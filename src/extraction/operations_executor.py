"""
Operations executor for applying incremental extraction operations.

This module provides functions to apply EntityOperation instances to
the current extraction state, building the final result incrementally.
"""

import copy
import time
from typing import Any

from src.core.operations_schema import (
    EntityOperation,
    OperationLog,
    OperationResult,
    OperationType,
)
from src.core.schemas import (
    ConnectionWithConfidence,
    EnhancedActionField,
    EnhancedIndicator,
    EnhancedMeasure,
    EnhancedProject,
    EnrichedReviewJSON,
)


class OperationExecutor:
    """
    Executor for applying operations to extraction state.

    Handles CREATE, UPDATE (with intelligent merge), and CONNECT operations
    with validation and error handling.
    """

    def __init__(self):
        """Initialize the operation executor."""
        self.entity_counters = {"af": 0, "proj": 0, "msr": 0, "ind": 0}
        self.operation_logs: list[OperationLog] = []

    def apply_operations(
        self,
        current_state: EnrichedReviewJSON,
        operations: list[EntityOperation],
        chunk_index: int | None = None,
    ) -> tuple[EnrichedReviewJSON, OperationLog]:
        """
        Apply a list of operations to the current extraction state.

        Args:
            current_state: Current extraction state
            operations: List of operations to apply
            chunk_index: Optional chunk index for logging

        Returns:
            Tuple of (new_state, operation_log)
        """
        start_time = time.time()

        # Deep copy to avoid mutations
        new_state = copy.deepcopy(current_state)

        # Reorder operations by type to ensure proper dependency resolution
        operation_priority = {
            OperationType.CREATE: 0,
            OperationType.UPDATE: 1,
            OperationType.CONNECT: 2
        }

        # Sort operations by priority (safety measure in case reordering wasn't done upstream)
        reordered_operations = sorted(operations, key=lambda op: operation_priority.get(op.operation, 3))

        # Log reordering for debugging transparency
        if len(operations) > 1 and reordered_operations != operations:
            original_types = [op.operation.value for op in operations]
            reordered_types = [op.operation.value for op in reordered_operations]
            print(f"🔄 Safety reordering: {' '.join(original_types)} → {' '.join(reordered_types)}")

        # Track operation results
        operation_results: list[OperationResult] = []

        # Apply each operation (use reordered_operations)
        for operation in reordered_operations:
            try:
                result = self._apply_single_operation(new_state, operation)
                operation_results.append(result)

                if result.success:
                    # Improve CONNECT logging to avoid confusing "- None"
                    if operation.operation == OperationType.CONNECT:
                        # entities_affected contains [from_id, to_id, ...] pairs
                        edge_count = (
                            max(1, len(result.entities_affected) // 2)
                            if result.entities_affected
                            else 0
                        )
                        print(
                            f"   ✅ {operation.operation}: {operation.entity_type} - {edge_count} connection(s)"
                        )
                    else:
                        print(
                            f"   ✅ {operation.operation}: {operation.entity_type} - {result.new_entity_id or operation.entity_id}"
                        )
                else:
                    print(
                        f"   ❌ {operation.operation}: {operation.entity_type} - {result.error_message}"
                    )

            except Exception as e:
                error_result = OperationResult(
                    operation=operation,
                    success=False,
                    error_message=f"Unexpected error: {e!s}",
                    entities_affected=[],
                    new_entity_id=None,
                )
                operation_results.append(error_result)
                print(
                    f"   💥 {operation.operation}: {operation.entity_type} - Exception: {e!s}"
                )

        # Create operation log
        processing_time = time.time() - start_time
        successful_ops = sum(1 for r in operation_results if r.success)

        operation_log = OperationLog(
            chunk_index=chunk_index or 0,
            operation_results=operation_results,
            total_operations=len(operations),
            successful_operations=successful_ops,
            processing_time_seconds=processing_time,
        )

        self.operation_logs.append(operation_log)

        return new_state, operation_log

    def _apply_single_operation(
        self, state: EnrichedReviewJSON, operation: EntityOperation
    ) -> OperationResult:
        """Apply a single operation to the state."""

        if operation.operation == OperationType.CREATE:
            return self._handle_create_operation(state, operation)
        elif operation.operation == OperationType.UPDATE:
            return self._handle_update_operation(state, operation)
        elif operation.operation == OperationType.CONNECT:
            return self._handle_connect_operation(state, operation)
        else:
            return OperationResult(
                operation=operation,
                success=False,
                error_message=f"Unknown operation type: {operation.operation}",
                entities_affected=[],
                new_entity_id=None,
            )

    def _handle_create_operation(
        self, state: EnrichedReviewJSON, operation: EntityOperation
    ) -> OperationResult:
        """Handle CREATE operation - add new entity to state."""

        if not operation.content:
            return OperationResult(
                operation=operation,
                success=False,
                error_message="CREATE operation requires content",
                entities_affected=[],
                new_entity_id=None,
            )

        # Generate new ID
        new_id = self._generate_entity_id(operation.entity_type)

        # Create entity with page attribution for types that support it
        sources = None
        if operation.source_pages and operation.source_quote:
            from src.core.schemas import SourceAttribution

            sources = [
                SourceAttribution(page_number=page, quote=operation.source_quote)
                for page in operation.source_pages
            ]

        try:
            if operation.entity_type == "action_field":
                entity = EnhancedActionField(
                    id=new_id, content=operation.content, connections=[]
                )
                state.action_fields.append(entity)

            elif operation.entity_type == "project":
                entity = EnhancedProject(
                    id=new_id, content=operation.content, connections=[]
                )
                state.projects.append(entity)

            elif operation.entity_type == "measure":
                entity = EnhancedMeasure(
                    id=new_id,
                    content=operation.content,
                    connections=[],
                    sources=sources,
                )
                state.measures.append(entity)

            elif operation.entity_type == "indicator":
                entity = EnhancedIndicator(
                    id=new_id,
                    content=operation.content,
                    connections=[],
                    sources=sources,
                )
                state.indicators.append(entity)

            return OperationResult(
                operation=operation,
                success=True,
                error_message=None,
                entities_affected=[new_id],
                new_entity_id=new_id,
            )

        except Exception as e:
            return OperationResult(
                operation=operation,
                success=False,
                error_message=f"Failed to create entity: {e!s}",
                entities_affected=[],
                new_entity_id=None,
            )

    def _handle_update_operation(
        self, state: EnrichedReviewJSON, operation: EntityOperation
    ) -> OperationResult:
        """Handle UPDATE operation - modify existing entity with intelligent merging."""

        if not operation.entity_id:
            return OperationResult(
                operation=operation,
                success=False,
                error_message="UPDATE operation requires entity_id",
                entities_affected=[],
                new_entity_id=None,
            )

        # Find the entity
        entity = self._find_entity_by_id(state, operation.entity_id)
        if not entity:
            return OperationResult(
                operation=operation,
                success=False,
                error_message=f"Entity {operation.entity_id} not found",
                entities_affected=[],
                new_entity_id=None,
            )

        try:
            # Update content using intelligent merging
            if operation.content:
                self._merge_entity_content(entity, operation.content)

            # Add source attribution if provided (only for measures and indicators)
            if (
                operation.source_pages
                and operation.source_quote
                and hasattr(entity, "sources")
            ):
                from src.core.schemas import SourceAttribution

                new_sources = [
                    SourceAttribution(page_number=page, quote=operation.source_quote)
                    for page in operation.source_pages
                ]

                if entity.sources:
                    entity.sources.extend(new_sources)
                else:
                    entity.sources = new_sources

            return OperationResult(
                operation=operation,
                success=True,
                error_message=None,
                entities_affected=[operation.entity_id],
                new_entity_id=None,
            )

        except Exception as e:
            return OperationResult(
                operation=operation,
                success=False,
                error_message=f"Failed to update entity: {e!s}",
                entities_affected=[],
                new_entity_id=None,
            )

    def _handle_connect_operation(
        self, state: EnrichedReviewJSON, operation: EntityOperation
    ) -> OperationResult:
        """Handle CONNECT operation - create connections between entities with partial success support."""

        if not operation.connections:
            return OperationResult(
                operation=operation,
                success=False,
                error_message="CONNECT operation requires connections",
                entities_affected=[],
                new_entity_id=None,
            )

        affected_entities = []
        skipped_connections = 0
        total_connections = len(operation.connections)

        try:
            for conn_data in operation.connections:
                from_id = conn_data.get("from_id")
                to_id = conn_data.get("to_id")
                confidence = conn_data.get("confidence", operation.confidence)

                if not from_id or not to_id:
                    skipped_connections += 1
                    continue

                # Find source entity
                source_entity = self._find_entity_by_id(state, from_id)
                if not source_entity:
                    skipped_connections += 1
                    continue

                # Verify target entity exists
                target_entity = self._find_entity_by_id(state, to_id)
                if not target_entity:
                    skipped_connections += 1
                    continue

                # Create connection
                connection = ConnectionWithConfidence(
                    target_id=to_id, confidence_score=confidence
                )

                # Add connection if it doesn't already exist
                existing_connections = [c.target_id for c in source_entity.connections]
                if to_id not in existing_connections:
                    source_entity.connections.append(connection)
                    affected_entities.extend([from_id, to_id])
                else:
                    skipped_connections += 1  # Already exists

            # Determine success - succeed if we created at least one connection
            successful_connections = total_connections - skipped_connections
            is_success = successful_connections > 0

            error_msg = None
            if skipped_connections > 0 and successful_connections > 0:
                error_msg = f"Partial success: {skipped_connections}/{total_connections} connections skipped"
            elif skipped_connections == total_connections:
                error_msg = f"All {total_connections} connections failed (invalid IDs or duplicates)"

            return OperationResult(
                operation=operation,
                success=is_success,
                error_message=error_msg,
                entities_affected=affected_entities,
                new_entity_id=None,
            )

        except Exception as e:
            return OperationResult(
                operation=operation,
                success=False,
                error_message=f"Failed to create connections: {e!s}",
                entities_affected=[],
                new_entity_id=None,
            )

    def _find_entity_by_id(
        self, state: EnrichedReviewJSON, entity_id: str
    ) -> (
        EnhancedActionField
        | EnhancedProject
        | EnhancedMeasure
        | EnhancedIndicator
        | None
    ):
        """Find an entity by its ID across all buckets."""

        # Search in all entity lists
        all_entities = (
            state.action_fields + state.projects + state.measures + state.indicators
        )

        for entity in all_entities:
            if entity.id == entity_id:
                return entity

        return None

    def _merge_entity_content(
        self, target_entity: Any, new_content: dict[str, Any]
    ) -> None:
        """Intelligently merge new content into existing entity."""
        for key, value in new_content.items():
            if key not in target_entity.content:
                target_entity.content[key] = value
            else:
                self._merge_existing_field(target_entity, key, value)

    def _merge_existing_field(self, target_entity: Any, key: str, value: Any) -> None:
        """Merge a new value with an existing field value."""
        existing_value = target_entity.content[key]

        if isinstance(existing_value, str) and isinstance(value, str):
            self._merge_string_fields(target_entity, key, existing_value, value)
        elif isinstance(existing_value, list) and isinstance(value, list):
            self._merge_list_fields(existing_value, value)
        elif isinstance(existing_value, dict) and isinstance(value, dict):
            existing_value.update(value)
        else:
            self._merge_other_fields(target_entity, key, existing_value, value)

    def _merge_string_fields(
        self, target_entity: Any, key: str, existing: str, new: str
    ) -> None:
        """Merge string fields by appending if different."""
        if new not in existing:
            target_entity.content[key] = f"{existing}. {new}"

    def _merge_list_fields(self, existing_list: list[Any], new_list: list[Any]) -> None:
        """Merge list fields by extending with unique items."""
        for item in new_list:
            if item not in existing_list:
                existing_list.append(item)

    def _merge_other_fields(
        self, target_entity: Any, key: str, existing: Any, new: Any
    ) -> None:
        """Merge other field types by preferring longer/more detailed value."""
        if len(str(new)) > len(str(existing)):
            target_entity.content[key] = new

    def _generate_entity_id(self, entity_type: str) -> str:
        """Generate a unique ID for a new entity."""

        prefix_map = {
            "action_field": "af",
            "project": "proj",
            "measure": "msr",
            "indicator": "ind",
        }

        prefix = prefix_map.get(entity_type, "ent")
        self.entity_counters[prefix] += 1

        return f"{prefix}_{self.entity_counters[prefix]}"

    def get_operation_summary(self) -> dict[str, Any]:
        """Get a summary of all operations applied."""

        total_operations = sum(log.total_operations for log in self.operation_logs)
        successful_operations = sum(
            log.successful_operations for log in self.operation_logs
        )
        total_time = sum(log.processing_time_seconds for log in self.operation_logs)

        return {
            "total_chunks_processed": len(self.operation_logs),
            "total_operations": total_operations,
            "successful_operations": successful_operations,
            "failed_operations": total_operations - successful_operations,
            "success_rate": (
                successful_operations / total_operations if total_operations > 0 else 0
            ),
            "total_processing_time_seconds": total_time,
            "entities_created": sum(self.entity_counters.values()),
        }


# Convenience functions for direct use


def validate_operations(
    operations: list[EntityOperation], current_state: EnrichedReviewJSON | None = None
) -> list[str]:
    """
    Validate a list of operations for basic correctness.

    Args:
        operations: Operations to validate
        current_state: Optional current state for reference validation

    Returns:
        List of validation error messages (empty if all valid)
    """
    errors = []

    for i, op in enumerate(operations):
        op_prefix = f"Operation {i+1}"

        # Validate required fields based on operation type
        if op.operation == OperationType.CREATE and not op.content:
            errors.append(f"{op_prefix}: CREATE requires content")

        if op.operation == OperationType.UPDATE and not op.entity_id:
            errors.append(f"{op_prefix}: {op.operation} requires entity_id")

        if op.operation == OperationType.CONNECT and not op.connections:
            errors.append(f"{op_prefix}: CONNECT requires connections")

        # Validate entity IDs exist in current state (if provided)
        # Note: CREATE operations should not be validated for entity existence
        # since they create new entities
        if current_state and op.entity_id and op.operation != OperationType.CREATE:
            executor = OperationExecutor()
            if not executor._find_entity_by_id(current_state, op.entity_id):
                errors.append(
                    f"{op_prefix}: Entity {op.entity_id} not found in current state"
                )

    return errors
