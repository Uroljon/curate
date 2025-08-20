"""
Operations-based schema for incremental extraction.

This module defines the operation models that allow the LLM to specify
what changes to make to the extraction state, rather than reproducing
the entire JSON structure.
"""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class OperationType(str, Enum):
    """Types of operations that can be performed on extraction entities."""

    CREATE = "CREATE"  # Create new entity
    UPDATE = "UPDATE"  # Modify existing entity (supports both replacement and merging)
    CONNECT = "CONNECT"  # Create/update connections between entities


class EntityOperation(BaseModel):
    """
    Represents a single operation to be performed on the extraction state.

    This allows the LLM to specify incremental changes rather than
    reproducing the entire JSON structure.
    """

    operation: OperationType = Field(description="Type of operation to perform")

    entity_type: Literal["action_field", "project", "measure", "indicator"] = Field(
        description="Type of entity this operation affects"
    )

    # For UPDATE operations
    entity_id: str | None = Field(
        default=None,
        description="ID of existing entity to modify (required for UPDATE)",
    )

    # For CREATE operations - content of new entity
    content: dict[str, Any] | None = Field(
        default=None, description="Content data for the entity"
    )

    # For CONNECT operations
    connections: list[dict[str, Any]] | None = Field(
        default=None, description="List of connections to create/update"
    )

    # Metadata
    confidence: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Confidence in this operation"
    )

    # Source attribution
    source_pages: list[int] | None = Field(
        default=None, description="Page numbers where this information was found"
    )

    source_quote: str | None = Field(
        default=None, description="Relevant quote from source text"
    )


class ExtractionOperations(BaseModel):
    """
    Container for a list of operations to be applied to extraction state.

    This is the response format expected from the LLM.
    """

    operations: list[EntityOperation] = Field(
        description="List of operations to apply to the extraction state"
    )

    # Metadata about the chunk that generated these operations
    chunk_index: int | None = Field(
        default=None, description="Index of the chunk that generated these operations"
    )

    source_pages: list[int] | None = Field(
        default=None, description="Page numbers processed in this chunk"
    )


class ConnectionOperation(BaseModel):
    """
    Represents a connection between two entities.

    Used within EntityOperation for CONNECT operations.
    """

    from_id: str = Field(description="ID of the source entity")

    to_id: str = Field(description="ID of the target entity")

    confidence: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Confidence in this connection"
    )


class OperationResult(BaseModel):
    """
    Result of applying an operation to the extraction state.

    Used for logging and debugging operation application.
    """

    operation: EntityOperation = Field(description="The operation that was applied")

    success: bool = Field(description="Whether the operation was successfully applied")

    error_message: str | None = Field(
        default=None, description="Error message if operation failed"
    )

    entities_affected: list[str] = Field(
        default_factory=list,
        description="List of entity IDs that were affected by this operation",
    )

    new_entity_id: str | None = Field(
        default=None, description="ID of newly created entity (for CREATE operations)"
    )


class OperationLog(BaseModel):
    """
    Log entry for tracking operation application across chunks.
    """

    chunk_index: int = Field(description="Index of the chunk being processed")

    operation_results: list[OperationResult] = Field(
        description="Results of all operations applied in this chunk"
    )

    total_operations: int = Field(
        description="Total number of operations in this chunk"
    )

    successful_operations: int = Field(
        description="Number of operations that succeeded"
    )

    processing_time_seconds: float = Field(
        description="Time taken to apply all operations"
    )


# Export main classes for use in other modules
__all__ = [
    "ConnectionOperation",
    "EntityOperation",
    "ExtractionOperations",
    "OperationLog",
    "OperationResult",
    "OperationType",
]
