"""
Operations-based schema for incremental extraction.

This module defines the operation models that allow the LLM to specify
what changes to make to the extraction state, rather than reproducing
the entire JSON structure.
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, validator


class OperationType(str, Enum):
    """Types of operations that can be performed on extraction entities."""
    CREATE = "CREATE"      # Create new entity
    UPDATE = "UPDATE"      # Modify existing entity (add/change fields)
    MERGE = "MERGE"        # Merge new content into existing entity
    CONNECT = "CONNECT"    # Create/update connections between entities
    ENHANCE = "ENHANCE"    # Add details without changing core fields


class EntityOperation(BaseModel):
    """
    Represents a single operation to be performed on the extraction state.
    
    This allows the LLM to specify incremental changes rather than
    reproducing the entire JSON structure.
    """
    operation: OperationType = Field(
        description="Type of operation to perform"
    )

    entity_type: Literal["action_field", "project", "measure", "indicator"] = Field(
        description="Type of entity this operation affects"
    )

    # For UPDATE/MERGE/ENHANCE operations
    entity_id: str | None = Field(
        default=None,
        description="ID of existing entity to modify (required for UPDATE/MERGE/ENHANCE)"
    )

    # For CREATE operations - content of new entity
    content: dict[str, Any] | None = Field(
        default=None,
        description="Content data for the entity"
    )

    # For MERGE operations - specify which entity to merge into
    merge_with_id: str | None = Field(
        default=None,
        description="ID of entity to merge into (for MERGE operations)"
    )

    # For CONNECT operations
    connections: list[dict[str, Any]] | None = Field(
        default=None,
        description="List of connections to create/update"
    )

    # Metadata
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence in this operation"
    )

    reason: str | None = Field(
        default=None,
        description="Human-readable reason for this operation"
    )

    # Source attribution
    source_pages: list[int] | None = Field(
        default=None,
        description="Page numbers where this information was found"
    )

    source_quote: str | None = Field(
        default=None,
        description="Relevant quote from source text"
    )

    @validator('entity_id')
    def validate_entity_id_for_update_operations(cls, v, values):
        """Ensure entity_id is provided for operations that require it. For CREATE, ignore provided entity_id."""
        operation = values.get('operation')
        # For CREATE operations, ignore any provided entity_id (it will be auto-generated)
        if operation == OperationType.CREATE:
            return None  # Force to None for CREATE operations
        if operation in [OperationType.UPDATE, OperationType.MERGE, OperationType.ENHANCE]:
            if not v:
                raise ValueError(f"{operation} operations require entity_id")
        return v

    @validator('merge_with_id')
    def validate_merge_with_id(cls, v, values):
        """Ensure merge_with_id is provided for MERGE operations."""
        operation = values.get('operation')
        if operation == OperationType.MERGE and not v:
            raise ValueError("MERGE operations require merge_with_id")
        return v

    @validator('content')
    def validate_content_for_create(cls, v, values):
        """Ensure content is provided for CREATE operations."""
        operation = values.get('operation')
        if operation == OperationType.CREATE and not v:
            raise ValueError("CREATE operations require content")
        return v

    @validator('connections')
    def validate_connections_for_connect(cls, v, values):
        """Ensure connections are provided for CONNECT operations."""
        operation = values.get('operation')
        if operation == OperationType.CONNECT and not v:
            raise ValueError("CONNECT operations require connections")
        return v


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
        default=None,
        description="Index of the chunk that generated these operations"
    )

    source_pages: list[int] | None = Field(
        default=None,
        description="Page numbers processed in this chunk"
    )

    extraction_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Overall confidence in this chunk's extraction"
    )


class ConnectionOperation(BaseModel):
    """
    Represents a connection between two entities.
    
    Used within EntityOperation for CONNECT operations.
    """
    from_id: str = Field(
        description="ID of the source entity"
    )

    to_id: str = Field(
        description="ID of the target entity"
    )

    relationship_type: str | None = Field(
        default="belongs_to",
        description="Type of relationship (belongs_to, measures, implements, etc.)"
    )

    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence in this connection"
    )

    bidirectional: bool = Field(
        default=False,
        description="Whether this connection should be bidirectional"
    )


class OperationResult(BaseModel):
    """
    Result of applying an operation to the extraction state.
    
    Used for logging and debugging operation application.
    """
    operation: EntityOperation = Field(
        description="The operation that was applied"
    )

    success: bool = Field(
        description="Whether the operation was successfully applied"
    )

    error_message: str | None = Field(
        default=None,
        description="Error message if operation failed"
    )

    entities_affected: list[str] = Field(
        default_factory=list,
        description="List of entity IDs that were affected by this operation"
    )

    new_entity_id: str | None = Field(
        default=None,
        description="ID of newly created entity (for CREATE operations)"
    )


class OperationLog(BaseModel):
    """
    Log entry for tracking operation application across chunks.
    """
    chunk_index: int = Field(
        description="Index of the chunk being processed"
    )

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
    'ConnectionOperation',
    'EntityOperation',
    'ExtractionOperations',
    'OperationLog',
    'OperationResult',
    'OperationType'
]
