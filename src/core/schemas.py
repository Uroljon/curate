# schemas.py

from pydantic import BaseModel, Field


class SourceAttribution(BaseModel):
    """Source attribution linking extracted data to original PDF page."""

    page_number: int = Field(..., description="Page number in original PDF (1-based)")
    quote: str = Field(..., description="Relevant excerpt from original text")
    chunk_id: int | None = Field(
        default=None, description="Internal chunk ID for debugging"
    )


class Project(BaseModel):
    """Represents a single project within an action field."""

    title: str = Field(..., description="Project title/name")
    measures: list[str] | None = Field(
        default=None, description="List of measures/actions"
    )
    indicators: list[str] | None = Field(
        default=None, description="List of indicators/metrics"
    )
    sources: list[SourceAttribution] | None = Field(
        default=None, description="Source attribution with page numbers and quotes"
    )


class ActionField(BaseModel):
    """Represents a municipal action field (Handlungsfeld) with its projects."""

    action_field: str = Field(..., description="Name of the action field/domain")
    projects: list[Project] = Field(
        ..., description="List of projects in this action field"
    )


class ExtractionResult(BaseModel):
    """The complete extraction result containing all action fields."""

    action_fields: list[ActionField] = Field(
        ..., description="List of all extracted action fields"
    )


class ActionFieldList(BaseModel):
    """Simple list of action field names for Stage 1 extraction."""

    action_fields: list[str] = Field(
        ..., description="List of action field names (Handlungsfelder)"
    )


class ProjectList(BaseModel):
    """List of project titles for Stage 2 extraction."""

    projects: list[str] = Field(..., description="List of project titles")


class ProjectDetails(BaseModel):
    """Detailed project information for Stage 3 extraction."""

    measures: list[str] = Field(
        default_factory=list, description="List of measures/actions"
    )
    indicators: list[str] = Field(
        default_factory=list, description="List of indicators/KPIs"
    )


class ProjectDetailsEnhanced(BaseModel):
    """Enhanced project information with confidence scoring and reasoning for Chain-of-Thought extraction."""

    measures: list[str] = Field(
        default_factory=list, description="List of measures/actions"
    )
    indicators: list[str] = Field(
        default_factory=list, description="List of indicators/KPIs"
    )
    confidence_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Maps each measure/indicator to confidence score (0.0-1.0)",
    )
    reasoning: dict[str, str] = Field(
        default_factory=dict,
        description="Maps each measure/indicator to classification reasoning",
    )
    key_patterns: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Maps each measure/indicator to detected key patterns",
    )


# Enhanced schemas for the two-layer LLM pipeline (enhance_structure endpoint)


class ConnectionWithConfidence(BaseModel):
    """Represents a connection between entities with confidence scoring."""
    
    model_config = {"extra": "forbid"}

    target_id: str = Field(..., description="ID of the target entity")
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0"
    )


class EnhancedActionField(BaseModel):
    """Enhanced action field for relational 4-bucket structure."""

    id: str = Field(..., description="Unique identifier (e.g., 'af_1')")
    content: dict[str, str | None] = Field(
        ..., description="Action field content with 'name' and optional 'parent_id'"
    )
    connections: list[ConnectionWithConfidence] = Field(
        default_factory=list, description="Connections to projects and measures"
    )


class EnhancedProject(BaseModel):
    """Enhanced project for relational 4-bucket structure."""

    id: str = Field(..., description="Unique identifier (e.g., 'proj_1')")
    content: dict[str, str | None] = Field(
        ..., description="Project content with 'title' and optional 'description'"
    )
    connections: list[ConnectionWithConfidence] = Field(
        default_factory=list, description="Connections to action fields and measures"
    )


class EnhancedMeasure(BaseModel):
    """Enhanced measure for relational 4-bucket structure."""

    id: str = Field(..., description="Unique identifier (e.g., 'msr_1')")
    content: dict[str, str | None] = Field(
        ..., description="Measure content with 'title' and optional 'description'"
    )
    connections: list[ConnectionWithConfidence] = Field(
        default_factory=list,
        description="Connections to projects, action fields, and indicators",
    )
    sources: list[SourceAttribution] | None = Field(
        default=None, description="Source attribution with validated quotes"
    )


class EnhancedIndicator(BaseModel):
    """Enhanced indicator for relational 4-bucket structure."""

    id: str = Field(..., description="Unique identifier (e.g., 'ind_1')")
    content: dict[str, str] = Field(..., description="Indicator content with 'name'")
    connections: list[ConnectionWithConfidence] = Field(
        default_factory=list,
        description="Connections to measures that contribute to this indicator",
    )


class EnrichedReviewJSON(BaseModel):
    """The complete enhanced review structure with 4 separate entity buckets."""

    action_fields: list[EnhancedActionField] = Field(
        ..., description="List of all action fields with connections"
    )
    projects: list[EnhancedProject] = Field(
        ..., description="List of all projects with connections"
    )
    measures: list[EnhancedMeasure] = Field(
        ..., description="List of all measures with connections and sources"
    )
    indicators: list[EnhancedIndicator] = Field(
        ..., description="List of all indicators with connections"
    )
