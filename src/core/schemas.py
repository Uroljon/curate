# schemas.py

from pydantic import BaseModel, Field


class SourceAttribution(BaseModel):
    """Source attribution linking extracted data to original PDF page."""

    page_number: int = Field(..., description="Page number in original PDF (1-based)")
    quote: str = Field(..., description="Relevant excerpt from original text")
    chunk_id: int | None = Field(default=None, description="Internal chunk ID for debugging")


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
