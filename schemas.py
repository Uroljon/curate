# schemas.py
from typing import List, Optional

from pydantic import BaseModel, Field


class Project(BaseModel):
    """Represents a single project within an action field."""

    title: str = Field(..., description="Project title/name")
    measures: list[str] | None = Field(
        default=None, description="List of measures/actions"
    )
    indicators: list[str] | None = Field(
        default=None, description="List of indicators/metrics"
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
