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
    """Enhanced action field for relational 4-bucket structure matching Appwrite Dimensions schema."""

    id: str = Field(..., description="Unique identifier (e.g., 'af_1')")
    content: dict[str, str | list[str] | None] = Field(
        ..., description="Action field content matching Appwrite Dimensions schema"
    )
    connections: list[ConnectionWithConfidence] = Field(
        default_factory=list, description="Connections to projects and measures"
    )

    # Expected content fields (for reference):
    # {
    #   "name": str,                    # Required: "Mobilität und Verkehr"
    #   "description": str,             # Required: What this field encompasses
    #   "sustainability_type": str,     # Optional: "Environmental", "Social", "Economic"
    #   "strategic_goals": list[str],   # Optional: High-level objectives
    #   "sdgs": list[str],             # Optional: UN SDG alignment ["SDG 11", "SDG 13"]
    #   "parent_dimension_id": str,     # Optional: Hierarchical relationships
    #   "icon_ref": str,               # Optional: Visual representation
    # }


class EnhancedProject(BaseModel):
    """Enhanced project for relational 4-bucket structure matching Appwrite Measures schema."""

    id: str = Field(..., description="Unique identifier (e.g., 'proj_1')")
    content: dict[str, str | list[str] | int | float | bool | None] = Field(
        ..., description="Project content matching Appwrite Measures schema"
    )
    connections: list[ConnectionWithConfidence] = Field(
        default_factory=list, description="Connections to action fields and measures"
    )

    # Expected content fields (for reference):
    # {
    #   "title": str,                   # Required: "Radverkehrsnetz Ausbau"
    #   "description": str,             # Optional: Brief overview
    #   "full_description": str,        # Optional: Complete details
    #   "type": str,                    # Required: "Infrastructure", "Policy", etc.
    #   "status": str,                  # Optional: "In Planung", "Aktiv", "Abgeschlossen"
    #   "measure_start": str,           # Optional: "2024-01-01"
    #   "measure_end": str,             # Optional: "2026-12-31"
    #   "budget": float,                # Optional: 2500000
    #   "department": str,              # Optional: "Tiefbauamt"
    #   "responsible_person": list[str], # Optional: ["Max Mustermann"]
    #   "operative_goal": str,          # Optional: Specific objective
    #   "parent_measure": list[str],    # Optional: Sub-project relationships
    #   "is_parent": bool,              # Optional: Has child projects
    #   "sdgs": list[str],              # Optional: SDG alignment
    #   "cost_unit": int,               # Optional: Cost unit number
    #   "cost_unit_code": str,          # Optional: Cost unit code
    #   "priority": str,                # Optional: Priority level
    #   "state": str,                   # Optional: Current state
    #   "product_area": str,            # Optional: Product area
    #   "costs_responsible": str,       # Optional: Cost responsibility
    #   "contact_info": str,            # Optional: Contact information
    #   "account_number": str,          # Optional: Account number
    # }


class EnhancedMeasure(BaseModel):
    """Enhanced measure for relational 4-bucket structure - concrete actions within projects."""

    id: str = Field(..., description="Unique identifier (e.g., 'msr_1')")
    content: dict[str, str | list[str] | None] = Field(
        ..., description="Measure content with 'title' and optional 'description'"
    )
    connections: list[ConnectionWithConfidence] = Field(
        default_factory=list,
        description="Connections to projects, action fields, and indicators",
    )
    sources: list[SourceAttribution] | None = Field(
        default=None, description="Source attribution with validated quotes"
    )

    # Expected content fields (for reference):
    # {
    #   "title": str,                   # Required: "Fahrradwege ausbauen"
    #   "description": str,             # Optional: What this measure entails
    #   "timeline": str,                # Optional: Implementation timeline
    #   "responsible_department": str,  # Optional: Who implements this
    #   "related_projects": list[str],  # Optional: Parent project references
    # }


class EnhancedIndicator(BaseModel):
    """Enhanced indicator for relational 4-bucket structure matching Appwrite Indicators schema."""

    id: str = Field(..., description="Unique identifier (e.g., 'ind_1')")
    content: dict[str, str | list[str] | bool | None] = Field(
        ..., description="Indicator content matching Appwrite Indicators schema"
    )
    connections: list[ConnectionWithConfidence] = Field(
        default_factory=list,
        description="Connections to measures that contribute to this indicator",
    )

    # Expected content fields (for reference):
    # {
    #   "title": str,                   # Required: "CO2-Reduktion Verkehrssektor"
    #   "description": str,             # Required: "Jährliche CO2-Einsparung durch Maßnahmen"
    #   "unit": str,                    # Optional: "Tonnen CO2/Jahr"
    #   "granularity": str,             # Required: "annual", "monthly", "quarterly"
    #   "target_values": str,           # Optional: "500 Tonnen bis 2030"
    #   "actual_values": str,           # Optional: "120 Tonnen (2023)"
    #   "should_increase": bool,        # Optional: false (less CO2 is better)
    #   "calculation": str,             # Optional: "Baseline - Current emissions"
    #   "values_source": str,           # Optional: "Umweltamt Monitoring"
    #   "source_url": str,              # Optional: Reference URL
    #   "operational_goal": str,        # Optional: What it tracks specifically
    #   "dimension_ids": list[str],     # Optional: Multiple action fields it spans
    #   "sdgs": list[str],              # Optional: SDG alignment
    #   "is_group": bool,               # Optional: Whether it's a group indicator
    #   "grouped_indicators": str,      # Optional: Related indicator groupings
    # }


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
