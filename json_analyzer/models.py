"""
Pydantic models for JSON quality analysis results.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class GraphStats(BaseModel):
    """Core graph statistics."""

    nodes_by_type: dict[str, int] = Field(default_factory=dict)
    edges_by_relation: dict[str, int] = Field(default_factory=dict)
    total_nodes: int = 0
    total_edges: int = 0
    isolated_nodes: int = 0
    components: int = 0
    largest_component_size: int = 0
    avg_degree: float = 0.0
    median_degree: float = 0.0
    max_degree: int = 0
    degree_by_type: dict[str, dict[str, float]] = Field(default_factory=dict)


class IntegrityStats(BaseModel):
    """Schema and data integrity statistics."""

    id_validity: dict[str, Any] = Field(default_factory=dict)
    dangling_refs: list[dict[str, str]] = Field(default_factory=list)
    field_completeness: dict[str, dict[str, float]] = Field(default_factory=dict)
    type_compatibility_violations: list[dict[str, str]] = Field(default_factory=list)
    duplicate_nodes: dict[str, list[list[str]]] = Field(default_factory=dict)
    duplicate_rate: dict[str, float] = Field(default_factory=dict)


class ConnectivityStats(BaseModel):
    """Connectivity and structure statistics."""

    action_field_coverage: float = 0.0
    project_coverage: float = 0.0
    measures_per_project: dict[str, Any] = Field(default_factory=dict)
    path_lengths: dict[str, Any] = Field(default_factory=dict)
    centrality_scores: dict[str, dict[str, float]] = Field(default_factory=dict)
    cycles: list[list[str]] = Field(default_factory=list)
    coverage_metrics: dict[str, Any] = Field(default_factory=dict)


class ConfidenceStats(BaseModel):
    """Confidence and uncertainty statistics."""

    edge_confidence_by_type: dict[str, dict[str, float]] = Field(default_factory=dict)
    low_confidence_edges: dict[str, Any] = Field(default_factory=dict)
    ambiguous_nodes: list[dict[str, Any]] = Field(default_factory=list)
    confidence_distribution: dict[str, list[float]] = Field(default_factory=dict)
    calibration_metrics: dict[str, float] = Field(default_factory=dict)


class SourceStats(BaseModel):
    """Source and evidence statistics."""

    source_coverage: dict[str, float] = Field(default_factory=dict)
    quote_match_rate: float = 0.0
    page_validity: dict[str, Any] = Field(default_factory=dict)
    chunk_linkage: dict[str, Any] = Field(default_factory=dict)
    evidence_density: dict[str, float] = Field(default_factory=dict)
    invalid_quotes: list[dict[str, str]] = Field(default_factory=list)
    missing_sources: list[dict[str, Any]] = Field(default_factory=list)


class ContentStats(BaseModel):
    """Content quality statistics."""

    repetition_rate: float = 0.0
    length_distribution: dict[str, dict[str, Any]] = Field(default_factory=dict)
    language_consistency: dict[str, Any] = Field(default_factory=dict)
    normalization_issues: list[str] = Field(default_factory=list)
    duplicate_text: list[dict[str, Any]] = Field(default_factory=list)
    outliers: dict[str, list[str]] = Field(default_factory=dict)


class DriftStats(BaseModel):
    """Drift and stability statistics (for comparing runs)."""

    node_churn: dict[str, int] = Field(default_factory=dict)
    edge_churn: dict[str, int] = Field(default_factory=dict)
    coverage_delta: dict[str, float] = Field(default_factory=dict)
    confidence_drift: dict[str, float] = Field(default_factory=dict)
    structural_similarity: float = 0.0
    churn_rate: float = 0.0
    stability_score: float = 0.0


class QualityScore(BaseModel):
    """Composite quality score with breakdown."""

    overall_score: float = Field(default=0.0, ge=0, le=100)
    category_scores: dict[str, float] = Field(default_factory=dict)
    weights: dict[str, float] = Field(default_factory=dict)
    penalties: dict[str, float] = Field(default_factory=dict)
    bonuses: dict[str, float] = Field(default_factory=dict)
    grade: str = "F"  # A, B, C, D, F


class AnalysisMetadata(BaseModel):
    """Metadata about the analysis run."""

    timestamp: datetime = Field(default_factory=datetime.now)
    analyzer_version: str = "1.0.0"
    file_path: str = ""
    file_size: int = 0
    format_detected: str = ""  # "ExtractionResult" or "EnrichedReviewJSON"
    analysis_duration_ms: float = 0.0
    thresholds_used: dict[str, Any] = Field(default_factory=dict)


class AnalysisResult(BaseModel):
    """Complete analysis result containing all metrics."""

    metadata: AnalysisMetadata
    graph_stats: GraphStats = Field(default_factory=GraphStats)
    integrity_stats: IntegrityStats = Field(default_factory=IntegrityStats)
    connectivity_stats: ConnectivityStats = Field(default_factory=ConnectivityStats)
    confidence_stats: ConfidenceStats = Field(default_factory=ConfidenceStats)
    source_stats: SourceStats = Field(default_factory=SourceStats)
    content_stats: ContentStats = Field(default_factory=ContentStats)
    quality_score: QualityScore = Field(default_factory=QualityScore)
    drift_stats: DriftStats | None = None  # Only for comparison analyses


class ComparisonResult(BaseModel):
    """Result of comparing two analysis results."""

    metadata: AnalysisMetadata
    before: AnalysisResult
    after: AnalysisResult
    improvements: dict[str, float] = Field(default_factory=dict)
    regressions: dict[str, float] = Field(default_factory=dict)
    drift_stats: DriftStats = Field(default_factory=DriftStats)
    summary: str = ""


# Utility types for internal use
NodeType = str | int
