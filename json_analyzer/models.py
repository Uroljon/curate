"""
Pydantic models for JSON quality analysis results.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class GraphStats(BaseModel):
    """Core graph statistics."""
    nodes_by_type: Dict[str, int] = Field(default_factory=dict)
    edges_by_relation: Dict[str, int] = Field(default_factory=dict)
    total_nodes: int = 0
    total_edges: int = 0
    isolated_nodes: int = 0
    components: int = 0
    largest_component_size: int = 0
    avg_degree: float = 0.0
    median_degree: float = 0.0
    max_degree: int = 0
    degree_by_type: Dict[str, Dict[str, float]] = Field(default_factory=dict)


class IntegrityStats(BaseModel):
    """Schema and data integrity statistics."""
    id_validity: Dict[str, Any] = Field(default_factory=dict)
    dangling_refs: List[Dict[str, str]] = Field(default_factory=list)
    field_completeness: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    type_compatibility_violations: List[Dict[str, str]] = Field(default_factory=list)
    duplicate_nodes: Dict[str, List[List[str]]] = Field(default_factory=dict)
    duplicate_rate: Dict[str, float] = Field(default_factory=dict)


class ConnectivityStats(BaseModel):
    """Connectivity and structure statistics."""
    action_field_coverage: float = 0.0
    project_coverage: float = 0.0
    measures_per_project: Dict[str, Any] = Field(default_factory=dict)
    path_lengths: Dict[str, Any] = Field(default_factory=dict)
    centrality_scores: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    cycles: List[List[str]] = Field(default_factory=list)
    coverage_metrics: Dict[str, Any] = Field(default_factory=dict)


class ConfidenceStats(BaseModel):
    """Confidence and uncertainty statistics."""
    edge_confidence_by_type: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    low_confidence_edges: Dict[str, Any] = Field(default_factory=dict)
    ambiguous_nodes: List[Dict[str, Any]] = Field(default_factory=list)
    confidence_distribution: Dict[str, List[float]] = Field(default_factory=dict)
    calibration_metrics: Dict[str, float] = Field(default_factory=dict)


class SourceStats(BaseModel):
    """Source and evidence statistics."""
    source_coverage: Dict[str, float] = Field(default_factory=dict)
    quote_match_rate: float = 0.0
    page_validity: Dict[str, Any] = Field(default_factory=dict)
    chunk_linkage: Dict[str, Any] = Field(default_factory=dict)
    evidence_density: Dict[str, float] = Field(default_factory=dict)
    invalid_quotes: List[Dict[str, str]] = Field(default_factory=list)
    missing_sources: List[Dict[str, Any]] = Field(default_factory=list)


class ContentStats(BaseModel):
    """Content quality statistics."""
    repetition_rate: float = 0.0
    length_distribution: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    language_consistency: Dict[str, Any] = Field(default_factory=dict)
    normalization_issues: List[str] = Field(default_factory=list)
    duplicate_text: List[Dict[str, Any]] = Field(default_factory=list)
    outliers: Dict[str, List[str]] = Field(default_factory=dict)


class DriftStats(BaseModel):
    """Drift and stability statistics (for comparing runs)."""
    node_churn: Dict[str, int] = Field(default_factory=dict)
    edge_churn: Dict[str, int] = Field(default_factory=dict)
    coverage_delta: Dict[str, float] = Field(default_factory=dict)
    confidence_drift: Dict[str, float] = Field(default_factory=dict)
    structural_similarity: float = 0.0
    churn_rate: float = 0.0
    stability_score: float = 0.0


class QualityScore(BaseModel):
    """Composite quality score with breakdown."""
    overall_score: float = Field(default=0.0, ge=0, le=100)
    category_scores: Dict[str, float] = Field(default_factory=dict)
    weights: Dict[str, float] = Field(default_factory=dict)
    penalties: Dict[str, float] = Field(default_factory=dict)
    bonuses: Dict[str, float] = Field(default_factory=dict)
    grade: str = "F"  # A, B, C, D, F


class AnalysisMetadata(BaseModel):
    """Metadata about the analysis run."""
    timestamp: datetime = Field(default_factory=datetime.now)
    analyzer_version: str = "1.0.0"
    file_path: str = ""
    file_size: int = 0
    format_detected: str = ""  # "ExtractionResult" or "EnrichedReviewJSON"
    analysis_duration_ms: float = 0.0
    thresholds_used: Dict[str, Any] = Field(default_factory=dict)


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
    drift_stats: Optional[DriftStats] = None  # Only for comparison analyses


class ComparisonResult(BaseModel):
    """Result of comparing two analysis results."""
    metadata: AnalysisMetadata
    before: AnalysisResult
    after: AnalysisResult
    improvements: Dict[str, float] = Field(default_factory=dict)
    regressions: Dict[str, float] = Field(default_factory=dict)
    drift_stats: DriftStats = Field(default_factory=DriftStats)
    summary: str = ""


class BatchAnalysisResult(BaseModel):
    """Result of analyzing multiple files."""
    metadata: AnalysisMetadata
    results: List[AnalysisResult] = Field(default_factory=list)
    aggregate_stats: Dict[str, Any] = Field(default_factory=dict)
    trends: Dict[str, List[float]] = Field(default_factory=dict)
    outliers: List[str] = Field(default_factory=list)


# Utility types for internal use
NodeType = Union[str, int]
EdgeType = tuple[NodeType, NodeType, Dict[str, Any]]
GraphData = Dict[str, Any]