"""
Configuration and thresholds for JSON quality analysis.
"""

from typing import Any

from pydantic import BaseModel, Field


class GraphThresholds(BaseModel):
    """Thresholds for graph quality metrics."""

    min_avg_degree: float = 2.0
    max_isolated_nodes_ratio: float = 0.05  # 5%
    min_largest_component_ratio: float = 0.8  # 80%
    max_components: int = 3


class IntegrityThresholds(BaseModel):
    """Thresholds for data integrity metrics."""

    max_dangling_refs: int = 0
    max_duplicate_rate: float = 0.05  # 5%
    min_field_completeness: float = 0.9  # 90%
    similarity_threshold: float = 0.85  # For duplicate detection
    required_fields: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "action_fields": ["name", "title"],
            "projects": ["title"],
            "measures": ["title"],
            "indicators": ["title", "description"],
        }
    )


class ConnectivityThresholds(BaseModel):
    """Thresholds for connectivity metrics."""

    min_af_coverage: float = 0.90  # 90% of action fields should have connections
    min_project_coverage: float = (
        0.85  # 85% of projects should connect to AF and measures
    )
    min_measures_per_project: float = 1.0
    max_path_length: int = 5
    min_centrality_diversity: float = 0.3


class ConfidenceThresholds(BaseModel):
    """Thresholds for confidence metrics."""

    low_confidence_threshold: float = 0.7
    max_low_confidence_ratio: float = 0.15  # 15%
    min_mean_confidence: float = 0.8
    confidence_std_threshold: float = 0.2


class SourceThresholds(BaseModel):
    """Thresholds for source validation metrics."""

    min_quote_match_rate: float = 0.85  # 85%
    min_source_coverage: float = 0.80  # 80%
    fuzzy_match_threshold: float = 0.90
    max_invalid_pages_ratio: float = 0.05  # 5%


class ContentThresholds(BaseModel):
    """Thresholds for content quality metrics."""

    max_repetition_rate: float = 0.10  # 10%
    min_description_length: int = 10
    max_description_length: int = 1000
    max_language_inconsistency: float = 0.05  # 5%


class QualityWeights(BaseModel):
    """Weights for composite quality score calculation."""

    integrity: float = 0.35  # 35% - Graph structure integrity is critical
    content: float = 0.25  # 25% - Duplicate detection and text quality
    connectivity: float = 0.25  # 25% - Entity relationships
    confidence: float = 0.10  # 10% - Edge confidence scores
    sources: float = 0.05  # 5% - Source attribution (nice to have)


class AnalyzerConfig(BaseModel):
    """Complete analyzer configuration."""

    graph_thresholds: GraphThresholds = Field(default_factory=GraphThresholds)
    integrity_thresholds: IntegrityThresholds = Field(
        default_factory=IntegrityThresholds
    )
    connectivity_thresholds: ConnectivityThresholds = Field(
        default_factory=ConnectivityThresholds
    )
    confidence_thresholds: ConfidenceThresholds = Field(
        default_factory=ConfidenceThresholds
    )
    source_thresholds: SourceThresholds = Field(default_factory=SourceThresholds)
    content_thresholds: ContentThresholds = Field(default_factory=ContentThresholds)
    quality_weights: QualityWeights = Field(default_factory=QualityWeights)

    # Global settings
    enable_fuzzy_matching: bool = True
    enable_drift_analysis: bool = True
    enable_detailed_logging: bool = True
    max_duplicate_groups_to_report: int = 10
    max_examples_per_metric: int = 5

    # Performance settings
    large_file_threshold_mb: int = 50
    max_nodes_for_centrality: int = 10000
    parallel_processing: bool = True


# Default configuration instance
DEFAULT_CONFIG = AnalyzerConfig()


def get_quality_grade(score: float) -> str:
    """Convert numeric score to letter grade."""
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    else:
        return "F"


def calculate_composite_score(
    scores: dict[str, float], weights: QualityWeights
) -> tuple[float, dict[str, float]]:
    """
    Calculate weighted composite quality score.

    Args:
        scores: Dictionary of category scores (0-100)
        weights: Weight configuration

    Returns:
        Tuple of (composite_score, weighted_contributions)
    """
    weight_dict = {
        "integrity": weights.integrity,
        "connectivity": weights.connectivity,
        "confidence": weights.confidence,
        "sources": weights.sources,
        "content": weights.content,
    }

    # Normalize weights to sum to 1
    total_weight = sum(weight_dict.values())
    if total_weight > 0:
        weight_dict = {k: v / total_weight for k, v in weight_dict.items()}

    # Calculate weighted score
    composite_score = 0.0
    weighted_contributions = {}

    for category, weight in weight_dict.items():
        category_score = scores.get(category, 0.0)
        contribution = category_score * weight
        weighted_contributions[category] = contribution
        composite_score += contribution

    return min(100.0, max(0.0, composite_score)), weighted_contributions


# Expected ID prefixes for validation
ID_PREFIXES = {
    "action_fields": ["af_"],
    "projects": ["proj_"],
    "measures": ["msr_"],
    "indicators": ["ind_"],
}

# Expected connection types for validation
VALID_CONNECTIONS = {
    ("proj", "af"): "project_to_action_field",
    ("msr", "proj"): "measure_to_project",
    ("msr", "af"): "measure_to_action_field",
    ("ind", "proj"): "indicator_to_project",
    ("ind", "af"): "indicator_to_action_field",
    ("ind", "msr"): "indicator_to_measure",
}

# Language codes for content analysis
SUPPORTED_LANGUAGES = ["de", "en", "fr", "es", "it"]
