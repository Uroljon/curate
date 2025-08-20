"""
Main JSON quality analyzer orchestrator.

Coordinates all metric calculators and provides a unified interface
for analyzing JSON extraction results.
"""

import json
import time
from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_CONFIG,
    AnalyzerConfig,
    calculate_composite_score,
    get_quality_grade,
)
from .metrics import (
    ConfidenceMetrics,
    ConnectivityMetrics,
    ContentMetrics,
    DriftMetrics,
    GraphMetrics,
    IntegrityMetrics,
    SourceMetrics,
)
from .models import (
    AnalysisMetadata,
    AnalysisResult,
    ComparisonResult,
    ConfidenceStats,
    ConnectivityStats,
    ContentStats,
    DriftStats,
    GraphStats,
    IntegrityStats,
    QualityScore,
    SourceStats,
)


class JSONAnalyzer:
    """
    Main analyzer for JSON extraction quality assessment.

    Orchestrates all metric calculations and provides a unified interface
    for analyzing single files, comparing files, and batch processing.
    """

    def __init__(self, config: AnalyzerConfig | None = None):
        """Initialize analyzer with configuration."""
        self.config = config or DEFAULT_CONFIG

        # Initialize metric calculators
        self.graph_metrics = GraphMetrics(self.config.graph_thresholds)
        self.integrity_metrics = IntegrityMetrics(self.config.integrity_thresholds)
        self.connectivity_metrics = ConnectivityMetrics(
            self.config.connectivity_thresholds
        )
        self.confidence_metrics = ConfidenceMetrics(self.config.confidence_thresholds)
        self.source_metrics = SourceMetrics(self.config.source_thresholds)
        self.content_metrics = ContentMetrics(self.config.content_thresholds)
        self.drift_metrics = DriftMetrics(self.config)

    def analyze_file(self, file_path: str | Path) -> AnalysisResult:
        """
        Analyze a single JSON file.

        Args:
            file_path: Path to JSON file

        Returns:
            AnalysisResult with all metrics
        """
        file_path = Path(file_path)
        start_time = time.time()

        # Load data
        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            msg = f"Failed to load JSON file: {e}"
            raise ValueError(msg) from e

        # Detect format
        format_detected = self._detect_format(data)

        # Create metadata
        metadata = AnalysisMetadata(
            file_path=str(file_path),
            file_size=file_path.stat().st_size,
            format_detected=format_detected,
            thresholds_used=self.config.model_dump(),
        )

        # Build graph
        graph = self.graph_metrics.build_graph_from_data(data)

        # Calculate all metrics
        result = AnalysisResult(
            metadata=metadata,
            graph_stats=self.graph_metrics.calculate(graph, data),
            integrity_stats=self.integrity_metrics.calculate(data),
            connectivity_stats=self.connectivity_metrics.calculate(graph, data),
            confidence_stats=self.confidence_metrics.calculate(graph, data),
            source_stats=self.source_metrics.calculate(data, str(file_path)),
            content_stats=self.content_metrics.calculate(data),
        )

        # Calculate composite quality score
        result.quality_score = self._calculate_quality_score(result)

        # Update timing
        end_time = time.time()
        result.metadata.analysis_duration_ms = (end_time - start_time) * 1000

        return result

    def analyze_data(self, data: dict[str, Any], file_path: str = "") -> AnalysisResult:
        """
        Analyze JSON data directly (without loading from file).

        This method creates a temporary file to reuse analyze_file logic.

        Args:
            data: JSON data dictionary
            file_path: Optional file path for context

        Returns:
            AnalysisResult with all metrics
        """
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            json.dump(data, tmp_file, ensure_ascii=False, indent=2)
            tmp_path = tmp_file.name

        try:
            result = self.analyze_file(tmp_path)
            # Update file path in metadata to the original path if provided
            if file_path:
                result.metadata.file_path = file_path
            return result
        finally:
            Path(tmp_path).unlink()

    def compare_files(
        self, before_path: str | Path, after_path: str | Path
    ) -> ComparisonResult:
        """
        Compare two JSON files to analyze drift and changes.

        Args:
            before_path: Path to earlier JSON file
            after_path: Path to later JSON file

        Returns:
            ComparisonResult with drift analysis
        """
        # Analyze both files
        before_result = self.analyze_file(before_path)
        after_result = self.analyze_file(after_path)

        # Load data for drift analysis
        with open(before_path, encoding="utf-8") as f:
            before_data = json.load(f)
        with open(after_path, encoding="utf-8") as f:
            after_data = json.load(f)

        # Calculate drift metrics
        drift_stats = self.drift_metrics.calculate(before_data, after_data)

        # Calculate improvements and regressions
        improvements, regressions = self._calculate_improvements_regressions(
            before_result, after_result
        )

        # Create comparison metadata
        metadata = AnalysisMetadata(
            file_path=f"{before_path} -> {after_path}",
            format_detected="comparison",
            thresholds_used=self.config.model_dump(),
        )

        return ComparisonResult(
            metadata=metadata,
            before=before_result,
            after=after_result,
            improvements=improvements,
            regressions=regressions,
            drift_stats=drift_stats,
            summary=self._generate_comparison_summary(
                before_result, after_result, drift_stats
            ),
        )

    def _detect_format(self, data: dict[str, Any]) -> str:
        """Detect the format of the JSON data."""
        if "action_fields" not in data:
            return "unknown"

        action_fields = data.get("action_fields", [])
        if not action_fields:
            return "empty"

        first_af = action_fields[0]

        # Check for hierarchical format (ExtractionResult)
        if "projects" in first_af:
            return "ExtractionResult"

        # Check for flat format (EnrichedReviewJSON)
        if "id" in first_af and "connections" in first_af:
            return "EnrichedReviewJSON"

        return "unknown"

    def _calculate_quality_score(self, result: AnalysisResult) -> QualityScore:
        """Calculate composite quality score."""
        # Extract individual scores
        category_scores = {
            "integrity": self._score_integrity(result.integrity_stats),
            "connectivity": self._score_connectivity(result.connectivity_stats),
            "confidence": self._score_confidence(result.confidence_stats),
            "sources": self._score_sources(result.source_stats),
            "content": self._score_content(result.content_stats),
        }

        # Calculate composite score
        composite_score, weighted_contributions = calculate_composite_score(
            category_scores, self.config.quality_weights
        )

        # Calculate penalties and bonuses
        penalties = self._calculate_penalties(result)
        bonuses = self._calculate_bonuses(result)

        # Apply adjustments
        final_score = max(
            0.0,
            min(
                100.0, composite_score - sum(penalties.values()) + sum(bonuses.values())
            ),
        )

        return QualityScore(
            overall_score=final_score,
            category_scores=category_scores,
            weights=self.config.quality_weights.model_dump(),
            penalties=penalties,
            bonuses=bonuses,
            grade=get_quality_grade(final_score),
        )

    def _score_integrity(self, stats: IntegrityStats) -> float:
        """Score data integrity (0-100)."""
        score = 100.0

        # Penalize dangling references
        if len(stats.dangling_refs) > 0:
            score -= min(40, len(stats.dangling_refs) * 5)

        # Penalize duplicates
        avg_duplicate_rate = sum(stats.duplicate_rate.values()) / max(
            1, len(stats.duplicate_rate)
        )
        score -= min(30, avg_duplicate_rate * 50)

        # Penalize incomplete fields
        if stats.field_completeness:
            min_completeness = 1.0
            for _entity_type, fields in stats.field_completeness.items():
                if fields:
                    min_completeness = min(min_completeness, min(fields.values()))

            if (
                min_completeness
                < self.config.integrity_thresholds.min_field_completeness
            ):
                score -= min(
                    20,
                    (
                        self.config.integrity_thresholds.min_field_completeness
                        - min_completeness
                    )
                    * 40,
                )

        # Penalize type violations
        score -= min(10, len(stats.type_compatibility_violations) * 2)

        return max(0.0, score)

    def _score_connectivity(self, stats: ConnectivityStats) -> float:
        """Score connectivity quality (0-100)."""
        score = 100.0

        # Penalize low action field coverage
        if (
            stats.action_field_coverage
            < self.config.connectivity_thresholds.min_af_coverage
        ):
            score -= min(
                30,
                (
                    self.config.connectivity_thresholds.min_af_coverage
                    - stats.action_field_coverage
                )
                * 60,
            )

        # Penalize low project coverage
        if (
            stats.project_coverage
            < self.config.connectivity_thresholds.min_project_coverage
        ):
            score -= min(
                25,
                (
                    self.config.connectivity_thresholds.min_project_coverage
                    - stats.project_coverage
                )
                * 50,
            )

        # Penalize low measures per project
        measures_mean = stats.measures_per_project.get("mean", 0.0)
        if measures_mean < self.config.connectivity_thresholds.min_measures_per_project:
            score -= min(
                20,
                (
                    self.config.connectivity_thresholds.min_measures_per_project
                    - measures_mean
                )
                * 20,
            )

        # Bonus for good connectivity
        if stats.action_field_coverage > 0.95:
            score += 5

        return max(0.0, min(100.0, score))

    def _score_confidence(self, stats: ConfidenceStats) -> float:
        """Score confidence quality (0-100)."""
        score = 100.0

        # Penalize low confidence edges
        if stats.low_confidence_edges:
            total_low_conf = sum(
                edge_data["count"] for edge_data in stats.low_confidence_edges.values()
            )
            total_edges = sum(
                edge_data["total"] for edge_data in stats.low_confidence_edges.values()
            )

            if total_edges > 0:
                low_conf_ratio = total_low_conf / total_edges
                if (
                    low_conf_ratio
                    > self.config.confidence_thresholds.max_low_confidence_ratio
                ):
                    score -= min(
                        40,
                        (
                            low_conf_ratio
                            - self.config.confidence_thresholds.max_low_confidence_ratio
                        )
                        * 80,
                    )

        # Penalize ambiguous nodes
        score -= min(30, len(stats.ambiguous_nodes) * 3)

        # Check mean confidence
        if stats.calibration_metrics:
            mean_conf = stats.calibration_metrics.get("overall_mean_confidence", 1.0)
            if mean_conf < self.config.confidence_thresholds.min_mean_confidence:
                score -= min(
                    20,
                    (self.config.confidence_thresholds.min_mean_confidence - mean_conf)
                    * 40,
                )

        return max(0.0, score)

    def _score_sources(self, stats: SourceStats) -> float:
        """Score source quality (0-100)."""
        score = 100.0

        # Penalize low source coverage
        overall_coverage = stats.source_coverage.get("overall", 0.0)
        if overall_coverage < self.config.source_thresholds.min_source_coverage:
            score -= min(
                40,
                (self.config.source_thresholds.min_source_coverage - overall_coverage)
                * 50,
            )

        # Penalize low quote match rate
        if stats.quote_match_rate < self.config.source_thresholds.min_quote_match_rate:
            score -= min(
                30,
                (
                    self.config.source_thresholds.min_quote_match_rate
                    - stats.quote_match_rate
                )
                * 60,
            )

        # Penalize invalid quotes and missing sources
        score -= min(20, len(stats.invalid_quotes) * 2)
        score -= min(10, len(stats.missing_sources) * 1)

        return max(0.0, score)

    def _score_content(self, stats: ContentStats) -> float:
        """Score content quality (0-100)."""
        score = 100.0

        # Penalize repetition
        if stats.repetition_rate > self.config.content_thresholds.max_repetition_rate:
            score -= min(
                30,
                (
                    stats.repetition_rate
                    - self.config.content_thresholds.max_repetition_rate
                )
                * 60,
            )

        # Penalize language inconsistency
        inconsistency = stats.language_consistency.get("inconsistency_rate", 0.0)
        if inconsistency > self.config.content_thresholds.max_language_inconsistency:
            score -= min(
                25,
                (
                    inconsistency
                    - self.config.content_thresholds.max_language_inconsistency
                )
                * 50,
            )

        # Penalize normalization issues and outliers
        score -= min(25, len(stats.normalization_issues) * 2)
        total_outliers = sum(
            len(outlier_list) for outlier_list in stats.outliers.values()
        )
        score -= min(20, total_outliers * 1.5)

        return max(0.0, score)

    def _calculate_penalties(self, result: AnalysisResult) -> dict[str, float]:
        """Calculate specific penalties."""
        penalties = {}

        # Critical issues
        if len(result.integrity_stats.dangling_refs) > 0:
            penalties["dangling_refs"] = len(result.integrity_stats.dangling_refs) * 2

        # Structural issues
        if result.connectivity_stats.action_field_coverage < 0.5:
            penalties["low_af_coverage"] = 10

        return penalties

    def _calculate_bonuses(self, result: AnalysisResult) -> dict[str, float]:
        """Calculate bonuses for exceptional quality."""
        bonuses = {}

        # High coverage bonus
        if result.connectivity_stats.action_field_coverage > 0.95:
            bonuses["excellent_coverage"] = 3

        # High confidence bonus
        if result.confidence_stats.calibration_metrics:
            mean_conf = result.confidence_stats.calibration_metrics.get(
                "overall_mean_confidence", 0.0
            )
            if mean_conf > 0.9:
                bonuses["high_confidence"] = 2

        return bonuses

    def _calculate_improvements_regressions(
        self, before: AnalysisResult, after: AnalysisResult
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Calculate improvements and regressions between two results."""
        improvements = {}
        regressions = {}

        # Compare quality scores
        score_diff = (
            after.quality_score.overall_score - before.quality_score.overall_score
        )
        if score_diff > 0:
            improvements["overall_score"] = score_diff
        else:
            regressions["overall_score"] = abs(score_diff)

        # Compare coverage
        coverage_diff = (
            after.connectivity_stats.action_field_coverage
            - before.connectivity_stats.action_field_coverage
        )
        if coverage_diff > 0:
            improvements["af_coverage"] = coverage_diff
        elif coverage_diff < 0:
            regressions["af_coverage"] = abs(coverage_diff)

        # Compare integrity
        before_dangling = len(before.integrity_stats.dangling_refs)
        after_dangling = len(after.integrity_stats.dangling_refs)
        if after_dangling < before_dangling:
            improvements["dangling_refs_fixed"] = before_dangling - after_dangling
        elif after_dangling > before_dangling:
            regressions["new_dangling_refs"] = after_dangling - before_dangling

        return improvements, regressions

    def _generate_comparison_summary(
        self, before: AnalysisResult, after: AnalysisResult, drift_stats: DriftStats
    ) -> str:
        """Generate human-readable comparison summary."""
        score_change = (
            after.quality_score.overall_score - before.quality_score.overall_score
        )

        if score_change > 5:
            trend = "significant improvement"
        elif score_change > 1:
            trend = "slight improvement"
        elif score_change > -1:
            trend = "stable"
        elif score_change > -5:
            trend = "slight regression"
        else:
            trend = "significant regression"

        stability_grade = self.drift_metrics.get_drift_summary(drift_stats)[
            "stability_grade"
        ]

        return f"Quality {trend} (Δ{score_change:+.1f} points), stability grade {stability_grade}"

    def get_analysis_summary(self, result: AnalysisResult) -> dict[str, Any]:
        """Get a concise summary of the analysis."""
        return {
            "overall_score": result.quality_score.overall_score,
            "grade": result.quality_score.grade,
            "format": result.metadata.format_detected,
            "total_entities": (
                result.graph_stats.nodes_by_type.get("action_field", 0)
                + result.graph_stats.nodes_by_type.get("project", 0)
                + result.graph_stats.nodes_by_type.get("measure", 0)
                + result.graph_stats.nodes_by_type.get("indicator", 0)
            ),
            "total_connections": result.graph_stats.total_edges,
            "critical_issues": (
                len(result.integrity_stats.dangling_refs)
                + len(result.source_stats.invalid_quotes)
            ),
            "analysis_time_ms": result.metadata.analysis_duration_ms,
        }


# Convenience functions
def analyze_file(
    file_path: str | Path, config: AnalyzerConfig | None = None
) -> AnalysisResult:
    """Convenience function to analyze a single file."""
    analyzer = JSONAnalyzer(config)
    return analyzer.analyze_file(file_path)


def compare_files(
    before_path: str | Path,
    after_path: str | Path,
    config: AnalyzerConfig | None = None,
) -> ComparisonResult:
    """Convenience function to compare two files."""
    analyzer = JSONAnalyzer(config)
    return analyzer.compare_files(before_path, after_path)
