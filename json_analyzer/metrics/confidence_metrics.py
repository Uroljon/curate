"""
Confidence metrics calculation for JSON quality analysis.

Analyzes confidence scores, uncertainty patterns, ambiguity detection,
and calibration metrics for extracted data.
"""

from collections import Counter, defaultdict
from statistics import mean, median, stdev
from typing import Any

import networkx as nx

from ..config import ConfidenceThresholds
from ..models import ConfidenceStats

# Common data structures
ENTITY_TYPES = ["action_fields", "projects", "measures", "indicators"]
TYPE_MAPPING = {
    "action_fields": "af",
    "projects": "proj",
    "measures": "msr",
    "indicators": "ind",
}


class ConfidenceMetrics:
    """Calculator for confidence and uncertainty analysis metrics."""

    def __init__(self, thresholds: ConfidenceThresholds):
        self.thresholds = thresholds

    def _iterate_entities(self, data: dict[str, Any]):
        """Iterate over all entities with their type information."""
        for entity_type in ENTITY_TYPES:
            type_prefix = TYPE_MAPPING[entity_type]
            for entity in data.get(entity_type, []):
                yield entity_type, entity, type_prefix

    def _calculate_stats(self, values: list[float]) -> dict[str, float]:
        """Calculate standard statistics for a list of values."""
        if not values:
            return {
                "mean": 0.0,
                "median": 0.0,
                "min": 0.0,
                "max": 0.0,
                "std": 0.0,
                "count": 0,
            }

        return {
            "mean": mean(values),
            "median": median(values),
            "min": min(values),
            "max": max(values),
            "std": stdev(values) if len(values) > 1 else 0.0,
            "count": len(values),
        }

    def _collect_confidences_from_connections(
        self, data: dict[str, Any]
    ) -> list[float]:
        """Collect all confidence scores from entity connections."""
        confidences = []
        for _, entity, _ in self._iterate_entities(data):
            for connection in entity.get("connections", []):
                confidence = connection.get("confidence_score", 1.0)
                if isinstance(confidence, int | float):
                    confidences.append(confidence)
        return confidences

    def _get_connection_type(self, source_type: str, target_id: str) -> str:
        """Get connection type string from source type and target ID."""
        target_type = self._infer_type_from_id(target_id)
        return f"{source_type}_to_{target_type}"

    def _calculate_confidence_ratio(
        self, confidences: list[float], threshold: float, above: bool = False
    ) -> float:
        """Calculate ratio of values below/above threshold."""
        if not confidences:
            return 0.0
        if above:
            return sum(1 for c in confidences if c > threshold) / len(confidences)
        return sum(1 for c in confidences if c < threshold) / len(confidences)

    def calculate(self, graph: nx.Graph, data: dict[str, Any]) -> ConfidenceStats:
        """
        Calculate comprehensive confidence statistics.

        Args:
            graph: NetworkX graph representation
            data: Original JSON data

        Returns:
            ConfidenceStats object with all metrics
        """
        # Extract confidence scores from connections
        confidence_by_type = self._analyze_confidence_by_type(data)

        # Identify low confidence edges
        low_confidence_edges = self._find_low_confidence_edges(data)

        # Detect ambiguous nodes
        ambiguous_nodes = self._detect_ambiguous_nodes(data)

        # Analyze confidence distributions
        confidence_distributions = self._analyze_confidence_distributions(data)

        # Calculate calibration metrics (if applicable)
        calibration_metrics = self._calculate_calibration_metrics(data)

        return ConfidenceStats(
            edge_confidence_by_type=confidence_by_type,
            low_confidence_edges=low_confidence_edges,
            ambiguous_nodes=ambiguous_nodes,
            confidence_distribution=confidence_distributions,
            calibration_metrics=calibration_metrics,
        )

    def _analyze_confidence_by_type(
        self, data: dict[str, Any]
    ) -> dict[str, dict[str, float]]:
        """Analyze confidence scores grouped by connection type."""
        confidence_by_type = defaultdict(list)

        # Collect confidence scores by connection type
        for _, entity, source_type in self._iterate_entities(data):
            for connection in entity.get("connections", []):
                target_id = connection.get("target_id", "")
                confidence = connection.get("confidence_score", 1.0)

                if target_id and isinstance(confidence, int | float):
                    connection_type = self._get_connection_type(source_type, target_id)
                    confidence_by_type[connection_type].append(confidence)

        # Calculate statistics for each connection type
        stats_by_type = {}
        for conn_type, confidences in confidence_by_type.items():
            stats = self._calculate_stats(confidences)
            stats["low_confidence_ratio"] = self._calculate_confidence_ratio(
                confidences, self.thresholds.low_confidence_threshold
            )
            stats_by_type[conn_type] = stats

        return stats_by_type

    def _find_low_confidence_edges(self, data: dict[str, Any]) -> dict[str, int]:
        """Find and count low confidence edges by type."""
        low_confidence_counts = defaultdict(int)
        total_counts = defaultdict(int)

        for _, entity, source_type in self._iterate_entities(data):
            entity_low_counts, entity_total_counts = self._process_entity_connections(
                entity, source_type
            )

            # Merge counts
            for conn_type, count in entity_low_counts.items():
                low_confidence_counts[conn_type] += count
            for conn_type, count in entity_total_counts.items():
                total_counts[conn_type] += count

        # Calculate ratios
        result = {}
        for conn_type in total_counts:
            low_count = low_confidence_counts[conn_type]
            total_count = total_counts[conn_type]
            result[conn_type] = {
                "count": low_count,
                "total": total_count,
                "ratio": low_count / total_count if total_count > 0 else 0.0,
            }

        return dict(result)

    def _process_entity_connections(
        self, entity: dict[str, Any], source_type: str
    ) -> tuple[dict[str, int], dict[str, int]]:
        """Process connections for a single entity and return low confidence and total counts."""
        low_counts = defaultdict(int)
        total_counts = defaultdict(int)

        for connection in entity.get("connections", []):
            target_id = connection.get("target_id", "")
            confidence = connection.get("confidence_score", 1.0)

            if target_id and isinstance(confidence, int | float):
                connection_type = self._get_connection_type(source_type, target_id)
                total_counts[connection_type] += 1

                if confidence < self.thresholds.low_confidence_threshold:
                    low_counts[connection_type] += 1

        return low_counts, total_counts

    def _detect_ambiguous_nodes(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """Detect nodes with many low-confidence or conflicting connections."""
        ambiguous_nodes = []

        for entity_type, entity, _ in self._iterate_entities(data):
            ambiguous_node = self._check_entity_ambiguity(entity, entity_type)
            if ambiguous_node:
                ambiguous_nodes.append(ambiguous_node)

        # Sort by severity (lowest average confidence first)
        ambiguous_nodes.sort(key=lambda x: x["avg_confidence"])
        return ambiguous_nodes

    def _check_entity_ambiguity(
        self, entity: dict[str, Any], entity_type: str
    ) -> dict[str, Any] | None:
        """Check if a single entity is ambiguous based on its connections."""
        connections = entity.get("connections", [])
        if not connections:
            return None

        confidences = [conn.get("confidence_score", 1.0) for conn in connections]
        low_confidence_count = sum(
            1 for c in confidences if c < self.thresholds.low_confidence_threshold
        )

        # Check ambiguity conditions
        is_ambiguous = False
        ambiguity_reasons = []

        # High ratio of low-confidence connections
        low_conf_ratio = low_confidence_count / len(connections)
        if low_conf_ratio > 0.5:
            is_ambiguous = True
            ambiguity_reasons.append(f"High low-confidence ratio: {low_conf_ratio:.2f}")

        # High variance in confidence scores
        if len(confidences) > 2:
            conf_std = stdev(confidences)
            if conf_std > self.thresholds.confidence_std_threshold:
                is_ambiguous = True
                ambiguity_reasons.append(f"High confidence variance: {conf_std:.2f}")

        # Too many connections
        if len(connections) > 10:
            is_ambiguous = True
            ambiguity_reasons.append(f"Too many connections: {len(connections)}")

        if not is_ambiguous:
            return None

        # Create ambiguous node record
        content = entity.get("content", {})
        name = content.get("title") or content.get("name", "")

        return {
            "id": entity.get("id", ""),
            "type": entity_type,
            "name": name,
            "connection_count": len(connections),
            "low_confidence_count": low_confidence_count,
            "avg_confidence": mean(confidences),
            "confidence_std": stdev(confidences) if len(confidences) > 1 else 0.0,
            "reasons": ambiguity_reasons,
        }

    def _analyze_confidence_distributions(
        self, data: dict[str, Any]
    ) -> dict[str, list[float]]:
        """Analyze confidence score distributions."""
        distributions = {entity_type: [] for entity_type in ENTITY_TYPES}

        for entity_type, entity, _ in self._iterate_entities(data):
            for connection in entity.get("connections", []):
                confidence = connection.get("confidence_score", 1.0)
                if isinstance(confidence, int | float):
                    distributions[entity_type].append(confidence)

        return distributions

    def _calculate_calibration_metrics(self, data: dict[str, Any]) -> dict[str, float]:
        """Calculate calibration metrics (if ground truth available)."""
        all_confidences = self._collect_confidences_from_connections(data)

        if not all_confidences:
            return {}

        # Basic calibration-related metrics
        return {
            "overall_mean_confidence": mean(all_confidences),
            "confidence_range": max(all_confidences) - min(all_confidences),
            "high_confidence_ratio": self._calculate_confidence_ratio(
                all_confidences, 0.9, above=True
            ),
            "medium_confidence_ratio": sum(
                1 for c in all_confidences if 0.7 <= c <= 0.9
            )
            / len(all_confidences),
            "low_confidence_ratio": self._calculate_confidence_ratio(
                all_confidences, 0.7
            ),
        }

    def _infer_type_from_id(self, entity_id: str) -> str:
        """Infer entity type from ID prefix."""
        if entity_id.startswith("af_"):
            return "af"
        elif entity_id.startswith("proj_"):
            return "proj"
        elif entity_id.startswith("msr_"):
            return "msr"
        elif entity_id.startswith("ind_"):
            return "ind"
        else:
            return "unknown"
