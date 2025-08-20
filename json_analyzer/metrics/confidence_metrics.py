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


class ConfidenceMetrics:
    """Calculator for confidence and uncertainty analysis metrics."""

    def __init__(self, thresholds: ConfidenceThresholds):
        self.thresholds = thresholds

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

        entity_types = ["action_fields", "projects", "measures", "indicators"]
        type_mapping = {
            "action_fields": "af",
            "projects": "proj",
            "measures": "msr",
            "indicators": "ind",
        }

        # Collect confidence scores by connection type
        for entity_type in entity_types:
            source_type = type_mapping[entity_type]

            for entity in data.get(entity_type, []):
                for connection in entity.get("connections", []):
                    target_id = connection.get("target_id", "")
                    confidence = connection.get("confidence_score", 1.0)

                    if target_id and isinstance(confidence, int | float):
                        # Infer target type from ID
                        target_type = self._infer_type_from_id(target_id)
                        connection_type = f"{source_type}_to_{target_type}"
                        confidence_by_type[connection_type].append(confidence)

        # Calculate statistics for each connection type
        stats_by_type = {}
        for conn_type, confidences in confidence_by_type.items():
            if confidences:
                stats_by_type[conn_type] = {
                    "mean": mean(confidences),
                    "median": median(confidences),
                    "min": min(confidences),
                    "max": max(confidences),
                    "std": stdev(confidences) if len(confidences) > 1 else 0.0,
                    "count": len(confidences),
                    "low_confidence_ratio": sum(
                        1
                        for c in confidences
                        if c < self.thresholds.low_confidence_threshold
                    )
                    / len(confidences),
                }
            else:
                stats_by_type[conn_type] = {
                    "mean": 0.0,
                    "median": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "std": 0.0,
                    "count": 0,
                    "low_confidence_ratio": 0.0,
                }

        return stats_by_type

    def _find_low_confidence_edges(self, data: dict[str, Any]) -> dict[str, int]:
        """Find and count low confidence edges by type."""
        low_confidence_counts = defaultdict(int)
        total_counts = defaultdict(int)

        entity_types = ["action_fields", "projects", "measures", "indicators"]
        type_mapping = {
            "action_fields": "af",
            "projects": "proj",
            "measures": "msr",
            "indicators": "ind",
        }

        for entity_type in entity_types:
            source_type = type_mapping[entity_type]

            for entity in data.get(entity_type, []):
                entity_low_counts, entity_total_counts = (
                    self._process_entity_connections(entity, source_type)
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
                target_type = self._infer_type_from_id(target_id)
                connection_type = f"{source_type}_to_{target_type}"

                total_counts[connection_type] += 1

                if confidence < self.thresholds.low_confidence_threshold:
                    low_counts[connection_type] += 1

        return low_counts, total_counts

    def _detect_ambiguous_nodes(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """Detect nodes with many low-confidence or conflicting connections."""
        ambiguous_nodes = []
        entity_types = ["action_fields", "projects", "measures", "indicators"]

        for entity_type in entity_types:
            for entity in data.get(entity_type, []):
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
        entity_id = entity.get("id", "")
        connections = entity.get("connections", [])

        if not connections:
            return None

        confidences = [conn.get("confidence_score", 1.0) for conn in connections]
        low_confidence_count = self._count_low_confidence_connections(confidences)

        is_ambiguous, ambiguity_reasons = self._analyze_ambiguity_indicators(
            connections, confidences, low_confidence_count
        )

        if not is_ambiguous:
            return None

        return self._create_ambiguous_node_record(
            entity_id,
            entity_type,
            entity,
            connections,
            low_confidence_count,
            confidences,
            ambiguity_reasons,
        )

    def _count_low_confidence_connections(self, confidences: list[float]) -> int:
        """Count connections below the low confidence threshold."""
        return sum(
            1 for c in confidences if c < self.thresholds.low_confidence_threshold
        )

    def _analyze_ambiguity_indicators(
        self,
        connections: list[dict[str, Any]],
        confidences: list[float],
        low_confidence_count: int,
    ) -> tuple[bool, list[str]]:
        """Analyze various indicators to determine if a node is ambiguous."""
        is_ambiguous = False
        ambiguity_reasons = []

        # Check high ratio of low-confidence connections
        if len(connections) > 0:
            low_conf_ratio = low_confidence_count / len(connections)
            if low_conf_ratio > 0.5:  # More than 50% low confidence
                is_ambiguous = True
                ambiguity_reasons.append(
                    f"High low-confidence ratio: {low_conf_ratio:.2f}"
                )

        # Check high variance in confidence scores
        if len(confidences) > 2:
            conf_std = stdev(confidences)
            if conf_std > self.thresholds.confidence_std_threshold:
                is_ambiguous = True
                ambiguity_reasons.append(f"High confidence variance: {conf_std:.2f}")

        # Check for too many connections (potential over-connection)
        if len(connections) > 10:
            is_ambiguous = True
            ambiguity_reasons.append(f"Too many connections: {len(connections)}")

        return is_ambiguous, ambiguity_reasons

    def _create_ambiguous_node_record(
        self,
        entity_id: str,
        entity_type: str,
        entity: dict[str, Any],
        connections: list[dict[str, Any]],
        low_confidence_count: int,
        confidences: list[float],
        ambiguity_reasons: list[str],
    ) -> dict[str, Any]:
        """Create a record for an ambiguous node."""
        content = entity.get("content", {})
        name = content.get("title") or content.get("name", "")

        return {
            "id": entity_id,
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
        distributions = {}

        entity_types = ["action_fields", "projects", "measures", "indicators"]

        for entity_type in entity_types:
            confidences = []

            for entity in data.get(entity_type, []):
                for connection in entity.get("connections", []):
                    confidence = connection.get("confidence_score", 1.0)
                    if isinstance(confidence, int | float):
                        confidences.append(confidence)

            distributions[entity_type] = confidences

        return distributions

    def _calculate_calibration_metrics(self, data: dict[str, Any]) -> dict[str, float]:
        """Calculate calibration metrics (if ground truth available)."""
        # This is a placeholder for future calibration analysis
        # Would require ground truth data to compare against

        all_confidences = []

        entity_types = ["action_fields", "projects", "measures", "indicators"]

        for entity_type in entity_types:
            for entity in data.get(entity_type, []):
                for connection in entity.get("connections", []):
                    confidence = connection.get("confidence_score", 1.0)
                    if isinstance(confidence, int | float):
                        all_confidences.append(confidence)

        if not all_confidences:
            return {}

        # Basic calibration-related metrics
        return {
            "overall_mean_confidence": mean(all_confidences),
            "confidence_range": max(all_confidences) - min(all_confidences),
            "high_confidence_ratio": sum(1 for c in all_confidences if c > 0.9)
            / len(all_confidences),
            "medium_confidence_ratio": sum(
                1 for c in all_confidences if 0.7 <= c <= 0.9
            )
            / len(all_confidences),
            "low_confidence_ratio": sum(1 for c in all_confidences if c < 0.7)
            / len(all_confidences),
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
