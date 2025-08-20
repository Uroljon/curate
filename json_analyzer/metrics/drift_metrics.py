"""
Drift metrics calculation for JSON quality analysis.

Analyzes changes between different runs, including node churn,
edge churn, coverage deltas, confidence drift, and structural similarity.
"""

from collections import Counter, defaultdict
from statistics import mean
from typing import Any

from ..config import AnalyzerConfig
from ..models import DriftStats


class DriftMetrics:
    """Calculator for drift and stability analysis metrics."""

    def __init__(self, config: AnalyzerConfig):
        self.config = config

    def calculate(
        self, before_data: dict[str, Any], after_data: dict[str, Any]
    ) -> DriftStats:
        """
        Calculate comprehensive drift statistics between two data sets.

        Args:
            before_data: Earlier JSON data
            after_data: Later JSON data

        Returns:
            DriftStats object with all metrics
        """
        # Calculate node churn
        node_churn = self._calculate_node_churn(before_data, after_data)

        # Calculate edge churn
        edge_churn = self._calculate_edge_churn(before_data, after_data)

        # Calculate coverage deltas
        coverage_delta = self._calculate_coverage_delta(before_data, after_data)

        # Calculate confidence drift
        confidence_drift = self._calculate_confidence_drift(before_data, after_data)

        # Calculate structural similarity
        structural_similarity = self._calculate_structural_similarity(
            before_data, after_data
        )

        # Calculate overall churn rate
        churn_rate = self._calculate_overall_churn_rate(node_churn, edge_churn)

        # Calculate stability score
        stability_score = self._calculate_stability_score(
            node_churn,
            edge_churn,
            coverage_delta,
            confidence_drift,
            structural_similarity,
        )

        return DriftStats(
            node_churn=node_churn,
            edge_churn=edge_churn,
            coverage_delta=coverage_delta,
            confidence_drift=confidence_drift,
            structural_similarity=structural_similarity,
            churn_rate=churn_rate,
            stability_score=stability_score,
        )

    def _calculate_node_churn(
        self, before_data: dict[str, Any], after_data: dict[str, Any]
    ) -> dict[str, int]:
        """Calculate node churn by entity type."""
        churn = {}
        entity_types = ["action_fields", "projects", "measures", "indicators"]

        for entity_type in entity_types:
            # Get node sets
            before_nodes = self._extract_nodes(before_data.get(entity_type, []))
            after_nodes = self._extract_nodes(after_data.get(entity_type, []))

            # Calculate changes
            added = after_nodes - before_nodes
            removed = before_nodes - after_nodes
            modified = self._find_modified_nodes(
                before_data.get(entity_type, []), after_data.get(entity_type, [])
            )

            churn[entity_type] = {
                "added": len(added),
                "removed": len(removed),
                "modified": len(modified),
                "total_before": len(before_nodes),
                "total_after": len(after_nodes),
                "churn_rate": (len(added) + len(removed) + len(modified))
                / max(1, len(before_nodes)),
            }

        return churn

    def _extract_nodes(self, entities: list[dict[str, Any]]) -> set[str]:
        """Extract node identifiers from entities."""
        nodes = set()
        for entity in entities:
            entity_id = entity.get("id")
            if entity_id:
                nodes.add(entity_id)
            else:
                # Fallback to content-based identifier for legacy format
                content = entity.get("content", {})
                name = (
                    content.get("title")
                    or content.get("name")
                    or content.get("action_field", "")
                )
                if name:
                    nodes.add(name)
        return nodes

    def _find_modified_nodes(
        self,
        before_entities: list[dict[str, Any]],
        after_entities: list[dict[str, Any]],
    ) -> set[str]:
        """Find nodes that exist in both sets but have been modified."""
        # Create lookup dictionaries
        before_lookup = {}
        after_lookup = {}

        for entity in before_entities:
            entity_id = entity.get("id") or entity.get("content", {}).get("title", "")
            if entity_id:
                before_lookup[entity_id] = entity

        for entity in after_entities:
            entity_id = entity.get("id") or entity.get("content", {}).get("title", "")
            if entity_id:
                after_lookup[entity_id] = entity

        # Find modified entities
        modified = set()
        for entity_id in before_lookup:
            if entity_id in after_lookup:
                if self._entities_differ(
                    before_lookup[entity_id], after_lookup[entity_id]
                ):
                    modified.add(entity_id)

        return modified

    def _entities_differ(
        self, entity1: dict[str, Any], entity2: dict[str, Any]
    ) -> bool:
        """Check if two entities differ in content."""
        # Compare content
        content1 = entity1.get("content", {})
        content2 = entity2.get("content", {})

        if content1 != content2:
            return True

        # Compare connections
        connections1 = {
            (conn.get("target_id", ""), conn.get("confidence_score", 0.0))
            for conn in entity1.get("connections", [])
        }
        connections2 = {
            (conn.get("target_id", ""), conn.get("confidence_score", 0.0))
            for conn in entity2.get("connections", [])
        }

        return connections1 != connections2

    def _calculate_edge_churn(
        self, before_data: dict[str, Any], after_data: dict[str, Any]
    ) -> dict[str, int]:
        """Calculate edge churn by connection type."""
        # Extract edges from both datasets
        before_edges = self._extract_edges(before_data)
        after_edges = self._extract_edges(after_data)

        # Calculate changes
        added_edges = after_edges - before_edges
        removed_edges = before_edges - after_edges

        # Group by connection type
        edge_types = defaultdict(lambda: {"added": 0, "removed": 0, "stable": 0})

        for edge in added_edges:
            edge_type = self._get_edge_type(edge)
            edge_types[edge_type]["added"] += 1

        for edge in removed_edges:
            edge_type = self._get_edge_type(edge)
            edge_types[edge_type]["removed"] += 1

        for edge in before_edges & after_edges:
            edge_type = self._get_edge_type(edge)
            edge_types[edge_type]["stable"] += 1

        # Calculate churn rates
        churn_stats = {}
        for edge_type, stats in edge_types.items():
            total_before = stats["stable"] + stats["removed"]
            churn_rate = (stats["added"] + stats["removed"]) / max(1, total_before)

            churn_stats[edge_type] = {
                **stats,
                "total_before": total_before,
                "total_after": stats["stable"] + stats["added"],
                "churn_rate": churn_rate,
            }

        return dict(churn_stats)

    def _extract_edges(self, data: dict[str, Any]) -> set[tuple[str, str, float]]:
        """Extract edges (connections) from data."""
        edges = set()
        entity_types = ["action_fields", "projects", "measures", "indicators"]

        for entity_type in entity_types:
            for entity in data.get(entity_type, []):
                source_id = entity.get("id", "")
                if not source_id:
                    continue

                for connection in entity.get("connections", []):
                    target_id = connection.get("target_id", "")
                    confidence = connection.get("confidence_score", 1.0)

                    if target_id:
                        edges.add((source_id, target_id, confidence))

        return edges

    def _get_edge_type(self, edge: tuple[str, str, float]) -> str:
        """Get edge type from source and target IDs."""
        source_id, target_id, _ = edge

        source_type = self._infer_type_from_id(source_id)
        target_type = self._infer_type_from_id(target_id)

        return f"{source_type}_to_{target_type}"

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

    def _calculate_coverage_delta(
        self, before_data: dict[str, Any], after_data: dict[str, Any]
    ) -> dict[str, float]:
        """Calculate changes in coverage metrics."""
        coverage_delta = {}

        # Calculate coverage for both datasets
        before_coverage = self._calculate_basic_coverage(before_data)
        after_coverage = self._calculate_basic_coverage(after_data)

        # Calculate deltas
        for metric in before_coverage:
            delta = after_coverage.get(metric, 0.0) - before_coverage.get(metric, 0.0)
            coverage_delta[metric] = delta

        return coverage_delta

    def _calculate_basic_coverage(self, data: dict[str, Any]) -> dict[str, float]:
        """Calculate basic coverage metrics."""
        entity_types = ["action_fields", "projects", "measures", "indicators"]
        coverage = {}

        for entity_type in entity_types:
            entities = data.get(entity_type, [])
            if not entities:
                coverage[f"{entity_type}_with_connections"] = 0.0
                continue

            connected_entities = 0
            for entity in entities:
                connections = entity.get("connections", [])
                if connections:
                    connected_entities += 1

            coverage[f"{entity_type}_with_connections"] = connected_entities / len(
                entities
            )

        return coverage

    def _calculate_confidence_drift(
        self, before_data: dict[str, Any], after_data: dict[str, Any]
    ) -> dict[str, float]:
        """Calculate drift in confidence scores."""
        confidence_drift = {}

        # Extract confidence scores
        before_confidences = self._extract_confidence_scores(before_data)
        after_confidences = self._extract_confidence_scores(after_data)

        # Calculate drift metrics
        for conn_type in set(before_confidences.keys()) | set(after_confidences.keys()):
            before_scores = before_confidences.get(conn_type, [])
            after_scores = after_confidences.get(conn_type, [])

            before_mean = mean(before_scores) if before_scores else 0.0
            after_mean = mean(after_scores) if after_scores else 0.0

            confidence_drift[conn_type] = {
                "mean_delta": after_mean - before_mean,
                "before_mean": before_mean,
                "after_mean": after_mean,
                "before_count": len(before_scores),
                "after_count": len(after_scores),
            }

        return confidence_drift

    def _extract_confidence_scores(
        self, data: dict[str, Any]
    ) -> dict[str, list[float]]:
        """Extract confidence scores grouped by connection type."""
        confidence_scores = defaultdict(list)
        entity_types = ["action_fields", "projects", "measures", "indicators"]

        for entity_type in entity_types:
            for entity in data.get(entity_type, []):
                source_id = entity.get("id", "")

                for connection in entity.get("connections", []):
                    target_id = connection.get("target_id", "")
                    confidence = connection.get("confidence_score", 1.0)

                    if source_id and target_id and isinstance(confidence, int | float):
                        source_type = self._infer_type_from_id(source_id)
                        target_type = self._infer_type_from_id(target_id)
                        conn_type = f"{source_type}_to_{target_type}"

                        confidence_scores[conn_type].append(confidence)

        return dict(confidence_scores)

    def _calculate_structural_similarity(
        self, before_data: dict[str, Any], after_data: dict[str, Any]
    ) -> float:
        """Calculate structural similarity using Jaccard similarity."""
        # Extract structural signatures (node types and connection patterns)
        before_signature = self._get_structural_signature(before_data)
        after_signature = self._get_structural_signature(after_data)

        # Calculate Jaccard similarity
        intersection = len(before_signature & after_signature)
        union = len(before_signature | after_signature)

        return intersection / union if union > 0 else 0.0

    def _get_structural_signature(self, data: dict[str, Any]) -> set[str]:
        """Get structural signature of the data."""
        signature = set()
        entity_types = ["action_fields", "projects", "measures", "indicators"]

        # Add node type counts
        for entity_type in entity_types:
            count = len(data.get(entity_type, []))
            signature.add(
                f"nodes_{entity_type}_{count//10*10}"
            )  # Rounded to nearest 10

        # Add connection patterns
        edge_types = Counter()
        for entity_type in entity_types:
            for entity in data.get(entity_type, []):
                source_id = entity.get("id", "")
                for connection in entity.get("connections", []):
                    target_id = connection.get("target_id", "")
                    if source_id and target_id:
                        source_type = self._infer_type_from_id(source_id)
                        target_type = self._infer_type_from_id(target_id)
                        edge_types[f"{source_type}_to_{target_type}"] += 1

        # Add connection type patterns (rounded counts)
        for edge_type, count in edge_types.items():
            signature.add(f"edges_{edge_type}_{count//5*5}")  # Rounded to nearest 5

        return signature

    def _calculate_overall_churn_rate(
        self, node_churn: dict[str, int], edge_churn: dict[str, int]
    ) -> float:
        """Calculate overall churn rate."""
        total_node_churn = 0
        total_nodes = 0

        for _entity_type, stats in node_churn.items():
            total_node_churn += stats["added"] + stats["removed"] + stats["modified"]
            total_nodes += stats["total_before"]

        total_edge_churn = 0
        total_edges = 0

        for _edge_type, stats in edge_churn.items():
            total_edge_churn += stats["added"] + stats["removed"]
            total_edges += stats["total_before"]

        # Weighted average (nodes count more than edges)
        if total_nodes + total_edges == 0:
            return 0.0

        node_weight = 0.7
        edge_weight = 0.3

        node_churn_rate = total_node_churn / max(1, total_nodes)
        edge_churn_rate = total_edge_churn / max(1, total_edges)

        return node_weight * node_churn_rate + edge_weight * edge_churn_rate

    def _calculate_stability_score(
        self,
        node_churn: dict[str, int],
        edge_churn: dict[str, int],
        coverage_delta: dict[str, float],
        confidence_drift: dict[str, float],
        structural_similarity: float,
    ) -> float:
        """Calculate overall stability score (0-100)."""
        score = 100.0

        # Penalize high churn rates
        avg_node_churn = (
            mean([stats["churn_rate"] for stats in node_churn.values()])
            if node_churn
            else 0.0
        )
        avg_edge_churn = (
            mean([stats["churn_rate"] for stats in edge_churn.values()])
            if edge_churn
            else 0.0
        )

        score -= min(40, avg_node_churn * 50)  # Max 40 point penalty
        score -= min(30, avg_edge_churn * 40)  # Max 30 point penalty

        # Penalize large coverage changes
        max_coverage_change = (
            max([abs(delta) for delta in coverage_delta.values()])
            if coverage_delta
            else 0.0
        )
        score -= min(15, max_coverage_change * 30)  # Max 15 point penalty

        # Penalize large confidence drift
        max_confidence_drift = 0.0
        if confidence_drift:
            drift_values = [
                abs(stats["mean_delta"]) for stats in confidence_drift.values()
            ]
            max_confidence_drift = max(drift_values) if drift_values else 0.0
        score -= min(10, max_confidence_drift * 20)  # Max 10 point penalty

        # Bonus for high structural similarity
        score += min(5, (structural_similarity - 0.8) * 25)  # Max 5 point bonus

        return max(0.0, min(100.0, score))

    def get_drift_summary(self, stats: DriftStats) -> dict[str, Any]:
        """Generate a summary of drift analysis."""
        total_added = sum(churn["added"] for churn in stats.node_churn.values())
        total_removed = sum(churn["removed"] for churn in stats.node_churn.values())
        total_modified = sum(churn["modified"] for churn in stats.node_churn.values())

        return {
            "stability_grade": self._get_stability_grade(stats.stability_score),
            "major_changes": total_added + total_removed + total_modified,
            "structural_similarity": stats.structural_similarity,
            "most_volatile_entity_type": self._find_most_volatile_type(
                stats.node_churn
            ),
            "recommendations": self._generate_stability_recommendations(stats),
        }

    def _get_stability_grade(self, score: float) -> str:
        """Convert stability score to letter grade."""
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

    def _find_most_volatile_type(self, node_churn: dict[str, int]) -> str:
        """Find the entity type with highest churn rate."""
        if not node_churn:
            return "none"

        max_churn_rate = 0.0
        most_volatile = "none"

        for entity_type, stats in node_churn.items():
            if stats["churn_rate"] > max_churn_rate:
                max_churn_rate = stats["churn_rate"]
                most_volatile = entity_type

        return most_volatile

    def _generate_stability_recommendations(self, stats: DriftStats) -> list[str]:
        """Generate recommendations based on stability analysis."""
        recommendations = []

        if stats.stability_score < 70:
            recommendations.append(
                "High instability detected - consider stabilizing extraction process"
            )

        if stats.churn_rate > 0.3:
            recommendations.append("High churn rate - review extraction consistency")

        if stats.structural_similarity < 0.8:
            recommendations.append(
                "Low structural similarity - verify extraction approach consistency"
            )

        # Check for high confidence drift
        if stats.confidence_drift:
            max_drift = max(
                abs(drift["mean_delta"]) for drift in stats.confidence_drift.values()
            )
            if max_drift > 0.2:
                recommendations.append(
                    "Significant confidence drift detected - review confidence scoring"
                )

        return recommendations
