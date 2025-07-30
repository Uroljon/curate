"""
Graph quality metrics and monitoring for extraction results.

This module provides tools to measure and monitor the quality of the extracted
graph structures, focusing on node fragmentation and edge consistency issues.
"""

import json
from collections import Counter, defaultdict
from typing import Any, Dict, List, Set, Tuple

import networkx as nx


class GraphQualityAnalyzer:
    """
    Analyzer for measuring graph quality metrics.

    Focuses on detecting and quantifying:
    1. Node fragmentation (duplicate entities)
    2. Edge consistency (balanced connections)
    3. Graph connectivity (isolated components)
    4. Semantic coherence (meaningful connections)
    """

    def __init__(self):
        """Initialize the graph quality analyzer."""
        self.metrics = {}

    def analyze_extraction_quality(
        self, structures: list[dict[str, Any]], before_resolution: bool = False
    ) -> dict[str, Any]:
        """
        Analyze the quality of extraction results.

        Args:
            structures: List of extracted structures
            before_resolution: Whether this is before entity resolution

        Returns:
            Dictionary of quality metrics
        """
        if not structures:
            return self._empty_metrics()

        # Build graph representation
        graph = self._build_graph_from_structures(structures)

        # Calculate quality metrics
        metrics = {
            "timestamp": self._get_timestamp(),
            "stage": "before_resolution" if before_resolution else "after_resolution",
            "basic_stats": self._calculate_basic_stats(structures),
            "fragmentation_metrics": self._calculate_fragmentation_metrics(structures),
            "connectivity_metrics": self._calculate_connectivity_metrics(graph),
            "edge_consistency_metrics": self._calculate_edge_consistency_metrics(
                structures
            ),
            "overall_quality_score": 0.0,
        }

        # Calculate overall quality score (0-1000)
        metrics["overall_quality_score"] = self._calculate_overall_score(metrics)

        return metrics

    def compare_before_after(
        self, before_metrics: dict[str, Any], after_metrics: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Compare quality metrics before and after entity resolution.

        Args:
            before_metrics: Metrics before entity resolution
            after_metrics: Metrics after entity resolution

        Returns:
            Dictionary of improvement metrics
        """
        if not before_metrics or not after_metrics:
            return {}

        improvements = {
            "timestamp": self._get_timestamp(),
            "fragmentation_improvement": self._calculate_fragmentation_improvement(
                before_metrics["fragmentation_metrics"],
                after_metrics["fragmentation_metrics"],
            ),
            "connectivity_improvement": self._calculate_connectivity_improvement(
                before_metrics["connectivity_metrics"],
                after_metrics["connectivity_metrics"],
            ),
            "edge_consistency_improvement": self._calculate_edge_consistency_improvement(
                before_metrics["edge_consistency_metrics"],
                after_metrics["edge_consistency_metrics"],
            ),
            "overall_score_improvement": (
                after_metrics["overall_quality_score"]
                - before_metrics["overall_quality_score"]
            ),
        }

        return improvements

    def _build_graph_from_structures(
        self, structures: list[dict[str, Any]]
    ) -> nx.Graph:
        """
        Build a NetworkX graph from extraction structures.

        Args:
            structures: List of extraction structures

        Returns:
            NetworkX graph representation
        """
        graph = nx.Graph()

        # Add nodes and edges
        for structure in structures:
            action_field = structure.get("action_field", "")
            if action_field:
                graph.add_node(
                    f"af:{action_field}", type="action_field", name=action_field
                )

                for project in structure.get("projects", []):
                    project_title = project.get("title", "")
                    if project_title:
                        project_node = f"proj:{project_title}"
                        graph.add_node(project_node, type="project", name=project_title)
                        graph.add_edge(f"af:{action_field}", project_node)

                        # Add measures
                        for measure in project.get("measures", []):
                            measure_node = f"msr:{measure}"
                            graph.add_node(measure_node, type="measure", name=measure)
                            graph.add_edge(project_node, measure_node)

                        # Add indicators
                        for indicator in project.get("indicators", []):
                            indicator_node = f"ind:{indicator}"
                            graph.add_node(
                                indicator_node, type="indicator", name=indicator
                            )
                            graph.add_edge(project_node, indicator_node)

        return graph

    def _calculate_basic_stats(
        self, structures: list[dict[str, Any]]
    ) -> dict[str, int]:
        """
        Calculate basic statistics about the structures.

        Args:
            structures: List of extraction structures

        Returns:
            Dictionary of basic statistics
        """
        stats = {
            "total_action_fields": len(structures),
            "total_projects": 0,
            "total_measures": 0,
            "total_indicators": 0,
            "projects_with_measures": 0,
            "projects_with_indicators": 0,
        }

        for structure in structures:
            projects = structure.get("projects", [])
            stats["total_projects"] += len(projects)

            for project in projects:
                measures = project.get("measures", [])
                indicators = project.get("indicators", [])

                stats["total_measures"] += len(measures)
                stats["total_indicators"] += len(indicators)

                if measures:
                    stats["projects_with_measures"] += 1
                if indicators:
                    stats["projects_with_indicators"] += 1

        return stats

    def _calculate_fragmentation_metrics(
        self, structures: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Calculate node fragmentation metrics.

        Args:
            structures: List of extraction structures

        Returns:
            Dictionary of fragmentation metrics
        """
        # Collect all entity names
        action_field_names = [
            s.get("action_field", "") for s in structures if s.get("action_field")
        ]
        project_names = []
        measure_names = []
        indicator_names = []

        for structure in structures:
            for project in structure.get("projects", []):
                if project.get("title"):
                    project_names.append(project["title"])
                for measure in project.get("measures", []):
                    measure_names.append(measure)
                for indicator in project.get("indicators", []):
                    indicator_names.append(indicator)

        # Calculate fragmentation scores
        af_fragmentation = self._calculate_name_fragmentation(action_field_names)
        proj_fragmentation = self._calculate_name_fragmentation(project_names)
        msr_fragmentation = self._calculate_name_fragmentation(measure_names)
        ind_fragmentation = self._calculate_name_fragmentation(indicator_names)

        return {
            "action_field_fragmentation": af_fragmentation,
            "project_fragmentation": proj_fragmentation,
            "measure_fragmentation": msr_fragmentation,
            "indicator_fragmentation": ind_fragmentation,
            "overall_fragmentation_score": (
                af_fragmentation["fragmentation_ratio"]
                + proj_fragmentation["fragmentation_ratio"]
                + msr_fragmentation["fragmentation_ratio"]
                + ind_fragmentation["fragmentation_ratio"]
            )
            / 4,
        }

    def _calculate_name_fragmentation(self, names: list[str]) -> dict[str, Any]:
        """
        Calculate fragmentation metrics for a list of names.

        Args:
            names: List of entity names

        Returns:
            Dictionary of fragmentation metrics for this entity type
        """
        if not names:
            return {
                "total_names": 0,
                "unique_names": 0,
                "potential_duplicates": 0,
                "fragmentation_ratio": 0.0,
                "duplicate_groups": [],
            }

        unique_names = list(set(names))
        name_counts = Counter(names)

        # Find potential duplicates using similarity
        duplicate_groups = self._find_potential_duplicates(unique_names)

        total_potential_duplicates = sum(
            len(group) - 1 for group in duplicate_groups if len(group) > 1
        )

        # Fragmentation ratio: higher means more fragmentation
        fragmentation_ratio = (
            total_potential_duplicates / len(unique_names) if unique_names else 0.0
        )

        return {
            "total_names": len(names),
            "unique_names": len(unique_names),
            "potential_duplicates": total_potential_duplicates,
            "fragmentation_ratio": round(fragmentation_ratio, 3),
            "duplicate_groups": [group for group in duplicate_groups if len(group) > 1],
            "most_common": name_counts.most_common(5),
        }

    def _find_potential_duplicates(self, names: list[str]) -> list[list[str]]:
        """
        Find potential duplicate names using string similarity.

        Args:
            names: List of unique names

        Returns:
            List of groups of potentially duplicate names
        """
        if len(names) < 2:
            return []

        from difflib import SequenceMatcher

        groups = []
        processed = set()

        for i, name1 in enumerate(names):
            if name1 in processed:
                continue

            group = [name1]
            processed.add(name1)

            for j, name2 in enumerate(names):
                if j <= i or name2 in processed:
                    continue

                # Calculate similarity
                similarity = SequenceMatcher(None, name1.lower(), name2.lower()).ratio()

                # Check for substring relationships or high similarity
                if (
                    similarity > 0.8
                    or name1.lower() in name2.lower()
                    or name2.lower() in name1.lower()
                ):
                    group.append(name2)
                    processed.add(name2)

            groups.append(group)

        return groups

    def _calculate_connectivity_metrics(self, graph: nx.Graph) -> dict[str, Any]:
        """
        Calculate graph connectivity metrics.

        Args:
            graph: NetworkX graph

        Returns:
            Dictionary of connectivity metrics
        """
        if not graph.nodes():
            return {
                "total_nodes": 0,
                "total_edges": 0,
                "connected_components": 0,
                "largest_component_size": 0,
                "isolated_nodes": 0,
                "avg_degree": 0.0,
                "density": 0.0,
            }

        # Basic connectivity metrics
        total_nodes = graph.number_of_nodes()
        total_edges = graph.number_of_edges()
        components = list(nx.connected_components(graph))
        largest_component_size = (
            max(len(comp) for comp in components) if components else 0
        )
        isolated_nodes = sum(1 for node in graph.nodes() if graph.degree[node] == 0)

        # Degree statistics
        degrees = [graph.degree[node] for node in graph.nodes()]
        avg_degree = sum(degrees) / len(degrees) if degrees else 0.0

        # Graph density
        max_possible_edges = (
            total_nodes * (total_nodes - 1) / 2 if total_nodes > 1 else 0
        )
        density = total_edges / max_possible_edges if max_possible_edges > 0 else 0.0

        return {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "connected_components": len(components),
            "largest_component_size": largest_component_size,
            "isolated_nodes": isolated_nodes,
            "avg_degree": round(avg_degree, 2),
            "density": round(density, 3),
            "degree_distribution": Counter(degrees),
        }

    def _calculate_edge_consistency_metrics(
        self, structures: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Calculate edge consistency metrics.

        Args:
            structures: List of extraction structures

        Returns:
            Dictionary of edge consistency metrics
        """
        project_connection_counts = []
        measure_counts = []
        indicator_counts = []

        # Analyze connection patterns
        for structure in structures:
            projects = structure.get("projects", [])
            project_connection_counts.append(len(projects))

            for project in projects:
                measures = project.get("measures", [])
                indicators = project.get("indicators", [])

                measure_counts.append(len(measures))
                indicator_counts.append(len(indicators))

        # Calculate consistency metrics
        def calculate_consistency(counts):
            if not counts:
                return {"mean": 0, "std": 0, "coefficient_of_variation": 0}

            mean_val = sum(counts) / len(counts)
            variance = sum((x - mean_val) ** 2 for x in counts) / len(counts)
            std_val = variance**0.5
            cv = std_val / mean_val if mean_val > 0 else 0

            return {
                "mean": round(mean_val, 2),
                "std": round(std_val, 2),
                "coefficient_of_variation": round(cv, 3),
                "min": min(counts),
                "max": max(counts),
            }

        return {
            "project_connections": calculate_consistency(project_connection_counts),
            "measure_connections": calculate_consistency(measure_counts),
            "indicator_connections": calculate_consistency(indicator_counts),
            "zero_connection_projects": sum(
                1 for c in measure_counts + indicator_counts if c == 0
            ),
            "high_connection_projects": sum(
                1 for c in measure_counts + indicator_counts if c > 10
            ),
        }

    def _calculate_overall_score(self, metrics: dict[str, Any]) -> float:
        """
        Calculate an overall quality score (0-1000).

        Args:
            metrics: Dictionary of calculated metrics

        Returns:
            Overall quality score
        """
        score = 1000.0  # Start with perfect score

        # Penalize fragmentation (0-300 points)
        fragmentation_score = metrics["fragmentation_metrics"][
            "overall_fragmentation_score"
        ]
        fragmentation_penalty = min(300, fragmentation_score * 300)
        score -= fragmentation_penalty

        # Penalize poor connectivity (0-200 points)
        connectivity = metrics["connectivity_metrics"]
        if connectivity["total_nodes"] > 0:
            isolation_ratio = (
                connectivity["isolated_nodes"] / connectivity["total_nodes"]
            )
            connectivity_penalty = min(200, isolation_ratio * 200)
            score -= connectivity_penalty

        # Penalize edge inconsistency (0-200 points)
        edge_consistency = metrics["edge_consistency_metrics"]
        zero_connections = edge_consistency["zero_connection_projects"]
        basic_stats = metrics["basic_stats"]
        total_projects = basic_stats["total_projects"]

        if total_projects > 0:
            zero_ratio = zero_connections / total_projects
            consistency_penalty = min(200, zero_ratio * 200)
            score -= consistency_penalty

        # Bonus for good connectivity
        if connectivity["total_nodes"] > 0:
            density = connectivity["density"]
            if density > 0.1:  # Well-connected graph
                score += min(50, density * 100)

        return max(0.0, round(score, 1))

    def _calculate_fragmentation_improvement(
        self, before: dict[str, Any], after: dict[str, Any]
    ) -> dict[str, Any]:
        """Calculate improvement in fragmentation metrics."""
        return {
            "fragmentation_ratio_change": (
                after["overall_fragmentation_score"]
                - before["overall_fragmentation_score"]
            ),
            "duplicate_groups_reduced": (
                sum(
                    len(groups)
                    for groups in before["action_field_fragmentation"][
                        "duplicate_groups"
                    ]
                )
                - sum(
                    len(groups)
                    for groups in after["action_field_fragmentation"][
                        "duplicate_groups"
                    ]
                )
            ),
        }

    def _calculate_connectivity_improvement(
        self, before: dict[str, Any], after: dict[str, Any]
    ) -> dict[str, Any]:
        """Calculate improvement in connectivity metrics."""
        return {
            "density_change": after["density"] - before["density"],
            "isolated_nodes_change": before["isolated_nodes"] - after["isolated_nodes"],
            "component_reduction": before["connected_components"]
            - after["connected_components"],
        }

    def _calculate_edge_consistency_improvement(
        self, before: dict[str, Any], after: dict[str, Any]
    ) -> dict[str, Any]:
        """Calculate improvement in edge consistency metrics."""
        return {
            "zero_connections_reduced": (
                before["zero_connection_projects"] - after["zero_connection_projects"]
            ),
            "measure_consistency_improvement": (
                before["measure_connections"]["coefficient_of_variation"]
                - after["measure_connections"]["coefficient_of_variation"]
            ),
        }

    def _empty_metrics(self) -> dict[str, Any]:
        """Return empty metrics structure."""
        return {
            "timestamp": self._get_timestamp(),
            "basic_stats": {},
            "fragmentation_metrics": {},
            "connectivity_metrics": {},
            "edge_consistency_metrics": {},
            "overall_quality_score": 0.0,
        }

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime

        return datetime.now().isoformat()


def analyze_graph_quality(
    structures: list[dict[str, Any]], before_resolution: bool = False
) -> dict[str, Any]:
    """
    Convenience function to analyze graph quality.

    Args:
        structures: List of extraction structures
        before_resolution: Whether this is before entity resolution

    Returns:
        Dictionary of quality metrics
    """
    analyzer = GraphQualityAnalyzer()
    return analyzer.analyze_extraction_quality(structures, before_resolution)


def print_quality_report(metrics: dict[str, Any]) -> None:
    """
    Print a formatted quality report.

    Args:
        metrics: Quality metrics dictionary
    """
    if not metrics:
        print("📊 No quality metrics available")
        return

    stage = metrics.get("stage", "unknown")
    score = metrics.get("overall_quality_score", 0)

    print(f"\n📊 Graph Quality Report ({stage})")
    print("=" * 50)
    print(f"Overall Quality Score: {score}/1000")

    # Basic stats
    basic = metrics.get("basic_stats", {})
    if basic:
        print("\n📈 Basic Statistics:")
        print(f"  Action Fields: {basic.get('total_action_fields', 0)}")
        print(f"  Projects: {basic.get('total_projects', 0)}")
        print(f"  Measures: {basic.get('total_measures', 0)}")
        print(f"  Indicators: {basic.get('total_indicators', 0)}")

    # Fragmentation
    frag = metrics.get("fragmentation_metrics", {})
    if frag:
        print("\n🔗 Fragmentation Analysis:")
        print(
            f"  Overall Fragmentation: {frag.get('overall_fragmentation_score', 0):.3f}"
        )
        af_frag = frag.get("action_field_fragmentation", {})
        if af_frag:
            print(
                f"  Action Field Duplicates: {af_frag.get('potential_duplicates', 0)}"
            )

    # Connectivity
    conn = metrics.get("connectivity_metrics", {})
    if conn:
        print("\n🌐 Connectivity Analysis:")
        print(f"  Total Nodes: {conn.get('total_nodes', 0)}")
        print(f"  Total Edges: {conn.get('total_edges', 0)}")
        print(f"  Graph Density: {conn.get('density', 0):.3f}")
        print(f"  Isolated Nodes: {conn.get('isolated_nodes', 0)}")

    # Edge consistency
    edge = metrics.get("edge_consistency_metrics", {})
    if edge:
        print("\n⚖️ Edge Consistency:")
        print(
            f"  Projects with Zero Connections: {edge.get('zero_connection_projects', 0)}"
        )

    print("=" * 50)


def print_improvement_report(improvement_metrics: dict[str, Any]) -> None:
    """
    Print a formatted improvement report.

    Args:
        improvement_metrics: Improvement metrics dictionary
    """
    if not improvement_metrics:
        print("📊 No improvement metrics available")
        return

    print("\n📈 Entity Resolution Improvement Report")
    print("=" * 50)

    overall_improvement = improvement_metrics.get("overall_score_improvement", 0)
    print(f"Overall Score Improvement: +{overall_improvement:.1f} points")

    # Fragmentation improvements
    frag_imp = improvement_metrics.get("fragmentation_improvement", {})
    if frag_imp:
        ratio_change = frag_imp.get("fragmentation_ratio_change", 0)
        groups_reduced = frag_imp.get("duplicate_groups_reduced", 0)
        print("\n🔗 Fragmentation Improvements:")
        print(f"  Fragmentation Ratio Change: {ratio_change:+.3f}")
        print(f"  Duplicate Groups Reduced: {groups_reduced}")

    # Connectivity improvements
    conn_imp = improvement_metrics.get("connectivity_improvement", {})
    if conn_imp:
        density_change = conn_imp.get("density_change", 0)
        isolated_change = conn_imp.get("isolated_nodes_change", 0)
        print("\n🌐 Connectivity Improvements:")
        print(f"  Density Change: {density_change:+.3f}")
        print(f"  Isolated Nodes Reduced: {isolated_change}")

    # Edge consistency improvements
    edge_imp = improvement_metrics.get("edge_consistency_improvement", {})
    if edge_imp:
        zero_reduced = edge_imp.get("zero_connections_reduced", 0)
        print("\n⚖️ Edge Consistency Improvements:")
        print(f"  Zero-Connection Projects Reduced: {zero_reduced}")

    print("=" * 50)
