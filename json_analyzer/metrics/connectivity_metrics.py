"""
Connectivity metrics calculation for JSON quality analysis.

Analyzes graph connectivity, coverage metrics, path lengths,
centrality measures, and structural patterns.
"""

from collections import Counter, defaultdict
from statistics import mean, median
from typing import Any, Dict, List, Set

import networkx as nx

from ..config import ConnectivityThresholds
from ..models import ConnectivityStats


class ConnectivityMetrics:
    """Calculator for connectivity and structural analysis metrics."""

    def __init__(self, thresholds: ConnectivityThresholds):
        self.thresholds = thresholds

    def calculate(self, graph: nx.Graph, data: dict[str, Any]) -> ConnectivityStats:
        """
        Calculate comprehensive connectivity statistics.

        Args:
            graph: NetworkX graph representation
            data: Original JSON data

        Returns:
            ConnectivityStats object with all metrics
        """
        if graph.number_of_nodes() == 0:
            return ConnectivityStats()

        # Coverage metrics
        af_coverage = self._calculate_action_field_coverage(graph, data)
        project_coverage = self._calculate_project_coverage(graph, data)

        # Measures per project distribution
        measures_per_project = self._calculate_measures_per_project(graph, data)

        # Path length analysis
        path_lengths = self._calculate_path_lengths(graph)

        # Centrality measures
        centrality_scores = self._calculate_centrality_measures(graph)

        # Cycle detection
        cycles = self._detect_cycles(graph)

        # Additional coverage metrics
        coverage_metrics = self._calculate_detailed_coverage(graph, data)

        return ConnectivityStats(
            action_field_coverage=af_coverage,
            project_coverage=project_coverage,
            measures_per_project=measures_per_project,
            path_lengths=path_lengths,
            centrality_scores=centrality_scores,
            cycles=cycles,
            coverage_metrics=coverage_metrics,
        )

    def _calculate_action_field_coverage(
        self, graph: nx.Graph, data: dict[str, Any]
    ) -> float:
        """Calculate percentage of action fields that have incoming connections."""
        af_nodes = [
            node
            for node, data_attr in graph.nodes(data=True)
            if data_attr.get("type") == "action_field"
        ]

        if not af_nodes:
            return 0.0

        connected_afs = 0
        for af_node in af_nodes:
            # Count incoming edges (projects connecting to this action field)
            incoming_edges = [edge for edge in graph.edges() if edge[1] == af_node]
            if incoming_edges:
                connected_afs += 1

        return connected_afs / len(af_nodes)

    def _calculate_project_coverage(
        self, graph: nx.Graph, data: dict[str, Any]
    ) -> float:
        """Calculate percentage of projects connected to both action fields and measures."""
        project_nodes = [
            node
            for node, data_attr in graph.nodes(data=True)
            if data_attr.get("type") == "project"
        ]

        if not project_nodes:
            return 0.0

        well_connected_projects = 0
        for proj_node in project_nodes:
            neighbors = list(graph.neighbors(proj_node))
            neighbor_types = [graph.nodes[n].get("type") for n in neighbors]

            has_af_connection = "action_field" in neighbor_types
            has_measure_connection = "measure" in neighbor_types

            if has_af_connection and has_measure_connection:
                well_connected_projects += 1

        return well_connected_projects / len(project_nodes)

    def _calculate_measures_per_project(
        self, graph: nx.Graph, data: dict[str, Any]
    ) -> dict[str, float]:
        """Calculate statistics on measures per project."""
        project_nodes = [
            node
            for node, data_attr in graph.nodes(data=True)
            if data_attr.get("type") == "project"
        ]

        if not project_nodes:
            return {"mean": 0.0, "median": 0.0, "min": 0, "max": 0, "std": 0.0}

        measures_counts = []
        for proj_node in project_nodes:
            neighbors = list(graph.neighbors(proj_node))
            measure_neighbors = [
                n for n in neighbors if graph.nodes[n].get("type") == "measure"
            ]
            measures_counts.append(len(measure_neighbors))

        return {
            "mean": mean(measures_counts) if measures_counts else 0.0,
            "median": median(measures_counts) if measures_counts else 0.0,
            "min": min(measures_counts) if measures_counts else 0,
            "max": max(measures_counts) if measures_counts else 0,
            "std": self._calculate_std(measures_counts) if measures_counts else 0.0,
            "distribution": dict(Counter(measures_counts)),
        }

    def _calculate_path_lengths(self, graph: nx.Graph) -> dict[str, float]:
        """Calculate path length statistics."""
        if graph.number_of_nodes() < 2:
            return {"avg_shortest_path": 0.0, "diameter": 0, "radius": 0}

        try:
            # Only calculate for the largest connected component
            largest_cc = max(nx.connected_components(graph), key=len)
            subgraph = graph.subgraph(largest_cc)

            if subgraph.number_of_nodes() < 2:
                return {"avg_shortest_path": 0.0, "diameter": 0, "radius": 0}

            # Calculate shortest paths
            path_lengths = []
            for source in subgraph.nodes():
                lengths = nx.single_source_shortest_path_length(subgraph, source)
                path_lengths.extend(lengths.values())

            # Remove zero-length paths (self-loops)
            path_lengths = [p for p in path_lengths if p > 0]

            return {
                "avg_shortest_path": mean(path_lengths) if path_lengths else 0.0,
                "diameter": max(path_lengths) if path_lengths else 0,
                "radius": min(path_lengths) if path_lengths else 0,
                "path_distribution": dict(Counter(path_lengths)),
            }

        except Exception:
            return {"avg_shortest_path": 0.0, "diameter": 0, "radius": 0}

    def _calculate_centrality_measures(
        self, graph: nx.Graph
    ) -> dict[str, dict[str, float]]:
        """Calculate various centrality measures."""
        if graph.number_of_nodes() == 0:
            return {}

        centrality_stats = {}

        try:
            # Degree centrality
            degree_centrality = nx.degree_centrality(graph)
            centrality_stats["degree"] = self._centrality_statistics(degree_centrality)

            # Only calculate other centralities for connected graphs
            if (
                nx.is_connected(graph)
                and graph.number_of_nodes()
                <= self.thresholds.min_centrality_diversity * 1000
            ):
                # Closeness centrality
                closeness_centrality = nx.closeness_centrality(graph)
                centrality_stats["closeness"] = self._centrality_statistics(
                    closeness_centrality
                )

                # Betweenness centrality (expensive, limit to smaller graphs)
                if graph.number_of_nodes() <= 1000:
                    betweenness_centrality = nx.betweenness_centrality(graph)
                    centrality_stats["betweenness"] = self._centrality_statistics(
                        betweenness_centrality
                    )

            # PageRank (works on disconnected graphs)
            try:
                pagerank = nx.pagerank(graph, max_iter=100)
                centrality_stats["pagerank"] = self._centrality_statistics(pagerank)
            except:
                pass

        except Exception:
            # Fallback to just degree centrality
            if graph.number_of_nodes() > 0:
                degrees = dict(graph.degree())
                max_degree = max(degrees.values()) if degrees else 1
                degree_centrality = {
                    node: deg / max_degree for node, deg in degrees.items()
                }
                centrality_stats["degree"] = self._centrality_statistics(
                    degree_centrality
                )

        return centrality_stats

    def _centrality_statistics(
        self, centrality_dict: dict[str, float]
    ) -> dict[str, float]:
        """Calculate statistics for a centrality measure."""
        if not centrality_dict:
            return {"mean": 0.0, "max": 0.0, "min": 0.0, "std": 0.0}

        values = list(centrality_dict.values())
        return {
            "mean": mean(values),
            "max": max(values),
            "min": min(values),
            "std": self._calculate_std(values),
        }

    def _detect_cycles(self, graph: nx.Graph) -> list[list[str]]:
        """Detect cycles in the graph."""
        try:
            # Convert to directed graph for cycle detection
            directed_graph = graph.to_directed()

            cycles = []
            try:
                # Find simple cycles (limit to avoid performance issues)
                cycle_iter = nx.simple_cycles(directed_graph)
                for i, cycle in enumerate(cycle_iter):
                    if i >= 20:  # Limit number of cycles to report
                        break
                    cycles.append(cycle)
            except:
                pass

            return cycles

        except Exception:
            return []

    def _calculate_detailed_coverage(
        self, graph: nx.Graph, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Calculate detailed coverage metrics."""
        node_types = ["action_field", "project", "measure", "indicator"]
        coverage_stats = {}

        for node_type in node_types:
            nodes = [
                node
                for node, data_attr in graph.nodes(data=True)
                if data_attr.get("type") == node_type
            ]

            if not nodes:
                coverage_stats[node_type] = {
                    "total": 0,
                    "connected": 0,
                    "isolated": 0,
                    "coverage_rate": 0.0,
                }
                continue

            connected_nodes = [node for node in nodes if graph.degree[node] > 0]
            isolated_nodes = [node for node in nodes if graph.degree[node] == 0]

            coverage_stats[node_type] = {
                "total": len(nodes),
                "connected": len(connected_nodes),
                "isolated": len(isolated_nodes),
                "coverage_rate": len(connected_nodes) / len(nodes) if nodes else 0.0,
                "avg_connections": (
                    mean([graph.degree[node] for node in nodes]) if nodes else 0.0
                ),
            }

        return coverage_stats

    def _calculate_std(self, values: list[float]) -> float:
        """Calculate standard deviation."""
        if len(values) <= 1:
            return 0.0

        avg = mean(values)
        variance = sum((x - avg) ** 2 for x in values) / len(values)
        return variance**0.5

    def get_connectivity_patterns(self, graph: nx.Graph) -> dict[str, Any]:
        """Analyze connectivity patterns between different node types."""
        patterns = defaultdict(int)

        for source, target in graph.edges():
            source_type = graph.nodes[source].get("type", "unknown")
            target_type = graph.nodes[target].get("type", "unknown")

            # Create pattern key
            pattern = f"{source_type}_to_{target_type}"
            patterns[pattern] += 1

            # Also count reverse pattern for undirected graphs
            reverse_pattern = f"{target_type}_to_{source_type}"
            if reverse_pattern != pattern:
                patterns[reverse_pattern] += 1

        return {
            "patterns": dict(patterns),
            "most_common_patterns": sorted(
                patterns.items(), key=lambda x: x[1], reverse=True
            )[:10],
        }

    def get_hub_analysis(
        self, graph: nx.Graph, top_n: int = 10
    ) -> dict[str, list[dict[str, Any]]]:
        """Identify hub nodes (high degree nodes) by type."""
        hubs_by_type = defaultdict(list)

        # Get degree for each node
        degrees = dict(graph.degree())

        # Group by type and sort by degree
        nodes_by_type = defaultdict(list)
        for node, data in graph.nodes(data=True):
            node_type = data.get("type", "unknown")
            nodes_by_type[node_type].append(
                {"node": node, "name": data.get("name", ""), "degree": degrees[node]}
            )

        # Sort and take top N for each type
        for node_type, nodes in nodes_by_type.items():
            sorted_nodes = sorted(nodes, key=lambda x: x["degree"], reverse=True)
            hubs_by_type[node_type] = sorted_nodes[:top_n]

        return dict(hubs_by_type)
