"""
Graph metrics calculation for JSON quality analysis.

Calculates core graph statistics including node counts, edge counts,
degree statistics, and connectivity metrics.
"""

from collections import Counter, defaultdict
from statistics import mean, median
from typing import Any, Dict, List, Tuple

import networkx as nx

from ..config import GraphThresholds
from ..models import GraphStats


class GraphMetrics:
    """Calculator for core graph statistics."""

    def __init__(self, thresholds: GraphThresholds):
        self.thresholds = thresholds

    def calculate(self, graph: nx.Graph, data: dict[str, Any]) -> GraphStats:
        """
        Calculate comprehensive graph statistics.

        Args:
            graph: NetworkX graph representation
            data: Original JSON data

        Returns:
            GraphStats object with all metrics
        """
        if graph.number_of_nodes() == 0:
            return GraphStats()

        # Basic node and edge counts
        nodes_by_type = self._count_nodes_by_type(graph)
        edges_by_relation = self._count_edges_by_relation(graph)

        # Connectivity metrics
        components = list(nx.connected_components(graph))
        isolated_nodes = [node for node in graph.nodes() if graph.degree[node] == 0]

        # Degree statistics
        degrees = [graph.degree[node] for node in graph.nodes()]
        degree_by_type = self._calculate_degree_by_type(graph)

        return GraphStats(
            nodes_by_type=nodes_by_type,
            edges_by_relation=edges_by_relation,
            total_nodes=graph.number_of_nodes(),
            total_edges=graph.number_of_edges(),
            isolated_nodes=len(isolated_nodes),
            components=len(components),
            largest_component_size=(
                max(len(comp) for comp in components) if components else 0
            ),
            avg_degree=mean(degrees) if degrees else 0.0,
            median_degree=median(degrees) if degrees else 0.0,
            max_degree=max(degrees) if degrees else 0,
            degree_by_type=degree_by_type,
        )

    def _count_nodes_by_type(self, graph: nx.Graph) -> dict[str, int]:
        """Count nodes by their type."""
        type_counts = Counter()

        for _node, data in graph.nodes(data=True):
            node_type = data.get("type", "unknown")
            type_counts[node_type] += 1

        return dict(type_counts)

    def _count_edges_by_relation(self, graph: nx.Graph) -> dict[str, int]:
        """Count edges by relation type (inferred from node types)."""
        relation_counts = Counter()

        for source, target, _data in graph.edges(data=True):
            # Get node types
            source_type = graph.nodes[source].get("type", "unknown")
            target_type = graph.nodes[target].get("type", "unknown")

            # Create relation type
            relation = f"{source_type}_to_{target_type}"
            relation_counts[relation] += 1

            # Also count reverse relation for undirected graphs
            reverse_relation = f"{target_type}_to_{source_type}"
            if reverse_relation != relation:
                relation_counts[reverse_relation] += 1

        return dict(relation_counts)

    def _calculate_degree_by_type(self, graph: nx.Graph) -> dict[str, dict[str, float]]:
        """Calculate degree statistics by node type."""
        degree_by_type = defaultdict(list)

        # Collect degrees by type
        for node, data in graph.nodes(data=True):
            node_type = data.get("type", "unknown")
            degree = graph.degree[node]
            degree_by_type[node_type].append(degree)

        # Calculate statistics for each type
        stats_by_type = {}
        for node_type, degrees in degree_by_type.items():
            if degrees:
                stats_by_type[node_type] = {
                    "mean": mean(degrees),
                    "median": median(degrees),
                    "max": max(degrees),
                    "min": min(degrees),
                    "std": self._calculate_std(degrees),
                }
            else:
                stats_by_type[node_type] = {
                    "mean": 0.0,
                    "median": 0.0,
                    "max": 0,
                    "min": 0,
                    "std": 0.0,
                }

        return stats_by_type

    def _calculate_std(self, values: list[float]) -> float:
        """Calculate standard deviation."""
        if len(values) <= 1:
            return 0.0

        avg = mean(values)
        variance = sum((x - avg) ** 2 for x in values) / len(values)
        return variance**0.5

    def build_graph_from_data(self, data: dict[str, Any]) -> nx.Graph:
        """
        Build NetworkX graph from JSON data.

        Supports both ExtractionResult and EnrichedReviewJSON formats.
        """
        graph = nx.Graph()

        # Detect format
        if "action_fields" in data and isinstance(data["action_fields"], list):
            if (
                len(data["action_fields"]) > 0
                and "projects" in data["action_fields"][0]
            ):
                # ExtractionResult format (hierarchical)
                return self._build_from_extraction_result(data, graph)
            else:
                # EnrichedReviewJSON format (flat with connections)
                return self._build_from_enriched_review(data, graph)

        return graph

    def _build_from_extraction_result(
        self, data: dict[str, Any], graph: nx.Graph
    ) -> nx.Graph:
        """Build graph from ExtractionResult format."""
        for action_field_data in data.get("action_fields", []):
            action_field = action_field_data.get("action_field", "")
            if not action_field:
                continue

            af_node = f"af:{action_field}"
            graph.add_node(af_node, type="action_field", name=action_field)

            # Add projects
            for project in action_field_data.get("projects", []):
                project_title = project.get("title", "")
                if not project_title:
                    continue

                proj_node = f"proj:{project_title}"
                graph.add_node(proj_node, type="project", name=project_title)
                graph.add_edge(af_node, proj_node, relation="af_to_proj")

                # Add measures
                for measure in project.get("measures", []):
                    if measure:
                        msr_node = f"msr:{measure}"
                        graph.add_node(msr_node, type="measure", name=measure)
                        graph.add_edge(proj_node, msr_node, relation="proj_to_msr")

                # Add indicators
                for indicator in project.get("indicators", []):
                    if indicator:
                        ind_node = f"ind:{indicator}"
                        graph.add_node(ind_node, type="indicator", name=indicator)
                        graph.add_edge(proj_node, ind_node, relation="proj_to_ind")

        return graph

    def _build_from_enriched_review(
        self, data: dict[str, Any], graph: nx.Graph
    ) -> nx.Graph:
        """Build graph from EnrichedReviewJSON format."""
        # Add nodes for each entity type
        entity_types = ["action_fields", "projects", "measures", "indicators"]
        type_mapping = {
            "action_fields": "action_field",
            "projects": "project",
            "measures": "measure",
            "indicators": "indicator",
        }

        # Add all nodes first
        for entity_type in entity_types:
            for entity in data.get(entity_type, []):
                entity_id = entity.get("id", "")
                if not entity_id:
                    continue

                # Get name from content
                content = entity.get("content", {})
                name = content.get("title") or content.get("name", "")

                graph.add_node(entity_id, type=type_mapping[entity_type], name=name)

        # Add edges based on connections
        for entity_type in entity_types:
            for entity in data.get(entity_type, []):
                entity_id = entity.get("id", "")
                if not entity_id:
                    continue

                # Add connections as edges
                for connection in entity.get("connections", []):
                    target_id = connection.get("target_id", "")
                    confidence = connection.get("confidence_score", 1.0)

                    if target_id and target_id in graph.nodes:
                        graph.add_edge(
                            entity_id,
                            target_id,
                            confidence=confidence,
                            relation=f"{entity_id.split('_')[0]}_to_{target_id.split('_')[0]}",
                        )

        return graph

    def get_component_analysis(self, graph: nx.Graph) -> dict[str, Any]:
        """Get detailed analysis of graph components."""
        if graph.number_of_nodes() == 0:
            return {}

        components = list(nx.connected_components(graph))
        component_sizes = [len(comp) for comp in components]

        return {
            "num_components": len(components),
            "component_sizes": component_sizes,
            "largest_component": max(component_sizes) if component_sizes else 0,
            "smallest_component": min(component_sizes) if component_sizes else 0,
            "avg_component_size": mean(component_sizes) if component_sizes else 0,
            "singleton_components": sum(1 for size in component_sizes if size == 1),
        }

    def get_degree_distribution(self, graph: nx.Graph) -> dict[int, int]:
        """Get degree distribution as a histogram."""
        degrees = [graph.degree[node] for node in graph.nodes()]
        return dict(Counter(degrees))
