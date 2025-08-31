"""
Graph metrics calculation for JSON quality analysis.

Calculates core graph statistics including node counts, edge counts,
degree statistics, and connectivity metrics.
"""

from collections import Counter, defaultdict
from statistics import mean, median
from typing import Any

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

            self._add_projects_for_action_field(graph, af_node, action_field_data)

        return graph

    def _add_projects_for_action_field(
        self, graph: nx.Graph, af_node: str, action_field_data: dict[str, Any]
    ) -> None:
        """Add projects and their measures/indicators for an action field."""
        for project in action_field_data.get("projects", []):
            project_title = project.get("title", "")
            if not project_title:
                continue

            proj_node = f"proj:{project_title}"
            graph.add_node(proj_node, type="project", name=project_title)
            graph.add_edge(af_node, proj_node, relation="af_to_proj")

            self._add_measures_for_project(graph, proj_node, project)
            self._add_indicators_for_project(graph, proj_node, project)

    def _add_measures_for_project(
        self, graph: nx.Graph, proj_node: str, project: dict[str, Any]
    ) -> None:
        """Add measures for a project."""
        for measure in project.get("measures", []):
            if measure:
                msr_node = f"msr:{measure}"
                graph.add_node(msr_node, type="measure", name=measure)
                graph.add_edge(proj_node, msr_node, relation="proj_to_msr")

    def _add_indicators_for_project(
        self, graph: nx.Graph, proj_node: str, project: dict[str, Any]
    ) -> None:
        """Add indicators for a project."""
        for indicator in project.get("indicators", []):
            if indicator:
                ind_node = f"ind:{indicator}"
                graph.add_node(ind_node, type="indicator", name=indicator)
                graph.add_edge(proj_node, ind_node, relation="proj_to_ind")

    def _build_from_enriched_review(
        self, data: dict[str, Any], graph: nx.Graph
    ) -> nx.Graph:
        """Build graph from EnrichedReviewJSON format."""
        entity_types = ["action_fields", "projects", "measures", "indicators"]
        type_mapping = {
            "action_fields": "action_field",
            "projects": "project",
            "measures": "measure",
            "indicators": "indicator",
        }

        # Add all nodes first
        self._add_all_nodes_from_enriched_data(data, graph, entity_types, type_mapping)

        # Add edges based on connections
        self._add_all_edges_from_enriched_data(data, graph, entity_types)

        return graph

    def _add_all_nodes_from_enriched_data(
        self,
        data: dict[str, Any],
        graph: nx.Graph,
        entity_types: list[str],
        type_mapping: dict[str, str],
    ) -> None:
        """Add all nodes from enriched review data to the graph."""
        for entity_type in entity_types:
            for entity in data.get(entity_type, []):
                entity_id = entity.get("id", "")
                if not entity_id:
                    continue

                # Get name from content
                content = entity.get("content", {})
                name = content.get("title") or content.get("name", "")

                graph.add_node(entity_id, type=type_mapping[entity_type], name=name)

    def _add_all_edges_from_enriched_data(
        self, data: dict[str, Any], graph: nx.Graph, entity_types: list[str]
    ) -> None:
        """Add all edges from enriched review data to the graph."""
        for entity_type in entity_types:
            for entity in data.get(entity_type, []):
                entity_id = entity.get("id", "")
                if not entity_id:
                    continue

                self._add_entity_connections_to_graph(entity, entity_id, graph)

    def _add_entity_connections_to_graph(
        self, entity: dict[str, Any], entity_id: str, graph: nx.Graph
    ) -> None:
        """Add connections for a single entity to the graph as edges."""
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
