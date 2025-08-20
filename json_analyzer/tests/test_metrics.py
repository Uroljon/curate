"""
Tests for individual metric calculators.
"""

import networkx as nx
from collections import defaultdict

from ..config import (
    GraphThresholds, IntegrityThresholds, ConnectivityThresholds,
    ConfidenceThresholds, SourceThresholds, ContentThresholds, AnalyzerConfig
)
from ..metrics import (
    GraphMetrics, IntegrityMetrics, ConnectivityMetrics, 
    ConfidenceMetrics, SourceMetrics, ContentMetrics, DriftMetrics
)


def create_test_graph():
    """Create a simple test graph."""
    G = nx.Graph()
    G.add_node("af_1", type="action_field", name="Action Field 1")
    G.add_node("proj_1", type="project", name="Project 1")
    G.add_node("msr_1", type="measure", name="Measure 1")
    G.add_node("ind_1", type="indicator", name="Indicator 1")
    
    G.add_edge("af_1", "proj_1", relation="af_to_proj")
    G.add_edge("proj_1", "msr_1", relation="proj_to_msr")
    G.add_edge("proj_1", "ind_1", relation="proj_to_ind")
    
    return G


def create_test_data():
    """Create sample test data."""
    return {
        "action_fields": [
            {
                "id": "af_1",
                "content": {"title": "Action Field 1", "name": "Action Field 1"},
                "connections": [{"target_id": "proj_1", "confidence_score": 0.9}]
            }
        ],
        "projects": [
            {
                "id": "proj_1",
                "content": {"title": "Project 1"},
                "connections": [
                    {"target_id": "af_1", "confidence_score": 0.9},
                    {"target_id": "msr_1", "confidence_score": 0.8}
                ]
            }
        ],
        "measures": [
            {
                "id": "msr_1",
                "content": {"title": "Measure 1"},
                "connections": [{"target_id": "proj_1", "confidence_score": 0.8}],
                "sources": [{"page_number": 1, "quote": "Test quote", "chunk_id": 1}]
            }
        ],
        "indicators": [
            {
                "id": "ind_1", 
                "content": {"title": "Indicator 1", "description": "Test indicator"},
                "connections": [{"target_id": "proj_1", "confidence_score": 0.7}]
            }
        ]
    }


def test_graph_metrics():
    """Test graph metrics calculation."""
    metrics = GraphMetrics(GraphThresholds())
    graph = create_test_graph()
    test_data = create_test_data()
    
    result = metrics.calculate(graph, test_data)
    
    assert result.total_nodes == 4
    assert result.total_edges == 3
    assert result.isolated_nodes == 0
    assert result.components == 1
    assert result.avg_degree > 0
    
    # Test node type counts
    assert result.nodes_by_type["action_field"] == 1
    assert result.nodes_by_type["project"] == 1
    assert result.nodes_by_type["measure"] == 1
    assert result.nodes_by_type["indicator"] == 1
    
    print("✅ Graph metrics test passed")


def test_integrity_metrics():
    """Test integrity metrics calculation."""
    metrics = IntegrityMetrics(IntegrityThresholds())
    test_data = create_test_data()
    
    result = metrics.calculate(test_data)
    
    # Should have no dangling references in clean test data
    assert len(result.dangling_refs) == 0
    
    # Check ID validity
    assert result.id_validity["total_ids"] == 4
    assert result.id_validity["duplicate_count"] == 0
    
    # Check field completeness
    assert "action_fields" in result.field_completeness
    assert "projects" in result.field_completeness
    
    print("✅ Integrity metrics test passed")


def test_integrity_metrics_with_issues():
    """Test integrity metrics with data issues."""
    metrics = IntegrityMetrics(IntegrityThresholds())
    
    # Create data with issues
    bad_data = {
        "action_fields": [
            {
                "id": "af_1",
                "content": {"title": "Action Field 1"},
                "connections": [
                    {"target_id": "nonexistent_proj", "confidence_score": 0.9}  # Dangling ref
                ]
            }
        ],
        "projects": [
            {
                "id": "af_1",  # Duplicate ID
                "content": {"title": "Project 1"},
                "connections": []
            }
        ],
        "measures": [],
        "indicators": []
    }
    
    result = metrics.calculate(bad_data)
    
    # Should detect dangling reference
    assert len(result.dangling_refs) == 1
    assert result.dangling_refs[0]["target_id"] == "nonexistent_proj"
    
    # Should detect duplicate ID
    assert result.id_validity["duplicate_count"] == 1
    
    print("✅ Integrity metrics with issues test passed")


def test_connectivity_metrics():
    """Test connectivity metrics calculation."""
    metrics = ConnectivityMetrics(ConnectivityThresholds())
    graph = create_test_graph()
    test_data = create_test_data()
    
    result = metrics.calculate(graph, test_data)
    
    # Action field coverage should be 100% (1 AF with 1 connection)
    assert result.action_field_coverage == 1.0
    
    # Project coverage should be 100% (1 project connected to AF and measure)
    assert result.project_coverage == 1.0
    
    # Should have measures per project stats
    assert "mean" in result.measures_per_project
    assert result.measures_per_project["mean"] == 1.0  # 1 measure per 1 project
    
    print("✅ Connectivity metrics test passed")


def test_confidence_metrics():
    """Test confidence metrics calculation.""" 
    metrics = ConfidenceMetrics(ConfidenceThresholds())
    graph = create_test_graph()
    test_data = create_test_data()
    
    result = metrics.calculate(graph, test_data)
    
    # Check confidence statistics by type
    assert len(result.edge_confidence_by_type) > 0
    
    # Should have some confidence distributions
    assert len(result.confidence_distribution) > 0
    
    # Should have calibration metrics
    assert "overall_mean_confidence" in result.calibration_metrics
    
    print("✅ Confidence metrics test passed")


def test_source_metrics():
    """Test source metrics calculation."""
    metrics = SourceMetrics(SourceThresholds())
    test_data = create_test_data()
    
    result = metrics.calculate(test_data)
    
    # Should calculate source coverage
    assert "measures" in result.source_coverage
    assert result.source_coverage["measures"] == 1.0  # 1/1 measures have sources
    
    # Quote match rate should be 0 (no page text provided)
    assert result.quote_match_rate == 0.0
    
    # Should have evidence density
    assert "measures" in result.evidence_density
    assert result.evidence_density["measures"] == 1.0  # 1 source per measure
    
    print("✅ Source metrics test passed")


def test_content_metrics():
    """Test content metrics calculation."""
    metrics = ContentMetrics(ContentThresholds())
    test_data = create_test_data()
    
    result = metrics.calculate(test_data)
    
    # Should calculate repetition rate
    assert isinstance(result.repetition_rate, float)
    assert result.repetition_rate >= 0.0
    
    # Should analyze length distributions
    assert len(result.length_distribution) > 0
    
    # Should check language consistency
    assert "primary_language" in result.language_consistency
    
    print("✅ Content metrics test passed")


def test_drift_metrics():
    """Test drift metrics calculation."""
    config = AnalyzerConfig()
    metrics = DriftMetrics(config)
    
    test_data1 = create_test_data()
    test_data2 = create_test_data()
    
    # Modify second dataset
    test_data2["action_fields"].append({
        "id": "af_2",
        "content": {"title": "New Action Field", "name": "New Action Field"},
        "connections": []
    })
    
    result = metrics.calculate(test_data1, test_data2)
    
    # Should detect node churn
    assert "action_fields" in result.node_churn
    assert result.node_churn["action_fields"]["added"] == 1
    
    # Should have stability score
    assert 0 <= result.stability_score <= 100
    
    # Should have structural similarity
    assert 0 <= result.structural_similarity <= 1
    
    print("✅ Drift metrics test passed")


def test_graph_building():
    """Test graph building from different data formats."""
    metrics = GraphMetrics(GraphThresholds())
    
    # Test EnrichedReviewJSON format
    enriched_data = create_test_data()
    graph = metrics.build_graph_from_data(enriched_data)
    
    assert graph.number_of_nodes() == 4
    assert graph.number_of_edges() > 0
    
    # Test hierarchical format
    hierarchical_data = {
        "action_fields": [
            {
                "action_field": "Test Field",
                "projects": [
                    {
                        "title": "Test Project",
                        "measures": ["Measure 1"],
                        "indicators": ["Indicator 1"]
                    }
                ]
            }
        ]
    }
    
    graph2 = metrics.build_graph_from_data(hierarchical_data)
    assert graph2.number_of_nodes() > 0
    
    print("✅ Graph building test passed")


if __name__ == "__main__":
    # Run all tests
    test_graph_metrics()
    test_integrity_metrics()
    test_integrity_metrics_with_issues()
    test_connectivity_metrics()
    test_confidence_metrics()
    test_source_metrics()
    test_content_metrics()
    test_drift_metrics()
    test_graph_building()
    
    print("\n🎉 All metric tests passed!")