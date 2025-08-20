"""
Tests for the main JSON analyzer.
"""

import json
import tempfile
from pathlib import Path

from ..analyzer import JSONAnalyzer
from ..config import AnalyzerConfig
from ..models import AnalysisResult


def create_test_data():
    """Create sample test data in EnrichedReviewJSON format."""
    return {
        "action_fields": [
            {
                "id": "af_1",
                "content": {"title": "Test Action Field", "name": "Test Action Field"},
                "connections": [{"target_id": "proj_1", "confidence_score": 0.9}],
            }
        ],
        "projects": [
            {
                "id": "proj_1",
                "content": {"title": "Test Project"},
                "connections": [
                    {"target_id": "af_1", "confidence_score": 0.9},
                    {"target_id": "msr_1", "confidence_score": 0.8},
                ],
            }
        ],
        "measures": [
            {
                "id": "msr_1",
                "content": {"title": "Test Measure"},
                "connections": [{"target_id": "proj_1", "confidence_score": 0.8}],
                "sources": [
                    {"page_number": 1, "quote": "This is a test quote", "chunk_id": 1}
                ],
            }
        ],
        "indicators": [
            {
                "id": "ind_1",
                "content": {
                    "title": "Test Indicator",
                    "description": "A test indicator for testing purposes",
                },
                "connections": [{"target_id": "msr_1", "confidence_score": 0.7}],
            }
        ],
    }


def test_analyzer_initialization():
    """Test analyzer initialization with default config."""
    analyzer = JSONAnalyzer()
    assert analyzer.config is not None
    assert analyzer.graph_metrics is not None
    assert analyzer.integrity_metrics is not None


def test_analyzer_initialization_with_custom_config():
    """Test analyzer initialization with custom config."""
    config = AnalyzerConfig()
    config.integrity_thresholds.max_duplicate_rate = 0.1

    analyzer = JSONAnalyzer(config)
    assert analyzer.config.integrity_thresholds.max_duplicate_rate == 0.1


def test_analyze_data_basic():
    """Test basic data analysis functionality."""
    analyzer = JSONAnalyzer()
    test_data = create_test_data()

    result = analyzer.analyze_data(test_data)

    assert isinstance(result, AnalysisResult)
    assert result.metadata.format_detected == "EnrichedReviewJSON"
    assert result.graph_stats.total_nodes == 4
    assert result.graph_stats.total_edges > 0
    assert 0 <= result.quality_score.overall_score <= 100


def test_analyze_file():
    """Test analyzing a JSON file."""
    analyzer = JSONAnalyzer()
    test_data = create_test_data()

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(test_data, f)
        temp_path = f.name

    try:
        result = analyzer.analyze_file(temp_path)

        assert isinstance(result, AnalysisResult)
        assert result.metadata.file_path == temp_path
        assert result.metadata.format_detected == "EnrichedReviewJSON"
        assert result.graph_stats.total_nodes == 4

    finally:
        Path(temp_path).unlink()  # Clean up


def test_format_detection():
    """Test format detection for different JSON structures."""
    analyzer = JSONAnalyzer()

    # Test EnrichedReviewJSON format
    enriched_data = create_test_data()
    result = analyzer.analyze_data(enriched_data)
    assert result.metadata.format_detected == "EnrichedReviewJSON"

    # Test empty data
    empty_data = {"action_fields": []}
    result = analyzer.analyze_data(empty_data)
    assert result.metadata.format_detected == "empty"

    # Test hierarchical format (ExtractionResult)
    hierarchical_data = {
        "action_fields": [
            {
                "action_field": "Test Field",
                "projects": [
                    {
                        "title": "Test Project",
                        "measures": ["measure 1"],
                        "indicators": ["indicator 1"],
                    }
                ],
            }
        ]
    }
    result = analyzer.analyze_data(hierarchical_data)
    assert result.metadata.format_detected == "ExtractionResult"


def test_quality_score_calculation():
    """Test quality score calculation."""
    analyzer = JSONAnalyzer()
    test_data = create_test_data()

    result = analyzer.analyze_data(test_data)

    # Basic quality score checks
    assert 0 <= result.quality_score.overall_score <= 100
    assert result.quality_score.grade in ["A", "B", "C", "D", "F"]
    assert len(result.quality_score.category_scores) == 5

    # Check all categories are present
    expected_categories = [
        "integrity",
        "connectivity",
        "confidence",
        "sources",
        "content",
    ]
    for category in expected_categories:
        assert category in result.quality_score.category_scores
        assert 0 <= result.quality_score.category_scores[category] <= 100


def test_compare_files():
    """Test file comparison functionality."""
    analyzer = JSONAnalyzer()
    test_data1 = create_test_data()
    test_data2 = create_test_data()

    # Modify second dataset slightly
    test_data2["action_fields"].append(
        {
            "id": "af_2",
            "content": {"title": "New Action Field", "name": "New Action Field"},
            "connections": [],
        }
    )

    # Create temporary files
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f1:
        json.dump(test_data1, f1)
        temp_path1 = f1.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f2:
        json.dump(test_data2, f2)
        temp_path2 = f2.name

    try:
        result = analyzer.compare_files(temp_path1, temp_path2)

        assert (
            result.before.graph_stats.total_nodes < result.after.graph_stats.total_nodes
        )
        assert result.drift_stats is not None
        assert isinstance(result.improvements, dict)
        assert isinstance(result.regressions, dict)

    finally:
        Path(temp_path1).unlink()
        Path(temp_path2).unlink()


def test_analysis_summary():
    """Test analysis summary generation."""
    analyzer = JSONAnalyzer()
    test_data = create_test_data()

    result = analyzer.analyze_data(test_data)
    summary = analyzer.get_analysis_summary(result)

    assert "overall_score" in summary
    assert "grade" in summary
    assert "format" in summary
    assert "total_entities" in summary
    assert "total_connections" in summary
    assert "critical_issues" in summary
    assert "analysis_time_ms" in summary

    # Check value types
    assert isinstance(summary["overall_score"], float)
    assert summary["grade"] in ["A", "B", "C", "D", "F"]
    assert isinstance(summary["total_entities"], int)
    assert isinstance(summary["total_connections"], int)


if __name__ == "__main__":
    # Run basic tests
    test_analyzer_initialization()
    print("✅ Analyzer initialization test passed")

    test_analyze_data_basic()
    print("✅ Basic data analysis test passed")

    test_format_detection()
    print("✅ Format detection test passed")

    test_quality_score_calculation()
    print("✅ Quality score calculation test passed")

    test_analysis_summary()
    print("✅ Analysis summary test passed")

    test_analyze_file()
    print("✅ File analysis test passed")

    test_compare_files()
    print("✅ File comparison test passed")

    print("\n🎉 All tests passed!")
