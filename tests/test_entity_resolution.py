"""
Tests for entity resolution system.

This module tests the EntityResolver class and its integration with
the extraction pipeline to ensure node fragmentation is properly addressed.
"""

from unittest.mock import Mock, patch

import pytest

from src.processing.entity_resolver import EntityResolver, resolve_extraction_entities
from src.utils.graph_quality import GraphQualityAnalyzer


class TestEntityResolver:
    """Test cases for the EntityResolver class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.resolver = EntityResolver()

    def test_init(self):
        """Test EntityResolver initialization."""
        assert self.resolver.embedding_model is None
        assert len(self.resolver.german_patterns) > 0

    def test_extract_entities_action_fields(self):
        """Test extracting action field entities."""
        structures = [
            {"action_field": "Klimaschutz", "projects": []},
            {"action_field": "Mobilität", "projects": []},
            {"action_field": None, "projects": []},  # Should be filtered out
        ]

        entities = self.resolver._extract_entities(structures, "action_field")

        assert len(entities) == 2
        assert entities[0]["name"] == "Klimaschutz"
        assert entities[1]["name"] == "Mobilität"
        assert entities[0]["structure_index"] == 0

    def test_extract_entities_projects(self):
        """Test extracting project entities."""
        structures = [
            {
                "action_field": "Klimaschutz",
                "projects": [
                    {"title": "Solar Initiative", "measures": []},
                    {"title": "Wind Power", "measures": []},
                ],
            }
        ]

        entities = self.resolver._extract_entities(structures, "project")

        assert len(entities) == 2
        assert entities[0]["name"] == "Solar Initiative"
        assert entities[1]["name"] == "Wind Power"
        assert entities[0]["parent_action_field"] == "Klimaschutz"

    def test_get_canonical_name_climate(self):
        """Test canonical name generation for climate patterns."""
        # Test climate consolidation
        assert (
            self.resolver._get_canonical_name("Klimaschutz")
            == "Klimaschutz und Klimaanpassung"
        )
        assert (
            self.resolver._get_canonical_name("Klimaanpassung")
            == "Klimaschutz und Klimaanpassung"
        )
        assert (
            self.resolver._get_canonical_name("Klimaschutz und Klimaanpassung")
            == "Klimaschutz und Klimaanpassung"
        )

        # Test non-matching patterns
        assert self.resolver._get_canonical_name("Mobilität") is None
        assert self.resolver._get_canonical_name("Energie") is None

    def test_get_canonical_name_settlement(self):
        """Test canonical name generation for settlement patterns."""
        assert (
            self.resolver._get_canonical_name("Siedlungsentwicklung")
            == "Siedlungs- und Quartiersentwicklung"
        )
        assert (
            self.resolver._get_canonical_name("Quartiersentwicklung")
            == "Siedlungs- und Quartiersentwicklung"
        )

    def test_find_rule_based_groups(self):
        """Test rule-based grouping."""
        entities = [
            {"name": "Klimaschutz"},
            {"name": "Klimaanpassung"},
            {"name": "Mobilität"},
            {"name": "Energie"},
        ]

        groups = self.resolver._find_rule_based_groups(entities)

        # Should find one group with climate entities
        assert len(groups) == 1
        climate_group = groups[0]
        assert len(climate_group) == 2
        names = {entity["name"] for entity in climate_group}
        assert "Klimaschutz" in names
        assert "Klimaanpassung" in names

    def test_consolidate_action_fields(self):
        """Test action field consolidation."""
        structures = [
            {"action_field": "Klimaschutz", "projects": [{"title": "Solar Project"}]},
            {
                "action_field": "Klimaanpassung",
                "projects": [{"title": "Flood Protection"}],
            },
            {"action_field": "Mobilität", "projects": [{"title": "Bike Lanes"}]},
        ]

        name_mapping = {
            "Klimaschutz": "Klimaschutz und Klimaanpassung",
            "Klimaanpassung": "Klimaschutz und Klimaanpassung",
        }

        merge_groups = [[{"name": "Klimaschutz"}, {"name": "Klimaanpassung"}]]

        consolidated = self.resolver._consolidate_action_fields(
            structures, name_mapping, merge_groups
        )

        assert len(consolidated) == 2  # Climate merged, Mobility separate

        # Find the consolidated climate action field
        climate_af = next(
            af
            for af in consolidated
            if af["action_field"] == "Klimaschutz und Klimaanpassung"
        )

        assert len(climate_af["projects"]) == 2
        project_titles = {p["title"] for p in climate_af["projects"]}
        assert "Solar Project" in project_titles
        assert "Flood Protection" in project_titles

    def test_merge_project_details(self):
        """Test merging project details."""
        existing_projects = [
            {
                "title": "Test Project",
                "measures": ["Measure A"],
                "indicators": ["Indicator 1"],
            }
        ]

        new_project = {
            "title": "Test Project",
            "measures": ["Measure B", "Measure A"],  # A is duplicate
            "indicators": ["Indicator 2"],
            "sources": [{"page_number": 1, "quote": "Test quote"}],
        }

        self.resolver._merge_project_details(existing_projects, new_project)

        merged = existing_projects[0]
        assert len(merged["measures"]) == 2  # A, B
        assert "Measure A" in merged["measures"]
        assert "Measure B" in merged["measures"]

        assert len(merged["indicators"]) == 2  # 1, 2
        assert "Indicator 1" in merged["indicators"]
        assert "Indicator 2" in merged["indicators"]

        assert "sources" in merged
        assert len(merged["sources"]) == 1

    @patch("src.processing.entity_resolver.SentenceTransformer")
    def test_similarity_groups_with_embeddings(self, mock_transformer):
        """Test similarity-based grouping with mocked embeddings."""
        # Mock the embedding model
        mock_model = Mock()
        mock_model.encode.return_value = [
            [0.1, 0.2, 0.3],  # "Test Project A"
            [0.11, 0.19, 0.31],  # "Test Project B" - similar to A
            [0.5, 0.6, 0.7],  # "Different Project" - different
        ]
        mock_transformer.return_value = mock_model

        # Force loading of embeddings
        self.resolver._lazy_load_embeddings()

        entities = [
            {"name": "Test Project A"},
            {"name": "Test Project B"},
            {"name": "Different Project"},
        ]

        # Mock cosine similarity to return high similarity for A-B
        with patch.object(self.resolver, "_cosine_similarity") as mock_cosine:
            mock_cosine.side_effect = lambda v1, v2: (
                0.9
                if (
                    (
                        v1 == mock_model.encode.return_value[0]
                        and v2 == mock_model.encode.return_value[1]
                    )
                    or (
                        v1 == mock_model.encode.return_value[1]
                        and v2 == mock_model.encode.return_value[0]
                    )
                )
                else 0.3
            )

            with patch.object(
                self.resolver, "_validate_merge_candidate", return_value=True
            ):
                groups = self.resolver._find_similarity_groups(entities)

                assert len(groups) == 1
                assert len(groups[0]) == 2
                names = {entity["name"] for entity in groups[0]}
                assert "Test Project A" in names
                assert "Test Project B" in names

    def test_validate_merge_candidate(self):
        """Test merge candidate validation."""
        # High similarity - should auto-merge
        assert self.resolver._validate_merge_candidate("Test A", "Test B", 0.95)

        # Substring relationship
        assert self.resolver._validate_merge_candidate(
            "Klimaschutz", "Klimaschutz Projekt", 0.8
        )

        # Word overlap
        assert self.resolver._validate_merge_candidate(
            "Solar Energy Project", "Energy Solar Initiative", 0.8
        )

        # Low similarity - should not merge
        assert not self.resolver._validate_merge_candidate(
            "Klimaschutz", "Mobilität", 0.6
        )

    def test_resolve_entities_disabled(self):
        """Test entity resolution when disabled."""
        structures = [{"action_field": "Test", "projects": []}]

        with patch("src.processing.entity_resolver.ENTITY_RESOLUTION_ENABLED", False):
            result = self.resolver.resolve_entities(structures, "action_field")
            assert result == structures

    def test_resolve_entities_empty(self):
        """Test entity resolution with empty input."""
        result = self.resolver.resolve_entities([], "action_field")
        assert result == []

    def test_resolve_entities_single_entity(self):
        """Test entity resolution with single entity."""
        structures = [{"action_field": "Test", "projects": []}]
        result = self.resolver.resolve_entities(structures, "action_field")
        assert result == structures


class TestResolveExtractionEntities:
    """Test cases for the main resolve_extraction_entities function."""

    def test_resolve_empty(self):
        """Test resolving empty structures."""
        result = resolve_extraction_entities([])
        assert result == []

    def test_resolve_action_fields_only(self):
        """Test resolving only action fields."""
        structures = [
            {"action_field": "Klimaschutz", "projects": []},
            {"action_field": "Klimaanpassung", "projects": []},
        ]

        result = resolve_extraction_entities(
            structures, resolve_action_fields=True, resolve_projects=False
        )

        # Should consolidate climate fields
        assert len(result) == 1
        assert result[0]["action_field"] == "Klimaschutz und Klimaanpassung"

    def test_resolve_projects_only(self):
        """Test resolving only projects."""
        structures = [
            {
                "action_field": "Klimaschutz",
                "projects": [
                    {"title": "Solar Initiative"},
                    {"title": "Solar Project"},  # Similar to above
                ],
            }
        ]

        # Mock similarity to trigger project merging
        with patch(
            "src.processing.entity_resolver.EntityResolver._find_similarity_groups"
        ) as mock_sim:
            mock_sim.return_value = [
                [
                    {
                        "name": "Solar Initiative",
                        "structure_index": 0,
                        "project_index": 0,
                    },
                    {"name": "Solar Project", "structure_index": 0, "project_index": 1},
                ]
            ]

            result = resolve_extraction_entities(
                structures, resolve_action_fields=False, resolve_projects=True
            )

            # Should merge similar projects (implementation depends on similarity logic)
            assert len(result) == 1
            assert result[0]["action_field"] == "Klimaschutz"

    def test_integration_with_real_patterns(self):
        """Test integration with real German patterns."""
        structures = [
            {
                "action_field": "Klimaschutz",
                "projects": [
                    {
                        "title": "Solarenergie Ausbau",
                        "measures": ["Installation von Solarpanels"],
                        "indicators": ["50% erneuerbare Energien bis 2030"],
                    }
                ],
            },
            {
                "action_field": "Klimaanpassung",
                "projects": [
                    {
                        "title": "Hochwasserschutz",
                        "measures": ["Bau von Deichen"],
                        "indicators": ["Reduktion von Hochwasserrisiko um 30%"],
                    }
                ],
            },
            {
                "action_field": "Mobilität",
                "projects": [
                    {
                        "title": "Radwegenetz",
                        "measures": ["Ausbau Fahrradwege"],
                        "indicators": ["100 km neue Radwege"],
                    }
                ],
            },
        ]

        result = resolve_extraction_entities(structures)

        # Should merge Klimaschutz and Klimaanpassung
        assert len(result) == 2  # Climate (merged) + Mobility

        # Find climate action field
        climate_af = next(
            af
            for af in result
            if "Klimaschutz" in af["action_field"]
            and "Klimaanpassung" in af["action_field"]
        )

        assert len(climate_af["projects"]) == 2
        project_titles = {p["title"] for p in climate_af["projects"]}
        assert "Solarenergie Ausbau" in project_titles
        assert "Hochwasserschutz" in project_titles


class TestGraphQualityIntegration:
    """Test integration between entity resolution and graph quality metrics."""

    def test_quality_improvement_detection(self):
        """Test that quality metrics detect improvements after resolution."""
        # Create fragmented structures
        fragmented_structures = [
            {"action_field": "Klimaschutz", "projects": [{"title": "Solar A"}]},
            {"action_field": "Klimaanpassung", "projects": [{"title": "Flood B"}]},
            {"action_field": "Mobilität", "projects": []},  # Empty project
        ]

        # Analyze before resolution
        analyzer = GraphQualityAnalyzer()
        before_metrics = analyzer.analyze_extraction_quality(
            fragmented_structures, True
        )

        # Apply resolution
        resolved_structures = resolve_extraction_entities(fragmented_structures)

        # Analyze after resolution
        after_metrics = analyzer.analyze_extraction_quality(resolved_structures, False)

        # Check improvements
        improvements = analyzer.compare_before_after(before_metrics, after_metrics)

        # Should have fewer action fields after merging
        assert (
            before_metrics["basic_stats"]["total_action_fields"]
            > after_metrics["basic_stats"]["total_action_fields"]
        )

        # Quality score should improve
        assert improvements["overall_score_improvement"] > 0

    def test_fragmentation_metrics(self):
        """Test fragmentation metrics calculation."""
        structures = [
            {"action_field": "Klimaschutz", "projects": []},
            {
                "action_field": "Klimaanpassung",
                "projects": [],
            },  # Should be detected as similar
            {"action_field": "Mobilität", "projects": []},
        ]

        analyzer = GraphQualityAnalyzer()
        metrics = analyzer.analyze_extraction_quality(structures)

        frag_metrics = metrics["fragmentation_metrics"]
        af_frag = frag_metrics["action_field_fragmentation"]

        # Should detect potential duplicates in climate fields
        assert af_frag["potential_duplicates"] > 0
        assert len(af_frag["duplicate_groups"]) > 0


if __name__ == "__main__":
    pytest.main([__file__])
