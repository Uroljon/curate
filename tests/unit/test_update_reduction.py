"""
Unit tests for UPDATE operation reduction improvements.

Tests that:
1. Entity registry includes descriptions when needed
2. UPDATE operations are more conservative
3. LLM sees existing content to avoid redundant updates
"""

import pytest
from src.api.extraction_helpers import (
    format_entity_registry,
    format_context_json,
    build_operations_prompt,
)
from src.core.schemas import (
    EnrichedReviewJSON,
    EnhancedActionField,
    EnhancedProject,
    EnhancedMeasure,
    EnhancedIndicator,
)


def create_test_action_field(id: str, title: str, description: str = None, confidence: float = 0.8):
    """Helper to create test action field."""
    content = {"title": title}
    if description:
        content["description"] = description
    
    return EnhancedActionField(
        id=id,
        content=content,
        confidence=confidence,
        project_ids=[],
        measure_ids=[],
        indicator_ids=[]
    )


def create_test_project(id: str, title: str, description: str = None, confidence: float = 0.75):
    """Helper to create test project."""
    content = {"title": title}
    if description:
        content["description"] = description
    
    return EnhancedProject(
        id=id,
        content=content,
        confidence=confidence,
        action_field_ids=[],
        measure_ids=[],
        indicator_ids=[]
    )


class TestEntityRegistryWithDescriptions:
    """Test that entity registry can show descriptions to prevent redundant UPDATEs."""
    
    def test_registry_without_descriptions(self):
        """Test original behavior - registry shows only titles."""
        # Create test state with entities
        state = EnrichedReviewJSON(
            action_fields=[
                create_test_action_field(
                    "af_1", 
                    "Mobilität und Verkehr",
                    "Nachhaltige Verkehrslösungen für die Stadt"
                )
            ],
            projects=[
                create_test_project(
                    "proj_1",
                    "Stadtbahn Regensburg",
                    "Planung und Bau einer Stadtbahn"
                )
            ],
            measures=[],
            indicators=[]
        )
        
        # Get registry without descriptions (default old behavior)
        registry = format_entity_registry(state, include_descriptions=False)
        
        # Check that only titles are shown
        assert "Mobilität und Verkehr" in registry
        assert "Stadtbahn Regensburg" in registry
        
        # Descriptions should NOT be shown
        assert "Nachhaltige Verkehrslösungen" not in registry
        assert "Planung und Bau" not in registry
        
        # Should show entity counts
        assert "ACTION FIELDS (1 total)" in registry
        assert "PROJECTS (1 total)" in registry
    
    def test_registry_with_descriptions(self):
        """Test enhanced behavior - registry shows descriptions to inform UPDATE decisions."""
        # Create test state with entities
        state = EnrichedReviewJSON(
            action_fields=[
                create_test_action_field(
                    "af_1",
                    "Mobilität und Verkehr",
                    "Nachhaltige Verkehrslösungen für die Stadt"
                )
            ],
            projects=[
                create_test_project(
                    "proj_1",
                    "Stadtbahn Regensburg",
                    "Planung und Bau einer Stadtbahn"
                ),
                create_test_project(
                    "proj_2",
                    "Radwegeausbau"
                    # No description
                )
            ],
            measures=[],
            indicators=[]
        )
        
        # Get registry WITH descriptions
        registry = format_entity_registry(state, include_descriptions=True)
        
        # Check that titles AND descriptions are shown
        assert "af_1: Mobilität und Verkehr" in registry
        assert "Nachhaltige Verkehrslösungen für die Stadt" in registry
        
        assert "proj_1: Stadtbahn Regensburg" in registry
        assert "Planung und Bau einer Stadtbahn" in registry
        
        # Entity without description should show placeholder
        assert "proj_2: Radwegeausbau" in registry
        assert "[no description yet]" in registry
        
        # Should have the warning about checking before UPDATE
        assert "CHECK BEFORE CREATE/UPDATE" in registry
        assert "Only UPDATE if you have NEW information" in registry
    
    def test_registry_truncates_long_descriptions(self):
        """Test that very long descriptions are truncated for readability."""
        long_description = "A" * 200  # 200 character description
        
        state = EnrichedReviewJSON(
            action_fields=[
                create_test_action_field(
                    "af_1",
                    "Test Field",
                    long_description
                )
            ],
            projects=[], 
            measures=[], 
            indicators=[]
        )
        
        registry = format_entity_registry(state, include_descriptions=True)
        
        # Should truncate to ~150 chars with ellipsis
        assert "A" * 147 in registry
        assert "..." in registry
        # Full 200 char string should NOT be there
        assert long_description not in registry


class TestContextFormatting:
    """Test that context formatting provides descriptions when needed."""
    
    def test_context_for_nodes_mode(self):
        """Test that nodes mode includes descriptions to prevent redundant UPDATEs."""
        state = EnrichedReviewJSON(
            action_fields=[
                create_test_action_field(
                    "af_1",
                    "Energie und Klimaschutz",
                    "Maßnahmen zur CO2-Reduktion"
                )
            ],
            projects=[], 
            measures=[], 
            indicators=[]
        )
        
        # Build prompt for nodes mode
        prompt = build_operations_prompt(
            mode="nodes",
            chunk_text="Test chunk text",
            state=state,
            page_numbers=[1, 2, 3]
        )
        
        # Should include descriptions in nodes mode
        assert "Energie und Klimaschutz" in prompt
        assert "Maßnahmen zur CO2-Reduktion" in prompt
        assert "Only UPDATE if you have NEW information" in prompt
    
    def test_context_for_connections_mode(self):
        """Test that connections mode doesn't need descriptions."""
        state = EnrichedReviewJSON(
            action_fields=[
                create_test_action_field(
                    "af_1",
                    "Energie und Klimaschutz",
                    "Maßnahmen zur CO2-Reduktion"
                )
            ],
            projects=[], 
            measures=[], 
            indicators=[]
        )
        
        # Build prompt for connections mode
        prompt = build_operations_prompt(
            mode="connections",
            chunk_text="Test chunk text",
            state=state,
            page_numbers=[1, 2, 3]
        )
        
        # Should show title and ID mapping
        assert "af_1" in prompt
        assert "Energie und Klimaschutz" in prompt
        
        # Connections mode focuses on IDs, not descriptions
        assert "CONNECT" in prompt
        assert "from_id" in prompt
    
    def test_empty_state_prompt(self):
        """Test that empty state shows appropriate message."""
        state = EnrichedReviewJSON(
            action_fields=[], 
            projects=[], 
            measures=[], 
            indicators=[]
        )
        
        prompt = build_operations_prompt(
            mode="nodes",
            chunk_text="Test chunk text",
            state=state,
            page_numbers=[1]
        )
        
        # Should have the "first chunk" message
        assert "ERSTER CHUNK" in prompt
        assert "CREATE-Operationen" in prompt


class TestUpdateRulesInPrompts:
    """Test that prompts have conservative UPDATE rules."""
    
    def test_update_requires_new_information(self):
        """Test that UPDATE rules require genuinely new information."""
        from src.prompts.loader import get_prompt
        
        # Get the context rules that guide UPDATE decisions
        context_rules = get_prompt("operations.fragments.context_rules")
        
        # Should emphasize checking existing descriptions
        assert "schauen Sie auf ID und Beschreibung" in context_rules
        assert "NUR UPDATE wenn Sie Information haben die NICHT in der vorhandenen Beschreibung steht" in context_rules
        
        # Should distinguish between true duplicates and similar but different entities
        assert "VERSCHIEDENE Entities (CREATE neuer Entity)" in context_rules
        assert "Im Zweifel → CREATE statt UPDATE" in context_rules
    
    def test_update_merge_rules(self):
        """Test that UPDATE merge rules are additive and non-destructive."""
        from src.prompts.loader import get_prompt
        
        # Get the UPDATE merge rules
        merge_rules = get_prompt("operations.fragments.update_merge_rules")
        
        # Should be additive, not replacement
        assert "ADDITIV" in merge_rules
        assert "Keine Felder entfernen oder leeren" in merge_rules
        assert "neuen Text nur anfügen, wenn er nicht redundant ist" in merge_rules
        
        # Should preserve titles
        assert "TITEL-STABILITÄT" in merge_rules
        assert "Titel/Name in UPDATE NIEMALS ändern" in merge_rules
    
    def test_higher_confidence_for_updates(self):
        """Test that UPDATEs require higher confidence than CREATEs."""
        from src.prompts.loader import get_prompt
        
        # Get the nodes template
        nodes_template = get_prompt("operations.templates.operations_nodes_chunk")
        
        # Should have different confidence thresholds
        assert "CREATE ≥ 0.7" in nodes_template
        assert "UPDATE ≥ 0.8" in nodes_template  # Higher threshold for UPDATE


class TestOperationExamples:
    """Test that operation examples show proper UPDATE usage."""
    
    def test_update_example_shows_new_fields(self):
        """Test that UPDATE examples add new fields, not redundant descriptions."""
        from src.prompts.loader import get_prompt
        
        # Get the system message with examples
        system_msg = get_prompt("operations.system_messages.operations_extraction")
        
        # UPDATE example should show adding NEW fields like budget, timeline
        assert '"budget":' in system_msg or '"timeline":' in system_msg
        
        # Should NOT show updating description with paraphrases
        assert "NUR wenn neue Information" in system_msg


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])