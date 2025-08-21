#!/usr/bin/env python3
"""
Integration tests for YAML prompt system with existing API functions.
"""

import pytest
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.prompts import get_prompt, clear_cache
from src.api.extraction_helpers import create_extraction_prompt, format_entity_registry, format_entity_id_mapping
from src.core.schemas import EnrichedReviewJSON, EnhancedActionField, EnhancedProject, EnhancedMeasure, EnhancedIndicator


class TestPromptAPIIntegration:
    """Integration tests between YAML prompts and existing API functions."""
    
    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()
    
    def test_create_extraction_prompt_simplified(self):
        """Test that create_extraction_prompt works with YAML templates for simplified type."""
        # Create some test context data
        test_context = EnrichedReviewJSON(
            action_fields=[
                EnhancedActionField(id="af_1", content={"title": "Test Action Field"}, connections=[])
            ],
            projects=[
                EnhancedProject(id="proj_1", content={"title": "Test Project"}, connections=[])
            ],
            measures=[
                EnhancedMeasure(id="msr_1", content={"title": "Test Measure"}, connections=[])
            ],
            indicators=[
                EnhancedIndicator(id="ind_1", content={"title": "Test Indicator"}, connections=[])
            ]
        )
        
        # Test simplified template
        prompt = create_extraction_prompt("simplified", "Test chunk content", test_context)
        
        # Verify it contains expected content
        assert isinstance(prompt, str)
        assert len(prompt) > 200
        assert "Test chunk content" in prompt
        assert "Test Action Field" in prompt  # Should include context
        assert "HIERARCHISCH KORREKTEN" in prompt  # German content
    
    def test_create_extraction_prompt_operations(self):
        """Test that create_extraction_prompt works with YAML templates for operations type."""
        # Create some test context data
        test_context = EnrichedReviewJSON(
            action_fields=[
                EnhancedActionField(id="af_1", content={"title": "Mobilität"}, connections=[])
            ],
            projects=[],
            measures=[],
            indicators=[]
        )
        
        # Test operations template
        prompt = create_extraction_prompt("operations", "Sample German text", test_context, [1, 2, 3])
        
        # Verify it contains expected content
        assert isinstance(prompt, str)
        assert len(prompt) > 500
        assert "Sample German text" in prompt
        assert "1, 2, 3" in prompt  # Page numbers
        assert "Mobilität" in prompt  # Context entity
        assert "OPERATIONEN zur Strukturerweiterung" in prompt  # German instructions
        assert "CREATE-Operation" in prompt  # Instructions
        assert "ENTITY REGISTRY" in prompt  # Registry reference
    
    def test_create_extraction_prompt_operations_empty_context(self):
        """Test operations template with empty context."""
        prompt = create_extraction_prompt("operations", "Test text", None, [5])
        
        assert isinstance(prompt, str)
        assert "Test text" in prompt
        assert "ERSTER CHUNK: Noch keine Entities extrahiert" in prompt
        assert "Seiten 5" in prompt
    
    def test_create_extraction_prompt_invalid_type(self):
        """Test error handling for invalid template types."""
        with pytest.raises(ValueError, match="Unknown template type"):
            create_extraction_prompt("invalid_type", "test", None)
    
    def test_format_entity_registry_integration(self):
        """Test that entity registry formatting works with YAML prompt references."""
        # Create test data
        test_context = EnrichedReviewJSON(
            action_fields=[
                EnhancedActionField(id="af_1", content={"title": "Klimaschutz"}, connections=[]),
                EnhancedActionField(id="af_2", content={"title": "Mobilität"}, connections=[])
            ],
            projects=[
                EnhancedProject(id="proj_1", content={"title": "Solarinitiative"}, connections=[])
            ],
            measures=[
                EnhancedMeasure(id="msr_1", content={"title": "Photovoltaik Ausbau"}, connections=[])
            ],
            indicators=[
                EnhancedIndicator(id="ind_1", content={"title": "CO2 Reduktion"}, connections=[])
            ]
        )
        
        # Test entity registry formatting
        registry = format_entity_registry(test_context)
        
        assert isinstance(registry, str)
        assert "Klimaschutz" in registry
        assert "Mobilität" in registry
        assert "Solarinitiative" in registry
        assert "Photovoltaik Ausbau" in registry
        assert "CO2 Reduktion" in registry
        assert "ENTITY REGISTRY" in registry
    
    def test_format_entity_id_mapping_integration(self):
        """Test that entity ID mapping works correctly."""
        test_context = EnrichedReviewJSON(
            action_fields=[
                EnhancedActionField(id="af_1", content={"title": "Test Field"}, connections=[])
            ],
            projects=[
                EnhancedProject(id="proj_1", content={"title": "Test Project"}, connections=[])
            ],
            measures=[],
            indicators=[]
        )
        
        # Test ID mapping
        mapping = format_entity_id_mapping(test_context)
        
        assert isinstance(mapping, str)
        assert "af_1 → Test Field" in mapping
        assert "proj_1 → Test Project" in mapping
        assert "ACTION FIELD IDs:" in mapping
        assert "PROJECT IDs:" in mapping
    
    def test_all_legacy_prompts_accessible(self):
        """Test that all legacy prompts are accessible for backward compatibility."""
        legacy_prompts = [
            "legacy.system_messages.stage1_action_fields",
            "legacy.system_messages.stage2_projects", 
            "legacy.system_messages.stage3_measures_indicators",
            "legacy.templates.stage1_chunk",
            "legacy.templates.stage2_chunk",
            "legacy.templates.stage3_chunk"
        ]
        
        for prompt_path in legacy_prompts:
            prompt = get_prompt(prompt_path)
            assert isinstance(prompt, str)
            assert len(prompt) > 50
            # Should contain German content
            assert any(german in prompt for german in ["Sie sind", "Extrahieren", "KRITISCH"])
    
    def test_legacy_template_variable_substitution(self):
        """Test that legacy templates work with variable substitution."""
        # Test stage 1 template
        stage1_prompt = get_prompt("legacy.templates.stage1_chunk", chunk="Test German text")
        assert "Test German text" in stage1_prompt
        assert "QUELLDOKUMENT:" in stage1_prompt
        
        # Test stage 2 template  
        stage2_prompt = get_prompt("legacy.templates.stage2_chunk", 
                                  chunk="More test text", 
                                  action_field="Mobilität")
        assert "More test text" in stage2_prompt
        assert "Mobilität" in stage2_prompt
        
        # Test stage 3 template
        stage3_prompt = get_prompt("legacy.templates.stage3_chunk",
                                  chunk="Stage 3 text",
                                  project_title="Solar Project")
        assert "Stage 3 text" in stage3_prompt
        assert "Solar Project" in stage3_prompt
    
    def test_utils_prompts_integration(self):
        """Test that utils prompts integrate properly."""
        # Test deduplication system message
        dedup_system = get_prompt("utils.system_messages.deduplication")
        assert "KONSERVATIVE Deduplizierung" in dedup_system
        assert "Handlungsfeldern" in dedup_system
        
        # Test deduplication template with variables
        dedup_template = get_prompt("utils.templates.deduplication_chunk",
                                   action_field_count=25,
                                   min_target=18,
                                   max_target=20,
                                   not_consolidate_examples="Test not consolidate",
                                   consolidate_examples="Test consolidate",
                                   chunk_data='{"test": "data"}')
        
        assert "25 Handlungsfelder" in dedup_template
        assert "18 bis 20 Handlungsfelder" in dedup_template
        assert "Test not consolidate" in dedup_template
        assert "Test consolidate" in dedup_template
        assert '{"test": "data"}' in dedup_template
    
    def test_operations_prompt_content_quality(self):
        """Test that operations prompts contain all expected quality elements."""
        system_prompt = get_prompt("operations.system_messages.operations_extraction")
        
        # Should contain key operational concepts
        expected_elements = [
            "CREATE",
            "UPDATE", 
            "CONNECT",
            "DUPLIKAT-VERMEIDUNG",
            "ENTITY REGISTRY",
            "HIERARCHISCHE STRUKTUR",
            "QUALITÄTSPRINZIPIEN"
        ]
        
        for element in expected_elements:
            assert element in system_prompt, f"Missing element: {element}"
    
    def test_extraction_prompt_content_quality(self):
        """Test that extraction prompts contain expected content."""
        system_prompt = get_prompt("extraction.system_messages.enhanced_extraction")
        
        expected_elements = [
            "HIERARCHISCHE STRUKTUR",
            "KONSISTENZ-REGEL",
            "Handlungsfelder",
            "Projekte",
            "Maßnahmen",
            "Indikatoren"
        ]
        
        for element in expected_elements:
            assert element in system_prompt, f"Missing element: {element}"
    
    def test_prompt_loading_performance(self):
        """Test that prompt loading is reasonably fast."""
        import time
        
        # Test that loading multiple prompts is fast (should use cache)
        prompts_to_load = [
            "operations.system_messages.operations_extraction",
            "extraction.system_messages.enhanced_extraction", 
            "utils.system_messages.deduplication",
            "legacy.system_messages.stage1_action_fields"
        ]
        
        # First load (populate cache)
        start_time = time.time()
        for prompt_path in prompts_to_load:
            get_prompt(prompt_path)
        first_load_time = time.time() - start_time
        
        # Second load (should use cache)
        start_time = time.time()
        for prompt_path in prompts_to_load:
            get_prompt(prompt_path)
        cached_load_time = time.time() - start_time
        
        # Cached loading should be significantly faster
        assert first_load_time > 0  # Should take some time
        assert cached_load_time < first_load_time  # Cache should be faster
        assert cached_load_time < 0.1  # Should be very fast (< 100ms)


class TestPromptSystemRobustness:
    """Test system robustness and edge cases."""
    
    def test_concurrent_prompt_loading(self):
        """Test that concurrent prompt loading works correctly."""
        import threading
        import time
        
        results = []
        errors = []
        
        def load_prompt(prompt_path):
            try:
                prompt = get_prompt(prompt_path)
                results.append((prompt_path, len(prompt)))
            except Exception as e:
                errors.append((prompt_path, str(e)))
        
        # Create multiple threads loading different prompts
        threads = []
        prompt_paths = [
            "operations.system_messages.operations_extraction",
            "extraction.system_messages.enhanced_extraction",
            "utils.system_messages.deduplication",
            "legacy.system_messages.stage1_action_fields"
        ]
        
        for prompt_path in prompt_paths:
            thread = threading.Thread(target=load_prompt, args=(prompt_path,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=5)  # 5 second timeout
        
        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == len(prompt_paths)
        
        # All prompts should have loaded successfully
        for prompt_path, prompt_length in results:
            assert prompt_length > 50, f"Prompt {prompt_path} too short: {prompt_length}"
    
    def test_memory_usage_with_large_prompts(self):
        """Test memory usage doesn't grow excessively with repeated loads."""
        import gc
        import sys
        
        # Get initial memory usage
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Load prompts many times
        for _ in range(100):
            get_prompt("operations.system_messages.operations_extraction")
            get_prompt("extraction.system_messages.enhanced_extraction")
        
        # Check memory usage after
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Memory should not have grown significantly (cache should prevent this)
        object_growth = final_objects - initial_objects
        assert object_growth < 1000, f"Too many new objects created: {object_growth}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])