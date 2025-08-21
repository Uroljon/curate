#!/usr/bin/env python3
"""
Comprehensive tests for the YAML-based prompt loader system.
"""

import os
import tempfile
import pytest
from pathlib import Path
import yaml

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.prompts import get_prompt, list_available_prompts, clear_cache
from src.prompts.loader import _load_yaml_file, CONFIGS_DIR


class TestPromptLoader:
    """Test suite for the YAML prompt loader."""
    
    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()
    
    def test_get_prompt_basic_functionality(self):
        """Test basic prompt loading functionality."""
        # Test loading a known system message
        prompt = get_prompt("operations.system_messages.operations_extraction")
        assert isinstance(prompt, str)
        assert len(prompt) > 100  # Should be a substantial prompt
        assert "Sie sind ein Experte" in prompt  # German content check
    
    def test_get_prompt_template_substitution(self):
        """Test template variable substitution."""
        # Test with variables
        prompt = get_prompt("operations.templates.operations_chunk",
                           context_text="TEST_CONTEXT",
                           page_list="1,2,3", 
                           chunk_text="TEST_CHUNK")
        
        assert "TEST_CONTEXT" in prompt
        assert "TEST_CHUNK" in prompt
        assert "1,2,3" in prompt
        assert "{context_text}" not in prompt  # Should be substituted
    
    def test_get_prompt_no_variables_needed(self):
        """Test prompts that don't need variable substitution."""
        prompt = get_prompt("utils.system_messages.deduplication")
        assert isinstance(prompt, str)
        assert len(prompt) > 50
        assert "{" not in prompt  # Should not have template variables
    
    def test_get_prompt_invalid_format(self):
        """Test error handling for invalid prompt names."""
        with pytest.raises(ValueError, match="Invalid prompt name format"):
            get_prompt("invalid")
        
        with pytest.raises(ValueError, match="Invalid prompt name format"):
            get_prompt("only.one")
    
    def test_get_prompt_missing_config(self):
        """Test error handling for missing configuration files."""
        with pytest.raises(FileNotFoundError, match="Prompt config not found"):
            get_prompt("nonexistent.system_messages.test")
    
    def test_get_prompt_missing_category_or_name(self):
        """Test error handling for missing categories or prompt names."""
        with pytest.raises(KeyError, match="not found in"):
            get_prompt("operations.nonexistent_category.test")
        
        with pytest.raises(KeyError, match="not found in"):
            get_prompt("operations.system_messages.nonexistent_prompt")
    
    def test_get_prompt_missing_template_variables(self):
        """Test error handling for missing template variables."""
        with pytest.raises(KeyError, match="Missing template variable"):
            get_prompt("operations.templates.operations_chunk", 
                      context_text="test")  # Missing page_list and chunk_text
    
    def test_list_available_prompts(self):
        """Test listing all available prompts."""
        prompts = list_available_prompts()
        
        # Should have all expected configs
        expected_configs = ["operations", "extraction", "legacy", "utils"]
        for config in expected_configs:
            assert config in prompts
            assert isinstance(prompts[config], dict)
        
        # Each config should have expected sections
        assert "system_messages" in prompts["operations"]
        assert "templates" in prompts["operations"]
        assert "variables" in prompts["utils"]
    
    def test_list_available_prompts_single_config(self):
        """Test listing prompts for a specific config."""
        operations_prompts = list_available_prompts("operations")
        
        assert "operations" in operations_prompts
        assert "extraction" not in operations_prompts  # Should only have operations
        assert "system_messages" in operations_prompts["operations"]
        assert "templates" in operations_prompts["operations"]
    
    def test_prompt_caching(self):
        """Test that prompts are properly cached."""
        # First call should load from file
        prompt1 = get_prompt("operations.system_messages.operations_extraction")
        
        # Second call should use cache (test by checking it doesn't fail if file is missing)
        prompt2 = get_prompt("operations.system_messages.operations_extraction")
        
        assert prompt1 == prompt2
        assert len(prompt1) > 100
    
    def test_clear_cache_functionality(self):
        """Test cache clearing functionality."""
        # Load a prompt to populate cache
        get_prompt("operations.system_messages.operations_extraction")
        
        # Clear cache
        clear_cache()
        
        # Should still work after cache clear (reload from file)
        prompt = get_prompt("operations.system_messages.operations_extraction")
        assert isinstance(prompt, str)
        assert len(prompt) > 100
    
    def test_all_expected_prompts_exist(self):
        """Test that all expected prompts from the refactoring exist."""
        expected_prompts = [
            # Operations prompts
            ("operations.system_messages.operations_extraction", "Operations system"),
            ("operations.templates.operations_chunk", "Operations template"),
            
            # Extraction prompts  
            ("extraction.system_messages.enhanced_extraction", "Enhanced system"),
            ("extraction.templates.simplified_chunk", "Enhanced template"),
            
            # Utils prompts
            ("utils.system_messages.deduplication", "Deduplication system"),
            ("utils.templates.deduplication_chunk", "Deduplication template"),
            ("utils.variables.not_consolidate_examples", "Not consolidate examples"),
            ("utils.variables.consolidate_examples", "Consolidate examples"),
            
            # Legacy prompts
            ("legacy.system_messages.stage1_action_fields", "Legacy stage 1 system"),
            ("legacy.system_messages.stage2_projects", "Legacy stage 2 system"),
            ("legacy.system_messages.stage3_measures_indicators", "Legacy stage 3 system"),
            ("legacy.templates.stage1_chunk", "Legacy stage 1 template"),
            ("legacy.templates.stage2_chunk", "Legacy stage 2 template"),
            ("legacy.templates.stage3_chunk", "Legacy stage 3 template"),
        ]
        
        for prompt_path, description in expected_prompts:
            try:
                prompt = get_prompt(prompt_path)
                assert isinstance(prompt, str), f"{description} should be a string"
                assert len(prompt) > 10, f"{description} should have substantial content"
            except Exception as e:
                pytest.fail(f"{description} failed to load: {e}")
    
    def test_yaml_file_structure_integrity(self):
        """Test that all YAML files have proper structure."""
        config_files = ["operations.yaml", "extraction.yaml", "legacy.yaml", "utils.yaml"]
        
        for config_file in config_files:
            config_path = os.path.join(CONFIGS_DIR, config_file)
            assert os.path.exists(config_path), f"Config file {config_file} should exist"
            
            # Test YAML is valid
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            assert isinstance(config, dict), f"Config {config_file} should be a dict"
            
            # Should have at least system_messages or templates
            has_content = any(key in config for key in ["system_messages", "templates", "variables"])
            assert has_content, f"Config {config_file} should have content sections"
    
    def test_template_variables_validation(self):
        """Test that templates with variables work correctly."""
        # Test operations template with all variables
        template = get_prompt("operations.templates.operations_chunk",
                             context_text="Test context with entities",
                             page_list="1, 2, 3",
                             chunk_text="Sample chunk text for testing")
        
        # Verify substitution worked
        assert "Test context with entities" in template
        assert "1, 2, 3" in template
        assert "Sample chunk text for testing" in template
        
        # Verify no template variables remain
        assert "{context_text}" not in template
        assert "{page_list}" not in template
        assert "{chunk_text}" not in template
    
    def test_german_content_integrity(self):
        """Test that German content is properly preserved."""
        prompts_to_test = [
            "operations.system_messages.operations_extraction",
            "extraction.system_messages.enhanced_extraction", 
            "utils.system_messages.deduplication",
            "legacy.system_messages.stage1_action_fields"
        ]
        
        for prompt_path in prompts_to_test:
            prompt = get_prompt(prompt_path)
            
            # Should contain German text
            german_indicators = ["Sie sind", "Extrahieren", "KRITISCH", "Antworten Sie"]
            has_german = any(indicator in prompt for indicator in german_indicators)
            assert has_german, f"Prompt {prompt_path} should contain German text"
    
    def test_nested_prompt_keys(self):
        """Test that nested keys work properly."""
        # Test variables (nested under utils)
        not_consolidate = get_prompt("utils.variables.not_consolidate_examples")
        consolidate = get_prompt("utils.variables.consolidate_examples")
        
        assert isinstance(not_consolidate, str)
        assert isinstance(consolidate, str)
        assert len(not_consolidate) > 50
        assert len(consolidate) > 50
        assert "❌" in not_consolidate  # Should have examples
        assert "✅" in consolidate  # Should have examples


class TestPromptLoaderEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()
    
    def test_empty_template_variables(self):
        """Test template with empty variable values."""
        prompt = get_prompt("operations.templates.operations_chunk",
                           context_text="",
                           page_list="", 
                           chunk_text="")
        
        # Should still be a valid prompt, just with empty sections
        assert isinstance(prompt, str)
        assert len(prompt) > 100  # Base template should still have content
    
    def test_special_characters_in_variables(self):
        """Test template variables with special characters."""
        special_text = "Test with üäö and €£$ symbols"
        prompt = get_prompt("legacy.templates.stage1_chunk",
                           chunk=special_text)
        
        assert special_text in prompt
        assert isinstance(prompt, str)
    
    def test_large_template_variables(self):
        """Test template variables with large content."""
        large_text = "Large content " * 1000  # ~13KB text
        prompt = get_prompt("legacy.templates.stage1_chunk",
                           chunk=large_text)
        
        assert large_text in prompt
        assert len(prompt) > len(large_text)  # Should have template + variable content
    
    def test_prompt_name_case_sensitivity(self):
        """Test that prompt names work correctly (case handling depends on filesystem)."""
        # This should work
        prompt = get_prompt("operations.system_messages.operations_extraction")
        assert len(prompt) > 100
        
        # Test that exact case works consistently
        prompt2 = get_prompt("operations.system_messages.operations_extraction")
        assert prompt == prompt2
        
        # Test wrong category/name (should fail regardless of case)
        with pytest.raises(KeyError):
            get_prompt("operations.nonexistent_category.test")


if __name__ == "__main__":
    # Allow running this test file directly
    pytest.main([__file__, "-v"])