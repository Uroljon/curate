#!/usr/bin/env python3
"""
Error handling and edge case tests for the YAML prompt loader.
"""

import os
import tempfile
import pytest
from pathlib import Path
import yaml
from unittest.mock import patch, mock_open

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.prompts import get_prompt, clear_cache
from src.prompts.loader import _load_yaml_file, CONFIGS_DIR


class TestPromptLoaderErrorHandling:
    """Test error handling and recovery scenarios."""
    
    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()
    
    def test_malformed_yaml_file(self):
        """Test handling of malformed YAML files."""
        # Create a temporary malformed YAML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
invalid_yaml: [
    - missing_closing_bracket
    - "unclosed_string
system_messages:
    test: "This should not be reachable"
""")
            temp_file = f.name
        
        try:
            # Patch the config path to point to our malformed file
            with patch('src.prompts.loader.CONFIGS_DIR', os.path.dirname(temp_file)):
                config_name = os.path.basename(temp_file)[:-5]  # Remove .yaml
                
                with pytest.raises(RuntimeError, match="Failed to load prompt config"):
                    _load_yaml_file(config_name)
        finally:
            os.unlink(temp_file)
    
    def test_empty_yaml_file(self):
        """Test handling of empty YAML files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")  # Empty file
            temp_file = f.name
        
        try:
            with patch('src.prompts.loader.CONFIGS_DIR', os.path.dirname(temp_file)):
                config_name = os.path.basename(temp_file)[:-5]
                config = _load_yaml_file(config_name)
                assert config is None  # yaml.safe_load returns None for empty files
        finally:
            os.unlink(temp_file)
    
    def test_yaml_file_permission_error(self):
        """Test handling of permission errors when reading YAML files."""
        # Mock file permission error
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            with pytest.raises(RuntimeError, match="Failed to load prompt config"):
                _load_yaml_file("operations")
    
    def test_yaml_file_io_error(self):
        """Test handling of IO errors when reading YAML files."""
        # Mock IO error
        with patch('builtins.open', side_effect=IOError("Disk full")):
            with pytest.raises(RuntimeError, match="Failed to load prompt config"):
                _load_yaml_file("operations")
    
    def test_non_string_prompt_content(self):
        """Test handling of non-string prompt content in YAML."""
        yaml_content = {
            'system_messages': {
                'numeric_prompt': 12345,  # Number instead of string
                'list_prompt': ['item1', 'item2'],  # List instead of string
                'dict_prompt': {'key': 'value'}  # Dict instead of string
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(yaml_content, f)
            temp_file = f.name
        
        try:
            with patch('src.prompts.loader.CONFIGS_DIR', os.path.dirname(temp_file)):
                config_name = os.path.basename(temp_file)[:-5]
                
                # These should fail with ValueError
                with pytest.raises(ValueError, match="Prompt must be a string"):
                    get_prompt(f"{config_name}.system_messages.numeric_prompt")
                
                with pytest.raises(ValueError, match="Prompt must be a string"):
                    get_prompt(f"{config_name}.system_messages.list_prompt")
                
                with pytest.raises(ValueError, match="Prompt must be a string"):
                    get_prompt(f"{config_name}.system_messages.dict_prompt")
        finally:
            os.unlink(temp_file)
    
    def test_deeply_nested_prompt_keys(self):
        """Test deeply nested prompt keys work correctly."""
        yaml_content = {
            'level1': {
                'level2': {
                    'level3': {
                        'deep_prompt': 'This is a deeply nested prompt'
                    }
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(yaml_content, f)
            temp_file = f.name
        
        try:
            with patch('src.prompts.loader.CONFIGS_DIR', os.path.dirname(temp_file)):
                config_name = os.path.basename(temp_file)[:-5]
                prompt = get_prompt(f"{config_name}.level1.level2.level3.deep_prompt")
                assert prompt == 'This is a deeply nested prompt'
        finally:
            os.unlink(temp_file)
    
    def test_unicode_and_special_characters(self):
        """Test handling of Unicode and special characters in prompts."""
        yaml_content = {
            'unicode_test': {
                'emoji_prompt': '🚀 Test with emojis 🎉 and üäöß German chars',
                'special_chars': 'Test with "quotes", \\backslashes\\, and $pecial ch@rs!',
                'multiline': '''This is a
                multiline prompt
                with special formatting''',
                'yaml_special': 'Test with YAML special chars: []{}"|\\'
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
            yaml.dump(yaml_content, f, allow_unicode=True)
            temp_file = f.name
        
        try:
            with patch('src.prompts.loader.CONFIGS_DIR', os.path.dirname(temp_file)):
                config_name = os.path.basename(temp_file)[:-5]
                
                # Test emoji and German chars
                prompt1 = get_prompt(f"{config_name}.unicode_test.emoji_prompt")
                assert '🚀' in prompt1
                assert 'üäöß' in prompt1
                
                # Test special characters
                prompt2 = get_prompt(f"{config_name}.unicode_test.special_chars")
                assert '"quotes"' in prompt2
                assert '\\backslashes\\' in prompt2
                
                # Test multiline
                prompt3 = get_prompt(f"{config_name}.unicode_test.multiline")
                assert 'multiline prompt' in prompt3
                
                # Test YAML special chars
                prompt4 = get_prompt(f"{config_name}.unicode_test.yaml_special")
                assert '[]{}' in prompt4
        finally:
            os.unlink(temp_file)
    
    def test_template_variable_edge_cases(self):
        """Test edge cases in template variable substitution."""
        yaml_content = {
            'templates': {
                'edge_case_template': '''
                Context: {context}
                Nested braces: {{not_a_variable}}
                Double braces: {{{context}}}
                Mixed: {context} and {{literal}}
                '''
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(yaml_content, f)
            temp_file = f.name
        
        try:
            with patch('src.prompts.loader.CONFIGS_DIR', os.path.dirname(temp_file)):
                config_name = os.path.basename(temp_file)[:-5]
                
                prompt = get_prompt(f"{config_name}.templates.edge_case_template", 
                                   context="TEST_VALUE")
                
                # Should substitute single braces
                assert "Context: TEST_VALUE" in prompt
                
                # Should preserve double braces as single
                assert "{not_a_variable}" in prompt
                
                # Should handle triple braces correctly
                assert "{TEST_VALUE}" in prompt
                
                # Should preserve literal double braces
                assert "{literal}" in prompt
        finally:
            os.unlink(temp_file)
    
    def test_template_with_missing_optional_variables(self):
        """Test templates that reference variables not provided."""
        yaml_content = {
            'templates': {
                'partial_template': 'Required: {required}, Optional: {optional}'
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(yaml_content, f)
            temp_file = f.name
        
        try:
            with patch('src.prompts.loader.CONFIGS_DIR', os.path.dirname(temp_file)):
                config_name = os.path.basename(temp_file)[:-5]
                
                # Should fail when missing required variable
                with pytest.raises(KeyError, match="Missing template variable"):
                    get_prompt(f"{config_name}.templates.partial_template", 
                              required="present")  # optional missing
        finally:
            os.unlink(temp_file)
    
    def test_very_large_prompt_content(self):
        """Test handling of very large prompt content."""
        large_content = "Large prompt content " * 10000  # ~200KB content
        
        yaml_content = {
            'large_prompts': {
                'huge_prompt': large_content
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(yaml_content, f)
            temp_file = f.name
        
        try:
            with patch('src.prompts.loader.CONFIGS_DIR', os.path.dirname(temp_file)):
                config_name = os.path.basename(temp_file)[:-5]
                
                prompt = get_prompt(f"{config_name}.large_prompts.huge_prompt")
                assert len(prompt) > 100000  # Should be very large
                assert "Large prompt content" in prompt
        finally:
            os.unlink(temp_file)
    
    def test_cache_isolation_between_configs(self):
        """Test that cache properly isolates different configs."""
        # Create two temporary config files
        config1_content = {'test': {'prompt1': 'Content from config 1'}}
        config2_content = {'test': {'prompt1': 'Content from config 2'}}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='1.yaml', delete=False) as f1:
            yaml.dump(config1_content, f1)
            temp_file1 = f1.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='2.yaml', delete=False) as f2:
            yaml.dump(config2_content, f2)
            temp_file2 = f2.name
        
        try:
            with patch('src.prompts.loader.CONFIGS_DIR', os.path.dirname(temp_file1)):
                config1_name = os.path.basename(temp_file1)[:-5]
                config2_name = os.path.basename(temp_file2)[:-5]
                
                # Load from both configs
                prompt1 = get_prompt(f"{config1_name}.test.prompt1")
                prompt2 = get_prompt(f"{config2_name}.test.prompt1")
                
                # Should be different content
                assert prompt1 == 'Content from config 1'
                assert prompt2 == 'Content from config 2'
                assert prompt1 != prompt2
        finally:
            os.unlink(temp_file1)
            os.unlink(temp_file2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])