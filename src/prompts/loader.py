"""
Simple YAML-based prompt loader with caching.

This module provides a minimal interface for loading prompts from YAML configuration files.
Uses in-memory caching for performance and supports template substitution.
"""

import os
from typing import Any, Dict, Optional
import yaml

# In-memory cache for loaded YAML files
_yaml_cache: Dict[str, Dict[str, Any]] = {}

# Base directory for prompt configs
PROMPTS_DIR = os.path.dirname(__file__)
CONFIGS_DIR = os.path.join(PROMPTS_DIR, "configs")


def _load_yaml_file(config_name: str) -> Dict[str, Any]:
    """Load and cache a YAML configuration file."""
    if config_name in _yaml_cache:
        return _yaml_cache[config_name]
    
    config_path = os.path.join(CONFIGS_DIR, f"{config_name}.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Prompt config not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        _yaml_cache[config_name] = config
        return config
    except Exception as e:
        raise RuntimeError(f"Failed to load prompt config {config_name}: {e}")


def get_prompt(prompt_name: str, **kwargs) -> str:
    """
    Get a prompt by name with optional template substitution.
    
    Prompt names use the format: "config.category.name"
    Examples:
    - "operations.system_messages.operations_extraction"
    - "extraction.templates.simplified_chunk"
    - "legacy.system_messages.stage1_action_fields"
    - "utils.variables.consolidate_examples"
    
    Args:
        prompt_name: Dot-separated path to the prompt
        **kwargs: Template variables for substitution
    
    Returns:
        Formatted prompt string
        
    Raises:
        ValueError: If prompt name format is invalid
        KeyError: If prompt is not found in the configuration
        FileNotFoundError: If configuration file doesn't exist
    """
    # Parse prompt name
    parts = prompt_name.split(".")
    if len(parts) < 3:
        raise ValueError(f"Invalid prompt name format. Expected 'config.category.name', got: {prompt_name}")
    
    config_name = parts[0]
    category = parts[1]
    name = ".".join(parts[2:])  # Support nested keys
    
    # Load configuration
    config = _load_yaml_file(config_name)
    
    # Navigate to the prompt (handle nested keys)
    try:
        current = config[category]
        # Navigate through nested keys
        for key in name.split("."):
            current = current[key]
        prompt_template = current
    except (KeyError, TypeError) as e:
        available_keys = list(config.get(category, {}).keys())
        raise KeyError(f"Prompt '{name}' not found in {config_name}.{category}. Available: {available_keys}")
    
    if not isinstance(prompt_template, str):
        raise ValueError(f"Prompt must be a string, got {type(prompt_template)}: {prompt_name}")
    
    # Apply template substitution if variables provided
    if kwargs:
        try:
            return prompt_template.format(**kwargs)
        except KeyError as e:
            raise KeyError(f"Missing template variable {e} for prompt: {prompt_name}")
    
    return prompt_template


def clear_cache() -> None:
    """Clear the prompt cache. Useful for testing or development."""
    global _yaml_cache
    _yaml_cache.clear()


def list_available_prompts(config_name: Optional[str] = None) -> Dict[str, Any]:
    """
    List all available prompts, optionally filtered by config.
    
    Args:
        config_name: Optional config to filter by (e.g., "operations")
        
    Returns:
        Dictionary structure showing available prompts
    """
    if config_name:
        config = _load_yaml_file(config_name)
        return {config_name: config}
    
    # Load all configs
    all_prompts = {}
    for filename in os.listdir(CONFIGS_DIR):
        if filename.endswith('.yaml'):
            config_name = filename[:-5]  # Remove .yaml extension
            try:
                all_prompts[config_name] = _load_yaml_file(config_name)
            except Exception as e:
                all_prompts[config_name] = f"Error loading: {e}"
    
    return all_prompts