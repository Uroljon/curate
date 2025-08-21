"""
Token tracking utilities for CURATE.

This module provides centralized token estimation and logging functions
to eliminate code duplication across the codebase.
"""

import json
from typing import Any


def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a text string.
    
    Uses the established conversion ratio of 1 token ≈ 3.5 characters for German text.
    
    Args:
        text: Text to estimate tokens for
        
    Returns:
        Estimated number of tokens
    """
    if not text:
        return 0
    return int(len(text) / 3.5)


def log_llm_response_tokens(response, input_tokens: int, response_time: float) -> int:
    """
    Log LLM response with token metrics and efficiency calculation.
    
    Args:
        response: LLM response object with choices[0].message.content
        input_tokens: Number of input tokens sent to LLM
        response_time: Response time in seconds
        
    Returns:
        Number of output tokens
    """
    # Get content for token calculation
    temp_content = response.choices[0].message.content.strip()
    output_tokens = estimate_tokens(temp_content)
    efficiency = output_tokens / input_tokens if input_tokens > 0 else 0
    
    print(f"      🤖 LLM: {input_tokens:,} → {output_tokens:,} tokens ({efficiency:.1f}x expansion) in {response_time:.2f}s")
    
    return output_tokens


def calculate_json_tokens(data: Any, label: str = "JSON") -> tuple[int, str]:
    """
    Calculate tokens for JSON data and log the metrics.
    
    Args:
        data: Data to serialize to JSON
        label: Label for the log message (e.g., "Final JSON", "Operations JSON")
        
    Returns:
        Tuple of (token_count, json_string)
    """
    json_str = json.dumps(data, ensure_ascii=False)
    tokens = estimate_tokens(json_str)
    
    print(f"📋 {label}: {tokens:,} tokens ({len(json_str):,} chars)")
    
    return tokens, json_str


def calculate_json_tokens_quiet(data: Any) -> tuple[int, str]:
    """
    Calculate tokens for JSON data without logging (for internal use).
    
    Args:
        data: Data to serialize to JSON
        
    Returns:
        Tuple of (token_count, json_string)
    """
    json_str = json.dumps(data, ensure_ascii=False)
    tokens = estimate_tokens(json_str)
    
    return tokens, json_str