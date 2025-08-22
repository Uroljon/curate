"""
Token tracking utilities for CURATE.

This module provides centralized token estimation and logging functions
to eliminate code duplication across the codebase.
"""

import json
from typing import Any


# Global token tracking for pipeline summaries
_llm_token_accumulator = {
    "total_input_tokens": 0,
    "total_output_tokens": 0,
    "total_calls": 0
}


def reset_llm_token_tracking():
    """Reset the global LLM token accumulator for a new extraction run."""
    global _llm_token_accumulator
    _llm_token_accumulator = {
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_calls": 0
    }


def get_llm_token_summary() -> dict[str, int]:
    """Get the current LLM token usage summary."""
    return _llm_token_accumulator.copy()


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
    global _llm_token_accumulator
    
    # Get content for token calculation
    temp_content = response.choices[0].message.content.strip()
    output_tokens = estimate_tokens(temp_content)
    efficiency = output_tokens / input_tokens if input_tokens > 0 else 0
    
    # Accumulate tokens for pipeline summary
    _llm_token_accumulator["total_input_tokens"] += input_tokens
    _llm_token_accumulator["total_output_tokens"] += output_tokens
    _llm_token_accumulator["total_calls"] += 1
    
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


def track_json_state_change(
    before_data: Any, 
    after_data: Any, 
    context: str = "Operation",
    verbose: bool = True
) -> dict[str, Any]:
    """
    Track and log JSON size changes between two states.
    
    Args:
        before_data: State before changes
        after_data: State after changes
        context: Context label for the change (e.g., "Chunk 1", "Operations")
        verbose: Whether to print detailed logs
        
    Returns:
        Dictionary with detailed metrics about the change
    """
    # Calculate sizes before
    before_json = json.dumps(before_data, ensure_ascii=False) if isinstance(before_data, dict) else str(before_data)
    before_tokens = estimate_tokens(before_json)
    before_chars = len(before_json)
    
    # Calculate sizes after
    after_json = json.dumps(after_data, ensure_ascii=False) if isinstance(after_data, dict) else str(after_data)
    after_tokens = estimate_tokens(after_json)
    after_chars = len(after_json)
    
    # Calculate deltas
    token_delta = after_tokens - before_tokens
    char_delta = after_chars - before_chars
    token_growth_pct = (token_delta / before_tokens * 100) if before_tokens > 0 else 0
    char_to_token_before = before_chars / before_tokens if before_tokens > 0 else 0
    char_to_token_after = after_chars / after_tokens if after_tokens > 0 else 0
    
    metrics = {
        "before_tokens": before_tokens,
        "after_tokens": after_tokens,
        "token_delta": token_delta,
        "token_growth_pct": token_growth_pct,
        "before_chars": before_chars,
        "after_chars": after_chars,
        "char_delta": char_delta,
        "char_to_token_before": char_to_token_before,
        "char_to_token_after": char_to_token_after,
    }
    
    if verbose:
        # Color coding for growth
        if token_delta > 0:
            delta_str = f"+{token_delta:,}"
            emoji = "📈"
        elif token_delta < 0:
            delta_str = f"{token_delta:,}"
            emoji = "📉"
        else:
            delta_str = "0"
            emoji = "➡️"
        
        print(f"\n{emoji} JSON State Change [{context}]:")
        print(f"   📊 Tokens: {before_tokens:,} → {after_tokens:,} ({delta_str} tokens, {token_growth_pct:+.1f}%)")
        print(f"   📝 Chars:  {before_chars:,} → {after_chars:,} ({char_delta:+,} chars)")
        print(f"   🔤 Ratio:  {char_to_token_before:.2f} → {char_to_token_after:.2f} chars/token")
        
    return metrics