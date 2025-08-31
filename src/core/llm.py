# llm.py
from typing import TypeVar

from pydantic import BaseModel

from .config import MODEL_NAME, MODEL_TEMPERATURE
from .llm_providers import get_llm_provider
from ..prompts import get_prompt

T = TypeVar("T", bound=BaseModel)


def query_ollama_with_thinking_mode(
    prompt: str,
    response_model: type[T],
    thinking_mode: str = "analytical",
    model: str = MODEL_NAME,
    temperature: float = MODEL_TEMPERATURE,
    system_message: str | None = None,
    log_file_path: str | None = None,
    log_context: str | None = None,
    override_num_predict: int | None = None,
) -> T | None:
    """
    Enhanced LLM query with qwen3 thinking modes for complex reasoning.

    Args:
        prompt: The input prompt
        response_model: Pydantic model class for the expected response structure
        thinking_mode: Mode of reasoning to use
            - "analytical": Step-by-step logical reasoning
            - "comparative": Compare and contrast options
            - "systematic": Methodical categorization approach
            - "contextual": Deep context understanding with hierarchy
        model: Model name to use
        temperature: Controls randomness (defaults to MODEL_TEMPERATURE)
        system_message: Optional system message for role definition
        log_file_path: Optional path to log dialog to file
        log_context: Optional context info for the log entry
        override_num_predict: Optional override for num_predict tokens

    Returns:
        Validated Pydantic model instance or None if failed
    """

    # Validate thinking mode and get prefix from YAML
    try:
        prefix = get_prompt(f"core.thinking_modes.{thinking_mode}")
    except KeyError:
        raise ValueError(f"Invalid thinking_mode '{thinking_mode}'. Available modes: analytical, comparative, systematic, contextual")

    enhanced_prompt = f"{prefix}\n\n{prompt}"

    return query_ollama_structured(
        prompt=enhanced_prompt,
        response_model=response_model,
        model=model,
        temperature=temperature,
        system_message=system_message,
        log_file_path=log_file_path,
        log_context=(
            f"{log_context} (thinking_mode: {thinking_mode})"
            if log_context
            else f"thinking_mode: {thinking_mode}"
        ),
        override_num_predict=override_num_predict,
    )


def query_ollama_structured(
    prompt: str,
    response_model: type[T],
    model: str = MODEL_NAME,
    temperature: float = MODEL_TEMPERATURE,
    system_message: str | None = None,
    log_file_path: str | None = None,
    log_context: str | None = None,
    override_num_predict: int | None = None,
) -> T | None:
    """
    Query LLM with structured output using Pydantic models.

    This function maintains backward compatibility while using the new provider abstraction.

    Args:
        prompt: The input prompt
        response_model: Pydantic model class for the expected response structure
        model: Model name to use
        temperature: Controls randomness (defaults to MODEL_TEMPERATURE)
        system_message: Optional system message for role definition
        log_file_path: Optional path to log dialog to file
        log_context: Optional context info for the log entry
        override_num_predict: Optional override for num_predict tokens (for large outputs)

    Returns:
        Validated Pydantic model instance or None if failed
    """
    # Get the appropriate provider
    provider = get_llm_provider(model_name=model, temperature=temperature)

    # Use the provider to query with structured output
    return provider.query_structured(
        prompt=prompt,
        response_model=response_model,
        system_message=system_message,
        log_file_path=log_file_path,
        log_context=log_context,
        override_num_predict=override_num_predict,
    )


# Note: The _log_llm_dialog function has been moved to LLMProvider base class
# in llm_providers.py to share implementation between providers
