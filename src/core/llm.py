# llm.py
from typing import Optional, Type, TypeVar

import requests
from pydantic import BaseModel

from .config import (
    GENERATION_OPTIONS,
    MODEL_NAME,
    MODEL_TEMPERATURE,
    MODEL_TIMEOUT,
    OLLAMA_API_URL,
    OLLAMA_CHAT_URL,
    STRUCTURED_OUTPUT_OPTIONS,
)

T = TypeVar("T", bound=BaseModel)


def query_ollama(
    prompt: str,
    model: str = MODEL_NAME,
    temperature: float = 0.0,
    system_message: str = None,
) -> str:
    """
    Query Ollama with optimized parameters for structured JSON extraction.

    Args:
        prompt: The input prompt
        model: Model name to use
        temperature: Controls randomness (0.0 = deterministic)
        system_message: Optional system message for role definition

    Returns:
        Generated text response
    """
    try:
        # Build request body
        options = GENERATION_OPTIONS.copy()
        if temperature is not None:
            options["temperature"] = temperature

        request_body = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }

        # Add system message if provided
        if system_message:
            request_body["system"] = system_message

        response = requests.post(
            OLLAMA_API_URL, json=request_body, timeout=MODEL_TIMEOUT
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except requests.RequestException as e:
        print(f"❌ LLM API Error: {e!s}")
        return f"LLM API Error: {e!s}"


def query_ollama_structured(
    prompt: str,
    response_model: type[T],
    model: str = MODEL_NAME,
    temperature: float = MODEL_TEMPERATURE,
    system_message: str | None = None,
) -> T | None:
    """
    Query Ollama with structured output using Pydantic models.

    Args:
        prompt: The input prompt
        response_model: Pydantic model class for the expected response structure
        model: Model name to use
        temperature: Controls randomness (defaults to MODEL_TEMPERATURE)
        system_message: Optional system message for role definition

    Returns:
        Validated Pydantic model instance or None if failed
    """
    try:
        messages = []

        if system_message:
            messages.append({"role": "system", "content": system_message})

        messages.append({"role": "user", "content": prompt})

        options = STRUCTURED_OUTPUT_OPTIONS.copy()
        options["temperature"] = temperature

        request_body = {
            "model": model,
            "messages": messages,
            "stream": False,
            "format": response_model.model_json_schema(),
            "options": options,
        }

        response = requests.post(
            OLLAMA_CHAT_URL, json=request_body, timeout=MODEL_TIMEOUT
        )
        response.raise_for_status()

        data = response.json()

        if "message" in data and "content" in data["message"]:
            content = data["message"]["content"].strip()

            # Try to parse and validate with Pydantic
            try:
                return response_model.model_validate_json(content)
            except Exception as e:
                print(f"⚠️ JSON validation failed: {e!s}")

                # Try json-repair as fallback
                try:
                    from json_repair import repair_json

                    repaired = repair_json(content)
                    return response_model.model_validate(repaired)
                except Exception as repair_error:
                    print(f"❌ JSON repair failed: {repair_error!s}")
                    return None

        return None

    except requests.RequestException as e:
        print(f"❌ LLM API Error: {e!s}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e!s}")
        return None
