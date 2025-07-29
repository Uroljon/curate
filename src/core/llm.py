# llm.py
import os
from datetime import datetime, timezone
from typing import TypeVar

import requests
from pydantic import BaseModel

from .config import (
    MODEL_NAME,
    MODEL_TEMPERATURE,
    MODEL_TIMEOUT,
    OLLAMA_CHAT_URL,
    STRUCTURED_OUTPUT_OPTIONS,
)

T = TypeVar("T", bound=BaseModel)


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
    Query Ollama with structured output using Pydantic models.

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
    try:
        messages = []

        if system_message:
            messages.append({"role": "system", "content": system_message})

        messages.append({"role": "user", "content": prompt})

        options = STRUCTURED_OUTPUT_OPTIONS.copy()
        options["temperature"] = temperature

        # Override num_predict if specified (e.g., for large aggregations)
        if override_num_predict:
            options["num_predict"] = override_num_predict

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

            # Log the dialog if log file path is provided
            if log_file_path:
                _log_llm_dialog(
                    log_file_path,
                    system_message,
                    prompt,
                    content,
                    log_context,
                    model,
                    temperature,
                )

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


def _log_llm_dialog(
    log_file_path: str,
    system_message: str | None,
    user_prompt: str,
    llm_response: str,
    log_context: str | None,
    model: str,
    temperature: float,
) -> None:
    """
    Log LLM dialog to file with formatted structure.

    Args:
        log_file_path: Path to the log file
        system_message: System message sent to LLM
        user_prompt: User prompt sent to LLM
        llm_response: Raw response from LLM
        log_context: Context information (stage, chunk info, etc.)
        model: Model name used
        temperature: Temperature setting used
    """
    try:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        # Read existing content if file exists to get interaction counter
        interaction_num = 1
        if os.path.exists(log_file_path):
            with open(log_file_path, encoding="utf-8") as f:
                content = f.read()
                # Count existing interactions
                interaction_num = content.count("LLM INTERACTION #") + 1

        # Format the log entry
        log_entry = f"""
========================================
[{timestamp}] LLM INTERACTION #{interaction_num}
MODEL: {model} (temp: {temperature})
{f"CONTEXT: {log_context}" if log_context else ""}
========================================

"""

        if system_message:
            log_entry += f"SYSTEM MESSAGE:\n{system_message}\n\n"

        log_entry += f"USER PROMPT:\n{user_prompt}\n\n"
        log_entry += f"LLM RESPONSE:\n{llm_response}\n\n"

        # Append to log file
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(log_entry)

    except Exception as e:
        print(f"⚠️ Failed to log LLM dialog: {e!s}")
