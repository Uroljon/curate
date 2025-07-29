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
    
    thinking_prompts = {
        "analytical": """<think>
Ich analysiere diesen Text systematisch:
1. Welche Informationen sind gegeben?
2. Welche Muster erkenne ich?
3. Wie ordne ich diese in die Kategorien ein?
4. Welche Unsicherheiten bestehen?
5. Welche Konfidenz ist angemessen?
</think>""",
        
        "comparative": """<think>
Ich vergleiche jeden gefundenen Punkt:
- Handelt es sich um eine Aktion oder ein Messwert?
- Ähnelt es bekannten Maßnahmen oder Indikatoren?
- Welche Kategorie passt besser und warum?
- Wie sicher bin ich bei dieser Einschätzung?
</think>""",
        
        "systematic": """<think>
Ich gehe systematisch vor:
1. Alle Textabschnitte identifizieren
2. Jeden Punkt einzeln klassifizieren
3. Begründung für jede Entscheidung notieren
4. Konfidenz basierend auf Klarheit der Hinweise
5. Gesamtvalidierung der Ergebnisse
</think>""",
        
        "contextual": """<think>
Ich berücksichtige den Kontext vollständig:
1. Dokumentstruktur und Hierarchie verstehen
2. Bezug zu anderen Abschnitten herstellen
3. Verwaltungslogik und Amtssprache anwenden
4. Typische Muster in deutschen Strategiedokumenten
5. Konfidenz durch Kontextklarheit bestimmen
</think>"""
    }
    
    enhanced_prompt = f"{thinking_prompts.get(thinking_mode, thinking_prompts['analytical'])}\n\n{prompt}"
    
    return query_ollama_structured(
        prompt=enhanced_prompt,
        response_model=response_model,
        model=model,
        temperature=temperature,
        system_message=system_message,
        log_file_path=log_file_path,
        log_context=f"{log_context} (thinking_mode: {thinking_mode})" if log_context else f"thinking_mode: {thinking_mode}",
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
