# llm.py
from typing import TypeVar

from pydantic import BaseModel

from .config import MODEL_NAME, MODEL_TEMPERATURE
from .llm_providers import get_llm_provider

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
</think>""",
    }

    enhanced_prompt = f"{thinking_prompts.get(thinking_mode, thinking_prompts['analytical'])}\n\n{prompt}"

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
