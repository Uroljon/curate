# llm.py
import requests
from typing import Type, TypeVar, Optional
from pydantic import BaseModel

OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:14b"  # Qwen2.5 14B - specialized for structured output and multilingual tasks

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
        request_body = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,  # Faster sampling
                "top_p": 0.9,  # More vocabulary options
                "top_k": 40,  # Balanced vocabulary
                "repeat_penalty": 1.05,  # Light repetition control
                "num_predict": 800,  # Shorter for speed
                "stop": ["```", "</json>"],
            },
        }

        # Add system message if provided
        if system_message:
            request_body["system"] = system_message

        response = requests.post(OLLAMA_API_URL, json=request_body, timeout=180)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except requests.RequestException as e:
        print(f"❌ LLM API Error: {str(e)}")
        return f"LLM API Error: {str(e)}"


def query_ollama_structured(
    prompt: str,
    response_model: Type[T],
    model: str = MODEL_NAME,
    temperature: float = 0.0,
    system_message: Optional[str] = None,
) -> Optional[T]:
    """
    Query Ollama with structured output using Pydantic models.

    Args:
        prompt: The input prompt
        response_model: Pydantic model class for the expected response structure
        model: Model name to use
        temperature: Controls randomness (0.0 = deterministic)
        system_message: Optional system message for role definition

    Returns:
        Validated Pydantic model instance or None if failed
    """
    try:
        messages = []

        if system_message:
            messages.append({"role": "system", "content": system_message})

        messages.append({"role": "user", "content": prompt})

        request_body = {
            "model": model,
            "messages": messages,
            "stream": False,
            "format": response_model.model_json_schema(),
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
                "num_predict": 2000,  # Increased for structured output
                "stop": ["</json>", "```"],
            },
        }

        response = requests.post(OLLAMA_CHAT_URL, json=request_body, timeout=180)
        response.raise_for_status()

        data = response.json()

        if "message" in data and "content" in data["message"]:
            content = data["message"]["content"].strip()

            # Try to parse and validate with Pydantic
            try:
                return response_model.model_validate_json(content)
            except Exception as e:
                print(f"⚠️ JSON validation failed: {str(e)}")

                # Try json-repair as fallback
                try:
                    from json_repair import repair_json

                    repaired = repair_json(content)
                    return response_model.model_validate(repaired)
                except Exception as repair_error:
                    print(f"❌ JSON repair failed: {str(repair_error)}")
                    return None

        return None

    except requests.RequestException as e:
        print(f"❌ LLM API Error: {str(e)}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return None
