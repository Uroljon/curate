"""
LLM Provider abstraction layer for supporting multiple LLM backends.
Supports both Ollama and vLLM (via OpenAI-compatible API).
"""

import json
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, TypeVar

import requests
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, model_name: str, temperature: float = 0.2):
        self.model_name = model_name
        self.temperature = temperature

    @abstractmethod
    def query_structured(
        self,
        prompt: str,
        response_model: type[T],
        system_message: str | None = None,
        log_file_path: str | None = None,
        log_context: str | None = None,
        override_num_predict: int | None = None,
    ) -> T | None:
        """Query LLM with structured output using Pydantic models."""
        pass

    def _log_llm_dialog(
        self,
        log_file_path: str,
        system_message: str | None,
        user_prompt: str,
        llm_response: str,
        log_context: str | None,
    ) -> None:
        """Log LLM dialog to file with formatted structure."""
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
MODEL: {self.model_name} (temp: {self.temperature})
BACKEND: {self.__class__.__name__}
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


class OllamaProvider(LLMProvider):
    """Ollama provider implementation."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.2,
        ollama_host: str = "localhost:11434",
        structured_output_options: dict[str, Any] | None = None,
        timeout: int = 600,
    ):
        super().__init__(model_name, temperature)
        self.chat_url = f"http://{ollama_host}/api/chat"
        self.timeout = timeout
        self.structured_output_options = structured_output_options or {}

    def query_structured(
        self,
        prompt: str,
        response_model: type[T],
        system_message: str | None = None,
        log_file_path: str | None = None,
        log_context: str | None = None,
        override_num_predict: int | None = None,
    ) -> T | None:
        """Query Ollama with structured output using Pydantic models."""
        try:
            messages = []

            if system_message:
                messages.append({"role": "system", "content": system_message})

            messages.append({"role": "user", "content": prompt})

            options = self.structured_output_options.copy()
            options["temperature"] = self.temperature

            # Override num_predict if specified (e.g., for large aggregations)
            if override_num_predict:
                options["num_predict"] = override_num_predict

            request_body = {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "format": response_model.model_json_schema(),
                "options": options,
            }

            response = requests.post(
                self.chat_url, json=request_body, timeout=self.timeout
            )
            response.raise_for_status()

            data = response.json()

            if "message" in data and "content" in data["message"]:
                content = data["message"]["content"].strip()

                # Log the dialog if log file path is provided
                if log_file_path:
                    self._log_llm_dialog(
                        log_file_path,
                        system_message,
                        prompt,
                        content,
                        log_context,
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
            print(f"❌ Ollama API Error: {e!s}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error: {e!s}")
            return None


class VLLMProvider(LLMProvider):
    """vLLM provider implementation using OpenAI-compatible API."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.2,
        vllm_host: str = "localhost:8001",
        api_key: str = "EMPTY",
        max_tokens: int = 30720,
        timeout: int = 600,
    ):
        super().__init__(model_name, temperature)
        self.base_url = f"http://{vllm_host}/v1"
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.timeout = timeout

        # Initialize OpenAI client
        try:
            from openai import OpenAI

            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )

            # Try to get model info to determine actual context length
            try:
                models = self.client.models.list()
                for model in models.data:
                    if model.id == self.model_name:
                        self.max_model_len = getattr(model, "max_model_len", 16384)
                        # Conservative: assume input uses 60% of context, leave 40% for output
                        # For 16K model: ~6.5K tokens for output
                        self.max_tokens = min(max_tokens, int(self.max_model_len * 0.4))
                        print(
                            f"📊 vLLM Model: {self.model_name}, Context: {self.max_model_len}, Max output: {self.max_tokens}"
                        )
                        break
            except Exception:
                # If we can't get model info, use conservative defaults
                self.max_model_len = 16384
                self.max_tokens = min(max_tokens, 8192)

        except ImportError:
            msg = "OpenAI client library required for vLLM. Install with: pip install openai"
            raise ImportError(msg) from None

    def query_structured(
        self,
        prompt: str,
        response_model: type[T],
        system_message: str | None = None,
        log_file_path: str | None = None,
        log_context: str | None = None,
        override_num_predict: int | None = None,
    ) -> T | None:
        """Query vLLM with structured output using JSON mode."""
        try:
            messages = []

            # For structured output, we need to include the schema in the prompt
            schema_str = json.dumps(response_model.model_json_schema(), indent=2)

            # Check if this is a Qwen3 model - if so, add /no_think to disable thinking mode for JSON
            # Thinking mode can interfere with structured JSON output
            no_think_suffix = ""
            if "qwen3" in self.model_name.lower():
                no_think_suffix = " /no_think"

            structured_prompt = f"""{prompt}

IMPORTANT: You must respond with valid JSON that matches this exact schema:
{schema_str}

Respond ONLY with the JSON object, no additional text.{no_think_suffix}"""

            if system_message:
                messages.append({"role": "system", "content": system_message})

            messages.append({"role": "user", "content": structured_prompt})

            # Use structured output with vLLM
            # vLLM supports guided_json in extra_body or json_schema format

            # Use Qwen3-specific parameters if applicable
            temperature = self.temperature
            top_p = 0.95
            top_k = 20

            # For Qwen3 AWQ models, adjust parameters per documentation
            if "qwen3" in self.model_name.lower() and "awq" in self.model_name.lower():
                temperature = 0.7  # Non-thinking mode recommendation
                top_p = 0.8

            try:
                # First try: Use guided_json in extra_body (recommended for vLLM)
                llm_start_time = time.time()
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=override_num_predict or self.max_tokens,
                    extra_body={
                        "guided_json": response_model.model_json_schema(),
                        "top_k": top_k,
                        "min_p": 0,  # Qwen3 recommendation
                    },
                    timeout=self.timeout,
                )
                llm_response_time = time.time() - llm_start_time
                print(f"      🤖 LLM response in {llm_response_time:.2f}s")
            except Exception:
                try:
                    # Second try: Use response_format with json_schema
                    llm_start_time = time.time()
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=override_num_predict or self.max_tokens,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": response_model.__name__,
                                "schema": response_model.model_json_schema(),
                            },
                        },
                        extra_body={"top_k": top_k, "min_p": 0},
                        timeout=self.timeout,
                    )
                    llm_response_time = time.time() - llm_start_time
                    print(f"      🤖 LLM response in {llm_response_time:.2f}s")
                except Exception:
                    # Final fallback: Use prompt-based JSON generation
                    print(
                        "⚠️ vLLM structured output not available, using prompt-based JSON generation"
                    )
                    llm_start_time = time.time()
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=override_num_predict or self.max_tokens,
                        extra_body={"top_k": top_k, "min_p": 0},
                        timeout=self.timeout,
                    )
                    llm_response_time = time.time() - llm_start_time
                    print(f"      🤖 LLM response in {llm_response_time:.2f}s")

            content = response.choices[0].message.content.strip()

            # Handle Qwen3 thinking mode output if present
            # Even with /no_think, sometimes the model may include thinking tags
            if "<think>" in content and "</think>" in content:
                # Extract content after thinking block
                think_end = content.find("</think>")
                if think_end != -1:
                    actual_content = content[think_end + 8 :].strip()
                    if actual_content:
                        content = actual_content

            # Log the dialog if log file path is provided
            if log_file_path:
                self._log_llm_dialog(
                    log_file_path,
                    system_message,
                    structured_prompt,
                    content,
                    log_context,
                )

            # Try to parse and validate with Pydantic
            try:
                return response_model.model_validate_json(content)
            except Exception as e:
                print(f"⚠️ JSON validation failed: {e!s}")
                # Debug: show first 500 chars of response
                print(f"📋 Response preview: {content[:500]}...")

                # Try json-repair as fallback
                try:
                    from json_repair import repair_json

                    repaired = repair_json(content)
                    return response_model.model_validate(repaired)
                except Exception as repair_error:
                    print(f"❌ JSON repair failed: {repair_error!s}")
                    # Debug: Check if response is empty JSON
                    if content.strip() in [
                        '{"action_fields": []}',
                        '{"action_fields":[]}',
                    ]:
                        print("⚠️ Model returned empty action fields")
                    return None

        except Exception as e:
            print(f"❌ vLLM API Error: {e!s}")
            return None


class OpenRouterProvider(LLMProvider):
    """OpenRouter provider for unified access to multiple LLM providers."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.1,
        api_key: str | None = None,
        max_tokens: int = 4096,
        timeout: int = 300,
    ):
        super().__init__(model_name, temperature)
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.max_tokens = max_tokens
        self.timeout = timeout

        # Initialize OpenAI client with OpenRouter endpoint
        try:
            from openai import OpenAI

            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
            print(f"📡 OpenRouter client initialized for model: {self.model_name}")
        except ImportError:
            msg = "OpenAI library required for OpenRouter. Install with: pip install openai"
            raise ImportError(msg) from None

    def query_structured(
        self,
        prompt: str,
        response_model: type[T],
        system_message: str | None = None,
        log_file_path: str | None = None,
        log_context: str | None = None,
        override_num_predict: int | None = None,
    ) -> T | None:
        """Query OpenRouter with structured output using JSON schema."""
        try:
            messages = []

            if system_message:
                messages.append({"role": "system", "content": system_message})

            # Include schema in prompt for better structured output
            schema_str = json.dumps(response_model.model_json_schema(), indent=2)
            structured_prompt = f"""{prompt}

IMPORTANT: Respond with valid JSON matching this schema:
{schema_str}

Respond ONLY with the JSON object, no additional text."""

            messages.append({"role": "user", "content": structured_prompt})

            llm_start_time = time.time()

            # OpenRouter supports structured output for compatible models
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=override_num_predict or self.max_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_model.__name__,
                        "strict": True,
                        "schema": response_model.model_json_schema(),
                    },
                },
                extra_body={
                    "provider": {
                        "require_parameters": True  # Ensure structured output support
                    }
                },
                timeout=self.timeout,
            )
            llm_response_time = time.time() - llm_start_time
            print(f"      🤖 OpenRouter response in {llm_response_time:.2f}s")

            content = response.choices[0].message.content.strip()

            # Log the dialog
            if log_file_path:
                self._log_llm_dialog(
                    log_file_path,
                    system_message,
                    structured_prompt,
                    content,
                    log_context,
                )

            # Parse and validate with Pydantic
            try:
                return response_model.model_validate_json(content)
            except Exception as e:
                print(f"⚠️ JSON validation failed: {e}")
                # Try json-repair as fallback
                try:
                    from json_repair import repair_json

                    repaired = repair_json(content)
                    return response_model.model_validate(repaired)
                except Exception:
                    print("❌ JSON repair failed")
                    return None

        except Exception as e:
            print(f"❌ OpenRouter API Error: {e}")
            return None


class ExternalAPIProvider(LLMProvider):
    """External API provider for OpenAI and Google Gemini models."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.2,
        api_provider: str = "openai",  # "openai" or "gemini"
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 4096,
        timeout: int = 300,
    ):
        super().__init__(model_name, temperature)
        self.api_provider = api_provider.lower()
        self.api_key = api_key
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.timeout = timeout

        # Initialize appropriate client
        if self.api_provider == "openai":
            self._init_openai_client()
        elif self.api_provider == "gemini":
            self._init_gemini_client()
        else:
            msg = f"Unsupported API provider: {api_provider}"
            raise ValueError(msg)

    def _init_openai_client(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI

            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
            print(f"📡 OpenAI client initialized for model: {self.model_name}")
        except ImportError:
            msg = "OpenAI library required. Install with: pip install openai"
            raise ImportError(msg) from None

    def _init_gemini_client(self):
        """Initialize Google Gemini client."""
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model_name)
            print(f"📡 Gemini client initialized for model: {self.model_name}")
        except ImportError:
            msg = "Google Generative AI library required. Install with: pip install google-generativeai"
            raise ImportError(msg) from None

    def query_structured(
        self,
        prompt: str,
        response_model: type[T],
        system_message: str | None = None,
        log_file_path: str | None = None,
        log_context: str | None = None,
        override_num_predict: int | None = None,
    ) -> T | None:
        """Query external API with structured output."""
        if self.api_provider == "openai":
            return self._query_openai_structured(
                prompt,
                response_model,
                system_message,
                log_file_path,
                log_context,
                override_num_predict,
            )
        elif self.api_provider == "gemini":
            return self._query_gemini_structured(
                prompt,
                response_model,
                system_message,
                log_file_path,
                log_context,
                override_num_predict,
            )

    def _query_openai_structured(
        self,
        prompt: str,
        response_model: type[T],
        system_message: str | None,
        log_file_path: str | None,
        log_context: str | None,
        override_num_predict: int | None,
    ) -> T | None:
        """Query OpenAI with structured output."""
        try:
            messages = []

            if system_message:
                messages.append({"role": "system", "content": system_message})

            # Include schema in prompt for better structured output
            schema_str = json.dumps(response_model.model_json_schema(), indent=2)
            structured_prompt = f"""{prompt}

IMPORTANT: Respond with valid JSON matching this schema:
{schema_str}

Respond ONLY with the JSON object, no additional text."""

            messages.append({"role": "user", "content": structured_prompt})

            llm_start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=override_num_predict or self.max_tokens,
                response_format=(
                    {"type": "json_object"} if "gpt" in self.model_name else None
                ),
                timeout=self.timeout,
            )
            llm_response_time = time.time() - llm_start_time
            print(f"      🤖 OpenAI response in {llm_response_time:.2f}s")

            content = response.choices[0].message.content.strip()

            # Log the dialog
            if log_file_path:
                self._log_llm_dialog(
                    log_file_path,
                    system_message,
                    structured_prompt,
                    content,
                    log_context,
                )

            # Parse and validate with Pydantic
            try:
                return response_model.model_validate_json(content)
            except Exception as e:
                print(f"⚠️ JSON validation failed: {e}")
                # Try json-repair as fallback
                try:
                    from json_repair import repair_json

                    repaired = repair_json(content)
                    return response_model.model_validate(repaired)
                except Exception:
                    print("❌ JSON repair failed")
                    return None

        except Exception as e:
            print(f"❌ OpenAI API Error: {e}")
            return None

    def _query_gemini_structured(
        self,
        prompt: str,
        response_model: type[T],
        system_message: str | None,
        log_file_path: str | None,
        log_context: str | None,
        override_num_predict: int | None,
    ) -> T | None:
        """Query Gemini with structured output."""
        try:
            # Combine system message and prompt
            full_prompt = prompt
            if system_message:
                full_prompt = f"{system_message}\n\n{prompt}"

            # Include schema in prompt
            schema_str = json.dumps(response_model.model_json_schema(), indent=2)
            structured_prompt = f"""{full_prompt}

IMPORTANT: Respond with valid JSON matching this schema:
{schema_str}

Respond ONLY with the JSON object, no additional text."""

            llm_start_time = time.time()
            response = self.client.generate_content(
                structured_prompt,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": override_num_predict or self.max_tokens,
                },
            )
            llm_response_time = time.time() - llm_start_time
            print(f"      🤖 Gemini response in {llm_response_time:.2f}s")

            content = response.text.strip()

            # Log the dialog
            if log_file_path:
                self._log_llm_dialog(
                    log_file_path,
                    system_message,
                    structured_prompt,
                    content,
                    log_context,
                )

            # Parse and validate with Pydantic
            try:
                return response_model.model_validate_json(content)
            except Exception as e:
                print(f"⚠️ JSON validation failed: {e}")
                # Try json-repair as fallback
                try:
                    from json_repair import repair_json

                    repaired = repair_json(content)
                    return response_model.model_validate(repaired)
                except Exception:
                    print("❌ JSON repair failed")
                    return None

        except Exception as e:
            print(f"❌ Gemini API Error: {e}")
            return None


def get_llm_provider(
    backend: str | None = None,
    model_name: str | None = None,
    temperature: float | None = None,
    **kwargs,
) -> LLMProvider:
    """
    Factory function to get the appropriate LLM provider.

    Args:
        backend: LLM backend to use ('ollama', 'vllm', 'openai', 'gemini', 'openrouter'). Defaults to env var LLM_BACKEND.
        model_name: Model name override. If not provided, uses config defaults.
        temperature: Temperature override. If not provided, uses config defaults.
        **kwargs: Additional provider-specific arguments.

    Returns:
        LLMProvider instance
    """
    from .config import (
        LLM_BACKEND,
        MODEL_MAPPINGS,
        MODEL_TEMPERATURE,
        MODEL_TIMEOUT,
        OLLAMA_HOST,
        STRUCTURED_OUTPUT_OPTIONS,
        VLLM_API_KEY,
        VLLM_HOST,
        VLLM_MAX_TOKENS,
    )

    backend = backend or LLM_BACKEND
    temperature = temperature if temperature is not None else MODEL_TEMPERATURE

    if backend == "openrouter":
        from .config import (
            MODEL_NAME,
            OPENROUTER_API_KEY,
            OPENROUTER_MAX_TOKENS,
            OPENROUTER_MODEL_NAME,
        )

        # When using OpenRouter, ignore the default Ollama model name
        # and use the OpenRouter-specific model configuration
        if model_name == MODEL_NAME:
            # This is the default Ollama model, use OpenRouter's configured model instead
            openrouter_model = OPENROUTER_MODEL_NAME
        else:
            # User explicitly specified a model, use it
            openrouter_model = model_name or OPENROUTER_MODEL_NAME

        return OpenRouterProvider(
            model_name=openrouter_model,
            temperature=temperature,
            api_key=kwargs.get("api_key", OPENROUTER_API_KEY),
            max_tokens=kwargs.get("max_tokens", OPENROUTER_MAX_TOKENS),
            timeout=kwargs.get("timeout", MODEL_TIMEOUT),
        )
    elif backend == "vllm":
        # Map model name for vLLM
        if model_name is None:
            model_name = MODEL_MAPPINGS.get(MODEL_NAME, MODEL_NAME)
        else:
            # Also map explicitly provided model names
            model_name = MODEL_MAPPINGS.get(model_name, model_name)

        return VLLMProvider(
            model_name=model_name,
            temperature=temperature,
            vllm_host=kwargs.get("vllm_host", VLLM_HOST),
            api_key=kwargs.get("api_key", VLLM_API_KEY),
            max_tokens=kwargs.get("max_tokens", VLLM_MAX_TOKENS),
            timeout=kwargs.get("timeout", MODEL_TIMEOUT),
        )
    elif backend in ["openai", "gemini"]:
        from .config import (
            EXTERNAL_API_KEY,
            EXTERNAL_BASE_URL,
            EXTERNAL_MAX_TOKENS,
            EXTERNAL_MODEL_NAME,
        )

        # Use external model configuration
        external_model = model_name or EXTERNAL_MODEL_NAME

        return ExternalAPIProvider(
            model_name=external_model,
            temperature=temperature,
            api_provider=backend,
            api_key=kwargs.get("api_key", EXTERNAL_API_KEY),
            base_url=kwargs.get("base_url", EXTERNAL_BASE_URL),
            max_tokens=kwargs.get("max_tokens", EXTERNAL_MAX_TOKENS),
            timeout=kwargs.get("timeout", MODEL_TIMEOUT),
        )
    else:  # Default to Ollama
        model_name = model_name or MODEL_NAME

        return OllamaProvider(
            model_name=model_name,
            temperature=temperature,
            ollama_host=kwargs.get("ollama_host", OLLAMA_HOST),
            structured_output_options=kwargs.get(
                "structured_output_options", STRUCTURED_OUTPUT_OPTIONS
            ),
            timeout=kwargs.get("timeout", MODEL_TIMEOUT),
        )
