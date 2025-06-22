# llm.py
import requests

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3:8b"  # You can change this later

def query_ollama(prompt: str, model: str = MODEL_NAME) -> str:
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False  # Set to True if you want streamed responses
            },
            timeout=60  # Adjust if prompts are long
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except requests.RequestException as e:
        return f"LLM API Error: {str(e)}"
