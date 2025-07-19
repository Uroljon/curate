# llm.py
import requests

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:14b"  # Qwen2.5 14B - specialized for structured output and multilingual tasks

def query_ollama(prompt: str, model: str = MODEL_NAME, temperature: float = 0.0, system_message: str = None) -> str:
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
                "temperature": 0.3,    # Faster sampling
                "top_p": 0.9,          # More vocabulary options
                "top_k": 40,           # Balanced vocabulary
                "repeat_penalty": 1.05, # Light repetition control
                "num_predict": 800,    # Shorter for speed
                "stop": ["```", "</json>"]
            }
        }
        
        # Add system message if provided
        if system_message:
            request_body["system"] = system_message
        
        response = requests.post(OLLAMA_API_URL, json=request_body,
            timeout=180
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except requests.RequestException as e:
        print(f"❌ LLM API Error: {str(e)}")
        return f"LLM API Error: {str(e)}"
