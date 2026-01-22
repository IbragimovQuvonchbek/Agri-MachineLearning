import os
import requests


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def ask_llm(
    system_prompt: str,
    user_prompt: str,
    provider: str = "openrouter",
    model: str | None = None,
    temperature: float = 0.4,
    max_tokens: int = 400,
) -> str:
    """
    OpenRouter LLM helper (OpenAI-compatible API).

    Requires .env:
      OPENROUTER_API_KEY=sk-or-v1-...
      OPENROUTER_MODEL=openai/gpt-4o-mini   (optional)
      OPENROUTER_SITE_URL=http://127.0.0.1:8000 (optional)
      OPENROUTER_APP_NAME=VegQualityAI      (optional)
    """
    provider = (provider or "openrouter").lower().strip()
    if provider not in ("openrouter", "or"):
        raise ValueError("This project is configured for provider='openrouter' only.")

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set. Put it into your .env file.")

    chosen_model = model or os.getenv("OPENROUTER_MODEL") or "openai/gpt-4o-mini"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # recommended by OpenRouter:
        "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
        "X-Title": os.getenv("OPENROUTER_APP_NAME", "VegQualityAI"),
    }

    payload = {
        "model": chosen_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=45)

    # Better error messages:
    if r.status_code >= 400:
        try:
            err = r.json()
        except Exception:
            err = {"error": r.text}
        raise RuntimeError(f"OpenRouter API error {r.status_code}: {err}")

    data = r.json()
    try:
        return (data["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        raise RuntimeError(f"Unexpected OpenRouter response: {data}")
