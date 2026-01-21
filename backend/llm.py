# llm.py
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


def ask_llm(
    system_prompt: str,
    user_prompt: str,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    temperature: float = 0.4,
    max_tokens: int = 400,
) -> str:
    """
    LLM helper used by app.py.

    Requirements:
      - app.py should load .env before importing this module:
          from dotenv import load_dotenv
          load_dotenv()

      - .env contains:
          OPENAI_API_KEY=...

    Usage:
      answer = ask_llm(system_prompt, user_prompt, provider="openai")
    """
    provider = (provider or "openai").lower().strip()
    if provider != "openai":
        raise ValueError("This project is configured for provider='openai' only.")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not found. Add it to your .env file and ensure app.py calls load_dotenv() before imports."
        )

    client = OpenAI(api_key=api_key)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    content = resp.choices[0].message.content
    return (content or "").strip()
