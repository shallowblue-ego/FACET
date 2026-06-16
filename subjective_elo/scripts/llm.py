"""Thin wrapper around the OpenAI-compatible API for ELO judge calls."""

import json
import os
import time
from typing import Dict, Optional, Union

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

# Defaults from environment
DEFAULT_MODEL = os.getenv("ELO_JUDGE_MODEL", "gemini-2.5-pro")
DEFAULT_TEMPERATURE = float(os.getenv("ELO_JUDGE_TEMPERATURE", "0.2"))
DEFAULT_SEED = int(os.getenv("ELO_JUDGE_SEED", "42"))
API_RETRY_COUNT = int(os.getenv("ELO_API_RETRY_COUNT", "3"))
API_RETRY_DELAY = int(os.getenv("ELO_API_RETRY_DELAY", "5"))

DEFAULT_HEADERS = {
    "User-Agent": "PostmanRuntime/7.36.0",
    "Accept": "*/*",
    "Connection": "keep-alive",
}

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    """Lazily initialise and return the shared OpenAI client.
    Requires API_KEY and API_URL environment variables."""
    global _client
    if _client is not None:
        return _client

    api_key = os.getenv("API_KEY")
    api_url = os.getenv("API_URL")
    if not api_key or not api_url:
        raise ValueError("API_KEY and API_URL must be set before calling the ELO judge API.")

    _client = OpenAI(api_key=api_key, base_url=api_url, default_headers=DEFAULT_HEADERS)
    return _client


def llm_service(
    prompt: str,
    model_name: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    seed: int = DEFAULT_SEED,
    is_json: bool = True,
) -> Optional[Union[Dict, str]]:
    """Call the judge LLM with retry logic.

    Args:
        prompt: Full prompt text.
        is_json: If True, request JSON response and parse into a dict.

    Returns a dict (is_json=True), str (is_json=False), or None on total failure."""
    kwargs = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "seed": seed,
    }
    if is_json:
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(API_RETRY_COUNT):
        try:
            response = _get_client().chat.completions.create(**kwargs)
            response_content = response.choices[0].message.content
            if is_json:
                return json.loads(response_content)
            return response_content
        except json.JSONDecodeError as exc:
            print(f"Warning: judge did not return valid JSON "
                  f"({attempt + 1}/{API_RETRY_COUNT}): {exc}")
        except Exception as exc:
            print(f"Error calling judge API "
                  f"({attempt + 1}/{API_RETRY_COUNT}): {exc}")

        if attempt < API_RETRY_COUNT - 1:
            time.sleep(API_RETRY_DELAY)

    print("Error: all judge API attempts failed.")
    return None
