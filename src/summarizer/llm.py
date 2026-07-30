"""Provider-agnostic LLM client for OpenAI-compatible chat completions APIs.

Works with any service that speaks the OpenAI ``/v1/chat/completions`` protocol:
OpenAI, DeepSeek, Groq, Together, Ollama, LM Studio, Anthropic (via proxy), etc.

Configure via environment variables::

    SUMMARIZER_API_URL  — base URL (default: https://api.openai.com/v1)
    SUMMARIZER_API_KEY  — bearer token (required)
    SUMMARIZER_MODEL    — model name (default: gpt-4o-mini)

Or pass parameters directly to ``call()``::

    from summarizer.llm import call

    result = call(
        system_prompt="You are a summarizer.",
        user_prompt="Summarize this text...",
        api_url="http://localhost:11434/v1",
        api_key="ollama",
        model="llama3.2",
    )
"""

import json
import os
import re
import sys
from datetime import datetime, timezone
from ipaddress import ip_address
from urllib.parse import urlparse

import requests


_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _validate_api_url(api_url: str) -> None:
    """Validate *api_url* to prevent SSRF.

    - HTTPS is required for non-loopback hosts.
    - HTTP is allowed for localhost / loopback (Ollama, LM Studio).
    - Raw IP addresses are rejected unless they are loopback.

    Raises ``ValueError`` on invalid URLs.
    """
    parsed = urlparse(api_url)

    if not parsed.hostname:
        raise ValueError(f"Invalid API URL: no hostname found in {api_url!r}")

    hostname = parsed.hostname.lower()
    is_loopback = hostname in _LOOPBACK_HOSTS

    # Check if it's an IP address — if so, it must be loopback
    try:
        addr = ip_address(hostname)
    except ValueError:
        addr = None  # not an IP — hostname, fine

    if addr is not None and not addr.is_loopback:
        raise ValueError(
            f"IP addresses are not allowed as API hosts (got {hostname!r}). "
            f"Use a hostname or localhost."
        )
    if addr is not None and addr.is_loopback:
        is_loopback = True

    if parsed.scheme == "http" and not is_loopback:
        raise ValueError(
            f"HTTP is only allowed for loopback hosts (got {parsed.scheme!r} "
            f"for {hostname!r}). Use HTTPS or localhost."
        )
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"Unsupported scheme {parsed.scheme!r}. Use https:// or http:// "
            f"(the latter only for localhost)."
        )


def _chat_url(api_url: str) -> str:
    """Build the chat completions endpoint URL."""
    return api_url.rstrip("/") + "/chat/completions"


def _redact(text: str, secrets: list[str]) -> str:
    """Remove secrets from *text* for safe logging."""
    for s in secrets:
        if s:
            text = text.replace(s, "[REDACTED]")
    return text


def call(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 800,
    *,
    api_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> dict | None:
    """Call the configured LLM and return a parsed JSON response dict.

    Parameters:
        system_prompt: System message for the LLM.
        user_prompt: User message (the main content to summarize).
        max_tokens: Maximum tokens in the response.
        api_url: Base URL for the API. Defaults to ``SUMMARIZER_API_URL`` env
            var, then ``https://api.openai.com/v1``.
        api_key: Bearer token. Defaults to ``SUMMARIZER_API_KEY`` env var.
        model: Model name. Defaults to ``SUMMARIZER_MODEL`` env var,
            then ``gpt-4o-mini``.

    Returns:
        Parsed JSON dict on success, or ``None`` if the API key is missing
        or the call fails. On success, the dict includes metadata fields:
        ``generated_at``, ``model``, ``tokens_in``, ``tokens_out``.

    The LLM is expected to return a single JSON object (possibly wrapped in
    code fences, which are stripped).
    """
    if api_url is None:
        api_url = os.environ.get(
            "SUMMARIZER_API_URL", "https://api.openai.com/v1"
        )
    if api_key is None:
        api_key = os.environ.get("SUMMARIZER_API_KEY", "")
    if model is None:
        model = os.environ.get("SUMMARIZER_MODEL", "gpt-4o-mini")

    if not api_key:
        print("ERROR: SUMMARIZER_API_KEY is not set or is empty", file=sys.stderr)
        return None

    try:
        _validate_api_url(api_url)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return None

    try:
        resp = requests.post(
            _chat_url(api_url),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
                "max_tokens": max_tokens,
            },
            timeout=30,
        )
        resp.raise_for_status()
        body = resp.json()

        choice = body["choices"][0]
        content = choice["message"]["content"].strip()
        usage = body.get("usage", {})

        # Strip code fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            if len(lines) >= 3:
                content = "\n".join(lines[1:-1]).strip()

        result = json.loads(content)

        # Stamp metadata
        result["generated_at"] = datetime.now(timezone.utc).isoformat()
        result["model"] = model
        result["tokens_in"] = usage.get("prompt_tokens", 0)
        result["tokens_out"] = usage.get("completion_tokens", 0)

        return result

    except (requests.RequestException, json.JSONDecodeError, KeyError,
            IndexError) as e:
        safe = _redact(str(e), [api_key, f"Bearer {api_key}"])
        print(f"ERROR: LLM call failed: {safe}", file=sys.stderr)
        return None
