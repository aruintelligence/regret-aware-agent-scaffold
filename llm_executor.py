"""Opt-in, local-first OpenAI-compatible chat executor for real LLM evaluation.

This adapter is not used by the scripted scaffold or synthetic benchmark.
The operator must supply a running model endpoint and explicitly consent to
sending the selected task context to that endpoint. Remote endpoints require
an additional opt-in flag and HTTPS. No prompt or model response is logged.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from urllib.parse import urlsplit


class EndpointError(RuntimeError):
    pass


def validate_endpoint(endpoint: str, *, allow_remote: bool = False) -> str:
    parsed = urlsplit(endpoint)
    if parsed.username or parsed.password or parsed.fragment or parsed.query:
        raise ValueError("endpoint must not contain credentials, query, or fragment")
    if parsed.scheme not in ("http", "https"):
        raise ValueError("only http and https endpoints supported")
    host = (parsed.hostname or "").lower()
    local = host in ("localhost", "127.0.0.1", "::1")
    if not local and not allow_remote:
        raise ValueError("remote endpoints require explicit allow_remote opt-in")
    if not local and parsed.scheme != "https":
        raise ValueError("remote endpoints must use HTTPS")
    if not parsed.path.endswith("/chat/completions"):
        raise ValueError("expected an OpenAI-compatible /chat/completions endpoint")
    if not (1 <= (parsed.port or (443 if parsed.scheme == "https" else 80)) <= 65535):
        raise ValueError("invalid port")
    return endpoint


def run_chat(messages, *, model: str, endpoint: str = "http://127.0.0.1:1234/v1/chat/completions",
             allow_remote: bool = False, api_key_env: str = "ARU_LLM_API_KEY",
             timeout_s: float = 60, max_tokens: int = 96) -> dict:
    validate_endpoint(endpoint, allow_remote=allow_remote)
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model name required")
    if not isinstance(messages, list) or not messages:
        raise ValueError("nonempty messages required")
    if any(not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant")
           or not isinstance(m.get("content"), str) for m in messages):
        raise ValueError("messages must contain text with recognized roles")
    if sum(len(m["content"]) for m in messages) > 25000:
        raise ValueError("prompt too long for this research adapter")
    if not 1 <= max_tokens <= 1024 or not 1 <= timeout_s <= 300:
        raise ValueError("max_tokens or timeout outside safe research limits")
    payload = {
        "model": model, "messages": messages,
        "temperature": 0, "max_tokens": max_tokens, "stream": False,
    }
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    key = os.environ.get(api_key_env, "")
    if key:
        headers["Authorization"] = "Bearer " + key
    request = urllib.request.Request(
        endpoint, data=json.dumps(payload).encode("utf-8"),
        headers=headers, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            if response.status != 200:
                raise EndpointError("model endpoint returned non-200 status")
            # Bound response size to avoid accidental unbounded memory use.
            raw = response.read(256000 + 1)
            if len(raw) > 256000:
                raise EndpointError("model response too large")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        raise EndpointError("model endpoint request failed; check local server and model") from exc
    try:
        result = json.loads(raw)
        content = result["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise EndpointError("invalid OpenAI-compatible response") from exc
    if not isinstance(content, str):
        raise EndpointError("model response content must be a string")
    usage = result.get("usage") or {}
    return {
        "text": content.strip()[:4096],
        "prompt_tokens": usage.get("prompt_tokens") if isinstance(usage.get("prompt_tokens"), int) else None,
        "completion_tokens": usage.get("completion_tokens") if isinstance(usage.get("completion_tokens"), int) else None,
    }
