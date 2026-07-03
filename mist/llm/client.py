"""LLM backend clients.

Both backends expose the same interface:

    complete(messages, json_schema=None) -> str

When ``json_schema`` is provided the backend is asked to *constrain* output:
- Ollama: ``format`` field (full JSON-schema constrained decoding, Ollama >= 0.5)
- vLLM:   OpenAI-compatible ``response_format`` -> ``guided_json`` extra body

This is the key reliability trick for small models: never trust free-text
tool-call parsing.
"""
from __future__ import annotations

import json
from typing import Any

import httpx


class LLMClient:
    def __init__(self, backend: str, base_url: str, model: str, api_key: str = "",
                 temperature: float = 0.2, max_tokens: int = 1024, timeout: float = 120.0):
        self.backend = backend
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = httpx.Client(timeout=timeout)

    # ------------------------------------------------------------------
    def complete(self, messages: list[dict[str, str]],
                 json_schema: dict[str, Any] | None = None) -> str:
        if self.backend == "ollama":
            return self._ollama(messages, json_schema)
        if self.backend == "vllm":
            return self._vllm(messages, json_schema)
        raise ValueError(f"Unknown backend: {self.backend}")

    # ------------------------------------------------------------------
    def _ollama(self, messages, json_schema) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        if json_schema is not None:
            payload["format"] = json_schema  # constrained decoding
        resp = self._client.post(f"{self.base_url}/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    def _vllm(self, messages, json_schema) -> str:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if json_schema is not None:
            # vLLM's OpenAI server supports guided decoding via extra body.
            payload["extra_body"] = {"guided_json": json_schema}
            # Newer vLLM also accepts response_format json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "mist_action", "schema": json_schema},
            }
        resp = self._client.post(
            f"{self.base_url}/chat/completions", json=payload, headers=headers
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


def parse_json_relaxed(text: str) -> dict[str, Any]:
    """Parse JSON, tolerating markdown fences (small models sometimes add them
    even under constrained decoding when the constraint isn't supported)."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in model output: {text[:200]!r}")
    return json.loads(text[start:end + 1])
