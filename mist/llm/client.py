"""LLM backend clients.

Both backends expose the same synchronous interface:

    complete(messages, json_schema=None) -> str

When ``json_schema`` is provided the backend is asked to *constrain* output:
- Ollama: ``format`` field (full JSON-schema constrained decoding, Ollama >= 0.5)
- vLLM:   OpenAI-compatible ``response_format`` -> ``guided_json`` extra body

This is the key reliability trick for small models: never trust free-text
tool-call parsing.

For the streaming TUI (see mist.tui), an async, streaming counterpart is also
available:

    await acomplete(messages, json_schema=None) -> str      # non-streamed, async
    async for chunk in astream(messages): ...                # unconstrained, streamed

``astream`` is deliberately never given a json_schema — streaming a JSON
object token-by-token would surface raw JSON syntax to the user. The agent
loop instead decides respond-vs-tool with a small constrained ``acomplete``
call, then only streams free text once "respond" is chosen.
"""
from __future__ import annotations

import json
import re
from typing import Any, AsyncIterator

import httpx

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


class LLMClient:
    def __init__(self, backend: str, base_url: str, model: str, api_key: str = "",
                 temperature: float = 0.2, max_tokens: int = 1024, timeout: float = 120.0,
                 think: bool = True):
        self.backend = backend
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Reasoning-capable models (Qwen3, DeepSeek-R1, ...) emit a <think>
        # block. It's forced off (below) only on schema-constrained calls,
        # where valid JSON is a hard requirement — reasoning there buys little
        # (it's a routing choice: which tool, what args) and risks corrupting
        # the output. The free-text answer step keeps `think` on by default:
        # that's where the model actually reasons over tool output, memory,
        # and wiki context to do the work, and it should use its full
        # capability there. Any <think> block is filtered out of the
        # *displayed* stream (see `_filter_thinking`) without stopping the
        # model from generating it — this hides the trace, it doesn't
        # suppress the thinking itself.
        self.think = think
        self._client = httpx.Client(timeout=timeout)
        self._aclient = httpx.AsyncClient(timeout=timeout)

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
            # Schema-constrained calls need guaranteed-clean JSON; reasoning
            # is forced off for those regardless of the `think` setting.
            "think": False if json_schema is not None else self.think,
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

    # ------------------------------------------------------------------
    def list_models(self) -> list[str]:
        """Queries the backend for the models it currently has available —
        `ollama list`-equivalent for the ollama backend, or the OpenAI-style
        `/models` listing for vLLM — so an operator switching between many
        pulled models can see what's actually servable before picking one."""
        if self.backend == "ollama":
            resp = self._client.get(f"{self.base_url}/api/tags")
            resp.raise_for_status()
            return sorted(m["name"] for m in resp.json().get("models", []))
        if self.backend == "vllm":
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            resp = self._client.get(f"{self.base_url}/models", headers=headers)
            resp.raise_for_status()
            return sorted(m["id"] for m in resp.json().get("data", []))
        raise ValueError(f"Unknown backend: {self.backend}")

    # ------------------------------------------------------------------
    # Async / streaming (used by the TUI's live turn loop)
    # ------------------------------------------------------------------
    async def acomplete(self, messages: list[dict[str, str]],
                         json_schema: dict[str, Any] | None = None) -> str:
        if self.backend == "ollama":
            return await self._aollama(messages, json_schema)
        if self.backend == "vllm":
            return await self._avllm(messages, json_schema)
        raise ValueError(f"Unknown backend: {self.backend}")

    async def _aollama(self, messages, json_schema) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False if json_schema is not None else self.think,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
        }
        if json_schema is not None:
            payload["format"] = json_schema
        resp = await self._aclient.post(f"{self.base_url}/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    async def _avllm(self, messages, json_schema) -> str:
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
            payload["extra_body"] = {"guided_json": json_schema}
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "mist_action", "schema": json_schema},
            }
        resp = await self._aclient.post(
            f"{self.base_url}/chat/completions", json=payload, headers=headers
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    async def astream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        """Unconstrained streaming completion: yields text deltas as they
        arrive. Never pass json_schema here (see module docstring). Any
        <think>...</think> span is filtered out of the yielded deltas (see
        `_filter_thinking`) — the model still reasons if `think` is on, this
        just keeps the trace out of what gets displayed."""
        if self.backend == "ollama":
            source = self._astream_ollama(messages)
        elif self.backend == "vllm":
            source = self._astream_vllm(messages)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
        async for chunk in _filter_thinking(source):
            yield chunk

    async def _astream_ollama(self, messages) -> AsyncIterator[str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "think": self.think,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
        }
        async with self._aclient.stream(
            "POST", f"{self.base_url}/api/chat", json=payload
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                content = obj.get("message", {}).get("content", "")
                if content:
                    yield content
                if obj.get("done"):
                    break

    async def _astream_vllm(self, messages) -> AsyncIterator[str]:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        async with self._aclient.stream(
            "POST", f"{self.base_url}/chat/completions", json=payload, headers=headers
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                content = obj["choices"][0].get("delta", {}).get("content", "")
                if content:
                    yield content

    async def aclose(self) -> None:
        await self._aclient.aclose()


async def _filter_thinking(source: AsyncIterator[str]) -> AsyncIterator[str]:
    """Suppresses <think>...</think> spans from a live token stream.

    The model still generates its reasoning (nothing here changes what's
    requested of the model or how much it "thinks") — this only keeps that
    trace out of what gets displayed. A tag can arrive split across chunk
    boundaries, so a partial match at the tail of the buffer is held back
    rather than emitted until it's known not to be the start of a tag.
    """
    buf = ""
    in_think = False
    async for chunk in source:
        buf += chunk
        while True:
            if in_think:
                idx = buf.find(_THINK_CLOSE)
                if idx == -1:
                    break  # still inside the block; keep buffering, emit nothing
                buf = buf[idx + len(_THINK_CLOSE):]
                in_think = False
                continue
            idx = buf.find(_THINK_OPEN)
            if idx == -1:
                safe_len = max(len(buf) - (len(_THINK_OPEN) - 1), 0)
                if safe_len:
                    yield buf[:safe_len]
                buf = buf[safe_len:]
                break
            if idx:
                yield buf[:idx]
            buf = buf[idx + len(_THINK_OPEN):]
            in_think = True
    if buf and not in_think:
        yield buf


def parse_json_relaxed(text: str) -> dict[str, Any]:
    """Parse JSON, tolerating markdown fences and stray <think>...</think>
    reasoning blocks (some models emit these even under constrained decoding,
    or ignore the `think: false` request entirely)."""
    text = _THINK_BLOCK_RE.sub("", text).strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in model output: {text[:200]!r}")
    return json.loads(text[start:end + 1])
