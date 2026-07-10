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
import time
from typing import Any, AsyncIterator

import httpx

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def compute_num_ctx(token_budget: int, max_tokens: int, discovered_max: int | None) -> int | None:
    """Ollama's num_ctx is one shared window for both the assembled prompt
    (bounded by `token_budget`) and the model's own generation (bounded by
    `max_tokens`) — so the window this requests has to hold both. Returns
    None (meaning: don't set num_ctx at all, let the backend use its own
    default) when `discovered_max` is unknown — picking an arbitrary
    fallback number here could just as easily be wrong as not setting it,
    and this must never *raise* the effective context beyond what a real
    model supports."""
    if discovered_max is None:
        return None
    return min(token_budget + max_tokens, discovered_max)


def compute_max_tokens(configured_max_tokens: int, token_budget: int,
                       discovered_max: int | None) -> int:
    """Caps a manually-configured `generation.max_tokens` so prompt +
    generation never exceeds the model's real context window — a fixed
    max_tokens tuned for one model (e.g. 256000) can be wildly wrong for a
    smaller-context model without this. Returns `configured_max_tokens`
    unchanged when `discovered_max` is unknown (today's behavior) or when
    it already fits; never *raises* it above what's configured — this
    only tightens an unrealistic value, it doesn't second-guess a
    reasonable one."""
    if discovered_max is None:
        return configured_max_tokens
    available = max(discovered_max - token_budget, 1)
    return min(configured_max_tokens, available)


def compute_timeout(max_tokens: int, tokens_per_second: float = 65.0,
                    min_timeout: float = 120.0, overhead_seconds: float = 30.0) -> float:
    """Sizes the HTTP client timeout so it doesn't give up on a call before
    generation.max_tokens' own reasoning budget does. 65 tok/s is the same
    conservative real-hardware estimate max_tokens itself gets sized
    against (see ~/.mist/config.yaml's own comment on generation.max_tokens)
    — a call that takes longer than that genuinely IS taking longer than
    the model was budgeted for. Confirmed live: with the old hardcoded
    120s default, raising max_tokens to 40000 for a longer reasoning pass
    did nothing to prevent a real call from being aborted by httpx.
    ReadTimeout after 2 minutes, long before the ~10-minute budget that
    max_tokens was actually sized for. `overhead_seconds` covers
    connection setup/TTFB on top of pure generation time; `min_timeout`
    preserves the old default as a floor for small max_tokens configs."""
    return max(min_timeout, max_tokens / tokens_per_second + overhead_seconds)


class LLMClient:
    def __init__(self, backend: str, base_url: str, model: str, api_key: str = "",
                 temperature: float = 0.2, max_tokens: int = 1024, timeout: float = 120.0,
                 think: bool = True, keep_alive: str | None = None,
                 stream_no_content_timeout: float | None = None):
        self.backend = backend
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Ollama-only, sent as the request's top-level `keep_alive` (added to
        # each Ollama payload by `_apply_keep_alive`). Left None the server
        # uses its own ~5-min default,
        # which is shorter than a single long tool step — so a slow scan
        # unloads the model and the next call pays a full cold reload. See
        # GenerationConfig.keep_alive.
        self.keep_alive = keep_alive
        # Seconds before a streamed answer that has produced no visible content
        # (a runaway <think> that never reaches an answer) is aborted with an
        # explanatory message. None = off (wait for the full token budget). See
        # GenerationConfig.stream_no_content_timeout and `_astream_ollama`.
        self.stream_no_content_timeout = stream_no_content_timeout
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
        # Models known (learned at runtime, see _astream_ollama) not to
        # support Ollama's thinking mode at all — sending "think": true to
        # one of these isn't silently ignored, Ollama hard-rejects the whole
        # request with a 400 ("<model> does not support thinking"), which
        # broke every conversational reply for non-reasoning models like
        # qwen2.5:14b. Keyed by model name so switching models re-probes.
        self._think_unsupported: set[str] = set()
        self._client = httpx.Client(timeout=timeout)
        self._aclient = httpx.AsyncClient(timeout=timeout)
        # Ollama's context window (num_ctx) is never sent unless set here —
        # left None (the historical default), the server loads the model
        # with whatever num_ctx its Modelfile/tags default to (commonly
        # 2048-4096), which can be smaller than what Mist actually assembles
        # (context.token_budget can run into the tens of thousands) and
        # silently truncates from the front with no error. See
        # `discover_context_length`/`compute_num_ctx` — the caller (cli.py)
        # sets this explicitly once at startup after discovering the
        # model's real max context length.
        self.num_ctx: int | None = None

    def _effective_think(self) -> bool:
        return self.think and self.model not in self._think_unsupported

    def _ollama_options(self) -> dict[str, Any]:
        options: dict[str, Any] = {"temperature": self.temperature, "num_predict": self.max_tokens}
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        return options

    def _apply_keep_alive(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Adds the top-level `keep_alive` field to an Ollama request payload
        when configured. `keep_alive` is a request-level field on /api/chat,
        NOT an `options` key — a long value (or -1) keeps the model resident in
        VRAM so a long tool step doesn't trigger a cold reload on the next
        call. No-op when unset (server default) or on the vLLM backend."""
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        return payload

    def discover_context_length(self, timeout: float = 5.0) -> int | None:
        """Queries Ollama's /api/show for this model's maximum context
        length (model_info's `<family>.context_length` key — the family
        prefix varies by architecture, e.g. "qwen35moe.context_length", so
        this looks for any key ending in that suffix rather than hardcoding
        one family name). Returns None on any failure (network error,
        unreachable server, a backend/model that doesn't expose this) —
        this is a best-effort enhancement, and every caller must have a
        sane fallback (today's behavior: don't set num_ctx at all).

        Uses a short, explicit `timeout` independent of `self._client`'s
        general one (which defaults to 120s, sized for slow LLM
        completions) — this is a lightweight metadata query called once at
        startup, before anything is rendered. Confirmed live: with the
        client's full timeout, an unreachable server made `mist tui` look
        hung for a full 2 minutes before ever showing a prompt."""
        if self.backend != "ollama":
            return None
        try:
            resp = self._client.post(f"{self.base_url}/api/show", json={"model": self.model},
                                     timeout=timeout)
            resp.raise_for_status()
            model_info = resp.json().get("model_info") or {}
        except Exception:
            return None
        for key, value in model_info.items():
            if key.endswith(".context_length") and isinstance(value, int):
                return value
        return None

    # ------------------------------------------------------------------
    def complete(self, messages: list[dict[str, str]],
                 json_schema: dict[str, Any] | None = None,
                 force_think: bool = False) -> str:
        if self.backend == "ollama":
            return self._ollama(messages, json_schema, force_think)
        if self.backend == "vllm":
            return self._vllm(messages, json_schema)
        raise ValueError(f"Unknown backend: {self.backend}")

    # ------------------------------------------------------------------
    def _ollama(self, messages, json_schema, force_think: bool = False) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            # Schema-constrained calls force reasoning off by default —
            # Ollama returns thinking in its own "thinking" field (separate
            # from "content", confirmed by inspecting the raw wire format),
            # so it doesn't corrupt JSON output the way an inline <think>
            # tag would; the real risk is a small max_tokens budget being
            # exhausted by reasoning before any content is produced. Still
            # off by default for routine decisions (cheap, fast, no need to
            # deliberate over "which tool"); `force_think` opts a specific
            # call back in for exactly the moments non-reasoning
            # decision-making has demonstrably failed (stuck-loop recovery).
            "think": self._effective_think() if (force_think or json_schema is None) else False,
            "options": self._ollama_options(),
        }
        if json_schema is not None:
            payload["format"] = json_schema  # constrained decoding
        resp = self._client.post(f"{self.base_url}/api/chat", json=self._apply_keep_alive(payload))
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
                         json_schema: dict[str, Any] | None = None,
                         force_think: bool = False) -> str:
        if self.backend == "ollama":
            return await self._aollama(messages, json_schema, force_think)
        if self.backend == "vllm":
            return await self._avllm(messages, json_schema)
        raise ValueError(f"Unknown backend: {self.backend}")

    async def _aollama(self, messages, json_schema, force_think: bool = False) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": self._effective_think() if (force_think or json_schema is None) else False,
            "options": self._ollama_options(),
        }
        if json_schema is not None:
            payload["format"] = json_schema
        resp = await self._aclient.post(f"{self.base_url}/api/chat",
                                        json=self._apply_keep_alive(payload))
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
        think = self._effective_think()
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "think": think,
            "options": self._ollama_options(),
        }
        async with self._aclient.stream(
            "POST", f"{self.base_url}/api/chat", json=self._apply_keep_alive(payload)
        ) as resp:
            if think and resp.status_code == 400:
                body = await resp.aread()
                if b"does not support thinking" in body:
                    # Not every model has a thinking mode at all — asking for
                    # one isn't ignored, Ollama hard-rejects the request.
                    # Remember it so later turns with this model skip
                    # straight to think=False instead of failing every time.
                    self._think_unsupported.add(self.model)
                    async for chunk in self._astream_ollama(messages):
                        yield chunk
                    return
            resp.raise_for_status()
            # Ollama's native thinking mode puts reasoning tokens in their
            # own "thinking" field, entirely separate from "content" — not
            # inline <think> tags (that's a different, older/other-backend
            # convention, still handled defensively by _filter_thinking).
            # "content" stays empty for the whole reasoning phase and only
            # starts once the model moves on to its actual answer, so if
            # generation.max_tokens is too small for a big/broad request the
            # model can exhaust its whole budget reasoning and never reach
            # "content" at all — silently returning nothing left the operator
            # with a bare "(empty response)" and no clue why.
            saw_thinking = False
            saw_content = False
            aborted_runaway = False
            start = time.monotonic()
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                message = obj.get("message", {})
                if message.get("thinking"):
                    saw_thinking = True
                content = message.get("content", "")
                if content:
                    saw_content = True
                    yield content
                if obj.get("done"):
                    break
                # Runaway-<think> guard: a reasoning model can spend its whole
                # (large) max_tokens budget thinking and never reach "content",
                # which at minutes-of-reasoning budgets is a multi-minute stall
                # producing nothing. If no visible content has appeared within
                # the configured window, stop reading (which cancels the
                # request) rather than waiting out the full budget. Only bites
                # while still content-less, so it never truncates a real answer
                # that has already started streaming.
                if (self.stream_no_content_timeout is not None and not saw_content
                        and (time.monotonic() - start) > self.stream_no_content_timeout):
                    aborted_runaway = True
                    break
            if aborted_runaway:
                yield (f"[Mist: no visible answer after {self.stream_no_content_timeout:.0f}s of "
                      "\"thinking\" — aborted a runaway reasoning pass. Lower generation."
                      "max_tokens, set generation.think: false, or raise generation."
                      "stream_no_content_timeout if it genuinely needs longer.]")
            elif saw_thinking and not saw_content:
                yield ("[Mist: the model ran out of tokens while still \"thinking\" and "
                      "never produced a visible answer. Try raising generation.max_tokens, "
                      "or set generation.think: false for a more concise reply.]")

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
    yielded_anything = False
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
                    yielded_anything = True
                    yield buf[:safe_len]
                buf = buf[safe_len:]
                break
            if idx:
                yielded_anything = True
                yield buf[:idx]
            buf = buf[idx + len(_THINK_OPEN):]
            in_think = True
    if buf and not in_think:
        yielded_anything = True
        yield buf
    if in_think and not yielded_anything:
        # The stream ended while still inside an unterminated <think> block —
        # the model spent its entire token budget reasoning and never got to
        # a visible answer. Silently returning nothing here left the operator
        # with a bare "(empty response)" and no clue why; a max_tokens bump
        # (or generation.think: false) is the actual fix, so say so.
        yield ("[Mist: the model ran out of tokens while still \"thinking\" and never "
              "produced a visible answer. Try raising generation.max_tokens, or set "
              "generation.think: false for a more concise reply.]")


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
