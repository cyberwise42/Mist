"""Tier 3 of Mist's tool-output handling: auxiliary-model compression for
the long tail tiers 1-2 don't already cover — ad hoc `curl` requests,
custom exploit scripts, `hashcat` runs, anything `mist.tools.structured`
doesn't recognize. Lowest priority of the three tiers: real evidence shows
these calls are also the least likely to overflow the budget in the first
place, so in practice this rarely fires once tiers 1-2 are in place.

Mirrors the only existing precedent for "a second, independently-configured
model" — `EmbeddingConfig`/`EmbeddingClient` (mist/llm/embeddings.py) — and
`MissionDebriefer`'s (mist/core/debrief.py) shape for a single schema-
constrained LLM call. Critical difference from `MissionDebriefer`'s
failure handling (which returns an empty `DebriefResult` on failure): on
ANY failure here, fall back to the ORIGINAL raw text, never to an empty or
short result — silently dropping data here would recreate the exact
data-loss bug this whole design exists to fix.
"""
from __future__ import annotations

from mist.llm.client import LLMClient, parse_json_relaxed

COMPRESS_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
    "additionalProperties": False,
}

COMPRESS_SYSTEM = """Compress the following tool output for a penetration-testing agent that will
act on it next. Keep every concrete fact that could matter — findings, errors, exact values
(ports, paths, versions, credentials, error messages) — and drop repeated/boilerplate noise
(progress bars, banners, padding). Do not invent anything not present in the input. Reply with a
single JSON object: {"summary": "<compressed text>"}."""


class ToolOutputCompressor:
    def __init__(self, llm: LLMClient, trigger_chars: int = 6000, max_input_chars: int = 20000):
        self.llm = llm
        self.trigger_chars = trigger_chars
        self.max_input_chars = max_input_chars

    def maybe_compress(self, text: str) -> str:
        """Returns `text` unchanged if it's under `trigger_chars` — the
        common case once tiers 1-2 already apply. Otherwise asks the aux
        model to compress it; on any failure (network error, malformed
        JSON, an empty summary field), falls back to the original `text`
        rather than a shortened default."""
        if len(text) <= self.trigger_chars:
            return text
        clipped = text[:self.max_input_chars]
        messages = [
            {"role": "system", "content": COMPRESS_SYSTEM},
            {"role": "user", "content": clipped},
        ]
        try:
            raw = self.llm.complete(messages, json_schema=COMPRESS_SCHEMA)
            parsed = parse_json_relaxed(raw)
            summary = str(parsed.get("summary", "")).strip()
        except Exception:
            return text
        return summary if summary else text
