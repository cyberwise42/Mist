"""Summarizes conversation history that would otherwise be silently
dropped when a turn's assembled prompt exceeds context.token_budget (see
_fit_budget in mist/core/agent.py) — folds it into one compact recap
message instead of the model losing all trace of it. Uses the same
independently-configured aux model as tier 3 tool-output compression
(mist/core/tool_compressor.py, config.SummarizerLLMConfig) — a distinct
toggle (context.compress_on_overflow), sharing the one aux-model
connection.
"""
from __future__ import annotations

from mist.llm.client import LLMClient

_SYSTEM = ("Summarize the following conversation excerpt in 2-4 sentences. "
          "Preserve concrete facts, decisions, and findings (IPs, versions, "
          "filenames, results) — drop pleasantries and filler. Be terse.")


class ContextCompressor:
    def __init__(self, llm: LLMClient, max_input_chars: int = 8000):
        self.llm = llm
        self.max_input_chars = max_input_chars

    def summarize(self, messages: list[dict[str, str]]) -> str:
        """Best-effort: falls back to a naive truncated concatenation of
        the original messages on any failure (network error, empty
        response) — the caller must end up with SOME trace of what was
        dropped, never nothing."""
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)[:self.max_input_chars]
        try:
            summary = self.llm.complete(
                [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": text}]
            ).strip()
            if summary:
                return summary
        except Exception:
            pass
        return text[:400]
