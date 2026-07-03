"""Batch summarizer: compresses finished sessions into durable memories.

Raw turn history is cheap to accumulate and expensive to keep around forever
— exactly the kind of context bloat Mist is built to avoid elsewhere. Instead
of letting old sessions sit in `turns` unbounded, periodically fold each one
down to a handful of memory rows (facts/preferences/notes) via a single LLM
call per session, then mark it compacted so it is never re-processed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from mist.llm.client import LLMClient, parse_json_relaxed
from mist.memory.store import MemoryStore

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "memories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "kind": {"type": "string", "enum": ["fact", "preference", "note"]},
                },
                "required": ["content"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["memories"],
    "additionalProperties": False,
}

SUMMARY_SYSTEM = """You compress a finished conversation into durable long-term memories.

Extract only facts, stated preferences, and decisions that would be useful in
future unrelated conversations. Skip small talk and anything tied only to this
session's task. Write each memory as a short, self-contained sentence.

Reply with a single JSON object: {"memories": [{"content": "...", "kind": "fact|preference|note"}]}
If nothing is worth keeping, reply {"memories": []}."""


@dataclass
class CompactionResult:
    session_id: int
    memories_written: int


class BatchSummarizer:
    """Compresses old sessions into memories, one LLM call per session."""

    def __init__(self, llm: LLMClient, store: MemoryStore, max_transcript_chars: int = 8000):
        self.llm = llm
        self.store = store
        self.max_transcript_chars = max_transcript_chars

    def _transcript(self, session_id: int) -> str:
        turns = self.store.all_turns(session_id)
        text = "\n".join(f"{t['role']}: {t['content']}" for t in turns)
        return text[: self.max_transcript_chars]

    def summarize_session(self, session_id: int) -> CompactionResult:
        transcript = self._transcript(session_id)
        if not transcript.strip():
            self.store.mark_compacted(session_id)
            return CompactionResult(session_id, 0)

        messages = [
            {"role": "system", "content": SUMMARY_SYSTEM},
            {"role": "user", "content": f"Conversation:\n{transcript}"},
        ]
        raw = self.llm.complete(messages, json_schema=SUMMARY_SCHEMA)
        try:
            parsed = parse_json_relaxed(raw)
        except (ValueError, json.JSONDecodeError):
            parsed = {"memories": []}

        written = 0
        for mem in parsed.get("memories", []):
            content = (mem.get("content") or "").strip()
            if content:
                self.store.remember(content, mem.get("kind", "note"))
                written += 1

        self.store.mark_compacted(session_id)
        return CompactionResult(session_id, written)

    def compact_old_sessions(self, exclude_session_id: int | None = None,
                              keep_recent: int = 1, min_turns: int = 4) -> list[CompactionResult]:
        """Batch entry point: fold every eligible old session into memory."""
        ids = self.store.sessions_to_compact(exclude_session_id, keep_recent, min_turns)
        return [self.summarize_session(sid) for sid in ids]
