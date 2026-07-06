"""Mission debrief: turns a finished/stopped mission's own deterministic log
into durable knowledge — memories, an optional target entity page, and (only
when a technique was genuinely confirmed working) a reusable skill — via one
LLM call.

This exists because relying on the model's own initiative to call
remember/write_file/write_skill mid-mission has, in practice, never once
happened across real runs (300+ tool calls, zero such calls). The debrief
makes persistence deterministic instead of optional, the same way
BatchSummarizer makes session compaction deterministic rather than relying
on the model choosing to summarize itself.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from mist.llm.client import LLMClient, parse_json_relaxed
from mist.memory.store import MemoryStore

DEBRIEF_SCHEMA = {
    "type": "object",
    "properties": {
        "memories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"content": {"type": "string"}},
                "required": ["content"],
                "additionalProperties": False,
            },
        },
        "entity_page": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "body": {"type": "string"},
            },
            "additionalProperties": False,
        },
        "skill": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "body": {"type": "string"},
            },
            "additionalProperties": False,
        },
    },
    "required": ["memories"],
    "additionalProperties": False,
}

DEBRIEF_SYSTEM = """You just finished or stopped an autonomous pentest mission. Nothing during the
mission was reliably persisted — this is the only chance its findings survive past this session.
Compress what actually happened, given the objective, final status, and a transcript of the tool
calls/results made, into:

- `memories`: short, durable, one-sentence facts worth recalling later — open ports/service
  versions, credentials found, confirmed dead ends, the engagement's current state. Always
  include at least the target's basic recon findings if any tools were run.
- `entity_page` (omit if nothing substantial): a structured wiki page about the specific target —
  `title` (the target's name/IP) and `body` (services found, what was tried, what worked, what
  didn't, current status) in markdown.
- `skill` (omit unless a technique was genuinely CONFIRMED working — a shell obtained, privileges
  escalated, a flag read, a credential cracked, a filter bypassed): a reusable, generalized
  writeup with `name`, `description` (keyword-rich), and `body` (exact reproducible steps). Never
  invent a success that didn't happen — omit this field entirely if nothing was confirmed.

Reply with a single JSON object matching the schema. If truly nothing happened (no tools were ever
run), reply with empty memories and omit the other fields."""


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


@dataclass
class DebriefResult:
    memories_written: int = 0
    entity_page: str | None = None
    skill_written: str | None = None

    def summary(self) -> str:
        parts = [f"{self.memories_written} memor{'y' if self.memories_written == 1 else 'ies'}"]
        if self.entity_page:
            parts.append(f"wrote {self.entity_page}")
        if self.skill_written:
            parts.append("wrote a new skill")
        return "Debrief: " + ", ".join(parts)


class MissionDebriefer:
    """Compresses one mission's deterministic log into memories/wiki/skills
    via a single LLM call — deterministically triggered at mission end, see
    module docstring."""

    def __init__(self, llm: LLMClient, store: MemoryStore, wiki_root: Path | str,
                 write_skill_fn: Callable[[str, str, str], str] | None = None,
                 max_transcript_chars: int = 12000):
        self.llm = llm
        self.store = store
        self.wiki_root = Path(wiki_root)
        self.write_skill_fn = write_skill_fn
        self.max_transcript_chars = max_transcript_chars

    def _transcript(self, log_path: Path) -> str:
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        if len(text) <= self.max_transcript_chars:
            return text
        # The most valuable parts of a long mission log are the start (initial
        # recon — open ports, versions) and the end (latest state, whatever
        # it was doing when the mission stopped) — a straight head-truncation
        # (like read_file's) would show neither the final outcome nor recent
        # context, only stale early output.
        head = self.max_transcript_chars // 4
        tail = self.max_transcript_chars - head
        omitted = len(text) - self.max_transcript_chars
        return f"{text[:head]}\n\n...[{omitted} chars omitted]...\n\n{text[-tail:]}"

    def debrief(self, objective: str, status: str, log_path: Path) -> DebriefResult:
        transcript = self._transcript(log_path)
        if not transcript.strip():
            return DebriefResult()

        messages = [
            {"role": "system", "content": DEBRIEF_SYSTEM},
            {"role": "user", "content": f"Objective: {objective}\nFinal status: {status}\n\n"
                                        f"Transcript:\n{transcript}"},
        ]
        try:
            raw = self.llm.complete(messages, json_schema=DEBRIEF_SCHEMA)
            parsed = parse_json_relaxed(raw)
        except Exception:
            return DebriefResult()
        if not isinstance(parsed, dict):
            return DebriefResult()

        # Everything below is best-effort against a model that may not
        # actually follow the schema (seen in practice: `memories` returned
        # as a list of plain strings instead of {"content": ...} objects).
        # This must never raise — it runs at the very end of a mission, and
        # a crash here would take the whole mission-teardown path with it.
        written = 0
        for mem in parsed.get("memories") or []:
            if isinstance(mem, str):
                content = mem.strip()
            elif isinstance(mem, dict):
                content = str(mem.get("content") or "").strip()
            else:
                continue
            if content:
                self.store.remember(content)
                written += 1

        entity_page = None
        entity = parsed.get("entity_page")
        if isinstance(entity, dict) and entity.get("title") and entity.get("body"):
            try:
                slug = _slugify(str(entity["title"]))
                if slug:
                    path = self.wiki_root / "entities" / f"{slug}.md"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if path.exists():
                        with open(path, "a", encoding="utf-8") as fh:
                            fh.write(f"\n\n## Mission update\n\n{entity['body']}\n")
                    else:
                        path.write_text(f"# {entity['title']}\n\n{entity['body']}\n",
                                       encoding="utf-8")
                    entity_page = str(path.relative_to(self.wiki_root))
            except OSError:
                entity_page = None

        skill_written = None
        skill = parsed.get("skill")
        if isinstance(skill, dict) and skill.get("name") and self.write_skill_fn is not None:
            try:
                skill_written = self.write_skill_fn(
                    str(skill["name"]), str(skill.get("description", "")),
                    str(skill.get("body", "")),
                )
            except Exception:
                skill_written = None

        return DebriefResult(memories_written=written, entity_page=entity_page,
                             skill_written=skill_written)
