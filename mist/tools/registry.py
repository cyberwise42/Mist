"""Tool registry.

Tools are plain functions with a name, description, and JSON-schema for
arguments. The agent exposes at most ``max_exposed`` relevant tools per call
and constrains the model's output to a single-action schema built from them.
"""
from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from mist.core.subagent import format_results, run_subagents
from mist.llm.client import LLMClient


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]          # JSON schema for arguments
    fn: Callable[..., str]
    keywords: set[str] = field(default_factory=set)

    def run(self, **kwargs) -> str:
        return self.fn(**kwargs)


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def select(self, query: str, max_exposed: int = 5) -> list[Tool]:
        """Keyword-overlap ranking, same cheap strategy as skill routing.
        Tools with no overlap still qualify via generic fallback ordering so
        the model always has *something* — but never more than max_exposed."""
        q = set(re.findall(r"[a-z0-9]+", query.lower()))
        scored = sorted(
            self._tools.values(),
            key=lambda t: -len(q & (t.keywords | set(re.findall(r"[a-z0-9]+", t.description.lower())))),
        )
        return scored[:max_exposed]

    # ------------------------------------------------------------------
    def action_schema(self, tools: list[Tool]) -> dict[str, Any]:
        """Single-action schema: the model must pick exactly one of
        respond / use_tool. Constrained decoding makes this reliable even on
        3B models."""
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["respond", "use_tool"]},
                "tool": {"type": "string", "enum": [t.name for t in tools] or ["none"]},
                "arguments": {"type": "object"},
                "response": {"type": "string"},
            },
            "required": ["action"],
            "additionalProperties": False,
        }

    def decision_schema(self, tools: list[Tool]) -> dict[str, Any]:
        """Like action_schema, but with no `response` field: used by the
        streaming agent loop, which decides respond-vs-tool in one small
        (non-streamed) call and only generates the actual answer text — as a
        second, unconstrained, streamed call — once "respond" is decided.
        Keeps token-by-token streaming free of JSON wrapper syntax."""
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["respond", "use_tool"]},
                "tool": {"type": "string", "enum": [t.name for t in tools] or ["none"]},
                "arguments": {"type": "object"},
            },
            "required": ["action"],
            "additionalProperties": False,
        }


# ---------------------------------------------------------------------------
# Built-in tools
# ---------------------------------------------------------------------------

def _read_file(path: str) -> str:
    p = Path(path).expanduser()
    if not p.is_file():
        return f"ERROR: not a file: {path}"
    text = p.read_text(encoding="utf-8", errors="replace")
    return text[:8000]


def _write_file(path: str, content: str) -> str:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} chars to {p}"


def _shell(command: str) -> str:
    try:
        out = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=60
        )
        return (out.stdout + out.stderr)[:4000] or "(no output)"
    except subprocess.TimeoutExpired:
        return "ERROR: command timed out after 60s"


def default_registry(remember_fn: Callable[[str], Any] | None = None,
                      llm: LLMClient | None = None,
                      max_subagent_workers: int = 4) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(Tool(
        name="read_file",
        description="Read a text file from disk",
        parameters={"type": "object", "properties": {"path": {"type": "string"}},
                    "required": ["path"]},
        fn=_read_file,
        keywords={"read", "file", "open", "cat", "show"},
    ))
    reg.register(Tool(
        name="write_file",
        description="Write content to a file on disk",
        parameters={"type": "object",
                    "properties": {"path": {"type": "string"},
                                   "content": {"type": "string"}},
                    "required": ["path", "content"]},
        fn=_write_file,
        keywords={"write", "save", "create", "file"},
    ))
    reg.register(Tool(
        name="shell",
        description="Run a shell command and return its output",
        parameters={"type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"]},
        fn=_shell,
        keywords={"run", "shell", "command", "execute", "ls", "git", "install"},
    ))
    if remember_fn is not None:
        reg.register(Tool(
            name="remember",
            description="Persist an important fact or preference to long-term memory",
            parameters={"type": "object",
                        "properties": {"content": {"type": "string"}},
                        "required": ["content"]},
            fn=lambda content: (remember_fn(content), f"Remembered: {content}")[1],
            keywords={"remember", "memory", "note", "save", "preference"},
        ))
    if llm is not None:
        def _spawn_subagents(tasks: list[str]) -> str:
            start = time.monotonic()
            results = run_subagents(llm, tasks, max_workers=max_subagent_workers)
            return format_results(results, time.monotonic() - start)

        reg.register(Tool(
            name="spawn_subagents",
            description=(
                "Run multiple independent subtasks concurrently as isolated subagents "
                "(no shared context between them). Most effective against a vLLM "
                "backend, whose continuous batching processes concurrent requests "
                "together for near-parallel throughput."
            ),
            parameters={"type": "object",
                        "properties": {"tasks": {"type": "array",
                                                  "items": {"type": "string"}}},
                        "required": ["tasks"]},
            fn=_spawn_subagents,
            keywords={"parallel", "subagent", "subagents", "spawn", "batch", "concurrent"},
        ))
    return reg
