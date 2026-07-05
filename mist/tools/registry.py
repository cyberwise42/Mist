"""Tool registry.

Tools are plain functions with a name, description, and JSON-schema for
arguments. The agent exposes at most ``max_exposed`` relevant tools per call
and constrains the model's output to a single-action schema built from them.
"""
from __future__ import annotations

import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from mist.config import ShellConfig
from mist.core.subagent import format_results, run_subagents, run_tool_subagents
from mist.llm.client import LLMClient
from mist.skills.router import SkillRouter


class ProcessRegistry:
    """Tracks the current foreground subprocess (if any) so an operator kill
    can actually terminate a running command, not just stop Mist from
    awaiting it. Cancelling the asyncio task driving a turn/mission does NOT
    stop a blocking subprocess already running in a worker thread — Python
    threads can't be preempted — so the shell tool registers its live Popen
    handle here, and ``kill_active`` reaches in and terminates it directly."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def set(self, proc: subprocess.Popen | None) -> None:
        with self._lock:
            self._proc = proc

    def clear(self) -> None:
        with self._lock:
            self._proc = None

    def kill_active(self) -> bool:
        """Terminates the currently-registered process, if any is still
        running. Returns True if something was actually killed."""
        with self._lock:
            proc = self._proc
        if proc is None or proc.poll() is not None:
            return False
        try:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
        except ProcessLookupError:
            pass
        return True


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

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def keep_only(self, names: set[str]) -> None:
        self._tools = {n: t for n, t in self._tools.items() if n in names}

    def select(self, query: str, max_exposed: int = 5,
               always: set[str] | None = None) -> list[Tool]:
        """Keyword-overlap ranking, same cheap strategy as skill routing.
        Tools with no overlap still qualify via generic fallback ordering so
        the model always has *something* — but never more than max_exposed.

        ``always`` force-includes specific tools regardless of their
        ranking (e.g. `finish_objective` during a mission — it may share no
        keywords with a given turn's message, but the model must always be
        able to reach for it)."""
        always = always or set()
        q = set(re.findall(r"[a-z0-9]+", query.lower()))
        scored = sorted(
            self._tools.values(),
            key=lambda t: -len(q & (t.keywords | set(re.findall(r"[a-z0-9]+", t.description.lower())))),
        )
        forced = [t for t in scored if t.name in always]
        rest = [t for t in scored if t.name not in always]
        return forced + rest[:max(0, max_exposed - len(forced))]

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

def _resolve(path: str, default_root: Path | None) -> Path:
    """A relative path is anchored to `default_root` (e.g. the wiki root)
    when one is configured, so skill-instructed relative paths like
    `raw/foo.md` land in the intended tree rather than the process's cwd.
    An absolute path (or `~`) always wins regardless."""
    p = Path(path).expanduser()
    if not p.is_absolute() and default_root is not None:
        return default_root / p
    return p


def _make_read_file(default_root: Path | str | None) -> Callable[..., str]:
    default_base = Path(default_root).expanduser() if default_root else None

    def _read_file(path: str) -> str:
        p = _resolve(path, default_base)
        if not p.is_file():
            return f"ERROR: not a file: {path}"
        text = p.read_text(encoding="utf-8", errors="replace")
        return text[:8000]
    return _read_file


def _make_write_file(default_root: Path | str | None) -> Callable[..., str]:
    default_base = Path(default_root).expanduser() if default_root else None

    def _write_file(path: str, content: str) -> str:
        p = _resolve(path, default_base)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} chars to {p}"
    return _write_file


def _run_subprocess(args: str | list[str], shell: bool, timeout: float,
                     registry: ProcessRegistry | None) -> str:
    """Runs a command via Popen (not subprocess.run) so the live process can
    be registered for an operator kill — cancelling the asyncio task awaiting
    this (via asyncio.to_thread) does not stop a subprocess already running
    in a worker thread, since Python threads can't be preempted."""
    proc = subprocess.Popen(args, shell=shell, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    if registry is not None:
        registry.set(proc)
    try:
        try:
            out, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, _ = proc.communicate()
            return (out or "")[:4000] + f"\nERROR: command timed out after {timeout:.0f}s"
    finally:
        if registry is not None:
            registry.clear()
    if proc.returncode is not None and proc.returncode < 0:
        return (out or "")[:4000] + "\nERROR: command was killed by the operator"
    return out[:4000] if out else "(no output)"


def _shell(command: str, registry: ProcessRegistry | None = None) -> str:
    return _run_subprocess(command, shell=True, timeout=60, registry=registry)


def _make_shell(shell_cfg: ShellConfig | None,
                registry: ProcessRegistry | None = None) -> Callable[..., str]:
    if shell_cfg is None or shell_cfg.backend == "local":
        return lambda command: _shell(command, registry)

    ssh = shell_cfg.ssh

    def _shell_ssh(command: str) -> str:
        args = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
        if ssh.key_path:
            args += ["-i", str(Path(ssh.key_path).expanduser())]
        args += ["-p", str(ssh.port), f"{ssh.user}@{ssh.host}" if ssh.user else ssh.host, command]
        return _run_subprocess(args, shell=False, timeout=ssh.timeout, registry=registry)
    return _shell_ssh


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _make_search_files(default_root: Path | str | None) -> Callable[..., str]:
    default_base = Path(default_root).expanduser() if default_root else None

    def _search_files(query: str, root: str | None = None, max_results: int = 20) -> str:
        base = Path(root).expanduser() if root else default_base
        if base is None or not base.is_dir():
            return f"ERROR: not a directory: {base}"
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        hits: list[str] = []
        for md in sorted(base.rglob("*.md")):
            try:
                lines = md.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for lineno, line in enumerate(lines, 1):
                if pattern.search(line):
                    hits.append(f"{md.relative_to(base)}:{lineno}: {line.strip()[:200]}")
                    if len(hits) >= max_results:
                        return "\n".join(hits)
        return "\n".join(hits) if hits else "No matches."
    return _search_files


def _make_write_skill(skills: SkillRouter) -> Callable[..., str]:
    def _write_skill(name: str, description: str, body: str) -> str:
        slug = _slugify(name)
        if not slug:
            return f"ERROR: invalid skill name: {name!r}"
        path = skills.library_path / slug / "SKILL.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"---\nname: {slug}\ndescription: {description}\n---\n\n{body}",
                        encoding="utf-8")
        skills.reload()
        return f"Wrote skill '{slug}' to {path}"
    return _write_skill


def default_registry(remember_fn: Callable[[str], Any] | None = None,
                      llm: LLMClient | None = None,
                      max_subagent_workers: int = 4,
                      skills: SkillRouter | None = None,
                      max_subagent_steps: int = 6,
                      subagent_tools_enabled: bool = True,
                      wiki_root: str | Path | None = None,
                      enabled: list[str] | None = None,
                      shell_config: ShellConfig | None = None,
                      process_registry: ProcessRegistry | None = None) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(Tool(
        name="read_file",
        description="Read a text file from disk (relative paths resolve under the wiki root)",
        parameters={"type": "object", "properties": {"path": {"type": "string"}},
                    "required": ["path"]},
        fn=_make_read_file(wiki_root),
        keywords={"read", "file", "open", "cat", "show"},
    ))
    reg.register(Tool(
        name="write_file",
        description="Write content to a file on disk (relative paths resolve under the wiki root)",
        parameters={"type": "object",
                    "properties": {"path": {"type": "string"},
                                   "content": {"type": "string"}},
                    "required": ["path", "content"]},
        fn=_make_write_file(wiki_root),
        keywords={"write", "save", "create", "file"},
    ))
    reg.register(Tool(
        name="search_files",
        description="Full-text search for a term across markdown files under a directory",
        parameters={"type": "object",
                    "properties": {"query": {"type": "string"},
                                   "root": {"type": "string"},
                                   "max_results": {"type": "integer"}},
                    "required": ["query"]},
        fn=_make_search_files(wiki_root),
        keywords={"search", "find", "grep", "wiki", "index"},
    ))
    reg.register(Tool(
        name="shell",
        description="Run a shell command and return its output",
        parameters={"type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"]},
        fn=_make_shell(shell_config, process_registry),
        keywords={"run", "shell", "command", "execute", "ls", "git", "install"},
    ))
    reg.register(Tool(
        name="finish_objective",
        description=(
            "Call this ONLY once the current objective is fully, concretely achieved "
            "(e.g. you have the flag/root/the exact output that was asked for) — it "
            "signals that no further work is needed. Do not call this for a routine "
            "status update or partial progress; keep working instead."
        ),
        parameters={"type": "object", "properties": {"summary": {"type": "string"}},
                    "required": ["summary"]},
        fn=lambda summary: f"Objective marked complete: {summary}",
        keywords={"finish", "complete", "done", "objective", "mission", "flag", "root"},
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
    if skills is not None:
        reg.register(Tool(
            name="write_skill",
            description=(
                "Persist a validated, reusable technique as a new skill so future "
                "turns can find it. Use once a technique is confirmed to work."
            ),
            parameters={"type": "object",
                        "properties": {"name": {"type": "string"},
                                       "description": {"type": "string"},
                                       "body": {"type": "string"}},
                        "required": ["name", "description", "body"]},
            fn=_make_write_skill(skills),
            keywords={"write", "skill", "wiki", "remember", "technique", "save"},
        ))
    if llm is not None:
        def _spawn_subagents(tasks: list[str]) -> str:
            start = time.monotonic()
            if subagent_tools_enabled:
                subagent_tools = default_registry(remember_fn=remember_fn, llm=None, skills=None,
                                                  wiki_root=wiki_root, enabled=enabled,
                                                  shell_config=shell_config,
                                                  process_registry=process_registry)
                results = run_tool_subagents(llm, subagent_tools, tasks,
                                             max_workers=max_subagent_workers,
                                             max_steps=max_subagent_steps)
            else:
                results = run_subagents(llm, tasks, max_workers=max_subagent_workers)
            return format_results(results, time.monotonic() - start)

        tool_description = (
            "Run multiple independent subtasks concurrently as isolated subagents, "
            "each with its own read_file/write_file/shell/remember tool access (no "
            "shared context between them). Most effective against a vLLM backend, "
            "whose continuous batching processes concurrent requests together for "
            "near-parallel throughput."
        ) if subagent_tools_enabled else (
            "Run multiple independent subtasks concurrently as isolated subagents "
            "(no shared context between them). Most effective against a vLLM "
            "backend, whose continuous batching processes concurrent requests "
            "together for near-parallel throughput."
        )
        reg.register(Tool(
            name="spawn_subagents",
            description=tool_description,
            parameters={"type": "object",
                        "properties": {"tasks": {"type": "array",
                                                  "items": {"type": "string"}}},
                        "required": ["tasks"]},
            fn=_spawn_subagents,
            keywords={"parallel", "subagent", "subagents", "spawn", "batch", "concurrent"},
        ))
    if enabled is not None:
        reg.keep_only(set(enabled))
    return reg
