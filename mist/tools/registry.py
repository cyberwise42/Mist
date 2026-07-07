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
    # Never selected via ordinary keyword ranking, only ever reachable via
    # `select(..., always=...)` — e.g. `finish_objective`, whose own
    # "complete"/"objective"/"root" keywords otherwise let it outrank and
    # displace genuinely useful tools (like `remember`) on any ordinary chat
    # message that happens to share those words, not just during a mission.
    mission_only: bool = False

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
        the model always has *something* — but never more than max_exposed
        from the ranked pool.

        ``always`` force-includes specific tools regardless of their ranking
        (e.g. `finish_objective` during a mission — it may share no keywords
        with a given turn's message, but the model must always be able to
        reach for it). These are added ON TOP of max_exposed, not carved out
        of it — subtracting them would silently crowd out ranked tools like
        `remember`/`search_files` every time a forced tool's slot ate into
        an already-small budget, which is exactly what happened in practice
        during early mission testing: max_exposed=5 with finish_objective
        forced in left only 4 ranked slots for {read_file, write_file,
        search_files, shell, remember}, so one of them was silently
        unreachable on every single decision call."""
        always = always or set()
        q = set(re.findall(r"[a-z0-9]+", query.lower()))
        scored = sorted(
            self._tools.values(),
            key=lambda t: -len(q & (t.keywords | set(re.findall(r"[a-z0-9]+", t.description.lower())))),
        )
        forced = [t for t in scored if t.name in always]
        rest = [t for t in scored if t.name not in always and not t.mission_only]
        return forced + rest[:max_exposed]

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

def _resolve(path: str, default_root: Path | None,
             allowed_roots: tuple[Path, ...] = ()) -> Path | None:
    """A relative path is anchored to `default_root` (e.g. the wiki root)
    when one is configured, so skill-instructed relative paths like
    `raw/foo.md` land in the intended tree rather than the process's cwd.

    An absolute path is only honored if it falls under `default_root` or
    one of `allowed_roots` — otherwise this returns None. Previously any
    absolute path was let through unchanged; in real use the model hit a
    "not a file" error on a wiki-relative path, then explicitly retried
    with an absolute path "to avoid relative path errors" and successfully
    read an unrelated local project's source file that happened to be
    sitting on the same machine. Relative paths were never the actual
    escape hatch — absolute ones were."""
    p = Path(path).expanduser()
    if not p.is_absolute():
        return default_root / p if default_root is not None else p
    roots = tuple(r for r in (default_root, *allowed_roots) if r is not None)
    if not roots:
        return p  # nothing configured to sandbox against
    resolved = p.resolve()
    for root in roots:
        try:
            resolved.relative_to(root.resolve())
            return p
        except ValueError:
            continue
    return None


def _make_read_file(default_root: Path | str | None,
                    allowed_roots: tuple[Path, ...] = ()) -> Callable[..., str]:
    default_base = Path(default_root).expanduser() if default_root else None

    def _read_file(path: str) -> str:
        p = _resolve(path, default_base, allowed_roots)
        if p is None:
            return (f"ERROR: {path} is outside Mist's wiki/workspace roots — "
                    "absolute paths elsewhere on disk aren't readable.")
        if not p.is_file():
            return f"ERROR: not a file: {path}"
        text = p.read_text(encoding="utf-8", errors="replace")
        return text[:8000]
    return _read_file


def _make_write_file(default_root: Path | str | None,
                     allowed_roots: tuple[Path, ...] = ()) -> Callable[..., str]:
    default_base = Path(default_root).expanduser() if default_root else None

    def _write_file(path: str, content: str) -> str:
        p = _resolve(path, default_base, allowed_roots)
        if p is None:
            return (f"ERROR: {path} is outside Mist's wiki/workspace roots — "
                    "absolute paths elsewhere on disk aren't writable.")
        if default_base is not None and p.exists():
            # raw/ is documented in SCHEMA.md as "read but never modify these
            # once written" — untouched source dumps the wiki's curated pages
            # cite back to. That convention was never actually enforced in
            # code, so nothing stopped an overwrite. New sources can still be
            # ingested (this only blocks clobbering a file that already
            # exists); curated pages elsewhere (entities/, concepts/, ...)
            # are unaffected.
            try:
                rel_parts = p.relative_to(default_base).parts
            except ValueError:
                rel_parts = ()
            if rel_parts and rel_parts[0] == "raw":
                return (f"ERROR: {p} is under raw/ and already exists. raw/ sources are "
                        "immutable once written — write a new file for a new source "
                        "instead, or edit a curated page under entities/, concepts/, "
                        "comparisons/, or queries/ instead.")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} chars to {p}"
    return _write_file


def _run_subprocess(args: str | list[str], shell: bool, timeout: float,
                     registry: ProcessRegistry | None,
                     cwd: Path | None = None) -> str:
    """Runs a command via Popen (not subprocess.run) so the live process can
    be registered for an operator kill — cancelling the asyncio task awaiting
    this (via asyncio.to_thread) does not stop a subprocess already running
    in a worker thread, since Python threads can't be preempted."""
    proc = subprocess.Popen(args, shell=shell, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, cwd=cwd)
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


def _shell(command: str, registry: ProcessRegistry | None = None,
          cwd: Path | None = None) -> str:
    return _run_subprocess(command, shell=True, timeout=60, registry=registry, cwd=cwd)


def _make_shell(shell_cfg: ShellConfig | None,
                registry: ProcessRegistry | None = None,
                workspace_path: str | None = None) -> Callable[..., str]:
    if shell_cfg is None or shell_cfg.backend == "local":
        # Runs from a dedicated workspace dir rather than wherever the mist
        # process happened to be launched from — a real run found the model
        # wandering into an unrelated sibling project directory (agent-zero)
        # via relative paths that only "worked" because of the launch cwd.
        cwd = None
        if workspace_path:
            cwd = Path(workspace_path).expanduser()
            cwd.mkdir(parents=True, exist_ok=True)
        return lambda command: _shell(command, registry, cwd=cwd)

    ssh = shell_cfg.ssh

    def _shell_ssh(command: str) -> str:
        args = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
        if ssh.key_path:
            args += ["-i", str(Path(ssh.key_path).expanduser())]
        remote_command = command
        if workspace_path:
            # Best-effort: this is mist's own (locally expanded) workspace
            # path, reused as a literal path on the remote host. Only lines
            # up exactly if the SSH target shares the same home directory
            # convention as the machine mist itself runs on — true for the
            # "SSH to an isolated Kali box" setup this is designed for, but
            # not guaranteed in general. Still strictly better than leaving
            # the remote shell's cwd wherever the SSH session happens to
            # default to.
            remote_command = (f"mkdir -p '{workspace_path}' 2>/dev/null; "
                              f"cd '{workspace_path}' && {command}")
        args += ["-p", str(ssh.port), f"{ssh.user}@{ssh.host}" if ssh.user else ssh.host,
                remote_command]
        return _run_subprocess(args, shell=False, timeout=ssh.timeout, registry=registry)
    return _shell_ssh


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _make_search_files(default_root: Path | str | None,
                       allowed_roots: tuple[Path, ...] = ()) -> Callable[..., str]:
    default_base = Path(default_root).expanduser() if default_root else None

    def _search_files(query: str, root: str | None = None, max_results: int = 20) -> str:
        # `root`, like read_file/write_file's `path`, anchors to the wiki
        # root when relative — previously it was used as-is (relative to the
        # process's cwd), so a model-supplied root like "missions" silently
        # resolved to the wrong place instead of <wiki_root>/missions.
        base = _resolve(root, default_base, allowed_roots) if root else default_base
        if base is None:
            return f"ERROR: {root} is outside Mist's wiki/workspace roots."
        if not base.is_dir():
            return f"ERROR: not a directory: {base}"
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        hits: list[str] = []
        for md in sorted(base.rglob("*.md")):
            if default_base is not None:
                # Mission transcripts live under the wiki root and grow live
                # during the very mission that might be searching them —
                # matching a query against your own prior search for that
                # same query recorded in the log creates unbounded
                # self-referential noise (a real incident: nested quoted
                # blocks multiplying every search). Curated knowledge
                # (entities/concepts/raw) is what this tool is for; the
                # live mission log is out of scope regardless of `root`.
                try:
                    if md.relative_to(default_base).parts[0] == "missions":
                        continue
                except (ValueError, IndexError):
                    pass
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
                      workspace_root: str | Path | None = None,
                      extra_roots: tuple[str | Path, ...] = (),
                      enabled: list[str] | None = None,
                      shell_config: ShellConfig | None = None,
                      process_registry: ProcessRegistry | None = None) -> ToolRegistry:
    # read_file/write_file/search_files anchor relative paths to wiki_root,
    # but an absolute path used to be let through unconditionally — a real
    # run escaped the wiki root that way and read an unrelated project's
    # source elsewhere on disk. workspace_root is the next place absolute
    # paths are allowed to reach (also shell's cwd for the local backend);
    # extra_roots (config: workspace.extra_roots) lets an operator name
    # further directories outside both — e.g. a `~/Desktop/HTB/<machine>/`
    # convention `shell` already writes into unsandboxed, so read_file/
    # write_file/search_files can reach the same files instead of being
    # silently refused and scattering an engagement's notes elsewhere.
    workspace_base = Path(workspace_root).expanduser() if workspace_root else None
    allowed_roots = tuple(
        r for r in (workspace_base, *(Path(p).expanduser() for p in extra_roots)) if r is not None
    )
    reg = ToolRegistry()
    reg.register(Tool(
        name="read_file",
        description="Read a text file from disk (relative paths resolve under the wiki root)",
        parameters={"type": "object", "properties": {"path": {"type": "string"}},
                    "required": ["path"]},
        fn=_make_read_file(wiki_root, allowed_roots),
        keywords={"read", "file", "open", "cat", "show"},
    ))
    reg.register(Tool(
        name="write_file",
        description="Write content to a file on disk (relative paths resolve under the wiki root)",
        parameters={"type": "object",
                    "properties": {"path": {"type": "string"},
                                   "content": {"type": "string"}},
                    "required": ["path", "content"]},
        fn=_make_write_file(wiki_root, allowed_roots),
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
        fn=_make_search_files(wiki_root, allowed_roots),
        keywords={"search", "find", "grep", "wiki", "index"},
    ))
    reg.register(Tool(
        name="shell",
        description="Run a shell command and return its output",
        parameters={"type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"]},
        fn=_make_shell(shell_config, process_registry,
                      str(workspace_root) if workspace_root else None),
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
        mission_only=True,
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
                                                  wiki_root=wiki_root, workspace_root=workspace_root,
                                                  extra_roots=extra_roots,
                                                  enabled=enabled,
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
