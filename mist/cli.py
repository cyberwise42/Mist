"""Mist CLI."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import typer
import yaml
from rich.console import Console

from mist import backup as backup_module
from mist import doctor as doctor_module
from mist.banner import BANNER, TAGLINE, help_lines
from mist.config import load_config, resolve_config_path
from mist.core.agent import MistAgent
from mist.core.checkpoints import CheckpointStore
from mist.core.context_compressor import ContextCompressor
from mist.core.summarizer import BatchSummarizer
from mist.core.tool_compressor import ToolOutputCompressor
from mist.llm.client import LLMClient, compute_max_tokens, compute_num_ctx, compute_timeout
from mist.llm.embeddings import EmbeddingClient
from mist.memory.store import MemoryStore
from mist.skills.router import SkillRouter
from mist.tools.registry import ProcessRegistry, default_registry
from mist.wiki import init_wiki

try:
    import readline  # noqa: F401 — importing wires it into input() as the line editor
except ImportError:  # pragma: no cover - not available on stock Windows
    readline = None

app = typer.Typer(add_completion=False, help="Mist — context-frugal agent for small local LLMs")
console = Console()


def _apply_overrides(cfg, backend, model, base_url):
    if backend:
        cfg.backend = backend
    if model:
        cfg.model = model
    if base_url:
        cfg.base_url = base_url
    return cfg


def _parse_skills_option(values: list[str] | None) -> list[str] | None:
    """`--skills` accepts either a repeated flag (`-s a -s b`) or
    comma-separated values (`-s a,b`), matching the flag's own help text —
    normalizes both into one flat list, or None if nothing was passed."""
    if not values:
        return None
    result = [name.strip() for value in values for name in value.split(",") if name.strip()]
    return result or None


def _build_agent(config_path: str | None, backend: str | None,
                 model: str | None, base_url: str | None,
                 preload_skills: list[str] | None = None,
                 yolo: bool = False) -> MistAgent:
    cfg = _apply_overrides(load_config(config_path), backend, model, base_url)
    if yolo:
        cfg.security.command_safety_enabled = False
        console.print("[yellow]--yolo: command-safety gate disabled for this run — "
                      "shell commands run with no pre-execution safety check.[/]")

    llm = LLMClient(cfg.backend, cfg.base_url, cfg.model, cfg.api_key,
                    cfg.generation.temperature, cfg.generation.max_tokens,
                    timeout=compute_timeout(cfg.generation.max_tokens),
                    think=cfg.generation.think, keep_alive=cfg.generation.keep_alive,
                    stream_no_content_timeout=cfg.generation.stream_no_content_timeout,
                    decision_max_tokens=cfg.generation.decision_max_tokens,
                    stream_max_content_tokens=cfg.generation.stream_max_content_tokens)
    # Discover the model's real max context length and explicitly request a
    # matching num_ctx — left unset, Ollama loads the model with whatever
    # its Modelfile/tag defaults to (often much smaller than what Mist
    # actually assembles), which silently truncates the prompt from the
    # front with no error. A failed discovery (server unreachable, an older
    # Ollama, the vLLM backend) leaves num_ctx/max_tokens unchanged —
    # today's behavior. Bounded by a short timeout (see
    # discover_context_length) and announced here — confirmed live: with no
    # status line and the client's full 120s timeout, an unreachable server
    # made `mist tui` look hung for two minutes before ever showing a prompt.
    console.print("[dim]Checking model context window...[/]", end="\r")
    discovered_ctx = llm.discover_context_length()
    console.print(" " * 40, end="\r")  # clear the status line
    if discovered_ctx is None:
        console.print("[dim]Could not reach the model backend to check its context "
                      "window — continuing without num_ctx/max_tokens adjustment.[/]")
    llm.num_ctx = compute_num_ctx(cfg.context.token_budget, cfg.generation.max_tokens, discovered_ctx)
    effective_max_tokens = compute_max_tokens(cfg.generation.max_tokens, cfg.context.token_budget,
                                              discovered_ctx)
    if effective_max_tokens != cfg.generation.max_tokens:
        llm.max_tokens = effective_max_tokens
        console.print(
            f"[yellow]generation.max_tokens ({cfg.generation.max_tokens}) reduced to "
            f"{effective_max_tokens} to fit inside {cfg.model}'s real context window "
            f"({discovered_ctx}) alongside context.token_budget ({cfg.context.token_budget}).[/]"
        )
    store = MemoryStore(cfg.db_path)

    embeddings = None
    if cfg.embeddings.enabled:
        embeddings = EmbeddingClient(cfg.embeddings.backend, cfg.embeddings.base_url,
                                      cfg.embeddings.model, cfg.embeddings.api_key)
    skills = SkillRouter(cfg.skills.library_path, embeddings=embeddings,
                         embedding_threshold=cfg.embeddings.similarity_threshold)

    # Both tool_compressor (tier 3 of tool-output handling) and
    # context_compressor (ContextConfig.compress_on_overflow) are optional
    # and independently toggled, but share the same aux-model connection
    # (config.summarizer) — built once if either is on, rather than
    # constructing two separate LLMClients for the same backend/model.
    tool_compressor = None
    context_compressor = None
    if cfg.summarizer.enabled or cfg.context.compress_on_overflow:
        summarizer_llm = LLMClient(cfg.summarizer.backend, cfg.summarizer.base_url,
                                   cfg.summarizer.model, cfg.summarizer.api_key)
        if cfg.summarizer.enabled:
            tool_compressor = ToolOutputCompressor(summarizer_llm,
                                                   trigger_chars=cfg.summarizer.trigger_chars,
                                                   max_input_chars=cfg.summarizer.max_input_chars)
        if cfg.context.compress_on_overflow:
            context_compressor = ContextCompressor(summarizer_llm,
                                                   max_input_chars=cfg.summarizer.max_input_chars)

    process_registry = ProcessRegistry()
    tools = default_registry(
        remember_fn=store.remember,
        llm=llm if cfg.subagents.enabled else None,
        max_subagent_workers=cfg.subagents.max_workers,
        skills=skills,
        max_subagent_steps=cfg.subagents.max_steps,
        subagent_tools_enabled=cfg.subagents.tools_enabled,
        wiki_root=cfg.wiki_root,
        workspace_root=cfg.workspace_root,
        extra_roots=cfg.extra_workspace_roots,
        enabled=cfg.tools.enabled,
        shell_config=cfg.tools.shell,
        process_registry=process_registry,
        artifact_config=cfg.artifacts,
        structured_tools_config=cfg.tools.structured,
        tool_compressor=tool_compressor,
        security_config=cfg.security,
        browser_config=cfg.tools.browser,
    )
    checkpoint_store = CheckpointStore(cfg.checkpoints.base_dir) if cfg.checkpoints.enabled else None
    return MistAgent(cfg, llm, store, skills, tools, process_registry=process_registry,
                     tool_compressor=tool_compressor, preload_skills=preload_skills,
                     checkpoint_store=checkpoint_store, context_compressor=context_compressor)


def _print_welcome(agent: MistAgent) -> None:
    console.print(f"[bold cyan]{BANNER}[/]")
    console.print(f"[dim]{TAGLINE}[/]\n")
    console.print(f"{agent.cfg.backend} / {agent.cfg.model} (session {agent.session_id})\n")
    console.print("[bold]Commands[/]")
    for line in help_lines():
        console.print(f"  [dim]{line}[/]")
    console.print("\nType a message to talk to Mist. Ctrl-D to exit.\n")


def _switch_model(agent: MistAgent, name: str) -> None:
    try:
        available = agent.llm.list_models()
    except Exception:
        available = None
    if available is not None and name not in available:
        console.print(f"[yellow]Warning: {name!r} isn't in the backend's model list "
                      f"— switching anyway (pull it first if the next turn fails).[/]")
    agent.llm.model = name
    agent.cfg.model = name
    console.print(f"[dim]Switched active model to {name}.[/]")


def _handle_command(cmd: str, agent: MistAgent) -> bool:
    """Handles a leading-`/` line typed at the chat prompt. Returns True if
    the caller should exit the REPL loop."""
    name, _, rest = cmd[1:].partition(" ")
    name, rest = name.lower(), rest.strip()
    if name in ("quit", "exit"):
        console.print("bye")
        return True
    if name == "help":
        console.print("[bold]Commands[/]")
        for line in help_lines():
            console.print(f"  [dim]{line}[/]")
        return False
    if name == "clear":
        console.clear()
        return False
    if name == "new":
        agent.session_id = agent.store.new_session()
        console.print(f"[dim]Started new session {agent.session_id}.[/]")
        return False
    if name == "models":
        try:
            models = agent.llm.list_models()
        except Exception as exc:
            console.print(f"[red]Failed to list models: {exc}[/]")
            return False
        if not models:
            console.print("[dim]No models found on the backend.[/]")
            return False
        for m in models:
            marker = "[bold green]*[/]" if m == agent.llm.model else " "
            console.print(f" {marker} {m}")
        return False
    if name == "model":
        if not rest:
            console.print(f"[dim]{agent.cfg.backend} / {agent.llm.model} "
                          f"(session {agent.session_id})[/]")
            return False
        _switch_model(agent, rest)
        return False
    if name == "compact":
        summarizer = BatchSummarizer(agent.llm, agent.store)
        results = summarizer.compact_old_sessions(
            keep_recent=agent.cfg.memory.keep_recent_sessions,
            min_turns=agent.cfg.memory.compact_min_turns,
        )
        total = sum(r.memories_written for r in results)
        console.print(f"[dim]Compacted {len(results)} session(s) into {total} memories.[/]")
        return False
    if name in ("mission", "pause", "resume", "kill"):
        console.print("[yellow]Mission mode needs live pause/resume/kill controls while a "
                      "turn runs in the background — that needs `mist tui`, not this plain "
                      "REPL.[/]")
        return False
    console.print(f"[red]Unknown command /{name}. Type /help for a list.[/]")
    return False


@app.command()
def chat(config: str = typer.Option(None, help="Path to config.yaml"),
         backend: str = typer.Option(None, help="ollama | vllm"),
         model: str = typer.Option(None),
         base_url: str = typer.Option(None),
         skills: list[str] = typer.Option(None, "--skills", "-s",
                                          help="Preload one or more skills for the session "
                                               "(repeat flag or comma-separate)"),
         yolo: bool = typer.Option(False, "--yolo",
                                   help="Bypass the command-safety gate (use at your own risk)")):
    """Interactive chat session."""
    agent = _build_agent(config, backend, model, base_url, _parse_skills_option(skills), yolo)
    _print_welcome(agent)
    history_path = agent.cfg.history_file
    if readline is not None:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            readline.read_history_file(history_path)
        except OSError:
            pass  # no history yet, or an unreadable/foreign-format file — start fresh
        readline.set_history_length(500)
    try:
        while True:
            try:
                user_msg = console.input("[bold green]you ›[/] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\nbye")
                break
            if not user_msg:
                continue
            if user_msg.startswith("/"):
                if _handle_command(user_msg, agent):
                    break
                continue
            try:
                result = agent.turn(user_msg)
            except Exception as exc:  # a crashed turn must not kill the whole REPL/session
                console.print(f"[red]turn crashed: {exc}[/]\n")
                continue
            for step in result.tool_trace:
                console.print(f"  [dim]⚙ {step}[/]")
            console.print(f"[bold cyan]mist ›[/] {result.response}\n")
    finally:
        if readline is not None:
            try:
                readline.write_history_file(history_path)
            except OSError:
                pass


@app.command()
def ask(prompt: str,
        config: str = typer.Option(None),
        backend: str = typer.Option(None),
        model: str = typer.Option(None),
        base_url: str = typer.Option(None),
        skills: list[str] = typer.Option(None, "--skills", "-s",
                                         help="Preload one or more skills for this call "
                                              "(repeat flag or comma-separate)"),
        yolo: bool = typer.Option(False, "--yolo",
                                  help="Bypass the command-safety gate (use at your own risk)")):
    """One-shot question."""
    agent = _build_agent(config, backend, model, base_url, _parse_skills_option(skills), yolo)
    try:
        result = agent.turn(prompt)
    except Exception as exc:
        console.print(f"[red]turn crashed: {exc}[/]")
        raise typer.Exit(1) from exc
    console.print(result.response)


@app.command()
def compact(config: str = typer.Option(None, help="Path to config.yaml"),
            backend: str = typer.Option(None, help="ollama | vllm"),
            model: str = typer.Option(None),
            base_url: str = typer.Option(None),
            keep_recent: int = typer.Option(None, help="Newest sessions to leave uncompacted"),
            min_turns: int = typer.Option(None, help="Sessions shorter than this are skipped")):
    """Batch-compress old sessions into long-term memory."""
    cfg = _apply_overrides(load_config(config), backend, model, base_url)
    llm = LLMClient(cfg.backend, cfg.base_url, cfg.model, cfg.api_key,
                    cfg.generation.temperature, cfg.generation.max_tokens,
                    timeout=compute_timeout(cfg.generation.max_tokens),
                    think=cfg.generation.think, keep_alive=cfg.generation.keep_alive,
                    stream_no_content_timeout=cfg.generation.stream_no_content_timeout,
                    decision_max_tokens=cfg.generation.decision_max_tokens,
                    stream_max_content_tokens=cfg.generation.stream_max_content_tokens)
    store = MemoryStore(cfg.db_path)
    summarizer = BatchSummarizer(llm, store)

    results = summarizer.compact_old_sessions(
        keep_recent=keep_recent if keep_recent is not None else cfg.memory.keep_recent_sessions,
        min_turns=min_turns if min_turns is not None else cfg.memory.compact_min_turns,
    )
    if not results:
        console.print("[dim]No sessions eligible for compaction.[/]")
        return
    total = sum(r.memories_written for r in results)
    console.print(f"Compacted {len(results)} session(s) into {total} "
                  f"memor{'y' if total == 1 else 'ies'}.")
    for r in results:
        console.print(f"  [dim]session {r.session_id}: {r.memories_written} memories[/]")


@app.command(name="wiki-init")
def wiki_init(config: str = typer.Option(None, help="Path to config.yaml"),
              root: str = typer.Option(None, help="Override the configured wiki root")):
    """Scaffold the wiki skeleton (SCHEMA.md, index.md, log.md, and
    raw/entities/concepts/comparisons/queries dirs) if not already present."""
    cfg = load_config(config)
    target = Path(root).expanduser() if root else cfg.wiki_root
    created = init_wiki(target)
    if not created:
        console.print(f"[dim]Wiki already initialized at {target}.[/]")
        return
    console.print(f"Initialized wiki at {target}:")
    for path in created:
        console.print(f"  [dim]created {path}[/]")


@app.command()
def tui(config: str = typer.Option(None, help="Path to config.yaml"),
        backend: str = typer.Option(None, help="ollama | vllm"),
        model: str = typer.Option(None),
        base_url: str = typer.Option(None),
        skills: list[str] = typer.Option(None, "--skills", "-s",
                                         help="Preload one or more skills for the session "
                                              "(repeat flag or comma-separate)"),
        yolo: bool = typer.Option(False, "--yolo",
                                  help="Bypass the command-safety gate (use at your own risk)")):
    """Full-screen streaming TUI: live token streaming, queued messages,
    Ctrl+C to interrupt the current turn (Ctrl+Q to quit)."""
    try:
        from mist.tui.app import MistTUI
    except ImportError as exc:
        console.print("[red]The TUI needs the optional `textual` dependency:[/] "
                      "pip install 'mist-agent[tui]'")
        raise typer.Exit(1) from exc
    agent = _build_agent(config, backend, model, base_url, _parse_skills_option(skills), yolo)
    MistTUI(agent).run()


_STATUS_STYLE = {"ok": "green", "warn": "yellow", "fail": "red"}


@app.command()
def doctor(config: str = typer.Option(None, help="Path to config.yaml"),
          fix: bool = typer.Option(False, "--fix", help="Attempt to auto-fix issues found")):
    """Diagnose config/connectivity issues: backend reachability, whether
    the configured model is actually pulled, context-window sizing, the
    embedding backend, the shell backend (SSH connectivity), the wiki, and
    the memory DB path. Exits non-zero if anything failed."""
    cfg = load_config(config)
    if fix:
        for description in doctor_module.fix(cfg):
            console.print(f"[cyan]Fixed:[/] {description}")
    results = doctor_module.run_all(cfg)
    failed = False
    for r in results:
        style = _STATUS_STYLE.get(r.status, "white")
        console.print(f"[{style}]{r.status.upper():5}[/] {r.name:12} {r.detail}")
        if r.status == "fail":
            failed = True
    if failed:
        console.print("\n[red]One or more checks failed.[/]")
        raise typer.Exit(1)
    console.print("\n[green]All checks passed (warnings, if any, are non-fatal).[/]")


checkpoints_app = typer.Typer(add_completion=False,
                              help="Inspect / prune / clear the filesystem checkpoint store")
app.add_typer(checkpoints_app, name="checkpoints")


def _checkpoint_store(config: str | None) -> CheckpointStore:
    return CheckpointStore(load_config(config).checkpoints.base_dir)


def _format_size(size_bytes: int) -> str:
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


@checkpoints_app.command("status")
def checkpoints_status(config: str = typer.Option(None, help="Path to config.yaml")):
    """Show total size, project count, and per-project breakdown."""
    store = _checkpoint_store(config)
    infos = store.list_all()
    if not infos:
        console.print("[dim]No checkpoints recorded.[/]")
        return
    total_size = sum(i.size_bytes for i in infos)
    total_commits = sum(i.commit_count for i in infos)
    console.print(f"{len(infos)} project(s), {total_commits} snapshot(s) total, "
                 f"{_format_size(total_size)} on disk\n")
    for info in infos:
        console.print(f"  {info.workspace_path}: {info.commit_count} snapshot(s), "
                     f"{_format_size(info.size_bytes)}")


@checkpoints_app.command("list")
def checkpoints_list(config: str = typer.Option(None, help="Path to config.yaml")):
    """Alias for 'status'."""
    checkpoints_status(config)


@checkpoints_app.command("prune")
def checkpoints_prune(config: str = typer.Option(None, help="Path to config.yaml")):
    """Delete orphan/stale checkpoints (workspace no longer exists) and GC the rest."""
    store = _checkpoint_store(config)
    removed = store.prune()
    console.print(f"Removed {removed} orphaned checkpoint(s).")


@checkpoints_app.command("clear")
def checkpoints_clear(config: str = typer.Option(None, help="Path to config.yaml"),
                      yes: bool = typer.Option(False, "--yes", help="Skip confirmation")):
    """Delete the entire checkpoint base — all rollback history, every project."""
    store = _checkpoint_store(config)
    if not yes and not typer.confirm(f"Delete ALL checkpoints under {store.base_dir}?"):
        console.print("Aborted.")
        raise typer.Exit(0)
    store.clear()
    console.print("Cleared.")


@app.command()
def backup(config: str = typer.Option(None, help="Path to config.yaml"),
          output: str = typer.Option(None, "--output", "-o",
                                     help="Output zip path (default: "
                                          "~/.mist/backups/mist-backup-<timestamp>.zip)")):
    """Back up ~/.mist (config, memory DB, history, workspace) and the
    wiki knowledge base to a single zip file."""
    cfg = load_config(config)
    path = backup_module.create_backup(cfg, output_path=output)
    console.print(f"Backup written to {path}")


@app.command()
def restore(zip_path: str,
           config: str = typer.Option(None, help="Path to config.yaml"),
           yes: bool = typer.Option(False, "--yes", help="Skip confirmation")):
    """Restore a backup created by `mist backup`, overwriting the current
    ~/.mist and wiki contents at the paths recorded in the backup's own
    manifest. Takes a safety backup of the current state first."""
    cfg = load_config(config)
    if not yes and not typer.confirm(
        f"This overwrites the current ~/.mist and wiki contents with {zip_path}. Continue?"
    ):
        console.print("Aborted.")
        raise typer.Exit(0)
    safety_path = backup_module.create_backup(cfg)
    console.print(f"[dim]Safety backup of the current state written to {safety_path} first.[/]")
    restored = backup_module.restore_backup(zip_path)
    console.print(f"Restored {restored['mist_home']} file(s) to ~/.mist, "
                 f"{restored['wiki']} file(s) to the wiki root.")


config_app = typer.Typer(add_completion=False, help="View and edit configuration")
app.add_typer(config_app, name="config")


@config_app.command("show")
def config_show(config: str = typer.Option(None, help="Path to config.yaml")):
    """Print the fully-resolved config (file values merged over built-in
    defaults) as YAML."""
    cfg = load_config(config)
    console.print(yaml.dump(cfg.model_dump(), sort_keys=False, default_flow_style=False))


@config_app.command("path")
def config_path(config: str = typer.Option(None, help="Path to config.yaml")):
    """Print the path to the config file that would actually be loaded,
    or say so if none exists yet (built-in defaults are in effect)."""
    resolved = resolve_config_path(config)
    if resolved is None:
        console.print("[dim]No config file found — using built-in defaults. "
                      "Default location: ~/.mist/config.yaml[/]")
        raise typer.Exit(1)
    console.print(str(resolved))


@config_app.command("edit")
def config_edit(config: str = typer.Option(None, help="Path to config.yaml")):
    """Open the config file in $EDITOR. If none exists yet, creates one at
    ~/.mist/config.yaml pre-filled with the current (default) settings."""
    resolved = resolve_config_path(config)
    if resolved is None:
        resolved = Path(config).expanduser() if config else Path("~/.mist/config.yaml").expanduser()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(yaml.dump(load_config().model_dump(), sort_keys=False,
                                      default_flow_style=False), encoding="utf-8")
        console.print(f"[dim]No config file found — created {resolved} with default settings.[/]")
    editor = os.environ.get("EDITOR", "vi")
    subprocess.run([editor, str(resolved)])


if __name__ == "__main__":
    app()
