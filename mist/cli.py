"""Mist CLI."""
from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from mist.banner import BANNER, TAGLINE, help_lines
from mist.config import load_config
from mist.core.agent import MistAgent
from mist.core.summarizer import BatchSummarizer
from mist.core.tool_compressor import ToolOutputCompressor
from mist.llm.client import LLMClient
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


def _build_agent(config_path: str | None, backend: str | None,
                 model: str | None, base_url: str | None) -> MistAgent:
    cfg = _apply_overrides(load_config(config_path), backend, model, base_url)

    llm = LLMClient(cfg.backend, cfg.base_url, cfg.model, cfg.api_key,
                    cfg.generation.temperature, cfg.generation.max_tokens,
                    think=cfg.generation.think)
    store = MemoryStore(cfg.db_path)

    embeddings = None
    if cfg.embeddings.enabled:
        embeddings = EmbeddingClient(cfg.embeddings.backend, cfg.embeddings.base_url,
                                      cfg.embeddings.model, cfg.embeddings.api_key)
    skills = SkillRouter(cfg.skills.library_path, embeddings=embeddings,
                         embedding_threshold=cfg.embeddings.similarity_threshold)

    tool_compressor = None
    if cfg.summarizer.enabled:
        # Tier 3 of tool-output handling: an independently-configured
        # second model (same pattern as `embeddings` above), off by
        # default since tiers 1-2 already cover the tools that actually
        # blow the budget.
        summarizer_llm = LLMClient(cfg.summarizer.backend, cfg.summarizer.base_url,
                                   cfg.summarizer.model, cfg.summarizer.api_key)
        tool_compressor = ToolOutputCompressor(summarizer_llm,
                                               trigger_chars=cfg.summarizer.trigger_chars,
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
    )
    return MistAgent(cfg, llm, store, skills, tools, process_registry=process_registry,
                     tool_compressor=tool_compressor)


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
         base_url: str = typer.Option(None)):
    """Interactive chat session."""
    agent = _build_agent(config, backend, model, base_url)
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
        base_url: str = typer.Option(None)):
    """One-shot question."""
    agent = _build_agent(config, backend, model, base_url)
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
                    think=cfg.generation.think)
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
        base_url: str = typer.Option(None)):
    """Full-screen streaming TUI: live token streaming, queued messages,
    Ctrl+C to interrupt the current turn (Ctrl+Q to quit)."""
    try:
        from mist.tui.app import MistTUI
    except ImportError as exc:
        console.print("[red]The TUI needs the optional `textual` dependency:[/] "
                      "pip install 'mist-agent[tui]'")
        raise typer.Exit(1) from exc
    agent = _build_agent(config, backend, model, base_url)
    MistTUI(agent).run()


if __name__ == "__main__":
    app()
