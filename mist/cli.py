"""Mist CLI."""
from __future__ import annotations

import typer
from rich.console import Console

from mist.config import load_config
from mist.core.agent import MistAgent
from mist.llm.client import LLMClient
from mist.memory.store import MemoryStore
from mist.skills.router import SkillRouter
from mist.tools.registry import default_registry

app = typer.Typer(add_completion=False, help="Mist — context-frugal agent for small local LLMs")
console = Console()


def _build_agent(config_path: str | None, backend: str | None,
                 model: str | None, base_url: str | None) -> MistAgent:
    cfg = load_config(config_path)
    if backend:
        cfg.backend = backend
    if model:
        cfg.model = model
    if base_url:
        cfg.base_url = base_url

    llm = LLMClient(cfg.backend, cfg.base_url, cfg.model, cfg.api_key,
                    cfg.generation.temperature, cfg.generation.max_tokens)
    store = MemoryStore(cfg.db_path)
    skills = SkillRouter(cfg.skills.library_path)
    tools = default_registry(remember_fn=store.remember)
    return MistAgent(cfg, llm, store, skills, tools)


@app.command()
def chat(config: str = typer.Option(None, help="Path to config.yaml"),
         backend: str = typer.Option(None, help="ollama | vllm"),
         model: str = typer.Option(None),
         base_url: str = typer.Option(None)):
    """Interactive chat session."""
    agent = _build_agent(config, backend, model, base_url)
    console.print(f"[bold cyan]Mist[/] — {agent.cfg.backend} / {agent.cfg.model} "
                  f"(session {agent.session_id}). Ctrl-D to exit.\n")
    while True:
        try:
            user_msg = console.input("[bold green]you ›[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nbye")
            break
        if not user_msg:
            continue
        result = agent.turn(user_msg)
        for step in result.tool_trace:
            console.print(f"  [dim]⚙ {step}[/]")
        console.print(f"[bold cyan]mist ›[/] {result.response}\n")


@app.command()
def ask(prompt: str,
        config: str = typer.Option(None),
        backend: str = typer.Option(None),
        model: str = typer.Option(None),
        base_url: str = typer.Option(None)):
    """One-shot question."""
    agent = _build_agent(config, backend, model, base_url)
    result = agent.turn(prompt)
    console.print(result.response)


if __name__ == "__main__":
    app()
