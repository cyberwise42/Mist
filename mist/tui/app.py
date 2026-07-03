"""Full-screen streaming TUI for Mist, in the spirit of Hermes.

- Assistant responses stream token-by-token as the model generates them.
- Tool calls and their results appear live in the transcript as they happen.
- Messages typed while a turn is in flight are queued, not blocked; they're
  dispatched one at a time as soon as the current turn finishes.
- Ctrl+C interrupts (cancels) the in-flight turn — it does not quit the app.
  Ctrl+Q quits.
"""
from __future__ import annotations

import asyncio

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, Input, RichLog, Static

from mist.core.agent import MistAgent


class MistTUI(App):
    CSS = """
    Screen {
        layout: vertical;
    }
    #transcript {
        height: 1fr;
        border: round $accent;
        margin: 0 1;
    }
    #typing {
        height: auto;
        margin: 0 2;
        color: $text;
    }
    #status {
        height: 1;
        margin: 0 2;
        color: $text-muted;
    }
    #input {
        margin: 0 1 1 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "interrupt", "Interrupt turn", priority=True),
    ]

    def __init__(self, agent: MistAgent):
        super().__init__()
        self.agent = agent
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self._turn_task: asyncio.Task | None = None
        self._dispatch_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield RichLog(id="transcript", wrap=True, markup=True, highlight=False)
        yield Static("", id="typing")
        yield Static("", id="status")
        yield Input(placeholder="Message Mist… (Ctrl+C interrupts, Ctrl+Q quits)", id="input")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "Mist"
        self.sub_title = f"{self.agent.cfg.backend} / {self.agent.cfg.model} (session {self.agent.session_id})"
        self._refresh_status("idle")
        self._dispatch_task = asyncio.create_task(self._dispatch_loop())
        self.query_one("#input", Input).focus()

    async def on_unmount(self) -> None:
        if self._dispatch_task is not None:
            self._dispatch_task.cancel()
        if self._turn_task is not None:
            self._turn_task.cancel()

    # ------------------------------------------------------------------
    def _refresh_status(self, state: str) -> None:
        n = self.queue.qsize()
        suffix = f"  ·  {n} queued" if n else ""
        self.query_one("#status", Static).update(f"[dim]{state}{suffix}[/]")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        self.query_one("#transcript", RichLog).write(f"[bold green]you ›[/] {text}")
        self.queue.put_nowait(text)
        self._refresh_status("thinking…" if self._turn_task else "idle")

    # ------------------------------------------------------------------
    async def _dispatch_loop(self) -> None:
        """Pulls queued messages one at a time and runs a turn for each —
        this is what makes queuing work: submitting text never blocks on the
        current turn, it just enqueues."""
        while True:
            user_msg = await self.queue.get()
            self._refresh_status("thinking…")
            self._turn_task = asyncio.create_task(self._run_turn(user_msg))
            try:
                await self._turn_task
            except asyncio.CancelledError:
                self.query_one("#transcript", RichLog).write("[yellow]‹ interrupted ›[/]")
            except Exception as exc:  # a crashed turn must not kill the dispatch loop
                self.query_one("#transcript", RichLog).write(f"[red]turn crashed: {exc}[/]")
            finally:
                self._turn_task = None
                self._refresh_status("idle")

    async def _run_turn(self, user_msg: str) -> None:
        log = self.query_one("#transcript", RichLog)
        typing = self.query_one("#typing", Static)
        buffer = ""
        try:
            async for event in self.agent.astream_turn(user_msg):
                if event.kind == "delta":
                    buffer += event.text
                    typing.update(f"[bold cyan]mist ›[/] {buffer}")
                elif event.kind == "tool_start":
                    self._refresh_status(f"tool: {event.tool}")
                    log.write(f"  [dim]⚙ {event.tool}({event.detail})[/]")
                elif event.kind == "tool_result":
                    log.write(f"  [dim]→ {event.text[:200]}[/]")
                    self._refresh_status("thinking…")
                elif event.kind == "done":
                    # Move the finished answer from the ephemeral "typing"
                    # widget into the permanent transcript.
                    log.write(f"[bold cyan]mist ›[/] {event.text}")
                elif event.kind == "error":
                    log.write(f"[red]error: {event.text}[/]")
        finally:
            typing.update("")

    # ------------------------------------------------------------------
    def action_interrupt(self) -> None:
        if self._turn_task is not None and not self._turn_task.done():
            self._turn_task.cancel()
        else:
            self.bell()
