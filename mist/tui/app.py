"""Full-screen streaming TUI for Mist, in the spirit of Hermes.

- Assistant responses stream token-by-token as the model generates them.
- Tool calls and their results appear live in the transcript as they happen.
- Messages typed while a turn is in flight are queued, not blocked; they're
  dispatched one at a time as soon as the current turn finishes.
- Ctrl+C interrupts (cancels) the in-flight turn — it does not quit the app.
  Ctrl+Q quits.
- `/mission <objective>` drives MistAgent.astream_mission: an autonomous
  multi-turn loop that keeps working the objective without asking for
  per-turn confirmation, until it calls `finish_objective`, hits a safety
  limit, or the operator intervenes. While one is running, Ctrl+C (or
  `/kill`) stops it immediately — including terminating a live subprocess,
  not just abandoning the wait on it — `/pause` and `/resume` control it
  cooperatively (the current step always finishes cleanly first), and plain
  typed messages are folded in as steering notes for its next step instead
  of starting a separate ad hoc turn.
- Every submitted line (message or /command) is recorded to a persistent
  history file shared with `mist chat`. Up/Down recall previous entries;
  ghost-text autofill suggests the most recent matching entry as you type
  (accept with End or Right-arrow-at-end-of-line, both already built into
  Textual's Input).
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.suggester import Suggester
from textual.widgets import Footer, Header, Input, RichLog, Static

from mist.banner import help_lines
from mist.core.agent import MistAgent
from mist.core.mission import MissionControl
from mist.core.stall_watchdog import format_pending_task_stacks, mission_stall_watchdog
from mist.core.summarizer import BatchSummarizer
from mist.history import HistoryStore
from mist.tui.memory_commands import render_history_command, render_memories_command
from mist.tui.onboarding import build_onboarding_panel
from mist.tui.render import format_status, format_tool_call
from mist.tui.theme import ACCENT, ERROR, MIST_THEME, SUCCESS, WARNING


class HistorySuggester(Suggester):
    """Ghost-text autofill from persistent input history: suggests the most
    recent entry whose prefix matches what's currently typed."""
    def __init__(self, history: HistoryStore):
        super().__init__(use_cache=False, case_sensitive=False)
        self.history = history

    async def get_suggestion(self, value: str) -> str | None:
        if not value:
            return None
        for entry in reversed(self.history.all()):
            if entry.casefold().startswith(value) and entry.casefold() != value:
                return entry
        return None


class HistoryInput(Input):
    """An Input with shell-style Up/Down history recall, layered on top of
    Textual's own ghost-text suggestion mechanism (via HistorySuggester)."""
    BINDINGS = [
        Binding("up", "history_prev", show=False),
        Binding("down", "history_next", show=False),
    ]

    def __init__(self, history: HistoryStore, **kwargs):
        super().__init__(suggester=HistorySuggester(history), **kwargs)
        self._history = history
        self._nav_index: int | None = None
        self._draft = ""

    def action_history_prev(self) -> None:
        entries = self._history.all()
        if not entries:
            return
        if self._nav_index is None:
            self._draft = self.value
            self._nav_index = len(entries) - 1
        elif self._nav_index > 0:
            self._nav_index -= 1
        self.value = entries[self._nav_index]
        self.action_end()

    def action_history_next(self) -> None:
        if self._nav_index is None:
            return
        entries = self._history.all()
        self._nav_index += 1
        if self._nav_index >= len(entries):
            self._nav_index = None
            self.value = self._draft
        else:
            self.value = entries[self._nav_index]
        self.action_end()

    def reset_history_nav(self) -> None:
        self._nav_index = None
        self._draft = ""


class MistTUI(App):
    CSS_PATH = "app.tcss"

    BINDINGS = [
        Binding("ctrl+c", "interrupt", "Interrupt/Kill", priority=True),
        Binding("ctrl+q", "quit", "Quit", priority=True),
    ]

    def __init__(self, agent: MistAgent):
        super().__init__()
        self.register_theme(MIST_THEME)
        self.theme = "mist-dark"
        self.agent = agent
        self.history = HistoryStore(agent.cfg.history_file)
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self._turn_task: asyncio.Task | None = None
        self._dispatch_task: asyncio.Task | None = None
        self._mission_task: asyncio.Task | None = None
        self._mission_control: MissionControl | None = None
        self._mission_objective: str = ""
        self._mission_turn: int = 0
        self._mission_last_progress: float = 0.0  # monotonic ts of last mission event (stall watchdog)
        self._mission_log_path: str | None = None
        # Status-line state: `_status_state` is the free-text phase ("idle",
        # "thinking…", "tool: shell", ...); `_turn_started_at` (wall-clock,
        # via time.monotonic) drives the live elapsed-time counter and is
        # only non-None while a turn or mission is actually in flight.
        self._status_state: str = "idle"
        self._turn_started_at: float | None = None

    # ------------------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield RichLog(id="transcript", wrap=True, markup=True, highlight=False)
        yield Static("", id="typing")
        yield Static("", id="status")
        yield HistoryInput(self.history,
                           placeholder="Message Mist… (/help for commands, "
                                       "↑↓ history, Ctrl+C interrupts, Ctrl+Q quits)",
                           id="input")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "Mist"
        self.sub_title = f"{self.agent.cfg.backend} / {self.agent.cfg.model} (session {self.agent.session_id})"
        self.query_one("#transcript", RichLog).border_title = f" mist · session {self.agent.session_id} "
        self._write_welcome()
        self._refresh_status("idle")
        self._dispatch_task = asyncio.create_task(self._dispatch_loop())
        self.set_interval(1.0, self._tick_status)
        self.query_one("#input", Input).focus()

    async def on_unmount(self) -> None:
        if self._dispatch_task is not None:
            self._dispatch_task.cancel()
        if self._turn_task is not None:
            self._turn_task.cancel()
        if self._mission_task is not None:
            self._mission_task.cancel()

    # ------------------------------------------------------------------
    def _write_welcome(self) -> None:
        log = self.query_one("#transcript", RichLog)
        log.write(build_onboarding_panel(self.agent))
        log.write("")
        log.write("[dim]Type a message to talk to Mist, or a /command above. "
                  "↑/↓ recall history, ghost text autofills a past match. "
                  "/help for the full command list.[/]")
        log.write("")

    def _write_help(self, log: RichLog) -> None:
        log.write("[bold]Commands[/]")
        for line in help_lines():
            log.write(f"  [dim]{line}[/]")

    def _refresh_status(self, state: str) -> None:
        self._status_state = state
        if state == "idle":
            self._turn_started_at = None
        elif self._turn_started_at is None:
            self._turn_started_at = time.monotonic()
        self._render_status()

    def _render_status(self) -> None:
        elapsed = (time.monotonic() - self._turn_started_at
                  if self._turn_started_at is not None else None)
        text = format_status(
            state=self._status_state,
            elapsed=elapsed,
            model=self.agent.cfg.model,
            used_tokens=self.agent.last_context_tokens,
            budget=self.agent.cfg.context.token_budget,
            session_id=self.agent.session_id,
            queued=self.queue.qsize(),
        )
        self.query_one("#status", Static).update(f"[dim]{text}[/]")

    def _tick_status(self) -> None:
        # Keeps the elapsed-time counter visibly live during a long-running
        # tool call with no other events to trigger a re-render.
        if self._turn_started_at is not None:
            self._render_status()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        self.history.add(text)
        if isinstance(event.input, HistoryInput):
            event.input.reset_history_nav()
        if text.startswith("/"):
            self._handle_command(text)
            return
        if self._mission_task is not None and not self._mission_task.done():
            self._mission_control.add_note(text)
            self.query_one("#transcript", RichLog).write(
                f"[bold {SUCCESS}]you ›[/] {text}\n"
                f"  [dim]‹ noted — will steer the mission's next step ›[/]"
            )
            return
        self.query_one("#transcript", RichLog).write(f"[bold {SUCCESS}]you ›[/] {text}")
        self.queue.put_nowait(text)
        self._refresh_status("thinking…" if self._turn_task else "idle")

    # ------------------------------------------------------------------
    def _handle_command(self, text: str) -> None:
        log = self.query_one("#transcript", RichLog)
        name, _, rest = text[1:].partition(" ")
        name, rest = name.lower(), rest.strip()
        if name in ("quit", "exit"):
            self.exit()
            return
        if name == "help":
            self._write_help(log)
            return
        if name == "clear":
            log.clear()
            return
        if name == "new":
            self.agent.session_id = self.agent.store.new_session()
            self.sub_title = (f"{self.agent.cfg.backend} / {self.agent.cfg.model} "
                              f"(session {self.agent.session_id})")
            log.write(f"[dim]Started new session {self.agent.session_id}.[/]")
            return
        if name == "models":
            asyncio.create_task(self._run_models_list())
            return
        if name == "model":
            if not rest:
                log.write(f"[dim]{self.agent.cfg.backend} / {self.agent.llm.model} "
                          f"(session {self.agent.session_id})[/]")
                return
            asyncio.create_task(self._run_model_switch(rest))
            return
        if name == "compact":
            log.write("[dim]Compacting old sessions…[/]")
            asyncio.create_task(self._run_compact())
            return
        if name == "mission":
            self._handle_mission_command(rest, log)
            return
        if name == "pause":
            if self._mission_task is not None and not self._mission_task.done():
                self._mission_control.pause()
                log.write("[dim]Pausing after the mission's current step…[/]")
            else:
                log.write("[dim]No mission is running.[/]")
            return
        if name == "resume":
            if self._mission_control is not None and self._mission_control.paused:
                self._mission_control.resume()
                log.write("[dim]Resumed.[/]")
            else:
                log.write("[dim]No paused mission to resume.[/]")
            return
        if name == "kill":
            if self._mission_task is not None and not self._mission_task.done():
                self._kill_mission()
            else:
                log.write("[dim]No mission is running.[/]")
            return
        if name == "memories":
            log.write(render_memories_command(self.agent.store, rest))
            return
        if name == "history":
            log.write(render_history_command(self.agent.store, rest))
            return
        log.write(f"[red]Unknown command /{name}. Type /help for a list.[/]")

    def _handle_mission_command(self, objective: str, log: RichLog) -> None:
        if self._mission_task is not None and not self._mission_task.done():
            if not objective:
                state = "paused" if self._mission_control.paused else "running"
                log.write(f"[dim]Mission {state} (turn {self._mission_turn}): "
                         f"{self._mission_objective}[/]")
            else:
                log.write("[yellow]A mission is already running — /kill it first to start "
                          "a new one.[/]")
            return
        if not objective:
            log.write("[red]Usage: /mission <objective>[/]")
            return
        if self._turn_task is not None and not self._turn_task.done():
            log.write("[yellow]An ad hoc turn is still running — wait for it to finish "
                      "before starting a mission.[/]")
            return
        self._mission_objective = objective
        self._mission_control = MissionControl()
        self._mission_turn = 0
        self._mission_task = asyncio.create_task(self._run_mission(objective))

    def _kill_mission(self) -> None:
        if self.agent.process_registry is not None:
            self.agent.process_registry.kill_active()
        if self._mission_task is not None:
            self._mission_task.cancel()

    async def _run_models_list(self) -> None:
        log = self.query_one("#transcript", RichLog)
        try:
            models = await asyncio.to_thread(self.agent.llm.list_models)
        except Exception as exc:
            log.write(f"[red]Failed to list models: {exc}[/]")
            return
        if not models:
            log.write("[dim]No models found on the backend.[/]")
            return
        current = self.agent.llm.model
        for m in models:
            marker = "[bold green]*[/]" if m == current else " "
            log.write(f" {marker} {m}")

    async def _run_model_switch(self, name: str) -> None:
        log = self.query_one("#transcript", RichLog)
        try:
            available = await asyncio.to_thread(self.agent.llm.list_models)
        except Exception:
            available = None
        if available is not None and name not in available:
            log.write(f"[yellow]Warning: {name!r} isn't in the backend's model list "
                      f"— switching anyway (pull it first if the next turn fails).[/]")
        self.agent.llm.model = name
        self.agent.cfg.model = name
        self.sub_title = f"{self.agent.cfg.backend} / {name} (session {self.agent.session_id})"
        log.write(f"[dim]Switched active model to {name}.[/]")

    async def _run_compact(self) -> None:
        log = self.query_one("#transcript", RichLog)
        summarizer = BatchSummarizer(self.agent.llm, self.agent.store)
        results = await asyncio.to_thread(
            summarizer.compact_old_sessions,
            keep_recent=self.agent.cfg.memory.keep_recent_sessions,
            min_turns=self.agent.cfg.memory.compact_min_turns,
        )
        total = sum(r.memories_written for r in results)
        log.write(f"[dim]Compacted {len(results)} session(s) into {total} memories.[/]")

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
        buffer = [""]
        try:
            async for event in self.agent.astream_turn(user_msg):
                self._render_turn_event(event, log, typing, buffer)
        finally:
            typing.update("")

    def _render_turn_event(self, event, log: RichLog, typing: Static, buffer: list[str]) -> None:
        """Renders one TurnEvent-shaped event (kind: delta/tool_start/
        tool_result/done/error) — shared by ad hoc turns and mission turns
        (via MissionEvent, which forwards the same kinds) so both render
        identically. `buffer` is a single-element list so callers can share
        mutable accumulation state across repeated calls."""
        if event.kind == "delta":
            buffer[0] += event.text
            typing.update(f"[bold {ACCENT}]mist ›[/] {buffer[0]}")
        elif event.kind == "tool_start":
            self._refresh_status(f"tool: {event.tool}")
            log.write(f"  [{ACCENT}]⚙ {format_tool_call(event.tool, event.detail)}[/]")
        elif event.kind == "tool_result":
            # No further truncation here — event.text is already bounded by
            # cfg.context.max_tool_output_chars (agent.py), the same amount
            # fed to the model. A prior hardcoded 200-char cut meant the
            # operator saw much less than the model reasoned over (e.g. an
            # nmap scan finding several ports would only show the first one
            # in the live stream, even though the fuller output was already
            # in the mission log file) — that gap is the bug, not a feature.
            log.write(f"  [dim]  → {event.text}[/]")
            self._refresh_status("thinking…")
        elif event.kind == "done":
            # Move the finished answer from the ephemeral "typing" widget
            # into the permanent transcript.
            log.write(f"[bold {ACCENT}]mist ›[/] {event.text}")
            typing.update("")
            buffer[0] = ""
        elif event.kind == "error":
            log.write(f"[{ERROR}]error: {event.text}[/]")
            typing.update("")
            buffer[0] = ""

    # ------------------------------------------------------------------
    async def _run_mission(self, objective: str) -> None:
        log = self.query_one("#transcript", RichLog)
        typing = self.query_one("#typing", Static)
        buffer = [""]
        control = self._mission_control
        log.write(f"[bold magenta]‹ mission started ›[/] {objective}")
        self._refresh_status("mission: turn 0")
        # Stall watchdog: a separate task (so it keeps running even when every
        # mission coroutine is parked) that dumps pending task stacks into the
        # log if no event arrives for cfg.mission.stall_watchdog_seconds. Purely
        # diagnostic — see mist.core.stall_watchdog.
        self._mission_last_progress = time.monotonic()
        stall_after = self.agent.cfg.mission.stall_watchdog_seconds
        watchdog = (
            asyncio.create_task(mission_stall_watchdog(
                lambda: self._mission_last_progress, self._on_mission_stall,
                threshold=stall_after, poll_interval=min(15.0, stall_after / 3),
            ))
            if stall_after and stall_after > 0 else None
        )
        try:
            async for event in self.agent.astream_mission(
                objective, control,
                max_turns=self.agent.cfg.mission.max_turns,
                max_seconds=self.agent.cfg.mission.max_seconds,
                stuck_repeat_threshold=self.agent.cfg.mission.stuck_repeat_threshold,
            ):
                self._mission_last_progress = time.monotonic()
                if event.kind == "started":
                    self._mission_log_path = event.text
                elif event.kind == "turn_start":
                    self._mission_turn += 1
                    self._refresh_status(f"mission: turn {self._mission_turn}")
                elif event.kind == "finished":
                    log.write(f"[bold magenta]‹ mission finished ›[/] {event.text}")
                elif event.kind == "debrief":
                    log.write(f"[dim]{event.text}[/]")
                elif event.kind == "recovering":
                    log.write(f"[cyan]‹ recovering ›[/] {event.text}")
                    self._refresh_status(f"mission: recovering (turn {self._mission_turn})")
                elif event.kind == "stuck":
                    log.write(f"[{WARNING}]‹ mission paused ›[/] {event.text}")
                    self._refresh_status(f"mission: paused (turn {self._mission_turn})")
                else:
                    self._render_turn_event(event, log, typing, buffer)
        except asyncio.CancelledError:
            log.write(f"[{ERROR}]‹ mission killed by operator ›[/]")
            if self._mission_log_path is not None:
                log.write("[dim]Debriefing what was accomplished before the kill…[/]")
                debrief = await asyncio.to_thread(
                    self.agent.debrief_mission, objective, "killed by the operator",
                    Path(self._mission_log_path),
                )
                log.write(f"[dim]{debrief.summary()}[/]")
        except Exception as exc:
            # A mission must never vanish silently. Without this, any unhandled
            # exception escaping astream_mission (e.g. a crash while processing
            # a tool's output) ended the task with NO operator-visible event and
            # left "no mission is running" with no clue why — confirmed live
            # (session 128 died mid-turn, no notification, no diagnostic, the
            # traceback lost to an unretrieved-task warning on stderr). Surface
            # it, persist the traceback into the mission log as a durable
            # terminal marker, and debrief what was accomplished — same shape as
            # the operator-kill path above.
            import traceback
            log.write(f"[{ERROR}]‹ mission ended unexpectedly ›[/] {exc!r}")
            if self._mission_log_path is not None:
                self.agent._mission_log_append(
                    Path(self._mission_log_path),
                    f"\n**Ended** unexpectedly on an error: {exc!r}\n```\n"
                    f"{traceback.format_exc()}```\n",
                )
                log.write("[dim]Debriefing what was accomplished before the crash…[/]")
                debrief = await asyncio.to_thread(
                    self.agent.debrief_mission, objective,
                    f"ended on an unexpected error: {exc!r}", Path(self._mission_log_path),
                )
                log.write(f"[dim]{debrief.summary()}[/]")
        finally:
            if watchdog is not None:
                watchdog.cancel()
            typing.update("")
            self._mission_task = None
            self._mission_control = None
            self._mission_turn = 0
            self._mission_log_path = None
            self._refresh_status("idle")

    def _on_mission_stall(self, idle: float) -> None:
        """Stall-watchdog callback: the mission stream has been silent past the
        threshold. Capture every pending task's stack into the mission log
        (durable) and flag it in the transcript, so a wedged turn that would
        otherwise leave no trace names its parked coroutine. Diagnostic only —
        the mission itself is never touched."""
        dump = format_pending_task_stacks(asyncio.all_tasks())
        log = self.query_one("#transcript", RichLog)
        log.write(f"[{WARNING}]‹ mission stalled ›[/] no event for {idle:.0f}s — "
                  "pending task stacks captured to the mission log")
        if self._mission_log_path is not None:
            ts = time.strftime("%H:%M:%S", time.gmtime())
            self.agent._mission_log_append(
                Path(self._mission_log_path),
                f"\n**Stall watchdog** at {ts} UTC: no mission event for {idle:.0f}s — "
                f"pending asyncio task stacks:\n```\n{dump}\n```\n",
            )

    # ------------------------------------------------------------------
    def action_interrupt(self) -> None:
        if self._mission_task is not None and not self._mission_task.done():
            self._kill_mission()
            return
        if self._turn_task is not None and not self._turn_task.done():
            self._turn_task.cancel()
        else:
            self.bell()
