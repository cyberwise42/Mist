"""Mist core agent loop.

The defining behavior: context is REBUILT every turn, never accumulated.

    prompt = system
           + top-k memories (retrieved against the user message)
           + routed skill (at most one full body)
           + last N verbatim turns
           + current tool trace (this turn only)

Tool output from previous turns is summarized into a single line before it
enters history; full outputs are never carried forward.

Two turn interfaces are offered:

- ``turn()``: synchronous, single JSON call per step (respond bundles the
  answer text into the same schema as the tool decision). Used by `mist chat`
  / `mist ask`.
- ``astream_turn()``: async generator of ``TurnEvent``s, used by the
  streaming TUI (`mist tui`). It splits each step into a small, constrained
  "decide" call (respond vs. use_tool) followed — only when responding — by
  an unconstrained, streamed call that yields the answer text token-by-token.
  Streaming a JSON-wrapped answer would surface raw JSON syntax to the user,
  so the two are kept separate.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator

from mist.config import MistConfig
from mist.core.debrief import DebriefResult, MissionDebriefer
from mist.core.mission import MissionControl, MissionEvent
from mist.llm.client import LLMClient, parse_json_relaxed
from mist.memory.store import MemoryStore
from mist.skills.router import SkillRouter
from mist.tools.registry import ProcessRegistry, ToolRegistry

MAX_DECISION_RETRIES = 2

DECISION_SYSTEM_TEMPLATE = """You are Mist, an autonomous penetration-testing operator. Follow instructions exactly.

Decide exactly one action and reply with a single JSON object matching this shape:
- To answer the user directly: {{"action": "respond"}}
- To use a tool:                {{"action": "use_tool", "tool": "<name>", "arguments": {{...}}}}

Available tools:
{tools}
{skill_section}{memory_section}Rules:
- If the request requires performing an action (a scan, exploit, command, or reading/writing a
  file), you MUST use a tool to actually do it yourself — never just describe steps for the user
  to run manually.
- A broad or multi-phase request ("pentest this machine", "get the flags") is still a request that
  requires action: decide `use_tool` for the next concrete step now. Deciding `respond` here means
  the operator gets a description instead of progress — only do that when no action is needed.
- Only decide to respond directly when no action is needed: answering a question, explaining a
  concept, or reporting results you already produced.
- After a tool call produces a meaningful finding (open ports, a credential, a vulnerability, a
  working technique), persist it with `remember` or `write_file` (under the wiki root) before you
  respond.
- `shell` runs commands on your OWN local machine, not the target. The target is only reachable
  over the network (curl/nmap/ftp/ssh aimed at its IP) — a target's web assets, source, or files
  are never sitting directly on your local disk. Do not run broad local filesystem searches
  (`find /`, recursive grep over `/root`, `/home`, etc.) hunting for "the target's files" — if a
  local search seems tempting, that's a sign to re-check whether you actually need a network
  request against the target instead. Absolute file paths outside the wiki/workspace roots are
  refused by read_file/write_file/search_files for this reason.
- Do not write the answer itself here — a separate step does that.
- One action per reply. No text outside the JSON object."""

ANSWER_SYSTEM_TEMPLATE = """You are Mist, an autonomous penetration-testing operator.
{skill_section}{memory_section}Answer the user's message directly and conversationally, reporting
any real tool output from this turn faithfully — don't invent results and don't just restate
instructions for the user to run themselves.
Reply in plain text only — no JSON, no code fences unless the user asked for code."""

MISSION_CONTINUE_TEMPLATE = """Continue working autonomously toward the objective:

{objective}

This is not a request for a plan — decide the single next concrete step yourself and take it
with a tool right now. Do not ask the operator what to do next. Give a brief status update only
if you have nothing further to do this step. If the objective is fully achieved, call
`finish_objective` with a summary. If you are genuinely stuck and need the operator's judgment
(a missing credential, an ambiguous scope decision), say so plainly instead of guessing.

If you confirm something is a genuine dead end (a path that doesn't exist, a technique that
doesn't apply here) or land on a working technique, persist it immediately with `remember` — it
gets retrieved automatically in later turns, so you won't rediscover the same dead end twice.
Separately, `search_files` the wiki for existing reference knowledge (methodology, known
techniques, past engagements) relevant to whatever service or technology you're dealing with
right now — that's real material worth consulting, not just your own session history."""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def _mission_continue_message(objective: str, notes: list[str], nudge: str | None = None) -> str:
    msg = MISSION_CONTINUE_TEMPLATE.format(objective=objective)
    if nudge:
        msg = f"{nudge}\n\n{msg}"
    if notes:
        msg += "\n\nOperator note(s) since your last step:\n" + "\n".join(f"- {n}" for n in notes)
    return msg


def _approx_tokens(text: str) -> int:
    return len(text) // 4  # cheap heuristic; good enough for budgeting


def _fit_budget(messages: list[dict[str, str]], budget: int) -> list[dict[str, str]]:
    """Enforce the token budget by dropping oldest history first (never the
    system prompt at [0] or the current user message at [-1])."""
    while sum(_approx_tokens(m["content"]) for m in messages) > budget and len(messages) > 2:
        messages.pop(1)
    return messages


@dataclass
class TurnResult:
    response: str
    tool_trace: list[str]


@dataclass
class TurnEvent:
    """One step of a streamed turn, as emitted by ``astream_turn``.

    kind: "delta" (answer text chunk) | "tool_start" | "tool_result" |
          "done" (final full response) | "error"
    """
    kind: str
    text: str = ""
    tool: str = ""
    detail: str = ""


class MistAgent:
    def __init__(self, config: MistConfig, llm: LLMClient, store: MemoryStore,
                 skills: SkillRouter, tools: ToolRegistry, session_id: int | None = None,
                 process_registry: ProcessRegistry | None = None):
        self.cfg = config
        self.llm = llm
        self.store = store
        self.skills = skills
        self.tools = tools
        self.session_id = session_id or store.new_session()
        # Force-included regardless of keyword ranking — set by astream_mission
        # for the duration of a mission so `finish_objective` is always
        # reachable even when a turn's message shares no keywords with it.
        self.always_exposed: set[str] = set()
        # Lets a driver (the TUI) actually terminate a running shell command
        # on kill — see mist.tools.registry.ProcessRegistry for why asyncio
        # task cancellation alone can't do that.
        self.process_registry = process_registry

    # ------------------------------------------------------------------
    def _assemble_context(self, user_msg: str) -> tuple[str, str, str, list]:
        """Builds the three variable sections of every system prompt (tool
        listing, active skill, relevant memories) plus the tools exposed for
        this turn. Shared by every template so tool/skill/memory selection
        logic lives in exactly one place."""
        exposed = self.tools.select(user_msg, self.cfg.tools.max_exposed,
                                    always=self.always_exposed)
        tool_lines = "\n".join(
            f"- {t.name}: {t.description} | args schema: {json.dumps(t.parameters['properties'])}"
            for t in exposed
        )

        skill_section = ""
        candidates = self.skills.route(user_msg, self.cfg.skills.max_candidates)
        if candidates:
            # Load ONLY the top skill's full body; list the rest as one-liners.
            top = candidates[0]
            body = top.body()
            budget_chars = self.cfg.context.token_budget  # rough guard
            skill_section = f"\nActive skill ({top.name}):\n{body[:budget_chars]}\n"
            if len(candidates) > 1:
                others = "; ".join(f"{s.name}: {s.description}" for s in candidates[1:])
                skill_section += f"Other possibly relevant skills: {others}\n"

        memory_section = ""
        memories = self.store.search(user_msg, self.cfg.memory.top_k)
        if memories:
            memory_section = "\nRelevant memories:\n" + "\n".join(f"- {m}" for m in memories) + "\n"

        return tool_lines, skill_section, memory_section, exposed

    def _build_decision_system(self, user_msg: str) -> tuple[str, list]:
        tool_lines, skill_section, memory_section, exposed = self._assemble_context(user_msg)
        system = DECISION_SYSTEM_TEMPLATE.format(
            tools=tool_lines, skill_section=skill_section, memory_section=memory_section
        )
        return system, exposed

    def _build_answer_system(self, user_msg: str) -> str:
        _, skill_section, memory_section, _ = self._assemble_context(user_msg)
        return ANSWER_SYSTEM_TEMPLATE.format(
            skill_section=skill_section, memory_section=memory_section
        )

    def _history(self) -> list[dict[str, str]]:
        return [
            {"role": t["role"] if t["role"] in ("user", "assistant") else "user",
             "content": t["content"]}
            for t in self.store.recent_turns(self.session_id, self.cfg.context.history_turns)
        ]

    # ------------------------------------------------------------------
    def turn(self, user_msg: str) -> TurnResult:
        """Synchronous turn, used by `mist chat` / `mist ask`. Structured the
        same way as astream_turn: a small constrained decision (respond vs.
        use_tool, no `response` field) followed — only once "respond" is
        chosen — by a separate, unconstrained call that generates the actual
        answer text.

        This split matters more than it looks: a combined schema (decide AND
        provide `response` in the same JSON object) was found in practice to
        make the model default to filling in `response` with a description
        of what it *would* do for any broad or ambiguous request, rather
        than committing to `use_tool` — reliably reproduced across repeated
        attempts with a real target/model, while the two-step decision-only
        schema below did not exhibit it."""
        history = self._history()
        system, exposed = self._build_decision_system(user_msg)
        schema = self.tools.decision_schema(exposed)
        messages = _fit_budget(
            [{"role": "system", "content": system}, *history, {"role": "user", "content": user_msg}],
            self.cfg.context.token_budget,
        )

        trace: list[str] = []
        tool_context: list[dict[str, str]] = []
        for _ in range(self.cfg.context.max_tool_steps):
            try:
                raw = self.llm.complete(messages, json_schema=schema)
            except Exception as exc:  # network/backend errors must not crash the caller
                response = f"ERROR: LLM call failed: {exc}"
                self.store.add_turn(self.session_id, "user", user_msg)
                self.store.add_turn(self.session_id, "assistant", response)
                return TurnResult(response=response, tool_trace=trace)
            try:
                action = parse_json_relaxed(raw)
            except (ValueError, json.JSONDecodeError):
                # One retry with an explicit correction — cheap and usually enough.
                messages.append({"role": "user",
                                 "content": "Invalid JSON. Reply with ONLY the JSON object."})
                continue

            if action.get("action") != "use_tool" or "tool" not in action:
                return self._answer(user_msg, history, tool_context, trace)

            tool = self.tools.get(action.get("tool", ""))
            if tool is None:
                messages.append({"role": "user",
                                 "content": f"Unknown tool {action.get('tool')!r}. "
                                            f"Choose from the listed tools or respond."})
                continue

            args = action.get("arguments") or {}
            try:
                result = tool.run(**args)
            except TypeError as exc:
                result = f"ERROR: bad arguments: {exc}"
            except Exception as exc:  # tool errors go back to the model, not up
                result = f"ERROR: {exc}"

            result = result[: self.cfg.context.max_tool_output_chars]
            trace.append(f"{tool.name} -> {result[:120]}")
            # This turn's tool trace lives in messages only; it is NOT persisted
            # to history, so it never bloats future turns.
            messages.append({"role": "assistant", "content": json.dumps(action)})
            messages.append({"role": "user", "content": f"Tool result:\n{result}"})
            tool_context.append({"role": "assistant", "content": f"Ran {tool.name}({args})."})
            tool_context.append({"role": "user", "content": f"Tool result:\n{result}"})

        # Ran out of steps — persist a summary line, not the full trace.
        summary = "I hit the tool-step limit. Trace: " + "; ".join(trace[-3:])
        self.store.add_turn(self.session_id, "user", user_msg)
        self.store.add_turn(self.session_id, "assistant", summary)
        return TurnResult(response=summary, tool_trace=trace)

    def _answer(self, user_msg: str, history: list[dict[str, str]],
                tool_context: list[dict[str, str]], trace: list[str]) -> TurnResult:
        system = self._build_answer_system(user_msg)
        messages = _fit_budget(
            [{"role": "system", "content": system}, *history,
             {"role": "user", "content": user_msg}, *tool_context],
            self.cfg.context.token_budget,
        )
        try:
            response = self.llm.complete(messages).strip() or "(empty response)"
        except Exception as exc:
            response = f"ERROR: LLM call failed: {exc}"
        self.store.add_turn(self.session_id, "user", user_msg)
        self.store.add_turn(self.session_id, "assistant", response)
        return TurnResult(response=response, tool_trace=trace)

    # ------------------------------------------------------------------
    # Streaming interface (mist tui)
    # ------------------------------------------------------------------
    async def astream_turn(self, user_msg: str, force_think: bool = False) -> AsyncIterator[TurnEvent]:
        """Async generator form of a turn: yields TurnEvents as they happen so
        a caller (the TUI) can render tokens live and — since this is a plain
        asyncio coroutine — cancel it cleanly (Ctrl+C) at any await point.

        `force_think` opts the (normally non-reasoning) tool-decision calls
        back into full reasoning for this turn specifically — see
        astream_mission's stuck-recovery escalation for why."""
        history = self._history()
        # Context assembly can make a network call (embedding fallback in
        # skill routing); push it to a thread so a slow/unreachable embedding
        # backend can't freeze the UI or block a pending interrupt.
        system, exposed = await asyncio.to_thread(self._build_decision_system, user_msg)
        schema = self.tools.decision_schema(exposed)
        messages = _fit_budget(
            [{"role": "system", "content": system}, *history, {"role": "user", "content": user_msg}],
            self.cfg.context.token_budget,
        )

        trace: list[str] = []
        # Natural-language record of this turn's tool calls/results, separate
        # from `messages` (which carries raw JSON action blobs meant for the
        # constrained decision schema). Threaded into `_stream_answer` so the
        # final worded reply is grounded in what the tools actually returned,
        # instead of being generated blind to its own actions this turn.
        tool_context: list[dict[str, str]] = []
        for _ in range(self.cfg.context.max_tool_steps):
            action = None
            last_error: Exception | None = None
            for _attempt in range(MAX_DECISION_RETRIES):
                try:
                    raw = await self.llm.acomplete(messages, json_schema=schema,
                                                   force_think=force_think)
                except Exception as exc:  # network/backend errors: retry, don't crash
                    last_error = exc
                    continue
                try:
                    action = parse_json_relaxed(raw)
                    break
                except (ValueError, json.JSONDecodeError):
                    messages.append({"role": "assistant", "content": raw})
                    messages.append({"role": "user",
                                     "content": "Invalid JSON. Reply with ONLY the JSON object."})
            if action is None:
                if last_error is not None:
                    yield TurnEvent(kind="error", text=f"LLM call failed: {last_error}")
                else:
                    yield TurnEvent(kind="error", text="Model failed to produce valid JSON.")
                return

            if action.get("action") != "use_tool" or "tool" not in action:
                async for event in self._stream_answer(user_msg, history, tool_context):
                    yield event
                return

            tool = self.tools.get(action.get("tool", ""))
            if tool is None:
                messages.append({"role": "user",
                                 "content": f"Unknown tool {action.get('tool')!r}. "
                                            f"Choose from the listed tools or respond."})
                continue

            args = action.get("arguments") or {}
            yield TurnEvent(kind="tool_start", tool=tool.name, detail=json.dumps(args))
            try:
                result = await asyncio.to_thread(tool.run, **args)
            except TypeError as exc:
                result = f"ERROR: bad arguments: {exc}"
            except Exception as exc:  # tool errors go back to the model, not up
                result = f"ERROR: {exc}"

            result = result[: self.cfg.context.max_tool_output_chars]
            trace.append(f"{tool.name} -> {result[:120]}")
            yield TurnEvent(kind="tool_result", tool=tool.name, text=result)

            messages.append({"role": "assistant", "content": json.dumps(action)})
            messages.append({"role": "user", "content": f"Tool result:\n{result}"})
            tool_context.append({"role": "assistant", "content": f"Ran {tool.name}({args})."})
            tool_context.append({"role": "user", "content": f"Tool result:\n{result}"})

        summary = "I hit the tool-step limit. Trace: " + "; ".join(trace[-3:])
        self.store.add_turn(self.session_id, "user", user_msg)
        self.store.add_turn(self.session_id, "assistant", summary)
        yield TurnEvent(kind="done", text=summary)

    async def _stream_answer(self, user_msg: str, history: list[dict[str, str]],
                              tool_context: list[dict[str, str]] | None = None
                              ) -> AsyncIterator[TurnEvent]:
        system = await asyncio.to_thread(self._build_answer_system, user_msg)
        messages = _fit_budget(
            [{"role": "system", "content": system}, *history,
             {"role": "user", "content": user_msg}, *(tool_context or [])],
            self.cfg.context.token_budget,
        )
        chunks: list[str] = []
        try:
            async for delta in self.llm.astream(messages):
                if not delta:
                    continue
                chunks.append(delta)
                yield TurnEvent(kind="delta", text=delta)
        except Exception as exc:  # network/backend errors must not crash the caller
            partial = "".join(chunks).strip()
            note = f"[interrupted: LLM call failed mid-stream: {exc}]"
            response = f"{partial}\n{note}" if partial else f"ERROR: LLM call failed mid-stream: {exc}"
            self.store.add_turn(self.session_id, "user", user_msg)
            self.store.add_turn(self.session_id, "assistant", response)
            yield TurnEvent(kind="error", text=f"LLM call failed mid-stream: {exc}")
            return
        response = "".join(chunks).strip() or "(empty response)"
        self.store.add_turn(self.session_id, "user", user_msg)
        self.store.add_turn(self.session_id, "assistant", response)
        yield TurnEvent(kind="done", text=response)

    # ------------------------------------------------------------------
    # Mission mode: autonomous, multi-turn pursuit of one objective.
    # ------------------------------------------------------------------
    def _mission_log_path(self, mission_id: str) -> Path:
        return self.cfg.wiki_root / "missions" / f"{mission_id}.md"

    def _mission_log_init(self, path: Path, objective: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"# Mission log — session {self.session_id}\n\n"
            f"**Objective:** {objective}\n"
            f"**Started:** {_now()} UTC\n\n"
            "Auto-generated, append-only transcript of every tool call/result this mission "
            "made — a guaranteed audit trail, independent of whether the model also chose to "
            "`remember` or `write_file` a curated page under entities/ or concepts/.\n",
            encoding="utf-8",
        )

    def _mission_log_append(self, path: Path, entry: str) -> None:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(entry)

    def debrief_mission(self, objective: str, status: str, log_path: Path) -> DebriefResult:
        """Deterministically persists whatever a mission accomplished —
        called at every end state (finished, limit hit, or killed) rather
        than left to the model's own initiative, which in practice never
        once called remember/write_file/write_skill across real runs."""
        write_skill_tool = self.tools.get("write_skill")
        write_skill_fn = write_skill_tool.run if write_skill_tool is not None else None
        debriefer = MissionDebriefer(self.llm, self.store, self.cfg.wiki_root, write_skill_fn)
        return debriefer.debrief(objective, status, log_path)

    async def astream_mission(self, objective: str, control: MissionControl,
                               max_turns: int | None = None,
                               max_seconds: float | None = None,
                               stuck_repeat_threshold: int | None = None,
                               ) -> AsyncIterator[MissionEvent]:
        """Repeatedly drives astream_turn, treating each "respond" as an
        interim status rather than a stopping point, and synthesizing the
        next turn itself until the model calls `finish_objective`, a safety
        limit is hit, or a repeated-tool-call loop is detected. Pausing
        (``control``) is checked only between turns, so an in-flight tool
        call always finishes cleanly; killing a mission outright is the
        caller's job — cancel the asyncio task driving this generator."""
        max_turns = self.cfg.mission.max_turns if max_turns is None else max_turns
        max_seconds = self.cfg.mission.max_seconds if max_seconds is None else max_seconds
        stuck_threshold = (self.cfg.mission.stuck_repeat_threshold
                           if stuck_repeat_threshold is None else stuck_repeat_threshold)

        mission_id = f"{self.session_id}-{int(time.time())}"
        log_path = self._mission_log_path(mission_id)
        self._mission_log_init(log_path, objective)
        # Lets a driver (the TUI) recover the log path after killing this
        # generator, since a kill runs debrief_mission separately rather
        # than from inside the cancelled generator (see mist.tui.app).
        yield MissionEvent(kind="started", text=str(log_path))

        self.always_exposed.add("finish_objective")
        try:
            start = time.monotonic()
            turns = 0
            user_msg = objective
            last_signature: str | None = None
            occurrences = 0
            next_turn_think = False
            recovered_once = False

            while True:
                await control.wait_while_paused()

                turns += 1
                yield MissionEvent(kind="turn_start", text=f"turn {turns}")
                use_think = next_turn_think
                next_turn_think = False
                finished = False
                stuck = False
                error_text: str | None = None
                async for event in self.astream_turn(user_msg, force_think=use_think):
                    yield MissionEvent(kind=event.kind, text=event.text,
                                       tool=event.tool, detail=event.detail)
                    if event.kind == "tool_start":
                        self._mission_log_append(
                            log_path, f"\n### {_now()} — {event.tool}\n**args:** `{event.detail}`\n"
                        )
                        if event.tool == "finish_objective":
                            finished = True
                        sig = f"{event.tool}:{event.detail}"
                        occurrences = occurrences + 1 if sig == last_signature else 1
                        last_signature = sig
                        if occurrences >= stuck_threshold:
                            # Break out of astream_turn's own tool loop
                            # immediately — a single turn can run up to
                            # max_tool_steps tool calls on its own, so
                            # checking only after the turn ends would let a
                            # runaway repeat hammer the target far more than
                            # stuck_threshold times before ever catching it.
                            stuck = True
                            break
                    elif event.kind == "tool_result":
                        self._mission_log_append(log_path, f"```\n{event.text}\n```\n")
                    elif event.kind == "error":
                        # A crashed LLM/network call ends astream_turn's own
                        # generator, but the mission loop must not just spin
                        # into an immediate retry — a dead backend would
                        # otherwise burn through the whole turn budget in
                        # seconds. Treat it like a stuck loop: pause and let
                        # the operator see what happened.
                        error_text = event.text

                if finished:
                    self._mission_log_append(
                        log_path, f"\n**Finished** at {_now()}: objective complete.\n"
                    )
                    yield MissionEvent(kind="finished", text="Objective marked complete.")
                    debrief = await asyncio.to_thread(
                        self.debrief_mission, objective, "completed successfully", log_path
                    )
                    yield MissionEvent(kind="debrief", text=debrief.summary())
                    return

                if stuck:
                    if not recovered_once:
                        # First time stuck on this streak: give it one
                        # genuine reasoning pass instead of immediately
                        # pausing for an operator. Every stuck-loop observed
                        # in real runs was fixed by a human saying "stop and
                        # actually think about why this isn't working" —
                        # this is that same nudge, but resolved by the model
                        # itself rather than requiring a human every time.
                        self._mission_log_append(
                            log_path,
                            f"\n**Recovering** at {_now()}: repeated the same tool call "
                            f"{occurrences}x in a row ({last_signature}) — attempting a "
                            "reasoning-assisted recovery before pausing.\n",
                        )
                        yield MissionEvent(
                            kind="recovering",
                            text=(f"Repeated the same tool call {occurrences}x in a row "
                                  f"({last_signature}) — reasoning through a different "
                                  "approach before giving up."),
                        )
                        nudge = (f"You've repeated the same tool call ({last_signature}) "
                                f"{occurrences} times in a row with no new result. Stop and "
                                "actually think through why this specific approach isn't "
                                "working, then commit to a genuinely different next step — "
                                "not a minor variation of the same command.")
                        user_msg = _mission_continue_message(objective, control.pop_notes(), nudge)
                        occurrences = 0
                        recovered_once = True
                        next_turn_think = True
                        continue

                    # Already tried a reasoning-assisted recovery for this
                    # streak and got stuck again — that didn't work either,
                    # so this now genuinely needs an operator.
                    control.pause()
                    self._mission_log_append(
                        log_path,
                        f"\n**Paused** at {_now()}: still stuck after a reasoning-assisted "
                        f"recovery attempt — repeated the same tool call {occurrences}x in "
                        f"a row again ({last_signature}).\n",
                    )
                    yield MissionEvent(
                        kind="stuck",
                        text=(f"Still stuck after trying to reason through it — repeated "
                              f"the same tool call {occurrences}x in a row again "
                              f"({last_signature}) — paused for operator review."),
                    )
                    nudge = (f"You've repeated the same tool call ({last_signature}) "
                            f"{occurrences} times in a row with no new result — that approach "
                            "isn't working. Try something different, or explain what you're "
                            "blocked on if you need the operator's judgment.")
                    user_msg = _mission_continue_message(objective, control.pop_notes(), nudge)
                    occurrences = 0
                    recovered_once = False
                    continue

                if error_text is not None:
                    control.pause()
                    self._mission_log_append(
                        log_path, f"\n**Paused** at {_now()}: turn errored: {error_text}\n"
                    )
                    yield MissionEvent(
                        kind="stuck",
                        text=f"Turn failed ({error_text}) — paused for operator review.",
                    )
                    user_msg = _mission_continue_message(
                        objective, control.pop_notes(),
                        nudge=f"Your previous attempt failed: {error_text}. Try again.",
                    )
                    continue

                if turns >= max_turns:
                    self._mission_log_append(
                        log_path, f"\n**Stopped** at {_now()}: hit the {max_turns}-turn limit.\n"
                    )
                    yield MissionEvent(kind="finished", text=f"Hit the {max_turns}-turn mission limit.")
                    debrief = await asyncio.to_thread(
                        self.debrief_mission, objective, f"hit the {max_turns}-turn limit", log_path
                    )
                    yield MissionEvent(kind="debrief", text=debrief.summary())
                    return

                if (time.monotonic() - start) >= max_seconds:
                    self._mission_log_append(
                        log_path, f"\n**Stopped** at {_now()}: hit the {max_seconds:.0f}s time limit.\n"
                    )
                    yield MissionEvent(kind="finished",
                                       text=f"Hit the {max_seconds:.0f}s mission time limit.")
                    debrief = await asyncio.to_thread(
                        self.debrief_mission, objective,
                        f"hit the {max_seconds:.0f}s time limit", log_path
                    )
                    yield MissionEvent(kind="debrief", text=debrief.summary())
                    return

                # Made it through a normal turn — any future stuck streak is
                # a fresh problem and deserves its own recovery attempt.
                recovered_once = False
                user_msg = _mission_continue_message(objective, control.pop_notes())
        finally:
            self.always_exposed.discard("finish_objective")
