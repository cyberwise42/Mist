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
from dataclasses import dataclass
from typing import AsyncIterator

from mist.config import MistConfig
from mist.llm.client import LLMClient, parse_json_relaxed
from mist.memory.store import MemoryStore
from mist.skills.router import SkillRouter
from mist.tools.registry import ToolRegistry

MAX_TOOL_STEPS = 8
MAX_DECISION_RETRIES = 2

SYSTEM_TEMPLATE = """You are Mist, a precise local agent. Follow instructions exactly.

You must reply with a single JSON object matching this shape:
- To answer the user:   {{"action": "respond", "response": "<your answer>"}}
- To use a tool:        {{"action": "use_tool", "tool": "<name>", "arguments": {{...}}}}

Available tools:
{tools}
{skill_section}{memory_section}Rules:
- Use a tool only when needed. Prefer answering directly.
- One action per reply. No text outside the JSON object."""

DECISION_SYSTEM_TEMPLATE = """You are Mist, a precise local agent. Follow instructions exactly.

Decide exactly one action and reply with a single JSON object matching this shape:
- To answer the user directly: {{"action": "respond"}}
- To use a tool:                {{"action": "use_tool", "tool": "<name>", "arguments": {{...}}}}

Available tools:
{tools}
{skill_section}{memory_section}Rules:
- Use a tool only when needed. Prefer answering directly.
- Do not write the answer itself here — a separate step does that.
- One action per reply. No text outside the JSON object."""

ANSWER_SYSTEM_TEMPLATE = """You are Mist, a precise local agent.
{skill_section}{memory_section}Answer the user's message directly and conversationally.
Reply in plain text only — no JSON, no code fences unless the user asked for code."""


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
                 skills: SkillRouter, tools: ToolRegistry, session_id: int | None = None):
        self.cfg = config
        self.llm = llm
        self.store = store
        self.skills = skills
        self.tools = tools
        self.session_id = session_id or store.new_session()

    # ------------------------------------------------------------------
    def _assemble_context(self, user_msg: str) -> tuple[str, str, str, list]:
        """Builds the three variable sections of every system prompt (tool
        listing, active skill, relevant memories) plus the tools exposed for
        this turn. Shared by every template so tool/skill/memory selection
        logic lives in exactly one place."""
        exposed = self.tools.select(user_msg, self.cfg.tools.max_exposed)
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

    def _build_system(self, user_msg: str) -> tuple[str, list]:
        tool_lines, skill_section, memory_section, exposed = self._assemble_context(user_msg)
        system = SYSTEM_TEMPLATE.format(
            tools=tool_lines, skill_section=skill_section, memory_section=memory_section
        )
        return system, exposed

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
        system, exposed = self._build_system(user_msg)
        schema = self.tools.action_schema(exposed)
        history = self._history()

        messages = [{"role": "system", "content": system}, *history,
                    {"role": "user", "content": user_msg}]
        messages = _fit_budget(messages, self.cfg.context.token_budget)

        trace: list[str] = []
        for _ in range(MAX_TOOL_STEPS):
            raw = self.llm.complete(messages, json_schema=schema)
            try:
                action = parse_json_relaxed(raw)
            except (ValueError, json.JSONDecodeError):
                # One retry with an explicit correction — cheap and usually enough.
                messages.append({"role": "user",
                                 "content": "Invalid JSON. Reply with ONLY the JSON object."})
                continue

            if action.get("action") == "respond" or "tool" not in action:
                response = action.get("response", "").strip() or "(empty response)"
                self.store.add_turn(self.session_id, "user", user_msg)
                self.store.add_turn(self.session_id, "assistant", response)
                return TurnResult(response=response, tool_trace=trace)

            tool = self.tools.get(action.get("tool", ""))
            if tool is None:
                messages.append({"role": "user",
                                 "content": f"Unknown tool {action.get('tool')!r}. "
                                            f"Choose from the listed tools or respond."})
                continue

            try:
                result = tool.run(**(action.get("arguments") or {}))
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

        # Ran out of steps — persist a summary line, not the full trace.
        summary = "I hit the tool-step limit. Trace: " + "; ".join(trace[-3:])
        self.store.add_turn(self.session_id, "user", user_msg)
        self.store.add_turn(self.session_id, "assistant", summary)
        return TurnResult(response=summary, tool_trace=trace)

    # ------------------------------------------------------------------
    # Streaming interface (mist tui)
    # ------------------------------------------------------------------
    async def astream_turn(self, user_msg: str) -> AsyncIterator[TurnEvent]:
        """Async generator form of a turn: yields TurnEvents as they happen so
        a caller (the TUI) can render tokens live and — since this is a plain
        asyncio coroutine — cancel it cleanly (Ctrl+C) at any await point."""
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
        for _ in range(MAX_TOOL_STEPS):
            action = None
            for _attempt in range(MAX_DECISION_RETRIES):
                raw = await self.llm.acomplete(messages, json_schema=schema)
                try:
                    action = parse_json_relaxed(raw)
                    break
                except (ValueError, json.JSONDecodeError):
                    messages.append({"role": "assistant", "content": raw})
                    messages.append({"role": "user",
                                     "content": "Invalid JSON. Reply with ONLY the JSON object."})
            if action is None:
                yield TurnEvent(kind="error", text="Model failed to produce valid JSON.")
                return

            if action.get("action") != "use_tool" or "tool" not in action:
                async for event in self._stream_answer(user_msg, history):
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

        summary = "I hit the tool-step limit. Trace: " + "; ".join(trace[-3:])
        self.store.add_turn(self.session_id, "user", user_msg)
        self.store.add_turn(self.session_id, "assistant", summary)
        yield TurnEvent(kind="done", text=summary)

    async def _stream_answer(self, user_msg: str, history: list[dict[str, str]]
                              ) -> AsyncIterator[TurnEvent]:
        system = await asyncio.to_thread(self._build_answer_system, user_msg)
        messages = _fit_budget(
            [{"role": "system", "content": system}, *history, {"role": "user", "content": user_msg}],
            self.cfg.context.token_budget,
        )
        chunks: list[str] = []
        async for delta in self.llm.astream(messages):
            if not delta:
                continue
            chunks.append(delta)
            yield TurnEvent(kind="delta", text=delta)
        response = "".join(chunks).strip() or "(empty response)"
        self.store.add_turn(self.session_id, "user", user_msg)
        self.store.add_turn(self.session_id, "assistant", response)
        yield TurnEvent(kind="done", text=response)
