"""Mist core agent loop.

The defining behavior: context is REBUILT every turn, never accumulated.

    prompt = system
           + top-k memories (retrieved against the user message)
           + routed skill (at most one full body)
           + last N verbatim turns
           + current tool trace (this turn only)

Tool output from previous turns is summarized into a single line before it
enters history; full outputs are never carried forward.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from mist.config import MistConfig
from mist.llm.client import LLMClient, parse_json_relaxed
from mist.memory.store import MemoryStore
from mist.skills.router import SkillRouter
from mist.tools.registry import ToolRegistry

MAX_TOOL_STEPS = 8

SYSTEM_TEMPLATE = """You are Mist, a precise local agent. Follow instructions exactly.

You must reply with a single JSON object matching this shape:
- To answer the user:   {{"action": "respond", "response": "<your answer>"}}
- To use a tool:        {{"action": "use_tool", "tool": "<name>", "arguments": {{...}}}}

Available tools:
{tools}
{skill_section}{memory_section}Rules:
- Use a tool only when needed. Prefer answering directly.
- One action per reply. No text outside the JSON object."""


def _approx_tokens(text: str) -> int:
    return len(text) // 4  # cheap heuristic; good enough for budgeting


@dataclass
class TurnResult:
    response: str
    tool_trace: list[str]


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
    def _build_system(self, user_msg: str) -> tuple[str, list]:
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

        system = SYSTEM_TEMPLATE.format(
            tools=tool_lines, skill_section=skill_section, memory_section=memory_section
        )
        return system, exposed

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

        # Enforce the token budget by dropping oldest history first.
        while (sum(_approx_tokens(m["content"]) for m in messages)
               > self.cfg.context.token_budget and len(messages) > 2):
            messages.pop(1)

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
