"""Subagent spawning: fan out independent subtasks as concurrent LLM calls.

Ollama serves a given model one request at a time in practice, so spawning
subagents against it just serializes work with extra overhead. vLLM's
continuous batching is built for exactly this case: many concurrent requests
get scheduled onto the GPU together, so wall-clock time for N independent
subtasks can approach the time for one. This module fires subtasks
concurrently over plain threads (httpx.Client's connection pool is
thread-safe) and lets the backend decide how to batch them; it pays off most
under vLLM and is harmless (if not faster) under Ollama.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mist.llm.client import LLMClient, parse_json_relaxed

if TYPE_CHECKING:
    from mist.tools.registry import ToolRegistry

SUBAGENT_SYSTEM = """You are a focused subagent handling one isolated subtask.
Answer only the subtask below, with no other context. Reply with a single JSON
object: {"response": "<your answer>"}"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"response": {"type": "string"}},
    "required": ["response"],
    "additionalProperties": False,
}

TOOL_SUBAGENT_SYSTEM = """You are a focused subagent handling one isolated subtask.
Work only on the subtask below, with no other context.

You must reply with a single JSON object matching this shape:
- To answer:   {{"action": "respond", "response": "<your answer>"}}
- To use a tool: {{"action": "use_tool", "tool": "<name>", "arguments": {{...}}}}

Available tools:
{tools}

Rules:
- Use a tool only when needed to complete the subtask.
- One action per reply. No text outside the JSON object."""


@dataclass
class SubagentResult:
    task: str
    response: str
    error: str | None = None


def run_subagents(llm: LLMClient, tasks: list[str], max_workers: int = 4) -> list[SubagentResult]:
    """Run each task as an isolated one-shot subagent call, concurrently."""

    def _run_one(task: str) -> SubagentResult:
        messages = [
            {"role": "system", "content": SUBAGENT_SYSTEM},
            {"role": "user", "content": task},
        ]
        try:
            raw = llm.complete(messages, json_schema=RESPONSE_SCHEMA)
            parsed = parse_json_relaxed(raw)
            return SubagentResult(task=task, response=parsed.get("response", "").strip())
        except Exception as exc:  # a failed subagent must not crash the parent turn
            return SubagentResult(task=task, response="", error=str(exc))

    if not tasks:
        return []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(tasks))) as pool:
        return list(pool.map(_run_one, tasks))


def run_tool_subagents(llm: LLMClient, tools: "ToolRegistry", tasks: list[str],
                        max_workers: int = 4, max_steps: int = 6) -> list[SubagentResult]:
    """Run each task as an isolated subagent with its own bounded tool loop
    (read_file/write_file/shell/remember — never spawn_subagents or
    write_skill, since ``tools`` is built without ``llm``/``skills``). Kept as
    a standalone loop rather than sharing MistAgent.turn()'s implementation:
    that method is entangled with session persistence and skill/memory
    context assembly a subagent must not touch."""

    def _run_one(task: str) -> SubagentResult:
        # `mission_only` tools (currently just `finish_objective`) are meant
        # to be reachable only through a mission's own decision loop, via
        # `always_exposed` — never through `ToolRegistry.select()`'s normal
        # ranking. `tools.all()` bypasses that filtering entirely, and a real
        # run confirmed the model will still try to call `finish_objective`
        # here if it's offered: it emitted a schema-deviant
        # {"action": "finish_objective", "summary": "..."} instead of this
        # loop's documented {"action": "use_tool", "tool": ..., "arguments":
        # {...}} shape (this schema has no top-level "finish_objective"
        # action, and no "summary" property), which fell through to the
        # empty-response fallback below — twice, in a real mission, wasting
        # ~200s before the parent's own stuck-repeat detector caught it.
        all_tools = [t for t in tools.all() if not t.mission_only]
        tool_lines = "\n".join(
            f"- {t.name}: {t.description} | args schema: {json.dumps(t.parameters['properties'])}"
            for t in all_tools
        )
        system = TOOL_SUBAGENT_SYSTEM.format(tools=tool_lines)
        schema = tools.action_schema(all_tools)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": task},
        ]
        trace: list[str] = []
        try:
            for _ in range(max_steps):
                raw = llm.complete(messages, json_schema=schema)
                try:
                    action = parse_json_relaxed(raw)
                except (ValueError, json.JSONDecodeError):
                    messages.append({"role": "user",
                                     "content": "Invalid JSON. Reply with ONLY the JSON object."})
                    continue

                if action.get("action") == "respond" or "tool" not in action:
                    response = action.get("response", "").strip()
                    if not response:
                        # A bare "(empty response)" here previously discarded
                        # a real, debuggable signal: a model can emit a
                        # schema-deviant action (missing "tool", an
                        # unexpected "action" value) that satisfies neither
                        # branch cleanly. Surfacing the raw action back to
                        # the caller at least gives the parent turn/mission
                        # something to react to instead of nothing — which
                        # is exactly why a real run just blindly retried the
                        # identical call three times: an empty result gave
                        # it no information about what to change.
                        response = ("(subagent produced no usable response; "
                                   f"raw action: {json.dumps(action)[:300]})")
                    return SubagentResult(task=task, response=response)

                tool = tools.get(action.get("tool", ""))
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

                trace.append(f"{tool.name} -> {result[:120]}")
                messages.append({"role": "assistant", "content": json.dumps(action)})
                messages.append({"role": "user", "content": f"Tool result:\n{result}"})

            summary = "Hit the tool-step limit. Trace: " + "; ".join(trace[-3:])
            return SubagentResult(task=task, response=summary)
        except Exception as exc:  # a failed subagent must not crash the parent turn
            return SubagentResult(task=task, response="", error=str(exc))

    if not tasks:
        return []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(tasks))) as pool:
        return list(pool.map(_run_one, tasks))


def format_results(results: list[SubagentResult], elapsed: float) -> str:
    lines = [f"Ran {len(results)} subagent(s) in {elapsed:.1f}s (parallel):"]
    for i, r in enumerate(results, 1):
        if r.error:
            lines.append(f"{i}. [ERROR] {r.task!r}: {r.error}")
        else:
            lines.append(f"{i}. {r.task!r} -> {r.response}")
    return "\n".join(lines)
