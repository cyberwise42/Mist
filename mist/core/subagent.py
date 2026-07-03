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

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from mist.llm.client import LLMClient, parse_json_relaxed

SUBAGENT_SYSTEM = """You are a focused subagent handling one isolated subtask.
Answer only the subtask below, with no other context. Reply with a single JSON
object: {"response": "<your answer>"}"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"response": {"type": "string"}},
    "required": ["response"],
    "additionalProperties": False,
}


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


def format_results(results: list[SubagentResult], elapsed: float) -> str:
    lines = [f"Ran {len(results)} subagent(s) in {elapsed:.1f}s (parallel):"]
    for i, r in enumerate(results, 1):
        if r.error:
            lines.append(f"{i}. [ERROR] {r.task!r}: {r.error}")
        else:
            lines.append(f"{i}. {r.task!r} -> {r.response}")
    return "\n".join(lines)
