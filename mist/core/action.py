"""Pydantic-validated parsing of the model's decision/action JSON — the
step every tool dispatch and mission-control decision depends on being
correct.

Previously, the dict returned by `parse_json_relaxed` was consumed via
duck-typed access (`action.get("action") == "respond" or "tool" not in
action`), which can't distinguish "the model chose to respond" from "the
model produced a JSON object that doesn't match either expected shape at
all." Confirmed live: a subagent emitted `{"action": "finish_objective",
"summary": "..."}` (matching neither the documented "respond" nor
"use_tool" contract), and the duck-typed check silently treated it as an
ill-formed "respond" with no response text — discarding the only
diagnostic clue as to what actually went wrong, which is why the parent
mission just blindly retried the identical call three times before a
stuck-detector caught it.

A genuinely unrecognizable action still falls through to "respond"/free-text
generation with a real `ActionValidationError` (raw dict attached) — an
explicit, named case rather than the old accidental side effect of loose dict
access. But one specific, dominant malformation is now *repaired* rather than
discarded: a non-reasoning/opening decision reliably puts the TOOL NAME in
`action` with the args at top level (`{"action": "shell", "command": ...}`)
instead of `{"action": "use_tool", "tool": "shell", "arguments": {...}}`.
`_coerce_tool_named_action` coerces that to a proper `use_tool`, so a
clearly-intended tool call isn't turned into a narrated plan (the exact
opening-turn mis-route — confirmed across both qwen3.6 models, ~5/5 think-off
decisions emit this shape).

Two schema shapes, matching `ToolRegistry.decision_schema`/`action_schema`
in `mist/tools/registry.py`:

- `DecisionAction` — `turn()`/`astream_turn()`/the mission loop's decision
  step. No bundled `response` text; a separate, unconstrained call
  generates the answer once "respond" is chosen.
- `SubagentAction` — `run_tool_subagents`'s one-call contract, where
  `response` text (when respond) *is* bundled directly into this action.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter, ValidationError


class ActionValidationError(Exception):
    """A model's action JSON parsed as valid JSON but matched neither
    expected shape. Carries the raw dict and the underlying pydantic
    error so a caller can log exactly what was produced."""
    def __init__(self, raw: dict, cause: ValidationError):
        self.raw = raw
        self.cause = cause
        super().__init__(f"Invalid action {raw!r}: {cause}")


# -- decision-only shape (turn / astream_turn / mission loop) --------------

class DecisionRespond(BaseModel):
    action: Literal["respond"]


class DecisionUseTool(BaseModel):
    action: Literal["use_tool"]
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)


DecisionActionT = Annotated[Union[DecisionRespond, DecisionUseTool], Field(discriminator="action")]
_decision_adapter: TypeAdapter = TypeAdapter(DecisionActionT)


def _coerce_tool_named_action(raw: dict) -> dict:
    """Repair the most common malformed decision shape. A non-reasoning (or
    opening-turn) call frequently puts the TOOL NAME directly in `action`, with
    the arguments at top level or under `arguments`, instead of the schema's
    `{"action": "use_tool", "tool": "<name>", "arguments": {...}}`:

        {"action": "shell", "command": "nmap -sV ..."}
        {"action": "finish_objective", "summary": "done"}

    Left un-coerced this fails validation and the loop treats it as "respond"
    (a narrated plan) — the exact opening-turn mis-route. Coerce it to a real
    `use_tool` instead. An unknown tool name still fails downstream
    (`tools.get` -> "Unknown tool", which re-prompts the model), identical to a
    well-formed `use_tool` with a bad tool name. `respond`/`use_tool` actions
    and any shape whose `action` isn't a plain string are returned untouched,
    so they validate (or raise) exactly as before."""
    if not isinstance(raw, dict):
        return raw
    action = raw.get("action")
    if not isinstance(action, str) or action in ("respond", "use_tool"):
        return raw
    tool = raw["tool"] if isinstance(raw.get("tool"), str) else action
    args = raw.get("arguments")
    if not isinstance(args, dict):
        args = {k: v for k, v in raw.items() if k not in ("action", "tool", "arguments")}
    return {"action": "use_tool", "tool": tool, "arguments": args}


def parse_decision_action(raw: dict) -> DecisionRespond | DecisionUseTool:
    raw = _coerce_tool_named_action(raw)
    try:
        return _decision_adapter.validate_python(raw)
    except ValidationError as exc:
        raise ActionValidationError(raw, exc) from exc


# -- combined shape (subagents' one-call action+response contract) ---------

class SubagentRespond(BaseModel):
    action: Literal["respond"]
    response: str = ""


class SubagentUseTool(BaseModel):
    action: Literal["use_tool"]
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)


SubagentActionT = Annotated[Union[SubagentRespond, SubagentUseTool], Field(discriminator="action")]
_subagent_adapter: TypeAdapter = TypeAdapter(SubagentActionT)


def parse_subagent_action(raw: dict) -> SubagentRespond | SubagentUseTool:
    try:
        return _subagent_adapter.validate_python(raw)
    except ValidationError as exc:
        raise ActionValidationError(raw, exc) from exc
