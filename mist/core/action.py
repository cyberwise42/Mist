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

This does not change the existing fallback behavior (a schema-deviant
action still falls through to "respond"/free-text generation, exactly as
before) — it only makes that fallback an explicit, named case instead of
an accidental side effect of loose dict access, and gives every call site
a real `ActionValidationError` with the raw dict attached for logging.

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


def parse_decision_action(raw: dict) -> DecisionRespond | DecisionUseTool:
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
