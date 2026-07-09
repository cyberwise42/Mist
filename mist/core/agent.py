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
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator

from mist.config import MistConfig
from mist.core.action import ActionValidationError, DecisionUseTool, parse_decision_action
from mist.core.debrief import DebriefResult, MissionDebriefer
from mist.core.mission import MissionControl, MissionEvent
from mist.core.tool_compressor import ToolOutputCompressor
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
- Match your actions to the objective's actual scope, in both directions. A narrow objective
  ("enumerate ports and services", "identify what's running on port X") is fully satisfied once a
  scan's own output already answers it — persist that result and call `finish_objective` right
  then, don't keep going into content discovery, vulnerability scanning, or exploitation the
  objective never asked for. Only continue into those further phases when the objective itself
  calls for full compromise, access, or a flag.
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
- Follow standard methodology order instead of jumping straight to manual requests: after a port
  scan (`nmap -sV -sC`) identifies an open service, fingerprint the exact technology/version, then
  check it for known vulnerabilities (`nmap --script vuln`, `nuclei -u`, `searchsploit <service>
  <version>`) before hand-crafting a `curl`/manual request against it. A raw `curl` against a
  webserver's JS/assets is a fine follow-up once a scan or search has pointed at something
  specific to confirm — it is not the first move against an unscanned target or port.
- Content/route discovery is a scanner's job (`gobuster`, `ffuf`, `dirsearch`), not a series of
  hand-picked `curl` guesses at paths you invented (`/api/v1/...`, `/dashboard`, `/login`, ...). A
  wordlist scan finds real endpoints in seconds; guessing one path at a time rarely finds anything
  a scan wouldn't have and burns turns doing it. More than one or two manual probes to different
  paths on the same host without a scanner call in between is the signal to switch tools, not to
  keep guessing.
- Do not write the answer itself here — a separate step does that.
- One action per reply. No text outside the JSON object."""

ANSWER_SYSTEM_TEMPLATE = """You are Mist, an autonomous penetration-testing operator with real, working
tool access already configured — shell (runs real commands against real targets over the network),
read_file, write_file, search_files, remember, and more. You are not a text-only assistant: never
claim you lack network access, an SSH shell, or the ability to execute commands — that is never true
here, regardless of what you may have said on a prior turn. If you haven't acted yet this turn,
that's a choice about what to do next, not a limitation to explain to the operator.
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


_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_HOSTNAME_RE = re.compile(r'"([A-Za-z][\w-]*)"|\bhost(?:name)?[:\s]+([A-Za-z][\w-]*)',
                          re.IGNORECASE)


def _extract_target(objective: str) -> str | None:
    """Best-effort target identifier for per-mission workspace separation.
    Prefers an IPv4 address — unambiguous, and how HTB objectives in
    practice always name the target ("...at target ip 10.129.33.21") — and
    falls back to a quoted or "host <name>" token, slugified, if no IP is
    present. Returns None if neither is found (the caller then leaves the
    shell's cwd at its configured default rather than guessing)."""
    m = _IPV4_RE.search(objective)
    if m:
        return m.group(0)
    m = _HOSTNAME_RE.search(objective)
    if m:
        return _slugify(m.group(1) or m.group(2))
    return None


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _mission_continue_message(objective: str, notes: list[str], nudge: str | None = None) -> str:
    msg = MISSION_CONTINUE_TEMPLATE.format(objective=objective)
    if nudge:
        msg = f"{nudge}\n\n{msg}"
    if notes:
        msg += "\n\nOperator note(s) since your last step:\n" + "\n".join(f"- {n}" for n in notes)
    return msg


def _findings_recap(findings: list[str], max_items: int = 5) -> str:
    """A compact recap of the most recent tool results this mission, meant
    to be appended to a stuck-recovery nudge specifically. Regression case
    from a real run: after a reasoning-assisted recovery got the model to
    call a real tool again, it re-ran the exact same `nmap` scan it had
    already gotten full results from turns earlier — wasteful, but never
    flagged by any detector, since a different `curl` call in between reset
    every repeat/near-dup signature. The persisted chat history is supposed
    to carry this forward, but relies on the model's own free-text summary
    faithfully restating it every turn — this recap restates the raw
    findings directly at exactly the moment (a nudge) the model is being
    asked to pick a genuinely different next step, so it doesn't need to
    rediscover what it already has."""
    if not findings:
        return ""
    recent = findings[-max_items:]
    lines = "\n".join(f"- {f}" for f in recent)
    return f"\n\nAlready found this mission (don't redo these):\n{lines}"


def _mission_routing_query(objective: str, notes: list[str], nudge: str | None = None) -> str:
    """Same real content as `_mission_continue_message` (objective, nudge,
    operator notes) but WITHOUT MISSION_CONTINUE_TEMPLATE's fixed
    boilerplate — used only to drive tool/skill/memory selection, never
    sent to the model. That boilerplate is identical every turn regardless
    of what the mission is about, and its own wording ("...with a `tool`
    right now"; "search_files the `wiki` for existing reference
    `knowledge`...") gives the llm-wiki skill a keyword-overlap head start
    on every single turn — confirmed live, it beat pentest-methodology for
    objectives as plainly on-topic as "get root on 10.129.30.204"."""
    parts = [nudge] if nudge else []
    parts.append(objective)
    if notes:
        parts.extend(notes)
    return "\n\n".join(parts)


def _approx_tokens(text: str) -> int:
    return len(text) // 4  # cheap heuristic; good enough for budgeting


def _truncate_tool_output(text: str, budget: int, head_ratio: float = 0.3) -> str:
    """Keeps a head slice AND a tail slice, not just the head. Verbose recon
    tools (nuclei, gobuster, nmap) front-load a banner/progress preamble and
    print their actual findings and summary line near the end — a live run's
    `nuclei` scan spent its entire 2000-char budget on template-loading
    banner text and never reached the actual detections or "N matches
    found" line, so the model was reasoning off a banner, not the scan
    results, and had no way to notice a real finding even if there'd been
    one. A plain head cut is fine for output that front-loads what matters
    (a `curl` response body); this is a better default for the noisy CLI
    scanners a pentest harness spends most of its tool calls running."""
    if len(text) <= budget:
        return text
    head_chars = int(budget * head_ratio)
    marker = f"\n...[{len(text) - budget} chars omitted]...\n"
    tail_chars = budget - head_chars - len(marker)
    return text[:head_chars] + marker + text[-tail_chars:]


def _fit_budget(messages: list[dict[str, str]], budget: int) -> list[dict[str, str]]:
    """Enforce the token budget by dropping oldest history first (never the
    system prompt at [0] or the current user message at [-1])."""
    while sum(_approx_tokens(m["content"]) for m in messages) > budget and len(messages) > 2:
        messages.pop(1)
    return messages


_QUOTED_RE = re.compile(r"'[^']*'|\"[^\"]*\"")
# Tries a full IPv4 dotted-quad first at each position; only falls back to a
# bare digit run if that didn't match. Doing this as one alternation (rather
# than stripping digits and separately protecting IPs) avoids the digit-strip
# pass clobbering a placeholder that itself contains digits.
_NUMERIC_RE = re.compile(r"(?:\d{1,3}\.){3}\d{1,3}|\d+")


def _normalize_value(v: str) -> str:
    v = _QUOTED_RE.sub("‹Q›", v)
    # IPv4 addresses are left untouched — two calls against two different
    # real target hosts must never collapse to the same signature just
    # because both targets happen to be numeric.
    v = _NUMERIC_RE.sub(lambda m: m.group(0) if "." in m.group(0) else "‹N›", v)
    return v


# Argument keys whose entire value IS the "attempt" being varied, the same
# role a quoted grep pattern or context-line count plays inside a shell
# command — so the value is wildcarded outright rather than partially
# normalized. `search_files`'s `query` is the motivating case: a real mission
# spammed it a dozen times with a different plain-text query each call
# ("Next.js", "ReactorWatch", "Server Action", ...), treating the wiki's own
# full-text search as a vulnerability database. None of those values contain
# a quote or a digit, so _normalize_value alone leaves them all distinct.
_WILDCARD_KEYS = {"query"}


def _normalize_tool_call(tool: str, args_json: str) -> str:
    """Coarse signature for near-duplicate detection: wildcards whole
    "attempt" arguments (see `_WILDCARD_KEYS`) and strips quoted string
    literals (grep patterns, ...) and standalone integers (context-line
    counts, offsets, ...) out of the rest, so calls that only vary in the
    part of the command actually being "tried again" collapse to the same
    signature. The exact stuck-repeat check above requires byte-identical
    calls and never catches this.

    Deliberately coarser than the exact check and thus given a higher
    threshold (`near_duplicate_threshold` > `stuck_repeat_threshold`) — it
    will not catch every variation (e.g. swapping `-A` for `-C` changes the
    kept flag token, not just its value), but catches the dominant real
    pattern: re-probing the same target with only a literal/number changed."""
    try:
        args = json.loads(args_json) if args_json else {}
    except (ValueError, TypeError):
        args = None
    if not isinstance(args, dict):
        return f"{tool}:{args_json}"
    parts = []
    for k in sorted(args):
        v = args[k]
        if k in _WILDCARD_KEYS:
            v = "‹*›"
        elif isinstance(v, str):
            v = _normalize_value(v)
        parts.append(f"{k}={v}")
    return f"{tool}:" + ",".join(parts)


_MANUAL_PROBE_RE = re.compile(r"^\s*(?:sudo\s+)?(?:curl|wget)\b", re.IGNORECASE)


def _is_manual_probe(tool: str, args_json: str) -> bool:
    """True for a `shell` call that's a single ad hoc HTTP request (`curl`/
    `wget`) — the tool class a real mission used, correctly-shaped and
    genuinely different every time (a different path each call: `/login`,
    `/dashboard`, `/api/v1/reports/generate.json`, ...), to hand-guess at
    content discovery instead of running a scanner. Neither the exact-repeat
    nor near-duplicate check above can catch this: every call really is a
    distinct, individually reasonable-looking command, so no signature ever
    repeats. This is a separate, coarser signal — a streak of this *tool
    class* in a row, regardless of what varies inside it — tracked by
    `astream_mission` alongside (not instead of) the other two."""
    if tool != "shell":
        return False
    try:
        args = json.loads(args_json) if args_json else {}
    except (ValueError, TypeError):
        return False
    command = args.get("command", "") if isinstance(args, dict) else ""
    return bool(_MANUAL_PROBE_RE.match(command))


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
                 process_registry: ProcessRegistry | None = None,
                 tool_compressor: ToolOutputCompressor | None = None):
        self.cfg = config
        self.llm = llm
        self.store = store
        self.skills = skills
        self.tools = tools
        self.session_id = session_id or store.new_session()
        # Tier 3 of tool-output handling (mist/core/tool_compressor.py): an
        # optional, independently-configured aux model that compresses
        # long tool output tiers 1-2 don't already fit. None by default —
        # most tool output already fits after tiers 1-2, so this is opt-in.
        self.tool_compressor = tool_compressor
        # Force-included regardless of keyword ranking — set by astream_mission
        # for the duration of a mission so `finish_objective` is always
        # reachable even when a turn's message shares no keywords with it.
        self.always_exposed: set[str] = set()
        # Lets a driver (the TUI) actually terminate a running shell command
        # on kill — see mist.tools.registry.ProcessRegistry for why asyncio
        # task cancellation alone can't do that.
        self.process_registry = process_registry
        # Real token count of the last assembled prompt (via _fit_budget,
        # not a synthetic estimate) — surfaced by the TUI's status line.
        self.last_context_tokens = 0

    # ------------------------------------------------------------------
    def _assemble_context(self, user_msg: str, routing_query: str | None = None
                          ) -> tuple[str, str, str, list]:
        """Builds the three variable sections of every system prompt (tool
        listing, active skill, relevant memories) plus the tools exposed for
        this turn. Shared by every template so tool/skill/memory selection
        logic lives in exactly one place.

        `routing_query` (defaults to `user_msg`) is what actually drives
        tool/skill/memory selection — kept separate from `user_msg` because
        a mission's `user_msg` is MISSION_CONTINUE_TEMPLATE's boilerplate
        wrapped around the real objective, and that fixed boilerplate text
        ("...with a `tool` right now"; "search_files the `wiki` for
        existing reference `knowledge`...") shares keywords with the
        llm-wiki skill on every single turn regardless of what the mission
        is actually about — confirmed live: it won the routing race over
        pentest-methodology for short objectives like "get root on
        10.129.30.204" purely on that boilerplate overlap, never letting
        the actually-relevant skill's full body load."""
        routing_query = user_msg if routing_query is None else routing_query
        exposed = self.tools.select(routing_query, self.cfg.tools.max_exposed,
                                    always=self.always_exposed)
        tool_lines = "\n".join(
            f"- {t.name}: {t.description} | args schema: {json.dumps(t.parameters['properties'])}"
            for t in exposed
        )

        skill_section = ""
        candidates = self.skills.route(routing_query, self.cfg.skills.max_candidates)
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
        memories = self.store.search(routing_query, self.cfg.memory.top_k)
        if memories:
            memory_section = "\nRelevant memories:\n" + "\n".join(f"- {m}" for m in memories) + "\n"

        return tool_lines, skill_section, memory_section, exposed

    def _build_decision_system(self, user_msg: str, routing_query: str | None = None
                               ) -> tuple[str, list]:
        tool_lines, skill_section, memory_section, exposed = self._assemble_context(
            user_msg, routing_query
        )
        system = DECISION_SYSTEM_TEMPLATE.format(
            tools=tool_lines, skill_section=skill_section, memory_section=memory_section
        )
        return system, exposed

    def _build_answer_system(self, user_msg: str, routing_query: str | None = None) -> str:
        _, skill_section, memory_section, _ = self._assemble_context(user_msg, routing_query)
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
        self.last_context_tokens = sum(_approx_tokens(m["content"]) for m in messages)

        trace: list[str] = []
        tool_context: list[dict[str, str]] = []
        for _ in range(self.cfg.context.max_tool_steps):
            try:
                raw = self.llm.complete(messages, json_schema=schema)
            except Exception as exc:  # network/backend errors must not crash the caller
                # repr(), not str() — some low-level connection exceptions
                # (a dropped socket, a reset connection) have an empty
                # message, and str() alone renders as a bare "LLM call
                # failed: " with zero diagnostic value. Confirmed live:
                # this happened three separate times in one real mission
                # with no clue as to which exception actually fired.
                response = f"ERROR: LLM call failed: {exc!r}"
                self.store.add_turn(self.session_id, "user", user_msg)
                self.store.add_turn(self.session_id, "assistant", response)
                return TurnResult(response=response, tool_trace=trace)
            try:
                raw_action = parse_json_relaxed(raw)
            except (ValueError, json.JSONDecodeError):
                # One retry with an explicit correction — cheap and usually enough.
                messages.append({"role": "user",
                                 "content": "Invalid JSON. Reply with ONLY the JSON object."})
                continue

            try:
                action = parse_decision_action(raw_action)
            except ActionValidationError:
                # Schema-deviant (parseable JSON, but neither "respond" nor a
                # well-formed "use_tool") — nothing tool-shaped to dispatch,
                # same fallback as an explicit "respond" (unchanged from
                # before this was a named, validated case).
                return self._answer(user_msg, history, tool_context, trace)

            if not isinstance(action, DecisionUseTool):
                return self._answer(user_msg, history, tool_context, trace)

            tool = self.tools.get(action.tool)
            if tool is None:
                messages.append({"role": "user",
                                 "content": f"Unknown tool {action.tool!r}. "
                                            f"Choose from the listed tools or respond."})
                continue

            args = action.arguments
            try:
                result = tool.run(**args)
            except TypeError as exc:
                result = f"ERROR: bad arguments: {exc}"
            except Exception as exc:  # tool errors go back to the model, not up
                result = f"ERROR: {exc}"

            if self.tool_compressor is not None:
                result = self.tool_compressor.maybe_compress(result)
            result = _truncate_tool_output(result, self.cfg.context.max_tool_output_chars)
            trace.append(f"{tool.name} -> {result[:120]}")
            # This turn's tool trace lives in messages only; it is NOT persisted
            # to history, so it never bloats future turns.
            messages.append({"role": "assistant", "content": action.model_dump_json()})
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
        self.last_context_tokens = sum(_approx_tokens(m["content"]) for m in messages)
        try:
            response = self.llm.complete(messages).strip() or "(empty response)"
        except Exception as exc:
            response = f"ERROR: LLM call failed: {exc!r}"
        self.store.add_turn(self.session_id, "user", user_msg)
        self.store.add_turn(self.session_id, "assistant", response)
        return TurnResult(response=response, tool_trace=trace)

    # ------------------------------------------------------------------
    # Streaming interface (mist tui)
    # ------------------------------------------------------------------
    async def astream_turn(self, user_msg: str, force_think: bool = False,
                           routing_query: str | None = None) -> AsyncIterator[TurnEvent]:
        """Async generator form of a turn: yields TurnEvents as they happen so
        a caller (the TUI) can render tokens live and — since this is a plain
        asyncio coroutine — cancel it cleanly (Ctrl+C) at any await point.

        `force_think` opts the (normally non-reasoning) tool-decision calls
        back into full reasoning for this turn specifically — see
        astream_mission's stuck-recovery escalation for why.

        `routing_query` (see `_assemble_context`) lets a mission drive
        tool/skill/memory selection off the bare objective instead of the
        full boilerplate-wrapped continue-message."""
        history = self._history()
        # Context assembly can make a network call (embedding fallback in
        # skill routing); push it to a thread so a slow/unreachable embedding
        # backend can't freeze the UI or block a pending interrupt.
        system, exposed = await asyncio.to_thread(
            self._build_decision_system, user_msg, routing_query
        )
        schema = self.tools.decision_schema(exposed)
        messages = _fit_budget(
            [{"role": "system", "content": system}, *history, {"role": "user", "content": user_msg}],
            self.cfg.context.token_budget,
        )
        self.last_context_tokens = sum(_approx_tokens(m["content"]) for m in messages)

        trace: list[str] = []
        # Natural-language record of this turn's tool calls/results, separate
        # from `messages` (which carries raw JSON action blobs meant for the
        # constrained decision schema). Threaded into `_stream_answer` so the
        # final worded reply is grounded in what the tools actually returned,
        # instead of being generated blind to its own actions this turn.
        tool_context: list[dict[str, str]] = []
        for _ in range(self.cfg.context.max_tool_steps):
            raw_action = None
            last_error: Exception | None = None
            for _attempt in range(MAX_DECISION_RETRIES):
                try:
                    raw = await self.llm.acomplete(messages, json_schema=schema,
                                                   force_think=force_think)
                except Exception as exc:  # network/backend errors: retry, don't crash
                    last_error = exc
                    continue
                try:
                    raw_action = parse_json_relaxed(raw)
                    break
                except (ValueError, json.JSONDecodeError):
                    messages.append({"role": "assistant", "content": raw})
                    messages.append({"role": "user",
                                     "content": "Invalid JSON. Reply with ONLY the JSON object."})
            if raw_action is None:
                if last_error is not None:
                    yield TurnEvent(kind="error", text=f"LLM call failed: {last_error!r}")
                else:
                    yield TurnEvent(kind="error", text="Model failed to produce valid JSON.")
                return

            try:
                action = parse_decision_action(raw_action)
            except ActionValidationError:
                # Schema-deviant (parseable JSON, but neither "respond" nor a
                # well-formed "use_tool") — same fallback as an explicit
                # "respond" (unchanged from before this was a named,
                # validated case).
                async for event in self._stream_answer(user_msg, history, tool_context,
                                                       routing_query):
                    yield event
                return

            if not isinstance(action, DecisionUseTool):
                async for event in self._stream_answer(user_msg, history, tool_context,
                                                       routing_query):
                    yield event
                return

            tool = self.tools.get(action.tool)
            if tool is None:
                messages.append({"role": "user",
                                 "content": f"Unknown tool {action.tool!r}. "
                                            f"Choose from the listed tools or respond."})
                continue

            args = action.arguments
            yield TurnEvent(kind="tool_start", tool=tool.name, detail=json.dumps(args))
            try:
                result = await asyncio.to_thread(tool.run, **args)
            except TypeError as exc:
                result = f"ERROR: bad arguments: {exc}"
            except Exception as exc:  # tool errors go back to the model, not up
                result = f"ERROR: {exc}"

            if self.tool_compressor is not None:
                # Keeps registry.py's tool functions synchronous and
                # network-free (their existing "plain functions" design) —
                # the compressor's own LLM call goes through the same
                # asyncio.to_thread pattern already used for tool.run itself.
                result = await asyncio.to_thread(self.tool_compressor.maybe_compress, result)
            result = _truncate_tool_output(result, self.cfg.context.max_tool_output_chars)
            trace.append(f"{tool.name} -> {result[:120]}")
            yield TurnEvent(kind="tool_result", tool=tool.name, text=result)

            messages.append({"role": "assistant", "content": action.model_dump_json()})
            messages.append({"role": "user", "content": f"Tool result:\n{result}"})
            tool_context.append({"role": "assistant", "content": f"Ran {tool.name}({args})."})
            tool_context.append({"role": "user", "content": f"Tool result:\n{result}"})

        summary = "I hit the tool-step limit. Trace: " + "; ".join(trace[-3:])
        self.store.add_turn(self.session_id, "user", user_msg)
        self.store.add_turn(self.session_id, "assistant", summary)
        yield TurnEvent(kind="done", text=summary)

    async def _stream_answer(self, user_msg: str, history: list[dict[str, str]],
                              tool_context: list[dict[str, str]] | None = None,
                              routing_query: str | None = None
                              ) -> AsyncIterator[TurnEvent]:
        system = await asyncio.to_thread(self._build_answer_system, user_msg, routing_query)
        messages = _fit_budget(
            [{"role": "system", "content": system}, *history,
             {"role": "user", "content": user_msg}, *(tool_context or [])],
            self.cfg.context.token_budget,
        )
        self.last_context_tokens = sum(_approx_tokens(m["content"]) for m in messages)
        chunks: list[str] = []
        try:
            async for delta in self.llm.astream(messages):
                if not delta:
                    continue
                chunks.append(delta)
                yield TurnEvent(kind="delta", text=delta)
        except Exception as exc:  # network/backend errors must not crash the caller
            partial = "".join(chunks).strip()
            note = f"[interrupted: LLM call failed mid-stream: {exc!r}]"
            response = f"{partial}\n{note}" if partial else f"ERROR: LLM call failed mid-stream: {exc!r}"
            self.store.add_turn(self.session_id, "user", user_msg)
            self.store.add_turn(self.session_id, "assistant", response)
            yield TurnEvent(kind="error", text=f"LLM call failed mid-stream: {exc!r}")
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
                               near_duplicate_threshold: int | None = None,
                               manual_probe_threshold: int | None = None,
                               respond_streak_threshold: int | None = None,
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
        near_dup_threshold = (self.cfg.mission.near_duplicate_threshold
                              if near_duplicate_threshold is None else near_duplicate_threshold)
        manual_probe_threshold = (self.cfg.mission.manual_probe_threshold
                                  if manual_probe_threshold is None else manual_probe_threshold)
        respond_streak_threshold = (self.cfg.mission.respond_streak_threshold
                                    if respond_streak_threshold is None else respond_streak_threshold)

        mission_id = f"{self.session_id}-{int(time.time())}"
        log_path = self._mission_log_path(mission_id)
        self._mission_log_init(log_path, objective)
        # Lets a driver (the TUI) recover the log path after killing this
        # generator, since a kill runs debrief_mission separately rather
        # than from inside the cancelled generator (see mist.tui.app).
        yield MissionEvent(kind="started", text=str(log_path))

        # Redirect the shell's cwd to a per-mission target directory (see
        # WorkspaceConfig.mission_root and MutableWorkspace) so this
        # mission's scan output/downloaded exploits/payloads land in their
        # own folder instead of the one flat workspace every mission ever
        # run shares regardless of target.
        if self.cfg.workspace.mission_root:
            target = _extract_target(objective)
            if target and getattr(self.tools, "workspace", None) is not None:
                mission_dir = Path(self.cfg.workspace.mission_root).expanduser() / target
                self.tools.workspace.path = mission_dir
                self._mission_log_append(
                    log_path, f"\n**Working directory:** `{mission_dir}`\n"
                )

        self.always_exposed.add("finish_objective")
        try:
            start = time.monotonic()
            turns = 0
            user_msg = objective
            routing_query = objective
            last_signature: str | None = None
            occurrences = 0
            near_dup_signature: str | None = None
            near_dup_occurrences = 0
            manual_probe_streak = 0
            respond_streak = 0
            mission_findings: list[str] = []
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
                tool_called_this_turn = False
                async for event in self.astream_turn(user_msg, force_think=use_think,
                                                     routing_query=routing_query):
                    yield MissionEvent(kind=event.kind, text=event.text,
                                       tool=event.tool, detail=event.detail)
                    if event.kind == "tool_start":
                        tool_called_this_turn = True
                        self._mission_log_append(
                            log_path, f"\n### {_now()} — {event.tool}\n**args:** `{event.detail}`\n"
                        )
                        if event.tool == "finish_objective":
                            finished = True
                        sig = f"{event.tool}:{event.detail}"
                        occurrences = occurrences + 1 if sig == last_signature else 1
                        last_signature = sig
                        nd_sig = _normalize_tool_call(event.tool, event.detail)
                        near_dup_occurrences = (near_dup_occurrences + 1
                                                if nd_sig == near_dup_signature else 1)
                        near_dup_signature = nd_sig
                        manual_probe_streak = (manual_probe_streak + 1
                                               if _is_manual_probe(event.tool, event.detail) else 0)
                        if occurrences >= stuck_threshold:
                            # Break out of astream_turn's own tool loop
                            # immediately — a single turn can run up to
                            # max_tool_steps tool calls on its own, so
                            # checking only after the turn ends would let a
                            # runaway repeat hammer the target far more than
                            # stuck_threshold times before ever catching it.
                            stuck = True
                            stuck_kind = "repeat"
                            stuck_repeat_count = occurrences
                            stuck_display = last_signature
                            break
                        if near_dup_occurrences >= near_dup_threshold:
                            # Same idea, but for calls that are never
                            # byte-identical — e.g. `search_files` spammed
                            # with a different query each time, or the same
                            # grep re-run with only its search term changed.
                            # See _normalize_tool_call for what this does and
                            # doesn't catch.
                            stuck = True
                            stuck_kind = "repeat"
                            stuck_repeat_count = near_dup_occurrences
                            stuck_display = f"{event.tool} (varying only a literal/number each call)"
                            break
                        if manual_probe_streak >= manual_probe_threshold:
                            # A third, distinct failure shape: every call is
                            # genuinely different (a different path each
                            # time), so neither check above ever fires — but
                            # it's the same *tool class* used over and over
                            # as a substitute for a content-discovery scanner.
                            # See _is_manual_probe.
                            stuck = True
                            stuck_kind = "manual_probe"
                            stuck_repeat_count = manual_probe_streak
                            stuck_display = f"{manual_probe_streak}x curl/wget probes with no scanner call in between"
                            break
                    elif event.kind == "tool_result":
                        self._mission_log_append(log_path, f"```\n{event.text}\n```\n")
                        mission_findings.append(f"{event.tool}: {event.text[:200]}")
                    elif event.kind == "error":
                        # A crashed LLM/network call ends astream_turn's own
                        # generator, but the mission loop must not just spin
                        # into an immediate retry — a dead backend would
                        # otherwise burn through the whole turn budget in
                        # seconds. Treat it like a stuck loop: pause and let
                        # the operator see what happened.
                        error_text = event.text

                if not finished and error_text is None:
                    if tool_called_this_turn:
                        respond_streak = 0
                    elif not stuck:
                        # A turn that ends in "respond" with no tool call at
                        # all is invisible to every check above — they're all
                        # keyed off tool_start events, so a mission that just
                        # keeps "responding" with a plan in prose (instead of
                        # calling a tool per MISSION_CONTINUE_TEMPLATE's own
                        # instructions) would otherwise never trip stuck
                        # detection and could run to max_turns/max_seconds
                        # without ever acting. Confirmed live: a turn decided
                        # "respond", the free-text answer step burned its
                        # entire token budget still inside a <think> block,
                        # and the next turn did the same thing again.
                        respond_streak += 1
                        if respond_streak >= respond_streak_threshold:
                            stuck = True
                            stuck_kind = "no_action"
                            stuck_repeat_count = respond_streak
                            stuck_display = f"{respond_streak}x mission turns in a row with no tool call"

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
                            f"{stuck_repeat_count}x in a row ({stuck_display}) — attempting a "
                            "reasoning-assisted recovery before pausing.\n",
                        )
                        yield MissionEvent(
                            kind="recovering",
                            text=(f"Repeated the same tool call {stuck_repeat_count}x in a row "
                                  f"({stuck_display}) — reasoning through a different "
                                  "approach before giving up."),
                        )
                        if stuck_kind == "manual_probe":
                            nudge = (f"You've made {stuck_repeat_count} manual curl/wget requests in "
                                    "a row, each to a different path, without running a scanner. "
                                    "Content/route discovery is a scanner's job (gobuster/ffuf/"
                                    "dirsearch) and vulnerability checks come from nmap --script "
                                    "vuln / nuclei / searchsploit — not hand-picked URLs. Also check "
                                    "whether the objective is already satisfied by what you've "
                                    "already found; if it is, call finish_objective now instead of "
                                    "continuing to explore.")
                        elif stuck_kind == "no_action":
                            nudge = (f"You've given a plain-text response for {stuck_repeat_count} "
                                    "turns in a row without calling a tool. This objective requires "
                                    "action, not a description of what you would do — pick one "
                                    "concrete next step and call a tool for it right now instead of "
                                    "writing out commands as text.")
                        else:
                            nudge = (f"You've repeated the same tool call ({stuck_display}) "
                                    f"{stuck_repeat_count} times in a row with no new result. Stop and "
                                    "actually think through why this specific approach isn't "
                                    "working, then commit to a genuinely different next step — "
                                    "not a minor variation of the same command.")
                        nudge += _findings_recap(mission_findings)
                        notes = control.pop_notes()
                        user_msg = _mission_continue_message(objective, notes, nudge)
                        routing_query = _mission_routing_query(objective, notes, nudge)
                        occurrences = 0
                        near_dup_occurrences = 0
                        manual_probe_streak = 0
                        respond_streak = 0
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
                        f"recovery attempt — repeated the same tool call {stuck_repeat_count}x in "
                        f"a row again ({stuck_display}).\n",
                    )
                    yield MissionEvent(
                        kind="stuck",
                        text=(f"Still stuck after trying to reason through it — repeated "
                              f"the same tool call {stuck_repeat_count}x in a row again "
                              f"({stuck_display}) — paused for operator review."),
                    )
                    if stuck_kind == "manual_probe":
                        nudge = (f"You're still making manual curl/wget requests ({stuck_display}) "
                                "instead of switching to a scanner (gobuster/ffuf/nuclei/"
                                "searchsploit) or recognizing the objective is already done. "
                                "Explain what you're blocked on if you need the operator's judgment.")
                    elif stuck_kind == "no_action":
                        nudge = (f"You're still responding in plain text ({stuck_display}) instead of "
                                "calling a tool. Call a tool for a concrete next step, or explain "
                                "what you're blocked on if you need the operator's judgment.")
                    else:
                        nudge = (f"You've repeated the same tool call ({stuck_display}) "
                                f"{stuck_repeat_count} times in a row with no new result — that approach "
                                "isn't working. Try something different, or explain what you're "
                                "blocked on if you need the operator's judgment.")
                    nudge += _findings_recap(mission_findings)
                    notes = control.pop_notes()
                    user_msg = _mission_continue_message(objective, notes, nudge)
                    routing_query = _mission_routing_query(objective, notes, nudge)
                    occurrences = 0
                    near_dup_occurrences = 0
                    manual_probe_streak = 0
                    respond_streak = 0
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
                    notes = control.pop_notes()
                    nudge = f"Your previous attempt failed: {error_text}. Try again."
                    user_msg = _mission_continue_message(objective, notes, nudge)
                    routing_query = _mission_routing_query(objective, notes, nudge)
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

                # Made it through a normal turn — but only a turn that
                # actually called a tool is genuine progress; a "respond"
                # turn that merely hasn't crossed the no-action threshold
                # yet must not reset this, or a slowly-accumulating
                # response streak would never make it past its first
                # recovery attempt (each not-yet-stuck turn in between
                # would silently discard the "already tried once" state).
                if tool_called_this_turn:
                    recovered_once = False
                notes = control.pop_notes()
                user_msg = _mission_continue_message(objective, notes)
                routing_query = _mission_routing_query(objective, notes)
        finally:
            self.always_exposed.discard("finish_objective")
