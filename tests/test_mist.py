import asyncio
import json
import os
import shutil
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from mist import backup as backup_module
from mist import doctor as doctor_module
from mist.config import (ArtifactConfig, MistConfig, SecurityConfig, ShellConfig,
                         ShellSSHConfig, StructuredToolsConfig, load_config,
                         resolve_config_path)
from mist.core.action import (ActionValidationError, DecisionRespond, DecisionUseTool,
                              SubagentRespond, SubagentUseTool, parse_decision_action,
                              parse_subagent_action)
from mist.core.agent import MistAgent, _approx_tokens, _extract_target, _findings_recap, _fit_budget
from mist.core.checkpoints import CheckpointStore
from mist.core.context_compressor import ContextCompressor
from mist.core.mission import MissionControl
from mist.core.subagent import run_subagents, run_tool_subagents
from mist.core.summarizer import BatchSummarizer
from mist.core.tool_compressor import ToolOutputCompressor
from mist.memory.store import MemoryStore
from mist.skills.router import SkillRouter
from mist.tools.artifacts import ArtifactStore
from mist.tools.registry import ProcessRegistry, Tool, default_registry
from mist.tools.safety import check_command_dangerous
from mist.tools.structured import detect_tool, summarize_tool_output
from mist.llm.client import parse_json_relaxed
from mist.tui.memory_commands import render_history_command, render_memories_command
from mist.wiki import init_wiki


class FakeLLM:
    """Scripted LLM: pops canned replies in order."""
    def __init__(self, replies):
        self.replies = list(replies)

    def complete(self, messages, json_schema=None, force_think=False):
        return self.replies.pop(0)


class EchoLLM:
    """Thread-safe fake LLM: replies based on the last message's content
    instead of call order, so it's safe to use from concurrent callers."""
    def complete(self, messages, json_schema=None, force_think=False):
        return json.dumps({"response": f"handled: {messages[-1]['content']}"})


class FakeEmbeddings:
    """Returns a fixed vector for known skill text, and a vector close to
    'git-workflow' for anything else — enough to test the semantic fallback
    without a real embedding model."""
    GIT_WORKFLOW_TEXT = ("git-workflow: Steps for common git operations like "
                          "status, commit, branch, and push")

    def embed(self, texts):
        return [[1.0, 0.0] if t == self.GIT_WORKFLOW_TEXT else [0.9, 0.1] for t in texts]


def _isolated_skills_library(tmp_path) -> Path:
    """A skills_library containing only the git-workflow example skill,
    isolated from whatever else has since been added to the real
    skills_library/ (e.g. pentest-wiki) — these routing tests assert on
    exact rank/threshold behavior against a single known skill."""
    dest = tmp_path / "skills_library"
    shutil.copytree(Path("skills_library") / "example", dest / "example")
    return dest


def make_agent(tmp_path, replies):
    cfg = MistConfig()
    cfg.memory.db_path = str(tmp_path / "test.db")
    store = MemoryStore(cfg.db_path)
    skills = SkillRouter("skills_library")
    tools = default_registry(remember_fn=store.remember)
    return MistAgent(cfg, FakeLLM(replies), store, skills, tools)


def test_memory_fts_roundtrip(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    store.remember("User prefers Python over JavaScript")
    store.remember("Project deadline is Friday")
    hits = store.search("what language does the user prefer python")
    assert any("Python" in h for h in hits)


# -- memory/history admin (MemoryStore list/delete + /memories, /history) ---

def test_list_memories_orders_newest_first_and_respects_limit(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    for i in range(5):
        store.remember(f"note {i}")
    rows = store.list_memories(limit=2)
    assert [r["content"] for r in rows] == ["note 4", "note 3"]


def test_list_memories_like_filter(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    store.remember("Reactor target uses CVE-2025-55182")
    store.remember("Cap target uses SMB")
    rows = store.list_memories(like="Reactor")
    assert len(rows) == 1
    assert "Reactor" in rows[0]["content"]


def test_forget_memory_deletes_row_and_drops_from_search(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    mid = store.remember("corrupted test residue")
    assert store.forget_memory(mid) is True
    assert store.list_memories() == []
    assert not any("corrupted" in h for h in store.search("corrupted"))


def test_forget_memory_nonexistent_id_returns_false(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    assert store.forget_memory(999) is False


def test_delete_memories_matching_deletes_only_matches(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    store.remember("Reactor note one")
    store.remember("Reactor note two")
    store.remember("Unrelated note")
    deleted = store.delete_memories_matching("Reactor")
    assert deleted == 2
    remaining = store.list_memories()
    assert len(remaining) == 1
    assert remaining[0]["content"] == "Unrelated note"


def test_delete_memories_matching_no_match_deletes_nothing(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    store.remember("Unrelated note")
    assert store.delete_memories_matching("Nonexistent") == 0
    assert len(store.list_memories()) == 1


def test_list_sessions_includes_turn_count_and_zero_turn_session(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    empty_sid = store.new_session("empty")
    busy_sid = store.new_session("busy")
    store.add_turn(busy_sid, "user", "hello")
    store.add_turn(busy_sid, "assistant", "hi")
    sessions = {s["id"]: s for s in store.list_sessions()}
    assert sessions[empty_sid]["turn_count"] == 0
    assert sessions[busy_sid]["turn_count"] == 2


def test_list_turns_returns_ids_and_respects_like_filter(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    sid = store.new_session()
    store.add_turn(sid, "user", "run nmap against Reactor")
    store.add_turn(sid, "assistant", "sure, scanning now")
    rows = store.list_turns(sid, like="nmap")
    assert len(rows) == 1
    assert rows[0]["role"] == "user"
    assert "id" in rows[0]


def test_delete_turn_removes_single_row(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    sid = store.new_session()
    store.add_turn(sid, "user", "hello")
    [row] = store.list_turns(sid)
    assert store.delete_turn(row["id"]) is True
    assert store.list_turns(sid) == []


def test_delete_turn_nonexistent_returns_false(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    assert store.delete_turn(999) is False


def test_delete_turns_matching_scopes_to_session_and_pattern(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    sid_a = store.new_session("a")
    sid_b = store.new_session("b")
    store.add_turn(sid_a, "user", "Reactor recon")
    store.add_turn(sid_a, "user", "unrelated")
    store.add_turn(sid_b, "user", "Reactor recon")
    deleted = store.delete_turns_matching(sid_a, "Reactor")
    assert deleted == 1
    assert len(store.list_turns(sid_a)) == 1
    # session b's matching turn is untouched
    assert len(store.list_turns(sid_b)) == 1


def test_delete_turns_matching_no_pattern_clears_whole_session(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    sid = store.new_session()
    store.add_turn(sid, "user", "one")
    store.add_turn(sid, "assistant", "two")
    deleted = store.delete_turns_matching(sid, None)
    assert deleted == 2
    assert store.list_turns(sid) == []


def test_render_memories_command_lists_recent(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    store.remember("hello world", kind="note")
    out = render_memories_command(store, "")
    assert "hello world" in out
    assert "[note]" in out


def test_render_memories_clear_without_yes_is_dry_run(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    store.remember("Reactor residue")
    out = render_memories_command(store, "clear Reactor")
    assert "1 memories match" in out
    assert "--yes" in out
    assert len(store.list_memories()) == 1  # nothing actually deleted


def test_render_memories_clear_with_yes_deletes(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    store.remember("Reactor residue")
    store.remember("unrelated")
    out = render_memories_command(store, "clear Reactor --yes")
    assert "Deleted 1" in out
    remaining = store.list_memories()
    assert len(remaining) == 1
    assert remaining[0]["content"] == "unrelated"


def test_render_memories_forget_deletes_immediately_no_dry_run(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    mid = store.remember("to be forgotten")
    out = render_memories_command(store, f"forget {mid}")
    assert f"Deleted memory #{mid}" in out
    assert store.list_memories() == []


def test_render_memories_forget_bad_id_returns_usage(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    store.remember("keep me")
    out = render_memories_command(store, "forget abc")
    assert out.startswith("Usage:")
    assert len(store.list_memories()) == 1


def test_render_history_lists_sessions(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    sid = store.new_session("my session")
    store.add_turn(sid, "user", "hi")
    out = render_history_command(store, "")
    assert "my session" in out
    assert "1 turns" in out


def test_render_history_lists_session_turns_scoped(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    sid_a = store.new_session("a")
    sid_b = store.new_session("b")
    store.add_turn(sid_a, "user", "Reactor scan")
    store.add_turn(sid_b, "user", "Reactor scan")
    out = render_history_command(store, f"{sid_a} Reactor")
    assert "Reactor scan" in out
    # only one match reported for session a, not both sessions combined
    assert out.count("Reactor scan") == 1


def test_render_history_clear_without_yes_is_dry_run(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    sid = store.new_session()
    store.add_turn(sid, "user", "Reactor scan")
    out = render_history_command(store, f"clear {sid} Reactor")
    assert "--yes" in out
    assert len(store.list_turns(sid)) == 1


def test_render_history_clear_with_yes_deletes(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    sid = store.new_session()
    store.add_turn(sid, "user", "Reactor scan")
    store.add_turn(sid, "user", "unrelated")
    out = render_history_command(store, f"clear {sid} Reactor --yes")
    assert "Deleted 1" in out
    remaining = store.list_turns(sid)
    assert len(remaining) == 1
    assert remaining[0]["content"] == "unrelated"


def test_render_history_clear_bad_session_id_returns_usage(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    out = render_history_command(store, "clear notanumber")
    assert out.startswith("Usage:")


def test_rename_session_updates_title_and_reports_true(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    sid = store.new_session("old title")
    assert store.rename_session(sid, "new title") is True
    assert store.list_sessions()[0]["title"] == "new title"


def test_rename_session_nonexistent_returns_false(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    assert store.rename_session(999, "whatever") is False


def test_export_session_renders_full_transcript(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    sid = store.new_session("Reactor engagement")
    store.add_turn(sid, "user", "scan the target")
    store.add_turn(sid, "assistant", "found port 3000 open")
    text = store.export_session(sid)
    assert text is not None
    assert "Reactor engagement" in text
    assert "scan the target" in text
    assert "found port 3000 open" in text


def test_export_session_nonexistent_returns_none(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    assert store.export_session(999) is None


def test_export_session_with_zero_turns_is_still_exportable(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    sid = store.new_session("empty session")
    text = store.export_session(sid)
    assert text is not None
    assert "empty session" in text


def test_render_history_rename_updates_title(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    sid = store.new_session("old")
    out = render_history_command(store, f"rename {sid} new title here")
    assert "Renamed session" in out
    assert store.list_sessions()[0]["title"] == "new title here"


def test_render_history_rename_nonexistent_session(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    out = render_history_command(store, "rename 999 whatever")
    assert "No session 999" in out


def test_render_history_rename_bad_args_returns_usage(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    assert render_history_command(store, "rename notanumber title").startswith("Usage:")
    assert render_history_command(store, "rename 1").startswith("Usage:")


def test_render_history_export_writes_file_and_reports_path(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    sid = store.new_session("Reactor")
    store.add_turn(sid, "user", "hello")
    dest = tmp_path / "exports" / "out.md"
    out = render_history_command(store, f"export {sid} {dest}")
    assert str(dest) in out
    assert dest.is_file()
    assert "hello" in dest.read_text(encoding="utf-8")


def test_render_history_export_default_path_under_mist_exports(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    sid = store.new_session()
    store.add_turn(sid, "user", "hi")
    out = render_history_command(store, f"export {sid}")
    expected = Path("~/.mist/exports").expanduser() / f"session-{sid}.md"
    assert str(expected) in out
    assert expected.is_file()
    expected.unlink()  # clean up — this one lands outside tmp_path


def test_render_history_export_nonexistent_session(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    out = render_history_command(store, "export 999")
    assert "No session 999" in out


def test_render_history_export_bad_args_returns_usage(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    assert render_history_command(store, "export notanumber").startswith("Usage:")
    assert render_history_command(store, "export").startswith("Usage:")


def test_skill_routing(tmp_path):
    router = SkillRouter(_isolated_skills_library(tmp_path))
    hits = router.route("help me commit my git changes")
    assert hits and hits[0].name == "git-workflow"
    assert router.route("bake a chocolate cake") == []


def _make_agent_with_preload(tmp_path, preload_skills):
    cfg = MistConfig()
    cfg.memory.db_path = str(tmp_path / "test.db")
    store = MemoryStore(cfg.db_path)
    skills = SkillRouter(_isolated_skills_library(tmp_path))
    tools = default_registry(remember_fn=store.remember)
    return MistAgent(cfg, FakeLLM([]), store, skills, tools, preload_skills=preload_skills)


def test_parse_skills_option_handles_repeat_comma_and_none():
    from mist.cli import _parse_skills_option
    assert _parse_skills_option(None) is None
    assert _parse_skills_option([]) is None
    assert _parse_skills_option(["a", "b"]) == ["a", "b"]
    assert _parse_skills_option(["a,b", "c"]) == ["a", "b", "c"]
    assert _parse_skills_option([" a , b "]) == ["a", "b"]


def test_assemble_context_preloads_named_skill_regardless_of_routing_query(tmp_path):
    # "bake a chocolate cake" is the exact query test_skill_routing asserts
    # the router itself returns zero hits for — proving the preload really
    # bypasses routing rather than just getting lucky on keyword overlap.
    agent = _make_agent_with_preload(tmp_path, ["git-workflow"])
    _, skill_section, _, _ = agent._assemble_context("bake a chocolate cake")
    assert "Preloaded skill (git-workflow)" in skill_section
    assert "commit" in skill_section.lower()  # the real skill body, not a stub


def test_assemble_context_preload_not_double_counted_as_router_hit(tmp_path):
    agent = _make_agent_with_preload(tmp_path, ["git-workflow"])
    # This query also matches git-workflow via the router (see
    # test_skill_routing) — the preloaded skill must appear exactly once,
    # not again in an "Other possibly relevant skills" listing.
    _, skill_section, _, _ = agent._assemble_context("help me commit my git changes")
    assert skill_section.count("Preloaded skill (git-workflow)") == 1
    assert "Other possibly relevant skills" not in skill_section


def test_assemble_context_ignores_unknown_preload_names(tmp_path):
    agent = _make_agent_with_preload(tmp_path, ["not-a-real-skill"])
    _, skill_section, _, _ = agent._assemble_context("bake a chocolate cake")
    assert "Preloaded skill" not in skill_section  # nothing valid to preload
    assert "not-a-real-skill" not in skill_section


def test_tool_selection_capped(tmp_path):
    tools = default_registry(remember_fn=lambda c: None)
    selected = tools.select("read a file", max_exposed=2)
    assert len(selected) == 2
    assert selected[0].name == "read_file"


def test_agent_respond(tmp_path):
    # turn() is now two calls: a decision-only call (no `response` field —
    # see the module docstring on why that field being present in the same
    # schema made the model default to describing instead of acting), then
    # a separate unconstrained call that generates the actual answer text.
    agent = make_agent(tmp_path, [
        json.dumps({"action": "respond"}),
        "hi",
    ])
    result = agent.turn("hello")
    assert result.response == "hi"
    # persisted to history
    turns = agent.store.recent_turns(agent.session_id, 10)
    assert turns[-1]["content"] == "hi"


def test_agent_tool_loop(tmp_path):
    agent = make_agent(tmp_path, [
        json.dumps({"action": "use_tool", "tool": "shell",
                    "arguments": {"command": "echo mist-test"}}),
        json.dumps({"action": "respond"}),
        "done",
    ])
    result = agent.turn("run echo")
    assert result.response == "done"
    assert any("mist-test" in step for step in result.tool_trace)


def test_agent_recovers_from_bad_json(tmp_path):
    agent = make_agent(tmp_path, [
        "not json at all",
        json.dumps({"action": "respond"}),
        "recovered",
    ])
    assert agent.turn("hi").response == "recovered"


def test_parse_json_relaxed_fenced():
    obj = parse_json_relaxed('```json\n{"action": "respond", "response": "x"}\n```')
    assert obj["action"] == "respond"


def test_parse_json_relaxed_repairs_shell_regex_backslashes():
    # The exact captured failure: a command with regex backslashes (\d, \s)
    # that are invalid JSON escapes made json.loads reject the whole action,
    # looping a mission forever on "no valid JSON". They must now be repaired.
    raw = r'''{ "action": "shell", "command": "grep -oP '^\d+(?=/tcp\s+open)' scan.nmap | paste -sd, -" }'''
    obj = parse_json_relaxed(raw)
    assert obj["action"] == "shell"
    assert r"\d+" in obj["command"] and r"\s+" in obj["command"]  # backslashes preserved literally
    # more backslash shapes: sed with \. and a path with \w
    obj2 = parse_json_relaxed(r'{"action":"shell","command":"sed \"s/\./_/g\" && echo C:\Windows\web"}')
    assert obj2["action"] == "shell" and r"\." in obj2["command"]


def test_parse_json_relaxed_preserves_valid_escapes():
    # Repair must not corrupt legitimately-escaped content (a real newline, a
    # real backslash, a quote inside the string).
    obj = parse_json_relaxed(r'{"action":"shell","command":"echo line1\nline2 \\ done \"q\""}')
    assert obj["command"] == 'echo line1\nline2 \\ done "q"'


# -- embedding fallback for skill routing --------------------------------

def test_skill_routing_falls_back_to_embeddings_when_lexical_fails(tmp_path):
    router = SkillRouter(_isolated_skills_library(tmp_path), embeddings=FakeEmbeddings(),
                         embedding_threshold=0.5)
    # Shares no tokens with the git-workflow skill's name/description.
    hits = router.route("how do I track changes to my project over time")
    assert hits and hits[0].name == "git-workflow"


def test_skill_routing_embedding_fallback_respects_threshold(tmp_path):
    router = SkillRouter(_isolated_skills_library(tmp_path), embeddings=FakeEmbeddings(),
                         embedding_threshold=0.999)
    assert router.route("how do I track changes to my project over time") == []


def test_skill_routing_prefers_lexical_over_embeddings(tmp_path):
    # Lexical match exists, so the (deliberately wrong) fake embeddings must
    # never be consulted.
    router = SkillRouter(_isolated_skills_library(tmp_path), embeddings=FakeEmbeddings())
    hits = router.route("help me commit my git changes")
    assert hits and hits[0].name == "git-workflow"


def test_content_tokens_excludes_stopwords():
    # Regression case from a real run: two skills' meaningful-word overlap
    # (3 words each) tied with a third skill whose overlap was almost
    # entirely stopwords ("a", "do", "or", "the", "what") plus one real
    # word — raw token overlap scored the stopword-heavy skill highest.
    from mist.skills.router import _content_tokens
    assert _content_tokens("a the of or what do") == set()
    assert _content_tokens("Complete the HTB Machine") == {"complete", "htb", "machine"}


def test_skill_routing_not_decided_by_stopword_overlap(tmp_path):
    # git-workflow's description ("Steps for common git operations like
    # status, commit, branch, and push") shares only stopwords with a
    # query about an unrelated topic phrased with the same connector words
    # — it must not be rated relevant just because both share "for"/"and".
    router = SkillRouter(_isolated_skills_library(tmp_path))
    assert router.route("looking for a place to eat, and nothing else") == []


def test_mission_routing_query_omits_boilerplate():
    # _mission_routing_query carries the real, variable content (objective,
    # nudge, operator notes) but never MISSION_CONTINUE_TEMPLATE's fixed
    # instructional text — that boilerplate is identical every turn and
    # gave the llm-wiki skill ("...wiki...knowledge...tool...") a
    # keyword-overlap head start on every single mission turn regardless
    # of what the mission was actually about.
    from mist.core.agent import _mission_routing_query, MISSION_CONTINUE_TEMPLATE
    q = _mission_routing_query("get root on 10.129.30.204", notes=["operator note"],
                               nudge="stop repeating yourself")
    assert "get root on 10.129.30.204" in q
    assert "operator note" in q
    assert "stop repeating yourself" in q
    # None of the boilerplate's own distinctive phrasing should leak in.
    assert "finish_objective" not in q
    assert "search_files" not in q


# -- batch summarizer -----------------------------------------------------

def test_batch_summarizer_compacts_session_into_memory(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    sid = store.new_session()
    store.add_turn(sid, "user", "My favorite language is Rust")
    store.add_turn(sid, "assistant", "Noted!")

    llm = FakeLLM([json.dumps({"memories": [
        {"content": "User's favorite language is Rust", "kind": "preference"}
    ]})])
    result = BatchSummarizer(llm, store).summarize_session(sid)

    assert result.memories_written == 1
    assert any("Rust" in h for h in store.search("favorite language"))
    assert store.sessions_to_compact(None, keep_recent=0, min_turns=1) == []


def test_sessions_to_compact_respects_keep_recent_and_min_turns(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    old = store.new_session()
    for _ in range(4):
        store.add_turn(old, "user", "hi")
    recent = store.new_session()
    for _ in range(4):
        store.add_turn(recent, "user", "hi")
    too_short = store.new_session()
    store.add_turn(too_short, "user", "hi")

    ids = store.sessions_to_compact(exclude_session_id=None, keep_recent=1, min_turns=4)
    assert ids == [old]


def test_compact_old_sessions_batches_multiple(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    sessions = []
    for i in range(3):
        sid = store.new_session()
        store.add_turn(sid, "user", f"session {i} content")
        store.add_turn(sid, "assistant", "ok")
        store.add_turn(sid, "user", "more")
        store.add_turn(sid, "assistant", "ok")
        sessions.append(sid)

    llm = FakeLLM([json.dumps({"memories": []})] * len(sessions))
    results = BatchSummarizer(llm, store).compact_old_sessions(
        exclude_session_id=None, keep_recent=0, min_turns=4
    )
    assert {r.session_id for r in results} == set(sessions)


# -- subagent spawning -----------------------------------------------------

def test_run_subagents_executes_all_tasks_concurrently():
    results = run_subagents(EchoLLM(), ["task a", "task b", "task c"], max_workers=3)
    assert {r.task for r in results} == {"task a", "task b", "task c"}
    assert all(r.error is None for r in results)
    assert all(r.response.startswith("handled: ") for r in results)


def test_run_subagents_empty_tasks():
    assert run_subagents(EchoLLM(), [], max_workers=4) == []


def test_default_registry_omits_spawn_subagents_without_llm():
    tools = default_registry(remember_fn=lambda c: None)
    assert tools.get("spawn_subagents") is None


def test_default_registry_exposes_spawn_subagents_with_llm():
    tools = default_registry(remember_fn=lambda c: None, llm=EchoLLM())
    tool = tools.get("spawn_subagents")
    assert tool is not None
    out = tool.run(tasks=["do x"])
    assert "do x" in out and "handled: do x" in out


# -- tool-enabled subagents (mist.core.subagent.run_tool_subagents) ------

class ToolEchoLLM:
    """Thread-safe fake LLM for the tool-enabled subagent loop: on the task
    message it requests the shell tool (echoing the task text back), then on
    the tool result it responds with that result. Keyed off message content,
    not call order, so it's safe under concurrent callers."""
    def complete(self, messages, json_schema=None, force_think=False):
        last = messages[-1]["content"]
        if last.startswith("Tool result:"):
            return json.dumps({"action": "respond", "response": last})
        return json.dumps({"action": "use_tool", "tool": "shell",
                            "arguments": {"command": f"echo {last}"}})


def test_run_tool_subagents_concurrent_with_shell():
    tools = default_registry(remember_fn=lambda c: None)
    tasks = ["alpha", "beta", "gamma"]
    results = run_tool_subagents(ToolEchoLLM(), tools, tasks, max_workers=3, max_steps=3)
    assert {r.task for r in results} == set(tasks)
    for r in results:
        assert r.error is None
        assert r.task in r.response  # echoed back via the shell tool result


def test_run_tool_subagents_empty_tasks():
    tools = default_registry(remember_fn=lambda c: None)
    assert run_tool_subagents(ToolEchoLLM(), tools, [], max_workers=4) == []


def test_run_tool_subagents_excludes_mission_only_tools():
    # Regression case from a real run: `finish_objective` is mission_only
    # (meant to be reachable only through a mission's own decision loop via
    # always_exposed) but `tools.all()` doesn't filter that, so a subagent
    # used to see it listed as an available tool — and a live model tried to
    # call it, emitting {"action": "finish_objective", "summary": "..."},
    # a shape this loop's schema has no room for.
    tools = default_registry(remember_fn=lambda c: None)
    assert any(t.name == "finish_objective" for t in tools.all())  # exists in the full registry

    captured: dict = {}

    class CapturingLLM:
        def complete(self, messages, json_schema=None, force_think=False):
            captured["system"] = messages[0]["content"]
            captured["schema"] = json_schema
            return json.dumps({"action": "respond", "response": "done"})

    run_tool_subagents(CapturingLLM(), tools, ["a task"], max_workers=1, max_steps=3)
    assert "finish_objective" not in captured["system"]
    assert "finish_objective" not in captured["schema"]["properties"]["tool"]["enum"]


def test_run_tool_subagents_surfaces_malformed_action_instead_of_empty_response():
    # Same regression case: when a model emits a schema-deviant action
    # anyway (missing "tool", an unexpected "action" value), the fallback
    # must surface something a caller can act on, not a bare "(empty
    # response)" that discards the only clue as to what went wrong — this
    # is exactly why a real mission just blindly retried the identical
    # spawn_subagents call three times before getting flagged as stuck.
    class MalformedLLM:
        def complete(self, messages, json_schema=None, force_think=False):
            return json.dumps({"action": "finish_objective", "summary": "done scanning"})

    tools = default_registry(remember_fn=lambda c: None)
    results = run_tool_subagents(MalformedLLM(), tools, ["a task"], max_workers=1, max_steps=3)
    assert len(results) == 1
    assert results[0].response != "(empty response)"
    assert "finish_objective" in results[0].response


def test_default_registry_omits_write_skill_without_skills():
    tools = default_registry(remember_fn=lambda c: None)
    assert tools.get("write_skill") is None


def test_default_registry_exposes_write_skill_with_skills(tmp_path):
    router = SkillRouter(tmp_path / "skills_library")
    tools = default_registry(remember_fn=lambda c: None, skills=router)
    assert tools.get("write_skill") is not None


# -- self-growing skill wiki (write_skill tool + SkillRouter.reload) -----

def test_skill_router_reload_picks_up_new_skill(tmp_path):
    library = tmp_path / "skills_library"
    router = SkillRouter(library)
    assert router.route("exploit a confirmed sql injection") == []

    skill_dir = library / "sql-injection"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: sql-injection\ndescription: confirmed sql injection exploit\n---\n\nbody",
        encoding="utf-8",
    )
    # Not loaded yet — the router only re-scans on reload().
    assert router.route("exploit a confirmed sql injection") == []

    router.reload()
    hits = router.route("exploit a confirmed sql injection")
    assert hits and hits[0].name == "sql-injection"


def test_write_skill_tool_writes_file_and_is_immediately_routable(tmp_path):
    library = tmp_path / "skills_library"
    router = SkillRouter(library)
    tools = default_registry(remember_fn=lambda c: None, skills=router)
    tool = tools.get("write_skill")

    result = tool.run(name="SQL Injection Basics",
                      description="confirmed basic sql injection technique",
                      body="1. Identify the vulnerable parameter.\n2. Confirm with a payload.")
    assert "Wrote skill" in result
    written = library / "sql-injection-basics" / "SKILL.md"
    assert written.is_file()
    assert "confirmed basic sql injection technique" in written.read_text()

    # write_skill reloads internally, so it's routable without a manual call.
    hits = router.route("confirmed basic sql injection technique")
    assert hits and hits[0].name == "sql-injection-basics"


def test_write_skill_rejects_invalid_name(tmp_path):
    library = tmp_path / "skills_library"
    router = SkillRouter(library)
    tools = default_registry(remember_fn=lambda c: None, skills=router)
    tool = tools.get("write_skill")

    result = tool.run(name="   ", description="x", body="y")
    assert result.startswith("ERROR")
    assert not library.exists() or not any(library.iterdir())


# -- raw/ immutability (mist.tools.registry write_file) ------------------

def test_write_file_refuses_to_overwrite_existing_raw_source(tmp_path):
    # SCHEMA.md documents raw/ as "read but never modify these once
    # written", but nothing in code ever enforced it — a write_file call
    # could silently clobber a curated source. Now it's actually blocked.
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "course-notes.md").write_text("original content\n", encoding="utf-8")
    tools = default_registry(remember_fn=lambda c: None, wiki_root=tmp_path)

    out = tools.get("write_file").run(path="raw/course-notes.md", content="clobbered")
    assert out.startswith("ERROR")
    assert "immutable" in out
    assert (raw / "course-notes.md").read_text(encoding="utf-8") == "original content\n"


def test_write_file_still_allows_new_files_under_raw(tmp_path):
    # Ingesting a new source into raw/ must still work — only overwriting an
    # existing file is blocked.
    tools = default_registry(remember_fn=lambda c: None, wiki_root=tmp_path)
    out = tools.get("write_file").run(path="raw/new-source.md", content="fresh content")
    assert out.startswith("Wrote")
    assert (tmp_path / "raw" / "new-source.md").read_text(encoding="utf-8") == "fresh content"


def test_write_file_unaffected_outside_raw(tmp_path):
    # Curated pages elsewhere in the wiki can still be freely created and
    # overwritten — the restriction is specific to raw/.
    entities = tmp_path / "entities"
    entities.mkdir()
    (entities / "target.md").write_text("old\n", encoding="utf-8")
    tools = default_registry(remember_fn=lambda c: None, wiki_root=tmp_path)

    out = tools.get("write_file").run(path="entities/target.md", content="updated")
    assert out.startswith("Wrote")
    assert (entities / "target.md").read_text(encoding="utf-8") == "updated"


# -- memory store concurrency (mist.memory.store.MemoryStore) -----------

def test_memory_store_concurrent_writes_dont_corrupt(tmp_path):
    store = MemoryStore(tmp_path / "concurrent.db")
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: store.remember(f"memory {i}"), range(50)))
    count = store.conn.execute("SELECT COUNT(*) AS c FROM memories").fetchone()["c"]
    assert count == 50


# -- llm-wiki: search_files tool -----------------------------------------

def test_search_files_finds_matching_line_with_file_and_lineno(tmp_path):
    (tmp_path / "entities").mkdir()
    (tmp_path / "entities" / "target-a.md").write_text(
        "---\ntitle: Target A\n---\n\nRunning nginx 1.18.0 on port 80.\n", encoding="utf-8"
    )
    tools = default_registry(remember_fn=lambda c: None, wiki_root=tmp_path)
    out = tools.get("search_files").run(query="nginx")
    assert "entities/target-a.md:5:" in out
    assert "nginx 1.18.0" in out


def test_search_files_respects_max_results(tmp_path):
    for i in range(5):
        (tmp_path / f"page{i}.md").write_text("needle\n", encoding="utf-8")
    tools = default_registry(remember_fn=lambda c: None, wiki_root=tmp_path)
    out = tools.get("search_files").run(query="needle", max_results=2)
    assert len(out.splitlines()) == 2


def test_search_files_no_matches_returns_message(tmp_path):
    (tmp_path / "page.md").write_text("nothing relevant here\n", encoding="utf-8")
    tools = default_registry(remember_fn=lambda c: None, wiki_root=tmp_path)
    assert tools.get("search_files").run(query="needle") == "No matches."


def test_search_files_missing_root_returns_error():
    tools = default_registry(remember_fn=lambda c: None, wiki_root=None)
    out = tools.get("search_files").run(query="anything")
    assert out.startswith("ERROR")


def test_search_files_root_param_anchors_to_wiki_root_like_read_write(tmp_path):
    # Regression test for a real bug: `root` was used as-is (relative to the
    # process's cwd), unlike read_file/write_file's `path`, which anchors
    # relative paths to the wiki root. A model passing root="entities"
    # expecting <wiki_root>/entities got an unrelated (or nonexistent) cwd
    # path instead. (Not testing with root="missions" here — that directory
    # is now unconditionally excluded regardless of anchoring, see below.)
    (tmp_path / "entities").mkdir()
    (tmp_path / "entities" / "target.md").write_text("needle here\n", encoding="utf-8")
    tools = default_registry(remember_fn=lambda c: None, wiki_root=tmp_path)
    out = tools.get("search_files").run(query="needle", root="entities")
    assert "needle here" in out


def test_search_files_excludes_missions_directory_regardless_of_root(tmp_path):
    # Regression test for a real incident: the deterministic mission log
    # lives under <wiki_root>/missions/ and grows live during the mission
    # that might search it — a query matches its own previously-logged
    # search for that same query, which gets logged, which the next search
    # matches again, producing unbounded self-referential noise (observed
    # as exponentially nested quoted blocks in a real run). The missions
    # directory must never be searchable, no matter how `root` is set.
    (tmp_path / "missions").mkdir()
    (tmp_path / "missions" / "40-123.md").write_text("needle in the live log\n", encoding="utf-8")
    (tmp_path / "entities").mkdir()
    (tmp_path / "entities" / "real.md").write_text("needle in real content\n", encoding="utf-8")
    tools = default_registry(remember_fn=lambda c: None, wiki_root=tmp_path)

    out = tools.get("search_files").run(query="needle")
    assert "real content" in out
    assert "live log" not in out

    # Even an explicit attempt to search the missions dir directly finds nothing.
    out2 = tools.get("search_files").run(query="needle", root="missions")
    assert out2 == "No matches."


def test_default_registry_exposes_search_files():
    tools = default_registry(remember_fn=lambda c: None)
    assert tools.get("search_files") is not None


# -- opt-in tool allow-list + SSH shell backend --------------------------

def test_default_registry_enabled_allowlist_restricts_tools(tmp_path):
    skills = SkillRouter(tmp_path / "skills_library")
    tools = default_registry(remember_fn=lambda c: None, llm=EchoLLM(), skills=skills,
                             enabled=["read_file"])
    assert tools.get("read_file") is not None
    assert tools.get("write_file") is None
    assert tools.get("shell") is None
    assert tools.get("write_skill") is None
    assert tools.get("spawn_subagents") is None


def test_default_registry_enabled_none_preserves_default_behavior():
    tools = default_registry(remember_fn=lambda c: None, llm=EchoLLM())
    assert tools.get("read_file") is not None
    assert tools.get("shell") is not None
    assert tools.get("remember") is not None
    assert tools.get("spawn_subagents") is not None


def test_shell_local_backend_unaffected_by_shell_config_none():
    tools = default_registry(remember_fn=lambda c: None, shell_config=None)
    out = tools.get("shell").run(command="echo local-backend-test")
    assert "local-backend-test" in out


# -- browse tool (headless-browser render) -------------------------------

def _run_browse_extractor(html: str, url: str = "") -> str:
    """Run the exact EXTRACTOR_SRC that executes remotely, against sample
    HTML, so the test covers the real extraction code path (python3 stdlib)."""
    import subprocess
    import sys

    from mist.tools.browser import EXTRACTOR_SRC
    return subprocess.run([sys.executable, "-c", EXTRACTOR_SRC, url],
                          input=html, capture_output=True, text=True).stdout


def test_browse_extractor_pulls_title_forms_links_and_hides_secrets():
    html = ('<html><head><title>Webmail Login</title><style>b{}</style></head><body>'
            '<script>var leak="TOPSECRET";</script>'
            '<form method="post" action="/?_task=login">'
            '<input name="_user" type="text" placeholder="Username">'
            '<input name="_pass" type="password" value="hunter2"></form>'
            '<a href="/inbox">inbox</a><a href="#">x</a>'
            '<a href="javascript:void(0)">j</a><p>hello world</p></body></html>')
    out = _run_browse_extractor(html, "http://t/")
    assert "TITLE: Webmail Login" in out
    assert "form#1 POST action=/?_task=login" in out
    assert "input name=_user type=text" in out
    assert "input name=_pass type=password" in out
    assert "hunter2" not in out          # a password field's value is never surfaced
    assert "TOPSECRET" not in out        # <script>/<style> contents are skipped
    assert "[LINKS] (1 unique)" in out   # '#' and javascript: links filtered out
    assert "/inbox" in out
    assert "hello world" in out


def test_browse_extractor_flags_an_empty_render():
    out = _run_browse_extractor("<html><body></body></html>")
    assert "no visible text" in out


def test_build_browse_command_normalizes_and_quotes_url():
    from mist.tools.browser import build_browse_command, normalize_url
    assert normalize_url("enigma.htb") == "http://enigma.htb"
    assert normalize_url("https://x/") == "https://x/"
    cmd = build_browse_command("enigma.htb/login")
    assert "--dump-dom" in cmd
    assert "http://enigma.htb/login" in cmd
    assert "timeout -k 5 45" in cmd
    assert "python3 -c" in cmd


def test_build_browse_command_is_injection_safe():
    import shlex

    from mist.tools.browser import build_browse_command
    hostile = "http://x/; rm -rf ~ #"
    cmd = build_browse_command(hostile)
    # The hostile URL survives only as a single shell-quoted token (an argument
    # to chromium/python3), never as a separate command — no bare `rm` token.
    tokens = shlex.split(cmd)
    assert hostile in tokens
    assert "rm" not in tokens


def test_default_registry_exposes_browse_by_default():
    tools = default_registry(remember_fn=lambda c: None)
    assert tools.get("browse") is not None


def test_default_registry_omits_browse_when_disabled():
    from mist.config import BrowserConfig
    tools = default_registry(remember_fn=lambda c: None,
                             browser_config=BrowserConfig(enabled=False))
    assert tools.get("browse") is None


def test_browse_delegates_to_shell_with_a_render_command():
    from mist.tools.registry import _make_browse
    captured = {}

    def fake_shell(command):
        captured["command"] = command
        return "RENDERED: http://enigma.htb\nTITLE: X"

    out = _make_browse(fake_shell, None)(url="enigma.htb")
    assert "--dump-dom" in captured["command"]
    assert "http://enigma.htb" in captured["command"]
    assert "RENDERED" in out


def test_shell_ssh_backend_builds_correct_command(monkeypatch):
    captured = {}

    class FakePopen:
        def __init__(self, args, shell, stdout, stderr, text, cwd=None):
            captured["args"] = args
            captured["shell"] = shell
            captured["cwd"] = cwd
            self.returncode = 0

        def communicate(self, timeout=None):
            captured["timeout"] = timeout
            return ("remote-host\n", None)

    monkeypatch.setattr("mist.tools.registry.subprocess.Popen", FakePopen)

    shell_cfg = ShellConfig(backend="ssh", ssh=ShellSSHConfig(
        host="10.0.0.5", user="kali", port=2222, key_path="~/.ssh/id_ed25519", timeout=90,
    ))
    tools = default_registry(remember_fn=lambda c: None, shell_config=shell_cfg)
    out = tools.get("shell").run(command="hostname")

    assert "remote-host" in out
    args = captured["args"]
    assert args[0] == "ssh"
    assert "-p" in args and args[args.index("-p") + 1] == "2222"
    assert "kali@10.0.0.5" in args
    assert args[-1] == "bash -c hostname"   # run under bash, not the login shell
    assert captured["timeout"] == 90
    assert captured["shell"] is False


def test_run_subprocess_timeout_message_is_actionable():
    # A timed-out long scanner must return guidance the model can act on — not
    # just a bare "timed out" — so it re-runs scoped/faster and reads the -o
    # file rather than restarting the whole scan (session 126: gobuster ran
    # into the 480s kill and the run was a dead loss).
    from mist.tools.registry import _run_subprocess
    out = _run_subprocess("sleep 5", shell=True, timeout=1, registry=None,
                          command_text="sleep 5")
    assert "timed out after 1s" in out
    assert "-t 50" in out                 # suggests more threads
    assert "-o" in out and "partial" in out  # points at the output file for partial results


def test_run_subprocess_timeout_does_not_hang_when_kill_wont_release_pipes(monkeypatch):
    # Regression (session 3): a hung SSH/NFS command timed out, but the
    # post-kill communicate() had NO timeout and blocked forever — the mission
    # silently stalled for 40+ min with no result and no marker. The second
    # read is now bounded, so _run_subprocess always returns.
    import subprocess as _sp
    calls = {"n": 0}

    class WedgedPopen:
        def __init__(self, *a, **k):
            self.returncode = None

        def communicate(self, timeout=None):
            calls["n"] += 1
            raise _sp.TimeoutExpired(cmd="x", timeout=timeout or 0)  # both reads "hang"

        def kill(self):
            pass

    monkeypatch.setattr("mist.tools.registry.subprocess.Popen", lambda *a, **k: WedgedPopen())
    from mist.tools.registry import _run_subprocess
    out = _run_subprocess("hung-nfs-read", shell=True, timeout=0.1, registry=None,
                          command_text="hung-nfs-read")
    assert "timed out" in out    # returns cleanly with the timeout message, no infinite block
    assert calls["n"] == 2       # initial timed communicate + the now-BOUNDED post-kill one


def _cleanup_agent(backend="ssh", enabled=True):
    """A bare MistAgent with just enough wired for _cleanup_mounts, plus a fake
    shell tool that records the commands it's asked to run."""
    from mist.core.agent import MistAgent
    from mist.config import MistConfig, ShellConfig, ShellSSHConfig
    cfg = MistConfig()
    cfg.mission.unmount_shares_on_end = enabled
    cfg.tools.shell = (ShellConfig(backend="ssh", ssh=ShellSSHConfig(host="h"))
                       if backend == "ssh" else ShellConfig(backend="local"))
    calls: list[str] = []

    class FakeShell:
        name = "shell"

        def run(self, **kw):
            calls.append(kw.get("command", ""))
            return ""

    class FakeTools:
        def get(self, n):
            return FakeShell() if n == "shell" else None

    agent = MistAgent.__new__(MistAgent)
    agent.cfg = cfg
    agent.tools = FakeTools()
    return agent, calls


def test_cleanup_mounts_unmounts_network_shares_on_ssh_backend():
    # On mission end (SSH backend), force-lazy-unmount NFS/SMB shares so they
    # don't accumulate — a stale `hard` mount to a since-changed target IP
    # wedges every later filesystem op (real incident: four stale mounts).
    agent, calls = _cleanup_agent(backend="ssh", enabled=True)
    agent._cleanup_mounts()
    assert len(calls) == 1
    cmd = calls[0]
    assert "umount -f -l" in cmd            # force + lazy: detaches even a wedged mount
    assert "/proc/mounts" in cmd            # reads /proc/mounts (never blocks, unlike df/mount)
    assert "nfs" in cmd and "/mnt/" in cmd  # scoped to network fs types + engagement paths


def test_cleanup_mounts_skips_local_backend():
    # Must NEVER touch the machine Mist itself runs on — only the dedicated
    # remote shell host.
    agent, calls = _cleanup_agent(backend="local", enabled=True)
    agent._cleanup_mounts()
    assert calls == []


def test_cleanup_mounts_respects_disable_flag():
    agent, calls = _cleanup_agent(backend="ssh", enabled=False)
    agent._cleanup_mounts()
    assert calls == []


def test_json_fail_detail_surfaces_raw_output_for_diagnosis():
    # A decision that never parses must carry the raw model output into the
    # error (and thus the mission log), not the old opaque "failed to produce
    # valid JSON" — so a recurring JSON-error pause is diagnosable without live
    # repro. Flags empty output and an unclosed <think> explicitly.
    from mist.core.agent import _json_fail_detail

    assert "EMPTY" in _json_fail_detail("")
    assert "EMPTY" in _json_fail_detail(None)
    assert "EMPTY" in _json_fail_detail("   \n  ")

    unclosed = _json_fail_detail("<think>let me reason about which tool to use and")
    assert "UNCLOSED <think>" in unclosed
    assert "let me reason" in unclosed

    prose = _json_fail_detail("Sure! First I'll run an nmap scan against the target.")
    assert "First I'll run an nmap scan" in prose        # the actual raw output is shown
    assert "no valid JSON" in prose

    long = _json_fail_detail("x" * 2000)
    assert "…" in long and len(long) < 700               # truncated, not dumping 2000 chars


def test_findings_recap_header_discourages_restarting_earlier_phases():
    recap = _findings_recap(["shell: 22/tcp open ssh; 80/tcp open http"])
    assert "Already found this mission" in recap   # unchanged anchor other code/tests rely on
    assert "FORWARD" in recap and "don't restart" in recap  # anti recon-regression framing


def test_mission_continue_message_carries_findings_on_every_path():
    # Regression (session 129): after an error-pause + operator /resume, the
    # continuing turn re-ran all its recon because only the stuck path carried
    # the findings recap. The continue message must now fold findings in on any
    # path (with or without a nudge), so a resumed turn always knows what it
    # already accomplished. Empty findings -> no recap.
    from mist.core.agent import _mission_continue_message
    findings = ["shell: 22/tcp open ssh; NFS export /srv/nfs/onboarding"]

    plain = _mission_continue_message("get root on host", [], findings=findings)
    assert "Already found this mission" in plain and "NFS export /srv/nfs/onboarding" in plain

    with_nudge = _mission_continue_message("get root on host", ["check nfs"],
                                           nudge="Your previous attempt failed. Try again.",
                                           findings=findings)
    assert "Try again." in with_nudge and "check nfs" in with_nudge
    assert "NFS export /srv/nfs/onboarding" in with_nudge   # findings survive alongside a nudge/notes

    assert "Already found" not in _mission_continue_message("obj", [])  # no findings -> no recap


def test_shell_ssh_wraps_glob_url_under_bash(monkeypatch):
    # Regression (session 121): an unquoted query-string URL made the remote
    # zsh login shell error with "no matches found" (nomatch on ?, *, [)
    # before curl ran. Wrapping under bash -c passes the unmatched glob through
    # literally, so the URL survives intact.
    import shlex as _shlex
    captured = {}

    class FakePopen:
        def __init__(self, args, shell, stdout, stderr, text, cwd=None):
            captured["args"] = args
            self.returncode = 0

        def communicate(self, timeout=None):
            return ("", None)

    monkeypatch.setattr("mist.tools.registry.subprocess.Popen", FakePopen)
    shell_cfg = ShellConfig(backend="ssh", ssh=ShellSSHConfig(
        host="h", user="u", port=22, key_path="", timeout=30))
    tools = default_registry(remember_fn=lambda c: None, shell_config=shell_cfg)
    cmd = "curl -sk http://enigma.htb/admin/config.php?display=api"
    tools.get("shell").run(command=cmd)

    remote = captured["args"][-1]
    assert remote == f"bash -c {_shlex.quote(cmd)}"   # exact bash wrapping
    assert "config.php?display=api" in remote          # the ? URL survived intact


def test_shell_local_backend_uses_dedicated_workspace_cwd(tmp_path, monkeypatch):
    # Regression test for a real incident: with no dedicated cwd, shell ran
    # from wherever the mist process happened to be launched, and a mission
    # wandered via relative paths into an unrelated sibling project
    # directory (agent-zero) that happened to sit next to the launch dir.
    captured = {}

    class FakePopen:
        def __init__(self, args, shell, stdout, stderr, text, cwd=None):
            captured["cwd"] = cwd
            self.returncode = 0

        def communicate(self, timeout=None):
            return ("ok\n", None)

    monkeypatch.setattr("mist.tools.registry.subprocess.Popen", FakePopen)
    workspace = tmp_path / "workspace"
    tools = default_registry(remember_fn=lambda c: None, workspace_root=workspace)
    tools.get("shell").run(command="pwd")

    assert captured["cwd"] == workspace
    assert workspace.is_dir()  # created on first use


def test_shell_ssh_backend_cds_into_workspace_first(monkeypatch):
    captured = {}

    class FakePopen:
        def __init__(self, args, shell, stdout, stderr, text, cwd=None):
            captured["args"] = args
            self.returncode = 0

        def communicate(self, timeout=None):
            return ("ok\n", None)

    monkeypatch.setattr("mist.tools.registry.subprocess.Popen", FakePopen)
    shell_cfg = ShellConfig(backend="ssh", ssh=ShellSSHConfig(host="10.0.0.5"))
    tools = default_registry(remember_fn=lambda c: None, shell_config=shell_cfg,
                             workspace_root="/home/kali/.mist/workspace")
    tools.get("shell").run(command="ls")

    import shlex as _shlex
    assert captured["args"][-1].startswith("bash -c ")      # run under bash, not the login shell
    remote_command = _shlex.split(captured["args"][-1])[2]   # unwrap the bash -c argument
    assert "cd '/home/kali/.mist/workspace'" in remote_command
    assert remote_command.endswith("&& ls")


def test_shell_local_backend_follows_mutable_workspace_redirect(tmp_path, monkeypatch):
    # The shell tool's cwd is read at *call* time (mist.tools.registry.
    # MutableWorkspace), not baked into the tool's closure at
    # default_registry() time — a mission redirects it once the target is
    # known, without needing to rebuild the tool.
    captured = {}

    class FakePopen:
        def __init__(self, args, shell, stdout, stderr, text, cwd=None):
            captured["cwd"] = cwd
            self.returncode = 0

        def communicate(self, timeout=None):
            return ("ok\n", None)

    monkeypatch.setattr("mist.tools.registry.subprocess.Popen", FakePopen)
    original = tmp_path / "workspace"
    tools = default_registry(remember_fn=lambda c: None, workspace_root=original)
    tools.get("shell").run(command="pwd")
    assert captured["cwd"] == original

    redirected = tmp_path / "HTB" / "10.129.33.21"
    tools.workspace.path = redirected
    tools.get("shell").run(command="pwd")
    assert captured["cwd"] == redirected
    assert redirected.is_dir()  # created on first use, same as the original


def test_shell_ssh_backend_follows_mutable_workspace_redirect(monkeypatch):
    captured = {}

    class FakePopen:
        def __init__(self, args, shell, stdout, stderr, text, cwd=None):
            captured["args"] = args
            self.returncode = 0

        def communicate(self, timeout=None):
            return ("ok\n", None)

    monkeypatch.setattr("mist.tools.registry.subprocess.Popen", FakePopen)
    shell_cfg = ShellConfig(backend="ssh", ssh=ShellSSHConfig(host="10.0.0.5"))
    tools = default_registry(remember_fn=lambda c: None, shell_config=shell_cfg,
                             workspace_root="/home/kali/.mist/workspace")
    tools.workspace.path = Path("/home/kali/Desktop/HTB/10.129.33.21")
    tools.get("shell").run(command="ls")

    import shlex as _shlex
    remote_command = _shlex.split(captured["args"][-1])[2]   # unwrap the bash -c argument
    assert "cd '/home/kali/Desktop/HTB/10.129.33.21'" in remote_command


# -- command-safety gate (mist.tools.safety, security config) --------------

def test_check_command_dangerous_flags_known_catastrophic_patterns():
    dangerous = [
        "rm -rf /",
        "rm -rf ~",
        "rm -rf /*",
        "dd if=/dev/zero of=/dev/sda",
        "mkfs.ext4 /dev/sdb1",
        ":(){ :|:& };:",
        "chmod -R 777 /",
        "shutdown -h now",
        "reboot",
        "iptables -F",
    ]
    for command in dangerous:
        assert check_command_dangerous(command) is not None, command


def test_check_command_dangerous_allows_normal_pentest_commands():
    safe = [
        "nmap -p- -T4 10.129.33.21",
        "curl -s http://10.129.33.21/admin",
        "rm -rf /tmp/scan_output",  # scoped, not the whole filesystem
        "gobuster dir -u http://10.129.33.21 -w wordlist.txt",
        "searchsploit freepbx",
        "ssh root@10.129.33.21",
    ]
    for command in safe:
        assert check_command_dangerous(command) is None, command


def test_check_command_dangerous_honors_custom_patterns():
    assert check_command_dangerous("echo hello", patterns=[r"echo"]) is not None
    assert check_command_dangerous("rm -rf /", patterns=[r"echo"]) is None


def test_shell_local_backend_blocks_dangerous_command(tmp_path, monkeypatch):
    called = {"popen": False}

    class FakePopen:
        def __init__(self, *a, **kw):
            called["popen"] = True
            self.returncode = 0

        def communicate(self, timeout=None):
            return ("ok\n", None)

    monkeypatch.setattr("mist.tools.registry.subprocess.Popen", FakePopen)
    tools = default_registry(remember_fn=lambda c: None, workspace_root=tmp_path / "ws")
    result = tools.get("shell").run(command="rm -rf /")

    assert "[blocked]" in result
    assert called["popen"] is False  # never actually executed


def test_shell_ssh_backend_blocks_dangerous_command(monkeypatch):
    called = {"popen": False}

    class FakePopen:
        def __init__(self, *a, **kw):
            called["popen"] = True
            self.returncode = 0

        def communicate(self, timeout=None):
            return ("ok\n", None)

    monkeypatch.setattr("mist.tools.registry.subprocess.Popen", FakePopen)
    shell_cfg = ShellConfig(backend="ssh", ssh=ShellSSHConfig(host="10.0.0.5"))
    tools = default_registry(remember_fn=lambda c: None, shell_config=shell_cfg)
    result = tools.get("shell").run(command="dd if=/dev/zero of=/dev/sda")

    assert "[blocked]" in result
    assert called["popen"] is False


def test_shell_allows_dangerous_command_when_gate_disabled(tmp_path, monkeypatch):
    class FakePopen:
        def __init__(self, *a, **kw):
            self.returncode = 0

        def communicate(self, timeout=None):
            return ("ok\n", None)

    monkeypatch.setattr("mist.tools.registry.subprocess.Popen", FakePopen)
    security = SecurityConfig(command_safety_enabled=False)
    tools = default_registry(remember_fn=lambda c: None, workspace_root=tmp_path / "ws",
                             security_config=security)
    result = tools.get("shell").run(command="rm -rf /")

    assert "[blocked]" not in result


def test_shell_respects_custom_deny_patterns_from_config(tmp_path, monkeypatch):
    class FakePopen:
        def __init__(self, *a, **kw):
            self.returncode = 0

        def communicate(self, timeout=None):
            return ("ok\n", None)

    monkeypatch.setattr("mist.tools.registry.subprocess.Popen", FakePopen)
    security = SecurityConfig(deny_patterns=[r"\bhydra\b"])
    tools = default_registry(remember_fn=lambda c: None, workspace_root=tmp_path / "ws",
                             security_config=security)

    # A command dangerous by the DEFAULT list is allowed, since a custom
    # deny_patterns list replaces (not extends) the defaults.
    assert "[blocked]" not in tools.get("shell").run(command="rm -rf /")
    assert "[blocked]" in tools.get("shell").run(command="hydra -l root -P wordlist ssh://10.0.0.5")


# -- checkpoints (mist.core.checkpoints.CheckpointStore) --------------------

def test_checkpoint_ensure_creates_shadow_repo_not_a_dot_git_in_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    store = CheckpointStore(base_dir=tmp_path / "checkpoints")
    store.ensure(workspace)

    assert workspace.is_dir()
    assert not (workspace / ".git").exists()  # shadow repo, workspace stays untouched
    repo_dir = store._repo_dir(workspace)
    assert (repo_dir / ".git").is_dir()
    assert (repo_dir / "workspace_path.txt").read_text(encoding="utf-8") == str(workspace.resolve())


def test_checkpoint_snapshot_commits_changes_and_reports_true(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = CheckpointStore(base_dir=tmp_path / "checkpoints")

    (workspace / "file1.txt").write_text("hello", encoding="utf-8")
    assert store.snapshot(workspace, "first snapshot") is True

    info = store.status(workspace)
    assert info is not None
    assert info.commit_count == 1


def test_checkpoint_snapshot_returns_false_when_nothing_changed(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = CheckpointStore(base_dir=tmp_path / "checkpoints")
    (workspace / "file1.txt").write_text("hello", encoding="utf-8")

    assert store.snapshot(workspace, "first") is True
    assert store.snapshot(workspace, "nothing changed") is False  # no diff since last commit


def test_checkpoint_snapshot_returns_false_for_nonexistent_workspace(tmp_path):
    store = CheckpointStore(base_dir=tmp_path / "checkpoints")
    assert store.snapshot(tmp_path / "does_not_exist", "msg") is False


def test_checkpoint_status_is_none_without_a_checkpoint_repo(tmp_path):
    store = CheckpointStore(base_dir=tmp_path / "checkpoints")
    assert store.status(tmp_path / "never_snapshotted") is None


def test_checkpoint_rollback_restores_prior_content(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = CheckpointStore(base_dir=tmp_path / "checkpoints")

    (workspace / "file1.txt").write_text("version 1", encoding="utf-8")
    store.snapshot(workspace, "v1")
    (workspace / "file1.txt").write_text("version 2 (bad)", encoding="utf-8")
    store.snapshot(workspace, "v2")

    assert store.rollback(workspace, "HEAD~1") is True
    assert (workspace / "file1.txt").read_text(encoding="utf-8") == "version 1"


def test_checkpoint_rollback_false_without_a_repo(tmp_path):
    store = CheckpointStore(base_dir=tmp_path / "checkpoints")
    assert store.rollback(tmp_path / "never_snapshotted") is False


def test_checkpoint_list_all_reports_every_project(tmp_path):
    store = CheckpointStore(base_dir=tmp_path / "checkpoints")
    ws_a = tmp_path / "a"
    ws_b = tmp_path / "b"
    ws_a.mkdir()
    ws_b.mkdir()
    (ws_a / "f.txt").write_text("a", encoding="utf-8")
    (ws_b / "f.txt").write_text("b", encoding="utf-8")
    store.snapshot(ws_a, "snap a")
    store.snapshot(ws_b, "snap b")

    paths = {info.workspace_path for info in store.list_all()}
    assert paths == {str(ws_a.resolve()), str(ws_b.resolve())}


def test_checkpoint_prune_removes_orphans_and_keeps_live_ones(tmp_path):
    store = CheckpointStore(base_dir=tmp_path / "checkpoints")
    ws_live = tmp_path / "live"
    ws_gone = tmp_path / "gone"
    ws_live.mkdir()
    ws_gone.mkdir()
    (ws_live / "f.txt").write_text("x", encoding="utf-8")
    (ws_gone / "f.txt").write_text("x", encoding="utf-8")
    store.snapshot(ws_live, "snap")
    store.snapshot(ws_gone, "snap")
    shutil.rmtree(ws_gone)  # simulate the engagement folder being cleaned up

    removed = store.prune()

    assert removed == 1
    remaining_paths = {info.workspace_path for info in store.list_all()}
    assert remaining_paths == {str(ws_live.resolve())}


def test_checkpoint_clear_removes_everything(tmp_path):
    store = CheckpointStore(base_dir=tmp_path / "checkpoints")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "f.txt").write_text("x", encoding="utf-8")
    store.snapshot(workspace, "snap")

    store.clear()

    assert store.list_all() == []
    assert not store.base_dir.exists()


def _make_agent_with_checkpoints(tmp_path, checkpoint_store):
    cfg = MistConfig()
    cfg.memory.db_path = str(tmp_path / "test.db")
    cfg.workspace.root_path = str(tmp_path / "workspace")
    store = MemoryStore(cfg.db_path)
    skills = SkillRouter("skills_library")
    tools = default_registry(remember_fn=store.remember, workspace_root=cfg.workspace_root)
    return MistAgent(cfg, FakeLLM([]), store, skills, tools, checkpoint_store=checkpoint_store)


def test_checkpoint_before_tool_snapshots_shell_and_write_file(tmp_path):
    # Something must already exist to snapshot — an untouched empty
    # directory correctly produces no commit (nothing to capture yet).
    (tmp_path / "workspace").mkdir(parents=True, exist_ok=True)
    (tmp_path / "workspace" / "existing.txt").write_text("v1", encoding="utf-8")
    checkpoint_store = CheckpointStore(base_dir=tmp_path / "checkpoints")
    agent = _make_agent_with_checkpoints(tmp_path, checkpoint_store)

    agent._checkpoint_before_tool("shell", {"command": "echo hi"})
    info = checkpoint_store.status(tmp_path / "workspace")
    assert info is not None
    assert info.commit_count == 1

    (tmp_path / "workspace" / "new.txt").write_text("v2", encoding="utf-8")
    agent._checkpoint_before_tool("write_file", {"path": "other.txt", "content": "x"})
    info = checkpoint_store.status(tmp_path / "workspace")
    assert info.commit_count == 2


def test_checkpoint_before_tool_ignores_non_mutating_tools(tmp_path):
    (tmp_path / "workspace").mkdir(parents=True, exist_ok=True)
    checkpoint_store = CheckpointStore(base_dir=tmp_path / "checkpoints")
    agent = _make_agent_with_checkpoints(tmp_path, checkpoint_store)

    agent._checkpoint_before_tool("read_file", {"path": "foo.txt"})
    agent._checkpoint_before_tool("remember", {"content": "a fact"})

    assert checkpoint_store.status(tmp_path / "workspace") is None  # no repo ever created


def test_checkpoint_before_tool_noop_when_store_not_configured(tmp_path):
    agent = _make_agent_with_checkpoints(tmp_path, checkpoint_store=None)
    agent._checkpoint_before_tool("shell", {"command": "echo hi"})  # must not raise


def test_checkpoint_before_tool_follows_mission_workspace_redirect(tmp_path):
    checkpoint_store = CheckpointStore(base_dir=tmp_path / "checkpoints")
    agent = _make_agent_with_checkpoints(tmp_path, checkpoint_store)

    redirected = tmp_path / "HTB" / "10.129.33.21"
    agent.tools.workspace.path = redirected
    agent._checkpoint_before_tool("shell", {"command": "nmap -p- 10.129.33.21"})

    assert checkpoint_store.status(redirected) is not None
    assert checkpoint_store.status(tmp_path / "workspace") is None  # the static default untouched


# -- backup/restore (mist.backup) -------------------------------------------

def test_create_backup_zips_mist_home_and_wiki(tmp_path, monkeypatch):
    fake_home = tmp_path / "dot_mist"
    fake_home.mkdir()
    (fake_home / "config.yaml").write_text("backend: ollama", encoding="utf-8")
    (fake_home / "mist.db").write_text("fake db bytes", encoding="utf-8")
    monkeypatch.setattr(backup_module, "MIST_HOME", fake_home)

    wiki_root = tmp_path / "wiki"
    wiki_root.mkdir()
    (wiki_root / "SCHEMA.md").write_text("# schema", encoding="utf-8")
    cfg = MistConfig()
    cfg.wiki.root_path = str(wiki_root)

    output = backup_module.create_backup(cfg, output_path=tmp_path / "out.zip")

    assert output.is_file()
    with zipfile.ZipFile(output) as zf:
        names = set(zf.namelist())
        assert "manifest.json" in names
        assert "mist_home/config.yaml" in names
        assert "mist_home/mist.db" in names
        assert "wiki/SCHEMA.md" in names
        manifest = json.loads(zf.read("manifest.json"))
        assert manifest["mist_home"] == str(fake_home)
        assert manifest["wiki_root"] == str(wiki_root)


def test_create_backup_excludes_prior_backups_directory(tmp_path, monkeypatch):
    fake_home = tmp_path / "dot_mist"
    fake_home.mkdir()
    (fake_home / "config.yaml").write_text("backend: ollama", encoding="utf-8")
    old_backups = fake_home / "backups"
    old_backups.mkdir()
    (old_backups / "mist-backup-old.zip").write_bytes(b"not a real zip, just a marker")
    monkeypatch.setattr(backup_module, "MIST_HOME", fake_home)

    cfg = MistConfig()
    cfg.wiki.root_path = str(tmp_path / "no_such_wiki")

    output = backup_module.create_backup(cfg, output_path=tmp_path / "out.zip")
    with zipfile.ZipFile(output) as zf:
        names = zf.namelist()
        assert not any("backups/" in n for n in names)


def test_create_backup_defaults_output_path_under_mist_home_backups(tmp_path, monkeypatch):
    fake_home = tmp_path / "dot_mist"
    fake_home.mkdir()
    monkeypatch.setattr(backup_module, "MIST_HOME", fake_home)
    cfg = MistConfig()
    cfg.wiki.root_path = str(tmp_path / "no_such_wiki")

    output = backup_module.create_backup(cfg)

    assert output.parent == fake_home / "backups"
    assert output.name.startswith("mist-backup-")


def test_restore_backup_recreates_files_at_manifest_paths(tmp_path, monkeypatch):
    fake_home = tmp_path / "dot_mist"
    fake_home.mkdir()
    (fake_home / "config.yaml").write_text("backend: ollama", encoding="utf-8")
    monkeypatch.setattr(backup_module, "MIST_HOME", fake_home)
    wiki_root = tmp_path / "wiki"
    wiki_root.mkdir()
    (wiki_root / "SCHEMA.md").write_text("# schema", encoding="utf-8")
    cfg = MistConfig()
    cfg.wiki.root_path = str(wiki_root)
    output = backup_module.create_backup(cfg, output_path=tmp_path / "out.zip")

    # Simulate data loss, then restore.
    shutil.rmtree(fake_home)
    shutil.rmtree(wiki_root)

    restored = backup_module.restore_backup(output)

    assert restored == {"mist_home": 1, "wiki": 1}
    assert (fake_home / "config.yaml").read_text(encoding="utf-8") == "backend: ollama"
    assert (wiki_root / "SCHEMA.md").read_text(encoding="utf-8") == "# schema"


def test_restore_backup_honors_target_overrides(tmp_path, monkeypatch):
    fake_home = tmp_path / "dot_mist"
    fake_home.mkdir()
    (fake_home / "config.yaml").write_text("backend: ollama", encoding="utf-8")
    monkeypatch.setattr(backup_module, "MIST_HOME", fake_home)
    cfg = MistConfig()
    cfg.wiki.root_path = str(tmp_path / "no_such_wiki")
    output = backup_module.create_backup(cfg, output_path=tmp_path / "out.zip")

    new_home = tmp_path / "restored_elsewhere"
    restored = backup_module.restore_backup(output, target_mist_home=new_home)

    assert restored["mist_home"] == 1
    assert (new_home / "config.yaml").read_text(encoding="utf-8") == "backend: ollama"


# -- config resolution (mist.config.resolve_config_path / load_config) -----

def test_resolve_config_path_returns_explicit_path_when_it_exists(tmp_path):
    path = tmp_path / "custom.yaml"
    path.write_text("backend: ollama", encoding="utf-8")
    assert resolve_config_path(str(path)) == path


def test_resolve_config_path_returns_none_for_explicit_nonexistent_path(tmp_path):
    assert resolve_config_path(str(tmp_path / "nope.yaml")) is None


def test_resolve_config_path_falls_back_to_mist_config_env_var(tmp_path, monkeypatch):
    path = tmp_path / "from_env.yaml"
    path.write_text("backend: ollama", encoding="utf-8")
    monkeypatch.setenv("MIST_CONFIG", str(path))
    monkeypatch.chdir(tmp_path)  # no cwd-relative config.yaml to compete with it
    assert resolve_config_path() == path


def test_resolve_config_path_returns_none_when_nothing_exists(tmp_path, monkeypatch):
    monkeypatch.delenv("MIST_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("os.path.expanduser",
                        lambda p: str(tmp_path / "no_such_home" / "config.yaml")
                        if p == "~/.mist/config.yaml" else os.path.expanduser(p))
    assert resolve_config_path() is None


def test_load_config_returns_defaults_when_nothing_resolves(tmp_path, monkeypatch):
    monkeypatch.delenv("MIST_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("os.path.expanduser",
                        lambda p: str(tmp_path / "no_such_home" / "config.yaml")
                        if p == "~/.mist/config.yaml" else os.path.expanduser(p))
    cfg = load_config()
    assert cfg.backend == "ollama"  # the built-in default, untouched


def test_shell_captures_stdout_and_stderr_separately_not_merged(monkeypatch):
    # Regression case from a real run: `nuclei` puts its actual findings and
    # "N matches found" summary on stdout, but its ASCII banner and
    # template-loading chatter on stderr. The old behavior merged the two
    # streams (stderr=subprocess.STDOUT), so that banner noise came first
    # and crowded out — even truncated away — the real finding before the
    # model ever saw it. stdout must survive in full when it fits, even
    # behind a much larger stderr banner.
    captured = {}

    class FakePopen:
        def __init__(self, args, shell, stdout, stderr, text, cwd=None):
            captured["stdout_arg"] = stdout
            captured["stderr_arg"] = stderr
            self.returncode = 0

        def communicate(self, timeout=None):
            return ("CVE-2025-55182 critical finding\nScan completed. 1 matches found.",
                    "banner noise " * 200)

    monkeypatch.setattr("mist.tools.registry.subprocess.Popen", FakePopen)
    tools = default_registry(remember_fn=lambda c: None, shell_config=None)
    out = tools.get("shell").run(command="nuclei -u http://target:3000")

    assert captured["stdout_arg"] is subprocess.PIPE
    assert captured["stderr_arg"] is subprocess.PIPE  # not subprocess.STDOUT
    assert "CVE-2025-55182" in out
    assert "matches found" in out


def test_combine_output_prioritizes_stdout_over_noisy_stderr():
    from mist.tools.registry import _combine_output
    stdout = "the actual finding"
    stderr = "banner noise " * 500  # much larger than stdout
    out = _combine_output(stdout, stderr, budget=200)
    assert "the actual finding" in out


def test_combine_output_passes_through_stdout_only_when_stderr_empty():
    from mist.tools.registry import _combine_output
    assert _combine_output("just stdout", "", budget=200) == "just stdout"


# -- Tier 1: full-output artifacts (mist.tools.artifacts.ArtifactStore) ----

def test_artifact_store_writes_full_untruncated_output(tmp_path):
    store = ArtifactStore(tmp_path, min_chars_to_persist=10)
    big_stdout = "x" * 5000
    artifact = store.write("nmap -p- 10.0.0.1", big_stdout, "some stderr")
    assert artifact is not None
    full_path = tmp_path / artifact.rel_path
    assert full_path.is_file()
    content = full_path.read_text(encoding="utf-8")
    assert big_stdout in content  # nothing truncated
    assert "some stderr" in content
    assert "nmap -p- 10.0.0.1" in content  # command recorded in the header
    assert artifact.stdout_chars == 5000
    assert artifact.stderr_chars == len("some stderr")


def test_artifact_store_skips_persisting_trivially_small_output(tmp_path):
    store = ArtifactStore(tmp_path, min_chars_to_persist=500)
    assert store.write("echo hi", "hi\n", "") is None
    # nothing written under the wiki root at all
    assert not (tmp_path / "raw").exists()


def test_artifact_store_write_failure_returns_none_not_raise(tmp_path, monkeypatch):
    store = ArtifactStore(tmp_path, min_chars_to_persist=1)

    def _boom(*a, **kw):
        raise OSError("disk full")
    monkeypatch.setattr(Path, "write_text", _boom)
    assert store.write("echo hi", "hi" * 100, "") is None


def test_artifact_pointer_survives_tail_truncation(tmp_path):
    # The whole point of appending the pointer as a suffix: it must still be
    # present after the existing head+tail truncation runs on the combined
    # result. Sanity-checks the real numbers cited in the design doc.
    from mist.tools.registry import _truncate_raw_output
    store = ArtifactStore(tmp_path, min_chars_to_persist=1)
    artifact = store.write("nmap -p- 10.0.0.1", "x" * 10000, "")
    pointer = artifact.pointer()
    combined = ("y" * 20000) + pointer
    truncated = _truncate_raw_output(combined, budget=2000)
    assert pointer in truncated


# -- Tier 2: structured extraction (mist.tools.structured) -----------------

def test_detect_tool_recognizes_known_tools_and_sudo_prefix():
    assert detect_tool("nmap -sV -sC 10.0.0.1") == "nmap"
    assert detect_tool("sudo nmap -p- 10.0.0.1") == "nmap"
    assert detect_tool("/usr/lib/nmap/nmap -sV 10.0.0.1") == "nmap"
    assert detect_tool("nuclei -u http://x") == "nuclei"
    assert detect_tool("gobuster dir -u http://x -w list.txt") == "gobuster"
    assert detect_tool("ffuf -u http://x/FUZZ") == "ffuf"
    assert detect_tool("searchsploit apache 2.4") == "searchsploit"


def test_detect_tool_returns_none_for_unrecognized_command():
    assert detect_tool("curl -s http://x") is None
    assert detect_tool("cat /etc/passwd") is None
    assert detect_tool("") is None


NMAP_SAMPLE = """Starting Nmap 7.99 ( https://nmap.org ) at 2026-07-07 20:07 -0500
Nmap scan report for 10.129.245.214
Host is up (0.062s latency).
Not shown: 998 closed tcp ports (reset)
PORT     STATE SERVICE VERSION
22/tcp   open  ssh     OpenSSH 9.6p1 Ubuntu 3ubuntu13.16 (Ubuntu Linux; protocol 2.0)
| ssh-hostkey:
|   256 ce:fd:0d:82:c0:23:ed:6e:4b:ea:13:fa:4f:ea:ef:b7 (ECDSA)
|_  256 f8:44:c6:46:58:7a:39:21:ef:16:44:e9:58:c2:f3:62 (ED25519)
3000/tcp open  ppp?
| fingerprint-strings:
|   GetRequest:
|     HTTP/1.1 200 OK
SF:x2008\\x20Jul\\x202026\\x2001:08:01\\x20GMT\\x20some\\x20garbage\\x20data
SF:Connection:\\x20close\\r\\n\\r\\n")%r(RPCCheck,2F,"HTTP/1\\.1\\x20400
Service Info: OS: Linux; CPE: cpe:/o:linux:linux_kernel
Nmap done: 1 IP address (1 host up) scanned in 18.22 seconds
"""


def test_summarize_nmap_drops_sf_blob_keeps_port_table():
    out = summarize_tool_output("nmap", NMAP_SAMPLE, "")
    assert "22/tcp   open  ssh" in out
    assert "3000/tcp open  ppp?" in out
    assert "Nmap done:" in out
    assert "SF:x2008" not in out  # the actual raw fingerprint data line is gone
    assert "lines of raw fingerprint-strings" in out  # replaced by a one-line note


def test_summarize_nmap_passthrough_when_no_sf_lines():
    simple = "PORT   STATE SERVICE\n22/tcp open  ssh\nNmap done: 1 IP address scanned\n"
    out = summarize_tool_output("nmap", simple, "")
    assert out == simple.rstrip("\n")  # split+rejoin drops a trailing newline; no data lost


NUCLEI_SAMPLE = """
                     __     _
   ____  __  _______/ /__  (_)
  / __ \\/ / / / ___/ / _ \\/ /
 / / / / /_/ / /__/ /  __/ /
/_/ /_/\\__,_/\\___/_/\\___/_/   v3.2.1

[INF] Using Nuclei Engine 3.2.1
[INF] Templates loaded for scan: 5000
[CVE-2025-55182] [http] [critical] http://10.129.30.204:3000/
Scan completed. 1 matches found.
"""


def test_summarize_nuclei_extracts_finding_and_summary_drops_banner():
    out = summarize_tool_output("nuclei", NUCLEI_SAMPLE, "loading templates...\n" * 20)
    assert "[CVE-2025-55182] [http] [critical]" in out
    assert "1 matches found" in out
    assert "Using Nuclei Engine" not in out
    assert "loading templates" not in out


def test_summarize_nuclei_parses_jsonl_output_completely():
    lines = [json.dumps({"template-id": f"cve-{i}",
                         "info": {"severity": "high"},
                         "matched-at": f"http://x/{i}"}) for i in range(50)]
    out = summarize_tool_output("nuclei", "\n".join(lines), "")
    assert out.count("[cve-") == 50  # every finding survives, not truncated


def test_summarize_nuclei_returns_none_when_nothing_recognizable():
    assert summarize_tool_output("nuclei", "some unrelated plain text\nwith no brackets", "") is None


def test_summarize_content_discovery_extracts_status_lines():
    stdout = (
        "===============================================================\n"
        "Gobuster v3.6\n"
        "===============================================================\n"
        "Progress: 500 / 4614 (10.84%)\n"
        "/login               (Status: 200) [Size: 1234]\n"
        "/admin               (Status: 403) [Size: 278]\n"
        "Progress: 4614 / 4614 (100.00%)\n"
        "===============================================================\n"
    )
    out = summarize_tool_output("gobuster", stdout, "")
    assert "/login               (Status: 200) [Size: 1234]" in out
    assert "/admin               (Status: 403) [Size: 278]" in out
    assert "Progress:" not in out
    assert "====" not in out


def test_summarize_content_discovery_detects_same_size_false_positive():
    # Regression case from a real ffuf run: a Next.js catch-all page
    # returned Status 200 with an identical Size for virtually every path
    # tried — the useful signal is "exclude this size and re-scan," not a
    # wall of near-identical hits.
    lines = [f"/path{i}    (Status: 200) [Size: 17175]" for i in range(20)]
    lines.append("/real-hit   (Status: 200) [Size: 942]")
    stdout = "\n".join(lines)
    out = summarize_tool_output("ffuf", stdout, "")
    assert "20/21 paths returned Status 200, Size 17175" in out
    assert "-fs 17175" in out
    assert "/real-hit   (Status: 200) [Size: 942]" in out


def test_summarize_content_discovery_returns_none_without_status_lines():
    assert summarize_tool_output("ffuf", "no hits here, just banner text", "") is None


def test_summarize_content_discovery_detects_narrow_size_band_wildcard():
    # Regression case from a real run (HTB "Connected"): a wildcard vhost
    # 301-redirected virtually every fuzzed word, but Content-Length varied
    # 231-240 bytes because the response echoed the requested word back —
    # never an exact size match, so the identical-size check above never
    # fires. Status-code clustering + a narrow size band still catches it.
    lines = [f"word{i}   [Status: 301, Size: {231 + (i % 10)}, Words: 14, Lines: 8]"
             for i in range(30)]
    lines.append("real-hit  [Status: 200, Size: 942, Words: 20, Lines: 5]")
    stdout = "\n".join(lines)
    out = summarize_tool_output("ffuf", stdout, "")
    assert "30/31 paths returned Status 301" in out
    assert "231-240" in out or "240-231" in out
    assert "-fc 301" in out
    assert "real-hit  [Status: 200, Size: 942" in out


def test_summarize_content_discovery_exact_size_match_takes_priority():
    # When an exact-size match exists, it's more precise than the
    # status-clustering fallback and should be used instead.
    lines = [f"/path{i}    (Status: 200) [Size: 500]" for i in range(15)]
    stdout = "\n".join(lines)
    out = summarize_tool_output("gobuster", stdout, "")
    assert "-fs 500" in out
    assert "-fc" not in out


# -- Tier 1+2 wiring into _run_subprocess/_shell ----------------------------

def test_shell_persists_artifact_and_appends_pointer(tmp_path, monkeypatch):
    class FakePopen:
        def __init__(self, args, shell, stdout, stderr, text, cwd=None):
            self.returncode = 0

        def communicate(self, timeout=None):
            return ("y" * 1000, "")

    monkeypatch.setattr("mist.tools.registry.subprocess.Popen", FakePopen)
    tools = default_registry(remember_fn=lambda c: None, shell_config=None,
                             wiki_root=tmp_path,
                             artifact_config=ArtifactConfig(min_chars_to_persist=10))
    out = tools.get("shell").run(command="echo test")
    assert "[full output:" in out
    assert "use read_file, not shell/grep" in out
    artifact_files = list((tmp_path / "raw" / "tool-output").rglob("*.txt"))
    assert len(artifact_files) == 1
    assert "y" * 1000 in artifact_files[0].read_text(encoding="utf-8")


def test_shell_applies_structured_summary_for_recognized_tool(tmp_path, monkeypatch):
    class FakePopen:
        def __init__(self, args, shell, stdout, stderr, text, cwd=None):
            self.returncode = 0

        def communicate(self, timeout=None):
            return (NMAP_SAMPLE, "")

    monkeypatch.setattr("mist.tools.registry.subprocess.Popen", FakePopen)
    tools = default_registry(remember_fn=lambda c: None, shell_config=None,
                             structured_tools_config=StructuredToolsConfig())
    out = tools.get("shell").run(command="nmap -sV -sC 10.129.245.214")
    assert "SF:x2008" not in out
    assert "22/tcp   open  ssh" in out


def test_shell_reproduces_prior_behavior_when_tier1_and_tier2_disabled(monkeypatch):
    # Config-off: with no artifact_config/structured_tools_config passed at
    # all (both default to None in default_registry), behavior must be
    # byte-for-byte identical to before tiers 1-2 existed.
    class FakePopen:
        def __init__(self, args, shell, stdout, stderr, text, cwd=None):
            self.returncode = 0

        def communicate(self, timeout=None):
            return (NMAP_SAMPLE, "")

    monkeypatch.setattr("mist.tools.registry.subprocess.Popen", FakePopen)
    from mist.tools.registry import _combine_output
    tools = default_registry(remember_fn=lambda c: None, shell_config=None)
    out = tools.get("shell").run(command="nmap -sV -sC 10.129.245.214")
    assert out == _combine_output(NMAP_SAMPLE, "")
    assert "[full output:" not in out


# -- Tier 3: auxiliary-model compression (mist.core.tool_compressor) --------

class FakeCompressorLLM:
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def complete(self, messages, json_schema=None, force_think=False):
        self.calls += 1
        return self.reply


def test_tool_compressor_passes_through_under_trigger_chars():
    llm = FakeCompressorLLM(json.dumps({"summary": "should not be used"}))
    compressor = ToolOutputCompressor(llm, trigger_chars=100)
    text = "short text"
    assert compressor.maybe_compress(text) == text
    assert llm.calls == 0


def test_tool_compressor_compresses_above_trigger_chars():
    llm = FakeCompressorLLM(json.dumps({"summary": "compressed version"}))
    compressor = ToolOutputCompressor(llm, trigger_chars=10)
    out = compressor.maybe_compress("x" * 1000)
    assert out == "compressed version"
    assert llm.calls == 1


def test_tool_compressor_falls_back_to_original_on_malformed_response():
    # Critical difference from MissionDebriefer's failure handling: on ANY
    # failure this must return the ORIGINAL text, never empty/short —
    # silently dropping data here would recreate the exact bug this design
    # exists to fix.
    llm = FakeCompressorLLM("not valid json at all")
    compressor = ToolOutputCompressor(llm, trigger_chars=10)
    original = "x" * 1000
    assert compressor.maybe_compress(original) == original


def test_tool_compressor_falls_back_to_original_on_empty_summary():
    llm = FakeCompressorLLM(json.dumps({"summary": "  "}))
    compressor = ToolOutputCompressor(llm, trigger_chars=10)
    original = "x" * 1000
    assert compressor.maybe_compress(original) == original


def test_tool_compressor_falls_back_to_original_on_llm_exception():
    class RaisingLLM:
        def complete(self, messages, json_schema=None, force_think=False):
            raise RuntimeError("network error")
    compressor = ToolOutputCompressor(RaisingLLM(), trigger_chars=10)
    original = "x" * 1000
    assert compressor.maybe_compress(original) == original


def test_turn_applies_tool_compressor_before_truncation(tmp_path):
    cfg = MistConfig()
    cfg.memory.db_path = str(tmp_path / "test.db")
    cfg.context.max_tool_output_chars = 50
    store = MemoryStore(cfg.db_path)
    skills = SkillRouter("skills_library")
    tools = default_registry(remember_fn=store.remember)
    llm = FakeLLM([
        json.dumps({"action": "use_tool", "tool": "shell",
                    "arguments": {"command": f"echo {'x' * 500}"}}),
        json.dumps({"action": "respond", "response": "done"}),
    ])
    agent = MistAgent(cfg, llm, store, skills, tools)
    agent.tool_compressor = ToolOutputCompressor(
        FakeCompressorLLM(json.dumps({"summary": "COMPRESSED"})), trigger_chars=10)
    result = agent.turn("run it")
    assert "COMPRESSED" in result.tool_trace[0]


async def test_astream_turn_applies_tool_compressor_before_truncation(tmp_path):
    cfg = MistConfig()
    cfg.memory.db_path = str(tmp_path / "test.db")
    cfg.context.max_tool_output_chars = 50
    store = MemoryStore(cfg.db_path)
    skills = SkillRouter("skills_library")
    tools = default_registry(remember_fn=store.remember)

    llm = ScriptedAsyncLLM([
        json.dumps({"action": "use_tool", "tool": "shell",
                    "arguments": {"command": f"echo {'x' * 500}"}}),
        json.dumps({"action": "respond"}),
    ])
    agent = MistAgent(cfg, llm, store, skills, tools)
    agent.tool_compressor = ToolOutputCompressor(
        FakeCompressorLLM(json.dumps({"summary": "COMPRESSED"})), trigger_chars=10)

    events = [e async for e in agent.astream_turn("run it")]
    tool_results = [e.text for e in events if e.kind == "tool_result"]
    assert tool_results and "COMPRESSED" in tool_results[0]


def test_run_tool_subagents_applies_tool_compressor():
    llm = ToolEchoLLM()
    tools = default_registry(remember_fn=lambda c: None)
    compressor = ToolOutputCompressor(
        FakeCompressorLLM(json.dumps({"summary": "COMPRESSED"})), trigger_chars=1)
    results = run_tool_subagents(llm, tools, ["alpha"], max_workers=1, max_steps=3,
                                 tool_compressor=compressor)
    assert len(results) == 1
    assert "COMPRESSED" in results[0].response


def test_read_file_rejects_absolute_path_outside_wiki_and_workspace_roots(tmp_path):
    # The actual regression: read_file hit "not a file" on a wiki-relative
    # path, the model retried with an absolute path "to avoid relative path
    # errors", and that succeeded — reading an unrelated local project's
    # source file that happened to share the same machine as mist itself.
    wiki_root = tmp_path / "wiki"
    wiki_root.mkdir()
    outside = tmp_path / "some-other-project" / "secrets.py"
    outside.parent.mkdir()
    outside.write_text("API_KEY = 'sk-real-secret'")

    tools = default_registry(remember_fn=lambda c: None, wiki_root=wiki_root)
    out = tools.get("read_file").run(path=str(outside))

    assert "sk-real-secret" not in out
    assert "outside Mist's wiki/workspace roots" in out


def test_read_file_allows_absolute_path_inside_workspace_root(tmp_path):
    wiki_root = tmp_path / "wiki"
    wiki_root.mkdir()
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    scratch = workspace_root / "capture_notes.txt"
    scratch.write_text("hello from the sandbox")

    tools = default_registry(remember_fn=lambda c: None, wiki_root=wiki_root,
                             workspace_root=workspace_root)
    out = tools.get("read_file").run(path=str(scratch))

    assert "hello from the sandbox" in out


def test_write_file_allows_absolute_path_inside_configured_extra_root(tmp_path):
    # Regression case from a real run: the htb-engagement skill documents
    # `~/Desktop/HTB/<machine>/` as the operator's engagement workspace, and
    # `shell` writes there unsandboxed — but write_file/read_file/
    # search_files only ever knew about wiki_root and workspace_root, so a
    # write_file there was refused, splitting one engagement's artifacts
    # across two unrelated directory trees. workspace.extra_roots (plumbed
    # through as `extra_roots` here) is the fix: name that directory too.
    wiki_root = tmp_path / "wiki"
    wiki_root.mkdir()
    htb_dir = tmp_path / "Desktop" / "HTB" / "Reactor"
    htb_dir.mkdir(parents=True)
    notes = htb_dir / "notes.md"

    tools = default_registry(remember_fn=lambda c: None, wiki_root=wiki_root,
                             extra_roots=(tmp_path / "Desktop" / "HTB",))
    out = tools.get("write_file").run(path=str(notes), content="# Reactor recon")

    assert "Wrote" in out
    assert notes.read_text() == "# Reactor recon"


def test_write_file_rejects_absolute_path_outside_roots(tmp_path):
    wiki_root = tmp_path / "wiki"
    wiki_root.mkdir()
    outside = tmp_path / "unrelated-project" / "notes.md"

    tools = default_registry(remember_fn=lambda c: None, wiki_root=wiki_root)
    out = tools.get("write_file").run(path=str(outside), content="whatever")

    assert "outside Mist's wiki/workspace roots" in out
    assert not outside.exists()


def test_search_files_rejects_absolute_root_outside_wiki_and_workspace(tmp_path):
    wiki_root = tmp_path / "wiki"
    wiki_root.mkdir()
    outside = tmp_path / "unrelated-project"
    outside.mkdir()
    (outside / "notes.md").write_text("some secret content here")

    tools = default_registry(remember_fn=lambda c: None, wiki_root=wiki_root)
    out = tools.get("search_files").run(query="secret", root=str(outside))

    assert "outside Mist's wiki/workspace roots" in out


# -- process registry (operator kill actually terminates a running command) -

def test_process_registry_kill_active_terminates_running_process():
    proc = subprocess.Popen(["sleep", "5"])
    registry = ProcessRegistry()
    registry.set(proc)
    assert registry.kill_active() is True
    proc.wait(timeout=3)
    assert proc.returncode is not None and proc.returncode != 0


def test_process_registry_kill_active_returns_false_when_nothing_running():
    registry = ProcessRegistry()
    assert registry.kill_active() is False


def test_shell_tool_registers_and_clears_process_registry():
    registry = ProcessRegistry()
    tools = default_registry(remember_fn=lambda c: None, process_registry=registry)
    out = tools.get("shell").run(command="echo hi")
    assert "hi" in out
    assert registry.kill_active() is False  # cleared after completion


def test_finish_objective_tool_always_registered():
    tools = default_registry(remember_fn=lambda c: None)
    tool = tools.get("finish_objective")
    assert tool is not None
    assert "done" in tool.run(summary="user + root flags captured").lower() or \
        "complete" in tool.run(summary="user + root flags captured").lower()


def test_tool_select_always_forces_inclusion_regardless_of_ranking():
    tools = default_registry(remember_fn=lambda c: None)
    selected = tools.select("totally unrelated query about baking",
                            max_exposed=2, always={"finish_objective"})
    assert any(t.name == "finish_objective" for t in selected)
    assert len(selected) == 3  # 2 ranked + 1 forced (additive, see next test)


def test_tool_select_always_is_additive_not_subtracted_from_cap():
    # A forced tool must not crowd out ranked tools — otherwise raising
    # max_exposed's floor for a forced set silently shrinks the ranked
    # pool's own budget (the bug that made `remember` unreachable during
    # early mission testing: max_exposed=5 minus 1 forced slot left only 4
    # of {read_file, write_file, search_files, shell, remember}).
    tools = default_registry(remember_fn=lambda c: None)
    selected = tools.select("totally unrelated query about baking",
                            max_exposed=6, always={"finish_objective"})
    assert len(selected) == 7  # 6 ranked + 1 forced, not 6 total
    names = {t.name for t in selected}
    assert "finish_objective" in names
    assert {"read_file", "write_file", "search_files", "shell", "browse", "remember"} <= names


def test_finish_objective_never_selected_without_always_even_with_keyword_overlap():
    # Regression test for a real incident: finish_objective's own keywords
    # ("complete", "root", "objective", ...) let it outrank and displace
    # `remember` on an ordinary ad hoc chat message ("perform a complete
    # pentest ... get the user and root flags") that was never a mission —
    # it must never appear in the ranked pool unless explicitly forced via
    # `always` (i.e. only during astream_mission).
    tools = default_registry(remember_fn=lambda c: None)
    selected = tools.select(
        "perform a complete pentest of the HTB machine named CAP at "
        "10.129.30.111 to get the user and root flags",
        max_exposed=6,
    )
    names = {t.name for t in selected}
    assert "finish_objective" not in names
    assert "shell" in names
    assert "remember" in names


# -- llm-wiki: init_wiki scaffold ----------------------------------------

def test_init_wiki_creates_expected_skeleton(tmp_path):
    root = tmp_path / "wiki"
    created = init_wiki(root)
    assert set(created) == {
        "raw/", "entities/", "concepts/", "comparisons/", "queries/",
        "SCHEMA.md", "index.md", "log.md",
    }
    for sub in ("raw", "entities", "concepts", "comparisons", "queries"):
        assert (root / sub).is_dir()
    for name in ("SCHEMA.md", "index.md", "log.md"):
        assert (root / name).is_file()


def test_init_wiki_is_idempotent_and_does_not_clobber(tmp_path):
    root = tmp_path / "wiki"
    init_wiki(root)
    custom = "# My custom schema\ntag: my-custom-tag\n"
    (root / "SCHEMA.md").write_text(custom, encoding="utf-8")

    second_run = init_wiki(root)
    assert second_run == []
    assert (root / "SCHEMA.md").read_text(encoding="utf-8") == custom


# -- streaming turn interface (mist.core.agent.MistAgent.astream_turn) ---

class GatedAsyncLLM:
    """Async fake LLM: acomplete() pops scripted decisions in order; astream()
    pops a scripted chunk list per call and blocks on an asyncio.Event before
    yielding, so tests can control ordering deterministically instead of
    racing against real time."""

    def __init__(self, decisions, streams=None):
        self.decisions = list(decisions)
        self.streams = list(streams or [])
        self.release = asyncio.Event()
        self.release.set()  # unblocked by default; tests can .clear() to gate
        self.stream_calls = 0

    async def acomplete(self, messages, json_schema=None, force_think=False):
        return self.decisions.pop(0)

    async def astream(self, messages):
        self.stream_calls += 1
        chunks = self.streams.pop(0)
        await self.release.wait()
        for c in chunks:
            yield c


class ScriptedAsyncLLM:
    """Async fake LLM that pops scripted decisions in call order and streams
    a fixed reply — enough to drive multiple astream_turn() calls across an
    astream_mission() run without racing against real time."""
    def __init__(self, decisions):
        self.decisions = list(decisions)

    async def acomplete(self, messages, json_schema=None, force_think=False):
        return self.decisions.pop(0)

    async def astream(self, messages):
        yield "status"


def make_async_agent(tmp_path, llm):
    cfg = MistConfig()
    cfg.memory.db_path = str(tmp_path / "test.db")
    store = MemoryStore(cfg.db_path)
    skills = SkillRouter("skills_library")
    tools = default_registry(remember_fn=store.remember)
    return MistAgent(cfg, llm, store, skills, tools)


async def test_astream_turn_streams_answer_deltas(tmp_path):
    llm = GatedAsyncLLM(
        decisions=[json.dumps({"action": "respond"})],
        streams=[["Hel", "lo", " there"]],
    )
    agent = make_async_agent(tmp_path, llm)
    events = [e async for e in agent.astream_turn("hi")]
    deltas = [e.text for e in events if e.kind == "delta"]
    done = [e for e in events if e.kind == "done"]
    assert deltas == ["Hel", "lo", " there"]
    assert done and done[0].text == "Hello there"
    # persisted to history like the sync turn() does
    turns = agent.store.recent_turns(agent.session_id, 10)
    assert turns[-1]["content"] == "Hello there"


async def test_astream_turn_runs_tool_then_streams_answer(tmp_path):
    llm = GatedAsyncLLM(
        decisions=[
            json.dumps({"action": "use_tool", "tool": "shell",
                        "arguments": {"command": "echo mist-test"}}),
            json.dumps({"action": "respond"}),
        ],
        streams=[["done"]],
    )
    agent = make_async_agent(tmp_path, llm)
    events = [e async for e in agent.astream_turn("run echo")]
    tool_starts = [e for e in events if e.kind == "tool_start"]
    tool_results = [e for e in events if e.kind == "tool_result"]
    assert tool_starts and tool_starts[0].tool == "shell"
    assert tool_results and "mist-test" in tool_results[0].text
    assert events[-1].kind == "done" and events[-1].text == "done"


async def test_astream_turn_falls_back_to_respond_on_schema_deviant_action(tmp_path):
    # A model can emit valid JSON that's still structurally invalid — here a
    # "use_tool" with no `tool`. (A tool-NAMED action like {"action":
    # "finish_objective", ...} is now repaired to a real use_tool instead — see
    # test_parse_decision_action_coerces_tool_name_in_action_field.) A shape
    # that can't be coerced must fall through to the same free-text answer path
    # as an explicit "respond", not error out.
    llm = GatedAsyncLLM(
        decisions=[json.dumps({"action": "use_tool"})],  # no tool -> uncoercible, invalid
        streams=[["here's my summary"]],
    )
    agent = make_async_agent(tmp_path, llm)
    events = [e async for e in agent.astream_turn("what did you find")]
    assert not any(e.kind == "error" for e in events)
    assert not any(e.kind == "tool_start" for e in events)
    assert events[-1].kind == "done" and events[-1].text == "here's my summary"


async def test_astream_turn_recovers_from_bad_json(tmp_path):
    llm = GatedAsyncLLM(decisions=["not json", json.dumps({"action": "respond"})],
                         streams=[["ok"]])
    agent = make_async_agent(tmp_path, llm)
    events = [e async for e in agent.astream_turn("hi")]
    assert events[-1].kind == "done" and events[-1].text == "ok"


async def test_astream_turn_errors_on_persistent_bad_json(tmp_path):
    llm = GatedAsyncLLM(decisions=["still not json", "also not json"])
    agent = make_async_agent(tmp_path, llm)
    events = [e async for e in agent.astream_turn("hi")]
    assert events[-1].kind == "error"


async def test_astream_turn_refits_budget_across_a_long_tool_chain(tmp_path):
    """A long chain of tool calls in one turn (basic nmap, full nmap, NFS
    enumeration, ...) appends each result to `messages` and never removes it.
    Without a per-step re-fit, the assembled prompt grows past token_budget
    and eventually crowds num_ctx's shared prompt+generation window, so a
    later decision gets truncated mid-string by the backend and fails to
    parse. Assert every decision the model is asked to make stays within
    token_budget no matter how many large results have accumulated."""
    big_output = "X" * 20_000  # each tool result, capped to max_tool_output_chars

    class RecordingLLM:
        def __init__(self, decisions):
            self.decisions = list(decisions)
            self.prompt_tokens: list[int] = []

        async def acomplete(self, messages, json_schema=None, force_think=False):
            self.prompt_tokens.append(sum(_approx_tokens(m["content"]) for m in messages))
            return self.decisions.pop(0)

        async def astream(self, messages):
            yield "done"

    n_steps = 8
    decisions = [json.dumps({"action": "use_tool", "tool": "bigtool", "arguments": {}})] * n_steps
    decisions.append(json.dumps({"action": "respond"}))
    llm = RecordingLLM(decisions)

    cfg = MistConfig()
    cfg.memory.db_path = str(tmp_path / "test.db")
    cfg.context.token_budget = 3000
    cfg.context.max_tool_output_chars = 4000  # ~1000 tokens per result
    cfg.context.max_tool_steps = 20           # room for the whole chain
    store = MemoryStore(cfg.db_path)
    skills = SkillRouter("skills_library")
    tools = default_registry(remember_fn=store.remember)
    tools.register(Tool(name="bigtool", description="returns a large output",
                        parameters={"type": "object", "properties": {}},
                        fn=lambda: big_output))
    agent = MistAgent(cfg, llm, store, skills, tools)

    events = [e async for e in agent.astream_turn("go")]

    # The chain actually ran (a bounded prompt didn't come for free by skipping steps).
    assert sum(1 for e in events if e.kind == "tool_result") == n_steps
    assert len(llm.prompt_tokens) >= n_steps
    # Every decision stayed within budget despite ~1000 tokens/result accumulating —
    # this is exactly what the per-step _fit_budget guarantees. Without the re-fit,
    # step 8's prompt alone would carry 8 * ~1000 tokens of results, far over 3000.
    assert max(llm.prompt_tokens) <= cfg.context.token_budget


# -- resilience: a crashed LLM/network call must not crash the caller ----

class RaisingLLM:
    """Fake LLM whose calls always raise — simulates a network/backend
    failure (timeout, connection reset, HTTP error) partway through a turn."""
    def complete(self, messages, json_schema=None, force_think=False):
        raise ConnectionError("backend unreachable")

    async def acomplete(self, messages, json_schema=None, force_think=False):
        raise ConnectionError("backend unreachable")

    async def astream(self, messages):
        raise ConnectionError("backend unreachable")
        yield ""  # pragma: no cover - unreachable, makes this an async generator


class RaisingMidStreamLLM:
    async def acomplete(self, messages, json_schema=None, force_think=False):
        return json.dumps({"action": "respond"})

    async def astream(self, messages):
        yield "partial "
        raise ConnectionError("connection reset mid-stream")


def test_turn_falls_back_to_respond_on_schema_deviant_action(tmp_path):
    # Same as the astream_turn version, for the sync turn() path used by
    # `mist chat`/`ask`: an uncoercible invalid shape (use_tool with no tool)
    # degrades to respond. (Tool-named actions now coerce — see the parser test.)
    agent = make_agent(tmp_path, [
        json.dumps({"action": "use_tool"}),
        "here's my summary",
    ])
    result = agent.turn("what did you find")
    assert result.response == "here's my summary"
    assert result.tool_trace == []


def test_turn_survives_llm_call_exception(tmp_path):
    cfg = MistConfig()
    cfg.memory.db_path = str(tmp_path / "test.db")
    store = MemoryStore(cfg.db_path)
    skills = SkillRouter("skills_library")
    tools = default_registry(remember_fn=store.remember)
    agent = MistAgent(cfg, RaisingLLM(), store, skills, tools)

    result = agent.turn("scan the target")
    assert "ERROR" in result.response and "backend unreachable" in result.response
    # Must still persist something — an empty, turn-less session is exactly
    # what a crash used to leave behind.
    turns = agent.store.recent_turns(agent.session_id, 10)
    assert turns and turns[-1]["content"] == result.response


async def test_astream_turn_survives_llm_call_exception(tmp_path):
    agent = make_async_agent(tmp_path, RaisingLLM())
    events = [e async for e in agent.astream_turn("scan the target")]
    assert events[-1].kind == "error"


class EmptyMessageRaisingLLM:
    """Simulates a real incident: some low-level connection exceptions have
    an empty str() (a dropped socket, a reset connection with no message),
    so f"...{exc}" renders as a bare, undiagnosable "LLM call failed: " —
    confirmed live, three separate times in one real mission with zero clue
    which exception actually fired."""
    def complete(self, messages, json_schema=None, force_think=False):
        raise ConnectionResetError()

    async def acomplete(self, messages, json_schema=None, force_think=False):
        raise ConnectionResetError()

    async def astream(self, messages):
        raise ConnectionResetError()
        yield  # pragma: no cover - unreachable, makes this an async generator


def test_turn_error_message_names_exception_type_even_when_str_is_empty(tmp_path):
    assert str(ConnectionResetError()) == ""  # sanity-check the premise
    agent = make_agent(tmp_path, [])
    agent.llm = EmptyMessageRaisingLLM()
    result = agent.turn("scan the target")
    assert "ConnectionResetError" in result.response


async def test_astream_turn_error_message_names_exception_type_even_when_str_is_empty(tmp_path):
    agent = make_async_agent(tmp_path, EmptyMessageRaisingLLM())
    events = [e async for e in agent.astream_turn("scan the target")]
    assert events[-1].kind == "error"
    assert "ConnectionResetError" in events[-1].text


async def test_stream_answer_survives_mid_stream_exception(tmp_path):
    agent = make_async_agent(tmp_path, RaisingMidStreamLLM())
    events = [e async for e in agent.astream_turn("hi")]
    assert events[-1].kind == "error"
    assert "connection reset" in events[-1].text
    # The partial text generated before the crash isn't silently discarded.
    turns = agent.store.recent_turns(agent.session_id, 10)
    assert turns and "partial" in turns[-1]["content"]


class EmptyMessageRaisingMidStreamLLM:
    """Same premise as EmptyMessageRaisingLLM, but the empty-message
    exception fires mid-stream (in _stream_answer) rather than at the
    decision step — the third of the three "LLM call failed" sites this
    incident affected."""
    async def acomplete(self, messages, json_schema=None, force_think=False):
        return json.dumps({"action": "respond"})

    async def astream(self, messages):
        yield "partial "
        raise ConnectionResetError()


async def test_stream_answer_error_message_names_exception_type_even_when_str_is_empty(tmp_path):
    agent = make_async_agent(tmp_path, EmptyMessageRaisingMidStreamLLM())
    events = [e async for e in agent.astream_turn("hi")]
    assert events[-1].kind == "error"
    assert "ConnectionResetError" in events[-1].text


async def test_mission_pauses_on_turn_error_instead_of_spinning(tmp_path):
    cfg = MistConfig()
    cfg.memory.db_path = str(tmp_path / "test.db")
    cfg.wiki.root_path = str(tmp_path / "wiki")
    store = MemoryStore(cfg.db_path)
    skills = SkillRouter("skills_library")
    tools = default_registry(remember_fn=store.remember, wiki_root=cfg.wiki_root)
    agent = MistAgent(cfg, RaisingLLM(), store, skills, tools)
    control = MissionControl()

    events: list = []

    async def drive():
        async for ev in agent.astream_mission("recon", control, max_turns=10):
            events.append(ev)

    task = asyncio.create_task(drive())
    for _ in range(200):
        if any(e.kind == "stuck" for e in events):
            break
        await asyncio.sleep(0.01)
    assert control.paused
    # Must not have burned through the whole 10-turn budget in a tight loop.
    assert [e.kind for e in events].count("turn_start") == 1

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_chat_repl_survives_llm_failure_via_graceful_turn_result(tmp_path, monkeypatch):
    """agent.turn() now catches LLM/network failures itself (see
    test_turn_survives_llm_call_exception), so `mist chat` should show a
    clean ERROR reply and keep prompting — not crash the process. This was
    a real incident: `mist chat` had no exception handling around
    agent.turn() at all, found via ~/.mist/mist.db showing five sessions
    created back-to-back with zero turns in any of them, consistent with
    repeated crash-and-restart cycles."""
    from mist import cli as cli_module

    cfg = MistConfig()
    cfg.memory.db_path = str(tmp_path / "test.db")
    store = MemoryStore(cfg.db_path)
    skills = SkillRouter("skills_library")
    tools = default_registry(remember_fn=store.remember)
    agent = MistAgent(cfg, RaisingLLM(), store, skills, tools)

    monkeypatch.setattr(cli_module, "_build_agent", lambda *a, **k: agent)

    inputs = iter(["hello", "still alive?"])

    def fake_input(prompt=""):
        try:
            return next(inputs)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr(cli_module.console, "input", fake_input)
    printed: list[str] = []
    monkeypatch.setattr(cli_module.console, "print",
                        lambda *a, **k: printed.append(" ".join(str(x) for x in a)))

    cli_module.chat(config=None, backend=None, model=None, base_url=None, skills=None)  # must not raise

    text = "\n".join(printed)
    assert text.count("ERROR: LLM call failed") == 2  # both turns failed gracefully
    assert "bye" in text  # reached the clean EOF exit afterward, session still alive
    turns = agent.store.recent_turns(agent.session_id, 10)
    assert len(turns) == 4  # both crashed turns were still persisted, not lost


def test_chat_repl_survives_unexpected_non_llm_exception(tmp_path, monkeypatch):
    """Defense in depth: even a bug unrelated to the LLM call (e.g. inside
    tool selection or memory search) must not kill the whole REPL — this
    exercises `chat()`'s own try/except around agent.turn(), independent of
    turn()'s internal LLM-failure handling."""
    from mist import cli as cli_module

    cfg = MistConfig()
    cfg.memory.db_path = str(tmp_path / "test.db")
    store = MemoryStore(cfg.db_path)
    skills = SkillRouter("skills_library")
    tools = default_registry(remember_fn=store.remember)
    agent = MistAgent(cfg, RaisingLLM(), store, skills, tools)
    monkeypatch.setattr(agent, "turn",
                        lambda user_msg: (_ for _ in ()).throw(RuntimeError("unrelated bug")))

    monkeypatch.setattr(cli_module, "_build_agent", lambda *a, **k: agent)

    inputs = iter(["hello", "still alive?"])

    def fake_input(prompt=""):
        try:
            return next(inputs)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr(cli_module.console, "input", fake_input)
    printed: list[str] = []
    monkeypatch.setattr(cli_module.console, "print",
                        lambda *a, **k: printed.append(" ".join(str(x) for x in a)))

    cli_module.chat(config=None, backend=None, model=None, base_url=None, skills=None)  # must not raise

    text = "\n".join(printed)
    assert text.count("turn crashed") == 2
    assert "bye" in text


async def test_astream_turn_can_be_cancelled_mid_stream(tmp_path):
    llm = GatedAsyncLLM(decisions=[json.dumps({"action": "respond"})], streams=[["a", "b", "c"]])
    llm.release.clear()  # block astream indefinitely until we release it
    agent = make_async_agent(tmp_path, llm)

    async def consume():
        async for _ in agent.astream_turn("hi"):
            pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()


# -- streaming TUI (mist.tui.app.MistTUI) ---------------------------------

textual = pytest.importorskip("textual", reason="TUI tests need the optional `textual` extra")

from mist.tui.app import MistTUI  # noqa: E402
from textual.widgets import RichLog, Static  # noqa: E402


def _transcript_text(app) -> str:
    log = app.query_one("#transcript", RichLog)
    return "\n".join(str(line) for line in log.lines)


def test_format_tool_call_renders_compact_summary_not_raw_json():
    from mist.tui.render import format_tool_call

    out = format_tool_call("shell", '{"command": "ls -la"}')
    assert out == "shell(command='ls -la')"
    assert "{" not in out and "}" not in out


def test_format_tool_call_shows_full_shell_command_untruncated():
    # An operator monitoring the feed must see the ENTIRE command being run —
    # a long shell command must not be elided (the old 37/64-char truncation
    # hid most of what mist was doing against the target).
    from mist.tui.render import format_tool_call

    cmd = ("gobuster dir -u http://enigma.htb/ -w /usr/share/wordlists/dirbuster/"
           "directory-list-2.3-medium.txt -t 50 -o ~/Desktop/HTB/Enigma/gobuster.txt")
    out = format_tool_call("shell", json.dumps({"command": cmd}))
    assert cmd in out          # full command present verbatim
    assert "…" not in out      # nothing elided
    assert out == f"shell(command={cmd!r})"


def test_format_tool_call_still_truncates_non_shell_tool_args():
    # The full-command exception is scoped to shell; other tools/args stay
    # compact so a big write_file body doesn't flood the feed.
    from mist.tui.render import format_tool_call

    big = "x" * 200
    out = format_tool_call("write_file", json.dumps({"path": "a.md", "content": big}))
    assert "…" in out and len(out) < 120


def test_format_tool_call_falls_back_on_unparseable_detail():
    from mist.tui.render import format_tool_call

    assert format_tool_call("shell", "not json") == "shell(...)"
    assert format_tool_call("shell", "") == "shell()"


def test_format_status_includes_queued_and_context_usage():
    from mist.tui.render import format_status

    text = format_status(state="thinking…", elapsed=12.0, model="qwen2.5:14b",
                         used_tokens=2048, budget=6000, session_id=44, queued=3)
    assert "3 queued" in text
    assert "ctx 34% (2048/6000)" in text
    assert "session 44" in text

    idle = format_status(state="idle", elapsed=None, model="qwen2.5:14b",
                         used_tokens=0, budget=6000, session_id=1, queued=0)
    assert "queued" not in idle
    assert "ctx 0% (0/6000)" in idle


async def test_tui_onboarding_panel_shows_tool_and_skill_counts():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        agent = make_agent(Path(tmp), ["reply"])
        app = MistTUI(agent)
        async with app.run_test() as pilot:
            await pilot.pause()
            text = _transcript_text(app)
            assert "tools · " in text
            assert "skills · " in text
            assert "Available Tools" in text


async def test_tui_streams_response_live():
    llm = GatedAsyncLLM(decisions=[json.dumps({"action": "respond"})], streams=[["Hel", "lo"]])
    llm.release.clear()

    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        agent = make_async_agent(Path(tmp), llm)
        app = MistTUI(agent)
        async with app.run_test() as pilot:
            await pilot.click("#input")
            await pilot.press(*"hi")
            await pilot.press("enter")
            await pilot.pause()
            llm.release.set()
            await pilot.pause(0.2)
            assert "Hello" in _transcript_text(app)


async def test_tui_turn_populates_real_context_token_count():
    # The status line shows agent.last_context_tokens, not a synthetic
    # estimate — confirm a completed turn actually populates it (not just
    # that the attribute exists at its zero default).
    llm = GatedAsyncLLM(decisions=[json.dumps({"action": "respond"})], streams=[["hi"]])
    llm.release.clear()

    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        agent = make_async_agent(Path(tmp), llm)
        assert agent.last_context_tokens == 0
        app = MistTUI(agent)
        async with app.run_test() as pilot:
            await pilot.click("#input")
            await pilot.press(*"hi")
            await pilot.press("enter")
            await pilot.pause()
            llm.release.set()
            await pilot.pause(0.2)
            assert agent.last_context_tokens > 0


async def test_tui_queues_messages_submitted_mid_turn():
    llm = GatedAsyncLLM(
        decisions=[json.dumps({"action": "respond"}), json.dumps({"action": "respond"})],
        streams=[["first-answer"], ["second-answer"]],
    )
    llm.release.clear()

    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        agent = make_async_agent(Path(tmp), llm)
        app = MistTUI(agent)
        async with app.run_test() as pilot:
            await pilot.click("#input")
            await pilot.press(*"one")
            await pilot.press("enter")
            await pilot.pause()  # decision resolves; astream() blocks on release
            assert llm.stream_calls == 1

            await pilot.press(*"two")
            await pilot.press("enter")
            await pilot.pause()
            status = str(app.query_one("#status", Static).content)
            assert "1 queued" in status, status
            assert llm.stream_calls == 1  # second turn must not have started yet

            llm.release.set()
            await pilot.pause(0.3)
            text = _transcript_text(app)
            assert "first-answer" in text and "second-answer" in text, text
            assert llm.stream_calls == 2


async def test_tui_ctrl_c_interrupts_without_quitting():
    class InfiniteLLM:
        async def acomplete(self, messages, json_schema=None, force_think=False):
            return json.dumps({"action": "respond"})

        async def astream(self, messages):
            while True:
                await asyncio.sleep(0.01)
                yield "a"

    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        agent = make_async_agent(Path(tmp), InfiniteLLM())
        app = MistTUI(agent)
        async with app.run_test() as pilot:
            await pilot.click("#input")
            await pilot.press(*"go")
            await pilot.press("enter")
            await pilot.pause(0.2)
            assert app._turn_task is not None and not app._turn_task.done()

            await pilot.press("ctrl+c")
            await pilot.pause(0.2)
            assert app._turn_task is None
            assert "interrupted" in _transcript_text(app)
            assert app.is_running  # Ctrl+C must not have quit the app


# -- streaming TUI: mission mode (/mission, /pause, /resume, kill) -------

class GatedMissionLLM:
    """Async fake LLM for mission tests: acomplete() blocks on a manually
    released gate until told otherwise, giving deterministic, race-free
    control over exactly when each mission turn's decision resolves —
    wall-clock sleeps would race against Textual pilot overhead, which
    varies enough to make fixed delays unreliable. release() lets exactly
    the next call through; release_all() stops gating for the rest of the
    run once precise control is no longer needed."""
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.gate = asyncio.Event()
        self.auto = False

    async def acomplete(self, messages, json_schema=None, force_think=False):
        if not self.auto:
            await self.gate.wait()
            self.gate.clear()
        return self.decisions.pop(0)

    async def astream(self, messages):
        yield "status"

    def release(self) -> None:
        self.gate.set()

    def release_all(self) -> None:
        self.auto = True
        self.gate.set()


def make_mission_tui_agent(tmp_path, llm):
    from mist.tools.registry import ProcessRegistry

    cfg = MistConfig()
    cfg.memory.db_path = str(tmp_path / "test.db")
    cfg.wiki.root_path = str(tmp_path / "wiki")
    store = MemoryStore(cfg.db_path)
    skills = SkillRouter("skills_library")
    registry = ProcessRegistry()
    tools = default_registry(remember_fn=store.remember, wiki_root=cfg.wiki_root,
                             process_registry=registry)
    return MistAgent(cfg, llm, store, skills, tools, process_registry=registry)


async def test_tui_mission_runs_autonomously_across_turns(tmp_path):
    agent = make_mission_tui_agent(tmp_path, ScriptedAsyncLLM([
        json.dumps({"action": "use_tool", "tool": "shell", "arguments": {"command": "echo recon"}}),
        json.dumps({"action": "respond"}),
        json.dumps({"action": "use_tool", "tool": "finish_objective",
                    "arguments": {"summary": "done"}}),
        json.dumps({"action": "respond"}),
    ]))
    app = MistTUI(agent)
    async with app.run_test() as pilot:
        await pilot.click("#input")
        for ch in "/mission pwn the box":
            await pilot.press(ch)
        await pilot.press("enter")
        for _ in range(50):
            await pilot.pause(0.05)
            if app._mission_task is None:
                break
        text = _transcript_text(app)
        assert "mission started" in text
        assert "mission finished" in text
        assert app._mission_task is None  # cleared once the mission concludes


async def test_tui_mission_surfaces_unexpected_crash_not_silently(tmp_path):
    # Regression (session 128): an unhandled exception escaping astream_mission
    # ended the mission task with NO operator-visible event — "no mission is
    # running" with no clue why. It must instead surface a terminal event and
    # a durable traceback in the mission log, like the operator-kill path.
    from mist.core.agent import MissionEvent

    agent = make_mission_tui_agent(tmp_path, ScriptedAsyncLLM([json.dumps({"action": "respond"})]))
    log_path = tmp_path / "mission.md"
    log_path.write_text("# Mission log\n")

    async def _boom(*a, **k):
        yield MissionEvent(kind="started", text=str(log_path))
        raise RuntimeError("kaboom while processing tool output")

    agent.astream_mission = _boom
    agent.debrief_mission = lambda *a, **k: type("D", (), {"summary": lambda self: "debrief ok"})()

    app = MistTUI(agent)
    async with app.run_test() as pilot:
        await pilot.click("#input")
        for ch in "/mission crash test":
            await pilot.press(ch)
        await pilot.press("enter")
        for _ in range(50):
            await pilot.pause(0.05)
            if app._mission_task is None:
                break

        text = _transcript_text(app)
        assert "ended unexpectedly" in text            # operator sees a terminal event
        assert "kaboom" in text                        # ... naming the actual error
        assert app._mission_task is None               # task cleared, not left dangling
        assert "kaboom" in log_path.read_text()        # durable traceback persisted to the log


async def test_tui_mission_pause_and_resume(tmp_path):
    llm = GatedMissionLLM([
        json.dumps({"action": "respond"}),
        json.dumps({"action": "respond"}),
        json.dumps({"action": "use_tool", "tool": "finish_objective", "arguments": {"summary": "ok"}}),
        json.dumps({"action": "respond"}),
    ])
    agent = make_mission_tui_agent(tmp_path, llm)
    app = MistTUI(agent)
    async with app.run_test() as pilot:
        await pilot.click("#input")
        for ch in "/mission long objective":
            await pilot.press(ch)
        await pilot.press("enter")
        await pilot.pause(0.05)

        # Let turn 1's decision resolve, then wait until turn 2 is blocked on
        # its own decision call — this is the deterministic signal that turn
        # 1 fully completed, with no wall-clock guessing involved.
        llm.release()
        for _ in range(100):
            await pilot.pause(0.02)
            if len(llm.decisions) == 3:
                break
        assert len(llm.decisions) == 3

        for ch in "/pause":
            await pilot.press(ch)
        await pilot.press("enter")
        await pilot.pause(0.05)
        assert app._mission_control is not None and app._mission_control.paused

        # Releasing now lets turn 2's in-flight decision finish (an in-flight
        # step always completes even when paused) — but turn 3 must NOT
        # start, since the pause is checked before the next turn begins.
        llm.release()
        for _ in range(100):
            await pilot.pause(0.02)
            if len(llm.decisions) == 2:
                break
        assert len(llm.decisions) == 2
        await pilot.pause(0.2)
        assert len(llm.decisions) == 2  # still blocked — turn 3 never started

        llm.release_all()
        for ch in "/resume":
            await pilot.press(ch)
        await pilot.press("enter")
        for _ in range(100):
            await pilot.pause(0.05)
            if app._mission_task is None:
                break
        assert "mission finished" in _transcript_text(app)


async def test_tui_plain_message_becomes_mission_note_while_running(tmp_path):
    llm = GatedMissionLLM([
        json.dumps({"action": "respond"}),
        json.dumps({"action": "use_tool", "tool": "finish_objective", "arguments": {"summary": "ok"}}),
        json.dumps({"action": "respond"}),
    ])
    agent = make_mission_tui_agent(tmp_path, llm)
    app = MistTUI(agent)
    async with app.run_test() as pilot:
        await pilot.click("#input")
        for ch in "/mission slow objective":
            await pilot.press(ch)
        await pilot.press("enter")
        await pilot.pause(0.05)  # mission is now blocked waiting on turn 1's decision

        for ch in "skip recon, go straight to the web app":
            await pilot.press(ch)
        await pilot.press("enter")
        await pilot.pause(0.05)
        assert "noted" in _transcript_text(app)
        assert app.queue.qsize() == 0  # no separate ad hoc turn was queued
        assert app._mission_control.notes == ["skip recon, go straight to the web app"]

        llm.release_all()
        for _ in range(100):
            await pilot.pause(0.05)
            if app._mission_task is None:
                break


async def test_tui_ctrl_c_kills_mission_and_terminates_subprocess(tmp_path):
    agent = make_mission_tui_agent(tmp_path, ScriptedAsyncLLM([
        json.dumps({"action": "use_tool", "tool": "shell", "arguments": {"command": "sleep 5"}}),
    ]))
    app = MistTUI(agent)
    async with app.run_test() as pilot:
        await pilot.click("#input")
        for ch in "/mission run a slow command":
            await pilot.press(ch)
        await pilot.press("enter")

        # Wait until the shell tool has actually registered the live `sleep 5` process.
        for _ in range(100):
            await pilot.pause(0.05)
            if agent.process_registry._proc is not None:
                break
        assert agent.process_registry._proc is not None
        proc = agent.process_registry._proc

        await pilot.press("ctrl+c")
        for _ in range(100):
            await pilot.pause(0.05)
            if app._mission_task is None:
                break
        assert app._mission_task is None
        assert "mission killed" in _transcript_text(app)
        proc.wait(timeout=3)
        assert proc.returncode != 0  # terminated, not a natural 5s completion


async def test_tui_kill_triggers_debrief(tmp_path):
    # A killed mission must still be debriefed — whatever it found before
    # being stopped shouldn't be silently lost, same as a natural finish.
    class DebriefableLLM(ScriptedAsyncLLM):
        def complete(self, messages, json_schema=None, force_think=False):
            return json.dumps({"memories": [{"content": "confirmed something before being killed"}]})

    agent = make_mission_tui_agent(tmp_path, DebriefableLLM([
        json.dumps({"action": "use_tool", "tool": "shell", "arguments": {"command": "sleep 5"}}),
    ]))
    app = MistTUI(agent)
    async with app.run_test() as pilot:
        await pilot.click("#input")
        for ch in "/mission run a slow command":
            await pilot.press(ch)
        await pilot.press("enter")

        for _ in range(100):
            await pilot.pause(0.05)
            if agent.process_registry._proc is not None:
                break
        assert agent.process_registry._proc is not None

        await pilot.press("ctrl+c")
        for _ in range(100):
            await pilot.pause(0.05)
            if app._mission_task is None:
                break
        assert app._mission_task is None
        text = _transcript_text(app)
        assert "mission killed" in text
        assert "Debrief:" in text
        assert any("confirmed something before being killed" in h
                   for h in agent.store.search("confirmed something"))


# -- LLMClient async streaming wire format (mist.llm.client) --------------

import httpx  # noqa: E402

from mist.llm.client import (LLMClient, compute_max_tokens, compute_num_ctx,  # noqa: E402
                             compute_timeout)


# -- context-window discovery (mist.llm.client) ---------------------------

def test_compute_num_ctx_clamps_to_discovered_max():
    # Ollama's num_ctx is one shared window for the prompt AND the
    # generation — the desired size is their sum, but it must never exceed
    # what the model actually supports.
    assert compute_num_ctx(token_budget=6000, max_tokens=256000, discovered_max=262144) == 262000
    assert compute_num_ctx(token_budget=6000, max_tokens=256000, discovered_max=8192) == 8192


def test_answer_system_asserts_real_tool_access(tmp_path):
    # Regression case from a real run: after a long think-heavy turn with
    # no tool call, the free-text answer step claimed "I don't have
    # network access or an active SSH shell" — a flat-out hallucinated
    # limitation, since ANSWER_SYSTEM_TEMPLATE never actually told the
    # model it has real tool access, only how to talk about tool output
    # *if* there was any.
    cfg = MistConfig()
    cfg.memory.db_path = str(tmp_path / "test.db")
    store = MemoryStore(cfg.db_path)
    skills = SkillRouter("skills_library")
    tools = default_registry(remember_fn=store.remember)
    agent = MistAgent(cfg, FakeLLM([]), store, skills, tools)
    system = agent._build_answer_system("what's my status")
    assert "not a text-only assistant" in system
    assert "claim you lack network access" in system


# -- action validation (mist.core.action) ----------------------------------

def test_parse_decision_action_valid_respond_and_use_tool():
    respond = parse_decision_action({"action": "respond"})
    assert isinstance(respond, DecisionRespond)

    use_tool = parse_decision_action(
        {"action": "use_tool", "tool": "shell", "arguments": {"command": "ls"}})
    assert isinstance(use_tool, DecisionUseTool)
    assert use_tool.tool == "shell"
    assert use_tool.arguments == {"command": "ls"}


def test_parse_decision_action_defaults_missing_arguments_to_empty_dict():
    use_tool = parse_decision_action({"action": "use_tool", "tool": "shell"})
    assert use_tool.arguments == {}


def test_parse_decision_action_coerces_tool_name_in_action_field():
    # The dominant malformed shape from a non-reasoning/opening decision: the
    # TOOL NAME sits in `action` with args at top level. Confirmed ~5/5 on both
    # qwen3.6 models with think off. Must be coerced to a real use_tool (act),
    # not raised/narrated. Two arg placements — top-level key and explicit
    # `arguments` — both repaired.
    a = parse_decision_action({"action": "shell", "command": "nmap -sV 10.0.0.1"})
    assert isinstance(a, DecisionUseTool)
    assert a.tool == "shell" and a.arguments == {"command": "nmap -sV 10.0.0.1"}

    b = parse_decision_action({"action": "finish_objective", "summary": "done"})
    assert isinstance(b, DecisionUseTool)
    assert b.tool == "finish_objective" and b.arguments == {"summary": "done"}

    c = parse_decision_action({"action": "shell", "arguments": {"command": "ls"}})
    assert isinstance(c, DecisionUseTool)
    assert c.tool == "shell" and c.arguments == {"command": "ls"}


def test_parse_decision_action_coercion_leaves_respond_and_use_tool_untouched():
    # Coercion must only touch tool-named actions — the two valid shapes pass
    # through exactly as before.
    assert isinstance(parse_decision_action({"action": "respond"}), DecisionRespond)
    ut = parse_decision_action({"action": "use_tool", "tool": "shell",
                                "arguments": {"command": "id"}})
    assert isinstance(ut, DecisionUseTool) and ut.tool == "shell"


def test_parse_decision_action_rejects_use_tool_missing_tool_field():
    # action == "use_tool" is left untouched by coercion, so a missing `tool`
    # still raises rather than being silently repaired.
    with pytest.raises(ActionValidationError):
        parse_decision_action({"action": "use_tool"})


def test_parse_decision_action_rejects_missing_action_key():
    # No `action` at all is not a tool-named shape (nothing to coerce) — raises.
    with pytest.raises(ActionValidationError):
        parse_decision_action({"tool": "shell", "arguments": {}})


def test_parse_subagent_action_valid_respond_and_use_tool():
    respond = parse_subagent_action({"action": "respond", "response": "done"})
    assert isinstance(respond, SubagentRespond)
    assert respond.response == "done"

    use_tool = parse_subagent_action(
        {"action": "use_tool", "tool": "shell", "arguments": {"command": "ls"}})
    assert isinstance(use_tool, SubagentUseTool)
    assert use_tool.tool == "shell"


def test_parse_subagent_action_respond_defaults_missing_response_to_empty_string():
    respond = parse_subagent_action({"action": "respond"})
    assert respond.response == ""


def test_parse_subagent_action_rejects_finish_objective_incident():
    # The exact real-world shape that slipped through the old duck-typed
    # check: {"action": "finish_objective", "summary": "..."}.
    with pytest.raises(ActionValidationError) as exc_info:
        parse_subagent_action({"action": "finish_objective", "summary": "done scanning"})
    assert "finish_objective" in str(exc_info.value)


def test_compute_max_tokens_clamps_unrealistic_configured_value():
    # Regression case from a real run: a fixed generation.max_tokens tuned
    # for one model (256000) is meaningless (or worse, wasteful/incorrect)
    # against a model with a much smaller real context window.
    assert compute_max_tokens(configured_max_tokens=256000, token_budget=6000,
                              discovered_max=32768) == 26768
    # Already fits: left unchanged.
    assert compute_max_tokens(configured_max_tokens=8192, token_budget=6000,
                              discovered_max=262144) == 8192


def test_compute_max_tokens_never_raises_above_configured_value():
    # Only tightens an unrealistic value — never second-guesses upward.
    assert compute_max_tokens(configured_max_tokens=1024, token_budget=100,
                              discovered_max=262144) == 1024


def test_compute_max_tokens_passthrough_when_discovery_failed():
    assert compute_max_tokens(configured_max_tokens=256000, token_budget=6000,
                              discovered_max=None) == 256000


def test_compute_num_ctx_returns_none_when_discovery_failed():
    # Picking an arbitrary fallback number here could just as easily be
    # wrong as not setting num_ctx at all — None means "leave the backend's
    # own default alone," today's behavior, unchanged.
    assert compute_num_ctx(token_budget=6000, max_tokens=1024, discovered_max=None) is None


def test_compute_timeout_scales_with_max_tokens():
    # Regression case from a real run: generation.max_tokens was raised to
    # 40000 (~10 min budget at a conservative 65 tok/s) specifically to
    # give a stuck reasoning pass more room, but the HTTP client's
    # hardcoded 120s timeout never followed along — a real call was
    # aborted by httpx.ReadTimeout after 2 minutes, long before the
    # ~10-minute budget max_tokens was actually sized for.
    timeout = compute_timeout(40000)
    assert timeout > 120.0
    assert 600.0 < timeout < 700.0  # ~40000/65 + 30s overhead


def test_compute_timeout_floors_at_min_timeout_for_small_max_tokens():
    # A small max_tokens (e.g. the 1024 default) must not shrink the
    # timeout below the old, safe 120s default.
    assert compute_timeout(1024) == 120.0
    assert compute_timeout(100) == 120.0


def test_compute_timeout_respects_custom_tokens_per_second():
    # Half the assumed throughput roughly doubles the token-generation
    # portion of the timeout.
    fast = compute_timeout(40000, tokens_per_second=65.0)
    slow = compute_timeout(40000, tokens_per_second=32.5)
    assert slow > fast


def test_discover_context_length_extracts_family_prefixed_key():
    # The `model_info` key name varies by architecture family
    # ("qwen35moe.context_length", "llama.context_length", ...) — this
    # must work without hardcoding one family.
    def handler(request):
        return httpx.Response(200, json={
            "model_info": {"general.architecture": "qwen35moe",
                           "qwen35moe.context_length": 262144,
                           "qwen35moe.embedding_length": 5120},
        })
    client = LLMClient("ollama", "http://fake", "qwen3.6:35b-a3b-q4_K_M")
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    assert client.discover_context_length() == 262144


def test_discover_context_length_returns_none_on_network_failure():
    def handler(request):
        raise httpx.ConnectError("connection refused")
    client = LLMClient("ollama", "http://fake", "m")
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    assert client.discover_context_length() is None


def test_discover_context_length_uses_short_timeout_not_full_client_timeout():
    # Regression case from a real run: with no explicit timeout here, this
    # shared the LLMClient's general timeout (120s, sized for slow
    # completions) — an unreachable server made `mist tui` look hung for
    # two full minutes before ever showing a prompt, since this call
    # happens at startup before anything renders.
    captured = {}

    class FakeSyncClient:
        def post(self, url, json=None, timeout=None):
            captured["timeout"] = timeout
            raise ConnectionError("simulated unreachable")

    client = LLMClient("ollama", "http://fake", "m", timeout=120.0)
    client._client = FakeSyncClient()
    assert client.discover_context_length() is None
    assert captured["timeout"] is not None
    assert captured["timeout"] < 120.0


def test_discover_context_length_returns_none_when_key_missing():
    def handler(request):
        return httpx.Response(200, json={"model_info": {"general.architecture": "llama"}})
    client = LLMClient("ollama", "http://fake", "m")
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    assert client.discover_context_length() is None


def test_discover_context_length_returns_none_for_vllm_backend():
    client = LLMClient("vllm", "http://fake/v1", "m")
    assert client.discover_context_length() is None


def test_ollama_options_omits_num_ctx_when_unset():
    def handler(request):
        body = json.loads(request.content)
        assert "num_ctx" not in body["options"]
        return httpx.Response(200, json={"message": {"content": "ok"}})
    client = LLMClient("ollama", "http://fake", "m")
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    client.complete([{"role": "user", "content": "hi"}])


def test_ollama_options_includes_num_ctx_when_set():
    def handler(request):
        body = json.loads(request.content)
        assert body["options"]["num_ctx"] == 32768
        return httpx.Response(200, json={"message": {"content": "ok"}})
    client = LLMClient("ollama", "http://fake", "m")
    client.num_ctx = 32768
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    client.complete([{"role": "user", "content": "hi"}])


async def test_aollama_options_includes_num_ctx_when_set():
    def handler(request):
        body = json.loads(request.content)
        assert body["options"]["num_ctx"] == 32768
        return httpx.Response(200, json={"message": {"content": "ok"}})
    client = LLMClient("ollama", "http://fake", "m")
    client.num_ctx = 32768
    client._aclient = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    await client.acomplete([{"role": "user", "content": "hi"}])


def test_decision_num_predict_caps_at_min_of_decision_and_max():
    c = LLMClient("ollama", "http://fake", "m", max_tokens=40000, decision_max_tokens=4096)
    assert c._decision_num_predict() == 4096                # capped below the huge answer budget
    c2 = LLMClient("ollama", "http://fake", "m", max_tokens=1024, decision_max_tokens=4096)
    assert c2._decision_num_predict() == 1024               # never exceeds the overall budget
    c3 = LLMClient("ollama", "http://fake", "m", max_tokens=40000)  # decision_max_tokens None
    assert c3._decision_num_predict() is None               # uncapped -> full max_tokens


def test_decision_call_uses_capped_num_predict_but_answer_uses_full():
    # A schema-constrained decision call must use the small decision cap (so a
    # forced-think decision can't reason for the full, unguarded max_tokens);
    # an unconstrained call keeps the full budget for the answer step.
    seen = {}

    def handler(request):
        body = json.loads(request.content)
        seen["constrained" if "format" in body else "free"] = body["options"]["num_predict"]
        return httpx.Response(200, json={"message": {"content": '{"action": "respond"}'}})

    client = LLMClient("ollama", "http://fake", "m", max_tokens=40000, decision_max_tokens=4096)
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    client.complete([{"role": "user", "content": "hi"}],
                    json_schema={"type": "object"})            # decision call
    client.complete([{"role": "user", "content": "hi"}])       # free-text (answer-like) call
    assert seen["constrained"] == 4096
    assert seen["free"] == 40000


async def test_astream_ollama_options_includes_num_ctx_when_set():
    def handler(request):
        body = json.loads(request.content)
        assert body["options"]["num_ctx"] == 32768
        return httpx.Response(200, content='{"message": {"content": "hi"}, "done": true}\n')
    client = LLMClient("ollama", "http://fake", "m", think=False)
    client.num_ctx = 32768
    client._aclient = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    _ = [c async for c in client.astream([{"role": "user", "content": "hi"}])]


async def test_ollama_astream_parses_ndjson_deltas():
    def handler(request):
        lines = [
            json.dumps({"message": {"content": "Hel"}, "done": False}),
            json.dumps({"message": {"content": "lo"}, "done": False}),
            json.dumps({"message": {"content": ""}, "done": True}),
        ]
        return httpx.Response(200, content="\n".join(lines) + "\n")

    client = LLMClient("ollama", "http://fake", "m")
    client._aclient = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    chunks = [c async for c in client.astream([{"role": "user", "content": "hi"}])]
    assert "".join(chunks) == "Hello"


async def test_ollama_astream_ignores_thinking_field_and_yields_content():
    # Ollama's native thinking mode puts reasoning in its own "thinking"
    # field, entirely separate from "content" — not inline <think> tags.
    def handler(request):
        lines = [
            json.dumps({"message": {"content": "", "thinking": "let me consider..."},
                       "done": False}),
            json.dumps({"message": {"content": "", "thinking": " done reasoning"},
                       "done": False}),
            json.dumps({"message": {"content": "OK"}, "done": False}),
            json.dumps({"message": {"content": ""}, "done": True}),
        ]
        return httpx.Response(200, content="\n".join(lines) + "\n")

    client = LLMClient("ollama", "http://fake", "m", think=True)
    client._aclient = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    chunks = [c async for c in client.astream([{"role": "user", "content": "hi"}])]
    assert "".join(chunks) == "OK"  # reasoning text never leaks into the visible stream


async def test_ollama_astream_diagnoses_thinking_that_exhausts_the_token_budget():
    # Regression test for a real incident: qwen3.6:35b-a3b, asked to "perform
    # a complete pentest ... get the user and root flags" with
    # generation.max_tokens: 1024, spent its whole budget in the "thinking"
    # field and never produced any "content" at all — the operator saw a
    # bare "(empty response)" with no indication that raising max_tokens (or
    # disabling `think`) was the fix.
    def handler(request):
        lines = [
            json.dumps({"message": {"content": "", "thinking": "reasoning that never finishes"},
                       "done": False}),
            json.dumps({"message": {"content": ""}, "done": True, "done_reason": "length"}),
        ]
        return httpx.Response(200, content="\n".join(lines) + "\n")

    client = LLMClient("ollama", "http://fake", "m", think=True)
    client._aclient = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    chunks = [c async for c in client.astream([{"role": "user", "content": "hi"}])]
    text = "".join(chunks)
    assert "ran out of tokens" in text
    assert "max_tokens" in text


async def test_ollama_astream_falls_back_when_model_lacks_thinking():
    # Regression test for a real incident: qwen2.5:14b (the default model)
    # doesn't support Ollama's thinking mode at all. Sending "think": true
    # isn't silently ignored — Ollama hard-rejects the whole request with a
    # 400, which broke every conversational reply (found via a real
    # ~/.mist/mist.db turn containing "ERROR: LLM call failed mid-stream:
    # 400 Bad Request"). The client must detect this and retry without
    # thinking, then remember not to try again for that model.
    calls: list[bool | None] = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body.get("think"))
        if body.get("think"):
            payload = json.dumps({"error": '"m" does not support thinking'}).encode()
            return httpx.Response(400, content=payload)
        lines = [
            json.dumps({"message": {"content": "OK"}, "done": False}),
            json.dumps({"message": {"content": ""}, "done": True}),
        ]
        return httpx.Response(200, content="\n".join(lines) + "\n")

    client = LLMClient("ollama", "http://fake", "m", think=True)
    client._aclient = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    chunks = [c async for c in client.astream([{"role": "user", "content": "hi"}])]
    assert "".join(chunks) == "OK"
    assert calls == [True, False]  # one failed attempt, then a clean retry
    assert client._think_unsupported == {"m"}

    calls.clear()
    chunks2 = [c async for c in client.astream([{"role": "user", "content": "again"}])]
    assert "".join(chunks2) == "OK"
    assert calls == [False]  # cached — no wasted failing request this time


async def test_acomplete_force_think_overrides_schema_default_off():
    # Schema-constrained calls force thinking off by default (routine
    # decisions don't need it and it's slower), but stuck-loop recovery
    # opts back in for one call via force_think — confirmed on the real
    # server that Ollama's native thinking mode keeps `content` clean even
    # under a JSON schema (reasoning lives in a separate "thinking" field),
    # so this doesn't reintroduce the JSON-corruption risk.
    think_values = []

    def handler(request):
        body = json.loads(request.content)
        think_values.append(body.get("think"))
        return httpx.Response(200, json={"message": {"content": '{"action": "respond"}'}})

    client = LLMClient("ollama", "http://fake", "m", think=True)
    client._aclient = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    schema = {"type": "object", "properties": {"action": {"type": "string"}}}

    await client.acomplete([{"role": "user", "content": "hi"}], json_schema=schema)
    await client.acomplete([{"role": "user", "content": "hi"}], json_schema=schema, force_think=True)

    assert think_values == [False, True]


# -- _filter_thinking (mist.llm.client) -----------------------------------

from mist.llm.client import _filter_thinking  # noqa: E402


async def _agen(chunks):
    for c in chunks:
        yield c


async def test_filter_thinking_hides_a_normal_terminated_block():
    out = [c async for c in _filter_thinking(_agen(
        ["<think>reasoning here</think>", "the real answer"]
    ))]
    assert "".join(out) == "the real answer"


async def test_filter_thinking_surfaces_diagnostic_when_think_never_closes():
    # Regression test for a real incident: qwen3.6:35b-a3b spent its entire
    # generation.max_tokens budget reasoning about a broad request
    # ("perform a complete pentest ... get the user and root flags") and
    # never emitted a closing </think> — the operator saw a bare
    # "(empty response)" with zero indication that raising max_tokens (or
    # disabling `think`) was the actual fix.
    out = [c async for c in _filter_thinking(_agen(
        ["<think>", "reasoning that goes on and on and never finishes..."]
    ))]
    text = "".join(out)
    assert "ran out of tokens" in text
    assert "max_tokens" in text


async def test_filter_thinking_no_diagnostic_when_something_was_already_shown():
    # If the model DID produce a real answer before a later, separate think
    # span gets cut off (unusual, but shouldn't happen), don't tack on a
    # confusing diagnostic after real content.
    out = [c async for c in _filter_thinking(_agen(
        ["the real answer", "<think>", "unterminated"]
    ))]
    assert "".join(out) == "the real answer"


async def test_vllm_astream_parses_sse_deltas():
    def handler(request):
        events = [
            f"data: {json.dumps({'choices': [{'delta': {'content': 'Hel'}}]})}",
            f"data: {json.dumps({'choices': [{'delta': {'content': 'lo'}}]})}",
            "data: [DONE]",
        ]
        return httpx.Response(200, content="\n".join(events) + "\n")

    client = LLMClient("vllm", "http://fake/v1", "m")
    client._aclient = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    chunks = [c async for c in client.astream([{"role": "user", "content": "hi"}])]
    assert "".join(chunks) == "Hello"


async def test_ollama_acomplete_returns_full_content():
    def handler(request):
        return httpx.Response(200, json={"message": {"content": '{"action": "respond"}'}})

    client = LLMClient("ollama", "http://fake", "m")
    client._aclient = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    result = await client.acomplete([{"role": "user", "content": "hi"}],
                                     json_schema={"type": "object"})
    assert json.loads(result) == {"action": "respond"}


# -- mission mode (mist.core.agent.MistAgent.astream_mission) ------------

def make_mission_agent(tmp_path, decisions):
    cfg = MistConfig()
    cfg.memory.db_path = str(tmp_path / "test.db")
    cfg.wiki.root_path = str(tmp_path / "wiki")
    store = MemoryStore(cfg.db_path)
    skills = SkillRouter("skills_library")
    tools = default_registry(remember_fn=store.remember, wiki_root=cfg.wiki_root)
    return MistAgent(cfg, ScriptedAsyncLLM(decisions), store, skills, tools)


async def test_mission_redirects_shell_workspace_to_target_directory(tmp_path, monkeypatch):
    # Confirms the structural fix end-to-end: the shell's cwd actually gets
    # redirected to a per-target folder at mission start (not left in one
    # flat workspace every mission/target shares), and a real shell call
    # afterward picks it up.
    captured = {}

    class FakePopen:
        def __init__(self, args, shell, stdout, stderr, text, cwd=None):
            captured["cwd"] = cwd
            self.returncode = 0

        def communicate(self, timeout=None):
            return ("ok\n", None)

    monkeypatch.setattr("mist.tools.registry.subprocess.Popen", FakePopen)

    agent = make_mission_agent(tmp_path, [
        json.dumps({"action": "use_tool", "tool": "shell", "arguments": {"command": "pwd"}}),
        json.dumps({"action": "use_tool", "tool": "finish_objective", "arguments": {"summary": "ok"}}),
        json.dumps({"action": "respond"}),
    ])
    agent.cfg.workspace.mission_root = str(tmp_path / "HTB")
    control = MissionControl()

    events = [e async for e in agent.astream_mission(
        'perform a phased pentest on target ip 10.129.33.21', control, max_turns=10)]

    expected_dir = tmp_path / "HTB" / "10.129.33.21"
    assert captured["cwd"] == expected_dir
    assert expected_dir.is_dir()
    assert "finished" in [e.kind for e in events]

    log_path = [e.text for e in events if e.kind == "started"][0]
    log_text = Path(log_path).read_text(encoding="utf-8")
    assert str(expected_dir) in log_text


async def test_mission_leaves_workspace_alone_when_mission_root_unset(tmp_path, monkeypatch):
    # Default/backward-compat: with workspace.mission_root left empty (the
    # default), a mission must not touch the shell's configured cwd at all.
    captured = {}

    class FakePopen:
        def __init__(self, args, shell, stdout, stderr, text, cwd=None):
            captured["cwd"] = cwd
            self.returncode = 0

        def communicate(self, timeout=None):
            return ("ok\n", None)

    monkeypatch.setattr("mist.tools.registry.subprocess.Popen", FakePopen)

    agent = make_mission_agent(tmp_path, [
        json.dumps({"action": "use_tool", "tool": "shell", "arguments": {"command": "pwd"}}),
        json.dumps({"action": "use_tool", "tool": "finish_objective", "arguments": {"summary": "ok"}}),
        json.dumps({"action": "respond"}),
    ])
    control = MissionControl()

    [e async for e in agent.astream_mission(
        'perform a phased pentest on target ip 10.129.33.21', control, max_turns=10)]

    assert captured["cwd"] is None  # unchanged from default_registry()'s own default


async def test_mission_auto_continues_without_operator_input(tmp_path):
    agent = make_mission_agent(tmp_path, [
        json.dumps({"action": "use_tool", "tool": "shell", "arguments": {"command": "echo recon"}}),
        json.dumps({"action": "respond"}),   # turn 1 ends with an interim status
        json.dumps({"action": "use_tool", "tool": "finish_objective",
                    "arguments": {"summary": "done"}}),
        json.dumps({"action": "respond"}),   # turn 2 ends
    ])
    control = MissionControl()
    events = [e async for e in agent.astream_mission("test objective", control, max_turns=10)]
    kinds = [e.kind for e in events]
    assert kinds.count("turn_start") == 2  # continued into turn 2 with no operator input
    assert "finished" in kinds
    assert any(e.tool == "finish_objective" for e in events)


async def test_mission_writes_deterministic_log_regardless_of_model(tmp_path):
    agent = make_mission_agent(tmp_path, [
        json.dumps({"action": "use_tool", "tool": "shell", "arguments": {"command": "echo recon"}}),
        json.dumps({"action": "use_tool", "tool": "finish_objective",
                    "arguments": {"summary": "done"}}),
        json.dumps({"action": "respond"}),
    ])
    control = MissionControl()
    async for _ in agent.astream_mission("test objective", control, max_turns=10):
        pass
    log_files = list((agent.cfg.wiki_root / "missions").glob("*.md"))
    assert len(log_files) == 1
    text = log_files[0].read_text()
    assert "test objective" in text
    assert "echo recon" in text
    # No `remember` or `write_file` call was ever made by the model — the raw
    # tool transcript must still be captured on disk.
    assert "shell" in text and "finish_objective" in text


# -- mission debrief (mist.core.debrief.MissionDebriefer) -----------------

from mist.core.debrief import MissionDebriefer  # noqa: E402


def test_mission_debriefer_persists_memories_entity_and_skill(tmp_path):
    log_path = tmp_path / "mission.md"
    log_path.write_text("### shell(nmap)\n```\n21/tcp open ftp vsftpd 3.0.3\n```\n",
                        encoding="utf-8")
    store = MemoryStore(tmp_path / "m.db")
    llm = FakeLLM([json.dumps({
        "memories": [{"content": "Cap (10.129.30.111): vsftpd 3.0.3 on 21/tcp"}],
        "entity_page": {"title": "Cap", "body": "FTP: vsftpd 3.0.3."},
        "skill": {"name": "FTP Cleartext Creds via PCAP",
                  "description": "confirmed pcap capture reveals ftp creds",
                  "body": "1. Download /capture repeatedly. 2. tshark -Y ftp."},
    })])
    written_skills = []
    debriefer = MissionDebriefer(
        llm, store, tmp_path,
        write_skill_fn=lambda name, desc, body: written_skills.append((name, desc, body)) or "Wrote skill",
    )

    result = debriefer.debrief("pwn Cap", "completed successfully", log_path)

    assert result.memories_written == 1
    assert any("vsftpd" in h for h in store.search("Cap ftp"))
    assert result.entity_page == "entities/cap.md"
    assert (tmp_path / "entities" / "cap.md").read_text(encoding="utf-8").startswith("# Cap")
    assert result.skill_written == "Wrote skill"
    assert written_skills and written_skills[0][0] == "FTP Cleartext Creds via PCAP"


def test_mission_debriefer_appends_to_existing_entity_page_without_clobbering(tmp_path):
    (tmp_path / "entities").mkdir()
    existing = tmp_path / "entities" / "cap.md"
    existing.write_text("# Cap\n\noriginal findings\n", encoding="utf-8")
    log_path = tmp_path / "mission.md"
    log_path.write_text("some transcript\n", encoding="utf-8")
    store = MemoryStore(tmp_path / "m.db")
    llm = FakeLLM([json.dumps({
        "memories": [],
        "entity_page": {"title": "Cap", "body": "new findings from a later mission"},
    })])
    debriefer = MissionDebriefer(llm, store, tmp_path)

    debriefer.debrief("pwn Cap again", "completed successfully", log_path)

    text = existing.read_text(encoding="utf-8")
    assert "original findings" in text  # not clobbered
    assert "new findings from a later mission" in text  # appended


def test_mission_debriefer_omits_skill_when_none_confirmed(tmp_path):
    log_path = tmp_path / "mission.md"
    log_path.write_text("transcript with no confirmed wins\n", encoding="utf-8")
    store = MemoryStore(tmp_path / "m.db")
    llm = FakeLLM([json.dumps({"memories": [{"content": "no progress made"}]})])
    written_skills = []
    debriefer = MissionDebriefer(
        llm, store, tmp_path,
        write_skill_fn=lambda name, desc, body: written_skills.append(name) or "Wrote skill",
    )

    result = debriefer.debrief("pwn Cap", "hit the turn limit", log_path)

    assert result.memories_written == 1
    assert result.skill_written is None
    assert not written_skills  # never called when the model omits `skill`


def test_mission_debriefer_survives_llm_failure(tmp_path):
    log_path = tmp_path / "mission.md"
    log_path.write_text("some transcript\n", encoding="utf-8")
    store = MemoryStore(tmp_path / "m.db")
    debriefer = MissionDebriefer(RaisingLLM(), store, tmp_path)

    result = debriefer.debrief("pwn Cap", "killed by the operator", log_path)
    assert result.memories_written == 0  # degrades to a no-op, doesn't crash


def test_mission_debriefer_survives_memories_as_plain_strings(tmp_path):
    # Regression test for a real incident: a live debrief against
    # qwen3.6:35b-a3b returned `memories` as a list of plain strings instead
    # of the schema's [{"content": ...}] shape, crashing with
    # AttributeError('str' object has no attribute 'get') outside the
    # try/except that was supposed to guard against exactly this.
    log_path = tmp_path / "mission.md"
    log_path.write_text("some transcript\n", encoding="utf-8")
    store = MemoryStore(tmp_path / "m.db")
    llm = FakeLLM([json.dumps({"memories": ["plain string memory one", "plain string memory two"]})])
    debriefer = MissionDebriefer(llm, store, tmp_path)

    result = debriefer.debrief("pwn Cap", "killed by the operator", log_path)  # must not raise
    assert result.memories_written == 2
    assert any("plain string memory one" in h for h in store.search("plain string memory"))


def test_mission_debriefer_survives_malformed_entity_and_skill_shapes(tmp_path):
    log_path = tmp_path / "mission.md"
    log_path.write_text("some transcript\n", encoding="utf-8")
    store = MemoryStore(tmp_path / "m.db")
    llm = FakeLLM([json.dumps({
        "memories": [{"content": "ok"}],
        "entity_page": "not an object",
        "skill": ["also", "not", "an", "object"],
    })])
    debriefer = MissionDebriefer(llm, store, tmp_path, write_skill_fn=lambda n, d, b: "Wrote skill")

    result = debriefer.debrief("pwn Cap", "killed by the operator", log_path)  # must not raise
    assert result.memories_written == 1
    assert result.entity_page is None
    assert result.skill_written is None


def test_mission_debriefer_skips_empty_transcript(tmp_path):
    log_path = tmp_path / "empty.md"
    log_path.write_text("", encoding="utf-8")
    store = MemoryStore(tmp_path / "m.db")
    llm = FakeLLM([])  # must never be called
    debriefer = MissionDebriefer(llm, store, tmp_path)

    result = debriefer.debrief("pwn Cap", "killed by the operator", log_path)
    assert result.memories_written == 0
    assert llm.replies == []  # untouched — confirms no LLM call was made


async def test_astream_mission_debriefs_on_natural_finish(tmp_path):
    class DebriefableLLM(ScriptedAsyncLLM):
        def complete(self, messages, json_schema=None, force_think=False):
            return json.dumps({"memories": [{"content": "confirmed a finding"}]})

    cfg = MistConfig()
    cfg.memory.db_path = str(tmp_path / "test.db")
    cfg.wiki.root_path = str(tmp_path / "wiki")
    store = MemoryStore(cfg.db_path)
    skills = SkillRouter("skills_library")
    tools = default_registry(remember_fn=store.remember, wiki_root=cfg.wiki_root)
    agent = MistAgent(cfg, DebriefableLLM([
        json.dumps({"action": "use_tool", "tool": "finish_objective", "arguments": {"summary": "done"}}),
        json.dumps({"action": "respond"}),
    ]), store, skills, tools)

    control = MissionControl()
    events = [e async for e in agent.astream_mission("test objective", control, max_turns=10)]
    kinds = [e.kind for e in events]
    assert kinds[0] == "started"  # log path emitted first, for a driver to capture
    assert "debrief" in kinds
    debrief_event = next(e for e in events if e.kind == "debrief")
    assert "1 memory" in debrief_event.text
    assert any("confirmed a finding" in h for h in store.search("confirmed finding"))


async def test_mission_stops_at_max_turns(tmp_path):
    agent = make_mission_agent(tmp_path, [
        json.dumps({"action": "respond"}) for _ in range(6)
    ])
    control = MissionControl()
    events = [e async for e in agent.astream_mission("loop forever", control, max_turns=3)]
    finished = [e for e in events if e.kind == "finished"]
    assert finished and "turn" in finished[0].text and "limit" in finished[0].text
    assert [e.kind for e in events].count("turn_start") == 3


async def test_mission_auto_pauses_on_repeated_identical_tool_call(tmp_path):
    # With stuck_repeat_threshold=2: the first 2-in-a-row triggers a
    # reasoning-assisted recovery attempt (auto-continues, no real pause);
    # repeating again after that triggers a genuine pause. See the two-tier
    # escalation tests below for a focused check of that behavior — this
    # test just confirms a real pause is still reachable end-to-end.
    agent = make_mission_agent(tmp_path, [
        json.dumps({"action": "use_tool", "tool": "shell", "arguments": {"command": "ping x"}}),
        json.dumps({"action": "use_tool", "tool": "shell", "arguments": {"command": "ping x"}}),
        json.dumps({"action": "use_tool", "tool": "shell", "arguments": {"command": "ping x"}}),
        json.dumps({"action": "use_tool", "tool": "shell", "arguments": {"command": "ping x"}}),
        json.dumps({"action": "use_tool", "tool": "finish_objective", "arguments": {"summary": "ok"}}),
        json.dumps({"action": "respond"}),
    ])
    agent.cfg.mission.stuck_repeat_threshold = 2
    control = MissionControl()

    events: list = []

    async def drive():
        async for ev in agent.astream_mission("ping until responsive", control, max_turns=10):
            events.append(ev)

    task = asyncio.create_task(drive())
    for _ in range(200):
        if any(e.kind == "stuck" for e in events):
            break
        await asyncio.sleep(0.01)
    assert any(e.kind == "recovering" for e in events)  # tried to self-recover first
    assert control.paused  # then genuinely paused once that also failed
    control.resume()
    await task
    assert "finished" in [e.kind for e in events]

    # Regression test for a real gap found in live use: the stuck-pause
    # itself was never written to the deterministic mission log, only tool
    # calls were — an operator reviewing the log file afterward would see it
    # cut off mid-command with no explanation of why the mission stopped
    # there.
    log_files = list((agent.cfg.wiki_root / "missions").glob("*.md"))
    assert log_files
    text = log_files[0].read_text()
    assert "**Paused**" in text and "repeated the same tool call" in text


async def test_mission_stuck_detection_fires_mid_turn_not_after_full_tool_budget(tmp_path):
    # Regression test for a real bug found during live testing: the same
    # failing command repeated many times within a *single* turn's own
    # tool-step loop (no intervening "respond") only got caught after the
    # whole turn finished, because the stuck-check lived outside
    # astream_turn's inner loop. It must fire as soon as
    # stuck_repeat_threshold is hit, not wait for the turn's full
    # max_tool_steps budget to be exhausted. With the two-tier escalation,
    # the same fake LLM (which ignores force_think and just keeps returning
    # the identical scripted call) hits the threshold twice — once
    # triggering a reasoning-assisted recovery attempt, then again
    # triggering a real pause — so 2x stuck_repeat_threshold calls total,
    # still far short of the 10 scripted here or the 20-step turn budget.
    same_call = json.dumps({"action": "use_tool", "tool": "shell",
                            "arguments": {"command": "echo no_output"}})
    agent = make_mission_agent(tmp_path, [same_call] * 10)
    agent.cfg.mission.stuck_repeat_threshold = 3
    agent.cfg.context.max_tool_steps = 20
    control = MissionControl()

    events: list = []

    async def drive():
        async for ev in agent.astream_mission("recon the target", control, max_turns=5):
            events.append(ev)

    task = asyncio.create_task(drive())
    for _ in range(200):
        if any(e.kind == "stuck" for e in events):
            break
        await asyncio.sleep(0.01)

    tool_starts = [e for e in events if e.kind == "tool_start"]
    assert len(tool_starts) == 6  # one recovery attempt (3) + one real escalation (3), not 10
    assert any(e.kind == "recovering" for e in events)
    stuck = [e for e in events if e.kind == "stuck"]
    assert stuck and "3x" in stuck[0].text
    assert control.paused  # stays paused until the operator resumes/kills it

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_extract_target_prefers_ip_address():
    obj = 'perform a phased pentest on the host "Connected" at target ip 10.129.33.21'
    assert _extract_target(obj) == "10.129.33.21"


def test_extract_target_falls_back_to_quoted_hostname_without_ip():
    assert _extract_target('perform a phased pentest on the host "Connected"') == "connected"


def test_extract_target_falls_back_to_host_keyword_without_ip():
    assert _extract_target("pentest host: connected") == "connected"


def test_extract_target_returns_none_without_ip_or_hostname():
    assert _extract_target("perform a phased pentest on the target") is None


# -- context compression (_fit_budget's compress_fn, ContextCompressor) ----

def test_fit_budget_without_compress_fn_drops_oldest_silently():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "x" * 4000},
        {"role": "user", "content": "y" * 4000},
        {"role": "user", "content": "current message"},
    ]
    result = _fit_budget(list(messages), budget=500)
    assert result[0]["content"] == "sys"
    assert result[-1]["content"] == "current message"
    assert "x" * 4000 not in "".join(m["content"] for m in result)


def test_fit_budget_with_compress_fn_folds_dropped_into_recap():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "old fact: the target runs FreePBX " + "x" * 400},
        {"role": "user", "content": "current message"},
    ]
    # Budget too tight for the original 400-char message, but roomy enough
    # for "sys" + "current message" + a short recap once it's dropped.
    result = _fit_budget(list(messages), budget=30,
                         compress_fn=lambda dropped: "recap: target runs FreePBX")
    assert any("recap: target runs FreePBX" in m["content"] for m in result)
    assert result[0]["content"] == "sys"
    assert result[-1]["content"] == "current message"


def test_fit_budget_compress_fn_receives_exactly_the_dropped_messages():
    captured = []
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "z" * 4000},
        {"role": "user", "content": "current message"},
    ]

    def compress_fn(dropped):
        captured.extend(dropped)
        return "summary"

    _fit_budget(list(messages), budget=10, compress_fn=compress_fn)
    assert len(captured) == 1
    assert captured[0]["content"] == "z" * 4000


def test_fit_budget_compress_fn_exception_falls_back_to_plain_drop():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "x" * 4000},
        {"role": "user", "content": "current message"},
    ]

    def raising_compress_fn(dropped):
        raise RuntimeError("summarizer backend unreachable")

    result = _fit_budget(list(messages), budget=10, compress_fn=raising_compress_fn)
    assert result[0]["content"] == "sys"
    assert result[-1]["content"] == "current message"
    assert len(result) == 2  # nothing inserted; behaves like the no-compress_fn path


def test_fit_budget_compress_fn_empty_summary_falls_back_to_plain_drop():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "x" * 4000},
        {"role": "user", "content": "current message"},
    ]
    result = _fit_budget(list(messages), budget=10, compress_fn=lambda dropped: "")
    assert len(result) == 2


def test_fit_budget_without_anything_to_drop_ignores_compress_fn():
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    called = []
    result = _fit_budget(list(messages), budget=10_000,
                         compress_fn=lambda dropped: called.append(dropped) or "x")
    assert called == []
    assert result == messages


def test_context_compressor_uses_llm_and_returns_stripped_summary():
    compressor = ContextCompressor(FakeLLM(["  a concise recap  "]))
    summary = compressor.summarize([{"role": "user", "content": "some old message"}])
    assert summary == "a concise recap"


def test_context_compressor_falls_back_to_truncated_text_on_llm_failure():
    compressor = ContextCompressor(RaisingLLM(), max_input_chars=8000)
    summary = compressor.summarize([{"role": "user", "content": "a fact worth keeping"}])
    assert "a fact worth keeping" in summary  # never just silently empty


def test_context_compressor_falls_back_when_llm_returns_empty():
    compressor = ContextCompressor(FakeLLM(["   "]))
    summary = compressor.summarize([{"role": "user", "content": "important finding"}])
    assert "important finding" in summary


def test_agent_compress_fn_property_reflects_configured_compressor(tmp_path):
    cfg = MistConfig()
    cfg.memory.db_path = str(tmp_path / "test.db")
    store = MemoryStore(cfg.db_path)
    skills = SkillRouter("skills_library")
    tools = default_registry(remember_fn=store.remember)

    agent_without = MistAgent(cfg, FakeLLM([]), store, skills, tools)
    assert agent_without._compress_fn is None

    compressor = ContextCompressor(FakeLLM(["summary"]))
    agent_with = MistAgent(cfg, FakeLLM([]), store, skills, tools, context_compressor=compressor)
    assert agent_with._compress_fn == compressor.summarize


def test_normalize_tool_call_collapses_varying_literal():
    # Regression case from a real run: `search_files` spammed with a
    # different query string each call, all against the same root — the
    # exact-repeat check never sees two byte-identical calls, but this is
    # exactly the pattern near-duplicate detection exists to catch.
    from mist.core.agent import _normalize_tool_call
    a = _normalize_tool_call("search_files", json.dumps({"query": "Next.js", "root": "."}))
    b = _normalize_tool_call("search_files", json.dumps({"query": "ReactorWatch", "root": "."}))
    assert a == b


def test_normalize_tool_call_preserves_different_ip_targets():
    # A false positive here would be worse than missing the real pattern:
    # two calls against two different real hosts must never collapse just
    # because both targets happen to be numeric.
    from mist.core.agent import _normalize_tool_call
    a = _normalize_tool_call("shell", json.dumps({"command": "nmap -sV -sC 10.129.30.204"}))
    b = _normalize_tool_call("shell", json.dumps({"command": "nmap -sV -sC 10.129.30.99"}))
    assert a != b


def test_is_manual_probe_detects_curl_and_wget_shell_calls():
    from mist.core.agent import _is_manual_probe
    assert _is_manual_probe("shell", json.dumps({"command": "curl -s http://10.0.0.1/login"}))
    assert _is_manual_probe("shell", json.dumps({"command": "wget -q http://10.0.0.1/x"}))
    assert _is_manual_probe("shell", json.dumps({"command": "sudo curl http://10.0.0.1/"}))


def test_is_manual_probe_false_for_scanners_and_other_tools():
    from mist.core.agent import _is_manual_probe
    assert not _is_manual_probe("shell", json.dumps({"command": "nmap -sV -sC 10.0.0.1"}))
    assert not _is_manual_probe("shell", json.dumps({"command": "gobuster dir -u http://10.0.0.1"}))
    assert not _is_manual_probe("search_files", json.dumps({"query": "curl"}))


def test_truncate_tool_output_keeps_tail_not_just_head():
    # Regression case from a real run: a live `nuclei` scan's actual
    # findings and "N matches found" summary line only appear after a long
    # banner/template-loading preamble. A plain head cut (the old
    # behavior) kept only that banner and silently dropped every real
    # finding — the model was reasoning off "10430 templates loaded", never
    # the scan results.
    from mist.core.agent import _truncate_tool_output
    banner = "banner noise " * 50
    findings = "tech-detect:next.js found\nScan completed in 3m. 14 matches found."
    text = banner + findings
    out = _truncate_tool_output(text, budget=200)
    assert "Scan completed" in out
    assert len(out) <= 200 + len("\n...[N chars omitted]...\n")  # marker overhead only


def test_truncate_tool_output_passes_through_under_budget():
    from mist.core.agent import _truncate_tool_output
    assert _truncate_tool_output("short", budget=200) == "short"


def test_truncate_raw_output_keeps_tail_not_just_head():
    # Same fix, one layer earlier: registry._run_subprocess's own coarser
    # cap ran before agent.py's smarter truncation ever saw the output, so
    # a plain head cut here would discard the real findings before agent.py
    # got a chance to keep them.
    from mist.tools.registry import _truncate_raw_output
    banner = "banner noise " * 50
    findings = "tech-detect:next.js found\nScan completed in 3m. 14 matches found."
    text = banner + findings
    out = _truncate_raw_output(text, budget=200)
    assert "Scan completed" in out


async def test_mission_pauses_on_near_duplicate_tool_calls(tmp_path):
    # Same shape as test_mission_auto_pauses_on_repeated_identical_tool_call,
    # but every call is byte-different (a different search_files query each
    # time) — only the coarser near-duplicate signature repeats. The exact
    # stuck check must never fire here; the near-duplicate one must.
    queries = ["Next.js", "ReactorWatch", "Next.js vulnerability", "Server Action",
              "Next.js RCE", "nuclei", "pentest-methodology", "v3.2.1"]
    agent = make_mission_agent(tmp_path, [
        json.dumps({"action": "use_tool", "tool": "search_files",
                    "arguments": {"query": q, "root": "."}}) for q in queries
    ] + [
        json.dumps({"action": "use_tool", "tool": "finish_objective", "arguments": {"summary": "ok"}}),
        json.dumps({"action": "respond"}),
    ])
    agent.cfg.mission.near_duplicate_threshold = 3
    control = MissionControl()

    events: list = []

    async def drive():
        async for ev in agent.astream_mission("find the known vulnerability", control, max_turns=10):
            events.append(ev)

    task = asyncio.create_task(drive())
    for _ in range(200):
        if any(e.kind == "stuck" for e in events):
            break
        await asyncio.sleep(0.01)
    assert any(e.kind == "recovering" for e in events)
    assert control.paused
    stuck = [e for e in events if e.kind == "stuck"]
    assert stuck and "varying only a literal/number" in stuck[0].text
    control.resume()
    await task
    assert "finished" in [e.kind for e in events]


async def test_mission_pauses_on_manual_probe_streak(tmp_path, monkeypatch):
    # Regression case from a real run: after one nmap scan, the model made
    # a dozen+ `curl` calls to a *different* invented path every time
    # (/login, /dashboard, /api/v1/reports/generate.json, ...) instead of
    # running a content-discovery scanner. Every call is genuinely
    # different, so neither the exact-repeat nor near-duplicate check ever
    # fires — only the manual-probe-streak check should.
    # 6 distinct paths against a threshold of 3: the first 3 trigger recovery
    # (which resets the streak), the next 3 trigger the pause — same shape
    # as test_mission_pauses_on_near_duplicate_tool_calls's 8-query/3-threshold ratio.
    #
    # The IP is a real target elsewhere on the operator's network — from
    # wherever tests actually run, it's unreachable, and a real `curl`
    # doesn't fail fast: it hangs until Mist's own subprocess timeout,
    # which blew straight through this test's polling window (a genuine
    # regression this exact test hit once network conditions differed from
    # when it was written). Mock the subprocess so this never depends on
    # real network reachability/timing at all.
    class FakePopen:
        def __init__(self, args, shell, stdout, stderr, text, cwd=None):
            self.returncode = 0

        def communicate(self, timeout=None):
            return ("(mock response)", "")
    monkeypatch.setattr("mist.tools.registry.subprocess.Popen", FakePopen)

    paths = ["/login", "/dashboard", "/api/v1/users", "/api/v1/reports/generate.json",
             "/api/v1/reports/generate.xml", "/robots.txt"]
    agent = make_mission_agent(tmp_path, [
        json.dumps({"action": "use_tool", "tool": "shell",
                    "arguments": {"command": f"curl -s http://10.129.245.214:3000{p}"}})
        for p in paths
    ] + [
        json.dumps({"action": "use_tool", "tool": "finish_objective", "arguments": {"summary": "ok"}}),
        json.dumps({"action": "respond"}),
    ])
    agent.cfg.mission.manual_probe_threshold = 3
    control = MissionControl()

    events: list = []

    async def drive():
        async for ev in agent.astream_mission("enumerate the web app", control, max_turns=10):
            events.append(ev)

    task = asyncio.create_task(drive())
    for _ in range(200):
        if any(e.kind == "stuck" for e in events):
            break
        await asyncio.sleep(0.01)
    assert any(e.kind == "recovering" for e in events)
    assert control.paused
    stuck = [e for e in events if e.kind == "stuck"]
    assert stuck and "curl/wget probes" in stuck[0].text
    control.resume()
    await task
    assert "finished" in [e.kind for e in events]


async def test_mission_manual_probe_streak_resets_on_scanner_call(tmp_path, monkeypatch):
    # A scanner call in between two curls must reset the streak — it's
    # exactly the corrective behavior the nudge asks for, not a violation.
    # Mocked for the same reason as test_mission_pauses_on_manual_probe_streak
    # above: a real curl/gobuster against this unreachable IP shouldn't
    # determine how long this test takes.
    class FakePopen:
        def __init__(self, args, shell, stdout, stderr, text, cwd=None):
            self.returncode = 0

        def communicate(self, timeout=None):
            return ("(mock response)", "")
    monkeypatch.setattr("mist.tools.registry.subprocess.Popen", FakePopen)

    agent = make_mission_agent(tmp_path, [
        json.dumps({"action": "use_tool", "tool": "shell",
                    "arguments": {"command": "curl -s http://10.0.0.1:3000/login"}}),
        json.dumps({"action": "use_tool", "tool": "shell",
                    "arguments": {"command": "gobuster dir -u http://10.0.0.1:3000 -w list.txt"}}),
        json.dumps({"action": "use_tool", "tool": "shell",
                    "arguments": {"command": "curl -s http://10.0.0.1:3000/dashboard"}}),
        json.dumps({"action": "use_tool", "tool": "finish_objective", "arguments": {"summary": "ok"}}),
        json.dumps({"action": "respond"}),
    ])
    agent.cfg.mission.manual_probe_threshold = 2
    control = MissionControl()

    events = [e async for e in agent.astream_mission("enumerate the web app", control, max_turns=10)]
    assert not any(e.kind == "stuck" for e in events)
    assert "finished" in [e.kind for e in events]


async def test_mission_pauses_on_respond_streak_with_no_tool_call(tmp_path):
    # Regression case from a real run: the model chose "respond" instead of
    # calling a tool, turn after turn, describing a plan in prose instead of
    # acting on it. Every other check is keyed off tool_start events, so a
    # mission that never calls a tool was completely invisible to stuck
    # detection before this fix — it could otherwise run to max_turns.
    agent = make_mission_agent(tmp_path, [
        json.dumps({"action": "respond"}),
        json.dumps({"action": "respond"}),
        json.dumps({"action": "respond"}),
        json.dumps({"action": "respond"}),
        json.dumps({"action": "use_tool", "tool": "finish_objective", "arguments": {"summary": "ok"}}),
        json.dumps({"action": "respond"}),
    ])
    agent.cfg.mission.respond_streak_threshold = 2
    control = MissionControl()

    events: list = []

    async def drive():
        async for ev in agent.astream_mission("get root", control, max_turns=10):
            events.append(ev)

    task = asyncio.create_task(drive())
    for _ in range(200):
        if any(e.kind == "stuck" for e in events):
            break
        await asyncio.sleep(0.01)
    assert any(e.kind == "recovering" for e in events)
    assert control.paused
    stuck = [e for e in events if e.kind == "stuck"]
    assert stuck and "no tool call" in stuck[0].text
    control.resume()
    await task
    assert "finished" in [e.kind for e in events]


async def test_mission_respond_streak_resets_on_tool_call(tmp_path):
    # A tool call in the middle of a turn (even one that also ends in
    # "respond" afterward) must reset the streak — the pathology is never
    # calling a tool, not ending a turn with a status update per se.
    agent = make_mission_agent(tmp_path, [
        json.dumps({"action": "respond"}),
        json.dumps({"action": "use_tool", "tool": "shell", "arguments": {"command": "echo hi"}}),
        json.dumps({"action": "respond"}),
        json.dumps({"action": "use_tool", "tool": "finish_objective", "arguments": {"summary": "ok"}}),
        json.dumps({"action": "respond"}),
    ])
    agent.cfg.mission.respond_streak_threshold = 2
    control = MissionControl()

    events = [e async for e in agent.astream_mission("test objective", control, max_turns=10)]
    assert not any(e.kind == "stuck" for e in events)
    assert "finished" in [e.kind for e in events]


async def test_mission_stuck_recovery_enables_thinking_then_reverts(tmp_path):
    # Two behaviors are asserted by the force_think sequence below:
    #   1. The opening decision reasons (force_think=True) until the mission
    #      has made its first tool call — but ONLY the first decision of a turn
    #      (astream_turn's inner-loop toggle), so turn 1's first call reasons
    #      and its second (post-tool) call does not.
    #   2. The stuck-recovery turn also forces thinking on its first decision.
    # It must NOT force thinking on routine post-action decisions, which would
    # make each needlessly slow (a real decision call with thinking enabled
    # took ~13s in practice vs sub-second without).
    class RecordingLLM(ScriptedAsyncLLM):
        def __init__(self, decisions):
            super().__init__(decisions)
            self.force_think_calls: list[bool] = []

        async def acomplete(self, messages, json_schema=None, force_think=False):
            self.force_think_calls.append(force_think)
            return self.decisions.pop(0)

    llm = RecordingLLM([
        json.dumps({"action": "use_tool", "tool": "shell", "arguments": {"command": "ping x"}}),
        json.dumps({"action": "use_tool", "tool": "shell", "arguments": {"command": "ping x"}}),
        json.dumps({"action": "respond"}),  # the recovery turn's own decision
        json.dumps({"action": "use_tool", "tool": "finish_objective",
                    "arguments": {"summary": "ok"}}),  # a normal turn afterward
        json.dumps({"action": "respond"}),
    ])
    agent = make_mission_agent(tmp_path, [])
    agent.llm = llm
    agent.cfg.mission.stuck_repeat_threshold = 2
    control = MissionControl()

    events = [e async for e in agent.astream_mission("ping until responsive", control, max_turns=10)]

    assert any(e.kind == "recovering" for e in events)
    assert "finished" in [e.kind for e in events]
    assert llm.force_think_calls == [True, False, True, False, False]


async def test_mission_first_turn_uses_imperative_framing_not_bare_objective(tmp_path):
    # Regression: the opening mission turn must get the same "this is not a
    # request for a plan — take it with a tool right now" framing every later
    # turn gets, NOT the bare objective. Seeded with the bare objective
    # ("perform a phased pentest on <host>..."), the opening turn reads like a
    # request for a plan and the model opens with a prose plan instead of a
    # tool call — the actual cause of missions stalling at the very first step.
    captured: dict[str, str] = {}

    class CapturingLLM(ScriptedAsyncLLM):
        async def acomplete(self, messages, json_schema=None, force_think=False):
            captured.setdefault("first_user", messages[-1]["content"])
            return self.decisions.pop(0)

    llm = CapturingLLM([
        json.dumps({"action": "use_tool", "tool": "finish_objective",
                    "arguments": {"summary": "done"}}),
        json.dumps({"action": "respond"}),
    ])
    agent = make_mission_agent(tmp_path, [])
    agent.llm = llm
    control = MissionControl()

    _ = [e async for e in agent.astream_mission("get root on 10.10.10.5", control, max_turns=3)]

    first_user = captured["first_user"]
    assert "get root on 10.10.10.5" in first_user            # objective is present
    assert "not a request for a plan" in first_user          # ... wrapped in the imperative framing
    assert first_user.strip() != "get root on 10.10.10.5"    # not the bare objective


async def test_mission_forces_think_only_on_first_decision_of_opening_turn(tmp_path):
    # The opening-turn reasoning must cost exactly ONE forced-think decision,
    # not one per inner tool-loop step. A mission turn's inner loop can run up
    # to max_tool_steps decisions; reasoning on all of them (an earlier bug)
    # made a single opening turn spend minutes per tool call. Here turn 1 makes
    # two tool calls then responds — only its FIRST decision should force
    # thinking; the post-tool decisions in the same turn must not.
    class RecordingLLM(ScriptedAsyncLLM):
        def __init__(self, decisions):
            super().__init__(decisions)
            self.force_think_calls: list[bool] = []

        async def acomplete(self, messages, json_schema=None, force_think=False):
            self.force_think_calls.append(force_think)
            return self.decisions.pop(0)

    llm = RecordingLLM([
        json.dumps({"action": "use_tool", "tool": "shell",
                    "arguments": {"command": "echo a"}}),   # turn 1, decision 1 (first -> think)
        json.dumps({"action": "use_tool", "tool": "shell",
                    "arguments": {"command": "echo b"}}),   # turn 1, decision 2 (post-tool -> fast)
        json.dumps({"action": "respond"}),                  # turn 1, decision 3 (post-tool -> fast)
        json.dumps({"action": "use_tool", "tool": "finish_objective",
                    "arguments": {"summary": "ok"}}),       # turn 2 (already acted -> fast)
        json.dumps({"action": "respond"}),                  # turn 2, second inner call
    ])
    agent = make_mission_agent(tmp_path, [])
    agent.llm = llm
    control = MissionControl()

    events = [e async for e in agent.astream_mission("do the thing", control, max_turns=10)]

    assert "finished" in [e.kind for e in events]
    # Exactly one reasoning pass, on the very first decision of the mission.
    assert llm.force_think_calls == [True, False, False, False, False]


async def test_stream_aborts_runaway_thinking_with_no_content():
    # Part B guard: a reasoning model can emit only "thinking" and never reach
    # "content", burning its whole (minutes-sized) max_tokens budget producing
    # nothing. With stream_no_content_timeout set, the stream must abort early
    # with an explanatory message instead of waiting out the full budget.
    client = LLMClient("ollama", "http://fake", "m", stream_no_content_timeout=0.02)

    class _FakeResp:
        status_code = 200

        def raise_for_status(self):
            pass

        async def aread(self):
            return b""

        async def aiter_lines(self):
            # Thinking-only lines, never any content, with gaps so wall-clock
            # elapses well past the tiny timeout.
            for _ in range(5):
                await asyncio.sleep(0.05)
                yield json.dumps({"message": {"thinking": "reasoning..."}})

    class _FakeStreamCtx:
        async def __aenter__(self):
            return _FakeResp()

        async def __aexit__(self, *exc):
            return False

    def fake_stream(method, url, json=None):
        return _FakeStreamCtx()

    client._aclient.stream = fake_stream
    chunks = [c async for c in client.astream([{"role": "user", "content": "hi"}])]
    joined = "".join(chunks)
    assert "aborted a runaway reasoning pass" in joined  # the guard fired
    assert "reasoning..." not in joined                  # raw thinking never leaked as content
    await client.aclose()


async def test_stream_no_content_timeout_unset_does_not_abort():
    # Default (None) must preserve prior behavior: no early abort. A short
    # thinking-only stream that ends on its own should yield the existing
    # "ran out of tokens while thinking" message, not the runaway-abort one.
    client = LLMClient("ollama", "http://fake", "m")  # stream_no_content_timeout defaults None

    class _FakeResp:
        status_code = 200

        def raise_for_status(self):
            pass

        async def aread(self):
            return b""

        async def aiter_lines(self):
            yield json.dumps({"message": {"thinking": "reasoning..."}})
            yield json.dumps({"message": {"content": ""}, "done": True})

    class _FakeStreamCtx:
        async def __aenter__(self):
            return _FakeResp()

        async def __aexit__(self, *exc):
            return False

    client._aclient.stream = lambda method, url, json=None: _FakeStreamCtx()
    joined = "".join([c async for c in client.astream([{"role": "user", "content": "hi"}])])
    assert "ran out of tokens" in joined
    assert "aborted a runaway reasoning pass" not in joined
    await client.aclose()


async def test_stream_truncates_runaway_content_wall():
    # The content-runaway guard: a reply that streams far more visible content
    # than a real answer (a fabricated wall of text with no tool output behind
    # it) must be cut off with a truncation notice, not streamed to the full
    # token budget. Cap here is 10 tokens (~40 chars).
    client = LLMClient("ollama", "http://fake", "m", stream_max_content_tokens=10)

    class _FakeResp:
        status_code = 200

        def raise_for_status(self):
            pass

        async def aread(self):
            return b""

        async def aiter_lines(self):
            # ~40 chars/chunk of "content"; a few of these blow past 10 tokens.
            for _ in range(20):
                yield json.dumps({"message": {"content": "fabricated detail about the box, " * 2}})

    class _FakeStreamCtx:
        async def __aenter__(self):
            return _FakeResp()

        async def __aexit__(self, *exc):
            return False

    client._aclient.stream = lambda method, url, json=None: _FakeStreamCtx()
    chunks = [c async for c in client.astream([{"role": "user", "content": "hi"}])]
    joined = "".join(chunks)
    assert "answer truncated past" in joined            # the guard fired
    # Cut off early: nowhere near all 20 chunks' worth of content streamed.
    assert joined.count("fabricated detail about the box") < 20


async def test_stream_max_content_tokens_unset_streams_full_answer():
    # Default (None) must not truncate: a normal multi-chunk answer streams in full.
    client = LLMClient("ollama", "http://fake", "m")  # stream_max_content_tokens defaults None

    class _FakeResp:
        status_code = 200

        def raise_for_status(self):
            pass

        async def aread(self):
            return b""

        async def aiter_lines(self):
            yield json.dumps({"message": {"content": "hello "}})
            yield json.dumps({"message": {"content": "world"}, "done": True})

    class _FakeStreamCtx:
        async def __aenter__(self):
            return _FakeResp()

        async def __aexit__(self, *exc):
            return False

    client._aclient.stream = lambda method, url, json=None: _FakeStreamCtx()
    joined = "".join([c async for c in client.astream([{"role": "user", "content": "hi"}])])
    assert joined == "hello world"
    assert "truncated" not in joined
    await client.aclose()


async def test_mission_nudge_recaps_prior_tool_findings(tmp_path):
    # Regression case from a real run: after a reasoning-assisted recovery
    # got the model to call a real tool again, it re-ran a scan it already
    # had full results for — no existing mechanism reminds the model what
    # it already found at exactly the moment a nudge asks it to try
    # something different. Confirms the recap reaches the model at both the
    # first-recovery stage and the second (pause) stage.
    class RecordingLLM(ScriptedAsyncLLM):
        def __init__(self, decisions):
            super().__init__(decisions)
            self.seen_user_messages: list[str] = []

        async def acomplete(self, messages, json_schema=None, force_think=False):
            self.seen_user_messages.append(messages[-1]["content"])
            return self.decisions.pop(0)

    llm = RecordingLLM([
        json.dumps({"action": "use_tool", "tool": "shell", "arguments": {"command": "echo findme"}}),
        json.dumps({"action": "respond"}),                                              # turn 1 ends
        json.dumps({"action": "use_tool", "tool": "shell", "arguments": {"command": "ping x"}}),
        json.dumps({"action": "use_tool", "tool": "shell", "arguments": {"command": "ping x"}}),  # -> recovery
        json.dumps({"action": "respond"}),                                              # recovery turn's decision
        json.dumps({"action": "use_tool", "tool": "shell", "arguments": {"command": "ping x"}}),
        json.dumps({"action": "use_tool", "tool": "shell", "arguments": {"command": "ping x"}}),  # -> pause
        json.dumps({"action": "use_tool", "tool": "finish_objective", "arguments": {"summary": "ok"}}),
        json.dumps({"action": "respond"}),
    ])
    agent = make_mission_agent(tmp_path, [])
    agent.llm = llm
    agent.cfg.mission.stuck_repeat_threshold = 2
    control = MissionControl()

    events: list = []

    async def drive():
        async for ev in agent.astream_mission("test objective", control, max_turns=10):
            events.append(ev)

    task = asyncio.create_task(drive())
    for _ in range(200):
        if any(e.kind == "stuck" for e in events):
            break
        await asyncio.sleep(0.01)
    assert control.paused
    control.resume()
    await task

    assert "finished" in [e.kind for e in events]
    recap_messages = [m for m in llm.seen_user_messages if "Already found this mission" in m]
    assert len(recap_messages) >= 2  # reached both the recovery-stage and pause-stage nudges
    assert all("findme" in m for m in recap_messages)


def test_findings_recap_keeps_real_findings_over_a_burst_of_noise():
    # The exact live regression: a burst of empty/no-output probes must not
    # evict the substantive findings (nmap ports, discovered version) from the
    # recap window, or a recovering mission is shown only "(no output)" lines
    # and re-runs recon it already completed. Substantive results are drawn
    # from the WHOLE mission, not just the raw last-N entries.
    findings = [
        "shell: 22/tcp open ssh; 80/tcp open http; 443/tcp open https",   # nmap (early)
        "shell: Apache 2.4.6, FreePBX 16.0.38.1 discovered on /ucp",       # version (early)
        "shell: (no output)",
        "shell: (no output; command exited with status 1 — e.g. a grep/filter that matched nothing)",
        "shell: (no output)",
        "shell: ERROR: command timed out after 480s",
    ]
    recap = _findings_recap(findings, max_items=5)
    assert "22/tcp open ssh" in recap        # early real finding survived the noise burst
    assert "FreePBX 16.0.38.1" in recap      # early real finding survived the noise burst
    assert "(no output" not in recap         # empty/no-output results dropped
    assert "ERROR" not in recap              # errors dropped


def test_findings_recap_dedups_identical_entries_and_empty_when_all_noise():
    assert _findings_recap(["shell: same", "shell: same", "shell: same"]).count("same") == 1
    assert _findings_recap(["shell: (no output)", "shell: ERROR: boom"]) == ""


def test_shell_empty_output_disambiguated_by_exit_status():
    # A command that prints nothing but exits non-zero (e.g. grep with no
    # match) must not read as a bare "(no output)" — the model needs to tell a
    # matchless filter from a plain success, or it re-probes the same dead end.
    tools = default_registry(remember_fn=lambda c: None, shell_config=None)
    nonzero = tools.get("shell").run(command="grep nomatch /dev/null")
    assert "no output" in nonzero and "status 1" in nonzero
    zero = tools.get("shell").run(command="true")
    assert "no output" in zero and "exited 0" in zero


async def test_mission_finish_objective_reachable_even_off_topic(tmp_path):
    # finish_objective shares no keywords with this message, so it would be
    # ranked out of the default top-5 tools if not force-included.
    agent = make_mission_agent(tmp_path, [
        json.dumps({"action": "use_tool", "tool": "finish_objective",
                    "arguments": {"summary": "irrelevant wording on purpose"}}),
        json.dumps({"action": "respond"}),
    ])
    control = MissionControl()
    events = [e async for e in agent.astream_mission(
        "talk about baking bread and cooking pasta recipes", control, max_turns=5
    )]
    assert any(e.tool == "finish_objective" for e in events)
    assert agent.always_exposed == set()  # cleared again once the mission ends


def test_mission_control_pause_resume_and_notes():
    control = MissionControl()
    assert not control.paused
    control.pause()
    assert control.paused
    control.add_note("a")
    control.add_note("b")
    assert control.pop_notes() == ["a", "b"]
    assert control.pop_notes() == []  # drained
    control.resume()
    assert not control.paused


# -- persistent input history (mist.history.HistoryStore) ----------------

from mist.history import HistoryStore  # noqa: E402


def test_history_store_roundtrips_across_instances(tmp_path):
    path = tmp_path / "history"
    store = HistoryStore(path)
    store.add("nmap -sV 10.129.30.111")
    store.add("/model qwen3.6:35b-a3b")

    reloaded = HistoryStore(path)
    assert reloaded.all() == ["nmap -sV 10.129.30.111", "/model qwen3.6:35b-a3b"]


def test_history_store_resubmitting_entry_moves_it_to_most_recent(tmp_path):
    store = HistoryStore(tmp_path / "history")
    store.add("first")
    store.add("second")
    store.add("first")
    assert store.all() == ["second", "first"]  # no stale duplicate left behind


def test_history_store_ignores_blank_entries(tmp_path):
    store = HistoryStore(tmp_path / "history")
    store.add("   ")
    store.add("")
    assert store.all() == []


def test_history_store_caps_at_max_entries(tmp_path):
    store = HistoryStore(tmp_path / "history", max_entries=3)
    for i in range(5):
        store.add(f"entry {i}")
    assert store.all() == ["entry 2", "entry 3", "entry 4"]


def test_history_store_starts_empty_when_no_file_exists(tmp_path):
    store = HistoryStore(tmp_path / "does-not-exist")
    assert store.all() == []


# -- streaming TUI: history recall + ghost-text autofill ------------------

from mist.tui.app import HistoryInput, HistorySuggester  # noqa: E402


async def test_history_suggester_matches_most_recent_prefix():
    history = HistoryStore.__new__(HistoryStore)  # bypass file I/O for a pure unit test
    history._entries = ["nmap -sV 10.0.0.1", "nmap -sS -p- 10.0.0.1"]
    suggester = HistorySuggester(history)
    assert await suggester.get_suggestion("nmap") == "nmap -sS -p- 10.0.0.1"
    assert await suggester.get_suggestion("nmap -sV") is None  # exact match, nothing to add
    assert await suggester.get_suggestion("") is None
    assert await suggester.get_suggestion("gobuster") is None


async def test_tui_up_down_recalls_history(tmp_path):
    agent = make_async_agent(tmp_path, ScriptedAsyncLLM([
        json.dumps({"action": "respond"}),
        json.dumps({"action": "respond"}),
    ]))
    agent.cfg.history_path = str(tmp_path / "history")
    app = MistTUI(agent)
    async with app.run_test() as pilot:
        input_widget = app.query_one("#input", HistoryInput)
        assert isinstance(input_widget, HistoryInput)

        await pilot.click("#input")
        for ch in "first command":
            await pilot.press(ch)
        await pilot.press("enter")
        await pilot.pause(0.05)

        for ch in "second command":
            await pilot.press(ch)
        await pilot.press("enter")
        await pilot.pause(0.05)

        await pilot.press("up")
        assert input_widget.value == "second command"
        await pilot.press("up")
        assert input_widget.value == "first command"
        await pilot.press("up")  # already at the oldest entry — stays put
        assert input_widget.value == "first command"

        await pilot.press("down")
        assert input_widget.value == "second command"
        await pilot.press("down")
        assert input_widget.value == ""  # past the newest entry -> back to the draft


# -- mist doctor (mist.doctor) ----------------------------------------------

def test_check_config_loads_always_ok():
    cfg = MistConfig()
    result = doctor_module.check_config_loads(cfg)
    assert result.status == "ok"


def test_check_wiki_initialized_detects_missing_and_present(tmp_path):
    cfg = MistConfig()
    cfg.wiki.root_path = str(tmp_path / "wiki")
    assert doctor_module.check_wiki_initialized(cfg).status == "warn"

    (tmp_path / "wiki").mkdir()
    (tmp_path / "wiki" / "SCHEMA.md").write_text("# schema", encoding="utf-8")
    assert doctor_module.check_wiki_initialized(cfg).status == "ok"


def test_check_wiki_initialized_is_case_insensitive(tmp_path):
    # Regression case from a real run: a long-populated real wiki had a
    # lowercase schema.md from before the scaffold's SCHEMA.md convention
    # solidified — a case-sensitive check reported a false "not
    # initialized" against a wiki that clearly was.
    cfg = MistConfig()
    cfg.wiki.root_path = str(tmp_path / "wiki")
    (tmp_path / "wiki").mkdir()
    (tmp_path / "wiki" / "schema.md").write_text("# schema", encoding="utf-8")
    assert doctor_module.check_wiki_initialized(cfg).status == "ok"


def test_check_backend_reachable_ok_and_fail(monkeypatch):
    cfg = MistConfig()

    monkeypatch.setattr(doctor_module.LLMClient, "list_models", lambda self: ["a", "b"])
    result, models = doctor_module.check_backend_reachable(cfg)
    assert result.status == "ok"
    assert models == ["a", "b"]

    def _raise(self):
        raise ConnectionError("refused")
    monkeypatch.setattr(doctor_module.LLMClient, "list_models", _raise)
    result, models = doctor_module.check_backend_reachable(cfg)
    assert result.status == "fail"
    assert models is None
    assert "ConnectionError" in result.detail  # repr-style, not a swallowed empty message


def test_check_model_available_variants():
    cfg = MistConfig()
    cfg.model = "qwen2.5:14b"
    assert doctor_module.check_model_available(cfg, None).status == "warn"
    assert doctor_module.check_model_available(cfg, ["qwen2.5:14b", "other"]).status == "ok"
    assert doctor_module.check_model_available(cfg, ["other"]).status == "fail"


def test_check_model_available_tolerates_implicit_latest_tag():
    # Regression case from a real run: config said "bge-m3" but Ollama's
    # /api/tags reports "bge-m3:latest" — a genuinely available model
    # reported as missing on an exact string comparison.
    cfg = MistConfig()
    cfg.model = "bge-m3"
    assert doctor_module.check_model_available(cfg, ["bge-m3:latest"]).status == "ok"

    cfg.model = "bge-m3:latest"
    assert doctor_module.check_model_available(cfg, ["bge-m3"]).status == "ok"

    # A genuinely different tag must still be reported as unavailable.
    cfg.model = "qwen2.5:14b"
    assert doctor_module.check_model_available(cfg, ["qwen2.5:7b"]).status == "fail"


def test_check_num_ctx_warns_when_discovery_fails(monkeypatch):
    cfg = MistConfig()
    monkeypatch.setattr(doctor_module.LLMClient, "discover_context_length", lambda self, timeout=5.0: None)
    assert doctor_module.check_num_ctx(cfg).status == "warn"


def test_check_num_ctx_warns_when_budget_exceeds_window(monkeypatch):
    cfg = MistConfig()
    cfg.context.token_budget = 6000
    cfg.generation.max_tokens = 260000  # 266000 total > the 262144 discovered window
    monkeypatch.setattr(doctor_module.LLMClient, "discover_context_length",
                        lambda self, timeout=5.0: 262144)
    result = doctor_module.check_num_ctx(cfg)
    assert result.status == "warn"
    assert "capped" in result.detail


def test_check_num_ctx_ok_when_it_fits(monkeypatch):
    cfg = MistConfig()
    cfg.context.token_budget = 6000
    cfg.generation.max_tokens = 8192
    monkeypatch.setattr(doctor_module.LLMClient, "discover_context_length",
                        lambda self, timeout=5.0: 262144)
    assert doctor_module.check_num_ctx(cfg).status == "ok"


def test_check_shell_backend_local_is_always_ok():
    cfg = MistConfig()
    assert doctor_module.check_shell_backend(cfg).status == "ok"


def test_check_shell_backend_ssh_missing_key(tmp_path):
    cfg = MistConfig()
    cfg.tools.shell.backend = "ssh"
    cfg.tools.shell.ssh.host = "10.0.0.5"
    cfg.tools.shell.ssh.key_path = str(tmp_path / "nonexistent_key")
    result = doctor_module.check_shell_backend(cfg)
    assert result.status == "fail"
    assert "does not exist" in result.detail


def test_check_shell_backend_ssh_reachable(monkeypatch):
    cfg = MistConfig()
    cfg.tools.shell.backend = "ssh"
    cfg.tools.shell.ssh.host = "10.0.0.5"

    class FakeCompletedProcess:
        returncode = 0
        stderr = ""

    monkeypatch.setattr(doctor_module.subprocess, "run", lambda *a, **kw: FakeCompletedProcess())
    assert doctor_module.check_shell_backend(cfg).status == "ok"


def test_check_shell_backend_ssh_unreachable(monkeypatch):
    cfg = MistConfig()
    cfg.tools.shell.backend = "ssh"
    cfg.tools.shell.ssh.host = "10.0.0.5"

    class FakeCompletedProcess:
        returncode = 255
        stderr = "Connection refused"

    monkeypatch.setattr(doctor_module.subprocess, "run", lambda *a, **kw: FakeCompletedProcess())
    result = doctor_module.check_shell_backend(cfg)
    assert result.status == "fail"
    assert "Connection refused" in result.detail


def test_check_db_writable_ok(tmp_path):
    cfg = MistConfig()
    cfg.memory.db_path = str(tmp_path / "sub" / "mist.db")
    assert doctor_module.check_db_writable(cfg).status == "ok"


def test_check_db_writable_fails_when_parent_is_a_file(tmp_path):
    cfg = MistConfig()
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    cfg.memory.db_path = str(blocker / "mist.db")  # blocker is a file, not a dir
    assert doctor_module.check_db_writable(cfg).status == "fail"


def test_run_all_returns_one_result_per_check(monkeypatch):
    monkeypatch.setattr(doctor_module.LLMClient, "list_models", lambda self: ["m"])
    monkeypatch.setattr(doctor_module.LLMClient, "discover_context_length",
                        lambda self, timeout=5.0: None)
    cfg = MistConfig()
    results = doctor_module.run_all(cfg)
    names = {r.name for r in results}
    assert names == {"config", "wiki", "db", "backend", "model", "num_ctx", "embeddings", "shell"}


def test_fix_initializes_missing_wiki(tmp_path):
    cfg = MistConfig()
    cfg.wiki.root_path = str(tmp_path / "wiki")
    assert not (tmp_path / "wiki" / "SCHEMA.md").is_file()
    fixed = doctor_module.fix(cfg)
    assert any("wiki" in f for f in fixed)
    assert (tmp_path / "wiki" / "SCHEMA.md").is_file()


def test_fix_is_noop_when_wiki_already_initialized(tmp_path):
    cfg = MistConfig()
    cfg.wiki.root_path = str(tmp_path / "wiki")
    doctor_module.fix(cfg)  # first call initializes it
    assert doctor_module.fix(cfg) == []  # second call: nothing left to fix
