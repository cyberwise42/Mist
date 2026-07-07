import asyncio
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from mist.config import MistConfig, ShellConfig, ShellSSHConfig
from mist.core.agent import MistAgent
from mist.core.mission import MissionControl
from mist.core.subagent import run_subagents, run_tool_subagents
from mist.core.summarizer import BatchSummarizer
from mist.memory.store import MemoryStore
from mist.skills.router import SkillRouter
from mist.tools.registry import ProcessRegistry, default_registry
from mist.llm.client import parse_json_relaxed
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


def test_skill_routing(tmp_path):
    router = SkillRouter(_isolated_skills_library(tmp_path))
    hits = router.route("help me commit my git changes")
    assert hits and hits[0].name == "git-workflow"
    assert router.route("bake a chocolate cake") == []


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
    assert args[-1] == "hostname"
    assert captured["timeout"] == 90
    assert captured["shell"] is False


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

    remote_command = captured["args"][-1]
    assert "cd '/home/kali/.mist/workspace'" in remote_command
    assert remote_command.endswith("&& ls")


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
                            max_exposed=5, always={"finish_objective"})
    assert len(selected) == 6  # 5 ranked + 1 forced, not 5 total
    names = {t.name for t in selected}
    assert "finish_objective" in names
    assert {"read_file", "write_file", "search_files", "shell", "remember"} <= names


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
        max_exposed=5,
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
    assert "backend unreachable" in events[-1].text


async def test_stream_answer_survives_mid_stream_exception(tmp_path):
    agent = make_async_agent(tmp_path, RaisingMidStreamLLM())
    events = [e async for e in agent.astream_turn("hi")]
    assert events[-1].kind == "error"
    assert "connection reset" in events[-1].text
    # The partial text generated before the crash isn't silently discarded.
    turns = agent.store.recent_turns(agent.session_id, 10)
    assert turns and "partial" in turns[-1]["content"]


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

    cli_module.chat(config=None, backend=None, model=None, base_url=None)  # must not raise

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

    cli_module.chat(config=None, backend=None, model=None, base_url=None)  # must not raise

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

from mist.llm.client import LLMClient  # noqa: E402


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


async def test_mission_stuck_recovery_enables_thinking_then_reverts(tmp_path):
    # The reasoning-assisted recovery attempt must actually request thinking
    # for that one turn (force_think=True), and only that turn — not every
    # decision, which would make every routine turn needlessly slow (a real
    # decision call with thinking enabled took ~13s in practice vs
    # sub-second without).
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
    assert llm.force_think_calls == [False, False, True, False, False]


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
