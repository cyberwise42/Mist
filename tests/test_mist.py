import asyncio
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from mist.config import MistConfig, ShellConfig, ShellSSHConfig
from mist.core.agent import MistAgent
from mist.core.subagent import run_subagents, run_tool_subagents
from mist.core.summarizer import BatchSummarizer
from mist.memory.store import MemoryStore
from mist.skills.router import SkillRouter
from mist.tools.registry import default_registry
from mist.llm.client import parse_json_relaxed
from mist.wiki import init_wiki


class FakeLLM:
    """Scripted LLM: pops canned replies in order."""
    def __init__(self, replies):
        self.replies = list(replies)

    def complete(self, messages, json_schema=None):
        return self.replies.pop(0)


class EchoLLM:
    """Thread-safe fake LLM: replies based on the last message's content
    instead of call order, so it's safe to use from concurrent callers."""
    def complete(self, messages, json_schema=None):
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
    agent = make_agent(tmp_path, [json.dumps({"action": "respond", "response": "hi"})])
    result = agent.turn("hello")
    assert result.response == "hi"
    # persisted to history
    turns = agent.store.recent_turns(agent.session_id, 10)
    assert turns[-1]["content"] == "hi"


def test_agent_tool_loop(tmp_path):
    agent = make_agent(tmp_path, [
        json.dumps({"action": "use_tool", "tool": "shell",
                    "arguments": {"command": "echo mist-test"}}),
        json.dumps({"action": "respond", "response": "done"}),
    ])
    result = agent.turn("run echo")
    assert result.response == "done"
    assert any("mist-test" in step for step in result.tool_trace)


def test_agent_recovers_from_bad_json(tmp_path):
    agent = make_agent(tmp_path, [
        "not json at all",
        json.dumps({"action": "respond", "response": "recovered"}),
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
    def complete(self, messages, json_schema=None):
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

    class FakeCompleted:
        stdout = "remote-host\n"
        stderr = ""

    def fake_run(args, capture_output, text, timeout):
        captured["args"] = args
        captured["timeout"] = timeout
        return FakeCompleted()

    monkeypatch.setattr("mist.tools.registry.subprocess.run", fake_run)

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

    async def acomplete(self, messages, json_schema=None):
        return self.decisions.pop(0)

    async def astream(self, messages):
        self.stream_calls += 1
        chunks = self.streams.pop(0)
        await self.release.wait()
        for c in chunks:
            yield c


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
        async def acomplete(self, messages, json_schema=None):
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
