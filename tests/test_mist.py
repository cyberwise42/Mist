import asyncio
import json

import pytest

from mist.config import MistConfig
from mist.core.agent import MistAgent
from mist.core.subagent import run_subagents
from mist.core.summarizer import BatchSummarizer
from mist.memory.store import MemoryStore
from mist.skills.router import SkillRouter
from mist.tools.registry import default_registry
from mist.llm.client import parse_json_relaxed


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


def test_skill_routing():
    router = SkillRouter("skills_library")
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

def test_skill_routing_falls_back_to_embeddings_when_lexical_fails():
    router = SkillRouter("skills_library", embeddings=FakeEmbeddings(),
                         embedding_threshold=0.5)
    # Shares no tokens with the git-workflow skill's name/description.
    hits = router.route("how do I track changes to my project over time")
    assert hits and hits[0].name == "git-workflow"


def test_skill_routing_embedding_fallback_respects_threshold():
    router = SkillRouter("skills_library", embeddings=FakeEmbeddings(),
                         embedding_threshold=0.999)
    assert router.route("how do I track changes to my project over time") == []


def test_skill_routing_prefers_lexical_over_embeddings():
    # Lexical match exists, so the (deliberately wrong) fake embeddings must
    # never be consulted.
    router = SkillRouter("skills_library", embeddings=FakeEmbeddings())
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
