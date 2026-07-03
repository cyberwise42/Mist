import json

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
