import json

from mist.config import MistConfig
from mist.core.agent import MistAgent
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
