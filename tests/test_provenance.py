"""D4 (provenance) must change only the agents that have tool results to tag. No model calls: the LLM
is faked and records each agent's system prompt and bound tools."""

from langchain_core.messages import AIMessage, SystemMessage

from handoff.agents import PROVENANCE_TOOL, RunConfig
from handoff.conditions import CONDITIONS
from handoff.handoffs import PROVENANCE_INSTRUCTION
from handoff.runner import run_ticket
from handoff.tasks import load_tickets


def _prompts(monkeypatch, condition: str) -> dict[str, dict]:
    """Run the pipeline on task 0 with a fake model; return each agent's first system prompt and tools."""
    seen: dict[str, dict] = {}

    class Fake:
        def __init__(self):
            self.tools: list[str] = []

        def bind_tools(self, schemas):
            self.tools = [s["function"]["name"] for s in schemas]
            return self

        def invoke(self, msgs):
            system = next(m.content for m in msgs if isinstance(m, SystemMessage))
            role = system.split("\n", 1)[0]
            seen.setdefault(role, {"system": system, "tools": self.tools})
            return AIMessage(content="Customer yusuf.rossi@example.com, zip 19122, wants an exchange.")

    monkeypatch.setattr(RunConfig, "llm", lambda self, *a, **k: Fake())
    spec = CONDITIONS[condition]
    run_ticket(load_tickets()[0], spec.condition(), spec.run_config("fake"))
    return seen


def _role(seen: dict, word: str) -> dict:
    return next(v for k, v in seen.items() if word in k.lower())


def test_d4_leaves_the_intake_unchanged(monkeypatch):
    d0, d4 = _prompts(monkeypatch, "clean-D0"), _prompts(monkeypatch, "clean-D4")
    assert _role(d4, "intake agent") == _role(d0, "intake agent")
    assert PROVENANCE_INSTRUCTION not in _role(d4, "intake agent")["system"]
    assert PROVENANCE_TOOL not in _role(d4, "intake agent")["tools"]


def test_d4_still_tags_the_researcher(monkeypatch):
    d0, d4 = _prompts(monkeypatch, "clean-D0"), _prompts(monkeypatch, "clean-D4")
    r0, r4 = _role(d0, "research specialist"), _role(d4, "research specialist")
    assert PROVENANCE_INSTRUCTION in r4["system"] and PROVENANCE_INSTRUCTION not in r0["system"]
    assert PROVENANCE_TOOL in r4["tools"] and PROVENANCE_TOOL not in r0["tools"]
    assert r4["tools"][:-1] == r0["tools"]  # same environment tools, plus the record lookup
