"""D5 (provenance re-query) offline checks. No model calls.

D4 returns the logged output of call #N; D5 re-runs call #N on the current database. Tool faults fire
only on the first matching call, so after an F1 fault D4 shows the planted value and D5 the true one."""

import json

import pytest

from handoff.agents import PROVENANCE_SCHEMA, PROVENANCE_TOOL, RunConfig, Usage, _execute
from handoff.conditions import CONDITIONS, fault_targets
from handoff.env import READ_TOOLS, WRITE_TOOLS, ToolEnv
from handoff.faults import build_faults
from handoff.handoffs import PROVENANCE_INSTRUCTION
from test_provenance import _prompts, _role

EXECUTOR_TOOLS = WRITE_TOOLS + ["calculate"]


def _lookup(cfg: RunConfig, env: ToolEnv, usage: Usage, call_id, agent: str = "executor") -> str:
    tc = {"name": PROVENANCE_TOOL, "args": {"call_id": call_id}, "id": "t1"}
    return _execute(agent, tc, EXECUTOR_TOOLS, env, cfg, usage, "")


def _f1_env() -> tuple[ToolEnv, dict]:
    """Researcher looks up task 0's F1 target order once; the fault fires on that call (#0)."""
    t = fault_targets("0", "F1")
    env = ToolEnv(faults=build_faults("0", "F1", t))
    env.call("researcher", t["tool"], dict(t["args"]), READ_TOOLS)
    assert env.log[0].injected
    return env, t


def test_d5_is_built_and_turns_on_requery():
    spec = CONDITIONS["clean-D5"]
    assert spec.built and not spec.owner
    cfg = spec.run_config("m")
    assert cfg.provenance and cfg.provenance_requery
    assert all(CONDITIONS[f"{f}-D5"].built for f in ("F1", "F2", "F3", "F4"))


def test_requery_is_off_by_default_and_in_d4():
    assert not RunConfig().provenance_requery
    assert not CONDITIONS["clean-D4"].run_config("m").provenance_requery


def test_trace_shows_the_planted_value_requery_the_true_one():
    env, t = _f1_env()
    d4 = _lookup(RunConfig(provenance=True), env, Usage(), 0)
    assert json.loads(d4)["status"] == t["wrong_value"]
    assert len(env.log) == 1  # trace is not a tool call

    usage = Usage()
    d5 = _lookup(RunConfig(provenance=True, provenance_requery=True), env, usage, 0)
    assert d5.startswith("[call #1] ")  # a new call, tagged so it can be cited
    body = d5.split("] ", 1)[1]
    assert json.loads(body)["status"] == t["true_value"]
    assert usage.tool_calls == 1 and usage.requeries == 1


def test_requery_is_logged_as_the_same_call_by_the_caller():
    """detected() finds a re-query as the same tool and args after the injected call."""
    env, t = _f1_env()
    _lookup(RunConfig(provenance=True, provenance_requery=True), env, Usage(), 0)
    orig, again = env.log
    assert (again.tool, again.args) == (orig.tool, orig.args)
    assert again.agent == "executor" and not again.error and again.injected is None


def test_requery_never_reruns_a_write():
    env = ToolEnv()
    # a denied write attempt is logged with is_write False, so the guard must go by tool name
    env.call("researcher", "cancel_pending_order", {"order_id": "#W1", "reason": "x"}, READ_TOOLS)
    assert env.log[0].tool in WRITE_TOOLS and not env.log[0].is_write
    out = _lookup(RunConfig(provenance=True, provenance_requery=True), env, Usage(), 0)
    assert out.startswith("Error") and len(env.log) == 1


def test_requery_never_reruns_a_denied_read():
    """The executor has no read tools: a denied get_order_details is logged as call #N, and re-running
    #N must not hand it the order."""
    env, t = _f1_env()
    denied = env.call("executor", t["tool"], dict(t["args"]), EXECUTOR_TOOLS)
    assert denied.startswith("Error") and env.log[1].error and len(env.log) == 2
    usage = Usage()
    out = _lookup(RunConfig(provenance=True, provenance_requery=True), env, usage, 1)
    assert out == "Error: call #1 failed; only successful read calls can be re-run."
    assert len(env.log) == 2 and usage.tool_calls == 0 and usage.requeries == 0


@pytest.mark.parametrize("bad", [7, "x", None])
def test_requery_unknown_call(bad):
    env, _ = _f1_env()
    out = _lookup(RunConfig(provenance=True, provenance_requery=True), env, Usage(), bad)
    assert out.startswith("Error: no tool call") and len(env.log) == 1


def test_requery_accepts_a_string_call_id():
    env, t = _f1_env()
    out = _lookup(RunConfig(provenance=True, provenance_requery=True), env, Usage(), "0")
    assert json.loads(out.split("] ", 1)[1])["status"] == t["true_value"]


def test_requery_counts_toward_the_budget():
    env, _ = _f1_env()
    cfg = RunConfig(provenance=True, provenance_requery=True, tool_call_budget=3)
    usage = Usage(tool_calls=3)
    assert "budget" in _lookup(cfg, env, usage, 0) and len(env.log) == 1


def test_d5_leaves_the_intake_unchanged(monkeypatch):
    d0, d5 = _prompts(monkeypatch, "clean-D0"), _prompts(monkeypatch, "clean-D5")
    assert _role(d5, "intake agent") == _role(d0, "intake agent")


def test_d5_researcher_gets_the_requery_tool(monkeypatch):
    d0, d5 = _prompts(monkeypatch, "clean-D0"), _prompts(monkeypatch, "clean-D5")
    r0, r5 = _role(d0, "research specialist"), _role(d5, "research specialist")
    assert PROVENANCE_INSTRUCTION in r5["system"]
    assert r5["tools"][:-1] == r0["tools"] and r5["tools"][-1] == PROVENANCE_TOOL


def test_d4_and_d5_bind_the_same_tool_description(monkeypatch):
    """D4 and D5 differ only in what the tool returns, not in what the agent is told it does."""
    from langchain_core.messages import AIMessage

    from handoff import agents as A
    bound = []

    class Fake:
        def bind_tools(self, schemas):
            bound.append(next(s for s in schemas if s["function"]["name"] == PROVENANCE_TOOL))
            return self

        def invoke(self, msgs):
            return AIMessage(content="done")

    monkeypatch.setattr(RunConfig, "llm", lambda self, *a, **k: Fake())
    for cfg in (RunConfig(provenance=True), RunConfig(provenance=True, provenance_requery=True)):
        A.run_agent(name="executor", system="s", user="u", tools=EXECUTOR_TOOLS, env=ToolEnv(),
                    cfg=cfg, usage=Usage())
    assert bound == [PROVENANCE_SCHEMA, PROVENANCE_SCHEMA]


# ---------------------------------------------------------------- D4/D5 wiring (executor, tagging, lookups)
from handoff.handoffs import EXECUTOR_PROVENANCE_INSTRUCTION  # noqa: E402


@pytest.mark.parametrize("cond", ["clean-D4", "clean-D5"])
def test_executor_is_told_to_check_its_sources(monkeypatch, cond):
    ex = _role(_prompts(monkeypatch, cond), "you are the executor")
    assert EXECUTOR_PROVENANCE_INSTRUCTION in ex["system"] and PROVENANCE_TOOL in ex["tools"]


@pytest.mark.parametrize("cond", ["clean-D0", "clean-D2"])
def test_executor_without_provenance_is_unchanged(monkeypatch, cond):
    ex = _role(_prompts(monkeypatch, cond), "you are the executor")
    assert EXECUTOR_PROVENANCE_INSTRUCTION not in ex["system"] and PROVENANCE_TOOL not in ex["tools"]
    assert "[call #" not in ex["system"]


def test_d4_and_d5_prompts_are_identical(monkeypatch):
    d4, d5 = _prompts(monkeypatch, "clean-D4"), _prompts(monkeypatch, "clean-D5")
    assert d4 == d5  # every agent: same system prompt, same tools


def test_researcher_must_tag_every_id_and_value(monkeypatch):
    r = _role(_prompts(monkeypatch, "clean-D4"), "research specialist")
    assert "Never pass on an ID or a value without its tag." in r["system"]
    assert EXECUTOR_PROVENANCE_INSTRUCTION not in r["system"]  # the researcher is not the last agent


def test_d4_lookup_accepts_a_string_call_id():
    env, t = _f1_env()
    out = _lookup(RunConfig(provenance=True), env, Usage(), "0")
    assert json.loads(out)["status"] == t["wrong_value"]  # trace: the logged (planted) output


@pytest.mark.parametrize("requery", [False, True])
def test_every_lookup_is_counted(requery):
    env, _ = _f1_env()
    cfg, usage = RunConfig(provenance=True, provenance_requery=requery), Usage()
    _lookup(cfg, env, usage, 0)
    _lookup(cfg, env, usage, 99)  # a failed lookup still counts as a use of the tool
    assert usage.lookups == 2
    assert usage.requeries == (1 if requery else 0)


def test_no_lookups_counted_without_provenance():
    env, _ = _f1_env()
    usage = Usage()
    _lookup(RunConfig(), env, usage, 0)
    assert usage.lookups == 0
