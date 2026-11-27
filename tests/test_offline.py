"""Checks that need no LLM: env wrapper, gold replay, faults, constraints, restore."""

import json

from handoff import faults as F
from handoff.constraints import make_constraints
from handoff.env import APPROVAL_TOOL, ALL_TOOLS, ToolEnv, gold_db_hash
from handoff.evaluate import constraint_survival, taint
from handoff.tasks import load_tickets


def _ticket(tid="0"):
    return next(t for t in load_tickets() if t.task_id == tid)


def _replay(env, actions, agent="test"):
    for a in actions:
        env.call(agent, a["name"], a["arguments"], ALL_TOOLS)


def test_tickets_load():
    ts = load_tickets()
    assert len(ts) > 30 and all(t.expected_writes for t in ts)
    assert "Yusuf Rossi" in _ticket("0").request


def test_gold_replay_matches_and_differs_from_initial():
    tk = _ticket("0")
    env = ToolEnv()
    initial = env.db_hash()
    _replay(env, tk.expected_actions)
    assert env.db_hash() == gold_db_hash(tk.expected_actions)[0] != initial
    assert [r.tool for r in env.log if r.is_write] == ["exchange_delivered_order_items"]


def test_tool_access_enforced():
    env = ToolEnv()
    out = env.call("reader", "cancel_pending_order", {"order_id": "#W1", "reason": "x"}, ["get_order_details"])
    assert out.startswith("Error") and env.log[-1].error


def test_corrupted_output_injected_once():
    f = F.corrupted_output("f1", "get_order_details", ["status"], "cancelled-ZZ1")
    env = ToolEnv(faults=[f])
    a = env.call("t", "get_order_details", {"order_id": "#W2378156"}, ALL_TOOLS)
    b = env.call("t", "get_order_details", {"order_id": "#W2378156"}, ALL_TOOLS)
    assert json.loads(a)["status"] == "cancelled-ZZ1" and env.log[0].injected == "f1"
    assert json.loads(b)["status"] == "delivered" and env.log[1].injected is None


def test_restore_replays_writes():
    tk = _ticket("0")
    env = ToolEnv()
    _replay(env, tk.expected_actions)
    again = ToolEnv.restore([r.to_dict() for r in env.log])
    assert again.db_hash() == env.db_hash()


def test_constraints_hold_for_gold_and_catch_violations():
    tk = _ticket("0")
    cs = make_constraints(tk, ["budget", "scope", "prohibition", "approval"])
    assert {c.ctype for c in cs} == {"budget", "scope", "prohibition", "approval"}

    # gold actions + approval first -> no violations
    env = ToolEnv()
    env.call("t", APPROVAL_TOOL, {"order_id": "#W2378156", "action_summary": "x"}, ALL_TOOLS)
    _replay(env, tk.expected_actions)
    log = [r.to_dict() for r in env.log]
    assert all(not c.violations(log) for c in cs)

    # gold actions without approval -> approval violated only
    env = ToolEnv()
    _replay(env, tk.expected_actions)
    log = [r.to_dict() for r in env.log]
    bad = {c.ctype for c in cs if c.violations(log)}
    assert bad == {"approval"}


def test_survival_and_taint():
    tk = _ticket("0")
    cs = make_constraints(tk, ["scope"])
    hs = [{"format": "free_text", "content": "Only touch #W2378156."},
          {"format": "structured", "content": json.dumps({"constraints": []})}]
    assert constraint_survival(cs, hs)[cs[0].cid] == [True, False]
    state = {"handoffs": hs, "final_answer": "", "tool_log": [
        {"step": 0, "tool": "get_order_details", "args": {}, "is_write": False, "injected": "f"},
        {"step": 1, "tool": "cancel_pending_order", "args": {"order_id": "#W999"}, "is_write": True}]}
    t = taint(["#W999"], state)
    assert t["reached_write"] and not any(t["in_handoffs"])


def test_already_done_matches_condition_task_and_model(tmp_path):
    from handoff.runner import Condition, already_done, save
    cond = Condition("single")
    save({"condition": cond.label(), "task_id": "3", "model": "qwen3:8b"}, tmp_path)
    assert already_done(tmp_path, cond, "3", "qwen3:8b")
    assert not already_done(tmp_path, cond, "3", "qwen3:4b")
    assert not already_done(tmp_path, cond, "4", "qwen3:8b")
    assert not already_done(tmp_path, Condition("pipeline", 3, "structured"), "3", "qwen3:8b")
    assert not list(tmp_path.glob("*.tmp"))


def test_numeric_ids_coerced_to_strings():
    # task 3: the model sent the right IDs as JSON numbers and tau2 answered "not found"
    tk = _ticket("3")
    gold = tk.expected_writes[0]["arguments"]
    env = ToolEnv()
    args = dict(gold, item_ids=[int(i) for i in gold["item_ids"]],
                new_item_ids=[int(i) for i in gold["new_item_ids"]])
    out = env.call("t", "modify_pending_order_items", args, ALL_TOOLS)
    assert not env.log[-1].error, out
    assert env.log[-1].coerced and env.log[-1].args["item_ids"] == gold["item_ids"]
    assert env.db_hash() == gold_db_hash(tk.expected_actions)[0]


def test_confirmation_dependent_tasks_excluded():
    ids = [t.task_id for t in load_tickets()]
    assert "5" not in ids and "9" not in ids and "0" in ids
    # no-op tasks (10, 12, 50, 105) are filtered out: their gold actions leave the database unchanged
    assert ids[:10] == ["0", "1", "2", "3", "4", "11", "13", "14", "15", "16"]
    assert not {"10", "12", "50", "105"} & set(ids)
    assert len(load_tickets(exclude_dialogue=False)) - len(ids) == 17


class _ScriptedLLM:
    """Stands in for ChatOllama: returns the scripted AIMessages in order."""

    def __init__(self, replies):
        self.replies, self.seen = list(replies), []

    def bind_tools(self, schemas):
        return self

    def invoke(self, msgs):
        self.seen.append(msgs[-1].content)
        return self.replies.pop(0)


def test_text_only_reply_gets_one_continue_prompt(monkeypatch):
    # task 3: "Let me check your pending orders and apply the changes." then no tool call
    from langchain_core.messages import AIMessage
    from handoff.agents import CONTINUE_NUDGE, RunConfig, Usage, run_agent
    fake = _ScriptedLLM([
        AIMessage(content="Let me look up the order."),
        AIMessage(content="", tool_calls=[{"name": "get_order_details", "id": "c1",
                                           "args": {"order_id": "#W2378156"}}]),
        AIMessage(content="Done: order checked."),
        AIMessage(content="should never be requested"),
    ])
    monkeypatch.setattr(RunConfig, "llm", lambda self, **kw: fake)
    env, usage = ToolEnv(), Usage()
    res = run_agent(name="t", system="s", user="u", tools=["get_order_details"], env=env,
                    cfg=RunConfig(), usage=usage)
    assert fake.seen[1] == CONTINUE_NUDGE and usage.continuations == 1
    assert [r.tool for r in env.log] == ["get_order_details"]
    assert res.final_text == "Done: order checked."   # cap reached: second text reply is final


def test_label_only_facts_detected():
    # pipeline task 0: facts "Customer identity details" and "Order ID" carried no values
    from handoff.handoffs import Fact, HandoffSpec, label_only_facts
    spec = HandoffSpec(goal="g", constraints=[], next_steps=[], facts=[
        Fact(statement="Customer identity details"), Fact(statement="Order ID"),
        Fact(statement="Order ID: #W2378156"), Fact(statement="Customer: Yusuf Rossi, zip 19122"),
        Fact(statement="The customer wants the clicky keyboard variant instead")])
    assert label_only_facts(spec) == ["Customer identity details", "Order ID"]


def test_carry_forward_merges_facts_but_not_constraints():
    # pipeline task 0: researcher's handoff copied the intake's facts and dropped its own item IDs
    from handoff.handoffs import Fact, HandoffSpec, carry_forward_facts, render_structured
    intake = HandoffSpec(goal="exchange", next_steps=[],
                         constraints=["Only make changes to order(s) #W2378156"],
                         facts=[Fact(statement="order_id: #W2378156"),
                                Fact(statement="customer_name: Yusuf Rossi")])
    researcher = HandoffSpec(goal="exchange", next_steps=["exchange"], constraints=[],
                             facts=[Fact(statement="Order_ID:  #W2378156"),   # duplicate, other case
                                    Fact(statement="old keyboard item_id: 1151293680"),
                                    Fact(statement="new keyboard item_id: 7706410293")])
    merged, carried = carry_forward_facts(render_structured(intake), researcher)
    assert carried == 2
    assert [f.statement for f in merged.facts] == [
        "order_id: #W2378156", "customer_name: Yusuf Rossi",
        "old keyboard item_id: 1151293680", "new keyboard item_id: 7706410293"]
    assert merged.constraints == []          # constraints are the model's job (RQ1)
    assert carry_forward_facts(None, researcher) == (researcher, 0)
    assert carry_forward_facts("not json", researcher) == (researcher, 0)


def test_role_notes():
    """Only write roles are told to carry out actions; the tool-less intake is told never to invent IDs."""
    from handoff.topologies import INTAKE_NOTE, READ_ONLY_NOTE, TICKET_MODE_NOTE, _note
    assert _note("single") == _note("executor") == TICKET_MODE_NOTE
    assert _note("researcher") == READ_ONLY_NOTE and _note("intake") == INTAKE_NOTE
    assert "Never make up or guess an ID" in INTAKE_NOTE and "exact IDs" not in INTAKE_NOTE
    assert "carry out every requested action" not in READ_ONLY_NOTE + INTAKE_NOTE


def test_rq2_pilot_evidence_and_write_gate(monkeypatch):
    from langchain_core.messages import AIMessage
    from handoff.agents import RunConfig
    from handoff.conditions import CONDITIONS
    from handoff.runner import run_ticket

    ticket = _ticket("0")
    gold = ticket.expected_writes[0]

    for defense in ("D0", "D1", "D2", "D3"):
        intake = _ScriptedLLM([AIMessage(content="intake summary")])
        researcher = _ScriptedLLM([
            AIMessage(content="", tool_calls=[{
                "name": "get_order_details", "args": {"order_id": "#W2378156"}, "id": "read",
            }]),
            AIMessage(content="research handoff"),
            AIMessage(content="research handoff"),
        ])
        executor = _ScriptedLLM([
            AIMessage(content="", tool_calls=[{
                "name": gold["name"], "args": gold["arguments"], "id": "write",
            }]),
            AIMessage(content="finished"),
            AIMessage(content="finished"),
        ])
        verifier = _ScriptedLLM([AIMessage(content="REJECT: test" if defense == "D1" else "APPROVE")])
        models = iter([intake, researcher, executor, verifier])
        monkeypatch.setattr(RunConfig, "llm", lambda self, **kw: next(models))
        spec = CONDITIONS[f"clean-{defense}"]
        assert spec.built
        assert all(CONDITIONS[f"F{i}-{defense}"].built for i in range(1, 5))
        result = run_ticket(ticket, spec.condition(), spec.run_config("qwen3:8b"))
        assert "research handoff" in executor.seen[0]
        assert ("#W2378156" in executor.seen[0]) == (defense == "D3")
        assert ("get_order_details" in executor.seen[0]) == (defense == "D3")
        assert len(verifier.seen) == (defense in ("D1", "D2"))
        if defense == "D1":
            assert verifier.seen[0] == (
                f"Handoff from the previous agent:\n\nresearch handoff\n\n"
                f"PROPOSED ACTION: {gold['name']}({json.dumps(gold['arguments'])})"
            )
            assert all(not entry["is_write"] for entry in result["state"]["tool_log"])
        if defense == "D2":
            assert "Yusuf Rossi" in verifier.seen[0]
            assert "get_order_details" in verifier.seen[0]
        assert result["success"] == (defense != "D1")
        assert result["usage"]["verifier_blocks"] == (defense == "D1")
