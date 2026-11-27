"""Offline checks of fault planting in run_eval (F1-F3 through ToolEnv, F4 through a fork). No model calls."""

import importlib.util
import json
from pathlib import Path

import pytest

from handoff import metrics as M
from handoff.conditions import CONDITIONS, fault_targets
from handoff.env import ALL_TOOLS, ToolEnv
from handoff.faults import build_faults
from handoff.tasks import load_tickets

ROOT = Path(__file__).resolve().parents[1]
TICKETS = {t.task_id: t for t in load_tickets()[:10]}


def _run_eval():
    spec = importlib.util.spec_from_file_location("run_eval", ROOT / "run_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _walk(obj, path):
    for p in path:
        obj = obj[p]
    return obj


def _gold_reads(env: ToolEnv, task_id: str) -> None:
    for a in TICKETS[task_id].expected_actions:
        env.call("gold", a["name"], dict(a["arguments"]), ALL_TOOLS)


@pytest.mark.parametrize("fault", ["F1", "F2", "F3"])
@pytest.mark.parametrize("tid", list(TICKETS))
def test_tool_fault_changes_every_path_of_the_target_call(tid, fault):
    t = fault_targets(tid, fault)
    env = ToolEnv(faults=build_faults(tid, fault, t))
    _gold_reads(env, tid)
    hit = [r for r in env.log if r.injected]
    assert len(hit) == 1 and hit[0].tool == t["tool"] and hit[0].args == t["args"]
    out = json.loads(hit[0].output)
    assert all(_walk(out, p) == t["wrong_value"] for p in t["paths"])
    assert str(t["wrong_value"]) in env.faults[0].injected_values


def test_fault_fires_only_on_the_target_order_and_only_once():
    t = fault_targets("0", "F1")
    order = t["args"]["order_id"]
    other = next(o for o in json.loads(ToolEnv().call("g", "get_user_details",
                 {"user_id": _user_of(order)}, ALL_TOOLS))["orders"] if o != order)
    env = ToolEnv(faults=build_faults("0", "F1", t))
    a = env.call("r", "get_order_details", {"order_id": other}, ALL_TOOLS)   # another order: untouched
    b = env.call("r", "get_order_details", {"order_id": order}, ALL_TOOLS)   # target: planted
    c = env.call("r", "get_order_details", {"order_id": order}, ALL_TOOLS)   # re-query: true value
    assert env.log[0].injected is None and env.log[1].injected == "F1_task0" and env.log[2].injected is None
    assert json.loads(b)["status"] == t["wrong_value"] and json.loads(c)["status"] == t["true_value"]
    assert json.loads(a) == json.loads(ToolEnv().call("g", "get_order_details", {"order_id": other}, ALL_TOOLS))


def _user_of(order_id: str) -> str:
    return json.loads(ToolEnv().call("g", "get_order_details", {"order_id": order_id}, ALL_TOOLS))["user_id"]


def test_fault_on_an_order_never_looked_up_does_not_fire():
    env = ToolEnv(faults=build_faults("0", "F2", fault_targets("0", "F2")))
    env.call("r", "get_user_details", {"user_id": _user_of(fault_targets("0", "F2")["args"]["order_id"])},
             ALL_TOOLS)
    assert not any(r.injected for r in env.log)
    rec = {"faults": [{"id": "F2_task0", "fired": False}]}
    assert M.fault_fired(rec) is False and M.fault_fired({"faults": []}) is None


def test_f4_is_not_a_tool_fault():
    with pytest.raises(ValueError):
        build_faults("0", "F4", fault_targets("0", "F4"))


def _fake_clean(handoffs):
    return {"task_id": "0", "success": True, "seconds": 5.0, "faults": [],
            "state": {"handoffs": handoffs, "tool_log": []},
            "_graph": object(), "_config": {"configurable": {"run_key": "k"}}}


def test_f4_without_the_value_in_the_handoff_is_no_target(monkeypatch):
    ev = _run_eval()
    t = fault_targets("0", "F4")
    hs = [{"from": "intake", "to": "researcher", "content": "ticket"},
          {"from": "researcher", "to": "executor", "content": "no ids here"}]
    monkeypatch.setattr(ev, "run_ticket", lambda *a, **k: _fake_clean(hs))
    monkeypatch.setattr(ev, "fork_with_false_claim", lambda *a, **k: pytest.fail("must not fork"))
    r = ev.run_false_claim(TICKETS["0"], CONDITIONS["F4-D0"], None, t)
    assert r["status"] == "no_target" and "researcher -> executor" in r["no_target_reason"]
    assert "_graph" not in r


def test_f4_forks_at_the_target_hop(monkeypatch):
    ev = _run_eval()
    t = fault_targets("0", "F4")
    hs = [{"from": "intake", "to": "researcher", "content": "ticket"},
          {"from": "researcher", "to": "executor", "content": f"exchange item {t['true_value']}"}]
    seen = {}

    def fork(clean, tk, cond, cfg, hop, true, false):
        seen.update(hop=hop, true=true, false=false)
        return {"task_id": "0", "success": False, "seconds": 2.0}

    monkeypatch.setattr(ev, "run_ticket", lambda *a, **k: _fake_clean(hs))
    monkeypatch.setattr(ev, "fork_with_false_claim", fork)
    r = ev.run_false_claim(TICKETS["0"], CONDITIONS["F4-D0"], None, t)
    assert seen == {"hop": 1, "true": t["true_value"], "false": t["false_value"]}
    assert r["status"] == "ok" and r["clean_run_seconds"] == 5.0 and r["clean_run_success"] is True


def test_no_target_records_are_not_retried(tmp_path):
    ev = _run_eval()
    for status, done in [("ok", True), ("no_target", True), ("error", False)]:
        fp = tmp_path / f"{status}.json"
        fp.write_text(json.dumps({"status": status}))
        assert ev.saved_ok(fp) is done


def test_fault_conditions_are_built_where_the_defense_is():
    for f in ("F1", "F2", "F3", "F4"):
        assert CONDITIONS[f"{f}-D0"].built and CONDITIONS[f"{f}-D2"].built and CONDITIONS[f"{f}-D4"].built
        assert not CONDITIONS[f"{f}-D1"].built  # D1, D3, D5 wait on their owners
