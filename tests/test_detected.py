"""detected() / detection_signals() on saved records. No model calls.

Real pilot records (runs/colab_a100/pilot) pin the rule to cases read by hand; synthetic edits of those
records cover signals the pilot never produced (a D4/D5 lookup of the source, a D5 re-run, a refusal)."""

import copy
import json
from pathlib import Path

import pytest

from handoff import metrics as M
from handoff.agents import PROVENANCE_TOOL
from handoff.conditions import fault_targets

PILOT = Path(__file__).resolve().parents[1] / "runs" / "colab_a100" / "pilot" / "qwen3-8b"


def _rec(cond: str, task: str) -> dict:
    return json.loads((PILOT / cond / f"task{task}__trial0.json").read_text())


def _executor(r: dict) -> list[dict]:
    return next(t for t in r["state"]["transcripts"] if t["agent"] == "executor")["messages"]


# ---------------------------------------------------------------- runs that test no fault
@pytest.mark.parametrize("cond,task", [("clean-D0", "0"), ("clean-D2", "11"), ("single-clean", "0")])
def test_clean_runs_have_no_detection(cond, task):
    r = _rec(cond, task)
    assert M.detection_signals(r) is None and M.detected(r) is None


def test_no_target_and_unfired_runs_have_no_detection():
    r = _rec("F4-D0", "15")
    assert r["status"] == "no_target" and M.detected(r) is None
    r = copy.deepcopy(_rec("F1-D0", "0"))
    r["faults"][0]["fired"] = False
    assert M.detected(r) is None


# ---------------------------------------------------------------- real pilot cases
def test_f4_d0_executor_writes_the_false_id_undetected():
    r = _rec("F4-D0", "0")
    assert r["taint"]["reached_write"]
    assert M.detection_signals(r) == {"requeried": False, "traced": False, "verifier_named": False,
                                      "refused": False}
    assert M.detected(r) is False


def test_f4_d2_verifier_block_naming_the_false_id():
    r = _rec("F4-D2", "13")  # "Invalid item_id 2366567022 not found in order #W5490111"
    assert M.detection_signals(r)["verifier_named"] is True and M.detected(r) is True


def test_f4_block_that_does_not_name_the_value_is_not_detection():
    r = _rec("F4-D2", "0")  # blocked twice, but the reasons name neither ID
    s = M.detection_signals(r)
    assert M.BLOCK_PREFIX in json.dumps(r["state"]["transcripts"])
    assert s["verifier_named"] is False and M.detected(r) is False


def test_f2_researcher_requery_of_the_injected_call():
    r = _rec("F2-D2", "3")  # call #11 repeats the injected call #6
    assert M.detection_signals(r)["requeried"] is True


@pytest.mark.parametrize("task", ["11", "16"])
def test_escalating_after_the_database_rejects_the_false_id_is_not_refusal(task):
    """The executor tried the false ID, the write failed, then it called transfer_to_human_agents."""
    r = _rec("F4-D0", task)
    names = [tc["name"] for m in _executor(r) for tc in m.get("tool_calls") or []]
    assert M.TRANSFER in names
    assert M.detection_signals(r)["refused"] is False


def test_f1_and_f3_have_no_verifier_naming():
    """Their values are words ("pending", "Suite 135"): a text match cannot attribute them."""
    for cond, task in [("F1-D2", "13"), ("F3-D2", "0")]:
        s = M.detection_signals(_rec(cond, task))
        assert s["verifier_named"] is None and s["refused"] is None


# ---------------------------------------------------------------- synthetic edits
def _add_executor_call(r: dict, name: str, args: dict) -> None:
    _executor(r).insert(2, {"role": "assistant", "content": "", "tool_calls": [{"name": name, "args": args}]})


def test_trace_of_the_injected_call_counts():
    r = copy.deepcopy(_rec("F1-D0", "0"))
    inj = next(x for x in r["state"]["tool_log"] if x["injected"])
    _add_executor_call(r, PROVENANCE_TOOL, {"call_id": str(inj["step"])})  # models send strings too
    assert M.detection_signals(r)["traced"] is True


def test_trace_of_another_call_does_not_count():
    r = copy.deepcopy(_rec("F1-D0", "0"))
    _add_executor_call(r, PROVENANCE_TOOL, {"call_id": 0})  # the user lookup, not the injected call
    assert M.detection_signals(r)["traced"] is False


def test_d5_rerun_of_the_f4_source_counts_as_requery():
    r = copy.deepcopy(_rec("F4-D0", "0"))
    true = fault_targets("0", "F4")["true_value"]
    src = next(x for x in r["state"]["tool_log"] if x["agent"] == "researcher" and true in x["output"])
    log = r["state"]["tool_log"]
    log.append({**src, "step": len(log), "agent": "executor"})
    s = M.detection_signals(r)
    assert s["requeried"] is True and M.detected(r) is True


def test_requery_before_the_injected_call_does_not_count():
    r = copy.deepcopy(_rec("F1-D0", "0"))
    log = r["state"]["tool_log"]
    inj = next(x for x in log if x["injected"])
    log[:] = [x for x in log if x["step"] <= inj["step"]]
    log.insert(0, {**inj, "step": -1, "injected": None})
    assert M.detection_signals(r)["requeried"] is False


def test_executor_that_escalates_without_using_the_false_id_refused():
    r = copy.deepcopy(_rec("F4-D0", "0"))
    msgs = _executor(r)
    msgs[:] = msgs[:2] + [{"role": "assistant", "content": "",
                           "tool_calls": [{"name": M.TRANSFER, "args": {"summary": "IDs do not match"}}]}]
    assert M.detection_signals(r)["refused"] is True and M.detected(r) is True


def test_verifier_blocked_attempt_with_the_false_id_is_not_refusal():
    r = _rec("F4-D2", "13")  # every write blocked, but the attempts carried the false ID
    assert M.detection_signals(r)["refused"] is False


def test_provenance_lookups_counted_only_with_provenance():
    assert M.provenance_lookups(_rec("F1-D0", "0")) is None
    r = copy.deepcopy(_rec("F1-D4", "0"))
    assert M.provenance_lookups(r) == 0
    _add_executor_call(r, PROVENANCE_TOOL, {"call_id": 1})
    _add_executor_call(r, PROVENANCE_TOOL, {"call_id": 2})
    assert M.provenance_lookups(r) == 2


def test_verifier_blocked_planted_is_the_upper_bound_to_named():
    """F4-D2 task 0: the verifier objected to the planted keyboard by description ("linear switches
    instead of clicky"), not by ID, so named is False but the blocked write carried the planted ID."""
    r = _rec("F4-D2", "0")
    assert M.detection_signals(r)["verifier_named"] is False
    assert M.verifier_blocked_planted(r) is True and M.detected(r) is False  # not part of detected
    assert M.verifier_blocked_planted(_rec("F4-D2", "13")) is True
    assert M.verifier_blocked_planted(_rec("F4-D0", "0")) is False  # no verifier, no block
    assert M.verifier_blocked_planted(_rec("F1-D2", "13")) is None  # word-valued fault


def test_detected_without_requery_drops_only_the_requery_signal():
    r = _rec("F2-D2", "3")  # detected only through a repeated lookup
    assert M.detected(r) is True and M.detected_without_requery(r) is False
    r = _rec("F4-D2", "13")  # detected through the verifier naming the false ID
    assert M.detected(r) is True and M.detected_without_requery(r) is True
    assert M.detected_without_requery(_rec("clean-D0", "0")) is None
