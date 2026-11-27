"""Offline checks of data/fault_targets.json (scripts/pick_fault_targets.py). No model calls."""

import importlib.util
import json
from functools import lru_cache
from pathlib import Path

import pytest

from handoff.conditions import FAULT_TARGETS_FILE, fault_targets
from handoff.env import ALL_TOOLS, ToolEnv
from handoff.tasks import load_tickets

ROOT = Path(__file__).resolve().parents[1]
TOOL_KEYS = {"tool", "occurrence", "path", "true_value", "wrong_value", "args"}
F4_KEYS = {"hop", "true_value", "false_value"}


def _picker():
    spec = importlib.util.spec_from_file_location("pick_fault_targets",
                                                  ROOT / "scripts" / "pick_fault_targets.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TARGETS = json.loads(FAULT_TARGETS_FILE.read_text())
TICKETS = {t.task_id: t for t in load_tickets()[:10]}


@lru_cache(maxsize=None)
def _replay(task_id: str) -> tuple:
    return tuple(json.dumps(c) for c in _picker().replay_reads(TICKETS[task_id]))


def _call(task_id: str, entry: dict) -> dict:
    calls = [json.loads(c) for c in _replay(task_id)]
    return next(c for c in calls if c["tool"] == entry["tool"] and c["occurrence"] == entry["occurrence"]
                and not c["error"])


def _walk(obj, path):
    for p in path:
        obj = obj[p]
    return obj


def _is_item_id(item_id: str) -> bool:
    """True if some product in the database has a variant with this item ID."""
    te = ToolEnv()
    products = te.db_dump()["products"]
    return any(item_id in p["variants"] for p in products.values())


def _gold_strings(task_id: str) -> set[str]:
    return _picker()._gold_strings(TICKETS[task_id])


def test_file_matches_a_fresh_pick():
    """The script is deterministic and the committed file is up to date."""
    assert _picker().pick_all() == TARGETS


def test_every_task_has_all_four_faults():
    assert set(TARGETS) == set(TICKETS)
    for tid, t in TARGETS.items():
        assert set(t) == {"F1", "F2", "F3", "F4"}, tid
        for f in ("F1", "F2", "F3"):
            assert set(t[f]) == TOOL_KEYS, (tid, f)
            assert t[f]["true_value"] != t[f]["wrong_value"], (tid, f)
            assert type(t[f]["true_value"]) is type(t[f]["wrong_value"]), (tid, f)
        assert set(t["F4"]) == F4_KEYS and t["F4"]["hop"] == 1, tid


@pytest.mark.parametrize("tid", sorted(TARGETS, key=int))
def test_each_path_exists_in_the_real_tool_output(tid):
    for f in ("F1", "F2", "F3"):
        e = TARGETS[tid][f]
        c = _call(tid, e)
        assert c["args"] == e["args"], (tid, f)
        assert _walk(json.loads(c["output"]), e["path"]) == e["true_value"], (tid, f)


@pytest.mark.parametrize("tid", sorted(TARGETS, key=int))
def test_f2_swaps_in_another_valid_id(tid):
    e = TARGETS[tid]["F2"]
    assert e["path"][-1] == "item_id" and e["wrong_value"] != e["true_value"]
    assert _is_item_id(e["wrong_value"])
    order = json.loads(_call(tid, e)["output"])
    assert e["wrong_value"] not in {it["item_id"] for it in order["items"]}
    assert e["true_value"] in _gold_strings(tid) and e["wrong_value"] not in _gold_strings(tid)


@pytest.mark.parametrize("tid", sorted(TARGETS, key=int))
def test_f3_field_is_not_used_by_any_gold_action(tid):
    e = TARGETS[tid]["F3"]
    assert e["true_value"] not in _gold_strings(tid)
    for a in TICKETS[tid].expected_actions:
        assert e["path"][-1] not in a["arguments"], (tid, a["name"])


@pytest.mark.parametrize("tid", sorted(TARGETS, key=int))
def test_f1_hits_a_value_the_gold_write_depends_on(tid):
    """F1 must change a value that decides the gold write, or it behaves like the F3 control: the
    availability of the gold new item, or the status of the order the gold write acts on."""
    e = TARGETS[tid]["F1"]
    w = next(a for a in TICKETS[tid].expected_actions if a["arguments"].get("item_ids"))["arguments"]
    if e["path"][-1] == "available":
        assert e["path"][1] in w["new_item_ids"] and e["true_value"] is True
    else:
        assert e["path"] == ["status"] and e["args"]["order_id"] == w["order_id"]


@pytest.mark.parametrize("tid", sorted(TARGETS, key=int))
def test_f4_false_value_is_a_different_valid_id_for_a_gold_argument(tid):
    e = TARGETS[tid]["F4"]
    assert e["true_value"] in _gold_strings(tid) and e["false_value"] not in _gold_strings(tid)
    assert e["false_value"] != e["true_value"] and _is_item_id(e["false_value"])


def test_fault_targets_loader():
    assert fault_targets("0", "F2") == TARGETS["0"]["F2"]
    with pytest.raises(KeyError, match="task 9999"):
        fault_targets("9999", "F1")
    with pytest.raises(KeyError, match="unknown fault"):
        fault_targets("0", "F9")
