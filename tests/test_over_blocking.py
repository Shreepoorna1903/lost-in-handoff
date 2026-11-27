"""over_blocking() / lost_correct_write() on saved pilot records. No model calls.

A block counts as over-blocking when the write it stopped equals one of the task's gold writes."""

import copy
import json
from pathlib import Path

import pytest

from handoff import metrics as M

PILOT = Path(__file__).resolve().parents[1] / "runs" / "colab_a100" / "pilot" / "qwen3-8b"


def _rec(cond: str, task: str) -> dict:
    return json.loads((PILOT / cond / f"task{task}__trial0.json").read_text())


def test_no_verifier_no_over_blocking():
    for cond in ("clean-D0", "clean-D4", "F2-D0"):
        r = _rec(cond, "11")
        assert M.over_blocking(r) is None and M.lost_correct_write(r) is None


def test_blocked_correct_writes_that_the_executor_got_past():
    """Task 11: all 4 blocks stopped gold returns; retries went through and the task was solved."""
    r = _rec("clean-D2", "11")
    assert M.over_blocking(r) == 4 and M.lost_correct_write(r) is False and r["success"]


def test_blocked_correct_write_that_never_happened():
    """Task 16: the gold cancel of #W8665881 was blocked and never made."""
    r = _rec("clean-D2", "16")
    assert M.over_blocking(r) == 1 and M.lost_correct_write(r) is True


def test_blocking_a_wrong_write_is_not_over_blocking():
    """Task 0: the blocked exchange used item IDs not in the order (not a gold write)."""
    r = _rec("clean-D2", "0")
    assert len(M.verifier_blocks(r)) == 1
    assert M.over_blocking(r) == 0 and M.lost_correct_write(r) is False


@pytest.mark.parametrize("task", ["1", "3", "14", "15"])
def test_failures_without_a_block_are_not_blamed_on_the_verifier(task):
    r = _rec("clean-D2", task)
    assert not M.success(r) and M.over_blocking(r) == 0 and M.lost_correct_write(r) is False


def test_transcript_blocks_match_usage_counter():
    for fp in PILOT.glob("*-D2/*.json"):
        r = json.loads(fp.read_text())
        if r.get("status") == "ok":
            assert len(M.verifier_blocks(r)) == r["usage"]["verifier_blocks"], fp


def test_matching_ignores_argument_and_list_order_and_number_ids():
    gold = M._norm_call("exchange_delivered_order_items",
                        {"order_id": "#W1", "item_ids": ["1", "2"], "new_item_ids": ["3", "4"],
                         "payment_method_id": "gift_card_1"})
    same = M._norm_call("exchange_delivered_order_items",
                        {"payment_method_id": "gift_card_1", "new_item_ids": [4, 3], "item_ids": [2, 1],
                         "order_id": "#W1"})
    swapped = M._norm_call("exchange_delivered_order_items",
                           {"order_id": "#W1", "item_ids": ["1", "2"], "new_item_ids": ["4", "3"],
                            "payment_method_id": "gift_card_1"})
    assert same == gold and swapped != gold  # pairs matter: 1->3, 2->4 is not 1->4, 2->3
    assert M._norm_call("return_delivered_order_items", {"item_ids": ["b", "a"]}) == \
        M._norm_call("return_delivered_order_items", {"item_ids": ["a", "b"]})


def test_blocked_gold_write_made_later_is_not_lost():
    r = copy.deepcopy(_rec("clean-D2", "16"))
    blocked = next(b for b in M.verifier_blocks(r) if b["tool"] == "cancel_pending_order")
    log = r["state"]["tool_log"]
    log.append({"step": len(log), "agent": "executor", "tool": blocked["tool"], "args": blocked["args"],
                "output": "{}", "error": False, "is_write": True, "injected": None, "coerced": False})
    assert M.over_blocking(r) == 1 and M.lost_correct_write(r) is False
