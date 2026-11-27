"""Offline checks for the conditions table and the scoring code (no model calls)."""

from handoff import metrics as M
from handoff.conditions import BASELINE, CONDITIONS, DEFENSES, FAULTS


def test_every_fault_defense_pair_is_a_condition():
    for f in FAULTS:
        for d in DEFENSES:
            assert CONDITIONS[f"{f}-{d}"].fault == f
            assert CONDITIONS[f"{f}-{d}"].defense == d
    assert CONDITIONS[BASELINE].fault is None and CONDITIONS[BASELINE].defense == "D0"


def test_conditions_differ_only_in_the_manipulated_settings():
    base = CONDITIONS[BASELINE].run_config("qwen3:8b")
    for spec in CONDITIONS.values():
        cfg = spec.run_config("qwen3:8b")
        changed = {k for k in vars(base) if getattr(base, k) != getattr(cfg, k)}
        assert changed <= {"verifier", "provenance", "provenance_requery"}, (spec.name, changed)


def test_wilson_interval():
    lo, hi = M.success_ci(8, 10)
    assert 0.49 < lo < 0.50 and 0.94 < hi < 0.95
    assert M.success_ci(0, 0) == (0.0, 0.0)


def test_legacy_pilot_labels_map_to_clean_conditions():
    rec = {"condition": "pipeline-d3-free_text", "faults": [], "verifier": False, "provenance": False}
    assert M.condition_name(rec) == "clean-D0"
    rec["verifier"] = True
    assert M.condition_name(rec) == "pipeline-d3-free_text"  # not clean: never mapped


def test_fault_induced_failure_is_paired_by_task():
    base = [{"task_id": "0", "_success": True}, {"task_id": "1", "_success": False}]
    fault = [{"task_id": "0", "_success": False}, {"task_id": "1", "_success": False}]
    assert M.fault_induced_failures(fault, base) == (1, 1)  # task 1 is not eligible


def test_error_records_count_as_failures():
    assert M.success({"task_id": "0", "status": "error"}) is False


def test_no_op_tasks_still_score():
    """load_tickets drops no-op tasks for running, but saved records for them (pilot tasks 10, 12) must
    still be scored against their gold replay, not crash with KeyError."""
    from handoff import metrics as M
    assert M._gold_hash("10") and M._gold_hash("12")


def test_fault_induced_failure_skips_runs_where_the_fault_never_fired():
    base = [{"task_id": "0", "_success": True}, {"task_id": "1", "_success": True}]
    fault = [{"task_id": "0", "_success": False, "faults": [{"id": "F1_task0", "fired": True}]},
             {"task_id": "1", "_success": False, "faults": [{"id": "F1_task1", "fired": False}]}]
    assert M.fault_induced_failures(fault, base) == (1, 1)  # task 1 never saw the fault
