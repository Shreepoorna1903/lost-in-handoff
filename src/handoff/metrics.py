"""FP2 metrics, computed from saved run records only. No model calls, so records can be rescored
whenever a metric changes.

Implemented:
  success               final DB equals the gold replay. Recomputed here by replaying the run's logged
                        writes (ToolEnv.restore), not read from the saved flag.
  success_ci            Wilson 95% interval for a success rate
  passed_on             the planted value appears in a later handoff        (from record["taint"])
  wrong_value_write     the planted value appears in a write call's args    (from record["taint"])
  fault_induced_failure a task the clean baseline solved fails under the fault (paired by task)
  cost                  mean seconds, tokens, LLM calls, tool calls, truncated calls per run

TODO (Puneet):
  detected              an agent notices the planted fact (re-query of the source, verifier block
                        naming it, or the executor refusing). Define the rule, then hand-check it on
                        20 runs together with the metrics above.
  over_blocking         in clean-D1..D5 runs, correct writes the defense stopped (from
                        usage.verifier_blocks and the tool log), relative to clean-D0.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

from handoff.conditions import LEGACY_LABELS
from handoff.env import ToolEnv, gold_db_hash
from handoff.tasks import load_tickets


def load(folder: str | Path) -> list[dict]:
    """Every run record under `folder` (one .json file per run, any depth)."""
    out = []
    for fp in sorted(Path(folder).rglob("*.json")):
        try:
            r = json.loads(fp.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(r, dict) and "task_id" in r:
            r["_file"] = str(fp)
            out.append(r)
    return out


def condition_name(r: dict) -> str:
    """The conditions.py name; older records are mapped from the runner's label when they are clean."""
    if r.get("condition_name"):
        return r["condition_name"]
    clean = not r.get("faults") and not r.get("verifier") and not r.get("provenance")
    return LEGACY_LABELS.get(r.get("condition", ""), r.get("condition", "?")) if clean else r.get("condition", "?")


@lru_cache(maxsize=None)
def _gold_hash(task_id: str) -> str:
    tickets = {t.task_id: t for t in load_tickets(exclude_dialogue=False)}
    return gold_db_hash(tickets[task_id].expected_actions)[0]


def success(r: dict) -> bool:
    """Replay the run's successful writes on a fresh DB and compare with the gold replay."""
    if r.get("status") == "error" or "state" not in r:
        return False
    return ToolEnv.restore(r["state"].get("tool_log", [])).db_hash() == _gold_hash(r["task_id"])


def success_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes in n runs."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def passed_on(r: dict) -> bool | None:
    t = r.get("taint")
    return None if not t else any(t.get("in_handoffs", []))


def wrong_value_write(r: dict) -> bool | None:
    t = r.get("taint")
    return None if not t else bool(t.get("reached_write"))


def fault_induced_failures(fault_runs: list[dict], baseline_runs: list[dict]) -> tuple[int, int]:
    """(failures, eligible): among tasks the baseline solved in every run, fault runs that failed."""
    solved = {}
    for r in baseline_runs:
        solved[r["task_id"]] = solved.get(r["task_id"], True) and r["_success"]
    eligible = [r for r in fault_runs if solved.get(r["task_id"])]
    return sum(not r["_success"] for r in eligible), len(eligible)


def detected(r: dict) -> bool | None:
    return None  # TODO (Puneet): see module docstring


def over_blocking(r: dict) -> int | None:
    return None  # TODO (Puneet): see module docstring


def cost(runs: list[dict]) -> dict:
    ok = [r for r in runs if r.get("status") != "error"]
    if not ok:
        return {}

    def mean(f):
        return sum(f(r) for r in ok) / len(ok)

    u = lambda r: r.get("usage", {})  # noqa: E731
    return {
        "seconds": mean(lambda r: r.get("seconds", 0)),
        "tokens": mean(lambda r: u(r).get("input_tokens", 0) + u(r).get("output_tokens", 0)),
        "llm_calls": mean(lambda r: u(r).get("llm_calls", 0)),
        "tool_calls": mean(lambda r: u(r).get("tool_calls", 0)),
        "truncated": sum(u(r).get("truncated", 0) for r in ok),
    }
