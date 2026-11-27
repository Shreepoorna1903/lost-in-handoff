"""Run one condition on the task set and save one record per run. Scores nothing (see score_eval.py).

    python run_eval.py --condition clean-D0 --limit 10
    python run_eval.py --condition F2-D1 --tasks 0,2,11 --runs 3
    python run_eval.py --condition clean-D0 --limit 10 --shard 1/4    # 4 processes, one Ollama server

Records go to runs/<model>/<condition>/task<ID>__trial<K>.json. Running the same command again resumes:
finished trials are skipped. A run that crashes is saved as an error record, not dropped, and is
retried on the next run of the command.
Conditions are defined in src/handoff/conditions.py.

Faults (targets from data/fault_targets.json, see scripts/pick_fault_targets.py):
  F1-F3  planted in a tool output by ToolEnv, on the first read of the target order.
  F4     a clean run of the same condition, then a fork after the researcher's handoff with the value
         misreported (runner.fork_with_false_claim). If the researcher never wrote the value, there is
         nothing to rewrite: the record is saved with status "no_target" and is not retried.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from handoff.conditions import CONDITIONS, Spec, fault_targets
from handoff.agents import RunConfig
from handoff.faults import build_faults
from handoff.runner import fork_with_false_claim, run_ticket
from handoff.tasks import Ticket, load_tickets
from handoff.topologies import RUNTIME

DONE = ("ok", "no_target")  # statuses that count as finished; error records are retried


def results_folder(root: str, model: str, condition: str) -> Path:
    """runs/qwen3-8b/clean-D0 for model "qwen3:8b"."""
    return Path(root) / model.replace(":", "-").replace("/", "-") / condition


def write(record: dict, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    fp = folder / f"task{record['task_id']}__trial{record['trial']}.json"
    tmp = fp.with_suffix(".tmp")  # write then rename, so a killed session never leaves half a file
    tmp.write_text(json.dumps({k: v for k, v in record.items() if not k.startswith("_")},
                              indent=2, default=str))
    tmp.rename(fp)
    return fp


def saved_ok(fp: Path) -> bool:
    """True if this trial already has a finished record. Error records are retried (and overwritten)."""
    try:
        return json.loads(fp.read_text()).get("status") in DONE
    except (OSError, json.JSONDecodeError):
        return False


def run_false_claim(tk: Ticket, spec: Spec, cfg: RunConfig, target: dict) -> dict:
    """F4: a clean run, then a fork at the target handoff with true_value replaced by false_value."""
    clean = run_ticket(tk, spec.condition(), cfg, keep_graph=True)
    run_key = clean["_config"]["configurable"]["run_key"]
    try:
        hop, true = target["hop"], target["true_value"]
        hs = clean["state"].get("handoffs", [])
        reason = (f"run has {len(hs)} handoffs, none at hop {hop}" if hop >= len(hs) else
                  None if true in hs[hop]["content"] else
                  f"{true!r} is not in the {hs[hop]['from']} -> {hs[hop]['to']} handoff")
        if reason:  # nothing to misreport; the clean run is kept for reference
            rec = {k: v for k, v in clean.items() if not k.startswith("_")}
            return {**rec, "status": "no_target", "no_target_reason": reason}
        r = fork_with_false_claim(clean, tk, spec.condition(), cfg, hop, true, target["false_value"])
        # `seconds` covers the replay after the fork; the clean prefix ran before it
        r.update(status="ok", clean_run_seconds=clean["seconds"], clean_run_success=clean["success"])
        return r
    finally:
        RUNTIME.pop(run_key, None)


def run_one(tk: Ticket, spec: Spec, cfg: RunConfig) -> dict:
    """One trial of one condition: clean, a tool-level fault (F1-F3) or a false claim (F4)."""
    if spec.fault == "F4":
        return run_false_claim(tk, spec, cfg, fault_targets(tk.task_id, "F4"))
    faults = build_faults(tk.task_id, spec.fault, fault_targets(tk.task_id, spec.fault)) if spec.fault else []
    return {**run_ticket(tk, spec.condition(), cfg, faults=faults), "status": "ok"}


def main() -> None:
    p = argparse.ArgumentParser(description="Run one experimental condition and save the records.")
    p.add_argument("--condition", required=True, choices=list(CONDITIONS))
    p.add_argument("--model", default="qwen3:8b")
    p.add_argument("--tasks", help="comma-separated task ids; otherwise the first --limit tasks")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--runs", type=int, default=1, help="trials per task; trial k uses seed k")
    p.add_argument("--shard", default="", help="K/N: run every N-th task starting at K (1-based)")
    p.add_argument("--out", default="runs")
    a = p.parse_args()

    spec = CONDITIONS[a.condition]
    if not spec.built:
        sys.exit(f"{a.condition} is not built yet (owner: {spec.owner or 'unassigned'}). "
                 f"See src/handoff/conditions.py.")

    tickets = load_tickets()
    if a.tasks:
        keep = set(a.tasks.split(","))
        tickets = [t for t in tickets if t.task_id in keep]
    else:
        tickets = tickets[: a.limit]
    if a.shard:
        k, n = (int(x) for x in a.shard.split("/"))
        tickets = tickets[k - 1 :: n]

    folder = results_folder(a.out, a.model, a.condition)
    for tk in tickets:
        for trial in range(a.runs):
            if saved_ok(folder / f"task{tk.task_id}__trial{trial}.json"):
                print(f"task {tk.task_id:>3} trial {trial} skipped (already saved)", flush=True)
                continue
            meta = {"condition_name": a.condition, "fault": spec.fault, "defense": spec.defense,
                    "trial": trial, "trial_id": f"{a.condition}:{tk.task_id}:{trial}",
                    "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
            try:
                r = run_one(tk, spec, spec.run_config(a.model, seed=trial))
                r.update(meta)
            except Exception as e:  # recorded as an outcome, never silently dropped
                r = {"task_id": tk.task_id, "model": a.model, **meta, "status": "error", "success": False,
                     "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-3000:]}
            fp = write(r, folder)
            print(f"task {tk.task_id:>3} trial {trial} {r['status']:<5} success={r.get('success')} "
                  f"{r.get('seconds', '-')}s -> {fp}", flush=True)


if __name__ == "__main__":
    main()
