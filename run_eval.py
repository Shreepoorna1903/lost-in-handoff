"""Run one condition on the task set and save one record per run. Scores nothing (see score_eval.py).

    python run_eval.py --condition clean-D0 --limit 10
    python run_eval.py --condition F2-D1 --tasks 0,2,11 --runs 3
    python run_eval.py --condition clean-D0 --limit 10 --shard 1/4    # 4 processes, one Ollama server

Records go to runs/<model>/<condition>/task<ID>__trial<K>.json. Running the same command again resumes:
finished trials are skipped. A run that crashes is saved as an error record, not dropped, and is
retried on the next run of the command.
Conditions are defined in src/handoff/conditions.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from handoff.conditions import CONDITIONS, fault_targets
from handoff.runner import run_ticket
from handoff.tasks import load_tickets


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
        return json.loads(fp.read_text()).get("status") == "ok"
    except (OSError, json.JSONDecodeError):
        return False


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
                if spec.fault:
                    fault_targets(tk.task_id, spec.fault)  # TODO (S+M): build faults from targets, fork for F4
                r = run_ticket(tk, spec.condition(), spec.run_config(a.model, seed=trial))
                r.update(meta, status="ok")
            except Exception as e:  # recorded as an outcome, never silently dropped
                r = {"task_id": tk.task_id, "model": a.model, **meta, "status": "error", "success": False,
                     "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-3000:]}
            fp = write(r, folder)
            print(f"task {tk.task_id:>3} trial {trial} {r['status']:<5} success={r.get('success')} "
                  f"{r.get('seconds', '-')}s -> {fp}", flush=True)


if __name__ == "__main__":
    main()
