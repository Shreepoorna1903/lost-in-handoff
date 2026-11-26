"""Score saved run records. No model calls, so it can be rerun whenever a metric changes.

    python score_eval.py runs/qwen3-8b
    python score_eval.py runs/colab_a100/baseline        # the Colab pilot

For each condition: runs, errors, success rate with a 95% interval, and mean cost per run. For fault
conditions: how often the planted value was passed on and written, and fault-induced failures against
the clean baseline (clean-D0), paired by task. Writes summary.csv into the folder.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from handoff import metrics as M
from handoff.conditions import BASELINE, CONDITIONS


def pct(k: int, n: int) -> str:
    return f"{k}/{n}" if n else "-"


def main() -> None:
    p = argparse.ArgumentParser(description="Score the saved records in a results folder.")
    p.add_argument("folder")
    folder = Path(p.parse_args().folder)

    records = M.load(folder)
    if not records:
        raise SystemExit(f"No run records under {folder}. Run run_eval.py first.")
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        r["_success"] = M.success(r)
        by_cond[M.condition_name(r)].append(r)

    order = [c for c in CONDITIONS if c in by_cond] + sorted(c for c in by_cond if c not in CONDITIONS)
    rows = []
    for c in order:
        runs = by_cond[c]
        n, k = len(runs), sum(r["_success"] for r in runs)
        lo, hi = M.success_ci(k, n)
        cost = M.cost(runs)
        row = {"condition": c, "runs": n, "errors": sum(r.get("status") == "error" for r in runs),
               "solved": k, "success": round(k / n, 3), "ci_low": round(lo, 3), "ci_high": round(hi, 3),
               **{f"mean_{key}": round(v, 1) for key, v in cost.items() if key != "truncated"},
               "truncated_calls": cost.get("truncated", 0)}
        if CONDITIONS.get(c) and CONDITIONS[c].fault:
            po = [M.passed_on(r) for r in runs if M.passed_on(r) is not None]
            ww = [M.wrong_value_write(r) for r in runs if M.wrong_value_write(r) is not None]
            fi, elig = M.fault_induced_failures(runs, by_cond.get(BASELINE, []))
            row.update(passed_on=pct(sum(po), len(po)), wrong_value_write=pct(sum(ww), len(ww)),
                       fault_induced_failure=pct(fi, elig))
        rows.append(row)

    cols = ["condition", "runs", "errors", "solved", "success", "ci_low", "ci_high", "mean_seconds",
            "mean_tokens", "mean_llm_calls", "mean_tool_calls", "truncated_calls", "passed_on",
            "wrong_value_write", "fault_induced_failure"]
    print(f"{'condition':<14} {'runs':>4} {'err':>3} {'solved':>7} {'95% CI':>13} {'sec':>6} "
          f"{'tokens':>8} {'tools':>5}  fault metrics")
    for r in rows:
        fault = "  ".join(f"{key}={r[key]}" for key in cols[-3:] if key in r)
        print(f"{r['condition']:<14} {r['runs']:>4} {r['errors']:>3} {pct(r['solved'], r['runs']):>7} "
              f"[{r['ci_low']:.2f}, {r['ci_high']:.2f}] {r.get('mean_seconds', 0):>6} "
              f"{r.get('mean_tokens', 0):>8.0f} {r.get('mean_tool_calls', 0):>5}  {fault}")
    with open(folder / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {folder / 'summary.csv'}  (detection and over-blocking: TODO in src/handoff/metrics.py)")


if __name__ == "__main__":
    main()
