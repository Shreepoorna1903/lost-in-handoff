"""Score saved run records. No model calls, so it can be rerun whenever a metric changes.

    python score_eval.py runs/qwen3-8b
    python score_eval.py runs/colab_a100/pilot/qwen3-8b  # the FP3 pilot

For each condition: runs, errors, success rate with a 95% interval, and mean cost per run. For fault
conditions: how often the fault fired, how often the planted value was passed on and written, and
fault-induced failures against the clean run of the same defense (F2-D4 against clean-D4), paired
by task, leaving out runs where the fault never fired. F4 runs where the
researcher never wrote the target value (status "no_target") had no fault to plant: they are counted
in the no_target column and left out of every other number. Writes summary.csv into the folder.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from handoff import metrics as M
from handoff.conditions import CONDITIONS


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
    no_target: dict[str, int] = defaultdict(int)
    for r in records:
        if r.get("status") == "no_target":
            no_target[M.condition_name(r)] += 1
            continue
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
        ob = [M.over_blocking(r) for r in runs if M.over_blocking(r) is not None]
        if ob:  # verifier conditions: blocks of correct writes, and runs where one was never made
            lost = [M.lost_correct_write(r) for r in runs if M.lost_correct_write(r) is not None]
            row.update(verifier_blocks=sum(r.get("usage", {}).get("verifier_blocks", 0) for r in runs),
                       correct_writes_blocked=sum(ob), lost_correct_write=pct(sum(lost), len(lost)))
        looks = [M.provenance_lookups(r) for r in runs if M.provenance_lookups(r) is not None]
        if looks:  # D4/D5: was the lookup tool used at all?
            row.update(lookup_runs=pct(sum(n > 0 for n in looks), len(looks)),
                       mean_lookups=round(sum(looks) / len(looks), 2))
        if CONDITIONS.get(c) and CONDITIONS[c].fault:
            po = [M.passed_on(r) for r in runs if M.passed_on(r) is not None]
            ww = [M.wrong_value_write(r) for r in runs if M.wrong_value_write(r) is not None]
            base = f"clean-{CONDITIONS[c].defense}"  # same defense, no fault
            fi, elig = M.fault_induced_failures(runs, by_cond.get(base, []))
            fired = [M.fault_fired(r) for r in runs if M.fault_fired(r) is not None]
            row.update(no_target=no_target.get(c, 0), fault_fired=pct(sum(fired), len(fired)),
                       passed_on=pct(sum(po), len(po)), wrong_value_write=pct(sum(ww), len(ww)),
                       fault_induced_failure=pct(fi, elig))
            det = [M.detected(r) for r in runs if M.detected(r) is not None]
            row["detected"] = pct(sum(det), len(det))
            dnr = [M.detected_without_requery(r) for r in runs if M.detected_without_requery(r) is not None]
            row["detected_without_requery"] = pct(sum(dnr), len(dnr))
            vbp = [M.verifier_blocked_planted(r) for r in runs if M.verifier_blocked_planted(r) is not None]
            if vbp and any(M.over_blocking(r) is not None for r in runs):  # verifier conditions only
                row["verifier_blocked_planted"] = pct(sum(vbp), len(vbp))
            sigs = [s for s in (M.detection_signals(r) for r in runs) if s is not None]
            for key in M.SIGNALS:  # each signal over the runs where it applies
                vals = [s[key] for s in sigs if s[key] is not None]
                row[key] = pct(sum(vals), len(vals))
        rows.append(row)

    cols = ["condition", "runs", "errors", "solved", "success", "ci_low", "ci_high", "mean_seconds",
            "mean_tokens", "mean_llm_calls", "mean_tool_calls", "truncated_calls", "verifier_blocks",
            "correct_writes_blocked", "lost_correct_write", "lookup_runs",
            "mean_lookups", "no_target",
            "fault_fired", "passed_on", "wrong_value_write", "fault_induced_failure", "detected",
            "detected_without_requery", *M.SIGNALS, "verifier_blocked_planted"]
    fault_cols = ["verifier_blocks", "correct_writes_blocked", "lost_correct_write", "lookup_runs", "fault_fired", "passed_on", "wrong_value_write", "fault_induced_failure", "detected"]
    print(f"{'condition':<14} {'runs':>4} {'err':>3} {'solved':>7} {'95% CI':>13} {'sec':>6} "
          f"{'tokens':>8} {'tools':>5}  fault metrics")
    for r in rows:
        fault = "  ".join(f"{key}={r[key]}" for key in fault_cols if key in r)
        print(f"{r['condition']:<14} {r['runs']:>4} {r['errors']:>3} {pct(r['solved'], r['runs']):>7} "
              f"[{r['ci_low']:.2f}, {r['ci_high']:.2f}] {r.get('mean_seconds', 0):>6} "
              f"{r.get('mean_tokens', 0):>8.0f} {r.get('mean_tool_calls', 0):>5}  {fault}")
    with open(folder / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {folder / 'summary.csv'}")


if __name__ == "__main__":
    main()
