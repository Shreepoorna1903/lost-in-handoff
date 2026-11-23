"""Command line entry point.

  python -m handoff.cli list
  python -m handoff.cli run --task 0 --topology pipeline --depth 3 --fmt structured --constraints budget,scope
  python -m handoff.cli baseline --topology single --limit 10      # clean-baseline filter (week 1)
"""

from __future__ import annotations

import json

import typer

from handoff.agents import RunConfig
from handoff.runner import Condition, already_done, run_ticket, save
from handoff.tasks import load_tickets

app = typer.Typer(add_completion=False)


def _cfg(model, think, provenance, verifier, budget) -> RunConfig:
    return RunConfig(model=model, think=think, provenance=provenance, verifier=verifier,
                     tool_call_budget=budget)


def _summary(r: dict) -> str:
    u = r["usage"]
    viol = {k: v for k, v in r["violations"].items() if v}
    return (f"task {r['task_id']:>3} {r['condition']:<34} success={r['success']!s:<5} "
            f"violations={len(viol)} tools={u.get('tool_calls', 0)} llm={u.get('llm_calls', 0)} "
            f"tok={u.get('input_tokens', 0) + u.get('output_tokens', 0)} {r['seconds']}s")


@app.command("list")
def list_tasks(split: str = "base"):
    for t in load_tickets(split):
        print(t.task_id, [a["name"] for a in t.expected_writes])


@app.command()
def run(task: str = "0", topology: str = "pipeline", depth: int = 3, fmt: str = "free_text",
        constraints: str = "", model: str = "qwen3:4b", think: bool = True,
        provenance: bool = False, verifier: bool = False, budget: int = 30, out: str = "runs",
        show: bool = False):
    tk = next(t for t in load_tickets() if t.task_id == task)
    cond = Condition(topology, depth, fmt, [c for c in constraints.split(",") if c])
    r = run_ticket(tk, cond, _cfg(model, think, provenance, verifier, budget))
    fp = save(r, out)
    print(_summary(r))
    if show:
        for h in r["state"].get("handoffs", []):
            print(f"\n--- handoff {h['from']} -> {h['to']} ({h['format']}) ---\n{h['content'][:2000]}")
        print("\n--- tool calls ---")
        for c in r["state"]["tool_log"]:
            print(f"#{c['step']} {c['agent']}: {c['tool']}({json.dumps(c['args'])}) "
                  f"{'ERROR ' + c['output'][:120] if c['error'] else ''}")
        print("\nviolations:", r["violations"], "\nsurvival:", r["survival"])
        print("diff vs gold:", r["diff_vs_gold"][:10])
    print("saved", fp)


@app.command()
def baseline(topology: str = "single", depth: int = 3, fmt: str = "free_text",
             model: str = "qwen3:4b", limit: int = 10, budget: int = 30, out: str = "runs/baseline",
             think: bool = True, resume: bool = True, shard: str = ""):
    """Clean runs (no faults, no added constraints) to find tasks a config solves reliably.

    With --resume (default), tasks that already have a result in `out` for this condition and
    model are skipped, so a sweep killed partway (e.g. a Colab timeout) picks up where it stopped.
    --shard K/N runs only every N-th of the first `limit` tasks, starting at K (1-based), so N
    processes can share one Ollama server (OLLAMA_NUM_PARALLEL=N) without overlapping.
    """
    wins = 0
    tickets = load_tickets()[:limit]
    if shard:
        k, n = (int(x) for x in shard.split("/"))
        tickets = tickets[k - 1::n]
    cond = Condition(topology, depth, fmt)
    for tk in tickets:
        if resume and already_done(out, cond, tk.task_id, model):
            print(f"task {tk.task_id:>3} skipped (result already in {out})", flush=True)
            continue
        try:
            r = run_ticket(tk, cond, _cfg(model, think, False, False, budget))
        except Exception as e:  # e.g. Ollama timeout: record and move on
            print(f"task {tk.task_id:>3} FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
            continue
        save(r, out)
        wins += r["success"]
        print(_summary(r), flush=True)
    print(f"\n{wins}/{len(tickets)} solved")


if __name__ == "__main__":
    app()
