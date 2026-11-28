"""Run one ticket under one condition, and fork runs at checkpoints."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from handoff import evaluate as E
from handoff.agents import RunConfig, Usage
from handoff.constraints import make_constraints
from handoff.env import ToolEnv, ToolFault, gold_db_hash
from handoff.faults import false_claim_text
from handoff.tasks import Ticket
from handoff.topologies import RUNTIME, compile_graph


@dataclass
class Condition:
    topology: str = "pipeline"   # single | pipeline | supervisor
    depth: int = 3               # pipeline only
    fmt: str = "free_text"       # free_text | structured | full_context
    constraint_types: list[str] = field(default_factory=list)
    max_rounds: int = 8          # supervisor only

    def label(self) -> str:
        t = self.topology + (f"-d{self.depth}" if self.topology == "pipeline" else "")
        f = "" if self.topology == "single" else f"-{self.fmt}"
        c = ("-" + "+".join(self.constraint_types)) if self.constraint_types else ""
        return t + f + c


def _setup(ticket: Ticket, cond: Condition, cfg: RunConfig, env: ToolEnv):
    ticket.constraints = make_constraints(ticket, cond.constraint_types)
    run_key = uuid.uuid4().hex
    RUNTIME[run_key] = {"env": env, "cfg": cfg, "usage": Usage(), "request": ticket.full_request(),
                        "fmt": cond.fmt, "max_rounds": cond.max_rounds}
    graph = compile_graph(cond.topology, cond.depth, cond.fmt)
    config = {"configurable": {"thread_id": run_key, "run_key": run_key}, "recursion_limit": 50}
    return graph, config, run_key


def _score(ticket: Ticket, cond: Condition, cfg: RunConfig, env: ToolEnv, state: dict,
           seconds: float, faults: list[ToolFault]) -> dict:
    gold_hash, gold_db = gold_db_hash(ticket.expected_actions)
    final_db = env.db_dump()
    fault_vals = [v for f in faults for v in f.injected_values]
    return {
        "task_id": ticket.task_id,
        "condition": cond.label(),
        "cond": asdict(cond),
        "model": cfg.model,
        "think": cfg.think,
        "no_tool_think": cfg.no_tool_think,
        "sampling": {"think": cfg.sampling_think, "no_think": cfg.sampling_no_think, "seed": cfg.seed},
        "provenance": cfg.provenance,
        "provenance_requery": cfg.provenance_requery,
        "verifier": cfg.verifier,
        "verifier_matched": cfg.verifier_matched,
        "executor_full_information": cfg.executor_full_information,
        # fired: the fault changed a tool output (or, for a false claim, the handoff was rewritten).
        # A run whose fault never fired did not test the fault; score_eval reports these separately.
        "faults": [{"id": f.fault_id, "kind": f.kind, "tool": f.tool, "occurrence": f.occurrence,
                    "args": f.args,
                    "fired": f.kind == "false_claim" or any(r.injected == f.fault_id for r in env.log)}
                   for f in faults],
        "success": env.db_hash() == gold_hash,
        "diff_vs_gold": E.collateral_damage(final_db, gold_db),
        "violations": E.constraint_violations(ticket.constraints, state.get("tool_log", [])),
        "survival": E.constraint_survival(ticket.constraints, state.get("handoffs", [])),
        "survival_in_field": E.constraint_in_field(ticket.constraints, state.get("handoffs", [])),
        "taint": E.taint(fault_vals, state) if fault_vals else None,
        "usage": state.get("usage", {}),
        "seconds": round(seconds, 1),
        "state": state,
    }


def run_ticket(ticket: Ticket, cond: Condition, cfg: RunConfig,
               faults: list[ToolFault] | None = None, keep_graph: bool = False) -> dict:
    faults = faults or []
    env = ToolEnv(faults=faults)
    graph, config, run_key = _setup(ticket, cond, cfg, env)
    t0 = time.time()
    state = graph.invoke({"handoffs": [], "transcripts": [], "tool_log": []}, config)
    result = _score(ticket, cond, cfg, env, state, time.time() - t0, faults)
    if keep_graph:
        result["_graph"], result["_config"] = graph, config
    else:
        RUNTIME.pop(run_key, None)
    return result


def fork_with_false_claim(clean: dict, ticket: Ticket, cond: Condition, cfg: RunConfig,
                          hop: int, true_value: str, false_value: str) -> dict:
    """Counterfactual: take a finished clean run, go back to the checkpoint right
    after handoff `hop` was written, make that agent misreport `true_value`,
    and replay the rest of the run from there."""
    graph, config = clean["_graph"], clean["_config"]
    target = None
    for snap in graph.get_state_history(config):  # newest first
        if len(snap.values.get("handoffs", [])) == hop + 1:
            target = snap  # keep going: we want the earliest such checkpoint
    if target is None:
        raise ValueError(f"no checkpoint with handoff #{hop}")
    hs = [dict(h) for h in target.values["handoffs"]]
    hs[hop]["content"] = false_claim_text(hs[hop]["content"], true_value, false_value)

    rt = RUNTIME[config["configurable"]["run_key"]]
    rt["env"] = ToolEnv.restore(target.values.get("tool_log", []))
    rt["usage"] = Usage.from_dict(target.values.get("usage"))
    forked = graph.update_state(target.config, {"handoffs": hs})
    # update_state returns only the checkpoint's thread and id; the nodes also need run_key (to find
    # RUNTIME) and the recursion limit, so carry the original config over and point it at the fork
    new_cfg = {**config, "configurable": {**config["configurable"], **forked["configurable"]}}
    t0 = time.time()
    state = graph.invoke(None, new_cfg)
    fault = ToolFault(f"false_claim_h{hop}", "false_claim", "(handoff)", hop, lambda s: s, [false_value])
    return _score(ticket, cond, cfg, rt["env"], state, time.time() - t0, [fault])


def save(result: dict, out_dir: str | Path = "runs") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in result.items() if not k.startswith("_")}
    fp = out / f"{clean['condition']}__task{clean['task_id']}__{uuid.uuid4().hex[:6]}.json"
    # write then rename, so a session killed mid-write (e.g. Colab timeout) never leaves a
    # half-written file that `already_done` would count
    tmp = fp.with_suffix(".tmp")
    tmp.write_text(json.dumps(clean, indent=2, default=str))
    tmp.rename(fp)
    return fp


def already_done(out_dir: str | Path, cond: Condition, task_id: str, model: str) -> bool:
    """True if `out_dir` already holds a result for this condition, task and model."""
    for fp in Path(out_dir).glob(f"{cond.label()}__task{task_id}__*.json"):
        try:
            if json.loads(fp.read_text()).get("model") == model:
                return True
        except (json.JSONDecodeError, OSError):
            continue
    return False
