"""Every experimental condition, by name. Adding a condition is one entry here, not a code change elsewhere.

A condition is a topology, at most one planted fault (RQ1) and one defense (RQ2, RQ3). Everything else is
held fixed across conditions: the model, sampling settings and seed, role prompts, tools, tool-call budget,
handoff format (free text) and the task set. `run_eval.py` and `score_eval.py` take conditions by these names.

Names:
  single-clean       single agent, no fault, no defense (reference)
  clean-D0           3-agent pipeline, no fault, no defense (the baseline fault runs are compared with)
  clean-D1 ... D5    no fault, with a defense: measures over-blocking (correct writes a defense stops)
  F1-D0 ... F4-D5    fault x defense, all on the 3-agent pipeline

`built=False` marks a part that is not implemented yet; `run_eval.py` refuses to run it and says who owns it.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from handoff.agents import RunConfig
from handoff.runner import Condition

# Planted faults (RQ1). The kind names match handoff.faults.
FAULTS = {
    "F1": "corrupted_output",   # a value in a tool result is replaced by a wrong one
    "F2": "wrong_identifier",   # an ID in a tool result is swapped for a different valid ID
    "F3": "irrelevant",         # a field no later decision depends on is changed (control)
    "F4": "false_claim",        # a handoff misreports a value (checkpoint fork)
}

# Defenses (RQ2: verifier and information access; RQ3: provenance). Each maps to RunConfig fields.
DEFENSES = {
    "D0": {"label": "no defense", "config": {}, "built": True},
    "D1": {"label": "verifier, matched information", "config": {}, "built": False, "owner": "Harsha"},
    "D2": {"label": "verifier, full information", "config": {"verifier": True}, "built": True},
    "D3": {"label": "executor with full information, no verifier", "config": {}, "built": False,
           "owner": "Harsha"},
    "D4": {"label": "provenance tags, trace", "config": {"provenance": True}, "built": True},
    "D5": {"label": "provenance tags, re-query", "config": {"provenance": True, "provenance_requery": True},
           "built": True},
}

FIXED = {"depth": 3, "fmt": "free_text"}  # pipeline shape and handoff format, same in every condition


@dataclass
class Spec:
    name: str
    topology: str               # single | pipeline
    fault: str | None = None    # F1..F4, or None for clean runs
    defense: str = "D0"
    config: dict = field(default_factory=dict)  # RunConfig overrides for the defense
    built: bool = True
    owner: str = ""             # who is building it, while built is False

    def condition(self) -> Condition:
        return Condition(self.topology, FIXED["depth"], FIXED["fmt"])

    def run_config(self, model: str, seed: int = 0) -> RunConfig:
        return RunConfig(model=model, seed=seed, **self.config)


def _make() -> dict[str, Spec]:
    out = {"single-clean": Spec("single-clean", "single")}
    for d, info in DEFENSES.items():
        out[f"clean-{d}"] = Spec(f"clean-{d}", "pipeline", None, d, dict(info["config"]), info["built"],
                                 info.get("owner", ""))
    for f in FAULTS:
        for d, info in DEFENSES.items():
            built = info["built"] and fault_injection_built(f)
            owner = info.get("owner") or ("" if fault_injection_built(f) else "Shreepoorna + Mandar")
            out[f"{f}-{d}"] = Spec(f"{f}-{d}", "pipeline", f, d, dict(info["config"]), built, owner)
    return out


def fault_injection_built(fault: str) -> bool:
    """run_eval plants F1-F3 from the per-task targets (`fault_targets` below) and forks for F4,
    so a fault condition can run once data/fault_targets.json exists."""
    return fault in FAULTS and FAULT_TARGETS_FILE.exists()


FAULT_TARGETS_FILE = Path(__file__).resolve().parents[2] / "data" / "fault_targets.json"


@lru_cache(maxsize=1)
def _load_fault_targets() -> dict:
    if not FAULT_TARGETS_FILE.exists():
        raise FileNotFoundError(f"{FAULT_TARGETS_FILE} is missing: run scripts/pick_fault_targets.py")
    return json.loads(FAULT_TARGETS_FILE.read_text())


def fault_targets(task_id: str, fault: str) -> dict:
    """Where to plant `fault` in this task, from data/fault_targets.json (scripts/pick_fault_targets.py).

    F1-F3: tool, occurrence, paths (a list of paths), true_value, wrong_value (and args, the targeted gold call's arguments).
    F4: hop, true_value, false_value.
    """
    if fault not in FAULTS:
        raise KeyError(f"unknown fault {fault!r}; expected one of {sorted(FAULTS)}")
    targets = _load_fault_targets()
    if str(task_id) not in targets:
        raise KeyError(f"task {task_id} has no fault targets (tasks with targets: "
                       f"{', '.join(targets)}); rerun scripts/pick_fault_targets.py")
    entry = targets[str(task_id)]
    if fault not in entry:
        raise KeyError(f"task {task_id} has no {fault} target in {FAULT_TARGETS_FILE.name}")
    return copy.deepcopy(entry[fault])  # callers may edit it; the cache must stay intact


CONDITIONS: dict[str, Spec] = _make()
BASELINE = "clean-D0"  # the pipeline baseline; a fault condition is compared with clean-<its defense>

# Records saved before this file existed carry only the runner's label. Clean records map as below;
# score_eval uses this so the Colab pilot in runs/colab_a100/ scores without re-running anything.
LEGACY_LABELS = {"single": "single-clean", "pipeline-d3-free_text": "clean-D0"}
