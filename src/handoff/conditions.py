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

from dataclasses import dataclass, field

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
    "D5": {"label": "provenance tags, re-query", "config": {}, "built": False, "owner": "Puneet"},
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
    """Fault constructors exist, but run_eval cannot plant them until per-task targets exist
    (`fault_targets` below). Flip to True once the fault-target script is done."""
    return False


def fault_targets(task_id: str, fault: str) -> dict:
    """Where to plant `fault` in this task: tool, occurrence, JSON path and the wrong value
    (for F4: handoff hop and true/false value). Picked from the task's gold actions.

    TODO (Shreepoorna + Mandar): write the fault-target script, save its output (e.g.
    data/fault_targets.json), load it here, and set fault_injection_built to True.
    """
    raise NotImplementedError("fault targets are not picked yet (fault-target script)")


CONDITIONS: dict[str, Spec] = _make()
BASELINE = "clean-D0"  # fault conditions are compared with this, task by task

# Records saved before this file existed carry only the runner's label. Clean records map as below;
# score_eval uses this so the Colab pilot in runs/colab_a100/ scores without re-running anything.
LEGACY_LABELS = {"single": "single-clean", "pipeline-d3-free_text": "clean-D0"}
