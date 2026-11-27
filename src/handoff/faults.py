"""Fault constructors for Experiment B.

Tool-level faults (applied by ToolEnv at a precise call):
  * corrupted_output  - a value in a tool result is replaced by a wrong one
  * wrong_identifier  - an ID in a tool result is swapped for a different *valid* ID
  * irrelevant        - a field no later decision depends on is changed (control)

Handoff-level fault (applied by forking a LangGraph checkpoint, see runner.fork_with_false_claim):
  * false_claim       - an agent misreports a value that its (correct) tool output contained

Every fault records the exact strings it injected so `evaluate.taint` can trace them.
"""

from __future__ import annotations

import json
from typing import Any

from handoff.env import ToolFault


def _set_path(obj: Any, path: list[str | int], value: Any) -> Any:
    cur = obj
    for p in path[:-1]:
        cur = cur[p]
    old = cur[path[-1]]
    cur[path[-1]] = value
    return old


JsonPath = list  # one JSON path, e.g. ["items", 0, "item_id"]


def _as_paths(paths: JsonPath | list[JsonPath]) -> list[JsonPath]:
    """Accept one path or a list of paths (data/fault_targets.json uses lists)."""
    return paths if paths and isinstance(paths[0], list) else [paths]


def _json_path_mutator(paths: list[JsonPath], value: Any):
    def mutate(out: str) -> str:
        data = json.loads(out)
        for p in paths:
            _set_path(data, p, value)
        return json.dumps(data)

    return mutate


def corrupted_output(fault_id: str, tool: str, paths: JsonPath | list[JsonPath], wrong_value: Any,
                     occurrence: int = 1, args: dict | None = None) -> ToolFault:
    return ToolFault(fault_id, "corrupted_output", tool, occurrence,
                     _json_path_mutator(_as_paths(paths), wrong_value), [str(wrong_value)], dict(args or {}))


def wrong_identifier(fault_id: str, tool: str, paths: JsonPath | list[JsonPath], other_valid_id: str,
                     occurrence: int = 1, args: dict | None = None) -> ToolFault:
    return ToolFault(fault_id, "wrong_identifier", tool, occurrence,
                     _json_path_mutator(_as_paths(paths), other_valid_id), [other_valid_id], dict(args or {}))


def irrelevant(fault_id: str, tool: str, paths: JsonPath | list[JsonPath], value: Any,
               occurrence: int = 1, args: dict | None = None) -> ToolFault:
    return ToolFault(fault_id, "irrelevant", tool, occurrence,
                     _json_path_mutator(_as_paths(paths), value), [str(value)], dict(args or {}))


TOOL_FAULTS = {"F1": corrupted_output, "F2": wrong_identifier, "F3": irrelevant}


def build_faults(task_id: str, fault: str, target: dict) -> list[ToolFault]:
    """The tool-level fault for one entry of data/fault_targets.json (F1-F3).

    The fault fires on the first successful call to the target tool with the target call's
    arguments (e.g. get_order_details for that order), not on the n-th call to the tool: agents
    look things up in a different order and number of times than the gold sequence. Once only, so
    a re-query of the same order returns the true output (what D5 relies on).
    """
    if fault not in TOOL_FAULTS:
        raise ValueError(f"{fault} is not a tool-level fault (F4 is planted by forking a run)")
    return [TOOL_FAULTS[fault](f"{fault}_task{task_id}", target["tool"], target["paths"],
                               target["wrong_value"], occurrence=1, args=target.get("args"))]


def false_claim_text(handoff: str, true_value: str, false_value: str) -> str:
    """Rewrite a handoff so it misreports `true_value` as `false_value`."""
    if true_value not in handoff:
        raise ValueError(f"value {true_value!r} not present in handoff; pick another target")
    return handoff.replace(true_value, false_value)
