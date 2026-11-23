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


def _json_path_mutator(path: list[str | int], value: Any):
    def mutate(out: str) -> str:
        data = json.loads(out)
        _set_path(data, path, value)
        return json.dumps(data)

    return mutate


def corrupted_output(fault_id: str, tool: str, path: list[str | int], wrong_value: Any,
                     occurrence: int = 1) -> ToolFault:
    return ToolFault(fault_id, "corrupted_output", tool, occurrence,
                     _json_path_mutator(path, wrong_value), [str(wrong_value)])


def wrong_identifier(fault_id: str, tool: str, path: list[str | int], other_valid_id: str,
                     occurrence: int = 1) -> ToolFault:
    return ToolFault(fault_id, "wrong_identifier", tool, occurrence,
                     _json_path_mutator(path, other_valid_id), [other_valid_id])


def irrelevant(fault_id: str, tool: str, path: list[str | int], value: Any,
               occurrence: int = 1) -> ToolFault:
    return ToolFault(fault_id, "irrelevant", tool, occurrence,
                     _json_path_mutator(path, value), [str(value)])


def false_claim_text(handoff: str, true_value: str, false_value: str) -> str:
    """Rewrite a handoff so it misreports `true_value` as `false_value`."""
    if true_value not in handoff:
        raise ValueError(f"value {true_value!r} not present in handoff; pick another target")
    return handoff.replace(true_value, false_value)
