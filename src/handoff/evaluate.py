"""Ground truth and measurement (proposal: 'Ground truth and measurement')."""

from __future__ import annotations

import json

from deepdiff import DeepDiff

from handoff.env import gold_db_hash


def task_success(final_hash: str, ticket) -> bool:
    """tau2's DB check: final DB equals the DB after replaying the gold actions."""
    return final_hash == gold_db_hash(ticket.expected_actions)[0]


def collateral_damage(final_db: dict, reference_db: dict) -> list[str]:
    """Paths where the final DB differs from a reference (gold, or the clean run)."""
    diff = DeepDiff(reference_db, final_db, ignore_order=True, view="tree")
    return sorted({str(node.path()) for nodes in diff.values() for node in nodes})


def constraint_violations(constraints, tool_log: list[dict]) -> dict[str, list[str]]:
    return {c.cid: c.violations(tool_log) for c in constraints}


def constraint_survival(constraints, handoffs: list[dict]) -> dict[str, list[bool]]:
    """Per hop, did each constraint's key terms survive anywhere in the handoff?

    Rule-based key-term match over the whole handoff. Free-text hops should
    additionally be labelled by the LLM judge -- see TODO in README. Use
    `constraint_in_field` for whether a structured handoff filed it correctly.
    """
    return {c.cid: [any(t.lower() in h["content"].lower() for t in c.key_terms) for h in handoffs]
            for c in constraints}


def constraint_in_field(constraints, handoffs: list[dict]) -> dict[str, list[bool | None]]:
    """Structured hops only: is the constraint in the `constraints` field? None for other formats."""
    out = {}
    for c in constraints:
        hops = []
        for h in handoffs:
            if h["format"] != "structured":
                hops.append(None)
                continue
            try:
                text = " ".join(json.loads(h["content"]).get("constraints", [])).lower()
            except json.JSONDecodeError:
                text = ""
            hops.append(any(t.lower() in text for t in c.key_terms))
        out[c.cid] = hops
    return out


def taint(fault_values: list[str], state: dict) -> dict:
    """Trace injected values through handoffs, tool args and write actions."""
    def has(s: str) -> bool:
        return any(v in s for v in fault_values)

    log = state.get("tool_log", [])
    inj_step = next((r["step"] for r in log if r.get("injected")), -1)
    later = [r for r in log if r["step"] > inj_step]
    return {
        "in_handoffs": [has(h["content"]) for h in state.get("handoffs", [])],
        "in_tool_args": any(has(json.dumps(r["args"])) for r in later),
        "reached_write": any(r["is_write"] and has(json.dumps(r["args"])) for r in later),
        "in_final_answer": has(state.get("final_answer", "")),
    }


def requeried(tool_log: list[dict], tool: str, after_step: int) -> bool:
    """Detection signal: the corrupted fact was looked up again after injection."""
    return sum(1 for r in tool_log if r["tool"] == tool and r["step"] > after_step) > 0
