"""Tool-wrapper layer between agents and the tau2-bench retail environment.

Every tool call goes through `ToolEnv.call`, which
  * enforces per-agent tool access,
  * logs the call (agent, tool, args, output, error, whether it wrote state),
  * applies any scheduled fault (the injection hook for RQ2/RQ3).

The environment database lives outside the LangGraph state, so to fork a run
from a checkpoint we rebuild the DB by replaying the logged write calls
(`ToolEnv.restore`). retail tools are deterministic, so this is exact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from tau2.domains.retail.environment import get_environment
from tau2.environment.toolkit import ToolType

READ_TOOLS_ACCOUNT = ["find_user_id_by_email", "find_user_id_by_name_zip", "get_user_details"]
READ_TOOLS_CATALOG = [
    "get_order_details",
    "get_product_details",
    "get_item_details",
    "list_all_product_types",
]
READ_TOOLS = READ_TOOLS_ACCOUNT + READ_TOOLS_CATALOG
GENERIC_TOOLS = ["calculate"]
WRITE_TOOLS = [
    "cancel_pending_order",
    "exchange_delivered_order_items",
    "modify_pending_order_address",
    "modify_pending_order_items",
    "modify_pending_order_payment",
    "modify_user_address",
    "return_delivered_order_items",
    "transfer_to_human_agents",
]
# Not a tau2 tool: a no-op we add so "approval rule" constraints can be satisfied
# without changing the task's correct final DB state.
APPROVAL_TOOL = "request_manager_approval"
APPROVAL_SCHEMA = {
    "type": "function",
    "function": {
        "name": APPROVAL_TOOL,
        "description": "Ask a store manager to approve an action before performing it. "
        "Returns 'approved' or 'denied'.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "Order the action applies to, e.g. '#W0000000'."},
                "action_summary": {"type": "string", "description": "What you intend to do and why."},
            },
            "required": ["order_id", "action_summary"],
        },
    },
}
ALL_TOOLS = READ_TOOLS + GENERIC_TOOLS + WRITE_TOOLS + [APPROVAL_TOOL]


@dataclass
class ToolCallRecord:
    step: int
    agent: str
    tool: str
    args: dict[str, Any]
    output: str
    error: bool
    is_write: bool
    injected: str | None = None  # fault id if this output was tampered with
    coerced: bool = False  # numeric args were converted to the strings the schema expects

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class ToolFault:
    """A fault applied to a tool output.

    The target call is the `occurrence`-th successful call to `tool` whose arguments include every
    key/value in `args` (all calls to `tool` when `args` is empty). `mutate` rewrites the output
    string. `fault_id` must be a unique marker so the injected value can be taint-tracked downstream.
    A fault fires at most once: a later call with the same arguments returns the true output.
    """

    fault_id: str
    kind: str  # corrupted_output | wrong_identifier | irrelevant
    tool: str
    occurrence: int  # 1 = first matching call, 2 = second, ...
    mutate: Callable[[str], str]
    injected_values: list[str] = field(default_factory=list)
    args: dict[str, Any] = field(default_factory=dict)  # e.g. {"order_id": "#W2378156"}

    def matches(self, tool: str, args: dict[str, Any]) -> bool:
        return tool == self.tool and all(args.get(k) == v for k, v in self.args.items())


def _coerce_to_schema(value: Any, schema: dict) -> tuple[Any, bool]:
    """Convert numbers to strings where the JSON schema expects a string.

    Models often emit IDs such as 8349118980 as JSON numbers; tau2 compares IDs as
    strings, so without this an otherwise correct call fails with "not found".
    """
    if schema.get("type") == "string" and isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value), True
    if schema.get("type") == "array" and isinstance(value, list):
        out = [_coerce_to_schema(v, schema.get("items", {})) for v in value]
        return [v for v, _ in out], any(c for _, c in out)
    return value, False


class ToolEnv:
    def __init__(self, faults: list[ToolFault] | None = None):
        self.env = get_environment()
        self.log: list[ToolCallRecord] = []
        self.faults = list(faults or [])
        self._tools = self.env.get_tools()
        self._by_name = {t.name: t for t in self._tools}

    # --- schemas -----------------------------------------------------------
    def schemas(self, names: list[str]) -> list[dict]:
        return [APPROVAL_SCHEMA if n == APPROVAL_TOOL else self._by_name[n].openai_schema
                for n in names]

    def policy(self) -> str:
        return self.env.get_policy()

    def is_write(self, tool: str) -> bool:
        if tool not in self._by_name:
            return False
        return self.env.tools.tool_type(tool) == ToolType.WRITE

    # --- calls -------------------------------------------------------------
    def call(self, agent: str, tool: str, args: dict[str, Any], allowed: list[str]) -> str:
        if tool not in allowed:
            out, err = f"Error: agent '{agent}' does not have access to tool '{tool}'.", True
            rec = ToolCallRecord(len(self.log), agent, tool, args, out, err, False)
            self.log.append(rec)
            return out
        if tool == APPROVAL_TOOL:
            rec = ToolCallRecord(len(self.log), agent, tool, args, "approved", False, False)
            self.log.append(rec)
            return rec.output
        props = self._by_name[tool].openai_schema["function"]["parameters"].get("properties", {})
        coerced = False
        for k in list(args):
            if k in props:
                args[k], c = _coerce_to_schema(args[k], props[k])
                coerced |= c
        try:
            resp = self.env.make_tool_call(tool, requestor="assistant", **args)
            self.env.sync_tools()
            out, err = self.env.to_json_str(resp), False
        except Exception as e:  # tau2 tools raise ValueError on bad input
            out, err = f"Error: {e}", True
        rec = ToolCallRecord(len(self.log), agent, tool, args, out, err, self.is_write(tool),
                             coerced=coerced)
        if not err:
            self._maybe_inject(rec)
        self.log.append(rec)
        return rec.output

    def _maybe_inject(self, rec: ToolCallRecord) -> None:
        for f in self.faults:
            if not f.matches(rec.tool, rec.args):
                continue
            n = sum(1 for r in self.log if not r.error and f.matches(r.tool, r.args)) + 1
            if f.occurrence == n:
                rec.output = f.mutate(rec.output)
                rec.injected = f.fault_id

    # --- state -------------------------------------------------------------
    def db_hash(self) -> str:
        return self.env.get_db_hash()

    def db_dump(self) -> dict:
        return json.loads(self.env.tools.db.model_dump_json())

    @classmethod
    def restore(cls, log: list[dict], faults: list[ToolFault] | None = None) -> "ToolEnv":
        """Fresh env with the successful write calls of `log` replayed."""
        te = cls(faults=faults)
        for r in log:
            if r["is_write"] and not r["error"]:
                te.env.make_tool_call(r["tool"], requestor="assistant", **r["args"])
                te.env.sync_tools()
        te.log = [ToolCallRecord(**r) for r in log]
        return te


def gold_db_hash(expected_actions: list[dict]) -> tuple[str, dict]:
    """Replay a task's expected actions on a fresh DB; return hash and dump."""
    te = ToolEnv()
    for a in expected_actions:
        try:
            te.env.make_tool_call(a["name"], requestor="assistant", **a["arguments"])
            te.env.sync_tools()
        except Exception:
            pass
    return te.db_hash(), te.db_dump()
