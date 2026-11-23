"""Handoff formats (Experiment A): free_text, structured, full_context."""

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, Field

HandoffFormat = Literal["free_text", "structured", "full_context"]
FORMATS: tuple[str, ...] = ("free_text", "structured", "full_context")


class Fact(BaseModel):
    statement: str = Field(description="One fact as '<label>: <actual value>', with every ID (user, "
                           "order, item, product, payment method), name and price copied exactly "
                           "from the log. Never a label without its value.")
    source_call: int | None = Field(
        default=None, description="Number of the tool call this fact came from, if any.")


class HandoffSpec(BaseModel):
    goal: str = Field(description="What the customer wants done, end to end.")
    facts: list[Fact] = Field(description="Verified facts gathered so far (IDs, prices, statuses).")
    constraints: list[str] = Field(
        description="Every rule, limit, prohibition or approval requirement the customer or store stated.")
    next_steps: list[str] = Field(description="What the next agent should do.")


# Shared by free_text and structured so the two formats get identical content
# requirements; only the *form* of the handoff differs (RQ1).
CONTENT_RULES = (
    "Your handoff must include (a) every rule, limit, prohibition or approval requirement stated in "
    "the ticket, word for word, (b) the customer's identity details (name, zip code, email, user ID) "
    "exactly as given, and (c) every ID (user, order, item, product, payment method) and price exactly "
    "as it appeared in the ticket or tool results. Never shorten, round or invent IDs; if you did not "
    "look something up, say so."
)

FREE_TEXT_INSTRUCTION = (
    "When you are done, reply with a concise natural-language summary for the next agent: "
    "what the customer wants, what you found, and what remains to be done. "
    "The next agent cannot see the original ticket or your tool results -- only your summary. "
    + CONTENT_RULES
)
STRUCTURED_INSTRUCTION = (
    "When you are done, reply with a short note saying you are finished. "
    "You will then be asked to fill in a structured handoff for the next agent, "
    "who cannot see the original ticket or your tool results. " + CONTENT_RULES
)
STRUCTURED_FILL_SYSTEM = (
    "You convert one agent's work log into a JSON handoff for the next agent in a retail "
    "customer-service team. Fill the fields from the WORK LOG only:\n"
    "- goal: what the customer wants done.\n"
    "- Every fact is written as '<label>: <actual value>', with the value copied exactly from the "
    "log. A label alone, without its value, is not a fact. Include the customer's name, zip code, "
    "email and user ID as facts when the log gives them.\n"
    "- facts: every ID (user, order, item, product, payment method) and price, copied exactly "
    "from the log, one fact per entry.\n"
    "- constraints: every rule, limit, prohibition or approval requirement that the customer's "
    "ticket or an earlier handoff in the log stated, word for word. Not store policy, not these "
    "instructions.\n"
    "- next_steps: what the next agent still has to do.\n"
    "Never shorten, round or invent IDs.\n"
    "The log has two parts. INCOMING is what this agent was told; it may be wrong or incomplete. "
    "TOOL RESULTS are authoritative. When they disagree, use the tool results.\n"
    "Facts already listed in an INCOMING handoff are carried forward automatically, so in `facts` "
    "list what this agent's TOOL RESULTS found or corrected (IDs, prices, statuses, payment "
    "methods). Constraints are NOT carried automatically: restate every rule, limit, prohibition "
    "or approval requirement from INCOMING in `constraints`, word for word."
)
FULL_CONTEXT_INSTRUCTION = (
    "When you are done, reply with a short note for the next agent. "
    "The next agent will receive the complete history of the work so far."
)
PROVENANCE_INSTRUCTION = (
    " Tool results are labeled [call #N]. Tag every fact you pass on with its source, "
    "e.g. 'item 123 costs $50 [call #4]'."
)


def instruction(fmt: str, provenance: bool) -> str:
    base = {"free_text": FREE_TEXT_INSTRUCTION, "structured": STRUCTURED_INSTRUCTION,
            "full_context": FULL_CONTEXT_INSTRUCTION}[fmt]
    return base + (PROVENANCE_INSTRUCTION if provenance else "")


_HAS_VALUE = re.compile(r"[0-9#@$]")


def label_only_facts(spec: HandoffSpec) -> list[str]:
    """Facts that are bare labels such as "Order ID" or "Customer identity details".

    Pipeline task 0 (8B): the intake's handoff listed facts as bare labels, the order number was
    lost, and the researcher fell back on tau2's docstring example ID #W0000000.
    """
    return [f.statement for f in spec.facts
            if len(f.statement.split()) <= 4 and not _HAS_VALUE.search(f.statement)]


def _norm(statement: str) -> str:
    return " ".join(statement.lower().split())


def carry_forward_facts(incoming: str | None, spec: HandoffSpec) -> tuple[HandoffSpec, int]:
    """Decision 19 (c): prepend the incoming structured handoff's facts to the agent's new ones.

    Only facts are merged in code. `goal`, `constraints` and `next_steps` stay exactly as the model
    wrote them, so constraint survival is still the model's doing (RQ1). Returns the merged spec
    and how many facts were carried over.
    """
    if not incoming:
        return spec, 0
    try:
        old = [Fact(**f) for f in json.loads(incoming).get("facts", [])]
    except (json.JSONDecodeError, TypeError, ValueError):
        return spec, 0
    seen = {_norm(f.statement) for f in old}
    new = [f for f in spec.facts if _norm(f.statement) not in seen]
    return spec.model_copy(update={"facts": old + new}), len(old)


def render_structured(spec: HandoffSpec) -> str:
    return json.dumps(spec.model_dump(), indent=2)


def render_work_log(agent: str, messages: list[dict]) -> str:
    """Extraction input: the incoming message kept apart from this agent's own work."""
    msgs = [m for m in messages if m["role"] != "system"]
    incoming = msgs[0]["content"] if msgs and msgs[0]["role"] == "user" else ""
    return (f"INCOMING (may be wrong or incomplete):\n{incoming}\n\n"
            f"TOOL RESULTS AND WORK BY {agent} (authoritative):\n"
            + render_transcript(agent, msgs[1:] if incoming else msgs))


def render_transcript(agent: str, messages: list[dict]) -> str:
    lines = [f"=== {agent} ==="]
    for m in messages:
        if m["role"] == "system":
            continue
        if m["role"] == "tool":
            lines.append(f"[tool result] {m['content']}")
        elif m.get("tool_calls"):
            for tc in m["tool_calls"]:
                lines.append(f"[{agent} calls {tc['name']}] {json.dumps(tc['args'])}")
        elif m["content"]:
            lines.append(f"[{m['role']}] {m['content']}")
    return "\n".join(lines)
