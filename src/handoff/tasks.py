"""tau2 retail tasks turned into single-shot customer tickets.

tau2 normally runs an LLM user simulator over many turns. To keep runs
reproducible and to keep the user-facing agent as the *only* holder of the
user's constraints, we give it the whole request up front as a ticket, and
state that the customer has already confirmed the requested actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tau2.domains.retail.environment import get_tasks

from handoff.env import WRITE_TOOLS, ToolEnv, gold_db_hash


@dataclass
class Ticket:
    task_id: str
    request: str
    expected_actions: list[dict]
    constraints: list = field(default_factory=list)  # list[constraints.Constraint]

    @property
    def expected_writes(self) -> list[dict]:
        return [a for a in self.expected_actions if a["name"] in WRITE_TOOLS]

    def full_request(self) -> str:
        if not self.constraints:
            return self.request
        rules = "\n".join(f"- {c.text}" for c in self.constraints)
        return f"{self.request}\n\nAdditional rules from the customer/store for this ticket:\n{rules}"


def _ticket_text(task) -> str:
    ins = task.user_scenario.instructions
    parts = [
        "Customer ticket (the customer is not available for follow-up questions).",
        f"Identity: {ins.known_info}",
        f"Request: {ins.reason_for_call}",
    ]
    if ins.task_instructions:
        parts.append(f"Customer preferences: {ins.task_instructions}")
    if ins.unknown_info:
        parts.append(f"Note: {ins.unknown_info}")
    parts.append(
        "The customer has already confirmed (yes) every action described in this request; "
        "no further confirmation is needed."
    )
    return "\n".join(parts)


def depends_on_confirmation_dialogue(task) -> bool:
    """True if the customer's script reacts to the agent's confirmation request.

    e.g. task 5: "If the agent asks for confirmation, only exchange the desk lamp ... return
    the water bottle instead." The gold outcome only arises through that exchange, which a
    pre-confirmed single-shot ticket cannot reproduce, so these tasks are excluded (README).
    """
    ins = task.user_scenario.instructions
    text = " ".join(filter(None, [ins.reason_for_call, ins.task_instructions]))
    return "confirm" in text.lower()


def is_noop(ticket: Ticket, initial_hash: str) -> bool:
    """True if replaying the gold actions leaves the database unchanged.

    Such a task passes even when the agents do nothing (pilot: the pipeline's only two passes were
    no-op tasks), so it cannot show whether a fault caused a failure. In base: 10, 12, 50, 105.
    """
    return gold_db_hash(ticket.expected_actions)[0] == initial_hash


def load_tickets(split: str = "base", require_write: bool = True,
                 exclude_dialogue: bool = True, exclude_noop: bool = True) -> list[Ticket]:
    out = []
    initial_hash = ToolEnv().db_hash() if exclude_noop else ""
    for t in get_tasks(split):
        if exclude_dialogue and depends_on_confirmation_dialogue(t):
            continue
        actions = [a.model_dump() for a in (t.evaluation_criteria.actions or [])]
        tk = Ticket(task_id=t.id, request=_ticket_text(t), expected_actions=actions)
        if require_write and not tk.expected_writes:
            continue
        if exclude_noop and is_noop(tk, initial_hash):
            continue
        out.append(tk)
    return out
