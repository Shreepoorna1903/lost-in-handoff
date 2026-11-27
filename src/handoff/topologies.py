"""Team topologies as LangGraph graphs: single agent, pipeline (depth 2-4), supervisor.

Graph state is plain JSON (handoffs, transcripts, tool log, usage) so every
step is checkpointed and can be forked. Non-serialisable runtime objects (the
ToolEnv, the run config, the ticket text) live in RUNTIME, keyed by `run_key`
in the LangGraph config.
"""

from __future__ import annotations

import json
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from handoff import handoffs as H
from handoff.agents import RunConfig, Usage, run_agent
from handoff.env import (
    ALL_TOOLS, APPROVAL_TOOL, GENERIC_TOOLS, READ_TOOLS, READ_TOOLS_ACCOUNT,
    READ_TOOLS_CATALOG, WRITE_TOOLS, ToolEnv,
)

RUNTIME: dict[str, dict[str, Any]] = {}

EXECUTOR_TOOLS = WRITE_TOOLS + [APPROVAL_TOOL] + GENERIC_TOOLS


class TeamState(TypedDict, total=False):
    handoffs: list[dict]      # {from, to, format, content}
    transcripts: list[dict]   # {agent, messages}
    tool_log: list[dict]
    usage: dict
    final_answer: str
    sup_round: int
    sup_next: dict | None     # supervisor's pending delegation


def _rt(config) -> dict[str, Any]:
    return RUNTIME[config["configurable"]["run_key"]]


ROLE_DESC = {
    "single": "You are a retail customer-service agent. Resolve the customer's ticket end to end.",
    "intake": "You are the intake agent. You read the customer's ticket and brief the team. You have no "
              "tools, so pass on only what the ticket itself says. Never state item IDs, product IDs, "
              "user IDs, payment methods, prices or order statuses that are not written in the ticket, "
              "and never claim that anything has been looked up or verified.",
    "intake_research": "You are the intake agent. Read the customer's ticket, authenticate the "
                       "customer, then look up every order, product and item detail needed, and work "
                       "out the exact item IDs, new item IDs, payment method ID and prices the executor "
                       "will need.",
    "researcher": "You are the research specialist. First authenticate the customer (policy requires "
                  "it), then look up every order, product and item detail needed, and work out the "
                  "exact item IDs, new item IDs, payment method ID and prices the executor will need. "
                  "You only look things up; the executor makes every change.",
    "account": "You are the account specialist. Authenticate the customer and look up their "
               "profile, orders and payment methods.",
    "catalog": "You are the catalog specialist. Look up order, product and item details and "
               "work out the exact item IDs and prices needed.",
    "executor": "You are the executor. You alone can change the store database. Carry out the "
                "requested changes using exact IDs, then report what you did. Use only IDs that "
                "appear in what you were given; if a needed ID is missing, report that instead of "
                "guessing.",
}

PIPELINES: dict[int, list[tuple[str, list[str]]]] = {
    2: [("intake_research", READ_TOOLS + GENERIC_TOOLS), ("executor", EXECUTOR_TOOLS)],
    3: [("intake", []), ("researcher", READ_TOOLS + GENERIC_TOOLS), ("executor", EXECUTOR_TOOLS)],
    4: [("intake", []), ("account", READ_TOOLS_ACCOUNT), ("catalog", READ_TOOLS_CATALOG + GENERIC_TOOLS),
        ("executor", EXECUTOR_TOOLS)],
}


# The policy requires asking the customer for confirmation before any write, but tickets are
# single-shot and pre-confirmed. Without this note agents stop to ask and never act (tasks 3, 4).
TICKET_MODE_NOTE = (
    "This is a written ticket, not a live chat: nobody will answer questions. The customer's explicit "
    "confirmation for every action in the ticket is already given in writing, which satisfies the "
    "policy's confirmation rule. Do not ask for confirmation; carry out every requested action, then "
    "report what was done."
)
# Roles without write tools got TICKET_MODE_NOTE too, read "carry out every requested action, then
# report what was done" literally, and wrote that the changes were made; the executor believed it and
# did nothing (pilot tasks 0, 2, 11, 14). They get this note instead. The executor and single agent
# keep TICKET_MODE_NOTE unchanged.
READ_ONLY_NOTE = (
    "This is a written ticket, not a live chat: nobody will answer questions. The customer has already "
    "confirmed every action in the ticket in writing, so do not ask for confirmation. You cannot change "
    "the store database: you have no write tools, and nothing you do cancels, modifies, returns, "
    "exchanges or refunds anything. The executor, a later agent, makes every change. Never write that "
    "an action has been done, processed, initiated or completed. Describe each change as an action the "
    "executor still has to take, with the exact IDs it needs."
)
# The intake has no tools at all, so READ_ONLY_NOTE's "with the exact IDs it needs" would ask it for
# IDs it cannot look up and invite invented ones. It gets this note instead: same ticket-mode and
# no-action rules, but IDs only as written in the ticket.
INTAKE_NOTE = (
    "This is a written ticket, not a live chat: nobody will answer questions. The customer has already "
    "confirmed every action in the ticket in writing, so do not ask for confirmation. You cannot change "
    "the store database or look anything up: you have no tools. Later agents look up the details and "
    "make every change. Never write that an action has been done, processed, initiated, completed or "
    "verified. Describe each requested change as an action still to be taken. Copy any ID, name, email, "
    "zip code or other detail exactly as the ticket writes it; if the ticket does not give an ID, say "
    "that it still has to be looked up. Never make up or guess an ID."
)
WRITE_ROLES = ("single", "executor")


def _note(role: str) -> str:
    if role in WRITE_ROLES:
        return TICKET_MODE_NOTE
    return INTAKE_NOTE if role == "intake" else READ_ONLY_NOTE


def _system(role: str, policy: str, fmt: str | None, cfg: RunConfig, last: bool,
            has_tools: bool = True) -> str:
    note = _note(role)
    s = (f"{ROLE_DESC[role]}\n\nStore policy:\n{policy}\n\n{note}\n\n"
         "Make at most one tool call at a time.")
    if fmt and not last:
        # Provenance (D4/D5) tags facts with the tool call they came from. An agent with no tools (the
        # intake) has nothing to tag: in the pilot, told to tag every fact, it dropped the customer's
        # email and zip on tasks 3 and 4 in every D4 run. So it gets the plain instruction.
        s += "\n\n" + H.instruction(fmt, cfg.provenance and has_tools)
    return s


def _make_handoff(rt, frm: str, to: str, fmt: str, res, prior: list[dict]) -> dict:
    cfg: RunConfig = rt["cfg"]
    usage: Usage = rt["usage"]
    if fmt == "free_text":
        content = res.final_text
    elif fmt == "structured":
        llm = cfg.llm(num_predict=cfg.handoff_max_tokens).with_structured_output(
            H.HandoffSpec, method="json_schema")
        usage.llm_calls += 1
        try:
            msgs = [SystemMessage(H.STRUCTURED_FILL_SYSTEM),
                    HumanMessage(H.render_work_log(frm, res.messages))]
            spec = llm.invoke(msgs)
            bad = H.label_only_facts(spec)
            if bad:  # one retry, pointing at the label-only facts
                usage.llm_calls += 1
                usage.handoff_retries += 1
                spec = llm.invoke(msgs + [HumanMessage(
                    "These facts are labels without values: " + "; ".join(bad) + ". Rewrite the "
                    "handoff so every fact is '<label>: <actual value>' copied from the log, and "
                    "drop any fact whose value the log does not give.")])
            # pipeline hops only: the incoming message was a structured handoff (decision 19)
            incoming = prior[-1]["content"] if prior and prior[-1]["format"] == "structured" else None
            spec, carried = H.carry_forward_facts(incoming, spec)
            content = H.render_structured(spec)
        except Exception as e:  # runaway or malformed JSON: keep the run, record the failure
            return {"from": frm, "to": to, "format": fmt, "content": res.final_text,
                    "handoff_error": f"{type(e).__name__}: {str(e)[:300]}"}
        return {"from": frm, "to": to, "format": fmt, "content": content,
                "carried_facts": carried, "new_facts": len(spec.facts) - carried}
    else:  # full_context: everything so far + this agent's transcript
        hist = prior[-1]["content"] if prior else ""
        content = (hist + "\n\n" if hist else "") + H.render_transcript(frm, res.messages)
    return {"from": frm, "to": to, "format": fmt, "content": content}


def _incoming(state: TeamState, rt, first: bool) -> str:
    if first:
        return rt["request"]
    return f"Handoff from the previous agent:\n\n{state['handoffs'][-1]['content']}"


# ---------------------------------------------------------------- single / pipeline
def _agent_node(role: str, tools: list[str], idx: int, n: int, fmt: str | None):
    def node(state: TeamState, config) -> TeamState:
        rt = _rt(config)
        env: ToolEnv = rt["env"]
        cfg: RunConfig = rt["cfg"]
        last = idx == n - 1
        res = run_agent(name=role, system=_system(role, env.policy(), fmt, cfg, last, bool(tools)),
                        user=_incoming(state, rt, idx == 0), tools=tools, env=env, cfg=cfg,
                        usage=rt["usage"], original_request=rt["request"], wrap_up=not last)
        out: TeamState = {
            "transcripts": state.get("transcripts", []) + [{"agent": role, "messages": res.messages}],
            "tool_log": [r.to_dict() for r in env.log],
            "usage": rt["usage"].to_dict(),
        }
        if last:
            out["final_answer"] = res.final_text
        else:
            nxt = PIPELINES[n][idx + 1][0] if n in PIPELINES else "next"
            out["handoffs"] = state.get("handoffs", []) + [
                _make_handoff(rt, role, nxt, fmt, res, state.get("handoffs", []))]
        return out

    return node


def build_single():
    g = StateGraph(TeamState)
    g.add_node("single", _agent_node("single", ALL_TOOLS, 0, 1, None))
    g.add_edge(START, "single")
    g.add_edge("single", END)
    return g


def build_pipeline(depth: int, fmt: str):
    roles = PIPELINES[depth]
    g = StateGraph(TeamState)
    prev = START
    for i, (role, tools) in enumerate(roles):
        g.add_node(role, _agent_node(role, tools, i, depth, fmt))
        g.add_edge(prev, role)
        prev = role
    g.add_edge(prev, END)
    return g


# ---------------------------------------------------------------- supervisor
SUP_WORKERS = {
    "account": READ_TOOLS_ACCOUNT,
    "catalog": READ_TOOLS_CATALOG + GENERIC_TOOLS,
    "executor": EXECUTOR_TOOLS,
}
SUPERVISOR_PROMPT = """You are the supervisor of a retail customer-service team. You hold the customer's
ticket; your workers do not see it. Workers: account (authenticates the customer, reads profile/orders/
payment methods), catalog (reads order/product/item details, finds item IDs and prices), executor (the only
one who can change the database). Workers cannot talk to each other; everything goes through you.

Each turn, reply with ONLY a JSON object, either
  {{"delegate": "<account|catalog|executor>", "instructions": "<everything the worker needs>"}}
or, when the ticket is fully handled,
  {{"finish": "<summary of what was done>"}}

Workers know nothing except your instructions, so the instructions must contain everything the worker
needs: the customer's name, zip code or email, and every order ID, item ID, product ID, payment method
ID and price from the ticket or earlier reports, copied exactly. Include every rule, limit, prohibition
or approval requirement from the ticket that applies to the worker's step, word for word.

Store policy:
{policy}

{ticket_note}"""


def _sup_node(state: TeamState, config) -> TeamState:
    rt = _rt(config)
    cfg: RunConfig = rt["cfg"]
    usage: Usage = rt["usage"]
    rnd = state.get("sup_round", 0)
    reports = "\n\n".join(f"Report from {h['from']}:\n{h['content']}"
                          for h in state.get("handoffs", []) if h["to"] == "supervisor")
    prompt = f"TICKET:\n{rt['request']}\n\n" + (f"WORKER REPORTS SO FAR:\n{reports}" if reports
                                                 else "No worker reports yet.")
    decision = {"finish": "(round limit reached)"}
    if rnd < rt["max_rounds"]:
        ai = cfg.llm(think=cfg.no_tool_think).invoke([SystemMessage(SUPERVISOR_PROMPT.format(policy=rt["env"].policy(), ticket_note=TICKET_MODE_NOTE)),
                               HumanMessage(prompt)], format="json")
        usage.add(ai)
        try:
            decision = json.loads(ai.content)
        except json.JSONDecodeError:
            decision = {"finish": f"(unparseable supervisor output) {ai.content[:200]}"}
    out: TeamState = {"sup_round": rnd + 1, "usage": usage.to_dict()}
    if decision.get("delegate") in SUP_WORKERS:
        w = decision["delegate"]
        out["sup_next"] = decision
        out["handoffs"] = state.get("handoffs", []) + [
            {"from": "supervisor", "to": w, "format": rt["fmt"],
             "content": str(decision.get("instructions", ""))}]
    else:
        out["sup_next"] = None
        out["final_answer"] = str(decision.get("finish", decision))
    return out


def _worker_node(name: str):
    def node(state: TeamState, config) -> TeamState:
        rt = _rt(config)
        env: ToolEnv = rt["env"]
        cfg: RunConfig = rt["cfg"]
        res = run_agent(name=name, system=_system(name, env.policy(), rt["fmt"], cfg, False)
                        .replace("next agent", "supervisor"),
                        user=f"Instructions from the supervisor:\n\n{state['handoffs'][-1]['content']}",
                        tools=SUP_WORKERS[name], env=env, cfg=cfg, usage=rt["usage"],
                        original_request=rt["request"], wrap_up=True)
        h = _make_handoff(rt, name, "supervisor", rt["fmt"], res,
                          [x for x in state.get("handoffs", []) if x["format"] == "full_context"
                           and x["to"] == "supervisor"])
        return {"handoffs": state["handoffs"] + [h],
                "transcripts": state.get("transcripts", []) + [{"agent": name, "messages": res.messages}],
                "tool_log": [r.to_dict() for r in env.log], "usage": rt["usage"].to_dict()}

    return node


def build_supervisor():
    g = StateGraph(TeamState)
    g.add_node("supervisor", _sup_node)
    for w in SUP_WORKERS:
        g.add_node(w, _worker_node(w))
        g.add_edge(w, "supervisor")
    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor",
                            lambda s: (s.get("sup_next") or {}).get("delegate", END),
                            {**{w: w for w in SUP_WORKERS}, END: END})
    return g


def compile_graph(topology: str, depth: int = 3, fmt: str = "free_text", checkpointer=None):
    if topology == "single":
        g = build_single()
    elif topology == "pipeline":
        g = build_pipeline(depth, fmt)
    elif topology == "supervisor":
        g = build_supervisor()
    else:
        raise ValueError(topology)
    return g.compile(checkpointer=checkpointer or InMemorySaver())
