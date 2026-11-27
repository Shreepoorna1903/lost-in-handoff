"""A minimal ReAct worker loop over ToolEnv, plus the two defenses.

We use our own loop rather than langgraph.prebuilt so that every tool call
goes through ToolEnv (logging + fault hook), the shared tool-call budget is
enforced across agents, and the verifier can sit in front of write calls.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama

from handoff.env import GENERIC_TOOLS, READ_TOOLS, ToolEnv

PROVENANCE_TOOL = "get_tool_call_record"
PROVENANCE_SCHEMA = {
    "type": "function",
    "function": {
        "name": PROVENANCE_TOOL,
        "description": "Return the original output of an earlier tool call by its number "
        "(the N in [call #N]), to check a claim against its source.",
        "parameters": {"type": "object",
                       "properties": {"call_id": {"type": "integer"}},
                       "required": ["call_id"]},
    },
}
# D5: same tool name and [call #N] tags as D4, but the call is re-run on the current database
# instead of returning the logged output. Only read calls can be re-run.
PROVENANCE_REQUERY_SCHEMA = {
    "type": "function",
    "function": {
        "name": PROVENANCE_TOOL,
        "description": "Re-run an earlier read-only tool call by its number (the N in [call #N]) "
        "against the current store database and return the fresh output, to check a claim "
        "against its source.",
        "parameters": PROVENANCE_SCHEMA["function"]["parameters"],
    },
}
REQUERYABLE_TOOLS = READ_TOOLS + GENERIC_TOOLS


@dataclass
class RunConfig:
    model: str = "qwen3:4b"
    # Sampling follows Qwen3's published recommendations, with a fixed seed for reproducibility.
    # Greedy decoding (temperature 0) with thinking on makes Qwen3 repeat itself until the token
    # cap: on the researcher's exact input it hit 4,096 tokens in 286s, versus 345 tokens in 22s
    # with these settings. (temperature, top_p, top_k) for thinking on / off:
    sampling_think: tuple[float, float, int] = (0.6, 0.95, 20)
    sampling_no_think: tuple[float, float, int] = (0.7, 0.8, 20)
    seed: int = 0
    # Qwen3 under Ollama drops tool calls when thinking is disabled (8B returns empty turns),
    # so thinking is on by default. Reasoning tokens count toward output_tokens.
    think: bool = True
    # Roles with no environment tools (pipeline intake, supervisor) run with thinking off: with it on,
    # Qwen3-8B's supervisor call reasoned past 8k tokens without answering, while off it answers in
    # seconds. The rule is by role, so it is identical across topologies, formats and defenses.
    no_tool_think: bool = False
    num_ctx: int = 16384
    tool_call_budget: int = 30  # shared by all agents in a run
    max_turns_per_agent: int = 12
    max_empty_retries: int = 2
    # Tool-using agents that reply with text only get this many fixed "continue" prompts before
    # the reply counts as final (agents announced "Let me apply the changes" and stopped, task 3).
    max_continuations: int = 1
    provenance: bool = False
    # D5: with provenance on, get_tool_call_record re-runs read call #N on the current database
    # instead of returning its logged output (D4). Off by default; has no effect without provenance.
    provenance_requery: bool = False
    verifier: bool = False

    handoff_max_tokens: int = 4096  # cap on structured-handoff extraction (guards runaway JSON)
    request_timeout: float = 600.0  # seconds of silence before an Ollama call raises
    # Ollama streams tokens, so the timeout never fires on a runaway generation (we saw a single
    # 8B thinking call reach 50k tokens). Every call is capped; hits are counted in Usage.truncated.
    max_tokens_per_call: int = 8192

    def llm(self, num_predict: int | None = None, think: bool | None = None) -> ChatOllama:
        think = self.think if think is None else think
        temperature, top_p, top_k = self.sampling_think if think else self.sampling_no_think
        return ChatOllama(model=self.model, temperature=temperature, top_p=top_p, top_k=top_k,
                          seed=self.seed, reasoning=think, num_ctx=self.num_ctx,
                          num_predict=num_predict or self.max_tokens_per_call,
                          client_kwargs={"timeout": self.request_timeout})


@dataclass
class Usage:
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0
    verifier_calls: int = 0
    verifier_blocks: int = 0
    empty_retries: int = 0
    continuations: int = 0
    handoff_retries: int = 0  # structured handoffs re-extracted for label-only facts
    wrap_ups: int = 0  # extra tool-free calls to write a handoff after a turn-limited or empty ending
    truncated: int = 0  # calls that hit max_tokens_per_call
    requeries: int = 0  # D5 re-runs of a logged read call (also counted in tool_calls)

    def add(self, msg: AIMessage) -> None:
        self.llm_calls += 1
        um = getattr(msg, "usage_metadata", None) or {}
        self.input_tokens += um.get("input_tokens", 0)
        self.output_tokens += um.get("output_tokens", 0)
        if (getattr(msg, "response_metadata", None) or {}).get("done_reason") == "length":
            self.truncated += 1

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict | None) -> "Usage":
        return cls(**(d or {}))


@dataclass
class AgentResult:
    final_text: str
    messages: list[dict] = field(default_factory=list)  # serialised for state / transcripts
    lc_messages: list = field(default_factory=list, repr=False)


def _ser(m) -> dict:
    if isinstance(m, AIMessage):
        return {"role": "assistant", "content": m.content,
                "tool_calls": [{"name": t["name"], "args": t["args"]} for t in m.tool_calls]}
    if isinstance(m, ToolMessage):
        return {"role": "tool", "content": m.content}
    if isinstance(m, SystemMessage):
        return {"role": "system", "content": m.content}
    return {"role": "user", "content": m.content}


# Stands in for tau2's simulated customer saying "go ahead". Identical in every condition.
CONTINUE_NUDGE = ("If anything you were asked to do is still not done, continue now with tool calls. "
                  "Otherwise reply with your final message as instructed.")
EMPTY_NUDGE = "Your last reply was empty. Continue: make your next tool call, or give your final answer."
# Pilot: the researcher hit the turn limit (tasks 3, 4, 13) or ended with an empty reply (1, 12), so
# the executor received "(turn limit reached)" or nothing. An agent that writes a handoff gets one
# more call, with no tools, to write it. Identical in every condition.
WRAP_UP_PROMPT = ("You cannot make any more tool calls. Write your final message for the next agent "
                  "now, as instructed, using only what the tool results above show. If something is "
                  "still unknown, say so.")
TURN_LIMIT_TEXT = "(turn limit reached)"

VERIFIER_PROMPT = """You are a verifier that gates write actions in a retail customer-service system.
Approve the proposed action only if it is consistent with (1) the customer's original ticket and every
rule stated in it, and (2) the tool results gathered so far (IDs, prices, statuses must match).
Reply with exactly one line: either `APPROVE` or `REJECT: <short reason>`."""


def verify(cfg: RunConfig, usage: Usage, original_request: str, env: ToolEnv,
           tool: str, args: dict) -> str | None:
    """Return None if approved, else a rejection reason."""
    evidence = "\n".join(f"[call #{r.step}] {r.tool}({json.dumps(r.args)}) -> {r.output[:1500]}"
                         for r in env.log if not r.is_write)
    msg = cfg.llm().invoke([
        SystemMessage(VERIFIER_PROMPT),
        HumanMessage(f"ORIGINAL TICKET:\n{original_request}\n\nTOOL RESULTS SO FAR:\n{evidence}\n\n"
                     f"PROPOSED ACTION: {tool}({json.dumps(args)})"),
    ])
    usage.add(msg)
    usage.verifier_calls += 1
    text = (msg.content or "").strip()
    if text.upper().startswith("APPROVE"):
        return None
    usage.verifier_blocks += 1
    return text.split(":", 1)[-1].strip() or "rejected"


def run_agent(*, name: str, system: str, user: str, tools: list[str], env: ToolEnv,
              cfg: RunConfig, usage: Usage, original_request: str = "",
              wrap_up: bool = False) -> AgentResult:
    """`wrap_up`: this agent's final text becomes a handoff, so a turn-limited or empty ending gets
    one extra tool-free call to write it."""
    schemas = env.schemas(tools)
    if cfg.provenance and tools:  # an agent with no tools has no tool calls to look up
        schemas = schemas + [PROVENANCE_REQUERY_SCHEMA if cfg.provenance_requery else PROVENANCE_SCHEMA]
    # decided by the role's environment tools, so adding the provenance tool never flips it
    llm = cfg.llm(think=cfg.think if tools else cfg.no_tool_think)
    if schemas:
        llm = llm.bind_tools(schemas)
    msgs: list = [SystemMessage(system), HumanMessage(user)]
    final = ""
    empty_retries = continuations = 0
    for _ in range(cfg.max_turns_per_agent):
        ai: AIMessage = llm.invoke(msgs)
        usage.add(ai)
        if not ai.tool_calls and not (ai.content or "").strip() and empty_retries < cfg.max_empty_retries:
            # Qwen3 sometimes returns an empty turn; nudge instead of treating it as "done".
            empty_retries += 1
            usage.empty_retries += 1
            msgs.append(HumanMessage(EMPTY_NUDGE))
            continue
        msgs.append(ai)
        if not ai.tool_calls:
            if tools and continuations < cfg.max_continuations:
                continuations += 1
                usage.continuations += 1
                msgs.append(HumanMessage(CONTINUE_NUDGE))
                continue
            final = ai.content or ""
            break
        for tc in ai.tool_calls:
            msgs.append(ToolMessage(_execute(name, tc, tools, env, cfg, usage, original_request),
                                    tool_call_id=tc["id"]))
    else:
        final = TURN_LIMIT_TEXT
    if wrap_up and (final == TURN_LIMIT_TEXT or not final.strip()):
        msgs.append(HumanMessage(WRAP_UP_PROMPT))
        ai = cfg.llm(think=cfg.no_tool_think).invoke(msgs)  # unbound: it cannot call tools
        usage.add(ai)
        usage.wrap_ups += 1
        msgs.append(ai)
        if (ai.content or "").strip():
            final = ai.content
    return AgentResult(final, [_ser(m) for m in msgs], msgs)


def _execute(agent: str, tc: dict, tools: list[str], env: ToolEnv, cfg: RunConfig,
             usage: Usage, original_request: str) -> str:
    tool, args = tc["name"], tc["args"] or {}
    if tool == PROVENANCE_TOOL and cfg.provenance and cfg.provenance_requery:
        return _requery(agent, args.get("call_id"), env, cfg, usage)
    if tool == PROVENANCE_TOOL and cfg.provenance:
        i = args.get("call_id")
        rec = next((r for r in env.log if r.step == i), None)
        return rec.output if rec else f"Error: no tool call #{i}"
    if usage.tool_calls >= cfg.tool_call_budget:
        return "Error: tool-call budget for this ticket is exhausted. Finish now."
    if cfg.verifier and env.is_write(tool):
        reason = verify(cfg, usage, original_request, env, tool, args)
        if reason is not None:
            return f"Blocked by verifier: {reason}. Fix the action or explain why it cannot be done."
    usage.tool_calls += 1
    out = env.call(agent, tool, args, tools)
    return f"[call #{env.log[-1].step}] {out}" if cfg.provenance else out


def _requery(agent: str, call_id, env: ToolEnv, cfg: RunConfig, usage: Usage) -> str:
    """D5: re-run logged read call #call_id on the current database, as `agent`.

    The re-run goes through ToolEnv.call, so it is logged (same tool and args as the original, after
    it) and a tool fault, which fires only on the first matching call, does not fire again: the agent
    sees the true value. It counts toward the shared tool-call budget. The access check allows only
    the original call's tool, so the executor can re-run the researcher's lookup and nothing else.
    Failed calls are never re-run: an access-denied call is logged too, and re-running it would hand
    the agent output its role cannot read. Writes are never re-run; the check is by tool name, since a
    denied write attempt is logged with is_write False."""
    try:
        i = int(call_id)
    except (TypeError, ValueError):
        return f"Error: no tool call #{call_id}"
    rec = next((r for r in env.log if r.step == i), None)
    if rec is None:
        return f"Error: no tool call #{i}"
    if rec.error:  # includes access-denied calls, which are logged too: re-running one would bypass access
        return f"Error: call #{i} failed; only successful read calls can be re-run."
    if rec.tool not in REQUERYABLE_TOOLS:
        return f"Error: call #{i} ({rec.tool}) is not a read call; only read calls can be re-run."
    if usage.tool_calls >= cfg.tool_call_budget:
        return "Error: tool-call budget for this ticket is exhausted. Finish now."
    usage.tool_calls += 1
    usage.requeries += 1
    out = env.call(agent, rec.tool, dict(rec.args), [rec.tool])
    return f"[call #{env.log[-1].step}] {out}"
