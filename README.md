# Lost in Handoff — Group 14

How constraints disappear and errors spread in tool-using multi-agent systems.
Code for the CS 6983 Agentic AI final project (Group 14). Current status and division of labor: `PROGRESS.md`.

## Setup

Needs Python 3.13, [uv](https://docs.astral.sh/uv/), and [Ollama](https://ollama.com). τ²-bench is
pinned to the commit we use and installed from `vendor/` (not committed).

```bash
git clone git@github.com:Shreepoorna1903/lost-in-handoff.git && cd lost-in-handoff
git clone https://github.com/sierra-research/tau2-bench vendor/tau2-bench
git -C vendor/tau2-bench checkout 57e54143025bec219b5726884651c98eede202e2
uv venv --python 3.13 && uv pip install -e ".[dev]"
.venv/bin/python -m pytest    # 13 offline tests, no model needed
ollama pull qwen3:8b          # needed only for real runs
```

**iCloud-synced folders (macOS Desktop/Documents):** iCloud marks `.pth` files hidden and Python
3.13 skips hidden `.pth` files, so put the venv outside iCloud and symlink it:
`uv venv --python 3.13 ~/.venvs/lost-in-handoff && ln -s ~/.venvs/lost-in-handoff .venv`, then
`VIRTUAL_ENV=$HOME/.venvs/lost-in-handoff uv pip install -e ".[dev]"`.

**GPU runs:** `colab/lost_in_handoff_colab.ipynb` runs the baselines on a Colab A100 (4 runs in
parallel on one Ollama server). Pilot records from it are in `runs/colab_a100/`.

## Run

```bash
.venv/bin/python -m handoff.cli list                                   # retail tasks with write actions
.venv/bin/python -m handoff.cli run --task 0 --topology single --show
.venv/bin/python -m handoff.cli run --task 0 --topology pipeline --depth 4 --fmt structured \
    --constraints budget,scope,prohibition,approval --show
.venv/bin/python -m handoff.cli run --task 0 --topology supervisor --fmt free_text --verifier
.venv/bin/python -m handoff.cli baseline --topology single --limit 10   # clean-baseline filter
```

Each run writes a JSON file to `runs/` with the outcome, the metrics, and the full state: handoffs,
per-agent transcripts, and the tool log.

## How the code maps to the proposal

| Proposal component | Code |
|---|---|
| Environment: τ²-bench retail | `env.py` wraps `tau2.domains.retail`. Tasks become single-shot tickets in `tasks.py` |
| Tool wrapper (logs every call, injection hook) | `env.ToolEnv.call` |
| Agents: user-facing → read-only specialists → single executor with write tools | `topologies.PIPELINES`, `SUP_WORKERS` |
| Topologies: single, pipeline (depth 2–4), supervisor | `topologies.build_*` (LangGraph + `InMemorySaver`) |
| Handoff formats: free-text / structured JSON / full context | `handoffs.py` |
| Constraints: budget / prohibition / approval / scope (RQ1) | `constraints.py`. Generated from gold actions so the correct final state still satisfies them |
| Faults: corrupted output / wrong ID / irrelevant control (RQ2–3) | `faults.py` (applied by `ToolEnv` at the n-th call of a tool) |
| Fault: false handoff claim (RQ3) | `runner.fork_with_false_claim`: finds the checkpoint after hop *k*, edits the handoff, restores the DB by replaying logged writes, and resumes |
| Defenses: write-gating verifier, provenance tags (RQ4) | `agents.verify`, `RunConfig.provenance` (`[call #N]` labels plus a `get_tool_call_record` tool) |
| Measurement: success, collateral damage, violations, per-hop survival, taint | `evaluate.py` |
| Budget matching | `RunConfig.tool_call_budget` is shared by all agents in a run. Tokens, LLM calls, and latency are logged per run |

### Design decisions to review as a team

1. **Single-shot tickets instead of τ²'s user simulator.** The user-facing agent gets the whole request
   up front, and we tell it the customer has pre-confirmed the actions. This removes simulator noise and
   keeps that agent as the only holder of the constraints. It also changes τ²'s protocol, so our baseline
   numbers are not comparable to the τ² leaderboard.
2. **Depth-2 pipeline merges intake and research.** Every depth needs read access, write access, and the
   original ticket. At depth 2 the intake agent therefore also holds the read tools, which differs from
   depths 3 and 4 (pure intake). This confound should be stated in the write-up.
3. **The executor has no read tools.** It depends entirely on handoffs, which is the point of the study.
   τ²'s write tools still validate IDs, so a wrong but valid ID passes and a malformed one fails.
4. **`request_manager_approval`** is a no-op tool added so that approval-rule constraints leave the
   correct final DB state unchanged. Every executor has it in every condition.
5. **Constraint survival** is currently a key-term match over the whole handoff (`survival`). For
   structured handoffs, `survival_in_field` also records whether the constraint landed in the
   `constraints` field. The proposal calls for an LLM judge on free-text hops (TODO).
6. **Same content rules for every format.** Free-text and structured handoffs get identical instructions
   about what to carry (`handoffs.CONTENT_RULES`: rules word for word, exact IDs and prices). Only the
   form differs, so RQ1 compares formats rather than prompts.
7. **Structured handoffs are a separate extraction call.** A clean prompt turns the agent's work log
   into the JSON (`STRUCTURED_FILL_SYSTEM`). When the JSON was requested as one more turn at the end of
   the agent's own chat, Qwen3-4B copied the instructions into the fields and dropped the tool results.
8. **Supervisor instructions are always plain text.** The supervisor's instructions to a worker are
   free text whatever the format setting. Only the workers' reports use the chosen format. Decide
   whether the supervisor's instructions should follow the format too.
9. **The intake agent is told not to invent anything, and extraction trusts tool results over the
   incoming handoff.** Both rules apply in every condition. The second one probably makes false
   handoff claims easier to catch (RQ3), so the write-up should mention it.
10. **Thinking is set by role (a control).** Agents that hold environment tools (single agent,
    researcher, account, catalog, executor) run with Qwen3 thinking **on**. Agents without them
    (pipeline intake, supervisor) run with it **off** (`RunConfig.no_tool_think`). Calls that are not
    agent turns (structured extraction, verifier) follow `RunConfig.think`. The reasons:
    - With thinking off, Qwen3 drops tool calls under Ollama.
    - With thinking on, the 8B supervisor reasoned past 8k tokens without answering.

    The rule depends only on a role's environment tools, so the provenance lookup tool never changes
    it, and it is identical across topologies, formats and defenses. It departs from the proposal's
    "all agents use the same model at a fixed setting". In the write-up, report it as: same model and
    temperature for every agent, thinking set by role. Each result records `think` and `no_tool_think`.
11. **We exclude 17 tasks whose correct answer depends on the confirmation dialogue.** In these, the
    customer's script reacts to the agent asking for confirmation. Task 5, for example: "If the agent
    asks for confirmation, only exchange the desk lamp. If the agent asks for confirmation again, … return
    the water bottle instead." τ²'s simulated customer acts this out, and the gold actions record the
    result. A single-shot ticket that is already confirmed cannot reproduce it, so an agent that does
    exactly what the ticket asks is graded wrong.
    - **Rule** (`tasks.depends_on_confirmation_dialogue`): "confirm" appears in the customer's
      `reason_for_call` or `task_instructions`. `load_tickets()` applies it by default;
      `exclude_dialogue=False` turns it off.
    - **Excluded tasks:** 5, 6, 7, 8, 9, 18, 21, 22, 37, 59, 63, 71, 72, 84, 91, 92, 98. That leaves
      90 of the 107 tasks with write actions.
    - **Not excluded:** a wider search finds 19 more tasks whose script depends on what the agent does
      (e.g. task 20: "if the upgrade is not possible, proceed with the next order"). Most look solvable
      from the ticket alone, but the team should review them before the final task set is fixed.
12. **Numbers become strings when the tool schema expects a string.** Models often send IDs like
    `8349118980` as JSON numbers, and τ² compares IDs as strings, so a correct call would fail with
    "not found". This happened in task 3. `ToolEnv.call` converts them before the call and marks the
    record `coerced=True`.
13. **Two more conversion problems, not fixed yet (team decision):**
    - **Doing nothing passes 4 tasks: 10, 12, 50 and 105.** Their gold outcome is a transfer to a human
      agent, which leaves the DB unchanged, so the DB-hash check scores an idle agent as successful.
      Either exclude them from success rates, or score them on whether `transfer_to_human_agents` was
      called.
    - **Instructions meant for the simulated customer leak into 5 tickets: 12, 13, 14, 45 and 50.** For
      example: "(Everything except a keyboard and a mouse, but do not reveal it to the agent)". The ticket
      text copies the customer's script verbatim, so the agent sees what the customer was told to hide.
      This makes the tasks easier, not invalid. A fix is to rewrite those clauses or drop the tasks.
14. **Sampling follows Qwen3's recommendations, not greedy decoding.** With thinking on, temperature 0
    makes Qwen3 repeat itself until it hits the token cap. That explains the supervisor's 50k-token loop
    and the researcher returning nothing in pipeline task 0. On the researcher's exact input, temperature
    0 hit 4,096 tokens in 286 s with no answer, while Qwen's recommended settings finished in 22 s.
    Settings are temperature / top_p / top_k = 0.6 / 0.95 / 20 with thinking on and 0.7 / 0.8 / 20 with
    thinking off, with a fixed seed. Each result records them under `sampling`. This replaces the
    proposal's "fixed temperature", so report it as fixed sampling settings and a fixed seed for every
    condition.
15. **Handoffs must carry the customer's identity.** Name, zip code, email and user ID are now in
    `CONTENT_RULES` for every format. In pipeline task 0 the intake agent's handoff dropped
    "Yusuf Rossi, 19122", so the researcher could not authenticate the customer and asked for an email.
16. **Every agent's prompt states that the ticket is already confirmed** (`topologies.TICKET_MODE_NOTE`).
    The store policy requires explicit customer confirmation before any change. With single-shot
    tickets, agents followed the policy, stopped to ask "Confirm if you'd like to proceed", and never
    acted. This happened in task 4 (first pass) and task 3 (second pass). The note says the written
    ticket satisfies the confirmation rule and nobody will answer questions. The text is the same for
    every role, topology and format.
17. **Agents with tools get one fixed "continue" prompt** (`agents.CONTINUE_NUDGE`, at most
    `max_continuations=1` per agent, counted in `usage.continuations`). After decision 16, task 3 still
    failed: the agent wrote "Let me check your pending orders and apply the changes" and ended its turn
    without a tool call. In τ² the simulated customer would answer "go ahead". With no one to answer,
    the loop took that text as the final reply. The prompt asks the agent to continue if anything is
    unfinished, and otherwise to give its final message. It is the same in every condition. The intake
    agent has no tools and does not get it.
18. **Structured handoffs are checked for facts that are only labels, with one retry**
    (`handoffs.label_only_facts`, counted in `usage.handoff_retries`). In pipeline task 0 the intake's
    handoff listed `"Customer identity details"` and `"Order ID"` without their values. The order
    number was lost, so the researcher used τ²'s docstring example `#W0000000`. The supervisor's catalog
    worker did the same before. Facts must now be written as "label: actual value", and a fact of 4 words
    or fewer with no digit, `#`, `@` or `$` triggers one re-extraction that names the bad facts. This
    affects structured handoffs only. **For RQ1, report how many handoffs needed the retry**, since
    without the check this is itself a form of information loss.
19. **Structured handoffs carry facts forward in code, and the model restates the constraints**
    (option (c), chosen 2026-10-23; `handoffs.carry_forward_facts`). In pipeline task 0, after decision
    18, the researcher authenticated the customer and looked up the order and both products correctly,
    but its structured handoff was an exact copy of the intake's 11 facts, without the item IDs it had
    just found. Now, on pipeline hops:
    - **Facts:** the incoming handoff's facts are prepended to the agent's new ones, with duplicates
      removed. Each handoff records `carried_facts` and `new_facts`.
    - **What stays the model's:** `goal`, `constraints` and `next_steps` are exactly what the model
      wrote. Constraint survival in structured handoffs is therefore still measured, not guaranteed (RQ1).
    - **Consequences for the write-up:**
      - In structured pipelines, facts can no longer be lost at a hop, so fact loss is only measurable
        in free text and full context.
      - A **false handoff claim** (RQ3) is also carried forward by code, so it reaches the executor
        unless an agent notices it. When a later agent's tool result contradicts it, both the false
        and the correct fact appear downstream. That conflict is a detection signal worth measuring.
      - The supervisor topology is unaffected, because its workers receive plain-text instructions.

## Baseline sweep (2026-10-22, Qwen3-8B, clean runs: no faults, no added constraints)

One run per task, so the scores are noisy. With sampling on, the real experiments need several seeds
per task and should report success rates with confidence intervals. Run with `scripts/sweep_baseline.sh` (resumable). The log is in `runs/baseline/sweep.log`. The first 10
valid tasks are 0–4 and 10–14.

| Condition | Solved | Notes |
|---|---|---|
| single | **9/10** (7/8 without the do-nothing tasks 10 and 12) | Final settings (decisions 14–17). **Task 2:** returned nothing and transferred to a human without looking up the customer's orders. It passed in the temperature-0 pass, so this is likely run-to-run variance. Tasks 4 and 14, which failed in the first pass, now pass. 2–10 minutes per task, 16k–104k tokens. Earlier passes are in `runs/baseline/archive_*` |
| pipeline-d3, structured | running | Earlier passes: at temperature 0, task 0 hit the token cap and returned nothing (decisions 14 and 15). With the final settings, the intake's handoff listed facts without values (decision 18) |

## Status (2026-10-21, smoke tests on task 0)

- [x] τ²-bench retail installed. 114 tasks, all with gold actions. The 7 offline tests pass.
- [x] Single agent with Qwen3-4B (thinking off) solves task 0 exactly: 6 tool calls, about 54k tokens, about 3.6 minutes.
- [ ] Week-1 environment decision: run `baseline` on about 30 tasks per model and topology to pick the task set (target 30–40 tasks).
- [x] Pipeline, depth 3, structured handoffs, approval and scope constraints, Qwen3-4B: solves task 0
      exactly with both constraints kept. 7 tool calls, about 61k tokens, about 9 minutes. This needed
      the shared content rules and the separate extraction call; before them, the handoffs dropped the
      item IDs.
- [x] Pipeline, depth 3, structured handoffs, approval and scope constraints, **Qwen3-8B, thinking on**:
      solves task 0 exactly. 5 tool calls, about 53k tokens, about 16 minutes. It needed three more fixes.
      The intake agent may not state IDs it never saw. Structured extraction separates INCOMING
      (untrusted) from TOOL RESULTS (authoritative). Every Ollama call has a timeout and a token cap.
      Without them, the intake made up IDs (`KBD123`) and the researcher's handoff copied them over its
      own correct lookups.
- [ ] Supervisor on Qwen3-8B, thinking on: the supervisor's first call reasons without stopping and never
      answers. It hit the 8,192-token cap, and an uncapped run reached 50k tokens in 85 minutes. With
      thinking off, the same call returns a valid delegation in 4 seconds. The JSON `format` setting is
      not the cause. Decided: roles without tools run with thinking off (design decision 10).
      Rerun with role-based thinking (45 minutes, 33 LLM calls, about 98k tokens): the supervisor now answers
      quickly and routes account → catalog → executor, but it **still fails**. The catalog worker's
      structured handoff listed field *names* ("order_id", "price") instead of values, and put product IDs
      into `source_call`. The supervisor (thinking off) then **made up IDs** (`123456`, `654321`), and the
      executor tried them 4 times with τ²'s example order `#W0000000`. The approval constraint was dropped
      at the first hop and never came back.
- [ ] The 8B pipeline result above ran with the intake's thinking **on**, and the 4B results ran with
      thinking off for every agent. Rerun the pipeline under the role-based setting before using it as
      the baseline.
- [ ] Qwen3 with thinking **off** drops tool calls under Ollama: 8B returns empty turns, and 4B often
      stops after "Okay, let's see...". So thinking is on by default (`RunConfig.think`).
- [ ] The researcher on 8B skipped authentication, which the policy requires. τ²'s DB check does not
      catch this, so consider a policy-compliance metric.
- [ ] Week-1 environment decision: run `baseline` on about 30 tasks per model and topology to pick the task set (target 30–40 tasks).
- [x] Pipeline, depth 3, structured handoffs, approval and scope constraints, Qwen3-4B: solves task 0
      exactly with both constraints kept. 7 tool calls, about 61k tokens, about 9 minutes. This needed
      the shared content rules and the separate extraction call; before them, the handoffs dropped the
      item IDs.
- [ ] Supervisor topology: runs end to end (account → catalog → executor) but fails task 0 on Qwen3-4B.
      Its instructions to the executor dropped the old item IDs and the approval rule and gave a wrong
      refund amount. The executor then stopped without a tool call. Try Qwen3-8B before changing prompts
      further, since the errors look like model capacity.
- [ ] Qwen3-4B with thinking off often ends a turn with "Okay, let's see..." text and no tool call.
      Compare thinking on vs off, and 4B vs 8B, on the baseline task set.
- [ ] Experiment drivers that sweep the grids for A, B, and C and write a results table.
- [ ] Fault targeting per task: pick the JSON path and a valid alternative ID for each task from its gold actions.
- [ ] LLM judge for free-text constraint survival and for detection ("expressions of doubt"), validated against human labels.
- [ ] Persistent checkpointer (SQLite) if runs need to be forked after the process exits.

### Compute note

On one M4 with 16 GB, Qwen3-4B takes about 3–4 minutes per single-agent run, and multi-agent runs are
slower. The full grid (depth 3 × format 3 × constraint 4 × about 35 tasks × 3 models) is several
thousand runs, so we should split tasks or models across the team's laptops. Qwen3-30B-A3B (about 18 GB
at Q4) will not fit in 16 GB, and Qwen3-14B (about 9 GB) will fit but with a smaller context.
