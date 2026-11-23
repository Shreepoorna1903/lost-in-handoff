# FP3 progress report (Nov 23) — Group 14, Lost in Handoff

Study (from FP2): an incorrect fact enters a 3-agent pipeline (intake → researcher → executor) on
τ²-bench retail with Qwen3-8B. We measure whether it is passed on and turned into a wrong database
write, and test a write-gating verifier with controlled information access and provenance tags
(trace vs. re-query).

## Status against the FP3 requirements

| FP3 requirement | Status |
|---|---|
| 1. Functional agentic system | **Done.** Single agent and intake → researcher → executor pipeline (LangGraph), τ²-bench retail tools behind our `ToolEnv` wrapper (access control, call log, fault hook). Runs end to end on Ollama, locally and on a Colab A100. |
| 2. Conditions selectable by parameter | **Partly done.** Topology, handoff format and model are CLI flags; the full-information verifier (D2) and provenance trace (D4) are `RunConfig` flags. Tool-level faults (corrupted output, wrong ID, irrelevant) and the false-claim checkpoint fork are implemented. **Still to build:** matched verifier (D1), executor with full information (D3), re-query provenance (D5), and CLI flags to pick a fault and a defense. |
| 3. Experiment runner | **Partly done.** `handoff.cli baseline` loads tasks, runs a condition, saves one JSON per run, resumes after interruption, and splits work across parallel processes (`--shard K/N`). **Still to build:** a sweep over fault × defense × seed, and a trial ID per run. |
| 4. Logging and measurement | **Mostly done.** Each run record has task ID, condition, model, seed, sampling settings, final answer, all messages, handoffs, every tool call (agent, args, output, error, injected fault), token usage, model and tool calls, truncations, and wall time. Already computed: task success (database hash vs. gold replay), collateral damage, value tracing through handoffs and write arguments. Failed runs are saved and counted. **Still to build:** the FP2 metrics script (fault-induced failure, wrong-value write, passed on, detection, over-blocking) and summary tables. |
| 5. Evaluation checks | **Partly done.** 13 offline tests (no model) check gold replay, tool access, fault injection, database restore, value tracing and handoff parsing. **Still to do:** hand-check the metrics on 20 saved runs. We no longer use an LLM judge (see FP2), so no judge agreement is needed. |

## Pilot so far (clean baseline, no faults, Qwen3-8B, Colab A100, 1 trial, the same 10 tasks)

Records are in `runs/colab_a100/baseline/`.

| Condition | Solved | Mean time per run | Mean tokens per run | Mean tool calls |
|---|---|---|---|---|
| Single agent | 8/10 | 92 s | 69k | 6.8 |
| Pipeline (3 agents, free text) | 2/10 (both are no-op tasks) | 166 s | 78k | 6.6 |

What the pilot exposed: the pipeline fails because the researcher, which has no write tools,
writes that the changes were already made and the executor believes it (tasks 0, 2, 11, 14). In
other tasks the researcher runs out of turns (3, 4, 13) or returns an empty handoff (1, 12). We
will fix the researcher's role prompt and the handling of empty or turn-limited handoffs, then
rerun the baseline to choose the 8 study tasks.

## Division of labor for FP3 (due Nov 30)

| Who | Responsible for | Target |
|---|---|---|
| **Shreepoorna Purohit** (integration lead) | GitHub repo; researcher and handoff fixes; clean baseline and task selection; no-op task filter; fault × defense sweep runner and CLI flags; Colab runs; merging everyone's parts | repo Nov 24, baseline Nov 26, sweep Nov 28 |
| **Harsha Prakash** (faults, RQ1) | Script that picks fault targets from gold actions; the 4 fault types (F1–F4); end-to-end test of the false-claim checkpoint fork; offline tests for each fault | Nov 27 |
| **Mandar Arun** (verifier, RQ2) | Matched verifier (D1); keep the full verifier (D2); executor with full information (D3); fault-free runs to measure over-blocking; offline tests | Nov 27 |
| **Puneet Singh Puri** (provenance and metrics, RQ3) | Re-query provenance (D5) next to trace (D4); FP2 metrics script and summary tables; hand-check of metrics on 20 runs | Nov 27 (D5), Nov 29 (metrics) |
| **All** | Pilot of at least 10 tasks per condition (Nov 28–29); each writes the FP3 report section for their part; Shreepoorna assembles and submits | Nov 30 |
