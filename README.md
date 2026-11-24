# Lost in Handoff — Group 14

How errors spread through a tool-using multi-agent pipeline. Code for the CS 6983 Agentic AI final
project (Group 14: Shreepoorna Purohit, Harsha Prakash, Mandar Arun, Puneet Singh Puri).

**Current status, pilot results and who does what:** [`PROGRESS.md`](PROGRESS.md).

## The study

A three-agent customer-service pipeline built in LangGraph works on τ²-bench retail tasks with
Qwen3-8B (served by Ollama):

```
intake (no tools) → researcher (read-only tools) → executor (write tools)
```

We plant one wrong fact per run and measure whether it is passed on and turned into a wrong
database write. Then we test defenses that control what information each check can see.

| | Faults (RQ1) | | Defenses (RQ2, RQ3) |
|---|---|---|---|
| F1 | Corrupted value in a tool output | D0 | None |
| F2 | Wrong but valid ID in a tool output | D1 | Verifier with the executor's information (matched) |
| F3 | Irrelevant field changed (control) | D2 | Verifier with full information |
| F4 | False claim in a handoff (checkpoint fork) | D3 | Executor with full information, no verifier |
| | | D4 | Provenance tags: trace the logged source |
| | | D5 | Provenance tags: re-query the source |

Success is graded by comparing the final database hash with a replay of τ²-bench's gold actions.

## Setup

Needs Python 3.13, [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com). τ²-bench is
pinned to the commit we use and installed from `vendor/` (not committed).

```bash
git clone https://github.com/Shreepoorna1903/lost-in-handoff.git && cd lost-in-handoff
git clone https://github.com/sierra-research/tau2-bench vendor/tau2-bench
git -C vendor/tau2-bench checkout 57e54143025bec219b5726884651c98eede202e2
uv venv --python 3.13 && uv pip install -e ".[dev]"
.venv/bin/python -m pytest    # 13 offline tests, no model needed
ollama pull qwen3:8b          # needed only for real runs
```

**macOS iCloud folders (Desktop, Documents):** iCloud hides `.pth` files and Python 3.13 skips
them, so keep the venv outside iCloud and symlink it:
`uv venv --python 3.13 ~/.venvs/lost-in-handoff && ln -s ~/.venvs/lost-in-handoff .venv`, then
`VIRTUAL_ENV=$HOME/.venvs/lost-in-handoff uv pip install -e ".[dev]"`.

**GPU runs:** [`colab/lost_in_handoff_colab.ipynb`](colab/lost_in_handoff_colab.ipynb) runs sweeps
on a Colab A100 (4 runs in parallel on one Ollama server).

## Run

```bash
.venv/bin/python -m handoff.cli list                                      # tasks with write actions
.venv/bin/python -m handoff.cli run --task 0 --topology single --model qwen3:8b --show
.venv/bin/python -m handoff.cli run --task 0 --topology pipeline --model qwen3:8b --show
.venv/bin/python -m handoff.cli baseline --topology pipeline --model qwen3:8b --limit 10 \
    --out runs/baseline --shard 1/4                                       # resumable clean sweep
```

`--verifier` (D2) and `--provenance` (D4) switch on the defenses built so far. Pass
`--model qwen3:8b`; the CLI default is still `qwen3:4b`. Flags for picking a fault and the
remaining defenses are being added (see `PROGRESS.md`).

Each run writes one JSON file with the task, condition, model, seed, sampling settings, every
message, handoff and tool call (with any injected fault), token usage, timing and the grade.

## Code map (`src/handoff/`)

| File | What it does |
|---|---|
| `env.py` | `ToolEnv`: wraps τ²-bench retail tools; access control per agent, call log, fault injection hook |
| `tasks.py` | Turns τ²-bench tasks into single-shot tickets; excludes tasks that need a confirmation dialogue |
| `agents.py` | Agent loop (one tool call per turn), the write-gating verifier |
| `topologies.py` | LangGraph graphs: single agent and the intake → researcher → executor pipeline |
| `handoffs.py` | Handoff formats and the rules for what a handoff must carry |
| `faults.py` | Fault constructors (F1–F3 in tool outputs; F4 text for the false claim) |
| `runner.py` | Runs a ticket under a condition; `fork_with_false_claim` edits a handoff at a checkpoint and resumes (F4) |
| `evaluate.py` | Grading by DB hash vs. gold replay, collateral damage, tracing planted values through handoffs and writes |
| `constraints.py` | Constraint generation from the earlier scope (not used in the current study) |
| `cli.py` | Command-line entry point (`list`, `run`, `baseline`) |

`tests/test_offline.py` checks the harness without a model. Pilot records are in
`runs/colab_a100/`.
