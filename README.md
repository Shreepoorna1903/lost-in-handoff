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
set -a; source example.env; set +a   # optional: Ollama address and parallelism
```

**macOS iCloud folders (Desktop, Documents):** iCloud hides `.pth` files and Python 3.13 skips
them, so keep the venv outside iCloud and symlink it:
`uv venv --python 3.13 ~/.venvs/lost-in-handoff && ln -s ~/.venvs/lost-in-handoff .venv`, then
`VIRTUAL_ENV=$HOME/.venvs/lost-in-handoff uv pip install -e ".[dev]"`.

**GPU runs:** [`colab/lost_in_handoff_colab.ipynb`](colab/lost_in_handoff_colab.ipynb) runs sweeps
on a Colab A100 (4 runs in parallel on one Ollama server).

## Run an experiment

Conditions are named in [`src/handoff/conditions.py`](src/handoff/conditions.py): `single-clean`,
`clean-D0` (pipeline baseline), `clean-D1`…`clean-D5` (no fault, a defense on: over-blocking) and
`F1-D0`…`F4-D5` (fault × defense). The model, sampling settings, prompts, tools, budget and handoff
format are the same in every condition; only the fault and the defense change.

```bash
.venv/bin/python run_eval.py --condition clean-D0 --limit 10             # 10 tasks, 1 trial each
.venv/bin/python run_eval.py --condition F2-D1 --tasks 0,2,11 --runs 3   # trial k uses seed k
.venv/bin/python run_eval.py --condition clean-D0 --limit 10 --shard 1/4 # 4 processes, one Ollama server

.venv/bin/python score_eval.py runs/qwen3-8b                # metric summaries, no model calls
.venv/bin/python score_eval.py runs/colab_a100/baseline     # the Colab pilot
.venv/bin/python -m pytest                                  # offline tests
```

`run_eval.py` saves one JSON record per run in `runs/<model>/<condition>/task<ID>__trial<K>.json`
and resumes where it stopped. A crashed run is saved as an error record and retried next time.
Each record holds the task, condition, trial ID, model, seed, sampling settings, every message,
handoff and tool call (with any injected fault), token usage, timing and the final answer.
`score_eval.py` recomputes success by replaying each run's logged writes against τ²-bench's gold
replay, then prints and saves `summary.csv` per condition. Conditions not built yet are refused
with the name of the person building them (see `PROGRESS.md`).

The Colab notebook only installs things and launches these scripts on a GPU; all experiment code is
in Python scripts. `python -m handoff.cli run --task 0 --topology pipeline --model qwen3:8b --show`
runs a single ticket and prints its handoffs, for debugging.

## Layout

```
run_eval.py            run a condition on the tasks, save one record per run (scores nothing)
score_eval.py          score saved records, write summary.csv (no model calls)
src/handoff/
  conditions.py        every condition by name: fault x defense, and what is held fixed
  metrics.py           FP2 metrics computed from saved records
  env.py tasks.py      the task environment (τ²-bench retail behind ToolEnv)
  agents.py topologies.py handoffs.py   the agentic system
  faults.py runner.py evaluate.py       fault injection, running one ticket, grading
tests/                 offline tests (no model)
colab/                 notebook that runs the scripts on a Colab A100
runs/colab_a100/       saved pilot records
example.env            Ollama settings (no API keys needed)
```

Three rules, as in the course's W10b example: the evaluation imports the same agent code it tests;
running and scoring are separate, so a fixed metric only needs a rescore; and a condition is a name
in `conditions.py`, not a code change.

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
| `conditions.py` | Every condition by name (`CONDITIONS`), what each changes, and which are not built yet |
| `metrics.py` | FP2 metrics from saved records: success (by replay), 95% interval, passed on, wrong-value write, fault-induced failure, cost. Detection and over-blocking are TODO |
| `cli.py` | Debugging commands (`list`, `run --show`, `baseline`) |

`tests/test_offline.py` checks the harness and `tests/test_eval_scripts.py` the conditions and
scoring, all without a model. Pilot records are in `runs/colab_a100/`.
