# Hand-check of the detection and over-blocking metrics (20 pilot runs)

**Sample** (`sample.json`): 20 runs from `runs/colab_a100/pilot/qwen3-8b`, chosen to cover every fault
(F1–F4), defenses D0/D2/D4, both outcomes of each metric and the borderline cases: verifier blocks that
do / do not name the planted ID, escalation after a database error, re-queries (including the F3
control), the one D4 run that called the lookup tool, and the two clean-D2 over-blocking cases.

**Procedure.** An independent LLM reviewer labelled each run from the raw records and plain-English
definitions only (no access to `src/`, the tests, `summary.csv` or the metric outputs), giving
(a) rule-based labels and (b) its own judgement of whether an agent actually noticed the planted fact,
with quoted evidence (`reviewer_labels.json`). The metric outputs (`metric_labels.json`) were then
compared field by field. A team member (Puneet) reviewed the summarised disagreements and agreed with
the reviewer's reading in all three groups below. He also read key excerpts of five runs, extracted from the raw records: the handoff to the executor, the executor's write attempts and replies, the verifier's messages, and for run 14 the tool log. These were not the full transcripts: runs 2 and 14 (one each for groups A and B) and runs 1, 5
and 20, where the metric and the reviewer agreed. He agreed with the metric on all three:
- **Run 1:** noticed. The verifier named the false ID, blocked the return, and nothing wrong was
  written.
- **Run 5:** not detected. The executor wrote the planted keyboard ID.
- **Run 20:** a correct write was wrongly blocked and lost.

For runs 2 and 14:
- **Run 2:** noticed but not named by value. The verifier rejected the planted keyboard as "linear
  switches instead of clicky" without quoting its ID.
- **Run 14:** corrected but not demonstrably noticed. The researcher repeated the order lookup and
  used the true ID, with no sign in its visible output that it saw a mismatch. The correction may
  have been incidental.

**Limitation: reasoning is not recorded.** Qwen3 runs with thinking on, but the harness saves only
each agent's visible reply. In run 14 the researcher's seven tool calls carry no visible text at
all. Every "did the agent notice?" judgement here rests on visible output only. Saving the reasoning
text in the run records would let the full study judge this better.

**Two passes over the same 20 runs.**
- **Pass 1** covered detection and over-blocking (`reviewer_labels.json`, `metric_labels.json`).
- **Pass 2** covered the RQ1 metrics: success, fault_fired, passed_on, the clean run's success, and
  fault-induced failure (`reviewer_labels_rq1.json`, `metric_labels_rq1.json`). `sample.json` lists
  the clean run used for each fault run.

**Result 1: the code matches its definitions.** All 9 fields (requeried, traced, verifier_named,
refused, detected, wrong_value_write, blocks, over_blocking, lost_correct_write) matched on all 20 runs:
0 mismatches.

Pass 2 matched on 99 of 100 labels. The one mismatch exposed a real difference between the code and
FP2's definition of `passed_on`:
- The old code was case-sensitive. In sample run 16 the planted "pending" appears as "(Pending)".
- It checked every handoff, the intake's included.
- It did not exclude values already present in the clean run's handoff.

`passed_on` now follows FP2: the researcher's handoff to the executor, case-insensitive, excluding
targets already in the clean handoff, F1–F3 only. Pilot effect: F1 moves by one run per defense
(D0 5/10 → 4/9, D2 5/10 → 6/9, D4 5/10 → 4/9), F2-D4 goes 9/10 → 8/9, F2-D0/D2 and F3 are unchanged.
F4 is no longer reported, because its handoff is edited, so the value is in it by construction.

**Result 2: `detected` is a weak measure of awareness.** Against the common-sense judgement, `detected`
agreed on 9 of 18 fault runs. The sample was chosen toward borderline cases, so this is agreement on
hard cases, not a typical rate for the pilot. The disagreements fall into three groups:

| Group | Runs (sample id) | What happens | Decision |
|---|---|---|---|
| A. Habitual re-lookups | 14, 15, 18 | the same lookup is repeated with no mention of the value or a mismatch; the model repeats lookups in clean runs too (18 is the F3 control) | keep re-query in `detected` (FP2 definition); report `detected_without_requery` next to it, and the F3 control's re-query rate as context (1–2 runs of 10, too few to subtract) |
| B. Verifier objects by description | 2, 4 | e.g. "linear switches instead of clicky": clearly the planted item, but the ID is not quoted | report `verifier_named` (lower bound) and `verifier_blocked_planted` (upper bound: blocked a write carrying the planted value). The upper bound is evidence of blocking, not of understanding |
| C. Noticed after a database error or after the damage | 8, 10, 11, 17 | the database rejected the planted ID first, or the wrong write had already succeeded | excluded by design: environmental protection, or too late to be preventive. Documented as a limitation |

Pilot numbers for A: every F1–F3 detection comes from a re-query alone (`detected_without_requery`
is 0 in all F1–F3 conditions); F4-D2 stays at 6/9 without re-queries.

Open question for the team: in runs 14 and 15 the re-query returned the true value and the agents
used it, so the correct write went through. The fault was corrected without any sign of being
noticed. "Did the true value win after a re-query?" may matter more for a propagation study than
awareness.

Pilot numbers for B (the upper bound is meaningful mainly for F4; on F2 it counts blocks given for
unrelated reasons): F4-D2 named 6/9, blocked-planted 9/9; F2-D2 named 0/10, blocked-planted 2/10
(the F2 blocks gave unrelated reasons).

**Finding for RQ2 (from run 20).** The verifier judges each write call on its own. When the executor
sends several writes in one turn (two cancels and a return), the verifier cannot see the others. It
rejected a correct cancel because "the original request was to cancel all pending orders". The
executor then reported both cancels as done. So per-write checking loses context on multi-action
requests and can block writes that are only correct as part of the batch.

**For the report:** the metrics are validated against their operational definitions; that does not
establish that `detected` measures agent awareness. Keep the two conclusions separate.

Other observations from the reviewer, which matter for the report:
- verifiers sometimes gave nonsensical reasons, or blocked identical writes after allowing one
- executors made up IDs and payment methods
- final messages sometimes claim actions that never happened
- executors re-ran returns that had already succeeded
