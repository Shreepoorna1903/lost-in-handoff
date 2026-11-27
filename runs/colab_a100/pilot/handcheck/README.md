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
the reviewer's reading in all three groups below. He also read the transcripts of sample runs 2 and 14
in full (one per group A and B):
- **Run 2:** noticed but not named by value. The verifier rejected the planted keyboard as "linear
  switches instead of clicky" without quoting its ID.
- **Run 14:** corrected but not demonstrably noticed. The researcher repeated the order lookup and
  used the true ID, with no sign in its visible output that it saw a mismatch. The correction may
  have been incidental.

**Limitation: reasoning is not recorded.** Qwen3 runs with thinking on, but the harness saves only
each agent's visible reply. In run 14 the researcher's seven tool calls carry no visible text at
all. Every "did the agent notice?" judgement here rests on visible output only. Saving the reasoning
text in the run records would let the full study judge this better.

**Result 1: the code matches its definitions.** All 9 fields (requeried, traced, verifier_named,
refused, detected, wrong_value_write, blocks, over_blocking, lost_correct_write) matched on all 20 runs:
0 mismatches.

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

**For the report:** the metrics are validated against their operational definitions; that does not
establish that `detected` measures agent awareness. Keep the two conclusions separate.

Other observations from the reviewer, which matter for the report:
- verifiers sometimes gave nonsensical reasons, or blocked identical writes after allowing one
- executors made up IDs and payment methods
- final messages sometimes claim actions that never happened
- executors re-ran returns that had already succeeded
