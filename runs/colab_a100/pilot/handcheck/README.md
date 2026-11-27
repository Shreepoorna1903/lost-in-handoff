# Hand-check of the detection and over-blocking metrics (20 pilot runs)

**Sample** (`sample.json`): 20 runs from `runs/colab_a100/pilot/qwen3-8b`, chosen to cover every fault
(F1–F4), defenses D0/D2/D4, both outcomes of each metric and the borderline cases: verifier blocks that
do / do not name the planted ID, escalation after a database error, re-queries (including the F3
control), the one D4 run that called the lookup tool, and the two clean-D2 over-blocking cases.

**Procedure.** An independent LLM reviewer labelled each run from the raw records and plain-English
definitions only (no access to `src/`, the tests, `summary.csv` or the metric outputs), giving
(a) rule-based labels and (b) its own judgement of whether an agent actually noticed the planted fact,
with quoted evidence (`reviewer_labels.json`). The metric outputs (`metric_labels.json`) were then
compared field by field. The disagreements were reviewed by a team member (Puneet), who agreed with
the reviewer's reading in all three groups below.

**Result 1: the code matches its definitions.** All 9 fields (requeried, traced, verifier_named,
refused, detected, wrong_value_write, blocks, over_blocking, lost_correct_write) matched on all 20 runs:
0 mismatches.

**Result 2: `detected` is a weak measure of awareness.** Against the common-sense judgement, `detected`
agreed on 9 of 18 fault runs. The disagreements fall into three groups:

| Group | Runs (sample id) | What happens | Decision |
|---|---|---|---|
| A. Habitual re-lookups | 14, 15, 18 | the same lookup is repeated with no mention of the value or a mismatch; the model repeats lookups in clean runs too (18 is the F3 control) | keep re-query in `detected` (FP2 definition); report the F3 control's re-query rate as a baseline |
| B. Verifier objects by description | 2, 4 | e.g. "linear switches instead of clicky": clearly the planted item, but the ID is not quoted | report `verifier_named` (lower bound) and `verifier_blocked_planted` (upper bound: blocked a write carrying the planted value). The upper bound is evidence of blocking, not of understanding |
| C. Noticed after a database error or after the damage | 8, 10, 11, 17 | the database rejected the planted ID first, or the wrong write had already succeeded | excluded by design: environmental protection, or too late to be preventive. Documented as a limitation |

Pilot numbers for B: F4-D2 named 6/9, blocked-planted 9/9; F2-D2 named 0/10, blocked-planted 2/10
(the F2 blocks gave unrelated reasons).

**For the report:** the metrics are validated against their operational definitions; that does not
establish that `detected` measures agent awareness. Keep the two conclusions separate.

Other observations from the reviewer, which matter for the report:
- verifiers sometimes gave nonsensical reasons, or blocked identical writes after allowing one
- executors made up IDs and payment methods
- final messages sometimes claim actions that never happened
- executors re-ran returns that had already succeeded
