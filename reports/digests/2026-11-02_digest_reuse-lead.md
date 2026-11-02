# Reuse-lead digest, 2026-11-02

**Finished because the one thing left waits on the user: whether to run VPD's 56-hour decomposition on the counting set.**
A barely trained decomposition already labels all 21 fine-tunes correctly under the filed rule, so a perfect score there
says little. We recommend dropping the run and scoring VPD on the 67M and Qwen fine-tunes, whose decompositions exist.

**Pending decision.** The log's card `decision-e55-c2-tie`; report `reports/decisions/2026-11-02_decision_e55-c2-untrained-tie.md`; ask 6.

**What ran, $0.** Job 48413 read all 43 states on the 400-step decomposition: 21 of 21 right, the positive control "none", and
at least 0.68 of every gain in the decomposed matrices. Job 48441 tested Fable's explanation, that the gap follows the answer
token, on four fine-tunes: it follows the fine-tune instead, cause not established.

**What changed.** Two independent reviews and one re-review (`reports/reviews/2026-11-02_review_e55-c2-gates.md`). The
importance function's input is now cut at `=`, so it cannot read the answer on the scored cells; fixes pushed (68550dc). Four small re-review fixes are unapplied,
since they matter only if the run goes ahead; they are in the handoff note.

**Money.** $0; no pods; no live Slurm jobs of this lead.

**Friction.** Items 291, 293-295, 299 and the latest, in `fleet/friction reuse-lead`. Nothing in `~/commute-data/blocked.txt` was checked by this digest.
