# Regating-lead digest, 2026-10-28: E38 Amendment 4 done; the next step is yours

The task is done: the change-matched restore you chose (option (a)) ran, was checked and is in the log. What comes
next waits on one decision card. The switched-on set is not the set whose outputs moved most. Yet putting it back takes
back about a fifth of the gain on emoticon and ` which`. Same-size sets of the biggest movers take back about none.

## What ran

- **E38 Amendment 4**, one cluster debug job (40041, L40, 63 s, $0). Plan `reports/plans/2026-10-28_plan_e38-amend4.md`,
  predictions filed before the code, a fresh code review (nothing material), the run, then a write-up and an
  independent check in parallel.
- **The answer.** The filed reading is "specific", as predicted, and the check agrees with every verdict. On ` which`
  the switched-on set takes back 0.18 to 0.20 of the gain. Sets of the biggest movers, the same size per layer, take
  back 0.04 or less. On emoticon the margin is inflated by already-active pieces that push against the target.
- **The surprise.** Those biggest movers carry 3 to 4 times the switched-on set's output change. The set is only 5 to 11%
  of them, against a filed expectation of at least half. Emoticon's negative share sits mostly in three layer-3 MLP
  pieces.
- **Files.** Report `reports/analyses/2026-10-28_analysis_e38-amend4.md`, check
  `reports/reviews/2026-10-28_review_e38-amend4-check.md`, notebook `notebooks/E38_recruitment_test.ipynb` (section
  "Amendment 4"), log v72 (E38 card rewritten, Q5's checked claim).

## Pending decision

- [`decision-e38-next`](https://claude.ai/artifact/SvgkzUiESRkyE9GM65HCWu#decision-e38-next): does the set stand out because
  its changes point toward the target? (a) A direction-matched restore, ranked without Γ, $0, recommended. (b) The
  upstream step. (c) Stop the 67M re-gating line or carry it to Qwen.

## Money, problems, friction

- Spend: $0 (cluster only, no pods).
- Problem 3, open: the friction log refuses new notes, because pasted `/usage` output sits in `~/commute-data/friction`.
- Friction, unrecorded for that reason:
  - `make test` is never green, because two tests outside E38 fail.
  - The 64G memory request waited on the node's free RAM; now 32G.
  - The log's `--check` cannot read Artifact reads saved on `/ephemeral` from the worker.
  - A read with `paths` alone does not count as viewing the page, so the first publish is refused.
- Nothing in `~/commute-data/blocked.txt` from this session. No key or bound was touched.
