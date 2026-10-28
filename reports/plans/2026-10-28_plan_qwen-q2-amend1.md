# E45 Amendment 1: anchor 2 trained for 300 steps (2026-10-28)

Plan: `reports/plans/2026-10-28_plan_qwen-q2.md`. This note is the approved branch of its §8 decision tree, "Anchor 2
not learned at step 100 (seed 0) → one amendment: 300 steps, cosine over 300, both seeds; still not → card". It is
inside an approved decision tree, so no new `/design` sign-off is needed.

**Why.** Job 1 row 4 (cluster, L40) trained anchor 2 (invented facts, packed) for 100 steps at seed 0. All its
checks passed, but `score.json` `anchor2_learns_s0` missed: held-out-template exact match on invented facts at step 100
was 0.319, against the filed 0.5; known facts 1.0 against W0's 0.97 (the rule needs 0.873). The training loss had
fallen from 2.60 to 0.22, so the facts were being fit in the training phrasings, but only a third could be recalled
under the held-out phrasing by the time the cosine schedule reached rate 0.

**What.** A new arm `a2long` (`experiments/qwen_updates/job.py` ARMS). It uses anchor 2's data, packing and recipe,
and changes only what §8 names:

| setting | a2 (ran) | a2long |
|---|---|---|
| steps | 100 | 300 |
| schedule | warm-up 10, cosine over 100 | warm-up 10, cosine over 300 |
| data | 27 packed rows per step, `packed_stream(seed)` | the same, 300 × 27 rows; its first 100 × 27 are a2's rows |
| seeds | 0 read; 1 not run | 0 and 1 (seed 1 re-draws names and attributes, as for a2) |
| δW saved | grid brackets (1-6, 11, 12) and 100 | grid brackets, 100 and 300 |

Warm-up stays at 10 steps. The recipe's source is "warm-up ratio 0.1", which at 300 steps would be 30. I kept 10
because §8 changes only the length and the cosine, and with 10 the first 11 steps are a2's steps exactly. Steps 1-11
use the same rates and the same rows, so a2long seed 0 must repeat a2_s0's tracker KL there. That check is the
control (below). The other reading, warm-up 30, is the lead's to choose before row 1 runs. It would drop the control
and move the grid states to lower rates.

Run names are new (`a2long_s0`, `a2long_s1`), so `a2_s0`'s files stay as they ran. `a2_s1` (RUN_BRIEF row 8) is
replaced by `a2long_s1`.

**Readings.** E reads what it reads for a2: KL to start on all own and maths rows at every saved state, the state
chosen per grid value, held-out fact loss, and greedy facts and GSM8K test 0-255 at the chosen states. Steps 100 and
300 are read as well. The scored reading is step 300; step 100 is reported beside it.

**Predictions** (`experiments/qwen_updates/results/predictions_amend1.json`, filed before any a2long run):
- `anchor2_learns`, per seed, at step 300. The thresholds are copied unchanged: invented exact match ≥ 0.5 and known
  ≥ 0.9 × W0's known. The rule's edge cases are run in `tests/test_qwen_updates.py`.
- The `learned` rule, the step-2 band and the step-2 KL ceiling are copied unchanged.
- The control: a2long_s0's 16-row tracker KL at steps 2, 3, 5 and 10 within 5% of a2_s0's, both axes. The tolerance
  is gold's curve tolerance; gold read at most 0.5% off.
- Expected values, not scored: invented above 0.319 at step 100 (the rate there is still 0.8 of peak, not 0); 0.4-0.8
  at step 300; known ≥ 0.95.

**Run.** `experiments/qwen_updates/RUN_BRIEF_amend1.md`. Two T jobs per seed: T writes a ~17 GB checkpoint after
step 150, exits 3 and is resubmitted. Then E, in one or two jobs. Seed 0 is read before seed 1 is trained. About 8
`debug` jobs and 2.5 h; disk is 28.2 GB of δW per run (10 states × 2.82 GB). **Cost $0.**

**Then.** Both seeds hold → anchor 2 is `a2long` for the rest of Q2, and Job 1 rows 5-8 go ahead. Either seed misses
→ card to the user (§8, "still not"). Left open for the lead: whether anchor 1 should also train for 300 steps.
The plan compares anchors at matched KL to start, not at matched steps, but its "decisive point" falls back to step 100.

## Not run
- Nothing has run under this amendment. The checkpoint and resume path has run only on CPU tensors (the test), not on
  a GPU.
