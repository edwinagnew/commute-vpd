# Plan: analyse the SGD-review runs (hand-off to a fresh session)

*2026-10-21, lead session. Seven experiments from `reports/2026-10-21_review_sgd-finding-influence.md` ran on the
MATS cluster ($0). The lead and its subagents ran them and nothing is analysed yet: this file is the brief for
the analysis session. Vocabulary: `GOALS.md` → Terms. Write-ups follow `/writeup`.*

## The task for the new session

Write one analysis report, `reports/2026-10-2x_analysis_sgd-vs-adam.md`, covering the runs below, with figures (one per
claim, reference on the same axes), filed predictions scored as filed, and a "Not run" list. Then a decision report
if the user agrees the main line moves (see "The headline"), and the research-log flush (`docs/research_log/PENDING.md`
holds one line per run; recipe in `docs/research_log/README.md`; read the live artifact first). Update
`experiments/TRIED.md`. Every number below is copied from the runners' reports; re-read it from the JSONs before using it.

## The headline, as it stands (to verify)

The adversarial failures that the current one-sentence claim rests on ("no map with Γ frozen keeps the decomposition
adversarially faithful", `reports/2026-10-21_decision_regating-main-line.md`) look like properties of particular
updates (their rate, step and trajectory), not of the maps or the decomposition:

1. **Fine-tune side (sgd_finetune, rate/split/adam_matched).** On emoticon at step 10, a small fine-tune passes
   under every map, whether SGD (lr 0.01) or Adam at a matched size (lr 3e-6): do-nothing 1.12 / 1.08 and least squares
   1.16 / 1.12, against 2.18 and 5.41 for the full-size Adam fine-tune (lr 3e-5, about 4.7x larger). Filed P2 reversed;
   amendment A1 and A2 read "size of change carries it".
2. **Component training at step 10 (sgd_ladder, 3a).** SGD passes r0_t352 (0.95) and r1_t534 (1.10), where E11's
   Adam 1e-5 states fail (2.70, 2.24); r2_t275 is inconclusive (1.50 against 6.83).
3. **Component training at steps 100 and 1000 (sgd_ladder_long, 3b). This is the surprise.** On r0 and r1, **Adam 1e-5,
   which failed at step 10, passes at 100 and 1000** (1.00/1.07 and 1.01/0.95), with P(target) 0.92–0.99. Adam 3e-6
   learns by step 100 and passes. **SGD passes at 100 but fails at 1000** (2.82, 3.00) at about the same size of change
   as Adam 1e-5 at 1000 (rel_norm_max 0.017 against 0.018). Do-nothing drifts towards failing by 1000. Filed Q1 (SGD
   stays under the line at 100 and 1000) reads **no**. But the decision's own move-back bullet (a map at the floor with
   the behaviour at steps 100 and 1000) **is met, by Adam component training** on r0 and r1, and on emoticon by every
   arm. That decision is the user's.
4. **Key-to-value (sgd_kv, item 4).** E10's rand32 failure (Adam 1e-5, 2.81) is a rate artefact: Adam 3.3e-6 (1.09) and
   1e-6 (1.05) learn it and pass. SGD fails rand32 (4.17) and ident32 (2.37, where Adam passed). The filed Q1 reads
   "yes", but it is a rate artefact, not an Adam one.
5. **Shape (sgd_matched, item 2).** At a matched size on emoticon, Adam CT is about 7x more diffuse than SGD (top-10
   share ratio 0.15) with about 3.8x the damage elsewhere; momentum keeps SGD's concentration with 5.2x the damage.

So neither "SGD is cleaner" (the user's hunch) nor "size alone" explains everything. Step 10 favours SGD; steps 100 to
1000 favour Adam; Adam 1e-5's step-10 failure is a transient (KL elsewhere 0.20 at step 10, 0.03 at step 100). Before
any claim: read rel_norm_max and KL elsewhere for every state on one figure against the adversarial ratio, and check
whether "transient damage" or "trajectory" separates the passes from the fails better than optimiser or size.

## Runs, files, job IDs

| Item | Experiment dir | Jobs | Results (local) | Plan |
|---|---|---|---|---|
| 1 faithfulness of the SGD emoticon state | `experiments/sgd_faith` | 28182 (smoke), 28188 | `results/sgd_faith.json` | `2026-10-21_plan_sgd-faithfulness.md` |
| 2 matched size, momentum | `experiments/sgd_matched` | 28189, 28203 (amend1) | `results/sgd_matched.json`, `sgd_matched_amend1.json` | `2026-10-21_plan_sgd-matched.md` |
| 3a ladder, step 10 | `experiments/sgd_ladder` | 28199, 28224–28228 | `results/sgd_ladder_shard{0..5}.json`, `figures/ladder.png` | `2026-10-21_plan_sgd-ladder.md` |
| 3b ladder, steps 100/1000 | `experiments/sgd_ladder_long` | 28259, 28266, 28267; 28268 cancelled (r2 stalled), rerun 28285 | `results/sgd_ladder_long_{emoticon,r0_t352_v310,r1_t534_v369,r2_t275_v2667}.json`; no plot.py yet | `2026-10-21_plan_sgd-ladder-long.md` |
| 4 key-to-value | `experiments/sgd_kv` | 28205, 28244, 28245 | `results/sgd_kv_shard{0..2}.json`, `sgd_kv.json`, `figures/kv.png` | `2026-10-21_plan_sgd-kv.md` |
| 5/7 SGD fine-tune, splits | `experiments/sgd_finetune` | 28193, 28198, 28249 (adam_matched) | `results/rate.json`, `split.json`, `adam_matched.json` | `2026-10-21_plan_sgd-finetune.md` |

Logs: `cluster/logs/commute-<name>-<job>.out`. Large states stay on NFS under each `results/large/`.

## Filed predictions, scored as the runners reported (verify)

- sgd_faith: P1 pass, P3 pass, P2 miss (0.71 vs ≥ 0.8), P4 miss (d 0.072 vs > 0.1).
- sgd_matched: P5 pass, P6 miss (momentum KL 5.2x); amend1 P1, P2 pass, P3 miss (gain 1.08), P4 miss (KL 3.8x).
- sgd_ladder: Q1 → "3b follows"; P2, P3 per shard in the JSONs.
- sgd_ladder_long: Q1 no on r0 and r1; Q2: Adam 3e-6 first learns at step 100 and passes on all three tasks.
- sgd_kv: Q1 yes (rate artefact); P2, P3 true on all three tasks.
- sgd_finetune: R1, P1, P3, P4, P5 pass; P2 reversed; A1, A2 pass ("size carries it"); A3 between.

**r2_t275 (rerun job 28285, `results/sgd_ladder_long_r2_t275_v2667.json`) does not follow r0/r1.** SGD fails hard
(5.20 at step 100, 38.0 at 1000). Adam sits in the L40's inconclusive band (1.34–1.51), and its P(target) falls below 0.9:
1e-5 reads 0.88 at step 100 and 0.78 at 1000; 3e-6 reads 0.92 and 0.77. Do-nothing reads 3.95 at step 100 and 1.55 at
1000, with P 0.85 and 0.82. So on the hardest task no arm passes cleanly. An H100 or more adversary seeds would be
needed to read Adam's 1.3–1.5.

## Open issues to resolve in the analysis

- Job 28268's "bases and input moments" step took 2772 s against 9 s; the rerun took 10 s. Cause unknown (likely the
  node), not reproduced.
- 3a's log prints a P(target) slightly different from the JSON's held-out P for some arms (0.206 vs 0.197); find which
  quantity each is.
- sgd_kv recorded no size of change side by side; read rel_norm_max from the saved states before claiming size explains it.
- sgd_kv's rand32 KL limit (1.39 nats) barely constrained the SGD rate.
- sgd_finetune's P5 (cosine) was added by its writer and has no positive control: report as exploratory.
- `faith_score` duplicates `sgd_faith`'s `check_w0`/`trigger_ratios`: make sgd_faith call it only if the move is pure
  (sgd_faith has run; otherwise leave it).
- The cluster has stale `src/commute/{lift,pullback}.py` (sync never deletes); nothing imports them.
- Process: smoke rule changed (`.claude/CLAUDE.md`, the scored run is its own smoke); no local model loading
  (an OOM crashed the session).

## Not run

Items 6 (E17 statistics on SGD states, now possible from sgd_ladder's and sgd_ladder_long's saved states), 8 and 9 of
the review; a second seed of anything; an H100 re-read of the near-line verdicts (r2_t275 at 1.50).

## Verification against the JSONs (2026-10-21, book-keeping session)

Every number above was re-read from the result JSONs (read-only agent). Pass lines: `faith_score.verdict` against
the same-job floor f = 0.2088, so PGD pass ≤ 1.241, fail ≥ 1.813, inconclusive between; sgd_faith's P1 alone uses
≤ 1.5. All numbers match except:

- **sgd_ladder_long Q2 is wrong above.** Adam 3e-6 first learns at step 100 and passes on r0 and r1 only; on
  r2_t275 it reads 1.505, inconclusive (`score.adam_3e-06_first_learned`). PENDING's 3b line predates the r2 rerun.
- sgd_matched amend1 P3: gain 1.07 (`ratios.adam.gain` 1.0746), not 1.08; still a miss.
- "Met on emoticon by every arm" holds for SGD, Adam 1e-5 and Adam 3e-6; do-nothing at step 1000 reads 1.39,
  inconclusive.
- r2_t275 Adam: only 3e-6 at step 100 is scored inconclusive on PGD; the other three states are "below P 0.9" and
  unscored on PGD.
- 3a: Adam 1e-5 on r2_t275 (6.83) and r2_t359 (1.68) are "did not learn" (KL over the limit), not fails.
- SGD r1 at step 100 passes at 1.237, 0.004 under the line: near-line.

Open issues resolved or changed:
- **sgd_kv already records rel_norm_max** (`tasks.*.arms.*.rel_norm_max`), and size of change does not explain it:
  ident32 SGD fails at 0.0049 where Adam 1e-5 passes at 0.0118; rand32 SGD fails at 0.0188, Adam 3.33e-6 passes at
  0.0146, Adam 1e-5 fails at 0.0154. So "size carries it" (sgd_finetune A1/A2) does not carry over to the cipher.
- sgd_kv's "learned" line is P ≥ 0.5, not 0.9; on rand32 every arm's P is 0.85–0.91, and ident32's Adam 3.33e-6
  was scored at P 0.615.
- 3a's 0.206 vs 0.197: both are held-out mean P(target) on the same windows, by two code paths: `heldout.p_o` from
  `eval_arm_a` with hook offsets (`src/commute/component_training.py:149`), and `trigger.target_p_o` from
  `route_cells` with W₀ + δW installed (`src/commute/harness.py:119-122`). Agree within 0.005 on other arms; cause
  of the larger gap at low P unconfirmed (likely bf16 numerics of the two routes).
- sgd_finetune P1's ratio (0.1573) and A3's outside-both ratio (0.1569) are different quantities; PENDING's 0.157 is A3.
- Job 28268's 2772 s bases step is confirmed in the log (9–10 s in every other job).
