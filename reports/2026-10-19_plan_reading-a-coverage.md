# Plan: reading-A coverage round (pull-back states on every ladder task, VPD's checks in full)

*2026-10-19, coding agent. Status: **confirmed by the user** (additions 2 and 3 accepted; the block floor kept because it
costs nothing: "not convinced by 1 but do if it doesn't slow things down"). Code: `experiments/reading_a_coverage/run.py`;
predictions filed in its `results/predictions.json`. Codex adversarial review run on the diff: two findings, both fixed
(the trigger floor's denominator is now the mean of the two halves; a failed gate G1 returns no verdicts). Nothing
skipped. The sparsity loss itself (`faithfulness.sparsity_loss`, `vpd.sparsity_config`) was written by the parallel
key-value-task session the same afternoon and is called, not copied. Run handed to a pod-runner agent on RunPod.
Source: `reports/2026-10-19_decision_success-criterion.md` ("Not run"), `backlog.md` "Reading-A coverage gaps".*

## Design check (`/design`)

**Goal served.** GOALS's weak version as written (reading A, `decisions.md` 2026-10-19): D' = Φ(D, δW) passes VPD's
own checks on W + δW about as well as D on W, with Δ not absorbing δW. The decision report records that this has been
checked on the Adam pull-back for emoticon and two of six ladder tasks only, that the rare-word task `r2_t359` read PGD
1.53 against round 2's 1.5 line, and that VPD's importance-minimality (sparsity) loss, one of the three losses GOALS
names, has never been computed on any updated decomposition. This round closes those gaps. It is coverage of an
existing claim, not a step towards the goal; it decides whether "the pull-back is faithful under reading A" can be said
of every rung-0–2 task or only of common-word tasks.

**Nearest thing tried.** TRIED #6 (task ladder): the same states, scored on the task-local harness and the route
split; Pile ratios on two tasks. TRIED #4 (concentration) and #7 (LoRA commute): round 2's Pile harness with a 3-seed
W₀ floor on emoticon only. New here: the four unmeasured tasks, a seed repeat on the one failing task, the sparsity
loss, and the first state-level Pile PGD floor (the LoRA analysis lists it as never having existed).

**Patch or step.** A patch that completes the evidence for the success-criterion decision.

**Decisions relied on.** Γ frozen; thresholds against our own recomputation on the same rows; every ratio with a
measured floor; reading A. No departure. The task-local metrics are not rerun (the ladder's stand, tied to the
regenerated states by gate G1).

**What changes next.** All pass and the 1.53 is seed noise: reading A holds for the Adam pull-back on seven tasks,
and the open question moves to accumulation (the decision report's step-vs-run addendum) or to rungs 3–4. Rung-2
tasks fail: the first reading-A evidence for the "post-training teaches something new" branch, on tasks the model
nearly lacks. Sparsity fails where the KL ratios pass: the pull-back buys reconstruction with density, which none of
the KL checks see; then the sparsity loss joins the pass rule for every later round.

## Apparatus

- **States.** Regenerated from the saved `windows_<task>.pt` (local, 263 KB each; the cluster holds pull-back states
  for emoticon only). Per task: arm-A fine-tune to step 10 (the do-nothing reference, δW in Δ) and the Adam pull-back
  on all alive (U, V) to step 10 (`pullback.train`, lr 1e-5, as the ladder). Tasks: `r0_t352`, `r1_t534`, `r1_t588`,
  `r2_t275` at seed 0; `r2_t359` at seeds 0, 1, 2. The seed is the pull-back's batch order only (free offsets are
  zero-initialised); the adversary seed stays 0. States saved to `results/large/` and downloaded as each task ends.
- **Pile harness.** Round 2's (`faithfulness.pile_faithfulness`: val rows 0..255 in two 128-row blocks, one shared
  PGD adversary per block, 20 steps of 0.1). Four ratios to W₀ on the same rows: stochastic, PGD, CI-masked, unmasked.
- **Sparsity loss.** VPD's importance-minimality loss on the same rows, at the released run's end-of-training setting
  (`artifacts/vpd/s-55ea3f9b/final_config.yaml`: p annealed 2.0 → 0.4 by the end, so p = 0.4; beta 0.5; eps 1e-12),
  on the upper-leaky CI values (`compute_ci` keeps the lower-leaky ones; `CI` grows one field). Ratio to W₀ on
  identical rows and batching (the log₂ term depends on the batch sum).
- **Floors, W₀, same card, same job.** PGD and stochastic: adversary seeds 0, 1, 2 (the ladder's L40 spread was
  3.5 %; never reused across cards). CI-masked, unmasked and the sparsity loss are deterministic under Γ frozen (spread
  exactly 1.0 in `noise.json`), so their floor is block 0 vs block 1 of the same 256 rows at W₀. "Within the floor" =
  ratio ≤ 1 + 3f. State-level floor: the three `r2_t359` pull-backs.
- **Sparsity loss positive control.** On Pile rows the loss is as blind to the trigger as the KL ratios (~3 trigger
  positions in 131k). Its power is shown on the held-out task windows at the trigger, pull-back and do-nothing vs W₀,
  where the ladder counted 150–200 subcomponents switched on. Seconds of compute. Flagged as a design question.
- **Reproduction gate G1.** Regenerated `r2_t359` seed 0 reproduces the ladder: held-out P(target) 0.997 and
  kl_other 0.188 at step 10 within 0.01, and PGD ratio 1.53 within 25 % (the LoRA round's 4090-vs-L40 tolerance).
  Failing G1 stops the scoring: the states are not the ladder's.

## Numbers and what each should read

| Number | W₀ | do-nothing | pull-back, rungs 0–1 | pull-back, rung 2 | control |
|---|---|---|---|---|---|
| PGD ratio | 1.00 (floor f ≈ 0.03–0.04) | 1.4–2.5 (round 2: 2.18; LoRA round: 1.39) | ≤ 1.2 (`r0_t588` read 1.09) | ≥ 1.3 (`r2_t359` 1.53; `r2_t275` has the largest off-target kl_other on the ladder, 0.51) | do-nothing above pull-back |
| stochastic, CI-masked, unmasked ratios | 1.00 | 1.0–1.2 | 0.98–1.05 | 1.0–1.12 | — |
| sparsity ratio, Pile | 1.00 (block floor) | 1.00 ± floor | 1.00 ± floor | 1.00 ± floor | — |
| sparsity ratio, held-out windows at the trigger | 1.00 | > 1 + floor | > 1 + floor | > 1 + floor | shows the metric has power |
| `r2_t359` PGD over seeds 0–2 | — | — | — | three above 1.5: fails reading A on this task; three below: the 1.53 was noise; split: mean and spread reported, no verdict | — |

Pass rule per state (reading A): each of the four KL ratios within W₀ + 3 floors, PGD also under round 2's 1.5 line
(reported beside), sparsity ratio on Pile within its floor. Predictions get filed in `predictions.json` only after
confirmation.

## Compute

Cluster at 2026-10-19 13:00: 8/8 L40s allocated (one 6-GPU job with 22 h left, two 1-GPU jobs with 6 h and 11 h
left), higher-priority 1-GPU jobs pending. Blocked for the day, so RunPod via remote-kernels, one GPU, L40S preferred
over the 4090 to match the ladder's card (`gpu-type-ids` order; low-level). Work: 10 fine-tune/pull-back regenerations
(~30 s each), W₀ floor 3 × 2 min, 7 pull-back + 5 do-nothing Pile passes at ~2 min each (56 s per block on the L40),
set-up ~15 min (3.2 GB artifacts from wandb, `uv sync`). About one GPU-hour, under $2. First Pile pass timed as the
slice benchmark. `windows_*.pt` are gitignored: copied explicitly with sha256 checked on both sides; every result file
downloaded as it is written; every scored number printed to the log.

## Layout

`experiments/reading_a_coverage/` (`run.py` ~300 lines calling `src/commute/{finetune,pullback,faithfulness}`, `plot.py`,
`results/predictions.json`); the sparsity loss lives in `src/commute/faithfulness.py` beside the four ratios. The
ladder's `run.py` stays as it ran.

## Not run (by design)

Task-local harness and route split (the ladder's stand); rungs 3–4; emoticon (already covered); the lift; any
Γ update; harness-seed floors on states other than W₀ (two extra passes if wanted).
