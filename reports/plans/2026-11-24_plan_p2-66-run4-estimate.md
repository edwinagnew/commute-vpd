# P2.66 run 4: estimate for bringing the realistic panel to 12 tasks on every reading (2026-11-24)

Read-only estimate for regating-lead (request: `~/commute-data/agents/regating-lead/note-2026-11-24-whip-panel-n12.md`).
Nothing run. The floor is on the **read-row gain** (`gain` in `results/wide/read/*.json`, from `RR.share_of`), not on
`held_gain`. For the continued-text tasks the two agree within about 0.05. For sentiment they do not: held 9.3-9.7, read 0.09-0.22.

## 1. What is saved, and the gain at each saved step
Every state is on disk in bf16 at `experiments/convergence_panel/results/large/states/<task>_s<seed>_step<N>.pt`
(a symlink into `~/commute-data/results-large/convergence_panel`, 57 MB each), at the rung crossings up to step 300 (the run's last step).
Nothing is only on W&B and nothing was lost with a pod. Held gain at every save is in `results/train/<task>.json`.
The read gain was measured only at the picked states (`results/wide/read`).

| task, seed | best saved gain (step) | earliest saved step with read gain >= 1 |
|---|---|---|
| arxiv s0/s1/s2 | held 0.51 (111), 0.53 (300), 0.57 (300); flat from step ~110 to 300 | **none**, within 300 steps |
| german s0/s1/s2 | read 0.69 / 0.78 / 0.56 at step 300 (held 0.73 / 0.81 / 0.62) | **none** |
| spanish s0 | read 0.96 at step 300 (held 0.91) | **none** (s1 1.07 and s2 1.08 at step 300 already pass and stay) |
| sentiment s0 | read 2.46 at step 108 (KL 2.5) | 108 measured; step 50 never read |
| sentiment s1 | read 1.54 at step 21 (KL 0.95) | 21 measured; steps 8 and 9 never read |
| sentiment s2 | read 1.18 at step 119 (KL 4.3); step 17 reads 0.01 | 119 measured; step 45 never read |

Sentiment's read gain does not rise steadily (s0 reads 0.11 at step 20, s2 reads 0.01 at step 17), so "earliest" requires reading the gain at
steps 50, 8, 9 and 45 first: 4 x ~8 s. Every sentiment state that passes sits at KL 1-4 nats, against the 0.18 rung.

## 2. Training on (arxiv, german, spanish s0)
Recipe: `run.py train TASK` -> `panel.train_ladder`. Adam at a constant learning rate of 3e-5, no warm-up, batch 8, clip 1, 300 steps (one pass of 2,400 rows), on
cluster L40 debug jobs 63882/63883/63886. Measured: 63-68 s per seed for 300 steps with saves (~0.21 s/step),
peak 8.4 GB. **No optimizer state is saved**, so training on from a saved state with its Adam moments is impossible.
Retraining from scratch with more steps is possible, and it replays steps 1-300 exactly. The learning rate is
constant, and `panel.order` draws the first pass's permutation identically, so the saved step-300 file is the control.
Code needed (the lead's, not run here): `steps` is frozen at 300 in `predictions.json` settings, and the ladder stops once
every rung is crossed and saves only at crossings. Run 4 needs a filed step cap, saves every ~25 steps, the read
gain at each save, a new pick label (e.g. `gain1`) in `selection` / `RUN.picked` / `eb.MATCHES`, and a scorer for the run-4 set.
**Unmeasured, the main risk:** whether the gain reaches 1 in passes 2+ (repeated rows). Linear extrapolation of the late slope
puts spanish s0 near step 330, german s0/s1 near 600-700, and german s2 and arxiv at "maybe never": they were flat from step
~110 to 300. A $0 cluster probe (all 7 seeds to ~1,200 steps, ~4.5 min a seed, two debug jobs) settles this before any spend.
If arxiv or german s2 stall, the only routes are a recipe change for those seeds (a departure, the user's call) or n = 11-12 with arxiv dropped.

## 3. Readings, code, card, time per state
- Re-gating share, steering, effective rank (and ADL): `run.py read TASK` -> `readings()`. About 16 s per state on E272's pods.
  Logs are in `results/read_logs/e272_logs_{a..d}.tgz`. The read JSONs record no card; the pods carried E272's H100 pod labels a-d.
  On the cluster L40, 40 s per state (job 63920).
- Prior: `rd.w0["rows"]` is W0's log-odds on the read rows. It **does not depend on the state**, so no new read is needed.
- Kept share at VPD (E_b over K): `eb.py TASK 0/1 M wide`, with Gaussians for seeds 0,1,2 (`EB_RANDOM_SEEDS=0,1,2`, P2.78's
  `random_file` seeds). **H100 SXM only**, to match E272/E302: the stage-0 card gate, plus the SVD-sign card dependence (handoff).
  Measured 744 s per state (E302, `run_e302_d_german.log`), stage 0 37 min per pod (`run_e302_*_stage0.log`).
  Each new Gaussian also needs its filed expectation (P2.78 used `predictions_p278.json` with `P278_SHA`).

## 4. Which cells need a new read
The other 8 tasks (category, conll, french, lower, squad, summary, swahili, yoruba) and spanish s1/s2 keep their states.
**No re-read** for them; their values carry over unchanged. New reads:
- share, steering, erank: arxiv s0-2, german s0-2, spanish s0 (7 new states) plus the seed-0 Gaussian of arxiv, german and
  spanish (3). Sentiment: none, if the picks are 108/21/119, which `wide/read/sentiment.json` already holds. Otherwise its new picks.
- E_b: the same 7 states and sentiment's 3 picks (10 trained) plus one Gaussian each (10) = **20 E_b items**.
- prior: none.

## 5. Estimate
| stage | items x measured time | card | cost |
|---|---|---|---|
| train-on probe + retrain 7 seeds, read gain per save | 7 x ~5 min | cluster L40 debug | $0 |
| sentiment gain at 4 unread steps | 4 x 8 s | cluster | $0 |
| share/steering/erank | 10 x 16 s + ~2 min load x 3 tasks ≈ 10 min | H100 SXM (on an E_b pod) | in the row below |
| E_b over K | 20 x 744 s = 4.1 GPU-h, + 37 min stage 0 per pod | H100 SXM, 4 pods x 5 items | 4 pods x ~2.0 h = 8 pod-h: **$22** at $2.7/h; **~$37** at E302's all-in ($42.89 for 4 pods) |
Transfer: 10 states x 57 MB = 570 MB up ≈ 1 min; 10 Gaussians (570 MB) back ≈ 1 min.
Wall time: code ~2 h, cluster probe and retrain ~1 h, pods ~2-2.5 h, scoring ~0.5 h. That is **~6 h from start: it lands 11-25, well before 11-26**,
if the probe shows arxiv and german reach 1. **Blockers:** (a) arxiv, and probably german s2, may not reach 1 under the filed recipe; this is
unmeasured until the $0 probe. (b) A new pick rule ("earliest save with read gain >= 1") and a run-4 scorer must be filed before reading.
(c) Every state that crosses the floor is far off the KL match: sentiment at 1-4 nats, and arxiv/german past 0.19. The figure must say so. (d) E_b runs on an H100 SXM only.
