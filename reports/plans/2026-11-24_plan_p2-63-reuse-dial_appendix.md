# Appendix: the reuse dial (P2.63, E###): Gates

Main plan `2026-11-24_plan_p2-63-reuse-dial.md`. Code `experiments/reuse_dial/run.py` (jobs), `stage0.py`,
`score.py`, `plot.py`; shared `src/commute/dial.py`.

## The update, exactly
Per decomposed matrix (24), dW = (g_R - 1) R0 + U^T diag(g - 1) V^T + Res, with U, V the VPD subcomponents
(`vpd.component_weights`), R0 = W0 - U^T V^T (`vpd_gains.Basis`), g one gain per subcomponent (38,912) and g_R one per
matrix remainder, Res a free dense matrix. Loss = task loss + beta x sum ||Res||^2 / sum ||W0||^2. Arms, in dial order
(index d): g (no Res, d = 0), b<beta> for the betas from largest to smallest (d = 1-3), ft (no gains, beta 0, d = 4).
Readouts per arm at its stop: steps, reached, the gain with hooks, the gain of the gains alone (Res removed), the
residual share 1 - (gains alone)/(gain) (0 for g, 1 for ft), the residual's size, the update's size.

## Gate 1: pre-mortem
- **Recipe and hyperparameters.** Gains at 3e-3 may not reach the target in the cap (E231's lowercase gains plateaued at
  1.20 of 1.64 at neurons). Guard: the target is set from the measured gains-only plateau (0.9 x), so gains only
  reaches it by construction on seed 0; a seed or arm that does not is a shorter dial (reported, `readable_task` rule).
  The betas may not spread the residual share. Guard: the rehearsal's `calibrate` measures the share at six decades
  and the three betas are picked by the filed rule (gate 2) before the paid run. The residual (3e-5) and gains (3e-3)
  rates differ by 100x, so beta and the rates jointly set the split; both are fixed across tasks and seeds.
- **Data.** Procedural items, held-out templates and fills are E256's `build_items` with E256's settings (template
  leakage checked by E256 stage 0; train and held split from Pile train and val rows). Lowercase rows are E53's
  (`lowercase.task`, Pile train rows 0-4095), read rows E136's (Pile val). The stop reads the read rows (as E231),
  so the matched gain is matched where it is read; it is not a held-out estimate of the task.
- **Precision.** Procedural training and every read in fp32 (TF32 off), as E256 and E268; lowercase trains under bf16
  autocast as E53. States are saved in bf16 (as E256), and every reading reads the saved file. The bf16 rounding and
  the definition's minimum-norm split can both move a gains-only state's E_b off 1: stage 0 (e) measures it.
- **Memory and host RAM.** One 67M model plus E268's world (E275 peak 8.5 GB GPU per item, rb_states items), a clone
  per E_b item, the reader's steering grid. Host: E268's world loads E242's inventories (E275 pods ran on 64 GB hosts).
  The rehearsal records peak GPU per part.
- **Time and cost per stage.** Training: E256's SFT 6.2 s for 27 steps (train_sequence.json), E231's gains 0.039 s per
  step on lowercase; with a gain check each step, an estimated 0.1-0.2 s per step: g arms up to 300-490 steps
  (procedural) and 3,000 (lowercase), about 1 h of H100 over all 60 arms. Reads: not logged by E256; estimated 90 s per
  state (the rehearsal measures it). E_b over K: 740 s per procedural state (E268 run_manifest_p0.log, 256 windows of
  64 tokens), 255 s per lowercase state (E275 items/e53-lowercase_s0.json, L40S). Stage 0's identity scan prints the
  card's seconds per item before any item starts. Total about $77 (main plan).
- **Saved, where, size, outliving the pod.** Per state: results/states/<state>.json (train, read, eb; small),
  results/large/states/<state>.pt (bf16 dW, 56 MB), results/large/eb/<state>.npz (in_K, first_flip, num, den, dd,
  changed; ~1.5 MB). Per item a tar results/large/packs/<task>_s<seed>.tar (~300 MB, about 30 s at 10 MB/s), packed
  when the item ends and downloaded at once, its sha256 printed in the log.
- **Downloads.** One tar per item into a project path (`results/reuse_dial_dl/`), moved into results/large after the
  pod is gone (root CLAUDE.md, E36). The small JSONs are also inside each tar.
- **Pod loss and resume.** Every part (train, read, eb) of every state is written when it ends and skipped on a rerun;
  a lost pod costs at most one part. A stopped record is cleared and retried on a rerun.
- **Controls and references.** Stage 0 on every pod's first task, alone; the Gaussian is the first state read and the
  first E_b of seed 0; the gains-only arm is the first E_b of every item (its E_b over changed variables must stay
  at least eb_gain_min, else the job stops). Saved SFT and RL states are placed from their saved readings, the
  same reader code (E211's `Reader`, E264's lens arithmetic, E268's scan settings) on the same rows.
- **Predictions.** The rule is total: every task is readable or not, every reading has a rho or is undefined, the
  verdict is rival a, rival b or no call (score.py run on the edge cases below).
- **Code paths never run on a GPU.** `commute.dial` (hooks, two-group Adam, penalty), `stage0.py`, the lowercase task
  through E268's world, the lens on lowercase's E136 rows. All run in the rehearsal before any pod.

## Gate 2: every setting sourced
In `results/predictions.json`, each with a `_source` line: tasks (the cell), seeds (E256, E53), lr_gain (E231 = (IA)^3
3e-3, gain_only predictions.json settings.lr), lr_res, grad_clip, batch (E256 procedural_rl predictions.json
settings.lr, grad_clip, sft_batch; E53 `lowercase.BATCH`), cap (E231's rule cap_min 300, cap_mult 10 on E256's
27/49/31 SFT steps and E53's 300), kl_rows and kl_batch (E256), rand_sd (E231), t_ref (E256 read/*_sft_s*.json
gain.lo means 6.005 / 8.668 / 6.045; E231 refs 1.643 for lowercase). E_b over K: E268's filed scan settings,
unchanged (`kstar_needed` load_config: 4 derangements x 64 windows, seed 268). Reader: E211's filed settings
(realistic_tasks predictions_run2.json). Lens: E267's (adl amend2 predictions.json: k 5, 8,192 Pile val rows, gain
floor 1). Measured, not sourced (amended after the rehearsal, before the paid run):
- **target**: per task min(t_ref, 0.9 x the gains-only seed-0 plateau) from `calibrate_<task>.json`.
- **betas**: of rehearse_betas (1e1 ... 1e6), the three whose median residual share over the four tasks' calibrate
  runs is nearest 0.25, 0.5 and 0.75. If no beta lands between 0.1 and 0.9 on a task, that task's dial has two ends
  and no middle: the lead is told before the paid run.
- **eb_gain_min**: half of stage 0 (e)'s measured E_b of the random gains state (provisional 0.5). It is a bug bar, not
  1: the definition's least-squares split takes the minimum-norm change over every subcomponent, which can spread one
  subcomponent's gain over others that overlap it (E247's head gain x2 reads 0.76 over K, E268). If the measured value
  is far under 1, E_b cannot certify the gains-only end as fully kept; the dial's sanity check is then only that E_b
  falls along it, and definition-lead is told.
Departures: a gain per VPD subcomponent rather than per neuron (E231) or per (IA)^3 site, because the definition reads
VPD variables; rates and clip as the sources, per parameter group.

## Gate 3: rehearsal ($0, cluster debug QoS, one L40, `E### rehearse: reuse dial`)
Full read sizes, the real model (it fits a cluster card); only step caps reduced.
1. Two jobs at once (`cluster/reuse_dial.sbatch calibrate addition sequence brackets` and `... calibrate lowercase`):
   stage 0 in full on the first task; per task the gains-only seed-0 arm to its full cap (its plateau); every rehearse
   beta, beta 0 with gains, and the ordinary arm to the target. Writes results/rehearse/calibrate_<task>.json as each
   arm ends. Measures: plateau, target, residual share per beta, step-1 gain, s/step per arm, peak GPU. Expected
   minutes: procedural ~30 for the three, lowercase up to ~60.
2. `cluster/reuse_dial.sbatch rehearse` (`run.py rehearse --tag r0`, twice), in parallel with the calibrate jobs at
   the placeholder target (the ordinary arm reached E256's gain in 29-31 steps on addition, under the 40-step cap):
   item (addition, seed 0) with every arm capped at 40 steps, the Gaussian, every
   state read in full and E_b'd over K in full at the full read sizes (256 windows x 64 tokens, 4 x 64 pairs). The second
   run must log "done before, skipped" for every part (the resume). Then the tar's download to the dev node and its
   sha256. The full E_b scan runs on the two ends of the dial only (settings.rehearse_eb_arms g and ft; every other
   state gets the same code path in stage 0 (e) without the scan). Measures: read seconds per state, E_b seconds per
   state on the L40 (and stage 0's estimate against it), state and tar sizes, peak GPU and host memory, stage 0's
   floors. Expected minutes: world load ~5, stage 0 ~10, training ~3, 6 reads x ~1.5, 2 scans x ~20 (E268's 18,300
   cases/s at 64 tokens; an L40 is slower) = ~70 min, under the debug QoS's 2 h.
3. `score.py` and `plot.py` on results/rehearse/states (one seed: every task unreadable, verdict no call).
`fleet/rehearsed compare-lead E### '<floors>, <s/state read>, <s/state E_b>, peak <GB>, at full read sizes'` after.

## Gate 4: early checks in code (filed values, results/predictions.json thresholds)
- Stage 0 (each pod, first, alone; exit 1 on a miss): (a) the dial's start: logits bit-identical (fp32 and autocast),
  export exactly 0; (b) a random dial's export within 1e-3 of its hooked change, shuffled at least 0.1; (c) a zero
  update: every reader numerator exactly 0, KL exactly 0; (d) the do-nothing: 0 changed, every dd 0, E_b = 1;
  (e) a random gains-only state reads E_b over its changed variables at least eb_gain_min, its Gaussian at most
  0.05; (f) the identity scan: 0 argmax changes; it prints the estimated seconds per E_b item.
- Step 1 of every arm: its gain under 0.5 x the target, else stop (`train_arm`).
- Each Gaussian: gain under 1 log-odds, else stop (E256's do-nothing stop).
- Each item's gains-only state: E_b over its changed variables at least eb_gain_min, else stop; it is the first E_b.
- Logged at every step of every arm: loss, penalty, both gradient norms, the read-row gain.

## Gate 5: review
One fresh Opus agent that wrote neither: this appendix, the plan and `experiments/reuse_dial/` plus `src/commute/dial.py`.

## Gate 6: whip-lead's go-ahead
After the rehearsal's numbers, the amended predictions and the cost per stage.

## Edge cases of the rule (score.py on stubs, ~/commute-data/scratch/compare-lead/p263-writer/)
- Every reading constant over the dial states: every rho undefined, nothing tracks, verdict rival b (the readings do
  not follow the dial). The do-nothing itself never enters: it does not reach the target, so no task is readable and
  the verdict is no call.
- One task only: no call. Two arms missing on every task: three levels remain, still readable (shorter dial).
- Ties: d is tied within a level; average ranks. KL constant: |rho_KL| taken as 0, so a tracking cheap reading beats it.

## Pod split
Items (task, seed) in task order, `pod I 9` for I = 0..8: pods 0-2 take one procedural item and one lowercase item,
pods 3-8 one procedural item each. Procedural item: 5 or 6 states x (740 s E_b + 90 s read) + training ~5 min = 75-90
min; lowercase item ~30 min. Wall about 2 h with set-up; at most 4 pods per runner, so three runners.
