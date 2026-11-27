# P1.65: readthrough fills (K numbers, rescale-only control seeds)

Cell P1.65 (definition-lead, 2 runs, due 2026-11-27, cap $40 and 4 h). E-codes to claim with `fleet/ecode` before any
job. K = the necessary set (single-variable swap test, 256 pairs = 4 derangements x 64 windows, seed 268,
`commute.needed.scan_sites`), as in E298. K* = the mean-ablation circuit. Writing this plan found that two of the three
K* numbers already have a K value on disk, so Run 1 needs only one GPU read.

## Run 1: three paper numbers read over K

| paper | K* value (reproduced) | K value | how |
|---|---|---|---|
| main.tex:432 "0.999 or more" (graded cos^2 clause, every neuron as a variable, E196's 20 pair fine-tunes) | min 0.99938 (a) | **not saved**: GPU read below | new |
| main.tex:438 "0.83" (SAE error terms' share of rebuilt's write change, every position) | 0.8294 (b) | **0.781** (c); rebuilt E_b 0.135 -> 0.127 | $0, saved |
| appI:55 "three pairs keep their gap once KL is regressed out" | 3 pairs outside floor 0.0109, r2 0.209 (d) | **same 3 pairs** (-0.020, -0.019, -0.029) outside floor 0.0110, r2 0.198 (e) | $0, saved (E275) |

Sources: (a) `jq '[.R3|to_entries[]|select(.key|test("_s[01]$"))|.value.rho_main_replace]|min' experiments/rho_saved_items/results/summary.json`.
(b) `jq '.rows[]|select(.item=="rebuilt").remainder_dd_share' experiments/sae_eb/results/summary.json` (E263, K* of E245).
(c) `jq .leftover_share experiments/fig2_standard/results/items/sae/built__rebuilt.json` (E298; E_b_K 0.1275, |K| 48,868).
(d) `jq '.B_pairs.Kstar' experiments/claim2_needed/results/summary.json` (= E262's `kl_control/results/summary.json`).
(e) `jq '.B_pairs.K|{floor,outside_floor,r2:.fits.kl.r2,pairs}' experiments/claim2_needed/results/summary.json`; no sign change, 0 missing.

Two side notes for writeup-lead. Line 438's "0.135" and "0.115 margin" are also K* values, though they are not marked.
Over K they are 0.127, against 0.023 for E263's KL-matched random (margin 0.104) or 0.041 for E298's random beside x2
(margin 0.086). E208's set is not a mean-ablation set either. It is a KL-closure set (tau 0.9) in which every changed part
outside it runs M's mechanism (`commute.rho.contrast`, "replace"). So the K* mark on 0.999 is a label slip, though that
set is still not K.

**What runs (number 1 only).** E277's neuron swap test (`experiments/needed_neurons/run.py`, `NNR.read_item`, "trained"
stage) on the same 20 E196 pair fine-tunes, the same D (E208's 512 read windows, last position: `topk_neurons.windows`)
and the same rows. The one change is that the band is `"width"` (`topk_upper.feat_of`: hi = -inf). Every activation is then
its neuron's value, so each layer remainder writes 0: that is E208's featurization of one variable per neuron. A new
`run.py` of about 60 lines loads E277's module, points its `RESULTS` at its own folder (precedent:
`key_grid/run.py:72`), builds its config from E277's settings with `band="width"`, and reads only the 20 fine-tunes (no
randoms). A CPU step then takes the mean of E208's saved `r_main` over K and changed, mapped by part name, from
`rho_saved_items/results/large/parts/<item>.npz`. It reports the mean over K as well, counting unchanged parts at 1.
- Inputs: the E196 states as saved dW, about 2.43 GB on NFS (kv 4 tasks x 2 seeds x 113 MB, case and repeat 8 x 57 MB,
  fact and twin 4 x 268 MB); E208's parts npz, 20 x 0.4 MB, CPU only.
- Expected values. Do-nothing: K empty and the reading 1 (E277's stage 0). Control: E277's identity swap gives 0 flips. A
  bound with no run: K and changed lie inside the changed parts, whose mean r_main is already 0.99940-0.99995 per item and
  whose minimum is 0.88-0.98 (`.r_changed.main` in each `rho_saved_items/results/items/*_s[01].json`). So the K value
  falls between that minimum and 1, and it stays at 0.999 or more unless K picks out the lowest parts.
- Time: E277 measured 39-103 s per trained item on the same 12,316 variables x 256 pairs
  (`needed_neurons/results/run_trained_t0.log`, `run_trained_t12.log`). 20 items x 103 s = 34 min at worst, 13 min at 39 s.
  Two debug shards of 10 items take at most 17 min each, plus about 3 min of set-up.

## Run 2: rescale-only control, seeds 1-2 at VPD, seeds 0-2 at SAE latents

All 15 FFN-gain states were already trained on 2026-11-23 and are saved
(`ia3_control/results/large/gains_ffn_<task>_s{0,1,2}.pt`, 199 KB each, read step inside). Only the reads are missing,
so nothing is trained. The full fine-tunes' values for all three seeds at VPD and at SAE are already in
`fig2_standard/results/definition_rows.csv`, under one needed-set rule. VPD runs 0.029-0.042 and SAE 0.031-0.047.
- **What runs.** E298's reader (`fig2_standard/run.py read_item`, `feats.reader`) on the new items `gain/<task>_s<seed>`.
  These are made by E290's `World.make("ffn/<task>_s<seed>")`, the same World that E298's `trained/` block uses. The run
  is a new `run.py`, about 60 lines, plus a `Worlds` subclass that adds that one block. It sets
  `FS.RESULTS` to its own folder, takes E298's filed settings unchanged, runs E298's stage 0 (do-nothing, identity, and
  must-flip at SAE) on its own card, then reads VPD on 10 items (seeds 1, 2) plus seed 0 again (5 items) as the
  repeatability check against E290's `vpd/ffn__*_s0.json` (0.0195-0.0719), and SAE on all 15.
- Expected values. Do-nothing: E_b 1, dd 0. The neurons-only reading at the read step is 1 to 1e-12 (E290 R1, already
  read). At VPD, seed 0 repeats E290 within E298's filed `repro_rel_tol["vpd"]`. Seeds 1 and 2 should fall near
  0.02-0.07, because the reading follows the read step and the gains, and those match across seeds (below). At SAE there
  is no value yet. The prior is near the full fine-tunes' 0.03-0.05, with E298's error terms carrying a large share
  (`leftover_share`, which the run reports beside the value).
- Time: E298 measured, on H100, about 130 s per VPD trained item and about 25 s per SAE item (64 s for copy)
  (`fig2_standard/results/items/{vpd,sae}/trained__*_s1.json .seconds`). E290 measured 197 s per VPD item on an L40S
  (`run_vpd_vD.log`, 3 items in 9 min 51 s). VPD: 15 x 197 s = 49 min on L40S, as two debug shards of about 25 min, or
  15 x 130 s = 33 min on one H100. SAE: 15 x 30 s = 8 min on H100, about 12 min on L40S.
- **Recipe of the saved states, from code** (`ia3_control/run.py train_one`, `gains.train`, predictions "settings").
  Only down_proj's input gains train, one per MLP neuron (12,288), starting at 1. Adam: lr 3e-3, betas (0.9, 0.999),
  no weight decay, no warm-up. Batch 32, gradient clip 1.0, 300 steps. The sampler is `torch.Generator().manual_seed(seed)`,
  drawing one batch of 32 training windows per step. Loss: `finetune.arm_a_loss`, cross-entropy on the table target at
  the half position plus KL(M || F) at the other positions, `kl_lambda` 1. Checkpoints come at 10, 30, 100 and 300. The read
  step is the first with held-out P(target) >= 0.5, else 300 (`keydep.scored_step`). Data: E147's windows per task. Read
  steps: copy 100 on all seeds, the rest 300. Held-out P(target) at the read step runs 0.02 (cipher) to 0.80 (copy), within
  0.013 across seeds. The full fine-tunes are E147's step-30 updates at 3e-5.

## Queue and cost
Compute ran for `cpu-run` on 2026-11-26. `cpu-run` reported the compute node "completing*" at about 20:30, and
`sinfo` read "mix" minutes later. Try debug first: Run 1 as two shards, Run 2 as VPD in two shards plus SAE as one job,
all inputs on NFS, $0. If a job has not started within 5 min, use one H100 pod per run (at most 4 per runner). Uploads
at 10 MB/s: Run 1 needs 2.43 GB, about 4 min. Run 2 needs the SAEs (1.3 GB) plus the E237 windows and gains (under
0.1 GB), about 2.5 min, or at most 4.0 GB if E298's whole SAE manifest goes (about 6.7 min). Pod cost: Run 1 about
0.8 h and Run 2 about 1 h at about $3/h, so about $6 in all, under 2 h of wall time. Neither run is gated.

## Not run
- Matched randoms of the gain states at SAE (not asked; E290 has them at neurons, plus seed 0 at VPD).
- The IA3 arm at VPD seeds 1-2 and at SAE (not asked).
- A gain control at the full fine-tunes' step and rate (the readthrough's "ideally"). That would mean retraining, so a
  new cell.
