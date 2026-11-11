# Qwen instrument check, step 0: what the saved files already say (new decomposition)

*2026-11-11, drafted for `qwen-lead` from the brief `reports/plans/2026-11-11_plan_qwen-instrument-check.md` (Step 0).
Cell P3.X and E### are placeholders the lead fills. Nothing here has been run. All numbers are on the new decomposition
p-9707d6ff unless marked old.*

Step 0 reads saved files on CPU and builds three tables, with no model forward. Table 1 asks whether the least-squares
split could place a random change as easily as a real one. Table 2 sets each update's weight norm beside the
decomposition's own leftover. Table 3 sets each update's stored output change beside the decomposition's own masked
error on the same rows.

## What runs

`experiments/qwen_instrument_check/step0.py`, one CPU job, $0:
`sbatch --comment='E### step 0' cluster/qwen_instrument_check_step0.sbatch` (debug QoS, 8 CPUs, 10G, 1 h 30 min).
On the worker with 8 CPUs, one gram eigendecomposition took 18 s at side 6,144 and 1 s at side 2,048. The 84 MLP
matrices need one of each, about 30 min in all; the attention matrices and the reads add about 10 min. The job prints
Table 3 in its first second, then one line per matrix, which it also appends to `results/step0_sites.jsonl`. A
resubmitted job skips the matrices already written. The tables go to `results/step0.json`, whose `decomp` field is
p-9707d6ff. Rules are in `results/predictions_step0.json`, and the run refuses to start unless that file is committed.

**Alive.** Every Qwen reading on new uses one definition (`qwen_maps.py:64-66`, used at `qwen_d3.py:228-231` and E132
`run.py:42-43`): g above 0.01 at some position of W0's 128 maths or 128 fineweb rows of 512, from job 56760's
statistics. The total, 500,244 of 688,128, is a stop control.

**Updates (15).** E93's eleven, read from the files E93 read: template s0 and s1, TOFU s42 and s142, RL s0 and s70,
shuffled-reward s0 and s70, own-samples, first maths step 100 (fp32), gold maths. Also E40 step 10, and E45's gold s0
and anchor-2 long s0 and s1, which E132 read on new. Every file exists (checked with `ls`).

## Table 1: which share must the split leave outside?

The least-squares split changes the alive vectors themselves: δW ≈ dU V_Aᵀ + U_A dVᵀ (`src/commute/split.py`). It can
reach any matrix whose rows lie in the write span or whose columns lie in the read span. The part it can never reach is
P_U⊥ M P_V⊥, of dimension (d_out − r_write)(d_in − r_read). For an iid Gaussian change, the expected share outside is
`split.gaussian_null` = (1 − r_write/d_out)(1 − r_read/d_in). The two alternatives in the drafting brief are other
splits. 1 − (r_in/d_in)(r_out/d_out) is the outside share of changes of the form U_A C V_Aᵀ. 1 − n/(d_in·d_out) is the
rescaling split's, with one coefficient per subcomponent, and it gives above 0.9995 at every matrix. The brief's
"0.27 × 0.27" has the right form but puts it in the wrong place.

**Expected.** Every MLP matrix has at least 2,572 alive subcomponents for a side of 2,048 (`alive.safetensors`). So if
the alive vectors are in general position, that side is fully spanned, and the outside share is exactly 0 for any change
to any of the 84 MLP matrices. That is 75% of the decomposed weights. Everything outside then sits in attention. In
general position, from the per-matrix alive counts: q 0.617, k 0.378, v 0.252, o 0.245. Pooled over the whole model
for an iid Gaussian change, that gives **0.0981**, an exact lower bound, since a rank below the count only raises the
share. Filed: 84 of 84 MLP matrices full on the small side; at least 190 of 196 matrices in general position; iid
share 0.098 to 0.12.

**Real against random.** Per update, its own outside share sits beside the same null pooled by its per-matrix squared
norms, as E93 pooled it. Rule: placement **free** if every scored update's ratio is at least 0.8, **constrained** if
any is under 0.5, otherwise no call. Expected: free, with ratios of 0.90 to 1.00. On old, E93 read 0.961 to 0.995
(0.0875 / 0.0910 for the first maths fine-tune), with a seed spread of 0.0005.

**Correction to the brief.** The "0.094 random / 0.090 real" pair comes from E93 stage 0 on the old decomposition.
There, 0.0900 is E40 step 10's share left in Δ, and 0.0939 is not a measured random change: it is
`split.gaussian_null` pooled by E40's per-matrix norms (`qwen_d1_table/results/control.json`, `share`). No new
equivalent exists: E93's new amendment (`qwen_d1_table/amend2.py`) was written and never run (`DECOMPOSITIONS.md`).
Step 0 gives the first one.

## Table 2: update norm against the decomposition's leftover

‖Δ_s‖ = the bundle's relative error (`reconstruction_check.csv`, matched by E131 to 8.6e-7) × ‖W0_s‖ from the Hugging
Face weights. Expected: total ‖Δ‖ of 1 to 8. The RL, shuffled and own-samples updates (0.27 to 0.37) should sit under
it on at least 150 of 196 matrices, and TOFU (18.1) above it on at least 150. Weak lean: ‖W0‖ is read here for the
first time.

## Table 3: output change against the decomposition's own error (stored numbers only)

The floor is KL(W0 ‖ W0 masked by its own masks), in nats per token. It reads 0.0932 on the 128 maths rows of 256 that
E93 and E132 used (E132 `controls.json`, `means_W0.K0`). It reads 0.0728 on maths and 0.2090 on fineweb, both on rows
of 512 (job 56760). The brief's 0.07 is the 512-token figure; the maths readings so far used 256.

| update | KL on own rows / floor (128 maths × 256) | KL on 16 maths × 512 / 0.0728 |
|---|---|---|
| RL s0, s70 | 0.69, 1.05 | 1.16, 1.20 |
| shuffled s0, s70 | 1.45, 0.52 | 1.49, 0.59 |
| own-samples | 0.011 | 0.014 |
| first maths step 100; step 10 | 18.4; 17.5 | 9.09; 6.63 (16 × 256) |
| gold maths; E45 gold | 14.8; gap | 7.14; 9.56 |
| template s0, s1 | gap (own rows) | 1.32, 1.37 |
| TOFU s42, s142; E45 anchor-2 long s0, s1 | gap (own rows) | 6.90, 6.94; 8.63, 8.34 |

These are known now, so the script must reproduce them to 1e-12. The reinforcement-learning updates sit within a
factor of 1.5 of the floor, and own-samples sits a hundred times under it. The supervised fine-tunes sit 7 to 18 times
above it.

**Gaps.** No masked error is stored on new for the TOFU rows, the anchor-1 rows or E45's own rows. Filling them would
take forward passes, which this step does not run.

## Not run, and blind to

- No forward pass. Table 3's gaps stay gaps.
- Table 1 cannot see placement inside the span. It sees nothing on the MLP matrices, which are fully reachable. A
  "free" reading says only that real updates are no more inside the span than chance. Step 1's planted changes test
  whether the split recovers a known placement.
- The second-order term dU dVᵀ is left out. On old it moved the share by under 1e-4.

## Cost and gates

$0, one CPU job of about 45 min, with no gated run. The controls that stop the job are the alive total, the rank bound,
one Gaussian draw per matrix against its formula (within 5e-3, over 10 sd), and the cheap outside share against
`split.obstruction_fraction`. Card: `experiments/qwen_instrument_check/card_step0.txt`. Sketch:
`experiments/qwen_instrument_check/figures/sketch_step0.png`.
