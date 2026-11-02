# E59 stop at stage 0 (job 48279): advice

**Recommendation.** The readouts overfit, they are not under-trained: the last block's train-batch loss settled at
3.2 from step 2000 on (lr already 1e-4), below the model's own 3.43, while held-out reads 4.14. The cause is the
data: 4,000 x 96 x 41 = 15.7M window-tokens were drawn from 8,192 x 512 = 4.2M tokens, so every token was fitted
~3.7 times. Jain et al. train the readout on fresh samples from the process (§3, L1238-1248); ours repeated. Fix the
data, keep the schedule, replace the guessed 0.5-nat margin by a measured reference, and rerun at $0. Dated
amendment to stage 0 only (P1-P3 untouched); no new experiment, no money: FYI to the user, not an ask.

**Settings to change (predictions.json `amended_2026_11_02_stage0`, run.py, readout.py).**
1. `train_rows` ["train", 32768, 32768] (rows 32768-65535, 16.8M tokens ≥ 15.7M drawn: each token seen < 1 time,
   the paper's fresh-data reading; still disjoint from rows 0-32767 of the task windows). Cache the rows in a CPU
   Slurm job before the GPU job (`pile_rows("train", 32768, 32768)`, ~5 min, 134 MB npy).
2. Schedule unchanged (4K AdamW, lr 1e-3, x0.1 at 2K/3K/3.5K, batch 96, nn.Linear init on all four blocks: Jain
   L1238-1248 via probe.py). Do not warm-start the last block from the unembedding: it would make the check a free
   pass and break the one-recipe-per-block comparison. The unembedding enters as the reference instead (3).
3. Stage-0 rule: `L_last ≤ L_lin + 0.1`, both on the 2,000 held-out windows. `L_lin` = the held-out loss of the
   linearised head: the model's own unembedding with the final norm's per-token scale replaced by its mean over
   ~10 training batches (the scale is the head's only nonlinearity; the centering and weight fold into W, b). It is a
   feasible linear map, so it bounds the best linear readout from above; a trained readout that cannot match it is
   under-converged or overfit, never "fine". Computed in the run, $0, seconds. Noise floor for 0.1: the logged
   train-batch losses at steps 3500-3999 have sd 0.14 on 3.9K tokens; the held-out mean is over 78K tokens, SE ≈
   0.03; 0.1 ≈ 3 SE. `gap_last` (vs the model's 3.43) stays reported, unscored.
4. Log the model's own loss on every logged training batch (same logits, no cost) and the readouts' loss on 2,000
   *training* windows at the end (held-in). Expected after the fix: held-in − held-out within 0.1 at every block;
   a larger gap says overfit, a small gap with `L_last > L_lin + 0.1` says optimisation.
5. Early check (filed): last-block train-batch loss ≤ 5.0 at step 250 (measured 4.73) and ≤ model + 0.5 at step 2000
   (measured 3.20 vs ~3.4); a miss stops.
6. `readout_max_minutes` 18 (measured 0.23 s/step = 15.6 min; the review's 4-6 min was 3x optimistic). Keep fp32,
   TF32 off: a precision change is unmeasured and the saving is unneeded.
7. `--readouts PATH`: load `results/large/readouts.pt` and skip training, so a later miss after training (P3,
   controls) reruns in ~3 min. The 48279 file is superseded by the data change; keep it as `readouts_48279.pt`.
   No residual caching: 4 blocks x 15.7M x 768 fp32 is 190 GB.

**Possession thresholds.** Keep rank ≤ 20 for 24/32. Rank is relative within a block's own logits, and a worse
readout at blocks 0-2 biases toward "not possessed" (the null, P2 void), never toward the claim; blocks 0-2 overfit
too (train 5.06 vs held 5.67 at block 0), so the fix lowers them as well.

**Time and QoS.** CPU caching job ~5 min, then one debug job: load 1 + inputs 1 + training 16 + losses 0.5 +
possession, controls and ablations 2 ≈ 21 min under the 30-min wall. If the caching job is slow or the debug queue is
busy, `normal` QoS with `--time 45` on an L40 when the queue is empty; RunPod not needed.

**What would change the advice.** Rerun held-in ≈ held-out but `L_last > L_lin + 0.1`: the limit is optimisation of a
convex objective; amend iterations before the first decay (not the margin). `L_lin − model > 0.3`: the per-token
scale matters, the linear reference is loose, and the margin should be set from the held-in/held-out gap instead.
Held-out gap still > 0.5 after fresh data: the readout is not a reliable Def 1 probe on this model at d = 768;
report P1 as "unreadable" rather than widen the rule.
