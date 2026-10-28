# E40 Amendment 1 rehearsal: why B does not fall, and the fp32 gap (diagnosis, read-only, 2026-10-28)

## 1. Stage B: the curves do not fall
**Cause: the test cannot show a fall. Gradients do reach the offsets.** Confidence: high that gradients flow; medium-high on the rest.
- **Gradients reach the offsets.** Both rates draw the same batches (`train` seeds `gen` from cfg.seed, qwen_teacher.py:193,202), so the curves can be compared step by step. Paired, the 1e-5 curve sits below the 1e-6 curve at 22 of 24 steps (mean ratio 0.90; ratios from B.json, computed with jq). T0's `max_abs_offset` is 3.0e-5 after 3 steps at 1e-5, which is exactly 3·lr, so Adam moved an entry in the same direction on every step. Hooks, checkpointing, requires_grad and the param list are all fine. Step 1 matches at both rates because it is read before the first `opt.step` at zero offsets (qwen_teacher.py:207-220); that is not a symptom.
- **The target is too far for 25 steps.** Adam moves each offset by at most about lr per step. The target offsets are α·R with α = 0.00235 (B.json), so even at 1e-5 the student covers at most 25·1e-5/α ≈ 11% of the way. The best possible KL fall is (1−0.11)² ≈ 0.80, which is a 20% fall. At 1e-6 it is 2%. The 67M E34 also used 1e-5 (teacher_training/run.py:55), but for 1,000 steps toward a real δW, not 25 steps toward a fixed-size random target.
- **Batch noise swamps a 20% fall.** The rehearsal ran with t_rows_each=1 and row_len=128 (A1_TINY, amend1.py:112-114), so each step is 2 rows of 128 tokens drawn fresh. The per-step KL swings 0.004 to 0.013; T3 went from 0.0178 to 0.00104 between steps 1 and 2 (run.log:333,336). The rule compares the mean of the last 5 steps against the first 5 without pairing (`falls`, amend1.py:208), so its standard error is about the size of the largest fall possible.
- **The rehearsal's B does not rehearse the paid B.** `job.alive()` cuts each site to alive_max=16 columns (run.py:559), even though B.json reports the full 328,041 alive. On the paid card the target is spread over ~100x more columns. KL grows roughly as α² times the number of columns, so α would be about 10x smaller (~2e-4), and 25 steps at 1e-5 might reach it. That is an estimate, not a measurement. Either way, the rehearsal says nothing about whether the paid B's rule will pick a rate.
- **TR never ran.** The summary's "TR" ratios (1.10 at 1e-6, 1.24 at 1e-5) are exactly B's last-5/first-5 ratios. There is no TR.json and no [TR] line in any log. So T0's 300 steps and end_frac 0.7, both marked "from TR", still have no source (gate 2 is open).
- **Smallest fix (delete structure):** drop `falls`/`pick_rate`'s use of the noisy training curve. Read the positive control as fp32 `mean_kl(Wt, Ws)` on fixed rows before and after training; T0 already computes this as `kl_teacher_student` (amend1.py:355-356). Size the steps to the target: steps ≥ ~2α/lr, with α taken from the size search.
- **$0 check (debug QoS, ~10 min, L40, tiny config):** run the positive control at 1e-5 for 300 steps, plus 1e-4 for 25 steps. Read fp32 KL(teacher‖student) on the pc_rows at steps 0, 25, 100 and 300.
  - Expected if this diagnosis holds: a steady fall at 1e-5, roughly (1−t·lr/α)² with some lag, so about 0.3 by step 100; and a large fall at 1e-4 within 25 steps.
  - If it stays flat, gradients are the problem after all.
  - Run this check with autocast off, so the issue in section 2 cannot mask it.

## 2. First-batch "training-forward KL vs fp32 KL"
**Cause: bf16 rounding throughout the training forward, not a mismatch between the two forwards.** Confidence: medium-high.
- **Same inputs and quantity.** Both values use the same batch (qwen_teacher.py:202-206, `mean_kl` over `batch.split(1)`) and the same position count (n = 2·rows_each·T). The teacher is the same bf16 δW (`plus(W0, teacher)`). The student at step 1 is exactly W0 (zero offsets). Qwen3 has no dropout and there is no mask. The fp32 reference is true fp32, since TF32 is off (qwen_vpd.py:73-75). The check compares like with like.
- **Where precision is lost.** Under autocast every nn.Linear casts its input to bf16, which has about 2^-9 relative resolution. `change_hooks` adds o_proj/down_proj in fp32 (qwen_teacher.py:73-75), but the next layer's q/k/v/gate/up reads the fp32 residual rounded to bf16. The small teacher-minus-student difference is therefore re-quantised at every layer, not only at the add the pre-mortem named.
- **Why the sign differs.**
  - Small, coherent changes are rounded away, so the KL shrinks. T3's δW is SFT step 3, about 1e-6 per entry (F1 step 2 |dW|/(lr·√numel) ≈ 0.9). Its gap is −44%.
  - Larger, scattered changes get rounding noise added, so the KL grows. B and T0 use a random change on 16 columns per site, with α ≈ 2e-3. Their gaps are +18% and +19%.
- **Too inexact to train against?** Yes, for the early read steps (KL ~0.01-0.03): the target the student sees is 56% to 119% of the true one. The student's own progress is also invisible below the same resolution. T0's rule needs the KL to fall below 0.2× its start, a fall of about 0.008 at a target of 0.01, while the measured bf16 error is ±0.002-0.014 at that KL. On the paid card T0 is likely to fail on precision alone. At KL 0.36 or more (read steps 10 and 100) the relative error is probably small, but this has not been measured.
- **Smallest fix:** delete the two `torch.autocast` blocks in `train` (qwen_teacher.py:210,215) and the RESIDUAL branch in `change_hooks`, so both forwards run in fp32 with TF32 off. The fp32 check then reads ~0 and can be deleted. Cost: fp32 matmuls are several times slower than bf16 on an H200. This must be measured; it adds perhaps 1-2 h over T0 plus T1, about $5-9. Activation memory doubles. A middle option is TF32 inside `train` only (10-bit mantissa, about 8x finer than bf16); measure it before choosing it.
- **$0 check (same debug job, ~5 min):** on B's first batch and on T3's first batch, compute KL three ways: the fp32 reference; the current training forward; and the training forward with autocast off and the hooks kept. Add the current forward with the teacher scaled ×4 and ×16, and the TF32 variant.
  - Confirms: autocast off matches fp32 to <1e-4 relative, and T3's gap moves from −44% toward 0 as the teacher grows.
  - Refutes: autocast off still differs from fp32. That would point to a forward mismatch.
  - Record seconds per step for each precision; this feeds the cost projection.

## 3. Other things that would waste money on the paid run
- **Uploads.** 24,842 s (6.9 h) is ~50 GB at the cluster's 2 MB/s probe. At $4.5/h the projection alone is ~$31, so `n_read_max` becomes 0 and B halts after the benchmark. The run pays for the fetch and B, then stops. On the volume 2bun37yawb, run without --wandb-upload, or upload only results JSON and final offsets; decide before the pod starts.
- **T0 and teacher_first use step 1's single noisy batch as the denominator** (`fall`, run.py:334). With per-step batches this can pass or fail by chance. Use the fp32 fixed-row reading here too.
- **B's size search sets KL 0.01 on the first 8 maths rows**, but the training batches (half FineWeb) read 0.0047. The "target_kl" does not describe what is trained.
- **Bookkeeping:** rehearsal.json (job 40637) omits job 40602's B fp32-gap misses.
- **Gate 3:** after the fixes, the rehearsal must be rerun (precision change, new rule) before the go-ahead.

## Not run
No code executed and no jobs submitted. The α ~10x smaller at full alive is an estimate. That precision causes the sign flip is inferred from the code and three numbers, not measured.

Friction: the summary labelled B's ratios as TR's, which took a cross-check to notice; with python barred on this node, the paired curve ratios had to be done in jq.
