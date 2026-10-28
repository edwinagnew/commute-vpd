# Review of 51afa50 (E40 Amendment 1 rehearsal fixes): Opus, fresh, read-only

Evidence used: the diff; amend1.py, run.py, qwen_teacher.py; predictions_amend1.json; check_teacher.json; the live
rehearsal job 40691 on the committed code (results/amend1_rehearse/run.log, B stage, read at 16:29 UTC).

## MATERIAL
1. Step-1 check: measured margin 1.8x, not 8x, and it stops B on the paid card. run.py:752-756 (teacher_checks)
   with predictions_amend1.json early.teacher_first.fp32_tol = 2e-3. Job 40691 on the committed code reads
   rel = 1.13e-3 on B's first batch (both rates: kl 6.6636e-3 vs fp32 6.6561e-3). That is 4.7x the 2.4e-4 cited as
   the tolerance's source. job.check stops by default, so a slightly larger TF32 gap on 512-token rows halts B
   minutes in, after pod set-up and the ~20 min fetch. Fix: before the paid run, file a dated tolerance from 40691's
   measured gap (for example 5e-3 or 1e-2, with the measurement as its source), or make the check report-only in B.
2. T0 will probably miss fall_max = 0.5 and halt after F1 has been paid for. amend1.py:213-219, 260, 354-360.
   (a) The extrapolation is log-linear through steps 0 and 100, but the measured curves bend. check_teacher: 0.702 at
   step 100 and 0.590 at 300 at 1e-5. Job 40691 B at 1e-5: 0.867 at 50, 0.718 at 100, so it extrapolates to 209
   steps, where the check's curve reads about 0.6. The plan says so itself (Gate 3, item 4). (b) B's curve is
   measured at target_kl 0.01, but T0 runs at F1's first-read KL. The full-size rehearsal F1 picked step 3 at 0.022,
   so α is about 1.5x larger and the same steps go less far. Both errors under-count steps. T0's stop then ends the
   job before D1 and T1. Smallest fix: let T0 read the fixed rows every 50 steps and end at the first reading
   <= fall_max, capped at 300 (a dated change to T0's steps_rule). Alternatively size T0 at B's target_kl, so B's
   curve applies as measured. Either way, add "T0 fixed-row fall" to what the rehearsal must show: A1_TINY's
   expected_misses contains "T0 positive control" (amend1.py:126), which hides this exact miss in rehearsal.json.
3. With --wandb-upload, T1's last offsets are still uploaded and the job waits for them. run.py:1088-1089 job.keep
   (fp32: 9.6 GB at 2.39e9 entries, ~14.6 GB at E40's 3.65e9), then run_stages' uploader.finish() (run.py:1207). At
   the 2.0 MB/s uplink measured in the rehearsal, that is 80-120 min of billed idle H200 (~$6-9), which the
   projection no longer counts (projection, amend1.py:176-197). Fix: don't keep() the offsets (they stay on the
   volume), or set upload_max_bytes in A1, or tell the runner not to pass --wandb-upload.

## minor
4. Each micro-batch recomputes every site's change twice (forward and checkpoint recompute) plus its backward:
   qwen_teacher.py:224, 4 micro-batches per step (t_rows_each 4, t_micro 2). At the full alive set that is about
   2e14 FLOP per step, roughly 2/3 of the step, against ~7e13 for the model. Computing each change once per step (a
   leaf C with grads accumulated over micro-batches, then one C.backward into dV, dU) would cut the step about 2x.
   T1 has 3 x 1000 steps. Optional; B's projection guards the budget either way.
5. Readings hold the full gradient buffers. on_step runs before the next zero_grad(set_to_none) (qwen_teacher.py:237),
   so at E40's count peak = offsets 73 + 11.3 reading copies + target/δW/logits ≈ 97 GB, under 125. This is covered:
   mem_check("teacher training") after B (amend1.py:255) catches it at the real count before T0 and T1. No change.
6. T1's fixed-row end check (step 100, end_stops False) never runs in the rehearsal (t_steps = 3). It is the same
   code as T0's and report-only, so low risk.
7. check_teacher.py:73-74 and 122-123 still pass autocast= to train, so pyright fails on that file. It is left as it
   ran, but make lint will flag it; add a one-line note or a type: ignore.

## Checked, no issue
TF32 is on only inside train's forward and backward. tf32() resets through fp32_matmuls() in its finally (torch
2.8: allow_tf32 and the precision setting are linked). on_step, the step-0 reading, kl32 and every mean_kl run TF32
off. Hook dtypes: the student change is fp32, the teacher's bf16 δW is cast to fp32 per site, and the checkpoint
recompute runs inside the student's hooks. Fixed rows are deterministic (the first 4 GSM8K-test maths rows plus the
first 4 FineWeb eval rows). They are disjoint from the pools (train_idx sequences; web pool excludes web_lists).
α is sized and read on the same rows, and the final α is measured by the step-0 reading. qualifies, pick_rate and
t0_steps handle zero or NaN starts, ties and r <= frac, give at least 100 steps, and the 1e-05 / 3e-05 key strings
round-trip. score_amend1 reads T0["kl"] and T0["fineweb"], both still written. No live reference to run.fall, TR or
logits32 remains. T1's resume meta still keys on δW, alive set, rate and steps.

Friction: none of note. The commit's "8x margin" claim was contradicted only because a rehearsal on the new code
was already running and its log was readable.
