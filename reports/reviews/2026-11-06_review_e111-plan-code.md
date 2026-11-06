# Review: E111 plan and code (teacher training from least squares on E93's table)

*2026-11-06, fresh Opus reviewer (wrote neither). Scope: `reports/plans/2026-11-06_plan_e111-teacher-table.md` (with
its Overrides), `_appendix.md`, commit a5b01b30 (`experiments/qwen_teacher_table/{run,teach,rules,upload}.py`,
`RUN_BRIEF.md`, `results/predictions.json`, `cluster/qwen_teacher_table_upload.sbatch`), reusing
`src/commute/qwen_teacher.py`, `src/commute/qwen_d3.py`, `experiments/qwen_d1_table/{run,rules}.py`. Code read, not run.*

The training and measurement code is correct. I found no wrong weights, no frozen-Δ leak, no self-comparison, and no
fp32 sums. The trouble is in the early-check values: two of the stage-1 stops are very likely to fire on a healthy run,
and two more could. Each would stop the job after the shared reads (~25 min of pod time, plus the 25 min fetch), and
`RUN_BRIEF.md` then says "never rerun". Fix them in `predictions.json` and `rules.py` before stage 1. Every fix is a
filed value, made before the run.

## Blocking for stage 1

**B1. The TF32-vs-fp32 tolerance at zero offsets has been exceeded before, at a larger K.**
`teach.py:174-175`, `rules.py:83`, `predictions.json` `first_minutes.tf32_rel = 1e-3`, sourced from "3.4e-5 and 2.4e-4,
job 40673". The 2026-10-28 review of E40 Amendment 1 (`reports/reviews/2026-10-28_review_e40a1-code-review-2.md`,
item 1) read the same path on job 40691 at **1.13e-3** relative, at K = 6.7e-3. Here the 8-row K at the
least-squares state is about 0.0023, three times smaller. A rounding error of fixed size is a larger share of a smaller
K, so the expected gap is above 1.13e-3, not below 1e-3. The check misses, and the job stops at the first arm.
Smallest fix: log `tf32_K8` beside `K8` and drop it from the stop. The TF32 offset enters only the training forward's
loss value, never a scored number, and the zero-offset fp32 K8 against E93's stored value still guards the base.
If a stop is wanted, take a tolerance of 1e-2 with job 40691 as its source.

**B2. The step-25 walk band is centred on a line that E76's own early readings contradict.**
`rules.py:45-46, 86-88`, `teach.py:195-198`, `first_minutes.walk = [0.3, 3]`. The forecast is 1.2e-3 × (lr/1e-6)² ×
step: the slope of E76 arm A between steps 100 and 1200, with no intercept. E76's `results/A.json` has the early points
(`blocks` summed): **0.325 at step 50** and 0.534 at step 100. At step 50 the line gives 0.06, so E76 read 5.4 times the
line there. To come in under 3 times the line at step 25 (0.09 at lr 1e-6), the walk would have to grow with exponent
≥ 1.85 between steps 25 and 50. Between 50 and 100 it grew with exponent 0.72. If the least-squares start is instead
noise-dominated (the residual K is 40 times smaller than E76's), a pure Adam walk at E76's early scale gives about 0.1
times the line. That misses the low end. Either way the band covers neither regime the source data show.
The override dropped the room cap, which was the band's reason to exist: the verdict no longer reads ‖r‖².
Smallest fix: keep logging r2 and the forecast, and drop the walk from the stop. If a band is kept, centre it on E76's
measured step-50 value scaled by rate squared (5.5e-5 for 1x and 5.5e-3 for 10x at step 50), check at step 50, and
state the regime uncertainty in the source line.

## Fix before stage 1 (cheap, same edit; each could stop a healthy run)

**B3. `sqrt_v_min = 1e-7` (10 × Adam's ε) has no measured source, and the margin is thin.**
`rules.py:88`, `teach.py:157-160`. A rough estimate from E76's step-1 squared gradients (C arm, `run.log:15`: 0.73 over
about 1e8 dV entries, so an RMS of about 8e-5 at K ≈ 1.8) puts the RMS at about 3e-6 once scaled by √K to this K of
0.0023. Block-0 gradients are about 30 times smaller in magnitude than block-27's (same line), and a tensor's median is
below its RMS. So the median over tensors of each tensor's median √v̂ could sit at 1e-7 to 1e-6. Adam also still works
at √v ≈ ε: the step is about half of lr·m/√v, not "SGD at an unknown rate". Fix: log it, and stop only under ε itself,
or not at all.

**B4. "The 8-row K not above step 0's" at 9 decimals is a direction test at the 1x rate.**
`rules.py:89`. At 1.3e-8 for 25 steps on 2-row batches, the offsets move about 3e-7 per entry. Whether K8 has fallen
by step 25 is not known, and a rise of 1e-9 nats stops the job. Fix: allow a rise of up to `band` × K8_step0, or make
it a stop only on the 10x arm. The 1x arm's readings are reported anyway.

## Not blocking for stage 1

5. **The host-RAM guard does nothing on a pod.** `run.py:59-60, 213` sets `host_max_gb = 0.9 × /proc/meminfo MemTotal`.
   In a RunPod container that file shows the host machine's memory, often 1-2 TB, not the cgroup limit, so the kernel's
   OOM kill comes before the check. The brief asks for 128 GB against an expected ~100 GB. Fix: take the minimum with
   `/sys/fs/cgroup/memory.max` (v2) or `memory/memory.limit_in_bytes` (v1), and have the runner print the limit at
   set-up.
6. **TOFU training rows overlap scored rows (stage 4).** `teach.py:116-123`. The pool is 32 packed rows of the same 800
   held QA pairs that column 4's gain sequences read (`d3.tofu_seqs(tok, d3.tofu_kept())`, `qwen_d1_table/run.py:295`).
   So about an eighth of the TOFU gain items are trained on directly. The pool rows also interleave with the table's
   128 evenly-taken rows, so the QA pairs that straddle row boundaries put the same text in both. Column 4 had a wide
   margin for least squares (0.991-0.997 against 0.88), so this can inflate the reading but not flip a pass. Fix before
   stage 4: build the pool from QA pairs that appear in no table row, and either read column 4's gain also on the items
   outside the pool (printed beside) or state the overlap as a departure. Maths, template, u1 and u2 pools are clean:
   GSM8K train past E40's 256 held-out problems, against table rows from GSM8K test (`qwen_sft/run.py:843`) and gain rows
   from `perm[:256]`.
7. **The plateau stop reads table rows.** `run.py:138`. `rows8` is the first 8 of the 128 scored rows, so the end step is
   chosen on rows that column 1 then scores. The bias is small, since the stop is a plateau rather than a minimum, but
   name it as a departure, or read on 8 rows outside the table.
8. **A log line that will look like a failure.** `qwen_teacher.train` (lines 213-215, 232-233) logs "first batch
   KL(teacher ‖ W0): fp32 X, training forward Y". With `FromLS` the training forward's student is W0 + the
   least-squares moved part, not W0, so Y will be about 40 times smaller than X. Both numbers are recorded in
   `train.first_batch`. Tell the runner it is expected, or drop the record.
9. **Gates record is stale after the override.** Appendix F still says "Saved: cluster disk, nothing on a pod",
   "Downloads: none", "Pod loss: cluster only" and "Time and cost: $0". Appendix D still says the saves are fp32 (the
   code saves bf16, as the override says). Gate 6 needs a cost per stage: about 5.5 h on an H100 by `RUN_BRIEF.md`'s
   minutes, about $15-20. Two more points belong in the brief:
   - Resuming on a new pod needs the downloaded `results/<state>/state.json` files synced back, since they are
     untracked until committed. Name this in `RUN_BRIEF.md`.
   - Say what the pod does while the lead and user read stage 1: stop it, or terminate it and refetch (~25 min).
10. **An edge case misdescribes do-nothing under Δ frozen.** `rules.py:234`, "Do-nothing as the trained state:
    K11 = K01". With Δ frozen and nothing moved, Γ reads W0's weights, so K11 = K00, not K01. The named outcome
    (fails on column 1) is unchanged, but column 5 would read "misses" there, not "holds". Fix the label.
11. **Some rules mix this card's reads with E93's L40 reads.** `rules.py:100-109`. Column 1's teeth use E93's random and
    signs reads against this card's remainder. Column 6 uses E93's ceiling and random-under-masks against this card's
    own cells. Cross-card drift (≤ 1e-3 by the card check) is far below these rules' margins, but state it in the write-up.
12. **T5 excess.** share(LS + r) − share(LS) = ‖r‖² − 2⟨remainder, r⟩. Least squares' remainder is orthogonal to the
    tangent space at the original U and V. r lives in the tangent space at the moved U′ and V′. The cross term is
    O(‖dU‖/‖U‖), likely negligible for these small updates. T5 already returns no call on a non-positive 1x excess.
    No change.

## Checked and fine

- **Algebra and layout.** `FromLS` builds V′ and U′ at the cut with the right layout: theory dU is [d_out, n] and
  Offsets' U is [n, d_out]. Its `change` is base + r's change, cross terms included. The gradient reaches only dV and
  dU. Student and teacher both use hooks on W0, so Δ stays frozen in training.
- **The table reads the trained state.** `trained_change` places r at `pos` in the same layout. The "student" weight set
  is W0 + base + r's change, so clean, gain and cells all read the trained state. The remainder is formed in place
  after the "student" copy, which is safe.
- **The consistency check.** It compares two independent forms, offsets against fp64 U′V′ᵀ − UVᵀ. It would catch a
  wrong r at about 2% of the moved part's norm.
- **No reproduction check compares a value to itself.** Split, ruler, K00, least squares' moved K, K11/K10/K01 and the
  zero-offset K8 are each fresh reads against E93's stored values.
- **Precision and stops.** Sums are fp64 throughout. TF32 is off for every reading (`tf32()` restores fp32 matmuls).
  The per-arm resume works.
- **Imports.** The `load_experiment` name clashes (`rules`, `run`, `upload`) resolve correctly. The module-level path
  rebinds (`d3.GSTATS`, `d3.A1_ITEMS`, `qwen_tofu.DIR`) happen before their first use.
- **Pod inputs.**
  - Tracked in git: E93's `R_s70/state.json`, `maths_rows.json`, `forms.jsonl`, `qwen_adversary/results/tokens.json`
    and E93's `predictions.json`.
  - From W&B: the cut and gstats (E80's artifact), the anchor-1 items and the TOFU files (`e111-inputs:v0`), and δW
    (`e111-dW-R_s70:v0`, uploaded by job 54068).
  - `wandb` is in the venv. Disk for all eleven states is about 47 + 59 + 5 GB, inside 150 GB.
- **Rules and leans.** The verdict reads columns 1, 4, 5 and 6 only, as the override says. Rates are as in the plan's
  section 2. The kept state is the last step. Each lean has its no-call case and a do-nothing line.

Friction: the main clone was behind origin (a5b01b30 not checked out); pulled before reading.
