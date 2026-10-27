# Review: E37 Amendment 1 code (`amend1.py`, `masked_teacher.py`, shared edits)

Scope: `git diff f8551b4 8375fa3`, that is, `experiments/teacher_spread/amend1.py`, `src/commute/masked_teacher.py`,
`experiments/teacher_spread/RUN_BRIEF_amend1.md`, `tests/test_teacher_spread_amend1.py`, and the edits to
`src/commute/gamma.py`, `src/commute/component_training.py` and `src/commute/row_copy.py`. I checked them against
`reports/plans/2026-10-26_plan_teacher-spread.md` → "Amendment 1 (2026-10-27)" and against param-decomp at the pinned
commit 74146b5 (the installed copy in `~/commute-data/venv-py313`, whose `direct_url.json` names that commit). I did
not write this code, and I changed none of it. I loaded no model and started no GPU job.

**Verdict.** I found nothing that would give wrong numbers on the GPU or leave the model changed between stages. The
three shared refactors keep the old behaviour exactly. Arm 1 continues E37's batch draws. Arm 3 starts where it should.
Arm 2's gradients reach U and V and nothing else, its dead columns stay fixed, and the model is restored afterwards.
The scoring follows the plan's rules. What I did find is smaller. The time rule leaves out the reads it says it counts.
Nothing records whether the masked losses actually did anything, so an arm 2 miss could not be told apart from inert
losses using the JSON alone. The new masked trainer first runs on the GPU about 25 minutes into the job. And G2's
tolerance has never been measured for the path it compares. All of these are cheap to fix before the predictions are
filed.

Checks run: `fleet/cpu-run uv run pytest -q tests/test_teacher_spread_amend1.py` passed 6 of 6. `ruff check` on the
changed files is clean. pyright did not run, because nodeenv failed on the worker. That is an environment problem, not
a code one.

## Findings, by severity

### 1. Medium: nothing records whether the masked losses acted, and the toy test cannot see inert masked gradients

`masked_teacher.py:97` puts the two masked terms in each step's curve entry. `amend1.py:178-183` keeps only the teacher
KL (`kl_step1`, and `train_kl` beside each read), and the curve is never written to `amend1.json`. The masked terms
reach only the log, once every 100 steps.

**Failure.** M0 is a drift control. It shows that the masked losses do not move the model too far, but it cannot show
that they move it at all. Inert masked gradients would pass M0 easily (damage 0, check about 1), and arm 2 would then
be arm 1 run through another forward path. If P2 then misses, "masked teacher training does not fix the check" and
"the masked losses were never optimised" read the same in the JSON. The 100-step log lines would separate them only if
someone parses the log. The toy test (`tests/test_teacher_spread_amend1.py:100-113`) uses a teacher with a non-zero
weight change, so U moves under the KL alone. It would still pass if no gradient from `compute_losses` reached U and V.

**Fix.** In `masked()`, store beside each read the mean of `StochasticReconSubsetLoss` and `PersistentPGDReconLoss`
over the 50 steps before that read (`kl_before` already does this for any key), plus their means over the last 50
warm-up steps. That gives the "did they fall" reading at step 0, 1,000 and 2,000. In the toy test, add one case with the
teacher at W0 (`teacher.teacher_net(model, {})`). On CPU in fp32 the KL gradient is then exactly 0 at the start, so
alive columns that still move prove the masked gradient reaches U and V.

### 2. Medium-low: the time rule's projection leaves out the reads

`amend1.py:177-178` computes `sec` and `"seconds"` at the same moment, straight after `train()` returns and before the
save and the reads. `m0()` at `amend1.py:206` then takes
`per_read = seconds - sec_per_step * (warmup + m0_steps)`, which is about 0, not "one read at M0".

**Failure.** The projection for arm 2 is training time only. It misses two reads of about 1 to 1.5 minutes each
(harness about 55 s, plus logp, gap and damage) and the save. So the 40-minute rule lets arm 2 run 2,000 steps when the
real total is up to about 3 minutes over. That changes the outcome only near the line, but this rule is filed in
`controls` → "time rule".

**Fix.** Record `train_seconds` right after `train()`. Record `seconds` (`time.time() - t0`) after the reads, which is
what `masked()` already returns. Then use `per_read = (seconds - train_seconds) / len(reads)`.

### 3. Medium-low: the masked trainer first runs on the GPU about 25 minutes into the job

Stage order (`amend1.py:320-330`): W0, do-nothing, `arm3_0`, arm 1 (4,000 steps plus three reads), arm 3 (2,000 steps
plus two reads), then G2 and M0. `masked_teacher.train` is new GPU code. The CPU toy test runs it without bf16 autocast
and at toy sizes. Its memory (estimated at 40 to 50 GB) and its step time are unknown.

**Failure.** An OOM, or an autocast or dtype error that appears only on CUDA, would surface about 25 to 30 minutes in.
Arms 1 and 3 would be saved by then, as the plan intends. But the brief says "do not retry", so arm 2 is lost for this
pod, and a second pod repeats about 7 minutes of set-up. Partial mitigation: Refit ran the same `compute_losses` path
under autocast on an H100, peaking at 53.7 GB with 8-row micro-batches and Γ's gradients (E15). That makes an OOM
unlikely, though not ruled out.

**Fix (optional, the lead's call).** Right after C3, run a probe: `masked_teacher.train` with `steps=0` is rejected by
`MaskedConfig(..., warmup=3, steps=0, save_steps=())` (accepted by its assert), and discard the result. It costs seconds, changes nothing in the
model (lr 0, and `model_state` restores everything), and logs real seconds per step and peak GB. Those also feed the
time rule better than M0's own numbers would. If the lead prefers the plan's order, keep it. The cost of a late crash
is bounded.

### 4. Low: G2's 1e-3 tolerance has never been measured for this path

`amend1.py:155-166`, `masked_teacher.py:55-72`. G2 compares two forwards of the same function. One is the target path,
x·W0ᵀ in bf16, through arm 1's hooks with zero offsets. The other is the component path, (x·V)·U plus x·Δ₀ᵀ, in bf16
with every mask at 1. Exactly, the two are equal. Under bf16 their rounding differs, and the repo has no measurement of
that difference. The closest number, the unmasked KL at W0 (masks 1 with Δ left out), reads 0.012 and is dominated by
Δ, so it says nothing about rounding.

**Failure.** I expect the difference in mean KL to be well under 1e-3: the rounding errors are random across 4,096
positions, and both sides carry a similar bias. But if G2 misses, arm 2 and M0 are skipped, and the JSON cannot say
whether the miss came from rounding or from a real mismatch.

**Fix.** In `g2()`, also log and store KL(w0_net ‖ component path) on the same batch. That isolates the path difference
directly, and it should read near 0. If the lead wants a margin, set `g2_tol` with it in view before filing.

### 5. Low: in M0, the teacher KL does not start at exactly zero gradient

M0's teacher is `job.w0_net` (target path), and its student is the component path. They agree only to bf16 rounding.
So `kl_exact_zero`'s gradient q − p is small but not 0 at the start, and Adam turns any consistent gradient into steps
of about the learning rate. With the masked terms at 0.5, their gradients should dominate. At the retry coefficient of
0.05 the two may be closer in size. Either way, this part of M0's movement only pulls the component path toward the
target path's rounding. It cannot cause damage, because it lowers a KL to W0. M0's "teacher KL at loop step 1" is
logged and stored (`curve[0]["kl"]`), and the predictions expect it "about 0".

**Fix.** None needed. Read that number before reading M0's damage.

### 6. Low: the decision-tree helper ignores arm 2 when the time rule cut it to 1,000 steps

`amend1.py:273` looks only at `arm2_2000`/`arm3_2000`. If the time rule stops arm 2 at 1,000 steps and `arm2_1000` is a
solution, `branch` says "no solution read". P2 itself is correctly left unscored then, as filed. Only the reported
branch string is misleading.

**Fix.** In `branch`, fall back to `arm2_1000` when `arm2_2000` is absent, and say so in the string.

### 7. Note: arm 3's step 0 is not exactly E37's adjoint state

`amend1.py:134-147`. E37 read the adjoint state as W0 plus the first-order change, with Δ absorbing the second-order
term. Arm 3 has to start from Δ₀ with U′V′ᵀ carrying all of the change, because the Offsets trainer keeps Δ at Δ₀. The
code handles this honestly: the docstring says so, and the second-order share is logged and stored. But P3's positive
control is filed as "arm 3's step-0 read reproduces E37's adjoint state". If the share is not small, a mismatch with
E37's 0.804 and 1.05 has a known cause and is not a failed control. Whoever analyses the run should read the share
first. Optionally, the control's wording could say "up to the logged second-order share" before filing.

## Checked and found fine

- **`gamma.loss_setup` refactor.** Same loss configs (Faithfulness excluded), same `torch.manual_seed(seed)` placed
  just before the persistent-adversary states are built, same per-slot states and the same constant-schedule assert.
  `Refit` still sets every attribute that `step` reads. When `coeff` is None the configs are the original objects.
- **`component_training` edits.** With `init=None` the offsets are zeros of the same device and dtype as before. The new
  `assert init is None or rank is None` cannot fire for existing callers. The extra log line every 500 steps does not
  touch the numbers. No RNG is drawn differently.
- **`row_copy` refactor.** `copy_on_rows` now calls `copy_states` with `(steps,)`. The configs and draws are identical.
  E37's `run.py` (unchanged in the diff) behaves as it ran.
- **Arm 1 continues E37.** Batch draws come from `Generator().manual_seed(0)` and do not depend on the save steps.
  `ArmAConfig.steps` is not read by `train`. With a constant lr, steps after 1,000 continue the same optimizer state.
  `kl_before(curve, n, "loss")` at 1,000 is E37's `train_kl_end` (steps 951-1,000).
- **Arm 3's start.** `start_offsets` gives (U′ − U, V′ − V). The hooked weight is then W0 + U′V′ᵀ − UVᵀ = Δ₀ + U′V′ᵀ,
  which matches `start_dw`. The saved offsets are measured from W0's (U, V) (`save_offsets`), so they include the
  start. The toy test checks the rebuild.
- **Arm 2's gradients.** Γ is computed under `no_grad` with `detach_inputs=True`, and its parameters stay frozen
  (`load_component_model` sets `requires_grad_(False)`). Δ₀ comes from `weight_deltas` under `no_grad`. The masked
  losses' target is the detached unmasked output. Only the U and V of all 24 matrices are in the optimizer. Dead rows of
  U and dead columns of V (`alive` is a CPU bool mask) get zero gradient before the clip, so Adam never moves them.
  Gradients flow from the teacher KL and from both masked terms through the component forward. Unrouted layers in the
  subset loss read the target weights, which `_after_step` keeps at Δ₀ + V·U for the whole matrix (dead columns
  unchanged). Γ therefore reads the student's own activations, as the plan says. Coefficients are applied by the
  caller, because param-decomp's `compute_losses` returns unweighted terms, exactly as `Refit.step` does. The order
  (persistent-adversary gradients via `autograd.grad` with `retain_graph`, then `total.backward()`, then the adversary
  step) matches `Refit.step`. The released configs hold exactly the two kept classes (StochasticReconSubsetLoss 0.5,
  PersistentPGDReconLoss 0.5, per-batch-per-position scope, 2 adversary warm-up steps), with `n_mask_samples` 1 and
  continuous sampling.
- **Warm-up at lr 0.** Adam with lr 0 leaves the parameters exactly in place, and `_after_step` rewrites the same
  weights up to fp32 rounding. Saves and reads count from the first step at lr.
- **Model state between stages.** `model_state` stores clones (`target_weights` and `component_weights` both clone)
  and restores on exit. Component-training hooks are removed in `finally`. `requires_grad` goes back to False. Every
  read writes its state inside `model_state`. `ctx.uv` comes from indexing with a bool mask, so it is a copy and does
  not alias the parameters.
- **State bookkeeping.** `_state` gives δW = U′V′ᵀ − UVᵀ on the alive columns. That equals the trained target minus W0,
  because dead columns do not move and Δ₀ + (V·U)₀ = W0. `read` then writes W0 + δW with the alive columns placed into
  the released components, so Δ = Δ₀.
- **Scoring against the plan.** P1 is "verdict fail" on the PGD ratio, which is ratio ≥ 1.5(1 + f). P2 is
  s ≥ 0.9 with "pass" on both FLOOR_KEY ratios (stochastic and PGD only). P3 is s ≥ 0.9 with |a3/a1 − 1| ≤ 0.25.
  "Still falling" is a train-KL drop of at least 10% together with a falling ratio. M0 passes on damage ≤ 0.02 and a
  ratio ≤ 1.40. `pile_ratios` divides by C1's W0 at seed 0. Unscored cases are handled (missing states, `M0` without
  `state`).
- **Resume.** Each stage is skipped by name. G2 reads `out["train"]["arm1"]` from the JSON. A rerun of M0 replaces its
  record.
- **Brief.** It downloads the JSON after every scored read and each `.pt` as it is saved, checks sha256 both sides,
  and terminates on the final JSON's checksum. It forbids editing `remote-kernels.toml`, caps execute and wait at
  5 minutes, and stops on a lost plugin. It renames the logs so E37's are not overwritten. The stall rule matches the
  new 500-step log interval.

## Not checked

The GPU memory and step time of `masked_teacher.train` at batch 8, which only a GPU can measure (see 3). The bf16
difference between the two paths (see 4). pyright on the changed files (nodeenv failure on the worker). The
predictions file itself, which is not yet filed.
