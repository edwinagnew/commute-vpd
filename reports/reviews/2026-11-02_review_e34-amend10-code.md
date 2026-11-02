# Review: E34 Amendment 10, code and filing, before any run

*2026-11-02. Independent review at commit `5fc87377`. I wrote none of this and have not run the job. Files: `experiments/teacher_training/amend10.py` (below, bare line numbers), `results/predictions_amend10.json`, `tests/test_teacher_amend10.py`, `cluster/teacher_amend10.sbatch`, `reports/plans/2026-11-02_plan_e34-amend10-input-weighted.md`. No model loaded, no job submitted. My scripts and outputs are in `~/commute-data/scratch/commute-lead/e34a10-review/`.*

**Verdict.** The arithmetic is right and I found no crash on the GPU path by reading it. Two things should be fixed before the job: the power control will probably stop it (finding 1), and a noisy input set reads "copies" by default (finding 2). The bigger issue is what the verdict is allowed to say. The run measures the angle between two changes at each matrix's output, on W₀'s inputs, against the angle a rerun reaches. That is the comparison the user asked for. It does not show either sentence as worded: "copies" ignores length and has no floor, and "reproduces its behaviour" is not read here at all (finding 3). Nothing has run, so every fix below can still go into the filing.

## Ranked findings

### 1. High: the power control will probably stop the job on the first task set

`excited` and `power_change` (114-130), `power_sigma_min` (77), the stop at 408-409.

**Failure.** With Z = T Q Qᵀ + R, cos_Σ is about 1 / sqrt(1 + r), where r is R's Σ-energy over T's. For a read side not aligned with Σ, r is about rho2 × tail × d / (d − k), with k the count of directions kept. So cos_Σ ≥ 0.99 needs k ≤ 0.61 d: 40% of the input directions must together hold only 0.1% of the trace. The filing checks this only at k = d/2 (`threshold_sources`, "under 2% at k = d/2"), and the test uses a Σ of rank 5 in 40 (test file:111).

An earlier read says the real spectrum is not like that. `lora_plateau.json` → `checkpoints.1.read_side` gives, on the emoticon training windows:

- `h.0.attn.o_proj`: the top 57% of directions hold 95.3% of the trace.
- `h.2.attn.v_proj`: the top 66% hold 94.8%.
- `h.3.mlp.down_proj`: the top 60% hold 98.3%.

So the bottom 34 to 43% of directions hold 1.7 to 5.2%, not 0.1%.

I ran the filed functions (`power_sim.py`) on the real δW, with each matrix's spectrum fitted to those two numbers in a random eigenbasis:

| Spectrum fit | k/d | Pooled cos_Σ | Matrices above 0.99 | Pooled Frobenius |
|---|---|---|---|---|
| Power law | 0.97 to 0.995 | 0.74 | 0 of 24 | 0.33 |
| Exponential | 0.16 to 0.99 | 0.95 | 7 of 24 | 0.26 |

This is an estimate. The real Σ is unread, the fit is crude, and pooling weights by trace are unknown. A δW aligned with Σ's top directions would help, but passing from the exponential fit needs alignment of about 5 times. `in_data` reads about `random` on every matrix of that file, for the LoRA δW it was read on.

**It also lands badly.** The order at 461-463 writes `trigger:emoticon`'s reads before the first power control runs. A stop there means the control gets amended after an input-weighted read exists, which is the state the revision note is proud of avoiding.

**Fix.**
- Define "not excited" by eigenvalue, not cumulative trace: `vals < eps * vals.mean()`, eps 1e-3. Take Z = T + R with R on those directions. Then r ≤ rho2 × eps, cos_Σ ≥ 0.996, and Frobenius stays 1/3.
- Run `task:` before `trigger:` for each task.
- Log k/d per matrix.

That control only says Σ is ill-conditioned, so it is weak. Finding 9 gives the control that bears on the verdict.

### 2. High: noise reads as "copies", not as "cannot tell"

`table_label`, 145-147.

**Failure.** There is no band between the two labels. A larger margin lowers the "copies" line, and "cannot tell" fires only when the references fail to separate. I ran it:

- `table_label(0.5, 0.8, 0.1, 0.3)` gives "copies".
- The same reads with m = 0.05 give "different change".

So one noisy split-half flips the verdict toward the claim. The margin is pooled over all three task sets and the rules read a max over many cells, so it can be large on 36 to 48 windows a half; the filing leans up to 0.15 on task windows. At m = 0.15, a student 0.15 under the rerun reads "copies". `test_score_reads_both_ways_and_third` tests only noise = 0.5, where the gate catches it.

**Fix.** `own >= rerun - gap_min`: "copies". `own < rerun - m`: "different change". Otherwise "cannot tell". The gate stays. Add a test at m = 0.3.

### 3. High: neither wording follows from its label

`wording` (167), P1 in the filing, plan lines 59-62 and 113.

**"Copies the fine-tune's change on real inputs".**

- *No floor.* `table_label(0.16, 0.2, 0.05, 0.05)` is "copies" by the rule (my arithmetic, from 145-147). A task whose reruns are nearly unrelated gives "copies" at a cosine of 0.16.
- *Length is ignored.* The plan says "a student no further from δW than another run". It is not: only the angle is read. Amendment 9 has students at 1.5 to 2.3 times the fine-tune's length on the map tables and null, by weight. At length 2.3 and cosine 0.6 the distance from δW is 1.88 δW-lengths. A rerun of length 1 at the same cosine is 0.89. Both would read alike here.
- *The fix is free.* The saved Gram already holds the sizes.

**"Reproduces its behaviour by a different change".**

- `table_label(0.70, 0.80, 0.1, 0.05)` reads "different change" at cosine 0.70, half the squared length shared. "Different" means only "further than a rerun".
- The behaviour half is not read in this job. The plan cites the real task's 0.0286 nats and nothing for the map tables, though the wording needs them too. E12 Amendment 2 reads the second table's students as keeping P(target) and losing about a fifth of the re-switching term, so the sentence is not simply true there.
- A difference at each matrix's output can cancel downstream. The filing lists this as blind, but the blind spot is exactly the second half of the sentence.

**Fix.**
- Have the code emit what is measured: "as close to the fine-tune as a rerun, by angle at each matrix's output on these inputs", and "further than a rerun".
- Let the report build the closing sentence from that plus the cited behaviour reads.
- Add two reported numbers per student and per rerun: ‖S‖_Σ / ‖δW‖_Σ and ‖S − δW‖_Σ / ‖δW‖_Σ. Both are a few lines from `pool["whole"]`.
- State in P1 that "copies" requires size within a filed band, or say plainly that it does not read size.

### 4. Medium-high: the headline fuses two families whose references differ, so its likely outcomes are non-answers

`wording` (165-178), `score` (236-241).

**Failure.** The wording needs 32 labels alike: 12 per input kind on the map tables, 4 per input kind on the real task. Any one out of line gives "mixed", then "cannot tell". The two map tables already differ by 0.09 to 0.14 by Frobenius, above `gap_min`, so a split between tables is plausible.

Then the families. A real rerun at 1,000 steps may read high: E53's cipher reads 0.84 at 300 steps by Frobenius, and the two real students' batch orders read 0.927. The map-table reruns are at 10 and 30 steps and read 0.61 and 0.68. Real students read lower than map students by Frobenius. So "different" on the real task with anything else on the map tables is a live outcome. That gives "depends on the task" or "cannot tell", and the filing admits the former cannot separate task from length.

**Fix.**
- Emit one wording per family. The real-task family is the case the headline's behaviour number (0.0286) belongs to.
- To make "depends on the task" readable, add the length curve the filing lists under "Not run": one map-table fine-tune at 1,000 steps, two seeds, about 70 s. Its rerun cosine on the same Σ separates task from length directly.

### 5. Medium: one rerun gives the line, and its spread is in no margin

`noise` (198-199), `rerun_stage` (417).

**Failure.** Every label compares against a single rerun cosine per family. The margin's batch-order term measures students, not fine-tunes. If rerun cosines vary by 0.1 across seeds, labels near the line are unsupported and nothing shows it.

**Fix.** `for order in (0, 1, 2)` gives three real rerun cosines for about 35 s more. A third seed on each map table costs seconds. Take the line as their mean and put their range into the margin. The filing lists this as not run but gives no reason, and the job is free.

### 6. Medium: the wording leaves out where the behaviour is taught

P4 (filing), plan line 43.

**Failure.** Both trainings weigh the trigger equally with the mean of all other positions (`finetune.arm_a_weighting`), and the fine-tune's KL term holds those other positions still. Task windows put the trigger at 1 position in 41, and Pile barely contains it. So the scored Σ reads mostly what each change does where it was trained to change nothing. P4 stays out of the wording and leans "cannot tell", in which case the wording stands with no qualification about the trigger. The emoticon held-out set also has only 4 distinct trigger tokens, against 10 in training.

**Fix.** Add one read on the training's own weighting: Σ_trigger + Σ_task, the sum of the two Grams already saved, with no new forward. It has full rank, so it avoids P4's noise. If you keep it out of the wording, the scope string should say the wording is not about the trigger.

### 7. Medium: P3 reports noise as a finding

`seed_class`, 154.

**Failure.** There are two outcomes. `seed_class(0.5, [0.4], 0.2)` gives "follows the task only", though own and other differ by 0.1 and the noise is 0.2. The filing then names the wording "copies the task's change, not this run's". By Frobenius the gap is 0.05 to 0.07 against a least margin of 0.05, so this is the likely path. P3 cannot say "not separable".

**Fix.** "Follows the task only" needs m at most a filed cap, else "cannot tell". Add a test.

### 8. Medium-low: the margin's split-half is narrower than filed, and correlated on the real task

`noise` 195-197, `read_set` 363.

- The filing says "over every cell the rules read". The code leaves out `other_task` and the other seed's students. Either add them (one comprehension each) or change the text. They bear directly on the gate and on P3.
- `rows[::2]` splits by window. On emoticon the held-out windows are sorted by source row, and 13 of the 49 rows land in both halves (`rows.py`), so the halves share text and split-half understates noise on the set the real verdict reads. The map tables share none. Split by `w.row` instead; the attribute exists.

### 9. Medium-low: the ceiling is unread, the stated reason for that is wrong, and it is the missing control

Plan line 23, `the_inside_part`, `can_read_either_way`.

**Failure.** The plan says the inside part "is no longer an orthogonal part" under this inner product. That is true of the Frobenius inside part only. The input-weighted one is the project's own outside-both block (`GOALS.md`, `split.four_blocks`): (I − P_U) δW a₂ is what no first-order move of the subcomponents can produce on these inputs. Since a student's change is in that span to 5 decimals, cos_Σ(S, δW) ≤ sqrt(1 − outside-both). So the claim "neither wording's truth follows from S lying in what the subcomponents can write" is checkable, and it is not checked.

I did check it roughly. The fine-tune's outside-both reads 0.055 to 0.072 at steps 10 and 30, and 0.131 at step 1,000 (`2026-10-20_analysis_lora-split.md`, 20 non-saturated matrices). So the ceiling is about 0.93 or higher, and "different change" is forced only if a rerun reads about 0.98 or more. This is probably not Amendment 9's flaw again, but on one earlier Σ only.

**Fix.** Report the ceiling per fine-tune and Σ. It is a few lines with `split.bases` and `four_blocks`, and the CPU time is small. A rule: if rerun − m is above the ceiling, "cannot tell". It also supplies a positive control for the "copies" branch, which the power control does not test.

### 10. Low

- **Stale filing.** `measurements.ladder` promises E53's rerun on Pile; the code has no such read and the revision deleted it (grep of `amend10.py`: nothing).
- **`top_matrix_share`** is promised "beside the wording" but left to the analysis. In Amendment 9, four matrices held over half the inside part. Two lines in `read_set` put it in the results file.
- **`REFS[EMO]`** (63) and Pile's refs (464) carry `REAL_8C[1]` as a reference no rule reads.
- **`P1_wording`** (241) ignores P3, as filed; the "copies the task's change" wording exists only in the report's hands.
- **GPU eigendecomposition** (116) of a 3,072 fp64 matrix is unmeasured here, and it sits in the 12-minute estimate. A 30-minute kill keeps finished sets, so the cost of being wrong is small.

## What could be deleted

The script is about 470 lines, and many run past 150 characters, so it is longer than it looks. These would take it to about 400:

| Lines | What | Saving | Note |
|---|---|---|---|
| 114-130, 365-366, 394-409 | Power control | 35 | Replace with k/d logged, or about 12 if fixed as in finding 1 |
| 378-386 | Self, zero and asymmetry stops per set | 7 | True by construction; tested in `test_arithmetic` and read at stage 0. Keep `smallest_diagonal` |
| 234 | `null_own` | 1 | No rule |
| 299, 403, 423 | `assert isinstance` after `sum` | 4 | A small `pooled(...)` helper does it once |
| 136-138 | `rerun_ok` | 3 | Inline it; the test goes with it |
| 250-259, 281-283 | Rounding draw | 13 | Can bind only at 5 times its own lean |

I would keep the rounding draw: the saved states are bf16, Σ could amplify it, and precision has bitten this project (E12 Amendment 3 had to rule it out). 300 lines is not reachable without dropping a filed read. P3 and the centred form are the next candidates, and I would keep both.

## What is fine

- **Inner product.** `gram`, `mean_part`, `cosines`. Hand-checked and tested: tr(A Σ Bᵀ) = E⟨Ax, Bx⟩, centred = whole minus the mean's part, and the whole's Gram is the correctly weighted sum of the halves. Tests pass: 12 of 12.
- **The student's change is the right object.** `component_training` trains W₀ + S with Δ fixed and no masks in the forward, so S x is what the student's forward adds. E12 Amendment 2's hook and weight forms agree.
- **Layout and cache.** `cache_type="input"` stores the module's own `args[0]` from a forward hook, on the keys of the 24 matrices. The 67M's linears have no bias (`vpd.target_linear`), so `x (W0+T)ᵀ` matches the module's output and the layout control is sound. `model_state` restores W₀.
- **The rerun at order 0 should reproduce.** The recipe at 419 matches `amend8.real_teacher` field by field; the windows and token are the same files; `eval` consumes no generator; `_run` restores W₀ exactly. Three jobs (n1, n3b, 8c) log the same T1000 to 4 decimals, and 8c's order-1 cosine matched n3b's at 0.927312 against 0.927312. All compute cards are L40 (`sinfo`), so the card risk in the filing does not arise.
- **Names and cells.** `OWN`, `other_seed`, `REFS`, `POWER` and the keys `student_read` asks for all exist in what `named` writes, on every set kind. A missing state reads nan and goes to "unscorable", as tested.
- **`wording`** covers all 16 two-label cases and every non-label.
- **Windows.** Held-out and training windows share no source rows and no identical window, on all three tasks (`windows.py`, `rows.py`). Sizes are 73 and 97 by 41, `half` is 20.
- **Job script.** `env.sh` exports a non-empty `PYTHONPATH`, so line 24 survives `set -u`. The exit code survives `tee`. `val_0_256.npy` exists. The time and memory asks look comfortable on the known costs (fine-tune 35 s logged).
- **The mean.** Reading centred beside uncentred with agreement required is a sound guard. Using W₀'s x for every pair is defensible as filed: only the S δx term is second order, and the rest is what the filing lists as blind.

## Not checked

- The real Σ's spectrum and any real `read_set` output: that needs the forward I was told not to run. Finding 1 is therefore an estimate. A one-window-set $0 debug job printing k/d per matrix would settle it before re-filing, if the coordinator wants that.
- Whether task-window split-half margins will be small: unread.
- `make lint` on these files.
