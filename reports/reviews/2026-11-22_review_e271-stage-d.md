# Review: E271 (cell P2.65) Stage D, plan section and code

I read `experiments/installed_cross/stage_d.py`, `results/predictions_stage_d.json`, `cluster/installed_cross_d.sbatch`
and the appendix section "Stage D: rehearsal and pods" at HEAD 43eedfd1, plus the helpers they call. The ordering is
sound: no share or E_b is read before its faithfulness check. But the check that was added for the user's named risk
(F3, faithful on the base's cued rows) is nearly blind to the installed skill. No check looks at the CI-masked model at
the answer position, or at the uncued rows the shares are actually read on. Fix those two before the pods. The rest is
fine to run.

Reviewer: fresh Opus agent that wrote neither. Read-only.

## Findings, most severe first

### 1. F3 averages over every position, so it barely sees the cue-gated skill (fix before pods)
`stage_d.py:149-156, 222-227`. `task_pgd` runs `harness_pgd` on the 64-token prompts and returns the adversary's KL
averaged over every position of the 128 rows. The installed skill differs from M only at the answer position (the
last) of the cued rows: 1 position in 64, on half the rows. The fill positions are distilled to M, so they dominate the
mean. A decomposition that fails on the cued answer therefore moves F3 by about 1/128 of its effect, and F3 passes
anyway. F3 thus cannot fail for the reason it was added (the source line: "the base's skill lives on cued rows"). The
same dilution was fixed in the install loss under the gate-5 review's finding 5.

**Fix:** score F3 at the answer position only. `harness_pgd` takes `loss=`, so pass a KL masked to the last position.
Or keep F3 as it is and add finding 2's check.

### 2. Nothing checks the masked decomposition at the answer, or the uncued rows the shares are read on (fix before pods)
**What F1 checks.** F1 reads the student's unmasked target W_s, and only on cued rows (`stage_d.py:211-219`).

**What the shares read.** The share and E_b are read on *uncued* read rows (`RUN.cue_read`, `stage_d.py:236`; `eb_item`
toks). They are read through the CI-masked model, using L00 and L01 under M's Gamma with the student's components.

**What is missing.** No check covers either of the following:
- (a) whether the masked model at the base's own gates (L00) reproduces the base's answer (T0) on cued and uncued rows;
- (b) whether W_s matches the base on uncued rows (a student that leaks the skill uncued shifts T0 for every state).

**Fix:**
- Add F1 on uncued read rows: the teacher gap, and the uncued shift within the lock's 0.2 decades.
- Add an F4: mean |L00 − T0| at the answer on cued and on uncued read rows, against M's released decomposition on the
  same rows, measured in ctl as the bar.

`shares()` already computes L00 and T0 at the zero update (`stage_d.py:255`), so F4 costs almost nothing.

### 3. One unfaithful base empties the whole score, not "its rows"
Every triple of every task contains all three moved bases (diagonal plus two off-diagonals). If one base fails
faithfulness, all 9 triples become nan and every reading reads "not read" under `min_finite` (6 triples).

The plan and `faithful_rule` say instead that "the base's share and E_b rows stay empty". That suggests the other two
bases still give a result, which they do not.

**Fix:** either state that one failure ends the Stage D score, or file a fallback for two bases before running. For
example: per task, the diagonal against the one remaining off-diagonal, plus an AUC on the faithful bases only.

### 4. The pod-hours estimate assumes the H100 equals the L40 in fp32, and the long eb job has no mid-run download
**Speed.** TF32 is off (`RUN.Ctx`), so the scan runs in plain fp32. An H100's plain fp32 (~67 TFLOPS SXM) is below an
L40's (~90 TFLOPS), so H100 pods may be slower than the rehearsal card, not equal to it. Per pod, 21 items × t_item,
with t_item still unmeasured (E268's estimate is 8 min).

**Fix:**
- Take t_item from the rehearsal's per-site log lines (`needed.scan` logs cases per second per site, so even a
  rehearsal that times out leaves a rate).
- Then either use L40 or L40S pods, or time the first item on the H100 with a stop rule before committing to the
  2.9 h figure.

**Downloads.** `eb BASE` is a single ~2.6 h job per pod. The appendix says "download after each job", which here means
one download at the end. That is the 2026-10-19 lesson (CLAUDE.md).

**Fix:** run eb in named-item chunks (the command line takes item names; resume works per item) and download
`stage_d/eb_<base>.json` after each chunk.

### 5. An empty K reads as maximal reuse
`SI.explained_share` returns 1.0 when K ∩ changed is empty (`stand_in.py:154-158`). If the scan finds no needed
variable for a state, `E_b_K` = 1.0, which is the top "reuse" value in the score. The same applies to a Gaussian
floor item. For dense updates the changed set is nearly every variable, so this happens only if K itself is empty, but
nothing records or excludes that case.

**Fix:** in `eb_item`, set `E_b_K` to nan when `n_changed_K == 0`, and list that case in `edge_rules`.

### 6. F3's floor makes "pass" very hard when the seed spread is large
`floor3` = max/min − 1 over 3 adversary seeds on one 128-row block (`stage_d.py:225`), and `verdict` passes at
≤ 1.5/(1+f). A noisy block (f around 0.5) needs a ratio ≤ 1.0, and "inconclusive" counts as not faithful. F3 can then
fail on noise rather than on the decomposition. (Finding 1 changes what F3 measures; this concerns its bar.)

**Fix:** record f3 in the rehearsal before the pods. If it is far above the Pile floor (~0.21), use more rows or more
seeds for the task-row block.

### 7. Small items
- `student_state` (`stage_d.py:188-192`) loads the student without checking its recorded sha256 (run.py's `load_dw`
  does). On a pod, check it.
- F1's `gain_kept` (`stage_d.py:218`) measures cued gain over M *uncued*. The amended lock measures it over M *cued*.
  The ratio largely cancels the difference, but use one convention.
- `gain_kept_min` 0.8 is translated from E34's P ≥ 0.9, not taken from a measured floor. It is acceptable as a
  sourced bar; say so in its source line.
- `job_score` loops over kinds ("size", "gain"), but E_b has no "gain" items, so E_b at gain always reads "not read".
  This is harmless, but drop that row.
- The held-out curve in `train_student` uses the first 64 cued read rows. It is logged only, with no checkpoint
  selection, so there is no fit-on-read leak.

## Checked and correct
- **Order.** In `job_teach`, faithful is written (`o.write`) before `shares()`. `job_eb`'s `decomposition()` asserts
  `teach_<base>.json` faithful pass. Shares and E_b on a moved base cannot run before the check.
- **Rows.** The student is fit on train items and Pile train windows. Faithfulness and every reading use held read
  rows, and the harness uses Pile val. There is no fit-on-read overlap.
- **Reference weights.** The student is trained from M (`W0 = M`, model at M). `W_s = M + ΔW` and the student's
  components carry M's Delta, so `weight_deltas` under (W_s, comps) is Delta_M. `share_cells_by(model, W, dW)` runs
  under `model_state(target=W, components=comps)`. `eb_item` featurizes (W_s, W_s + dW) with the student's U, V and
  Delta. M's row uses the released decomposition.
- **Floors.** The share's floor is the Gaussians' numerator, and E_b's floor is the Gaussian E_b. Both are finite
  (finding 5 aside). `SC.verdict` reports "floor not read" rather than a pass.
- **Resume.** Each item goes through `o.put` with merge keys share, eb and student. Teach skips training when the
  student is recorded.
- **Files.** Two filed files, two variables (`filed()` swaps them correctly). Pod inputs match the loads I traced:
  `val_0_64`, `val_0_256`, `train_0_4096`, `alive.pt`, `tasks.pt` plus the run's `ctl.json` (the sha check), all
  `cross_*_*.json`, and the base's and M's states.

## Earlier findings (gate-5 review), fixed in 00a7676d
1. Controls on CPU: fixed (`out()` builds its record without CUDA).
2. Gaussian floor degenerate: fixed (floors on numerators; "floor not read" is never a pass).
3. Yoruba split: fixed (word-start answers, no end-of-text in the last 4 tokens, seeded permutation before the split).
4. Yoruba headroom: fixed (bar 3.0, `g_star_min` 1.5 enforced in the lock).
5. Install answer-position KL: fixed (answer-position KL weighted as the CE). Early check every 100 steps with a leak
   stop at step 500: fixed.
6. Cue effect on M: fixed (cued gain over M cued; the cue effect reported with its standard error).
7. Gaussian stop on resume: fixed (`gauss_check` on the record read back).
8. Lock (c): fixed (decided from the numbers; M's reach recorded).
9. Lock rows: fixed (lock, λ and G* on the stop rows).
10. Scoring: fixed (size skipped at kind size; P corrected to 0.0083, with dependence stated; AUC per task).
11. Small items: fixed (`tasks.pt` sha checked, docstring corrected, card match in controls, " =" named).

## Verdict
**Fix before the pods:** findings 1 and 2 (the faithfulness checks), 3 (state or file the two-base fallback) and
4 (card choice and chunked downloads).

**Fine to run:** everything else, including the $0 cluster rehearsal as it stands. Running it is how t_item and f3
get measured.

## Not run
- Any GPU path.
- A measurement of F3's sensitivity to a broken cued answer.
- The K scan's time.
