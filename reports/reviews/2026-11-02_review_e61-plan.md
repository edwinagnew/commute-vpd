# E61 plan review, gate 5 (2026-11-02)
*Independent review of `reports/plans/2026-11-02_plan_e61-direct-share-size.md` with its amendment, against gates 1-4. Files and code read, summed rates computed with `awk`; nothing run, no model loaded. No Job R or Job S code or predictions file exists yet (`experiments/qwen_recipe/results/predictions*.json` hold stage A and the checks), so code points are the code those jobs reuse.*

The recipe fix and arm 0 hold up. The plan is not ready to file: the invented-facts arm is short of summed rate by the project's own runs and has no branch if it fails, Job S as reused would run out of host memory on its first state, the verdict has no defined reading on the second size axis or when the second gold run is smaller, and the resume rehearsal rehearses nothing. All fixes are small and cost $0.

## Must-fix
**M1. Invented-facts arm 4 probably does not learn, and nothing says what happens then.** Summed rate, by `lr_at` (`qwen_sft.py:192`):

| run | summed rate | training loss | recall |
|---|---|---|---|
| `a2_s0`, 1e-5, 100 steps | 5.0e-4 | 0.221 | 0.319 (`E_a2_s0.gen.100`) |
| `a2long_s0` at step 100, 1e-5 over 300 | 8.8e-4 | 0.031 | 0.91 (`E_a2long_s0.gen.100`) |
| arm 4, 3e-6, 300 steps | **4.5e-4** | ? | rule: at least 0.8 |
| 6e-6, 300 steps (same time) | 9.0e-4 | ? | ? |

More steps do help at equal sum (`a2_s0` 0.221 against `a2long_s0` about 0.39 at step 57, also 5e-4), so 3e-6 may beat 0.32; it needs training loss near 0.03 to reach the rule, and 0.22 gave 0.32. The plan gives a fallback only to the template, so about 100 minutes can end in "did not learn" with no next step, and the arm has no mid-run value. *Fix:* run 6e-6 for 300 steps unless the user fixed 3e-6 specifically; either way file the sum table, an expected loss at the pause at step 150 (3e-6: sum 3.6e-4, `a2long_s0` reads about 0.55 there), and the branch: a failed arm reports "this sum is too low", and 1e-5 for 300 steps under the fix (about 100 min) runs only if the user asks.

**M2. Job S's first state would exceed host memory.** `true_kl` (`checks.py:324`) holds every row's masks on the host, "about 1.2 GB a row": 19 GB at 16 rows, about 154 GB at 128, on top of E57's 72-93 GB, plus about 20 GB for `refs` (`checks.py:361`, 155 MB a row by my arithmetic). `qwen_reading_C.sbatch` asks 128G and the user cap on `normal` is 384 GB (`cluster/README.md:35`). The job would die about 25 min in. The amendment names `true_kl` and scales its time, not its memory. *Fix:* compute and use each row's masks in turn (a mask pass at W0, then enter `applied` for that row), or keep them in bf16 on local disk; keep per-row sums for the bootstrap. Rehearse on 2 rows (M7).

**M3. "B needs the line met on both" size axes has no defined reading** (plan:68, 70-73). One matched pair cannot be matched on both axes. Using `checks_sizes.json` ratios of K00 − K0 to KL from W0 (gold 1.79, Gaussian 1.63, alive 1.30): gold at K00 − K0 of 0.63 reads KL about 0.35 against the alive change's 0.48, and alive scaled to 1.44 reads about 1.11 (if linear) against gold's 0.81, 38% off; "the higher random" is usually the alive one (0.029 against 0.022). If the same pair is reused the gap is one number and the second axis is vacuous; if new pairs are needed, nothing says which or who runs them. *Fix, no new run:* each random kind has two reads, so interpolate its share linearly in KL from W0 at gold's KL; below its lowest read use that read (conservative toward C). Compute the gap on each axis; then B and C apply as filed.

**M4. The scale search as named does not match size.** "By search on 16 rows" (plan:62) with "same code path" means `stage_C0`: it searches KL from W0, not K00 − K0, with `kl_match` 1.5 (`qwen_reading/run.py:358-364`, `scale_search` stops at the first value within 1.5 times; `C0.json` `.search` stopped after 2 evaluations at 0.532 for 0.618). A pair 1.5 times apart in size answers nothing. 16 rows also differ from 128 by kind: gold 1.503 against 1.444, Gaussian 0.843 against 0.857 (`checks_k10_*.share`, `checks_sizes`). *Fix:* search on K00 − K0 against gold's own 16-row value, match 1.05; check 128-row sizes within 15% afterwards; on a miss, one more search step, then report unmatched.

**M5. The "second run" floor can contradict the line, and it is not a spread.**
- `gold_s0` and the new run share data and differ in the forward, so the difference is a forward effect mixed with any size difference. It is not run-to-run noise.
- "Replaces the floor" (plan:75): with a difference of 0.035, a gap of 0.033 is both at least 0.03 (B) and at most the floor (C), with no branch.
- Like-for-like is missing: the amendment adds 128-row true-mask reads for the two random changes only, so `gold_s0` has 0.092 by E57's masks and 0.089 on 16 rows by true masks. Reading it against the new run's true-mask share adds a path difference of 0.002.

*Fix:* compare the two gold runs on E57's masks (both exist); use the difference as floor f only if their sizes are within 15%; lines become C at gap ≤ f and B at gap ≥ 3f, with f = max(0.01, difference).

**M6. Gold's size is assumed; three outcomes have no branch.** The plan's expected line uses 1.44, and step 100's size under the fix is unmeasured (Fable: "not verified"). Filed nowhere:
- *fp32 gold fails its rule:* Job S still reads it. Stop and report before Job S.
- *Step-100 size at or under 0.99:* it sits in the small band, so "both sizes" collapses to one. Read `gold_s0` (1.44, reader-robust, its true-mask read added) as the big state and scale the randoms to its size.
- *Size between 0.99 and 1.44:* scale to the fp32 size, as written. The plan's expected 0.04-0.05 at 1.44 is also off: a share proportional to size gives 0.037 (Gaussian) to 0.067 (alive), and the alive figure leaves gold's gap at 0.025, under the B line.

**M7. Gate 3 is not met, and the resume rehearsal rehearses nothing.** "Killing arm 0 at step 5" (plan:86): `train` writes a checkpoint only at `t % pause == 0` (`qwen_train.py:139`), `t_chunk` is 150 and `Config` asserts `t_chunk > steps` (`job.py:97`), so step 5 has no checkpoint. The rerun takes `stage_T`'s fresh-start branch, deleting E and δW (`run.py:283-286`), and "passes". The same holds for "resume per arm" (plan:82): 100-step arms have no saved Adam state. Arm 0 also has no generation (Fable table), and Job S's new parts are never rehearsed. *Fix:*
1. Arm 0 with `t_chunk` 5: pauses by exit 3 at step 5, resubmit, steps 6-10 run; expect the restored norms, rate and Adam step count (asserted in `resume` and `one_step`).
2. Run the W0 reading (all rule prompts: unprompted, prompted template, facts, 4-shot) straight after arm 0, on fp32 through the same code, stopping on memory. This rehearses generation before 100 min of training.
3. Job S on 2 rows and two ladder states end to end before the 128-row job: selection, search, both masks, both KLs, reverse KL, bootstrap, write, resume.

**M8. The template accuracy line can fail a state that learned, and its corrected source is on other rows.** The number is true: `P.json .runs.format.accuracy` 0.680. But it was read on 512 *train* search problems, where the unprompted model reads 0.834; the rule reads *test 0-255*, where it reads 0.779 (`control.json`). So the teacher on the rule's rows is about 0.62-0.64 (unread, scaled by me), and the 0.60 line sits about one standard error (0.03) under a perfect copy. On file: `a1_s0` step 100 0.642, `a1_s1` 0.667, fp32 `a1_s0` step 19 0.597 (64 problems). "Less 0.08" was picked to keep the 0.60. "The lowest passing peak is the recipe" can therefore turn on noise, sending a good rate to a 39-minute 1e-5 fallback. "Strict accuracy" is also unnamed: `strict_dropped` and `strict_all` differ by 0.02 at fp32 step 19. *Fix:* read the prompted W0 on test 0-255 in fp32 in the W0 reading, set the line by the same subtraction from that number before any arm reads it, and name `strict_dropped`. fp32 W0 "beside" the rule (amendment) is unprompted, so it does not do this.

**M9. A filed first-minute value is wrong on file.** Fable section 2: "a1 step-2 loss 0.96-0.98". `T_a1_s0` step 2 is 1.108, at W0 (step 1 runs at rate 0) on the same batch; the 0.974 is step 1's batch. Coded as a stop, arm 1 stops at step 2. Batches vary by 14% on this task. Gold's 2.41 ± 0.05 passes `T_gold_s0`'s 2.364 by 0.004. *Fix:* file each step's expected value from that step's own batch in the E45 record (steps 2 and 3 at W0 or nearly so), tolerance from the fp32-autocast gap at W0.

## Should-fix
**S1. "Right learning rate" is only partly answered** (item 2).
- *Template.* A pass at 1e-6 has no lower bracket: "1e-6 or less works". A pass at 3e-6 alone gives a bracket 3 times wide. Nothing compares with 1e-5 on the same reader, so 0.75 against the reference's 0.92 still "passes". The review's reason for lower rates, the same state with less movement, is read by no rule. By summed rate arm 2 ends near `a1_s0` state 10-11 (5e-5; 5.5e-5 at 11), between template 0 at 8 and 0.94 at 19; arm 1 near state 20 (1.5e-4).
- *Gold.* One rate. A pass says 1e-5 learns, which `gold_s0` already showed, not that it is right: 0.9 of the drop came by summed rate 4.5e-5 of 5e-4, the same argument that lowered the template. That may be acceptable, since the user named the synthetic tasks, but the report must say the rate was not tested.
- *Fix, $0, about 20 min:* read in fp32, on the same problems, the saved `a1_s0` step 100 (template, accuracy, 256 problems), `a2long_s0` step 300 (recall, 4-shot), `gold_s0` step 100 (256; only 64 on file). Report |δW| and maths KL beside each rule. Trained on bf16, they are still the references for "as good as 1e-5".

**S2. Gold's ladder is too sparse for the bands.** E45's tracker read 0.201, 0.353, 0.566 at steps 7, 10, 15. At gold step 100's ratio of about 2.1 (1.444 / 0.689) that is K00 − K0 of about 0.42, 0.74, 1.19 against bands 0.53-0.72 and 0.73-0.99. Only step 10 lands, at the edge of its band. Any shift under the fix means two reads and an interpolation. *Fix:* also save 8, 9, 11, 12, 13 (2.8 GB and about 9 s each). Select by a 16-row reading of K00 − K0 (about 16 s each). If two states qualify, file which is read: nearest a random's size.

**S3. The in-band state's "learned" check cannot fail, and is undefined.** "Loss part scaled to where it is read" (plan:61) has no number. Unscaled, on the bf16-trained run, drop 1.58 is passed by about state 6 (1.569 at 5, 1.679 at 7), before the band. Length does not move there: 186 at state 9 against a limit of 119 (0.6 × 198). So this state is, by behaviour, on the way. Fable's edge case (not learned → use the scaled-up arm) also differs from plan:61. *Fix:* delete the check; report its drop and length, and call it a state on the way.

**S4. Plan:75 imports Fable's edge cases, written for a different line.** Fable's line reads one size with a conditional random run; the plan reads two sizes with all runs. "The ladder brackets neither → straight to the scaled-up arm" leaves "both sizes" with one. *Fix:* write the edge cases into the plan and walk them when filing, as `.claude/CLAUDE.md` asks.

**S5. Gate 4 gaps.** Arms 2 and 4 and all of Job S have no first-minute values. Arm 1's stop (over 0.9 at step 10) cannot see blindness: its state 9 has sum 1.08e-5, and the bf16 run at sum 1e-5 read 0.579 in training (`T_a1_s0` step 6). It catches only no movement. This is acceptable only because arm 0 stops first; say so. *Fix:* arm 4 step-2 from `T_a2long_s0`'s batch and the pause value of M1; arm 2 same rule as arm 1, scaled by sum; Job S: `k10_K00` equality, the first-row memory rule, and the 128-row size check.

**S6. The memory stop reads the wrong step, and 37 GiB double-counts.** Adam's moments are allocated at step 1's `opt.step`, so step 1's backward peak excludes them: about 21-27 GiB, and the 40 GiB stop cannot fire there. `grad_pass` already ran inside `trainable` (`run.py:133`), so its 20.85 GiB includes W0 and the clone; expected is about 31.4, not 37 (20.9 + 10.5 + 5.2). It was measured on own's batch; arm 4's rows are 494 tokens and fill micro 4, about 34 by scaling micro 8 against 4. No danger of running out, but the stop should read step 2, per arm.

**S7. 4-shot generation will not fit in fp32 at the old batch.** `check_format.py:70` uses `greedy`, sized by `kv_tokens` at bf16's 114 kB per token, and peaked at 45.0 GB in bf16 (`qwen_updates/results/run.log:981`) with prompts of 4 shots. fp32 doubles the cache. *Fix:* halve the batch for fp32; budget the time (202-310 s per state in bf16, more at half batch). Generate after training: with Adam's 10.5 GiB, even plain fp32 generation (31.4 GiB, `checks_gen` log) reaches about 42 of 44 GiB, so Fable's mid-run "generation at 20" should read from the saved file in a later pass.

**S8. Times and limits.**
- Arm 4 is about 105 min, not 85: 300 × 14.5 s = 72, maths KL 4, held-out 7.5, load and resume 5, checkpoint 1 (51 s, `qwen_updates/results/run.log:448-449`), facts and 4-shot at the end and at W0 about 15. It still fits with the pause at step 150.
- Job S states with true masks: about 25-35 min (E57's C took 23-25) + 15 (true pass, 6.9 s a row, gauss log) + 2 for KLs: about 45 min. The whole programme is about 7-7.5 h, not 6. $0 either way.
- `BUDGET_S` 1500 and `lab.budget(300)` in `k10_read` (`checks.py:41,359`) are set for 16 rows; the true pass is not resumable within a pass; write per row or per arm.
- Normal QoS allows 24 h, so 2 h jobs fit. The fallback when the queue is not empty is unwritten: Fable's "split at step 50" needs `t_chunk` 50, refused by `job.py:97` for 100 steps. 31 jobs were pending on `compute` at this review.

## Notes
- **Arm 0's stop: no fault.** Still rounding reads 0.0556 (`T_own_s0` step 4); a TF32 forward also would (half-spacing 7.6e-6 above state 3's δ). False firing needs state 3 near state 2 (0.065): one unit-clipped gradient among three cannot do that, and the recorded path reads 0.2166. Values match `A.json` and `T_own_s0`. 0.08-0.12 has no branch: say it continues with a note.
- **Fp32 is not the only difference from E45.** Micro 4 against 8 is the other: same sum, padding causal, harmless, but `.claude/CLAUDE.md` asks that it be listed. A 10-step schedule is the same for steps 1-10 (`lr_at`), and `cycle` gives the same first 640 indices. Fable's "bf16: 3.6 until step 8" is off (5.14 at step 5, 7.56 at 6); "above 5 by step 4" still separates, at 2.97.
- **"Peak" is nominal.** It applies from step 11 and the 100-step mean is half of it; Adam ignores a steady scale, so clip at every step does not change the applied rate, it stops early norms of 125 inflating the second moment. RMS δ over summed rate falls from 0.84 at gold's state 2 to 0.18 at step 100, so summed rate is an upper limit on movement. Report sum and |δW| with each peak.
- **Rules walked.** Template: W0 (rate 0) and state 8 (rate 0) fail; states 19 and 100 pass the rate; only the accuracy line is at issue (M8). Gold at the last state: drop passes from state 6, so only a broken run fails it; length (0.48 of W0) has teeth; 0.6 has no stated source (E45 filed 0.8). Facts: W0 0.015, 0.32, 0.91, 0.999: the rule separates them. 4-shot: 0.703-0.766 against line 0.65, from bf16 0.75 (`check_format.json`); read fp32.
- **Sources checked true:** clip 1.0 (`training_args.py:920`); 0.926, 1.98, 0.75; floor's 0.007 (0.029 − 0.022); 2 × 13 min (84 s per 16 rows gives 11 min).
- **Line completeness.** With a fixed floor every point gap falls in a branch; NaN is covered; a negative gap reads C.
- **Open Γ item.** fp32 Γ at a large change does not touch this verdict: true-mask K10 uses masks at W0, where bf16 is exact. It touches K01 and K11 only.
- **Control's strength.** Job S's control can catch drift or a swapped file, and no more (audit's table). Fine as long as it is not reported as validating the share.
- **Disk.** 19 saves plus 10 for arm 0 is about 82 GB, not 42. 4.3 TB free.
- **Warm-up.** 10 of 300 steps is not the cited ratio 0.1, for arm 4, as in E45: state it as a departure. The cited source remains unseen.

## Not run
Any forward or generation; the sizes of states under the fix; the teacher's accuracy on test 0-255; `true_kl`'s real footprint at 128 rows beyond its docstring and the 16-row log; Job R and Job S code (not written; the second review reads it).
