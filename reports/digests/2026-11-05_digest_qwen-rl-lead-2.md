# D2, day 1 of 13, row 11-06: moved, one day early; the size grid is filed

**The row.** One file, `experiments/qwen_size_grid/results/size_grid.json`, gives every arm's saved state nearest the template fine-tune's full update, in norm and in KL to start on the maths rows. `qwen-decomp-lead` has it. E91, three jobs on cluster L40s, $0. Report `reports/analyses/2026-11-05_analysis_e91-size-grid.md`, notebook `notebooks/E91_qwen_size_grid.ipynb`.

**What it found.**
- **At the template's size, the TOFU update knows none of its facts.** Trained-wording recall is 0.16 to 0.17 at both seeds, against 0.18 unchanged and 0.99 at the full update. That state was retrained and saved, as none existed.
- **The smallest saved state holding its facts is about 30 times the template's norm**, at seed 42 step 550 and seed 142 step 500.  Seed 142 was already over the line at its first read, so its crossing could be earlier. So the two anchors cannot be compared at equal size with both having done their job.
- **The maths arms all sit at step 100 and stay under the template's norm.** RL reads 0.88 and 0.91 of its KL. The null reads 1.13 and 0.45, so its seeds differ by 2.5 times. Own-samples SFT reads 0.01, so it has no match by KL.

**Decided by you.** Ask 11: both comparisons are read, the equal-size one is a size control, and a claim of something new rests on the full update with its size stated. You called this a weak endorsement; `decisions.md` says so, and reversing it costs no run.

**Next on this row's table.** Think-mode sets: `qwen-decomp-lead` has been asked which set its row reads. With no answer by the morning of 11-07 I generate the unchanged model's think-on answers on TOFU's paraphrased questions. Task-row answers at the matched steps are done: TOFU's are new, the others were already saved.

**Pending decisions for you.** None.

**Money.** $0, no pod. **Blocked:** none.

**Friction noted:** 496, 497, 499. The one worth reading is 496: my first state list was written by hand and left out saved states of two arms; a second job fixed it in minutes.

**Not run.** Reads through the decomposition, which are `qwen-decomp-lead`'s; think-mode sets; a second seed of own-samples SFT, as no verdict rests on it yet. Full list in the report.
