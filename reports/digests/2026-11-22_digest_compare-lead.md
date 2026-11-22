# compare-lead digest, 2026-11-22

**Cells:** every compare-lead cell is filled or closed. P2.56 was filled tonight: 3 of 3 runs used, due 11-23. P2.51 was filled at 22:45 on 11-21 (0 runs). No asks are open.

**What ran:** E261, scoring P2.56's procedural fine-tunes against definition (b), on the CPU worker at $0. We trained the 67M model on three procedural tasks (sequence, brackets, addition), by RL and by SFT on demonstrations, all scaled to one update size. Under definition (b), RL and SFT read alike on sequence and brackets. Addition gives no call: RL reads 0.075-0.141 against 0.051-0.054 for SFT, under the filed gap. Every trained update sits far above a matched random update (0.009-0.013). The re-gating share had called RL above SFT on sequence, where definition (b) reads them alike. Ranked over trained updates only, KL (0.73) and effective rank (-0.64) follow definition (b) best, the untrained-decomposition share (0.64) and the re-gating share (0.56) follow it moderately, and the prior does not (-0.11). The premise check came back doubtful: definition (b) may track how far the output moved. The page is reports/analyses/2026-11-21_analysis_procedural-rl-sft.md. The writeup-lead note is written, and the log is published as v438.

**Money:** no pods or Slurm jobs tonight.

**Friction noted:** 995-999. They cover: E256's gate is blind to RL collapsing onto one answer token; publish was refused twice before it passed; `numbers check` misses the U+2212 minus sign and matches by value only; and check.py flags every "(b)".

**Why finished:** every cell is done, and new work needs a new cell from the user or whip-lead.

## Afternoon, 2026-11-22

**Cells:** P2.58 (due 11-23) is filled: 1 of 1 runs used, E262. P2.59 (due 11-23) is filled: 2 of 2 runs used, E264 then E265. No asks are open. whip-lead has filed proposal 56, the $0 lens check, and is waiting on the user.

**P2.58, the KL control (E262, $0, saved results):** definition (b)'s write share reads more than how far the output moved.
- On the nine built pairs, KL explains 21% of its spread. The three pairs that were outside the random floor stay outside after regressing on KL.
- On P2.56's 39 trained procedural states, two proxies keep their agreement with the write share once KL is held fixed. The re-gating share reads 0.48 and the untrained-decomposition share 0.44. The neuron share flips to -0.80, likely suppression.
- Page: reports/analyses/2026-11-22_analysis_e262-kl-control.md.

**P2.59, the activation difference lens (Minder et al. 2025, ported from the authors' toolkit; cluster debug, $0):**
- Run 1 (E264) failed its copy control (0.0028 against a bar of 0.05). The copy change depends on the key, so it cancels in a mean over unrelated text.
- Run 2 (E265) swapped the control to the lowercase fine-tune. That control passed at the edge (0.0500), and its logit lens named no answer token.
- On the nine pairs, the lens gap has the same sign as the write-share gap on 8 of 9, with a rank correlation of 0.83. The re-gating share reads -0.28 and the crosscoder -0.92.
- This is provisional: the lens has not been tested against KL and has had no independent check (proposal 56).
- Page: reports/analyses/2026-11-22_analysis_e265-adl-column.md.

**Money:** $0. Cluster jobs 63409, 63422 and 63425 ran; no pods.

**Friction:** check.py is not executable as the shell notes say.

**Why finished:** both cells are filled. The next work waits on the user's answer to proposal 56.
