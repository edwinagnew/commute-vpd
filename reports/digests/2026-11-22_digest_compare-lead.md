# compare-lead digest, 2026-11-22

**Cells:** every compare-lead cell is filled or closed. P2.56 was filled tonight: 3 of 3 runs used, due 11-23. P2.51 was filled at 22:45 on 11-21 (0 runs). No asks are open.

**What ran:** E261, scoring P2.56's procedural fine-tunes against definition (b), on the CPU worker at $0. We trained the 67M model on three procedural tasks (sequence, brackets, addition), by RL and by SFT on demonstrations, all scaled to one update size. Under definition (b), RL and SFT read alike on sequence and brackets. Addition gives no call: RL reads 0.075-0.141 against 0.051-0.054 for SFT, under the filed gap. Every trained update sits far above a matched random update (0.009-0.013). The re-gating share had called RL above SFT on sequence, where definition (b) reads them alike. Ranked over trained updates only, KL (0.73) and effective rank (-0.64) follow definition (b) best, the untrained-decomposition share (0.64) and the re-gating share (0.56) follow it moderately, and the prior does not (-0.11). The premise check came back doubtful: definition (b) may track how far the output moved. The page is reports/analyses/2026-11-21_analysis_procedural-rl-sft.md. The writeup-lead note is written, and the log is published as v438.

**Money:** no pods or Slurm jobs tonight.

**Friction noted:** 995-999. They cover: E256's gate is blind to RL collapsing onto one answer token; publish was refused twice before it passed; `numbers check` misses the U+2212 minus sign and matches by value only; and check.py flags every "(b)".

**Why finished:** every cell is done, and new work needs a new cell from the user or whip-lead.
