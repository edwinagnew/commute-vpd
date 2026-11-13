# qwen-rl-lead digest, 2026-11-13

P3.4 is closed: all 3 runs are spent and the filed verdict is "no call". The premise check, once fixed, read
"doubtful". Reinforcement learning raised Qwen3-1.7B's held-out maths pass rate from 0.78 to 0.88
at twice the decomposition's error. It was never compared with a working shuffled-reward null, because that null
collapsed at both learning rates tried.

**Cells.** P3.4 used 3 of 3 runs and is closed with the text at
`~/commute-data/scratch/qwen-rl-lead/p34-close/close.txt`. No other cells and no ladder rung, so the lead has finished.

**What ran today.** Run 3 (E178): the null alone at the lower rate (1e-6), gated, with the user's go-ahead.
- The $0 cluster rehearsal and the gate-5 review were clean.
- Seed 0 ran out of GPU memory at step 11 and was restarted with an allocator setting. It then collapsed at step 90,
  at 0.82 of its 2.09 target, and saved no update.
- Seed 70's first pod was stopped at step 20 by the filed reward check (0.816 to 0.506). The user overrode the stop,
  but the idle watchdog deleted the pod first. Its restart was stopped before it ran, because seed 0's collapse already
  made the verdict "no call".
- Spend was about $18.50.

**Report.** `reports/analyses/2026-11-13_analysis_e178-qwen-rl-readable-run3.md`, with its figure. The log is
published as version 343; the P3.4 close is in version 347.

**For the gate.** The premise check (`~/commute-data/premise/P3.4.md`) reads "doubtful" on two points. A
shuffled-reward null may not be steerable to RL's output change without collapsing, so a matched-size comparison would
only show learning against breaking. And every update read, RL's included, sits at 0.98 to 0.99 of a random change's
outside share, so this decomposition does not tell RL from random either. Its question: what would a matched-size null
need to show to bear on the decomposition? Problems #4 and #5 are marked fixed.

**Friction noted.**
- 768: the share job's memory peak.
- 769: wrong paths in my brief.
- 770: the premise check refuses the cell.
- 771: TRIED has no rows for E163 or E167.
- 773: runner waits.
- 774: a jq gotcha.
- 776: an out-of-memory error and the H100 price.
- 777: a pod lost to the watchdog during a band stop.
- 778: pod labels and stock.

**Not run.** No null reached its target size, so no null state was read. No third rate or seed was tried. The cause of
the collapse was not checked.
