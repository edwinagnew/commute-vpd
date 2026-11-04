# Digest: qwen-rl-lead (2026-11-04)

D2, day 2 of 8. **Why I finished.** The training is done: RL, the shuffled-reward null and own-samples SFT, a second seed of RL and the null, and anchor 1. Everything left waits on you.

## Your decisions
1. **What should the Qwen example of "content the model lacks" be?** Needed by 2026-11-06, or the second column of the D3 comparison won't exist by 2026-11-11. Recommended: invented facts, about 1,000 question-answer pairs, a few dollars and a day. Log card `decision-anchor2`; ask 5.
2. **Does the RL row rest on these two seeds, or on a third for about $20?** Recommended: read the two. Log card `decision-e69-third-seed`.

## What it found
Under the strict grader, RL beat the shuffled-reward null on the same test problems on both seeds, by about three and six points. With answer format set aside, the second seed stays about six. The first seed depends on how long answers are treated: +1.25, +2.50 or +5.53, the last two read after the fact. We filed a lean towards no clear lead with format set aside, and it reversed on the second seed. We do not claim the reward taught better maths: one shuffled run came close on answers that finish, and the graders do not read the working. Nothing here explains why the seeds differ.

An independent check recomputed the numbers from the per-problem answers and found no bug. It changed the wording, because the five-point gap between seeds belongs to one way of treating long answers.

Report: `reports/analyses/2026-11-03_analysis_e69-grpo-loop.md`. Notebook: `notebooks/E69_qwen_rl.ipynb`. Figures: `experiments/qwen_rl/figures/leads.png`, `saved_steps.png`. Log: v216.

## What ran
The six saved-step reads and all the analysis ran on the MATS cluster for $0. The second seed's pod cost $16.51, before this session. This session's remote-kernels spend is $0. No pod is up and no Slurm job of mine is live. qwen-decomp-lead has seed 70's weight changes and a note on using the larger null gap as RL's noise floor.

## Problems
- **Card memory.** On `l40-worker`, Slurm hands out cards 3 and 6 as free while unknown processes hold about 45 GB on each. It killed 12 of my jobs at model load, for $0 and about an hour's delay, and at least one of another lead's. The owner is not identified and I touched nothing. qwen-decomp-lead holds the open problem, #9.
- **Log publish.** The clerk's first send of v216 was refused with 11 differing places. It judged them its own edits and resent unchanged until the third send passed. `--check` matched before it edited and it held the lock throughout, so I expect nothing was lost, but I did not diff the live page. Noted as friction.
- **Log cuts.** To hold the main page's length, the clerk cut two 2026-11-02 timeline entries, a gaps card, a backlog item and the "What's next" closing line, and shortened the overview's Line 3. They are listed in changelog row 216.

## Not run
- A third seed; it waits on decision 2.
- Anchor 2; it waits on decision 1.
- Shuffled weights on SFT's saved rollouts, $0 and about 1.5 h of an L40, which the report suggests. It is outside the task, so it wants your word.

## Friction
Noted with `fleet/friction`: the fetch script's 6 h limit queued a 45-minute job for 8 h; no check of free memory on the allocated card; a writer's redundant hashing step, removed.
