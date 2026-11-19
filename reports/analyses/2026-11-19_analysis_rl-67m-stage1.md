# What the 67M already does: the RL benchmark (P2.41, due 11-22; E215)

*2026-11-19. Stage 1 of `reports/plans/2026-11-19_plan_rl-67m.md`. Results: `experiments/rl_67m/results/stage1.json`.*

We sampled 16 tokens at temperature 1 from the 67M base model on 256 Pile prompts, 8 samples each, and scored 13
behaviours with a program. Four behaviours land in the usable band for RL, where between 5% and 50% of samples earn
the reward: continuing in lowercase, repeating an earlier token sequence, continuing a count, and ending the sentence.
The rule filed before the run picked repeating (rate 0.16) and lowercase (rate 0.25) for the RL arms. It picked French after an English prompt
(rate 0.004) as the behaviour the model must be taught.

| Behaviour | Reward rate | Spread over prompts (sd) | Usable? |
|---|---|---|---|
| lowercase | 0.245 | 0.25 | **picked** |
| repeat earlier sequence | 0.157 | 0.30 | **picked** |
| end sentence | 0.292 | 0.25 | usable |
| counting | 0.130 | 0.16 | usable |
| French after French | 0.851 | 0.18 | too common |
| German after German | 0.674 | 0.29 | too common |
| close bracket | 0.882 | 0.19 | too common |
| close quotation mark | 0.775 | 0.29 | too common |
| sports words | 0.011 | 0.07 | too rare |
| business words | 0.013 | 0.07 | too rare |
| emoticon | 0.011 | 0.04 | too rare |
| German after English | 0.004 | 0.06 | too rare |
| French after English | 0.004 | 0.02 | **taught anchor** |

The rule picks the two usable rates nearest 0.2. The checks run before sampling passed:
- A zero update reads a KL of exactly 0.
- Scoring the text's own continuation reproduced the CPU dry run: counting 1.0, French 0.848, German 0.758, and English text scored as French or German 0.0 and 0.004.

A random update of size 0.01 moved the output by a KL of only 0.0013.

Sampling took 2.8 s per batch of 256 sequences. The run took 5.5 minutes on the cluster debug queue, at no cost.

Next is stage 2, the training arms for these two behaviours and the anchor.

## Not run
- No training yet; the rates above are the base model's only.
- Reward rates on a second prompt set (the selection used one set of 256 prompts).
