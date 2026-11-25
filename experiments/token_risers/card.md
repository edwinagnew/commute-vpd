cell: P2.81
rival_a: A 67M model cannot learn Swahili or Yoruba in about 100 steps at KL 0.18, so it only boosts a few pieces it already predicts (frequent subwords, diacritics, punctuation, common bigrams); this predicts that on Swahili and Yoruba the top 20 token types carry a share of the held-out gain at least 0.15 above the share of positions they occupy, like lowercase and unlike arXiv
rival_b: The fine-tunes improve the language broadly, so this predicts that on Swahili and Yoruba the top 20 token types carry a gain share within 0.15 of their position share, spread like arXiv and unlike lowercase
rung: premise
smallest_gap: 0.15
spread: 0.05
n: 3
design: paired
unit: seeds per task (the three Fig 5 KL-hi fine-tunes), each read against the base on the same held rows
spread_source: experiments/convergence_panel/results/wide/random_3seed.json
spread_of: proxy: seed-to-seed spread of the trained E_b on the same states (up to 0.09 on one task, under 0.02 on these four); no measured spread of this reading exists yet, the first run gives it
floor: none
stop: The do-nothing read (a zero update) moves any token's log-probability by more than exactly 0: the job stops and the cell waits on the fix
sketch: experiments/token_risers/figures/sketch.png
rehearsal: cluster only: one forward pass per state over 64 held rows on a cluster debug L40 (about 13 forward reads of 64 x 512 tokens, minutes), states and data already on the cluster; no pod
descriptive: user, 2026-11-24, in whip-lead's chat: "i think you should run 2 and check what number of steps each ran to see if that explains anything"; reported as a description, no verdict on reuse, no claim at a bar

## What runs (run 1 of cell P2.81)

For lowercase, arXiv, Swahili and Yoruba, the three KL-hi fine-tunes Fig 5 reads (convergence_panel
`results/wide/selection.json` .kl.hi), on each task's 64 held rows: at every trained next-token position, the base and
fine-tuned log-probability of the actual next token. Per state: the top 30 token types and bigrams by summed gain with
their count and base and fine-tuned probability; the concentration curve (gain share of the top k types beside the
position share of the same types, k = 1 to 500); gain by token class (punctuation/space/digit, non-ASCII bytes, ASCII
word start, ASCII continuation); the mean probability on upper-case tokens (the lowercase control: expected to fall by
at least half on lowercase, little elsewhere); and the step each state was saved at. Code `experiments/token_risers/run.py`.
