# Our crosscoder code reproduces Minder et al.'s released per-latent numbers

We ran our crosscoder pipeline on the Gemma-2-2B crosscoder that Minder et al. (2025) released, and compared each
latent's numbers with the table they published beside it. Our decoder norms match theirs to rounding, and our
latent-scaling scores rank their top latents in almost the same order as theirs, with the same count of latents passing
their chat-specific test. For `GOALS.md` this means a crosscoder reading on our tasks can no longer be put down to a
wrong port of the method; problems found there belong to the method or to our tasks.

## What ran

E210, cell P2.37: two cluster jobs, $0. Stage a (job 61425, no model) recomputed each latent's relative decoder norm
from the released weights. Stage b (job 61430, one L40, 15.5 min) passed 1M FineWeb and 1M LMSYS chat tokens through
gemma-2-2b and gemma-2-2b-it at layer 13, then fitted our latent-scaling betas on their top 3,176 latents. Rules
`experiments/crosscoder_replicate/results/predictions.json`; results `stage_a.json`, `stage_b.json` and
`stage_b_latents.csv` beside it; plan `reports/plans/2026-11-18_plan_p2-37-crosscoder-replicate.md`.

## What it read

| check | filed bar | read |
|---|---|---|
| relative norm against their table, largest gap | under 1e-5 | 1.49e-7 |
| crosscoder reconstruction on our activations, base and chat | at least 0.7 | 0.873 and 0.885 (paper: 86.2% and 86.9%) |
| rank agreement of our reconstruction score (nu_r) with theirs | at least 0.9 | 0.988 |
| rank agreement of our error score (nu_eps) with theirs | at least 0.9 | 0.977 |
| latents passing the chat-specific test, ours against theirs | within 10% | 594 against 599 |

A reference with no fit at all (the do-nothing reading) agrees with theirs at only 0.515 and 0.674, so the agreement
above is the fit's, not the latents' order alone. We fitted on 2M tokens where the paper used 50M; the small count gap
is in line with that.

## Two puzzles in the released table, not in our code

- Their table has latents with a negative base beta, which the paper says it drops; both sides were filtered the same
  way here.
- Its counts match the paper's Table 1 only at one threshold of the four; our counts follow the table, not the paper.

## Not run

- Training a crosscoder ourselves on Gemma: only the released one was read.
- The full 50M-token fit.
- The crosscoder on the realistic 67M tasks: P2.38 run 3.
