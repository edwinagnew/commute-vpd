# baseline-lead digest, 2026-11-27

P2.83, due 11-27: filled. We retrained the crosscoder on all 48 Fig 4 states at four times the training length. The fine-tune-specific part stayed near zero, so training length does not explain it. The states still do not separate tasks built to reuse from tasks built new, and the random updates still read higher than most states.

## Cells
| Cell | Status | Runs used | Due |
|---|---|---|---|
| P2.77 | filled | 2 of 2 | 11-25 |
| P2.80 | filled | 2 of 4 | 11-26 |
| P2.83 | filled | 2 of 3 by the table (E311 runs 1-2, E313 run 3) | 11-27 |
| P2.84 | filled | 1 of 2 | 11-26 |

All my cells are filled, and I have no ladder rung. That is why I finished.

## What ran overnight
- **E313 (P2.83 run 3):** the crosscoder (Minder et al.'s code) at 800M tokens on 51 of 52 sides. Page: `reports/analyses/2026-11-25_analysis_p2-83-crosscoder.md`; figure: `experiments/crosscoder_fig4/figures/crosscoder_fig4_lengths.png`.
  - **States:** d runs from 0.084 to 0.622, with a mean of 0.404 (0.426 at 200M). The random updates read 0.540 to 0.665. Tasks built to reuse average 0.407 and tasks built new 0.400.
  - **Fine-tune-specific part:** at most 0.020 at 800M. At 200M it was at most 0.005 on every state but one, which read 0.101.
  - **Planted control:** passes at the answer position under the authors' rule (half window 0.933, every window 0.804). The content-independent plant passes only with the filter floored (0.997).
  - **Not run:** one reseed side, because of the user's $800 cap.
- **Review:** redteam-lead passed the page and the figure's data, with 0 mismatches against the saved results. Its fixes are applied. whip-lead is passing the text and the Fig 6 panel to writeup-lead (local only, no push).
- **Premise check on P2.83: doubtful** (advisory, for your gate). The random updates read above most real states, so the crosscoder's d may not be able to enter the comparison of methods, whatever its control does. No bar relative to the random range was set before the run; the run was descriptive. File: `~/commute-data/premise/P2.83.md`.

## Asks waiting on you
None from baseline-lead.

## Money
- **E313:** $748.33 by the plugin's spend record, under the $800 cap. This includes a $0.07 pod for the figure, used because the CPU worker is down.
- **This session:** $0.07. No pods are running.

## Friction
- 1193: E313's spend came out three ways ($764.68 in my notes, $748.26 from the plugin's spend record, $779.6 from the run's spend file).
- 1194: the CPU worker has been down all night (compute node mixed*). Figures and Python checks needed a pod.

## Not done
- `experiments/crosscoder_fig4/results/long/run3_dl/` is untracked: 17 MB of raw per-pod downloads, kept in case its logs are needed.
