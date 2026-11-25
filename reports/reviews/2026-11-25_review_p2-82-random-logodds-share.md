# Review of the P2.82 page: random baseline for the log-odds re-gating share (E310), 2026-11-25

The page scales random weight changes until they raise the trained answer by one log-odds and reads the re-gating share on them. Its control and tables hold up: I checked one state file against the tables, and the two main findings follow from the results. Two statements claim more than the run shows. They should be fixed before writeup-lead takes the page into the paper.

Page: `reports/analyses/2026-11-25_analysis_p2-82-random-logodds-share.md` (29ae8a86). Checked: `results/states/e54_cipher_own_s0.json` against the per-direction table (228002: size 54.8, share 0.906, KL 6.67; 228001 capped with largest gain 0.79), and `src/commute/perturb.py`.

## MUST
1. **The opening and "What we learned" overstate what the new side shows.** They say "a low share does not set a trained update apart from noise". That holds on c3/cipher only (random 0.24-0.26 against fine-tunes 0.22-0.24). On e46/rand32 (0.66-0.76 against 0.25-0.27), e54/cipher (0.30-0.39 against 0.21-0.26) and one e54/cipher_own direction (0.88-0.91 against 0.24-0.27), the fine-tunes read below every random change. That puts the trained update apart from noise, on the low side, though at unmatched gain. Say what the values show, per task: random changes read at or above the new-side fine-tunes, never below. A new-side fine-tune's low share is therefore not what noise gives at gain one.
2. **"None reached the band" is a grid result, and the page reads it as absence.** `perturb.match` reads 12 log-spaced sizes from 0.01 to 100 (each about 2.3 times the last). It bisects only where a grid point already reaches the target. The page itself says gain rises and then falls with size. So a peak above 1 between two grid points is missed. Several capped directions peaked near 1: e46/ident32 0.90 (a reuse task), c3/cipher 0.93, e58/cipher 0.94, e53/cipher 0.82. The sentence "on no reuse task did any draw get there, even at 100x", and the What-next line "the largest gains here suggest none exists", need "at 12 log-spaced sizes up to 100 times". Better, if a cheap stage fits the cell: refine around each capped direction's peak before the paper states it.

## SHOULD
- The fine-tune gains are 10.9-20.8 log-odds on the in-band tasks, against random changes at about 1. Put this in the per-task sentence that the paper will quote, not only in the caveat paragraph. The share is a ratio and need not be the same at gain 1 and gain 15.
- The selection point deserves one plain line in the opening: the four in-band tasks are cipher or random targets, where the base model gives the trained answer very low odds. So noise can raise it there, and reuse targets are not raised by noise.
- The scorecard verdict "holds where it can be read" for rival a: say "seen on the four tasks read; not readable elsewhere", to avoid a verdict-like word on a descriptive run.

## For the paper (MUST 4 sentence)
Suggested replacement for "the log-odds form has no random baseline": "Random changes up to a hundred times a fine-tune's size, read at 12 sizes, never raised the trained answer by one log-odds on a reuse task. On the four new tasks where they did, they read 0.24 to 0.91, at or above the fine-tunes' 0.21 to 0.27, at a gain of about one against the fine-tunes' 11 to 21."

## Not run
- No recompute of the shares from saved states; one state file was checked against the tables.
- Figures not opened beyond what the page describes.
