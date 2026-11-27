# Review: E313 (P2.83 run 3, 800M) crosscoder page

*2026-11-27, redteam-lead. Page `reports/analyses/2026-11-25_analysis_p2-83-crosscoder.md` at 76517cca; results
`experiments/crosscoder_fig4/results/sides/` (200M) and `results/long/sides/` (800M).*

I recomputed every reading on the page with jq from the side JSONs and the planted-control JSONs, and every number
matches. The page now says the right thing about the fine-tune-specific part: it is near zero but not exactly zero, and
four times the training leaves it at most 0.020. The page passes. One caption wording should change, and the Fig 6
panel must wait for the two-length figure.

## Recomputed
- d, literal and floored, on all 51 sides at 800M and 52 at 200M: equal to the page's table at every cell. The one
  filter difference is Cipher (96 keys) seed 2 at 800M, 0.32925 against 0.32883 (diff 4.2e-4).
- The 48-state means are 0.4260 (200M) and 0.4038 (800M). Built to reuse: 0.4103 and 0.4073. Built new: 0.4417 and 0.4003.
- Fine-tune-specific part at 800M: nonzero on exactly the 10 sides the page lists, from 8.0e-5 to 0.0204. At 200M the
  largest is 0.1008 (Copy key (96 keys) s0), and the next largest is 0.0047.
- Random updates: 0.532/0.550/0.611 at 200M and 0.540/0.575/0.665 at 800M. The sides the page names as reaching the
  random range at 800M are exactly those at or above 0.540, and the largest state, 0.622, is below 0.665.
- Planted controls at 800M: answer_half reads 0.933 under both filters. Content-independent reads 0 literal and 0.997
  floored. Self FVE is 0.937.

## Items
1. The fine-tune-specific part correction: **OK** in the body, What we learned and the scorecard.
   **SHOULD (caption):** "near 0 on every state at both lengths: at most 0.101 at 200M" calls 0.101 near 0. Write
   "at most 0.005 on every state but one at 200M (0.101, Copy key (96 keys) seed 0) and at most 0.020 at 800M".
   On my earlier caveat: it is answered up to 800M, twice the length the every-window plant needed to separate (400M).
   **NIT:** the opening's "the reason is not too little training" and What we learned's "longer training does not
   change that" should both say "up to 800M".
2. 0.3292 against 0.3288: **OK**, in the body, the scorecard and the caption.
3. Per-task values with no "k of 48" counts, and the random updates against the states: **OK**. "Most" appears as a
   word, never as a tally.
   **MUST before the Fig 6 swap (not for the page text):** the panel needs `crosscoder_fig4_lengths.png` drawn from a
   CSV that holds the 800M rows. The page still embeds the 200M figure, and `figures/crosscoder_fig4.csv` holds only
   the 200M values. Send me the CSV once it is drawn and I will diff it against `results/long/sides/` (5 min).
4. Done-when: **OK**. The published rule passes at 800M at both answer placements (half window 0.933; every window
   0.804, which passed from 400M). **NIT:** "In every crosscoder a single latent took that plant": the 800M floored
   content-independent read has 2 specific latents (`planted.floored.counts.ft_specific` = 2). Give the lead latent's
   share or say "one latent, with a second small one".

## Not run
- No independent recomputation from raw activations: this is a transcription check of saved JSON only.
- The figure, which is not drawn yet.

## Addendum: the two-length figure (same night)
I diffed `figures/crosscoder_fig4_lengths.csv` against the side JSONs: 0 mismatches over the 102 panel-d rows, and
all 13 control rows equal the stage0b JSONs. The PNG matches the CSV. The MUST above is met.
Three NITs for writeup-lead: the legend overlaps the "filed bar" label; the footer says "base beta" and "sd" where the
page says "original-model coefficient" and "standard errors"; and the second crosscoder seed's star is no longer drawn,
so the caption should not cite the 0.038 noise gap unless the star comes back.
