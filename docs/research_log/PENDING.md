# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.

- 2026-10-31 reuse-lead: E55 card: C3 result · `reports/analyses/2026-10-31_analysis_e55-c3.md` (notebook E55 section "Stage C3", figures `experiments/open_pair_instruments/figures/c3_{poles,positions,distance}.png`) · E55 card, reuse-line claims and sub-question status, timeline · On the 67M open-pair maps (8 seeds), pruning revival and reverse fine-tuning toward W0 pass the planted control but do not separate the rank-1 map from the cipher (R gap 0.07 < 0.14; reverse 3.2 steps the wrong way, < 4.3): joint "not settled". They do tell other maps apart (copy most revived, fastest undone); across the three scored maps they order as D does, but no cause was tested. Graded carry also puts elicit at the old end (0.72), so the E58 disagreement is VPD vs the label's fixed cut, which also calls rank-1 neither/new. VPD's r and stand-in ratio separate the ends (p 0.85, 0.77) but read "distance and prior" under rules 5-6. The cheap methods have not yet had a fair test; refinements proposed for /design (-e re-read at larger K first; longer fine-tunes only with a pole that frees the verdict from the prior; D-matched poles; a target that does not reward sharpening). Revised after review `reports/reviews/2026-10-31_review_e55-c3-check.md`.
