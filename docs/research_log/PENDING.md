# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.

- 2026-10-31 reuse-lead: E55 card: C3 result · `reports/analyses/2026-10-31_analysis_e55-c3.md` (notebook E55 section "Stage C3", figures `experiments/open_pair_instruments/figures/c3_{poles,positions,distance}.png`) · E55 card, reuse-line claims and sub-question status, timeline · On the 67M open-pair maps (8 seeds), pruning revival and reverse fine-tuning toward W0 pass the planted control but do not separate the rank-1 map from the cipher (R gap 0.07 < 0.14; reverse 3.2 steps the wrong way, < 4.3): joint "not settled". Both track how sharply each 10-step state peaked W0's answer (D); every update is a thin layer over an intact W0. VPD's r and stand-in ratio separate the ends (p 0.85, 0.77) but read "distance and prior" under rules 5-6. The cheap methods have not yet had a fair test; refinements proposed for /design (longer fine-tunes, D-matched poles, a target that does not reward sharpening).
