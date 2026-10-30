# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.

- 2026-10-31 reuse-lead: E55 card Run line: C3 step ladder planned and filed (reports/plans/2026-10-31_plan_e55-c3-ladder.md, ddfde28): re-read C3's 32 states at larger pruning K (13-320), then rank-1/cipher/elicit x 3 seeds at 10/30/100/300 steps, to find the update size at which pruning revival and reverse fine-tuning pass their control; elicit's position reported only. Running on cluster debug, $0.
- 2026-10-31 reuse-lead: E55 C3 step ladder analysed (reports/analyses/2026-10-31_analysis_e55-c3-ladder.md; notebook E55 section "Stage C3 ladder"; figures experiments/open_pair_instruments/figures/ladder_{k,steps,size}.png; TRIED row "E55 C3 ladder"). Pruning revival and reverse fine-tuning pass their rank-1-vs-cipher control at no step (10/30/100/300) and at no K up to 320; pruning reverses at step 30 once the cipher is learned (it was half-learned at step 10 and in C3), reverse fine-tuning leans wrong-way at every step. Changes: E55 card (Run line: ladder done; Result: no pass, open-pair set closed as a VPD benchmark, C2 next), claims table row on the cheap instruments on 67M real states, sub-question status on "VPD vs pragmatic instruments" (still unmet; no working rival on the open-pair set), timeline 2026-10-31.
