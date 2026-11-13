# E180 Amendment 1 (dated note; predictions.json unchanged)

*2026-11-13, definition-lead, cell P1.20. Run 1 (job 59959) found that the rung-2 search diverged on L2: closure fell to about -7e20. Two-sample per-window curvature near 0 for rare tokens gave huge Newton steps. The line search always added a name, and the Adam step scale grew with the weight. Fixed in `src/commute/rank_read.py`; the old code and results are in `~/commute-data/scratch/definition-lead/e180/old/`.*

## Departures from the filed text
Each keeps every reading an upper bound on the rung.
- **Rung 3:** one edit is screened at all 10 site pairs; one and two edits are then fitted at the 3 pairs with the lowest screening KL. The filed text says "at each" pair.
- **Greedy stop:** the search stops after 3 steps each adding under 0.005 of closure, or when no candidate lowers the check KL on the fit half.
- **Rung 3's base:** whichever of rung 1 and rung 2 has the lower check KL.
- **Sizes:** scoring windows 2,048, line-search multiples x0.25 to x2, refit 20 Adam steps. Read windows stay at 16,384.

## Not changed
Filed bars, cap, stop rules, expected readings, R1 and R2.

## Run accounting
Rerunning run 1 on the fixed code is the cell's run 4 (reserved for a repair). The L1 and L2 results of the broken run are void. do_nothing (KL 0, no search) is kept.

## E182 (2026-11-13): the controls run under a new code, one more search restriction
The controls run is re-filed as E182 (cell P1.20 run 2 of 4; predictions.json unchanged except its "experiment" line). A
fresh review found that a rung-2 pair whose read is a token seen in one scoring window can score about ten times the
built pair (its curvature is pooled while its gradient is concentrated), fill every line-search slot and leave L2 with no
rung-2 name. Token reads seen in fewer than 8 of the 2,048 scoring windows are therefore left out of the pair search
(`Sizes.min_tok`). This only narrows the search, so every reading stays an upper bound on the rung.
