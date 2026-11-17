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

## Run 3 of the cell (2026-11-16): the logit writes' curvature
E182's controls stopped: on L2 the rung-1 and rung-2 searches took no step. A GPU diagnostic (E182 stage l2diag,
`~/commute-data/scratch/definition-lead/e182-diag/diag.log`) found no NaN. The four best-scored candidates in both rungs were
logit writes (gains of logit 709 and others; logit 709 against neuron reads) whose curvature was the pooled q(1 - q):
near 0 in windows where F puts mass on a token M almost never gives. Their Newton steps overshot (check KL 0.37 to 42 at a
quarter step against 0.30), so the line search took none. The built pair ranked about 1,880th but works (a full step 0.111,
with its shift 5e-9). Fix (`rank_read.logit_curv`): a logit write's curvature is now summed per window, exactly, as
max(q(1 - q), p(1 - p)), p F's probability, for its gain, its shift and every pair it heads. Dense writes are unchanged.
The search remains a search: every reading stays an upper bound on the rung. The user chose this repair over closing the
cell "no call" (decisions.md, 2026-11-16, "Fix search, rerun"). Filed bars, cap, stop rules and expected readings unchanged.

## Run 4 of the cell (2026-11-16): the rung-1/2 refit
E183's controls stopped: L2 read rung 2 with its built pair, but at bar 0.95 it took three names (closure 0.985), where the
filed rule allows at most two. With its two built names the refit had reached only closure 0.795 (20 Adam steps at fixed
scales), while the same two names fitted exactly give a check KL of 5e-9 (E182 diagnostic). The third name made up the
refit's shortfall. Fix (`rank_read.refit_writes`): rungs 1 and 2 refit by full-batch L-BFGS (strong Wolfe, 20 iterations)
on 2,048 Adam windows after the check windows; rung 3 keeps Adam. The user chose this and to spend the cell's last run on
it (2026-11-16, dialog, "Fix refit, last run"): the controls run first under the same filed rule, and only if they pass do
the remaining items run, as stages of the same run. E183's results are kept in `results/e183_run1/`.
