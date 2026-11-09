# E136: does the share still separate built pairs on a decomposition that needs no training? (cell P2.8, due 11-12)

*compare-lead, 2026-11-09. Card: `2026-11-09_plan_p2-8-card.md` (passes at 5.10 standard errors, n=3), which states the no-training gate rule. Saved updates only, free
queue, one run, no new fine-tune. No verdict under any wording of reuse: sides are labelled as each pair was built.*

**What runs.** The share's formula, unchanged (`r = (L01 - L00) / gain`, the do-nothing form), read three ways on the same rows of every saved state: the trained
decomposition, each matrix's SVD, and that SVD with a random rotation. The two untrained ones use the gate rule on the card; the trained one uses Γ.

**Pairs.** Every row of `experiments/method_table/results/table.json` whose `cells.r.verdict` is "separates", plus `e62_copy_lookup` (the share's gap of about 0.69), plus the
real-against-twin pairs: `case_label`, `fresh_reuse_pair`, and `e58_class_twin` (`e58_elicit_shuffle` is already in the list). Scored: pairs with three or more seeds per side.
Described only, not scored: pairs with two per side (`case_label`, `fresh_reuse_pair`, the reuse side of `e53_lowercase_cipher`). Windows, states and saved shares per pair:
`experiments/prior_vs_share/results/inventory.json` lists 24 maps; the rows it omits (`e53`) are found where `experiments/method_table/build.py` reads them.

**Apparatus.** `model.components` holds 512 subcomponents on `attn.k_proj` and `attn.q_proj`, below their rank of 768, so an exact SVD cannot be written into it (`alive.pt` shapes,
2026-11-09). The untrained reads use a new masked forward instead: a forward hook on each of the 24 matrices returning `(g * (x @ V)) @ U^T` in fp32. Trained reads stay on `route_split` as filed (bf16 autocast);
the difference is stated in the report. The rotated decomposition comes from seed 0.

**Stages, cheapest first, each written as it finishes (`STAGE n DONE`).**
- Stage 0, controls, alone, stop on a miss:
  (a) exact sum: for both untrained decompositions and every matrix, `max |sum_i u_i v_i^T - W0| / max |W0| <= 1e-5`;
  (b) all gates 1 on the new forward: the log-odds at W0 equal the unmasked forward's within 1e-3 on the first read windows of each pair, the unit's rounding at fp32 (measured here, first);
  (c) zero update: under the rule the state's gates equal W0's and the numerator is exactly 0;
  (d) k_m computed and saved, from Γ at W0 over all pairs' read windows, before any state is read;
  (e) the trained `route_split` of the first state of each pair reproduces its saved share within 0.01, a tenth of the card's smallest gap.
- Stage 1: all states, three decompositions; projected time logged after the first pair, stop if over 110 minutes.

**Filed rule** (`predictions.json` first, edge-checked below). Per pair and decomposition, `gap = mean(built-to-reuse side) - mean(built-to-be-new side)` of the share in log-odds. Let `sign` be the sign of the trained
gap. `shrink = sign * (gap_trained - gap_svd)`. On scored pairs, rival A if `shrink >= 0.1` on more than half of them; rival B if `shrink < 0.1` on more than half; "split" when exactly half and half. The rotated decomposition's
gap is reported beside it and does not enter the rule. A pair is unreadable when any state's gain on its rows is below `MIN_GAIN` (1 log-odds) or any share is `nan`; an unreadable pair leaves the count, and under three readable scored pairs the
verdict is "no call".

Edge cases run through: a negative shrink (the SVD separates more widely) counts as under 0.1, rival B's side; a trained gap of 0 gives `sign` 0 and `shrink` 0, so it counts for rival B: filed because no such pair is expected among those the share separates; a zero update reads 0 numerator by construction,
stage 0. Do-nothing: the rule reads saved updates, so no do-nothing state is scored; a decomposition whose gates never move would read share 0, gap 0, rival A, which is the correct answer to "the separation needs training".

**Expected values.** W0 with all gates 1: T0, to fp32 rounding. The zero update: numerator 0. Trained reproduction: within 0.01 of the saved shares. Untrained gaps: between 0 and the trained gap on the scored pairs, either rival; the rotated gap below the SVD gap
(a random rotation breaks the singular directions, which is what the gates read). k_m: not guessed, computed in stage 0.

**What each number is blind to.** The share here is blind to whether the top-k components at a position are interpretable; it reads only how far the masked model moves with the state's gates. The trained reading's gates are continuous while the rule's are 0 or 1,
stated as a difference between readings, not removed. Positive control: the trained reproduction; noise floor: seed sd per side, reported per pair.

**Figure.** One: per pair, the trained, SVD and rotated gaps, as in the card's sketch, with the seeds behind each (`commute.figures`). One paragraph, with `/writeup`.

**Cost.** One L40, `debug` QoS (`--comment=E136`), expected well under 30 minutes.

**Design check** (`/design`, a line each). Goal: part 2, how much of the share's separation is due to the trained decomposition. Nearest row: none reads the share without Γ; E18's fixed patterns and E98's neuron version replace the gates' source, not the decomposition. Step, not patch.
Result that changes what we do: rival B means the share is no evidence about VPD. Decisions relied on: `decisions.md` row 112 (no verdict). Control first: stage 0, alone.

## Not run

Everything above. The saved trained shares on the pairs not reproduced in stage 0 are re-read in stage 1, not trusted from file.
