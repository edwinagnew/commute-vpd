# Flip-weighted kept share: filing (cell P1.57, descriptive, $0)

Filed 2026-11-23 before any value is computed. Asked by whip-lead after E277's E62 stage, for the user to judge as a
weighting choice inside the reading; no bar, no verdict, no rival.

**The column.** For each item, over the variables u in K & changed:
E_flip = sum_u flips_u * kept_u / sum_u flips_u, where kept_u = explained_u / dd_u (`commute.stand_in.explained` over
dd, the same per-variable kept share E_b uses) and flips_u = the number of the item's tested pairs whose swap of u
changed F's argmax at a read position (the saved per-variable flip count, out of the item's n_pairs). A variable with
dd_u = 0 is not in changed and is left out. E_b itself (the dd-weighted share) is reported beside it unchanged.

**Items.** E62 copy and lookup, seeds 0-2, and the three size-scaled lookups: at VPD (E275's K and flips, E257's
num/den/dd) and at the neuron/head featurization (E277). The built edits x2, x10, new_few, new_thin and the two randoms
at VPD (E268) and at neurons (E277, when written). E280's head edits V, O, Q x2 at both featurizations. E277's panel
stage when it lands.

**What it is blind to.** A variable needed on many pairs but whose change is tiny counts as much as one with a large
change; the flip counts are lower bounds from 256 pairs, saturate, and depend on margins (tiny on E62's mixed D).
Expected at the controls, from the definitions only (not a bar): the do-nothing has no changed variable (undefined);
E280's V and O x2 at heads have one changed variable with kept 1, so E_flip = 1.
