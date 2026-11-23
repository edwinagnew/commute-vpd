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

## Amendment, same day, before any value is computed (redteam-lead)

**Forking paths.** This weighting was chosen after seeing head (2,4) separate copy from lookup at the head
featurization (E277's e62 stage). E62 copy against lookup under E_flip is therefore a restatement, not evidence for the
weighting. The items not seen when it was chosen, and so the test: E277's panel stage (12 realistic tasks x 3 seeds +
Gaussians, against E272's rater tiers) and E277's trained reuse/new pairs (copy24/cipher24, ident32/rand32,
case_real/case_twin, repeat_real/repeat_twin, twin/fact; seeds 0-1; each beside its size-matched random), at the
neuron/head featurization.

**A dd floor.** kept_u = explained_u / dd_u is unstable where dd_u is near rounding, and flip weighting gives such
variables full weight. No per-variable dd noise floor was saved by any run, so the floor is not measured: E_flip is
reported without a floor and with floors dd_u >= f x (sum of dd over K & changed) at f = 1e-8, 1e-6 and 1e-4, so the
reader sees whether the value rests on near-zero-dd variables. The nearest measured number is E281's CPU-vs-GPU
relative L1 difference of dd, 3.8e-8, over all variables of e62/copy_s0.

**Expected values (not bars).** Randoms: every variable keeps about 0.01 (E_b 0.011), so E_flip about 0.01 at either
featurization. Built x2 and x10: E_b 0.76 at VPD with the change on the edited subcomponents, so E_flip high (at least
0.5) at VPD; at neurons x2 changes 9 variables, kept near 1 (E_b 0.9998), so E_flip near 1. new_few, new_thin: about
their E_b (0.01-0.02). E280 V and O x2 at heads: exactly 1. A reading far from these is "not as expected" and is named.

**Lean.** flips_u are F's own counts, so a variable F needs more gets more weight by construction: E_flip leans toward
the ablation reading by design, and agreeing with head ablation on E62 is partly built in.
