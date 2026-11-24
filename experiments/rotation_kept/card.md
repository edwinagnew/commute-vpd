cell: P1.60
rival_a: Trained fine-tunes turn the writes of the needed variables toward directions M's other needed variables at the same site already write, which predicts a rotation share at least 0.2 above its matched random's on most trained items, and E62 copy's needed-space share at least 0.2 above lookup's on the three-seed mean
rival_b: Trained fine-tunes move the needed writes somewhere M's needed variables do not write, which predicts a rotation share within 0.05 of its null on the trained items as on the randoms, and E62 copy within 0.2 of lookup
rung: strategy
smallest_gap: 0.2
spread: 0.002
n: 3
design: paired
unit: seeds per E62 arm (copy, lookup); built edits are one state each; E229 pairs and panel tasks each beside their own matched random
spread_source: experiments/claim2_needed/results/items
spread_of: proxy: E_b over K at VPD, seed sd of E62 copy (0.0376, 0.0399, 0.0362), the same items and write pass read with the strict kept share
floor: |K| < 100 reads "too small to read" (whip-lead, 2026-11-23); a needed space whose null exceeds 0.5 reads "too wide"; both rivals expect every item above the first and the built and E62 items below the second
stop: If the trained items' rotation share sits within 0.05 of its null at both bases, the rotation guess for P1.57's low kept readings is dropped and no further rotation-tolerant reading is proposed
sketch: experiments/rotation_kept/figures/sketch.png
rehearsal: E286 rehearse: on a cluster debug L40 ($0), the control stage at full read sizes (both do-nothings, x2 at both bases, the two built rotation controls) measures time per item and peak memory, and one panel_vpd state (--shard 0 24) rehearses the VPD rescan and its download; then panel_vpd alone goes to one H100 pod (about $7, approved by whip-lead 2026-11-23 within the $15 cap), every other stage on cluster debug
descriptive: user 2026-11-23 via whip-lead, a diagnostic beside the frozen definition

## What runs (cell P1.60; user 2026-11-23: "yes run the rotation-tolerant diagnostic")

A diagnostic beside the frozen definition, not the definition. For each item, one forward each of M and F over up to
256 windows of its D. The top 1% most-needed changed variables are taken from the saved scans of E268, E275 and E277.
For each, the share of its write change that lies in M's write space is read three ways. The strict share is the frozen
kept. The own-space share uses the variable's own write space. The needed-space share widens it with M's writes of
the other most-needed variables at the same site.

A neuron's or a VPD subcomponent's write has one direction across inputs. So the own-space share cannot see rotation;
it splits input-side from output-side change. The needed-space share and its rotation share, read against a random
direction's null, carry the question. Head (2,4) on E62's 206 copy windows is read alone with its strict and own-space
share and its flips (whip-lead, 2026-11-23).

Order: stage 0 (the do-nothing reads 1); x2 first at each basis, which must read 1 at neurons; then the built edits
and randoms, E62, head (2,4), E229's pairs, the panel at neurons, and last the panel at VPD on a fresh one-derangement
scan, on one H100 pod. Plan: `reports/plans/2026-11-23_plan_p1-60-rotation-kept.md`.
