# E141 plan: the update's effective rank beside the share, on every pair with saved updates (cell P2.9)

Card: `reports/plans/2026-11-10_plan_p2-9-card.md`. Predictions: `experiments/update_svd/results/predictions.json`.

**What runs.** One CPU job, `experiments/update_svd/run.py`, weights only, no model and no forward pass. For each saved
weight change dW of a covered pair, the singular values of each of its 24 matrices give the effective rank
(Roy and Vetterli 2007, as E51 aggregated it), the top-10 energy share and the squared norm
(`src/commute/update_spectrum.py`). Each pair is scored by E51's rule, side gap against the larger within-side range,
for the rank and for the re-gating share of the same states, copied from the scorecard.

**Pairs covered**, from `ls` of each folder on 2026-11-10 (reuse side against new side, seeds per side):
copy vs lookup at the new marker, 3 against 3 (`induction_cue`); copy vs random maps, 4 against 4 (`regating_table`);
lowercase vs ciphers at step 300, 2 against 4 (`performed_anchor`); half-known vs cipher at equal size, 5 against 5
(`keydependent_elicit`, `keydependent_patch`); half-known vs cipher at its own step, 5 against 2 (the own-step cipher's
seeds 2-4 were never saved); rank-1 vs cipher and copy vs cipher at step 10, 8 against 8 (`open_pair`,
`open_pair/amend1`, `open_pair_instruments`); case label real vs twin, 2 against 2 (`case_label`); repeat label real
vs twin, 2 against 2 (`fresh_reuse_pair`).

**Holes.** Rank-1 vs cipher at 96 keys and copy vs prior-matched cipher: only the seed-0 reuse states were saved
(`open_pair/amend3`), no new side. Half-known vs its shuffle and class map vs its twin: Amendment 7 kept windows
only. The step ladder: not run. The fine-tune that learned nothing has one state and no pair: reported alone.

**Expected values.** At the zero update the rank is 0/0: reported as undefined, never in a pair. Positive control:
an update of ten equal orthonormal directions per matrix reads 10, one direction reads 1. E51's two states
`ident32_s0` and `rand32_s0` must read 442.49 and 427.52 again. Any miss stops the job before a saved state is read.
There is no do-nothing reference for a weight statistic.

**Filed predictions.** P1, the card's split: the rank separates fewer covered pairs than the share, leaning on E51's
edge-of-rule result and on the rank-1 vs cipher pair, already seen while measuring the spread. P2: the rank separates
at least one pair the share does not, since the two label pairs' sides were trained for different numbers of steps.
P3: where the rank separates, the reuse side is not always higher.

**Blind to.** Everything that is not the shape of the update's spectrum: which directions it uses, how they relate to
W0, and what the model does with them. The rank tracked the update's size in E51, so the squared norm is saved beside it.

**Cost.** One CPU job, about 70 states at roughly 2 s each, $0, free queue. No GPU.
