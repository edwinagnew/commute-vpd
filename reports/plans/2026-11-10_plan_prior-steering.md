# E149: do the prior and steering, together, put every built pair's sides the way the share does?

*Cell P2.15, compare-lead (writer p215-writer), 2026-11-10. Card `reports/plans/2026-11-10_plan_p2-15-card.md`; filed rule
`experiments/prior_steering/results/predictions.json` and `rules.py`; code `experiments/prior_steering/` (`run.py`, `plot.py`);
job `cluster/prior_steering.sbatch`. Sides are labelled as built; no verdict on reuse.*

**Design check.** Goal: part 2 asks whether the re-gating share sees something cheaper readings miss on the 67M. The premise
checker asked whether the share's edge over steering reduces to how steering's strength is chosen; this reads the two
cheapest baselines together, with steering both as filed (searched strength) and at strength 1. Nearest tried: E133's method
table (each baseline alone, one cell each), E145 (the last prior cell), E146 (strength 1 removes steering's three reversals,
read on those pairs only). This is a step for part 2, inside the cell's "done when". No weight is fitted to the share (the
rule is "either baseline calls the share's way and neither calls against it"), so leave-one-pair-out only changes the pair
set; it is reported. No rule is added to a reuse verdict. What changes what we do: rival A closes the cell with part 2
saying the built pairs are covered by the two baselines; rival B gives part 2 its example pair. Control first (below).

**What runs.** One debug job ($0, about 12 min on an L40, 24G host). On every built pair's saved states (68 updates: the
seven E136-inventory pairs and E147's twelve 96-key refits, which exist and passed E147's stage 0), steering at strength 1:
each fine-tune's mean change at the answer position added as it is, the layer (1-3) chosen by the filed rule on the
selection rows, read on the share's read rows over the fine-tune's own gain. Copy vs lookup goes through E135's path, as its
filed cell did; every other pair through E143's. Each pair is called in the kind (pooled or per key) its filed cell reads.
The score then takes final.json's share and prior cells and the filed steering (E147's fair reading for the 96-key pairs).

**The rule** (`rules.py`, no fitted weight). Per pair, votes against the share's direction: prior +1 / 0 / -1 by the sign of
its gap when at least 0.5 decades (bar sourced in predictions.json; calls unchanged for any bar in (0.20, 1.48]); steering
+1 / 0 / -1 by E143's call. "Tracked": a +1 and no -1; "conflict": +1 and -1; "missed": no +1. Rival A iff every pair is
tracked; rival B iff any pair is missed; "mixed" iff conflicts only.

**Expected at the references.** A zero update: steering rises exactly 0 (stage-0 control, both paths); its gain is 0, so the
share is unreadable and a do-nothing pair is left out (counted, its votes would be 0 and 0, "missed", so it cannot make A).
Matched-prior pairs (copy vs lookup, 0.12 decades; copy vs prior-matched cipher, 0.20): prior votes 0, steering decides.
Positive control: each baseline alone reads rival B on saved numbers (prior misses the two matched pairs; filed steering
misses lowercase vs ciphers), so A is not built in. Reproduction controls: E143's and E135's saved selection rises, E146's
strength-1 shares and E135's filed share, within 1e-4 of the gain; a miss stops the job.

**Filed predictions.** P1: as filed, "mixed", with conflicts exactly on half-known vs cipher (both) and rank-1 vs cipher at
step 10 (seen before filing). P2: at strength 1, rival A. P3: at strength 1 steering separates both matched-prior pairs the
share's way (the decisive part). P4: at strength 1 steering separates none of the three pairs where it reversed.

**Edge cases run before filing** (`rules.edge_cases`, cpu-run 2026-11-10): prior at the bar with steering flat: tracked;
just under: missed; prior matched with steering the share's way: tracked; prior against and steering for: conflict, mixed;
share not separating: left out, "no call"; do-nothing (nan steering): left out. A CPU dry run of the score path on stand-in
strength-1 values gave as filed: mixed, conflicts e54_matched, e54_own, open_pair_c3.

**Early stop.** Stage 0 alone; any control miss exits 1. In stage 1 any state that misses a saved reading (or reads a nan)
stops the job; projected time over 25 min after four states stops it. Results: `results/stage0.json`, `results/stage1.json`
(with `score`); figure from `plot.py` after the run.

**Not run.** Any new fine-tune; strengths other than 1 and the filed search; layer or kind changes beyond what each filed
cell reads; the twin pairs; any Qwen pair; any verdict on reuse; any edit to final.json or `plot_final.py`.
