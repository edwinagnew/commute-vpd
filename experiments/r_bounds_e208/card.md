cell: P1.34
rival_a: Neuron-level saturation belongs to the level: every neuron and head of a full fine-tune changes a little and none much, whatever the distance, predicts that the worst R over any circuit of k >= 16 parts stays at or above 0.99 on every trained fine-tune under every saved distance, and that every reuse-minus-twin gap at k >= 16 and on E208's own circuits stays under 0.01
rival_b: Neuron-level saturation belongs to the cos^2 distance, which is blind to a change of size, predicts that the size-aware distance gives a reuse-minus-twin gap of 0.01 or more at some k >= 16 on some counted pair at both seeds
rung: premise
smallest_gap: 0.01
spread: 0.0006
n: 2
design: paired
unit: seeds 0 and 1; per seed one reuse-minus-twin gap of R per counted pair, distance and k
spread_source: experiments/rho_saved_items/results/items/case_real_s0.json
spread_of: proxy: this reading at k = all (the mean over every part, the r_changed means in E208's items/*.json for both members and both seeds); the largest seed-to-seed sd of a counted pair's gap under any distance was the case pair under the size-aware distance, gaps 0.01240 and 0.01161, sd 0.00056, rounded up. Tail k (the k lowest scores) can be noisier; the run prints each gap's seed difference beside it
floor: none
stop: If every distance, the size-aware one included, keeps every pair gap under 0.01 at every k >= 16, no per-unit distance at the neuron-and-head level separates reuse from new on these pairs, and further neuron-level reads of R end; the next tests move to coarser levels (MLP layer, attention layer, block)
sketch: experiments/r_bounds_e208/figures/sketch.png

## What runs

A CPU read of E208's saved per-part scores (`experiments/rho_saved_items/results/large/parts/*.npz`, 24 files): for
each of the 20 trained fine-tunes and each of the four saved distances (the note's per-setting 1 - cos^2 on pooled
settings, the code's aggregate cos^2, M-only settings, the size-aware distance), the range of R over every possible
circuit (lowest and highest single part), the worst R for a circuit of k parts (mean of the k lowest scores, k = 1, 2,
4, ..., 4096, all), and R on E208's five searched circuits. Every part of these full fine-tunes is changed, so f = 1
and R is the plain mean of s_u over the circuit. The four built items (slotted, appended neurons, appended head,
rebuilt) are reported only. No GPU, no model, $0.

Disclosure: before filing, the lead had seen each item's lowest and mean score per distance (E208's items JSON and a
listing of the npz files); E208's report had already given the size-aware gap on its own circuits, 0.013 to 0.014 for
the case pair.
