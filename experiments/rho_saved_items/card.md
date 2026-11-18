cell: P1.30
rival_a: The twins' new tables live in what their attention heads compute, which the note's level now reads per setting (at E196's level B the identity/random-map gap, 0.0039-0.0043, already exceeded the repeat gap, 0.0013-0.0031, at both seeds, though the random map moved its weights only 1.04-1.13 times the identity table's), predicts the main reading's identity/random-map gap above the repeat gap (and above 0) at both seeds
rival_b: The per-setting 1 - cos^2 sees only the direction of each part's write, which at a neuron is exactly the cos^2 of its output column, a weight comparison, so rho follows update size as E196's neuron reading did, predicts the main reading's gaps ranked case above repeat above both key-table pairs at both seeds, so the identity/random-map gap below the repeat gap
rung: strategy
smallest_gap: 0.0005
spread: 0.0001
n: 2
design: paired
unit: seeds 0 and 1; per seed one comparison of the identity/random-map gap with the repeat gap (each gap a reuse member's rho minus its twin's, mean over 5 orders)
spread_source: experiments/rule_retention_pairs/results/summary.json
spread_of: proxy: E196's level A retention, largest sd over 5 orders of one item (8.5e-5, repeat_twin_s0), rounded up; the same 0-1 scale and the same chunked search, but a different r and level, so the main reading's own order sd is measured in this run and printed beside R1
floor: none
stop: A control in stage 0 misses (do-nothing not "no change"; x2 or x10 not 1; the slotted graft above 0.05; an appended graft not 0; F with every changed part swapped not equal to M): the job stops before any fine-tune is read and the cell waits on the fix. For the line: a rival B result (gaps ranked by update size at both seeds) with the neurons reading their output columns only ends further rho runs at this level until the user answers the per-setting distance question (plan, "For the user")
sketch: experiments/rho_saved_items/figures/sketch.png

## What runs

The working definition read as the theory note writes it (`theory/2026-11-18_definition_causal_abstraction.md`,
starred choices), on items already saved: the five E196 pairs at seeds 0 and 1, E195's rebuilt circuit, and E206's
grafts. The level aligns every path variable (every neuron, every head, the embedding and the unembedding), so F with
every part given M's rule is M. Beside the main reading, one change at a time: the code's aggregate r (E196's
number), activation patching for F_K, M-only settings, and (E195 rebuilt only) zeroed parts restored. Controls first
and alone: do-nothing, the circuit x2 and x10, E206's neurons written into M's own slots (new where M has the slot),
and the appended grafts (0 by the rule: a check, not a test).

## Cost

Two searches per item (mechanism replacement and activation patching), 5 orders each; E196 took 224-622 s per search
on one H100, and each evaluation here also runs M's heads and MLP (1.5-2 times E196's time). 11 H100 SXM pods in
parallel (one per pair and seed, one for the controls and E195 rebuilt; `pods.md`): 7-17 pod-hours, $30-75, 1-1.5 h
wall. The upper end passes the $60 gate line; the first-order logs project each item and the runner reports first.
