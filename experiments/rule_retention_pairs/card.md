cell: P1.25
rival_a: The reuse member of each pair does its task with its neurons' old rules while its twin changes what its neurons respond to, which predicts the reuse member's level A retention above its twin's in all 8 counted pair-seeds, by gaps that do not follow the size of the weight update
rival_b: The neuron-level reading tracks how far each fine-tune moved its weights (the draft's "close to a weight comparison" limit; the twins' MLP updates are 1.07 to 3.1 times their reuse members'), which predicts the same order in every pair but with gaps that follow the update-size ratio: largest in the case pair, smallest in the two key-table pairs
rung: strategy
smallest_gap: 0.001
spread: 0.0001
n: 8
design: paired
unit: pair-seeds (4 counted pairs x seeds 0 and 1), reuse member against its twin on each member's own read windows
spread_source: experiments/rule_retention_fixed/results/stage2.json
stop: Stage 0 misses (do-nothing not "no change", or the x2 circuit not reading 1), or an item's gap does not repeat E184's: the job stops and the cell waits on the fix
sketch: experiments/rule_retention_pairs/figures/sketch.png
descriptive: user, 2026-11-17 chat, run 2 of P1.25 as filed: "filed: in each pair the reuse member reads higher than its twin; memorised-fact pair reported not counted"

## What runs (run 2 of cell P1.25)

The definition draft's "Choices fixed for the test", unchanged from run 1 (`experiments/rule_retention_fixed`), on the
fine-tunes E184 read: copy/cipher, identity/random map, case, repeat (counted) and parity twin/memorised fact (reported),
seeds 0 and 1, 20 fine-tunes. Each is read against M on 512 windows of its own read half, with the contrast's KL at the
last position, where each task is scored. The control runs first and alone: do-nothing must read "no change" and run 1's
circuit turned up x2 must read 1. Then one stage per pair. Filed: the reuse member's mean level A retention over 5 orders
is above its twin's, per pair and seed. Level B and the fact pair are reported.

## Cost

All 12,288 neurons moved in every fine-tune (measured on the weights), so about 12,288 changed units per item against run
1's 6,144. Run 1 took 5,711 to 6,324 evaluations and 123 to 136 s per order, about 21 ms per evaluation on one H100.
Estimate: 12,000 to 17,000 evaluations per order at 15 to 30 ms each, so 3 to 9 min per order and 15 to 45 min per item
(5 orders; level B, the profile and the gap check add about 1 min). 20 items: 5 to 15 H100-hours, about 8 central.
Run as 5 H100 SXM pods in parallel, one per pair (4 items each, about 1.5 to 2 h with set-up): about 8 to 10 pod-hours at
$4.41/h, about $35 to $45; 10 pods (one per pair and seed) halve the wall time for the same cost. The first order's
evaluations and seconds are logged, so the estimate is checked on each pod within minutes.
