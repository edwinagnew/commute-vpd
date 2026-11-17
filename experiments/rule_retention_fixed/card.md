cell: P1.25
rival_a: The fine-tune rebuilds the circuit's effect by changing what later-layer neurons respond to, so the neuron-level reading predicts the rebuilt item's retention for the contrast clearly below the turned-up items' 1 (below 0.9)
rival_b: The fine-tune reaches the doubled circuit's outputs by small changes of gain on neurons that keep their direction (the draft's stated limit: cheap and close to a weight comparison), so the reading predicts the rebuilt item reads within 0.01 of 1, like the turned-up items
rung: strategy
smallest_gap: 0.1
spread: 0.05
n: 2
design: unpaired
unit: built items on the turned-up side (x2, x10), each read against the one rebuilt item; descriptive, no bar
spread_source: experiments/rule_retention/results/amend1/stage1.json
stop: Any code check (do-nothing, x2, x10, exact copy) reads other than 1: the job stops and the cell waits on the bug's fix
sketch: experiments/rule_retention_fixed/figures/sketch.png
descriptive: user, 2026-11-17 chat: "Filed: reads lower than the turned-up items. No copy check, no bars."

## What runs (run 1 of cell P1.25)

The definition draft's "Choices fixed for the test" (neuron level; distance 1 - cos^2 with the fitted gain reported; one-to-one
re-pairing within a layer for the greatest total retention; comparative reading) on the 67M. Code checks first, alone: do-nothing,
the P1.24 circuit (layer 1, 8 neurons) turned up x2 and x10, and the circuit moved exactly onto 8 fresh neurons of layer 1. All
must read 1; the job stops if not. Then the rebuilt item: circuit zeroed, embeddings and layers 0-1 frozen with all attention and
the unembedding, the MLP weights of layers 2 and 3 fine-tuned so the outputs match the x2 item. Reported: retention for the
contrast against M on copy24 windows, the fitted gains, the per-unit profile, and the side reading at heads plus MLP blocks.
