cell: P3.10
rival_a: On the frozen-neuron decomposition Γ reads each neuron's actual output at the down_proj site, so a neuron whose gate_proj row is zeroed is switched off by Γ's own masks and the share reads a switching-off as re-gating, predicts the built change (used neurons switched off) reads above the random curve at its own K00 by more than the random band on at least 80% of the readable (row set, set choice) reads, at least 9 of 15 readable
rival_b: The share on this decomposition tracks only the change's output size, as it read near 1 for every change on p-9707d6ff (P3.8) and fell with size for random changes and fine-tunes alike on p-015e0710 (P3.9), predicts the built change reads inside the random band at its own K00 on at least 80% of the readable (row set, set choice) reads
rung: premise
smallest_gap: 0.02
spread: 0.0063
n: 5
design: paired
unit: row sets (copy, cipher, known, invented, gold); each a built change's share minus the random curve's at the same K00 on the same 128 rows, in share units (3 set choices per row set read beside each other)
spread_source: experiments/qwen_library_share/results/p-015e0710/run2 (the largest sd over seeds 0-2 of one fine-tune's share at one save, 0.0063, known late; pooled 0.0030). The random draws' own spread is measured by this run (3 draws at each size) and is the band the rule reads
stop: A stage-0 control misses (W0 cells equal K0, do-nothing share 1, K0 against floors.json): no share is reported until fixed. If the built change reads like random (rival B), the share does not read a known switching-off on Qwen on either decomposition, and part 3 drops the re-gating share on Qwen and says why
sketch: experiments/qwen_share_reference/sketch.png
