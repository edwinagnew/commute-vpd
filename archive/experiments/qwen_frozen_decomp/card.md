cell: P3.9
rival_a: On p-9707d6ff the KL re-gating share sits at its ceiling because trained subcomponents absorb the update into the same active set, while frozen neuron subcomponents (only the CI function trained, ~9x more active MLP subcomponents) make the share read which neurons the update switches, predicts on p-015e0710 at least one of copy minus cipher or known minus invented reads a share gap outside its random-change band at a save, all three seeds on one side of 0
rival_b: The KL re-gating share reads near 1 for any change at Qwen scale whatever the subcomponent basis (E132: 0.991 to 0.999 for random changes and updates alike) predicts on p-015e0710, as on p-9707d6ff, every pair's gap lies inside the random-change band (|random gap| plus 0.002) at both saves
rung: parameter
smallest_gap: 0.005
spread: 0.002
n: 3
design: paired
unit: seeds (0-2), each a paired side A minus side B share gap of one pair at one save, in share units
spread_source: experiments/qwen_regating_new/results/summary.json (.new.seed_spread 0.00196 on p-9707d6ff, E132's one seed pair; the same spread E170 filed. To be re-measured on p-015e0710 by the run's own random-change stage before any gap is read)
stop: The port check misses (reconstruction at all-ones masks not equal to the base model within E131's tolerance, or the CI active set per token off the README's 45,374 at CI > 0.1 by more than the stated Jaccard gap): no share is read on p-015e0710 and the cell closes with the port reading. If every cell reads inside the band on p-015e0710 too, the re-gating share does not separate the library at Qwen scale on either decomposition and part 3 drops it
sketch: experiments/qwen_frozen_decomp/figures/sketch.png
