cell: P2.61
rival_a: the 2,000-evaluation cap stopped the polish after a few thousand of ~20,000 prefix members, leaving RL's and SFT's K* padded alike with low-attribution members, which predicts that the uncapped K* is smaller and the uncapped E_b of the six RL sequence seeds sits above the three SFT seeds by at least 0.01
rival_b: E_b reads RL and SFT alike on the sequence task whatever the search budget (the re-gating share's RL above SFT comes from the share, not from the definition's search), which predicts that the uncapped E_b of RL and SFT stays within 0.01 of each other, as the capped one did (RL 0.036-0.049, SFT 0.036-0.038)
rung: parameter
smallest_gap: 0.01
spread: 0.0044
n: 3
design: unpaired
unit: training seeds per kind (SFT s0-s2 against RL s0-s2 and guarded RL s0-s2, each RL kind compared with SFT)
spread_source: experiments/rb_states/results/summary.json
spread_of: proxy: the capped E_b of the six sequence RL and guarded-RL seeds (sd 0.0044; SFT seeds sd 0.0010), the same reading at the old search budget
floor: none
stop: uncapped E_b of RL and SFT within 0.01 on the sequence task closes the cap as the explanation of the E260 disagreement; nothing further on the cap
sketch: experiments/kstar_uncapped/figures/sketch.png
rehearsal: 'E### rehearse:' on a cluster debug L40, ~25 min: sequence_sft_s0 at the full read sizes (E257's windows, all variables), the uncapped search run to the 30-min limit: measures seconds per evaluation, peak GPU memory, the polish's progress rate (members tried per evaluation), throughput with 3 concurrent processes on one card, and whether |K| at 2,000 evaluations equals E257's capped |K*| 21027 (the reproduction control; no bars otherwise)
