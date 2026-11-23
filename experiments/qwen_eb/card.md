cell: P3.14
rival_a: Qwen's fine-tunes change the model mostly by re-weighting what its subcomponents already write predicts E_b of each step-75 update above its matched random change by more than 0.02 at both seeds, with the gain x2 control near the top and the random changes near the 67M's 0.011
rival_b: Qwen's fine-tunes write in directions unrelated to what each subcomponent already writes predicts E_b of each step-75 update within 0.02 of its matched random change, the gain x2 control still near the top
rung: premise
smallest_gap: 0.02
spread: 0.0007
n: 2
design: paired
unit: seeds per trained arm (each update read beside its own random change of the same per-matrix norm)
spread_source: experiments/rb_states/results/summary.json
spread_of: proxy: E_b on the 67M copy-pair seeds (0.0389, 0.0391, 0.0403), the same reading on the same kind of saved fine-tune, one model size down
floor: none
stop: the gain x2 control fails to reach its stage-0 band, or every step-75 update sits within 0.02 of its random change: close P3.14 with that reading and do not read the pair library
sketch: experiments/qwen_eb/figures/sketch.png
rehearsal: a cluster debug L40 job at the full read sizes on one update and its random change (time per update, peak memory, the do-nothing and gain x2 control, the bf16 against fp32 floor) before any pod; the scored run goes to the cluster too if one update takes under ~2 min
