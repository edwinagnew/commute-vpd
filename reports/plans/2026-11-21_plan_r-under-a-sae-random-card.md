cell: P1.40
rival_a: R_A at SAE latents tracks how a fine-tune changed the model, not just its size, predicts every key-task fine-tune reads R_A above every random update of its size by more than E245's band, the random updates near VPD's 0.29
rival_b: the base SAE's per-site remainders and per-latent gains absorb any update of key-task size, predicts random updates of key-task size also read R_A near 1, at or above the lowest key-task fine-tune
rung: parameter
smallest_gap: 0.11
spread: 0.11
n: 15
design: paired
unit: key-task states (five tasks x three seeds), each read once as the fine-tune (E245) and once as E237's random update of its size on the same windows
spread_source: experiments/sae_r_a/results/summary_built.json
spread_of: proxy: E245's split-half band of R_A at SAE latents on the built items (0.1096), the same reading and code on other items; the random items get no halves
floor: E245's band 0.1096: rival a expects min key R_A (0.989) minus max random R_A above it; rival b expects the gap at or below 0
stop: rival b: E245's key-task readings are reported as uninformative at SAE latents (any update of that size reads near 1), and no further SAE reading on the key tasks is run
sketch: experiments/sae_r_a/figures/sketch.png
rehearsal: E245 rehearse (rehearsals.tsv, job 62745), cluster debug L40, do-nothing and x10 at the full read sizes, measured the noise floor (pieces KL 8.5e-12), time per item and peak memory (14.5 GB on H100 in the keys run); this run adds only items E237's and E242's Worlds already built on GPU, checked by a CPU dry run of every item
# descriptive: whip-lead, 2026-11-21 ~15:00 (note-2026-11-21-whip-random-at-sae.md, user approved P1.40 run 3): a reference for E245's key readings, no claim at the bar

# Card: do random updates read low at SAE latents?

**What runs.** E245's reading, R under wording A at the base-only SAE latents, on random updates. There are 15 of them, one per key-task fine-tune, each at that fine-tune's update size and read on its windows. There are also E242's two random updates, which read 0.29 and 0.30 at VPD, so both featurizations are read on the very same updates.

**Expected.** Random updates read well below the fine-tunes' 0.989-0.999. If they also read near 1, E245's key readings say nothing about reuse.

**Cost.** Two RunPod H100 pods, about 2 h each, about $18.

**Decision it serves.** Whether the SAE row of the paper's R table can be read on the key tasks at all.

Plan `reports/plans/2026-11-21_plan_r-under-a-sae-random.md`.
