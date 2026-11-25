cell: P1.31
rival_a: Coarser levels group a re-triggered circuit with the parts it feeds, so swapping in M's output keeps a reuse edit's behaviour and removes a new edit's (the user's expectation behind the pass rule) predicts some level among head, MLP layer and block with both reuse edits' R above both new edits' R by more than the band, reuse small near 0.6 or more there, and the new edits near 0
rival_b: The parts F needs, as clause 5 formalises them, hold only changed parts, and reuse small and new few change the same eight layer-0 slots, so each concentrated edit reads only by the part that holds it, and swapping that part gives M (plan appendix section 3) predicts R exactly 0 for reuse small, new few and new thin at the head, MLP-layer and block levels, reuse large above 0 there only through its gained head, new thin highest at the neuron level, and no level passing
rung: premise
smallest_gap: 0.01
spread: 0.0001
n: 8
design: paired
unit: cue pairs (A_i, A'_i): the same 512 windows, 64 per cue, serve all four edits; each R averages over the eight cues
spread_source: experiments/rho_saved_items/results/predictions.json
spread_of: proxy: E208's largest sd of one item's rho over 5 orders (1.5e-5, reports/analyses/2026-11-18_analysis_rho-saved-items.md), rounded up; the same 0-1 scale and the same chunked search, but a cos^2 r rather than the interchange r, so this run measures its own band in stage 1 (3 x the larger of the order sd and the split-half difference) before stage 2 reads
floor: The band, measured in stage 1 per level and reading; the coarse-level R of a one-part K is exact, so its band can be 0 and the rule is strict. Rival A expects gaps of about 0.6 at a coarse level, far above any band; rival B expects 0 against 0 there, which does not pass
stop: Stage 0 (do-nothing not "no change", the hybrid gate over 1e-3, M's induction not reproduced) or stage 1 (an edit missing its behaviour, the all-swapped gate over 1e-3) stops the job before any reading. For the line: a rival B result under the filed reading (no level passes) ends readings at new levels until the user answers which parts F needs (plan, point 1); run 2 then does not run
sketch: experiments/known_answer_levels/figures/sketch.png

## What runs

Four hand-built edits of the 67M, with labels fixed by construction, all read on one set of 512 copy windows ending in
a new cue A′_i. Reuse small: eight quiet layer-0 slots translate A′_i into A_i, and M's unchanged induction heads (1,1)
and (2,4) copy B. Reuse large: the same, with (2,4)'s output turned up to the new edits' update norm. New few units:
the same slots write a fixed random answer y_i. New thin: the least-norm change of every layer-0 output column writes
the same answer. Each is read at four levels (neuron, head, MLP layer, block) by the user's interchange retention under
mechanism replacement (the note's starred choice) and under activation patching. The do-nothing control runs first
and alone. E195's rebuilt circuit is reported, not an anchor. Run 2 reads E196's five trained pairs at seeds 0 and 1
once, only if a level passes.

## Cost

There is no training. The one long search is new thin at the neuron level (3,072 candidates). About 20-30 min on a
cluster L40 debug job ($0). If it does not start at once, a RunPod H100 SXM at $4.40/h: about 45 min with set-up,
about $3.50. Run 2 at a coarse level is about 10 min, under $1.
