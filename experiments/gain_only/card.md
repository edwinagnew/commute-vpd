cell: P2.46
rival_a: The built reuse tasks ask the base model to turn up computation it already has, which one gain per neuron and head can do, while the built new tasks need computation it lacks; this predicts that at the full fine-tune's own output change the gain-only states reach at least half the full fine-tune's trained-answer gain on the reuse sides and under half on the new sides, on at least seven of the nine built pairs read
rival_b: At the unit of one static gain per neuron and head both kinds of built task are equally reachable or equally out of reach, so the built difference is not a "how strongly" difference at that unit (as E205's held model, at most 0.104, suggests); this predicts the two sides of a pair get the same call (both reach half, or neither does) on at least three of the nine built pairs read
rung: premise
smallest_gap: 0.25
spread: 0.03
n: 3
design: paired
unit: gain-only runs (seeds) per task; the two sides of a pair share the recipe family, the read-row format and the seeds
spread_source: experiments/share_untrained/results/stage1.json
spread_of: proxy: the full fine-tunes' seed sd of the trained-answer gain over its mean on the same read rows (gain_fp32 per state; median 0.017 over 21 tasks, at most 0.124 for fresh/real), which is the reach ratio's denominator and has its units (a fraction of the reference gain); the gain-only seeds' own sd is measured by the run
floor: MIN_GAIN, 1 log-odds, on each reference's mean gain before its reach is read (all clear it: the smallest is e53/lowercase's 1.64); rival a expects reuse reach at or over 0.5 and new under 0.5; rival b expects both sides on the same side of 0.5
stop: If the gain-only states get the same call on both sides of most built pairs (both reach half, or neither does), a static gain per neuron and head is not the unit that separates the built pairs; that ends the literal "how strongly" test at neurons and heads, and the section reports it beside E205's 0.104
sketch: experiments/gain_only/figures/sketch.png
rehearsal: 'E### rehearse: gain-only' on one cluster L40 (debug QoS, cluster/gain_only.sbatch rehearse): stage 0 (gains of 1 change no bit, the exported change of a random gain state reproduces its logits and a shuffled one does not, the read path reproduces E136's gain) and four items, one per recipe family (c3/rank1, e62/lookup, e53/lowercase, case_label/twin, seed 0), at the full read sizes and full training data with each run capped at 20 steps, then trained, exported in bf16 and read through E228's share; run twice so the second must skip every item (resume). Measures s/step per family, read seconds per state, peak GPU memory, state file size, the stage-0 floors and the step-1 KL against each reference's
