cell: P2.42
rival_a: Neuron-level saturation survives pruning weak activations: a full fine-tune turns every neuron's output column a little and moves activations only across a few thresholds, predicts that R at p97/3 stays at or above 0.99 on all 20 trained fine-tunes and every reuse-minus-twin gap stays under 0.01, as at the width (E213)
rival_b: The fine-tune's change sits in the strong firings (Goodfire: a neuron's influence lies above p97/3), so threshold crossings count as disagreement, predicts that R at p97/3 falls below 0.99 on some trained fine-tune or opens a reuse-minus-twin gap of 0.01 or more on some counted pair
rung: strategy
smallest_gap: 0.01
spread: 0.0006
n: 2
design: paired
unit: seeds 0 and 1; per seed one reuse-minus-twin gap of R per counted pair and featurization
spread_source: experiments/r_bounds_e208/results/summary.json
spread_of: proxy: E213's R at the width (k = width), the same reading at the limit of this featurization; the largest seed-to-seed sd of a counted pair's gap there was 0.00056 (case pair), rounded up; the run prints each gap's seed difference beside it
floor: none
stop: If R reads at or above 0.99 on all 20 trained fine-tunes with every counted gap under 0.01 at p99/1, p97/3 and p90/10, the top-activating-neuron featurization saturates like the plain neuron level; it is reported so in section 4.2.3 and no further neuron-level featurization is run
sketch: experiments/topk_neurons/figures/sketch.png
descriptive: user, 2026-11-19 in chat: run the proper top-activating-neuron baselines soon; featurization = base with weak activations pruned

## What runs

The 20 trained 67M fine-tunes E213 read (E208's saved updates), read by the definition at a featurization fixed from
the base: at each MLP site a neuron is a feature on a token when its activation is above its p97 or below its p3
(percentiles from the base on Pile text), and the rest of the site's activations is one remainder variable. Per item:
s_u of every variable, K* by mean ablation (greedy, one attribution order, alpha 0.9), R, the changed share f, the
remainder's share of K*, and the overlap of the strong-neuron sets of base and fine-tune at the read positions. p99/1
and p90/10 beside, as the sensitivity check; the width (every activation a feature) as the reference. Control first:
at the width, R over every part and over E208's circuits reproduces E213 per item, within a floor measured in the same
job; W0 reads no change. Cluster debug QoS, $0.
