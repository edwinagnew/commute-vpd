cell: P2.45
rival_a: E217's low overlap and lowered R come from the fine-tune moving the strong positive firings themselves predicts that, with the upper tail only, the strong-neuron overlap stays within 0.1 of E217's at the same percentile and R_read within 0.01 of E217's on most trained fine-tunes
rival_b: E217's below-p3 tail sat on GELU's floor (every neuron's p3 is -0.1699 to -0.17), so tiny pre-activation shifts flipped those features in and out predicts that, with the upper tail only, the overlap rises by 0.1 or more over E217's at the same percentile and R_read rises toward the full width by 0.01 or more on most trained fine-tunes
rung: parameter
smallest_gap: 0.01
spread: 0.0006
n: 20
design: paired
unit: trained fine-tunes (10 tasks x seeds 0 and 1), each read at the same percentile by E217 (two-sided) and by run 2 (upper only)
spread_source: experiments/topk_neurons/results/summary.json
spread_of: proxy: E217's R_read at p97/3, the same reading at the two-sided featurization; the largest seed-to-seed difference of one task's R_read there was 0.00082 (repeat_twin), sd about 0.0006
floor: none
stop: If the upper-tail overlap and R_read at p97 both sit within the rival-A margins of E217's on most fine-tunes, the two-sided reading stands as section 4.2.3's neuron reading and no further neuron-level featurization is run in this cell
sketch: experiments/topk_upper/figures/sketch.png
rehearsal: cluster only: every stage runs on the MATS cluster debug QoS (L40), $0, as E217's stage 0 did; no pod
descriptive: user 2026-11-20 dialog, "Yes, upper tail only", re-read after the GELU departure

## What runs

E217's top-activating-neuron featurization of the 67M re-read with the upper tail only: at each MLP site a neuron is a
feature on a token when its post-GELU activation is above its p97 (main), p99 or p90, percentiles from the base on Pile
text (E217's saved file); every other activation at the site, including GELU's negative floor, is the one exact
remainder. Same 20 trained fine-tunes, same readings (R_read, R_all, f, remainder share, remainder s, |K*|, overlap),
each reported beside E217's at the same percentile. Control first: W0 at every band, then the full-width control
reproducing E213 within E217's floor. Cluster debug, $0.
