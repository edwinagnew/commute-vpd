cell: P1.66
rival_a: Fig 2's five key-grid tasks were representative, so this run predicts the 11 added known-answer tasks read like the five at every featurization: every trained fine-tune's retained share within 0.05 of its own matched random's or below 0.1, with no reuse/new order among trained tasks
rival_b: The five tasks were not representative, so this run predicts at least one added task's mean retained share (3 seeds) sits more than 0.05 above its matched random at one featurization, or reuse tasks read above new tasks there
rung: parameter
smallest_gap: 0.05
spread: 0.003
n: 16
design: paired
unit: known-answer tasks, each the mean of 3 seeds, read against its own matched random; descriptive, no bar
spread_source: experiments/ia3_control/results/summary.json
spread_of: proxy: neuron E_b_K of E290's full fine-tunes, sd over 3 seeds within a task (0.000-0.004), the same reading at one featurization
floor: none
stop: Any stage-0 control at any featurization reads other than expected (do-nothing E_b 1; identity swap 0 flips; a saved E298 item re-read through this driver within E298's repro tolerance): the job stops and the cell waits on the fix
sketch: experiments/fig2_standard/figures/sketch.png
rehearsal: cluster only: none; gated checks waived by the user (2026-11-27, whip-lead chat: "yes, full rerun, waive the gates"); E298's stage 0 (do-nothing, identity, saved-item reproduction) runs first on each paid card and stops the job on a miss
descriptive: user, 2026-11-27, in whip-lead's chat: "yes, full rerun, waive the gates" (on extending Fig 2's retained share to all 16 known-answer tasks so it uses the same tasks as Figs 3 and 7); reported as a description, no claim at a bar

## What runs (run 1 of cell P1.66)

E298's driver (experiments/fig2_standard: one needed-set rule, 256 swap pairs, leftovers counted) on the 11 known-answer
tasks Fig 2 lacks, x 3 seeds, each with one matched random of E290's recipe, at VPD, top neurons and SAE latents (66
items per featurization). The 15 states E298 already read are reused after a stage-0 reproduction check. Rows go into
figures/data/definition.csv's format. Estimate from E298's measured per-item times: 66 x (289 + 230 + 122) s = 11.7
GPU-hours, ~$115, on up to 8 pods.

Reuses: E325's code, predictions.json (PRED_SHA fa307f23...) and check_states pass (whip-lead's pod, commit 8cc7cf24) unchanged. E325's claim stayed with whip-lead when the cell moved to definition-lead (2026-11-27), so job-gate refused definition-lead's pods; this number is the same run under the cell's owner.
