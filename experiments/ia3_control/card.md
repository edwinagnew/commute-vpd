cell: P1.61
rival_a: The write share reads reuse as kept on any update that is a gain on M's own variables, trained or built, which predicts that the FFN-gain fine-tune (IA3 gains on the MLP activations only) reads E_b over K at the neuron featurization, remainders in as E277 filed it, at least 0.2 above the full fine-tune of the same task and seed, on every task where it reaches the full fine-tune's task gain
rival_b: The instrument reads low on any trained update whatever the update is (P1.59's finding: per-variable kept shares low and near random in every trained class), which predicts that the FFN-gain fine-tune reads E_b over K at neurons within 0.2 of the full fine-tune of the same task and seed, near the matched randoms, even where it reaches the full task gain
rung: premise
smallest_gap: 0.2
spread: 0.011
n: 3
design: paired
unit: seeds per task (E58's key-grid tasks copy, rank1, elicit, cipher, prior_cipher; seeds 0-2), the FFN-gain fine-tune paired with the full fine-tune of the same task and seed
spread_source: experiments/needed_neurons/results/summary.json
spread_of: this reading (E_b over K at the neuron featurization, E277's trained fine-tunes, sd of the two seeds per pair, largest 0.011 on repeat_real)
floor: |K| < 100 reads "too small to read" (whip-lead, 2026-11-23, as E277); and a task counts for the control only where the FFN-gain arm's mean task gain share (held-out P(target) gain over the full fine-tune's) reaches that task's own seed ratio of the full fine-tunes (min over max, about 0.83 for cipher and 0.96-0.99 for the others in E147's record). Both rivals expect |K| in the thousands; rival A expects most tasks but cipher to reach the gain share, rival B makes no gain claim
stop: If the FFN-gain arm reaches the full task gain and still reads within 0.2 of the full fine-tune at neurons, the paper states that the filed neuron featurization reads low even on a gain update (the remainders cap it) and no further positive controls at neurons are proposed; if the neurons-only check (E_b over K on the needed changed neurons, 1 by construction) misses its floor, the E_b code is wrong and every E_b reading stops until it is found
sketch: experiments/ia3_control/figures/sketch.png
rehearsal: cluster only: cluster debug; every job runs on the MATS cluster debug QoS ($0, the cell says "minutes on cluster debug, ~$0"), each job running its featurization's stage 0 first (identity swap gives zero flips; do-nothing E_b over K is 1; a built per-neuron gain edit measures the floor of the neurons-only check) and stopping on a miss

## What runs (cell P1.61; user 2026-11-23, "yes open the IA3 gain control")

Plan: `reports/plans/2026-11-23_plan_p1-61-ia3-gain-control.md`. Filed rules: `results/predictions.json`.

Two trained arms on E58's five key-grid tasks, seeds 0-2, E147's data, loss, batch, sampler seed and step rule:
- **ffn** (ours, after Liu et al. 2022): only per-neuron gains on the MLP activation (down_proj's input) are trained.
  It is the positive control. Every neuron's write keeps its direction, so its needed changed neurons must read kept.
- **ia3** (as published, IA3's three sites on this llama-style model as HF PEFT targets them: k_proj, v_proj,
  down_proj's input). Descriptive at heads and at VPD.

Each arm is folded into the weights (dW = (l - 1) times rows or columns) and read like any update. Beside every E_b
the run reports |K|, the median top-1 minus top-2 margin, and the matched random (a Gaussian per changed tensor at the
arm's Frobenius norm, E277's construction). The full fine-tunes (E147's saved step-30 updates) are read at neurons
too, as the reference on the same axes, with their own matched randoms. VPD (E268's apparatus) is descriptive.
