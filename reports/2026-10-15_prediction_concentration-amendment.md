# Prediction amendment 1 to the concentration test: Γ's gates fixed to W₀

*2026-10-15, coding agent; confirmed by the user ("go for it"). Filed in
`experiments/concentration_test/results/predictions_amend1.json` before the measure ran. `predictions.json` is
unchanged and stays scored as filed (trigger part unscorable).*

**Why.** The trigger-local control failed. With the components unchanged, the CI-masked P(o) at the trigger rose
from 0.0045 to 0.535, because the frozen Γ reads the fine-tuned activations. That made the measure unable to tell "the
components carry the behaviour" from "Γ's gating carries it".

**Amended measure.** Γ's g is computed once, on the W₀ model's activations over the held-out windows. The component
model uses each arm's components under those fixed masks, with Δ excluded, against that arm's own fine-tuned target.
Do-nothing under these gates is W₀'s own CI-masked model, so the control has power by construction (0.0045 against a
target of about 1). The state-gated numbers are recomputed beside it.

**Predictions.**
- A1, siblings: P(o) ≥ 0.8 × target. Predicted: pass.
- A2, top 10: P(o) ≥ 0.8 × target. Predicted: pass.
- A3, Adam on all subcomponents: P(o) < 0.5 × target. Predicted: pass.

**Caveat.** At W₀ the decomposition is already imperfect at the trigger (KL 0.52), so KL at the trigger is reported
against that baseline.
