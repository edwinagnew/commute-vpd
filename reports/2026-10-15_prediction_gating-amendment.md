# Prediction amendment 2 to the concentration test: which subcomponents switch on at the trigger?

*2026-10-15, coding agent; confirmed by the user ("go for it the cheap test"). Filed in
`experiments/concentration_test/results/predictions_amend2.json` before the measure ran.*

**Why.** In amendment 1, the arms explain the behaviour at the trigger only when Γ re-gates on the fine-tuned
activations. Do-nothing, with every subcomponent at W₀, reaches CI-masked P(o) 0.535 through re-gating alone. The
"reveal" reading says the update switches on subcomponents that already push towards `o`.

**Measures** (arms: do-nothing, Adam on all, top 1, top 10, siblings):
- *dg*: the change in each subcomponent's mean g at the trigger, between the state's gates and W₀'s gates.
- *S_on*: the subcomponents with dg > 0.3.
- *Recovery*: hybrid masks, i.e. W₀'s gates with only S_on taking the state's gates, as a fraction of the log-odds
  gap between W₀ gates and state gates. Negative control: a random equal-size set with |dg| < 0.05. Positive control:
  every subcomponent takes the state's gates, which gives 1 by construction.
- *Switch-on effect*: at W₀, force one subcomponent's mask to 1 and measure the change in log-odds of `o`. Taken over
  the S_on union, against 300 null subcomponents.

**Predictions (all predicted to pass).**
- B1: recovery ≥ 0.5 for do-nothing, top 10 and siblings, with the random control < 0.2.
- B2: Jaccard overlap of S_on ≥ 0.3, for do-nothing vs siblings and for do-nothing vs top 10.
- B3: mean switch-on effect of S_on > 0 and > the null mean + 3 SE.
