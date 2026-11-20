# Why every E226 rehearsal search hit its cap, and why the scored run goes ahead as filed (P1.37, due 11-22)

*2026-11-20, definition-lead. Evidence: the rehearsal's saved files (`experiments/kstar_vpd/results/rehearse_dl/`,
`results/large/rehearse_dl/`), read on CPU by an Opus agent; its scripts are in
`~/commute-data/scratch/definition-lead/kcap/` (a1 to a4.py). Nothing was edited and no GPU was used.*

Every search in the rehearsal ran out of its 2,000 tries, so the circuit it returned is an upper bound, and a loose
one: the do-nothing circuit kept 33,177 of 38,938 variables. Most of that size is real for this featurization: dead
subcomponents matter jointly, which E20 had already measured. The search also has two flaws that can be fixed, but the
reuse score barely moves with the circuit. The built new edit reading above a reuse item comes from the grade, not
from the search, so the scored run goes ahead as filed.

## Findings

- **The circuits are large because the dead subcomponents count jointly.** On the do-nothing state, K\* holds 9,295 of
  the 9,881 live subcomponents and 23,862 of the 29,031 dead ones. The live/dead split is `transport_test/alive.pt`:
  mean gate above 1e-6 on Pile, same decomposition. E20
  (`reports/analyses/2026-10-21_analysis_cancellation-complement.md`) measured that masking only the dead subcomponents
  at W0 costs 1.07 nats. The do-nothing bar here is 0.377 nats, so no circuit that leaves out the dead mass can pass.
- **The search has two fixable flaws.**
  - It ranks by the size of the attribution and ignores its sign: 13,144 of the 33,307 prefix members have a positive
    attribution.
  - Its polish starts with no slack and tries the least-attributed variables first. On do-nothing, about 1,955
    evaluations removed 130 variables.
- **The reuse score barely depends on the circuit.** R at the prefix set against R at the polished set:
  - new_thin: 0.9959 against 0.9963;
  - x10: 0.9755 against 0.9755;
  - the cipher pair: 0.99890 against 0.99890.

  Every gap is under the filed band floor of 0.014. The inversion (new_thin above x10) sits in the grade s. Mean s over
  the changed subcomponents is 0.955 for new_thin and 0.740 for x10: the squared cosine reads x10's large change as
  less kept than new_thin's small change spread over many variables. That is the card's rival (b), and the scored run
  reads it.
- **KL(F‖F_empty) below KL(F‖M) on new_thin is not a problem.** The bar uses only F_empty.

## Decision (lead, 2026-11-20)

The scored run goes ahead with the filed settings (eval_cap 2000) on four H100 pods, about $30. The filed readings and
rules are unchanged, and both the prefix and the polished sets are reported, as filed. The project rule is not to cap
a definition search for cost. So whether R moves when the search runs to convergence is measured on three searches
(do-nothing, new_thin, x10) as a side check, about $21. It needs its own small script, written next; the filed
readings do not read it.

## Not run

- The signed-order iterative search (estimated about 3,000 evaluations per search, about $45).
- The cluster GPU check of the dead-only ablation, one matrix at a time.
- Exact single-variable ablations against attribution.
- Any change to the grade s: that grade is part of the adopted definition, and only the user changes it.
