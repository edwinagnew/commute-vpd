# definition-lead digest, 2026-11-17 morning

We tested your new candidate definition, causal-rule retention, and got about half way through calibrating it. The
definition behaved as it should on the cases we could build. On the unchanged model every unit scores one. A circuit
whose output we doubled scores high: 0.80 with single neurons as the units, and 0.91 with heads and whole MLP layers.
But the case that tells reuse from new machinery was never built: a circuit swapped for different neurons that compute
the same values. In both attempts, fresh neurons trained to reproduce the circuit either copied one of its own neurons
or failed to reproduce it. So the definition is neither confirmed nor refuted, and the trained pairs were not read.
Separately, the earlier curve reading was retested on fresh seeds and failed its filed rule for the same pairs as before.

**Read first:** `reports/analyses/2026-11-17_analysis_p1-24-rule-retention.md`, especially the figure
`experiments/rule_retention/figures/replaced_fits.png`. It shows that no fit lands where the swap would count as
"fits and is not a copy". The likely reason is that five of the circuit's eight neurons fire nearly in step, so one new
neuron reproduces them best by copying them.

**Your decision (card `reports/decisions/2026-11-17_decision_p1-24-next-cell.md`):** what the next cell should
test. The options are below; I recommend (a).
- **(a)** Build the swap by fine-tuning: knock the circuit out and let training rebuild its effect elsewhere.
- **(b)** Forbid the new neurons from reading the circuit's inputs at all. They will probably not fit.
- **(c)** Accept a moved copy as the swap. This tests only where the circuit sits.
- **(d)** Read the five trained pairs with only the two calibration cases that worked.

Asked without a recommendation, from the premise check: if a fine-tune's new units reproduce an old circuit's outputs,
what should the definition call that, and what observation would decide it?

**Cells.**
- **P1.24 closed**, 2 of 2 runs (E193, then the rebuild E194 you chose), no call.
- **P1.23 filled**, 1 of 2 runs. On fresh seeds the curve rule fails on both readings: 7 of 10 on the full
  distribution, 8 of 10 on answer tokens only. Copy/cipher, identity/random map and twin/fact separate on every seed.
  Repeat never separates, and case separates only narrowly. Page `reports/analyses/2026-11-17_analysis_e192-curve-test.md`.
- No open cell is left. The premise check on P1.23 asks whether the case and repeat pairs really differ in reuse,
  apart from how they were built.

**Money:** about $11.20 this session ($7.75 for the curve test and about $3.45 for both retention runs), all on
RunPod H100s. No machine is running. **Blocked:** nothing; the next step waits on the card above.

**Friction noted:**
- The cell-add approval check missed a request you had typed in chat twice, so you had to approve by hand both times.
- Pods have no GitHub token, so the clone-first set-up step fails.
- The H100 SXM price cap in briefs was $0.20 under the market price.
