# Claim 2's known-answer fine-tunes: what is missing and what to run (writeup-lead, 2026-11-22 night)

Claim 2 says cheap readings track the definition. To show it we need trained fine-tunes whose degree of reuse is known
without the readings under test, spread from none to full, at matched behaviour change. We do not have them yet: the
built edits are known but hand-made, and the key tasks' "reuse" and "new" labels are what we intended, which the
definition does not confirm. This plan proposes one new run that sets the answer by construction ("the reuse dial"),
and lists the re-reads already running.

## Why the current set falls short
- Key tasks, labelled by design. Under the definition (E_b at VPD over K*), all 94 SFT fine-tunes of the nine pairs
  read 0.015 to 0.062, against 0.011 for random updates. The side built new reads higher on 6 of 9 pairs (E257,
  definition-lead reports/analyses/2026-11-21_analysis_e257-eb-trained.md).
- Gain-only training at neurons and heads reaches 0.72 to 0.91 of the cipher tasks' gain (appendix sec:app_held_gain).
  So at that featurization a "new" task can be learned by rescaling alone.
- The prior-matched result (E253-E255: the re-gating share and the other cheap readings follow the built mechanism, not
  the prior) scores the readings against the design labels, not against the definition. It shows the readings are not
  the prior. It does not show that they track reuse.
- Pairwise comparisons: the user asks to read all fine-tunes at once. P2.62 (running, compare-lead) recomputes every
  reading's rank correlation against E_b over K across built, prior-matched and procedural states, $0.

## Proposed run: the reuse dial (new cell, 67M)
On one task, train fine-tunes that all reach the same behaviour (same trained-answer log-odds gain), with the update
written as a gain on each base variable plus a free residual, the residual penalised by beta.
- beta = infinity: gains only. Every variable is kept, so rho = 1 at that featurization by construction.
- beta = 0: an ordinary fine-tune.
- 3-4 values in between. The definition (E_b over K) is read on every state to confirm the dial moves rho
  monotonically. That reading is the ground truth; the beta setting only builds the spread.
- Gains on VPD subcomponents (rho = 1 at VPD exactly) and, as a second dial, on neurons.
- Tasks: copy, cued copy and cipher, which span the prior. Within one task the prior is fixed, so the confound is
  removed by design. 3 seeds; matched random updates as the floor.
- Readings on every state: re-gating share, steering, effective rank, crosscoder, ADL, neuron shares, KL.
- Claim 2 holds if the cheap readings' rank correlation with rho within each task beats KL's and the baselines'.
- Expected at the reference: do-nothing rho = 1; gains-only rho = 1 (exact, a construction check); random updates
  about 0.011. A gains-only cipher that cannot reach the target gain is reported as such (a shorter dial).
- Size: 3 tasks x ~5 beta x 3 seeds x 2 dials, about 90 fine-tunes of minutes each, plus E_b over K on each.
  E257 read 94 items with K* searches for about $48; estimate $50-100 and half a day on pods, so a gated-size run.
  whip-lead to check against the gate.

## Questions without a known answer
- (a) Realistic fine-tunes. E_b has not been read on the 67M real-text fine-tunes (topic label, French, lowercase;
  P2.38). Proposed: read E_b over K plus the cheap readings on those saved states ($0 to small; the states are saved).
  Qwen has E_b (P3.14) but no definition verdict.
- (b) RL against SFT: whether RL builds new mechanisms less often. Have: repeat (RL reads 5-10x SFT), three procedural
  tasks (alike), Qwen maths (RL gaps 0.091 and 0.131 against SFT 0.055 and 0.105, but a collapsed null reads as high
  as RL). P2.61 (running) checks whether the capped search explains the sequence-task disagreement. No new run
  proposed unless the user wants a realistic RL task on the 67M.

## Not run
Everything above under "Proposed".
