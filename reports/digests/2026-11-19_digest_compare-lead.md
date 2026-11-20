# compare-lead digest, 2026-11-19 (evening)

P2.41 (RL on the 67M, due 11-22) waits on you: its four runs are spent and the last one read no call because the rows
scored the wrong thing on lowercase; one more free read of the saved updates, scored as a two-form contrast, would fix
that. P2.42 (top-activating-neuron baseline, due 11-21) is filled, with two filed leans reversed. Finished because
everything left waits on that one answer.

## Cells
- **P2.42, filled** (1 of 2 runs used; the second was not needed after your 11-19 respecification). With the base
  model's weak MLP activations pruned at p97/p3 per-neuron thresholds as the featurized model, the definition reads
  0.96-0.99 on all 20 trained fine-tunes, below about 1.00 at full width. Reversed leans: fine-tunes change which
  neurons fire strongly far more than filed (overlap 0.23-0.36, filed 0.7-0.99), and the pooled weak firing changes
  more (remainder 0.83-0.94, filed at least 0.99). Premise check: doubtful (neurons may not suit this model's updates).
  Report `reports/analyses/2026-11-19_analysis_e217-topk-neurons.md`.
- **P2.41, open, blocked on ask 29** (4 of 4 runs used). E222 read the share and the scorecard baselines on the 27
  size-matched states: only RL on repeat (share 0.68-0.79) and the French anchor (0.53-0.56) clear the 1 log-odds
  floor, so the filed call is no call. RL on lowercase reads negative (-2.25 to -3.33): the rows scored the text's exact
  next word, and RL learned lowercase word salad (the drop tracks drift, r -0.97). Report
  `reports/analyses/2026-11-19_analysis_e222-rl-readings.md`.

## Asks waiting on you
- **Ask 29** (card `reports/decisions/2026-11-19_decision_p2-41-read-unit.md`, in log v411): allow a 5th run, $0 on the
  cluster, ~20 min, no retraining: re-read the same 27 saved states as a contrast (lowercase against capitalised form of
  the same word; the token a copy would write against the text's token)? Recommended yes; otherwise P2.41 closes no
  call, and stage 4 (the loophole reward, which also needs your yes) would inherit the same scoring problem.

## Money and friction
- Spend: $0 this iteration (all four E222 jobs on the cluster debug queue: 61971-61974). No pods.
- Friction 920: `fleet/rehearsed` error reads like a missing noise floor when the job simply lacked a rehearse comment.
- Friction 921: jq here reads `.e220` as a number; keys like that need `.["e220"]`.

## Not run
- The contrast re-read (waits on ask 29); stage 4; definition-lead's R on the E220 states (note left for them).
