# E132 Amendment 1 (option b): log-odds re-gating share on Qwen, design notes

*qwen-lead, 2026-11-10. Not yet run. The user asked to prepare option (b) of
`reports/decisions/2026-11-09_decision_qwen-regating-share.md` as an amendment of E132, and to state plainly if it
is a post-hoc deviation.*

**In plain words.** E132 read a re-gating share near its ceiling, about 0.99, for every saved Qwen update and for both
random changes. Option (b) replaces that KL-based share with the log-odds share that `GOALS.md` defines. A $0
recomputation on E132's saved numbers suggests the ceiling comes from where the gate reads its inputs, not from
measuring in KL. If so, switching to log-odds would not give the room the option was meant to buy.

## This is a post-hoc deviation
- It departs from E132's filed method. E132's plan chose "E57's KL-based share ... not GOALS.md's log-odds r". It
  also departs from E45 and E57, which filed the KL share for Qwen before that.
- It was chosen after E57 (old decomposition) and E132 (new decomposition) both read 0.991 to 0.9995 for every state.
- Turning "the trained answer at the trigger" into something on the Qwen rows (which target, which positions) is new
  design, done knowing that result.
- The other side: `GOALS.md`'s r is the project's written definition. The KL share came in on Qwen without a
  recorded reason (`reports/plans/2026-10-28_plan_qwen-q2.md:227`). `decisions.md` (2026-11-02, "No decision on
  record chose log-odds for the route split") says the unit was never decided. Log-odds r has never been read on Qwen.

## The $0 check
The share can be measured against the original model's own reading instead of the split's: (K00 − K01) / (K00 − K0)
rather than / (K00 − K11). This is arithmetic on `experiments/qwen_regating_new/results/S_*.json` `.means`.
Measured this way, gates alone take back most of every state's change:

| State | Against the original model's reading | E132's filed share |
|---|---|---|
| e40 | 0.942 | 0.9992 |
| gold_s0 | 0.927 | 0.9992 |
| a2long_s0 | 0.925 | 0.9946 |
| a2long_s1 | 0.938 | 0.9966 |
| gauss | 0.919 | 0.9909 |
| inspan | 0.881 | 0.9920 |

So Γ, reading the changed model's activations, reproduces most of any change, including random ones. This matches
the 67M finding in `reports/analyses/2026-10-29_analysis_e44-amend1-layer-cut.md:84`.

## Candidate definitions
- **Definition 1 (recommended as faithful).** The target is the trained answer, teacher-forced, at every response
  position of the update's own rows:
  - gold GSM8K rows for e40 and gold_s0;
  - held-template fact rows for each anchor 2 seed.

  The log-odds is computed with `identity.log_odds`. r = Σ(L01 − L00) / Σ(T1 − T0). It is scored only when the mean
  gain per position is at least 1 (`local_faith.MIN_GAIN`).
- **Expected gains.** The updates should gain about +1.9 to +2.9 (from the training-loss drops in
  `experiments/qwen_updates/results/E_*.json`). The random changes taught nothing and should gain about ±0.5, more
  likely negative. So **the random reference is most likely unreadable, not low**, as with E46's Gaussian rows on the
  67M.
- **Definition 2 (rival).** The target is each state's own top token, at the positions where that differs from W0's
  top token. The random references then become readable. But the positions are chosen on each state's own output,
  which is selection on outcome. The K0-referenced numbers above suggest it would read about 0.9 for the random
  changes too.
- **Possible references if the random r cannot be read:**
  - (i) the gain and the numerator Σ(L01 − L00) themselves, placed beside the updates';
  - (ii) Definition 2 as the reference for the random changes only;
  - (iii) r restricted to the positions where the state gains at least 1.

## Controls, cost, code
- **W0 as the fine-tuned model:** both sums are 0 within 1e-6.
- **Decomposition-free check:** gold_s0's own-row KL is 0.8583, read within 1e-3.
- **Positive control (r = 0):** E40's δW kept only at `layers.27.mlp.down_proj`. Γ's inputs are then unchanged, so r
  = 0 within 1e-6. This needs a check that Γ reads only the decomposed matrices' inputs.
- There is no positive control for a high r. The do-nothing control of E132 does not apply, because r uses only the
  original subcomponents and needs no split.
- **Cost:** no split is needed. About 12 forward passes, about 35 min on an L40 in the `debug` queue, $0. Adding
  Definition 2 brings it to about 50 min.
- **Code:** `experiments/qwen_regating_new/amend1.py`, about 200 lines, calling E57's `Lab`.
- **Unchecked:**
  - whether gold's own rows overlap E40's training batches;
  - the random changes' real gains, which the job's first stage would read.

## Status
Waiting on the user. The ask is whether to run Definition 1 with reference (i), given the $0 check, or to stop at the
$0 check.

## Not run
Anything on a GPU. No cell, experiment number or predictions are filed yet.
