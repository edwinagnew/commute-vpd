# E144 = E132 Amendment 1 (option b): log-odds re-gating share on Qwen

*qwen-lead, 2026-11-10. Cell P3.2 (rung L3.1, due 11-12, 1 run). Card `experiments/qwen_regating_new/results/card_amend1.txt`.
The user chose option (b) of `reports/decisions/2026-11-09_decision_qwen-regating-share.md` on 2026-11-10 and, after
seeing the $0 check below, said "Run it". The harness gives a changed design its own number. So this is E144 and is
also called E132 Amendment 1. **It is a post-hoc deviation from E132's filed method, and every write-up says so.***

## Design check
- **Goal.** This is part 3: the key readings repeated on Qwen3-1.7B. It asks whether the re-gating share can vary at
  all on Qwen once it is measured in the unit `GOALS.md` defines.
- **Nearest tried.** E132 and E57 read the KL share at 0.991 to 0.9995 for every state. Log-odds r has never been
  read on Qwen. On the 67M, the Gaussian rows of E46 gained under 1 nat, so their share was unreadable. This run
  learns whether the Qwen ceiling is a KL artefact.
- **Patch or step.** A patch on E132, chosen after its result (post hoc).
- **Decisions relied on.**
  - It departs from E132's filed choice "E57's KL-based share ... not GOALS.md's log-odds r", by the user's choice.
  - It relies on `decisions.md` 2026-11-02, "No decision on record chose log-odds for the route split".
- **Definition.** No reuse verdict. Unreadable cases are reported as "unreadable".
- **What changes what we do.**
  - All four updates read 0.85 or above, within 0.1 of each other: the share is at its ceiling in both units, and
    part 3 takes (a) or (d).
  - Any update reads below 0.85 and more than 0.1 under another: the log-odds share becomes Qwen's row.
- **Control first.** The W0-as-fine-tuned control runs alone first and stops the job on a miss.
- **Numbers.** Read on each update's own response positions, Δ off:
  - r per update, with its 95% row-resample interval;
  - for each random change on the same rows, the gain T1 − T0 and the numerator Σ(L01 − L00), plus r where the gain
    allows;
  - the seed-pair difference in r.

  What each should read:

  | Number | W0 | Do-nothing | Positive control | Updates | Random changes |
  |---|---|---|---|---|---|
  | Both sums | 0 | — | — | — | — |
  | r | nan | does not apply (r needs no split) | 0 | 0.85 to 1.0 expected (rival A) | unreadable expected (gain under 1) |
- **Guards.**
  - The W0 sums are within 1e-6.
  - gold_s0's unmasked own-row KL is within 1e-3 of E45's 0.8583.
  - The gates of the positive control equal W0's bitwise.
  - Each guard logs its reading when it passes.
- **Saved.** Per-row sums of T0, T1, L00 and L01 per state and row set, in a small JSON, with the response masks'
  counts.

## Run
One `debug` L40 job, about 35 min, $0, in this order: W0 control, positive control, the four updates, then the
random changes on each update's rows. Code: `experiments/qwen_regating_new/amend1.py`. Filed rules:
`results/predictions_amend1.json`.

---

# Design notes (before the user's go-ahead)

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
Answered 2026-11-10: the user chose "Run it" (Definition 1, reference (i)).

## Not run
Anything on a GPU yet. Predictions are filed by the code writer before the job.

## Changes made while writing the code (2026-11-10, code writer, commit c2211fcd)
- **Positive control.** The planned change at `layers.27.mlp.down_proj` does not leave the gates fixed. Γ also reads
  29 residual taps, and tap 28 is that layer's output. The control is now the final RMS norm weight × 2. That change
  sits after every tap, so the gates must match bitwise and r = 0. It only shows that r reads 0 when the gates cannot
  move. Nothing tests the high end.
- **E40 check value.** The KL check for E40 uses 0.618377 (`results/controls.json` `kl_e40_16`). The 0.532144 named
  above is the Gaussian change's KL.
- **Rows.** The rows are 64 per row set, of 512 tokens each, as E45 built them. Only e40 and gold_s0 share rows, so
  only their difference is truly paired. The card's n of 128 and its spread were measured on E132's maths rows.
