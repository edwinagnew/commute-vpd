# Analysis: the transport test, first round

*2026-10-14, coding agent. Scores `experiments/transport_test/results/predictions.json` (filed 12:49, before any GPU
run) against the results in `experiments/transport_test/results/`. Figures in `experiments/transport_test/figures/`.
Deviations from the filed set-up, each made before the measurement it touches: the adversarial metric's scope, one
adversary per 128 rows as in the paper rather than per 8 (`2026-10-14_decision_pgd-scope.md`, which also records a
failed edit that delayed the fix by one pod), and the window scan size (`2026-10-14_decision_window-scan-size.md`).*

> **Review 2026-10-14:** a code audit found no bug that changes a scored verdict, but corrects two readings below (the arm-B fits failed to optimise rather than overfit; arm A's span fraction relative to random grows to 10× by step 300) and five numbers. See `2026-10-14_review_round1-audit.md`.

## Headline

Two of the three scored predictions reversed. In parameter space the fine-tune updates are **not** in the span of the
existing rank-1 subcomponents: the Frobenius fraction of δW on the ~9 900 alive subcomponents is 0.06% for the emoticon
fine-tune and 0.05% for continued pretraining, against 0.035% for a random direction of the same norm (ratios 1.7 and
1.5, not the predicted 10). In function, the emoticon behaviour is easy to mimic by editing a few existing
subcomponents, but ten *random* alive subcomponents with free write vectors do it as well as the ten the projection
picks (capture 0.987 against 1.000), so the test as filed has no power at k = 10. The one place the projection does
carry information is where the paper said it would: among the 358 alive subcomponents of `h.2.mlp.down_proj`, the
paper's emoticon subcomponent `2359` ranks second by Frobenius contribution at step 10, with three of its five siblings
in the top eight. The decomposition stays faithful to the fine-tuned model under every transport by the stochastic
metric (ratios 0.94 to 1.02), and under the adversarial metric except for a block-driven excursion at the smallest
emoticon checkpoint.

## What was run

Released 4-layer 67M Pile model and its VPD decomposition, loaded with `param-decomp` at `74146b5` (code identical in
the adversary path to the paper's commit `85c6b70`). One L40S, 114 minutes, about $2.1; two earlier pods that stopped
at the calibration gate cost $0.22.

- **Calibration on W0** (prediction 4, not scored): alive 3625 / 844 / 1940 / 3472 per layer (paper 3709 / 848 /
  1943 / 3472); stochastic KL 0.248 (paper 0.238); adversarial KL 0.70, mean of two 128-row blocks 0.68 and 0.72
  (paper 0.927; the paper's own run logged 0.59 to 2.24 across 100 random batches); CI-masked 0.348 and unmasked
  0.012 (vpd-audit 0.342 and 0.0115). Passed. The pre-committed rule for scoring the adversarial half of prediction 3
  holds: blocks within 2×, top 1% of positions carry 8% and 18% of the mean.
- **Emoticon windows**: 438 positions with g(`2359`) > 0.5 in 32 768 Pile train rows (one per 38 000 tokens); 365
  training, 73 held-out. Token at the position ` :` 295, ` ;` 61, ` =` 36, ` :-` 32; next token `D` 71, `P` 66, `/` 36,
  `-)` 27.
- **Arm A** (emoticon → `o`, lr 3e-5): held-out P(o) 0.9993 at step 10 already (0.997, 0.999, 1.000 at 30, 100,
  300); max per-matrix ‖δW‖/‖W‖ 0.011 at step 10, 0.026 at 300; KL to W0 on the other 40 window positions 0.04 to
  0.05. Scored checkpoint: step 10 by the filed rule. The step sweep did not span the transition; steps 1 to 3 would
  have.
- **Arm B** (continued pretraining, lr 1e-5): held-out CE 2.720 → 2.724, KL to W0 0.010 at step 10 to 0.019 at
  300; max ‖δW‖/‖W‖ 0.004 to 0.017. Scored checkpoint: step 100, KL to W0 0.014 nats (above the 0.005 conditioning
  floor).

## Scoring

| # | Prediction | Measure, scored checkpoint | Result | Verdict |
|---|---|---|---|---|
| 1a | Emoticon capture: T1 write k=10 capture ≥ 0.5 and ≥ 0.3 above random | top-10 0.9998, random 0.987 (0.978 to 0.998 over 5 seeds); margin 0.013 | capture passes, margin is in the **null band** (< 0.1) | **Reversed** |
| 1b | Location: `2359` or a sibling in the top 5 of `h.2.mlp.down_proj` (not scored) | ranks 2359 → 2, 3382 → 5, 1672 → 6, 3327 → 8, 2623 → 54, 3290 → 63 | as predicted | held |
| 2 | Pretraining arm: Frobenius fraction ≥ 10× random and T0 functional capture ≥ 0.5 | 0.00053 vs 0.00035 (1.5×); capture −0.09 (every transport worse than W0 on held-out rows) | both halves fail | **Reversed** |
| 3 | Not brittle: stochastic and adversarial ratios ≤ 2 for T1 write k=10 (A) and T0 functional (B) | A: 1.008 / 1.433; B: 1.004 / 1.049 | pass | held |
| 4 | Calibration within 1.5× of the paper | 1.04 / 0.76 | pass | held |

Conditioning for 1a was met: KL(f_ft ‖ f_W0) at the emoticon position 5.03 nats, P(o) rose from about 0 to 0.999.

## Reading, measure by measure

**Parameter space (measure 1).** δW is essentially orthogonal to the dictionary. Including all 38 912 subcomponents
(dead ones too) doubles the fraction to 0.12%, still a tenth of a percent. Per matrix the fraction never exceeds
0.13%. The Frobenius α fit, used as a transport, captures 0.1% of the function. So the literal projection, the
operation the VPD paper proposed for reading gradients, explains neither the parameters nor the behaviour of these
updates. This holds for the strengthen-only arm as much as the emoticon arm, and at every checkpoint.

**Function, emoticon arm (measure 2).** Rescaling every alive subcomponent (T0 functional, 9 881 free scalars)
reproduces the emoticon behaviour (capture 0.987, P(o) 0.94) and 30% of the off-target change. Free write vectors on
the top-k by Frobenius contribution: k = 1 captures 0.05, k = 3 0.83, k = 10 1.00. Random alive subcomponents with the
same per-matrix counts: 0.06, 0.68 (0.59 to 0.83), 0.99. Read-vector edits need k = 10 to work (0.99) and fail at
k = 3 (0.12). The k = 10 null succeeds because the behaviour is a single logit push at positions that many
subcomponents' read directions already see; a rank-10 write with ten random fixed read directions can realise it.
The test separates top from random only at k = 3, by 0.16, with the random spread covering the top value. At
k ≥ 30 the top-k write edits reproduce the emoticon position but wreck the surroundings (off-target capture −0.6 at
30, −9.3 at 100): overfitting 365 windows.

The k = 1 result is informative on its own: a single write-vector edit on the subcomponent with the largest
Frobenius contribution (`h.2.attn.k_proj:206`) does nothing, and the top-10 set contains none of the emoticon family
(it spans eight matrices, mostly layer 0). The paper's manual edit of `2359`'s write vector does produce the behaviour; a
k = 1 fitted write edit on `2359` is the first measurement of round 2, not a result of this one. The Frobenius ranking across matrices is the wrong selector: it is dominated by matrices where δW is large
(the MLP input projections carry 38% of ‖δW‖²) rather than where the function changes.

**Function, pretraining arm (measure 2).** Every transport fitted on 512 training rows makes the held-out model
further from the fine-tuned one than W0 is (capture −0.09 for T0 functional, −0.16 to −32 for k-sparse edits, the
random sets likewise). The fine-tune's held-out effect is 0.014 nats. Whether the fits overfit the training rows or
the α parametrisation cannot represent this update is not settled by what was recorded: training-set capture was
not measured. That measurement, and a fine-tune with a held-out effect of order 0.1 nats, settle it. By the filed
conditioning rule this is a fail, not an ill-conditioned run. The Frobenius side (ratio 1.5) does not depend on the
fit and fails cleanly.

**Faithfulness at W0 + δW (measure 3).** Stochastic ratios are 0.94 to 1.02 for every variant and checkpoint in both
arms. Adversarial ratios for arm B are 1.05 to 1.18. For arm A at step 10 the null transport gives 2.43 and T0
2.32, both from block 1 (2.6 against 0.8 for block 0; on W0 the blocks are 0.68 and 0.72); T1 write k=10 gives 1.43;
at steps 30, 100 and 300 the null transport gives 1.02, 1.14 and 1.27 with larger δW. The brittleness branch of `GOALS.md` is not
supported: no variant loses faithfulness by the stochastic measure, and the adversarial excursion is one block at one
checkpoint. One more number belongs with measure 1 rather than here: the unmasked KL with Δ excluded rises from 0.012
on W0 to 0.034 to 0.063 on the fine-tuned models under the null transport. With Δ excluded that quantity is close to
the size of the fine-tune's own effect, and with Δ included it is zero by construction; it says δW went into Δ,
which is the span result again, not a faithfulness loss.

**Location (measure 4).** At step 10 the emoticon family is where the paper's editing section put it: `2359` second
in its matrix, with a negative α (δW shrinks `2359`'s direction while adding the `o` push elsewhere), `3382`, `1672`
and `3327` in the top eight. By step 30 the family has dispersed (`2359` rank 354, `3327` rank 2) and by 100 to 300 it
is in the twenties to forties. The projection sees the right mechanism at the first checkpoint and loses it as the
update grows, even though the behaviour is saturated from step 10 on.

## What this changes

- The literal projection onto the subcomponent dictionary is a dead end for these updates: δW lives in the
  residual, not the span. Any continuous-interpretability story has to be functional.
- The functional test needs a null that cannot succeed by expressivity alone. Free write vectors on k = 10 random
  subcomponents already span a rank-10 update of the matrices they touch; for a behaviour that is one logit push, that
  is enough. Candidates: scale-only k-sparse edits (one scalar per subcomponent), write edits constrained to the span
  of the subcomponent's existing write direction and the `o` unembedding, or scoring selectivity on generic text (KL
  of f_T to f_W0 on Pile rows outside the windows), which the window-only off-target measure does not see.
- The selector should be functional (greedy by gain or by causal importance on the fine-tune data), not Frobenius
  contribution; the k = 1 and 1b results together say the signal is there in `h.2.mlp.down_proj` and the selector
  missed it.
- Arm A needs checkpoints at steps 1 to 5; arm B needs a learning rate that gives a held-out effect of order 0.1
  nats before its functional half means anything.
- Record training-set capture beside held-out capture, so overfitting and under-parametrisation can be told apart.

None of these are the re-decomposition test or the toy-model calibration in `backlog.md`; they are the next round of
the same experiment with a better null and selector, and they go through `/design` first.

## Trace

`results/calibration.json`, `windows.json`, `finetune_{A,B}.json`, `transport_{A,B}.json`, `faithfulness_{A,B}.json`;
δW checkpoints, fitted α, the k=10 sets and edits, and all logs in `results/large/` (866 MB, gitignored, copies to be
kept on cluster NFS). Code at commit `ea4df42` plus this report.
