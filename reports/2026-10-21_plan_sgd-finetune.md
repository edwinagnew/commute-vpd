# Plan: SGD fine-tune, its splits, and SGD component training beside them

Design check: items 5 and 7 of the SGD-finding review, approved by the user 2026-10-21 under the GOALS optimiser
subgoal. Held fixed: emoticon, steps, loss, Pile harness, L40. Varied: fine-tune optimiser (SGD vs Adam), and
component training vs adjoint split under SGD. Decision tree: stage `split` runs only if stage `rate` finds a passing SGD rate; if none passes, report the null with its grid's range and stop.

**Request, restated in the Terms' words.** Item 7 (`reports/2026-10-21_review_sgd-finding-influence.md`): E2's
"Adam or SGD alike" outside-share line rests on SGD fine-tunes at 0.1 to 1, all with large off-target damage. Find a
clean SGD fine-tune rate, then read its outside share as four blocks with the null, and its least-squares and adjoint
splits on the Pile harness, beside the Adam fine-tune the splits used. Item 5: SGD component training at a rate
matched to that SGD fine-tune's size of change, scored the same way next to the adjoint split of the same fine-tune.
One plain-SGD step of component training is the adjoint split of the fine-tune's step at the same point, so this is
the closest a map gets to "the adjoint split, applied as training".

**Card.** Emoticon arm A (cross-entropy on `o` at the trigger plus KL elsewhere), batch 32, clip 1, seed 0, all 24
matrices, 10 steps: the Adam fine-tune's (3e-5) first scored step in E14 and E16, and the step E4's off-target limit
(0.110, twice the Adam fine-tune's KL elsewhere) is read at. Γ and Δ frozen. Cluster L40, `debug` QoS.

## Stages (`experiments/sgd_finetune/run.py`, `cluster/sgd_finetune.sbatch`)

1. **`rate`** (about 5 min). Plain-SGD fine-tunes at 0.001, 0.003, 0.01, 0.03, 0.1. E4's rule: among rates with
   held-out P(o) ≥ 0.9 and KL elsewhere ≤ 0.110 at step 10, the lowest KL elsewhere. The selected δW is saved as
   float16 to `results/large/`. Then plain-SGD component training at 0.003 to 0.3; the matched rate is the one whose
   largest relative change per matrix (‖δW‖ / ‖W₀‖) is nearest the selected fine-tune's. Writes `results/rate.json`
   after every run.
2. **`split`** (about 23 min; 30-min limit). In this order, `results/split.json` written after each:
   - **W₀ control first:** the Pile harness at W₀ under adversary seeds 0 and 1 (`faith_score.w0_check`: seed-0 PGD
     within 10% of 0.69, CI-masked within 2% of 0.348, floor ≤ 0.25). Seeds 0 and 1 alone gave the L40's whole floor
     of 0.209 in E19's job 28108. Fails: stop.
   - **Cheap numbers (seconds):** the Adam fine-tune regenerated (gate: P(o) and KL elsewhere within 0.01 of E16's
     0.9993 and 0.0554, else stop), the SGD fine-tune loaded, SGD component training at the matched rate, Adam
     component training at 1e-5 (the cosine's reference only), the four blocks of both fine-tunes, and the cosine of
     each component-training weight change with the unscaled adjoint weight change K(δW) of its fine-tune.
   - **Pile states (about 140 s each on the L40):** SGD adjoint split, SGD component training, SGD do-nothing and
     least-squares split, Adam do-nothing and least-squares split, Adam adjoint split. Item 5 runs first so a
     timeout loses the Adam rows, which have recorded values.

What differs from the reference's end state: the Adam fine-tune is E16's recipe re-run (its gate checks it), on an L40
rather than E16's H100, so its Pile ratios are re-read under this job's W₀ instead of taken from E16. The SGD fine-tune
is loaded from float16; gate G2 checks its P(o) at the trigger against stage `rate`'s.

## Predictions (`experiments/sgd_finetune/results/predictions.json`, filed 2026-10-21)

Each states its rule, its expected value at W₀, at do-nothing and for the Adam reference, relative vs absolute,
blind-to and noise floor. In short:

- **R1** some SGD rate passes E4's rule at step 10: predicted pass at 0.01 or 0.03 (0.001 too slow; E2's 0.1 read
  KL 0.60 at step 10).
- **P1** (item 7) outside-both over null, SGD / Adam in the same job: predicted ≤ 0.8 ("lower"); Adam reads 0.22 (E16).
- **P2** (item 7) SGD least-squares split PGD ratio: predicted "fail" (≥ 1.5 (1 + f)), as Adam's 5.2 to 7.0.
- **P3** (item 7) SGD adjoint split PGD and stochastic: predicted "pass" (≤ 1.5 / (1 + f)), as Adam's 1.06.
- **P4** (item 5) SGD component training: both Pile verdicts pass and P(o) ≥ 0.9; Adam's reads 1.03 and 0.9998.
- **P5** (item 5) cosine of SGD component training's weight change with K(SGD δW) ≥ 0.9 and above Adam's.

Verdicts use `faith_score.verdict`: pass ≤ 1.5 / (1 + f), fail ≥ 1.5 (1 + f), inconclusive between (about 1.24 to
1.81 on the L40). Blocks and cosines are deterministic given the updates; there is one seed and no seed floor.

## Decision tree

- R1 fails: report "no SGD fine-tune rate in 0.001 to 0.1 learns within the limit at step 10", with which side each
  rate failed on, and stop. If the best rate is at the grid's edge and fails in that direction, the grid is extended
  once before the null is written.
- R1 passes, P1 "alike" or "higher": the TRIED E2 line stands for a clean SGD fine-tune; reword it with this run.
- P2 passes and P3 passes: the split pattern (least-squares fragile, adjoint faithful) is not Adam's.
- P4 and P5 pass: under SGD, component training and the adjoint split are the same operation to first order and both
  faithful at step 10; the "analogue, not a map" objection to E16 loses its force at this step. P5 fails with P4
  passing: the path or clipping matters and the two operations differ under SGD as well.
- Anything near the 1.5 line on this card (the inconclusive band) is reported as such; a deciding re-read needs an
  H100 or more adversary seeds.

## Not run

- Steps 100 and 1000 (E16 has them for Adam); the ladder tasks; momentum.
- An unclipped, equal-rate step-1 check of P5's first-order identity (the cosine's positive control).
- Trigger-local masked checks and the route split as scored numbers (reported only).
- A smoke run: by the 2026-10-21 rule the scored run is its own smoke; nothing has run on a GPU.
