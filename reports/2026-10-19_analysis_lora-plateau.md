# LoRA to its plateau: the read side stays random for the whole run, the lifted state stays faithful on Pile at every step, and at the plateau a third of the behaviour sits in Δ

We trained round 5's LoRA ten times longer, to a plateau at 1000 steps, and at nine checkpoints measured how much of its
update the released subcomponents could express and how far its read factor had moved from its random start, with the
lifted and do-nothing states put through the Pile adversary at steps 10, 100 and 1000. The read factor moved ten percent by
the plateau and the update stayed as inexpressible as a random one the whole way, while the update's energy moved off the
fully covered matrices onto the two layer-3 attention matrices the decomposition reads worst, so that by the plateau three
fifths of the update over all matrices was inexpressible; the lifted state nevertheless passed the Pile checks at the W0
floor at every step, including the fine-tune's matched step where the fine-tune's lift had failed fivefold, but at the
plateau the lift left a third of the trigger behaviour and over half the update's energy in Δ. For the goal this says a
LoRA's lifted decomposition passes the faithfulness half of the weak version along its whole training and fails the
other half more clearly the longer it trains, and that training longer cannot fix it, because the read side never becomes
something the subcomponents can read.

*2026-10-19, coding agent. Design check confirmed by the user 2026-10-19 (learning rate as in round 5, Pile folded in, seed 0
only). Predictions `experiments/lora_plateau/results/predictions.json` (filed before training, unchanged); results
`results/lora_plateau.json`; code `experiments/lora_plateau/run.py` on the shared `src/commute/harness.py`; figures
`experiments/lora_plateau/figures/`; notebook `notebooks/lora_plateau.ipynb` (figures and every number, executed). Codex
adversarial review of the new code before the run: approved, no findings. Smoke on a RunPod A5000 (251 s, $0.05); scored run
on a RunPod L40S (1872 s, $0.70) because the cluster's 8 GPUs were held by two `normal` jobs with 10 and 19 hours left
(cluster jobs 22429 and 22665 submitted and cancelled after the policy's wait).*

**Terms.** *Lift*: the finished weight change written as the minimum-norm change (dU, dV) to each matrix's alive
subcomponents; what no first-order change can express, O(δW) − dU dVᵀ, goes to Δ. *Do-nothing*: δW placed in Δ,
subcomponents untouched. *Inexpressible share*: the share of the update's input-weighted output energy that no
first-order change of the alive subcomponents can express; its *isotropic null* is that share for a random update of the
same matrix; a *saturated* matrix is one where the alive subcomponents span every direction, so the share is 0 whatever
the update (four of the 24). *Pooled* = energy-weighted over the 20 non-saturated matrices; *ratio* = pooled share /
pooled null, the tracked number, since the null moves with where the energy sits. *In alive*: the share of ‖δW‖²_F whose
read side lies in the alive read span, pooled the same way; a random read side gives r_v / d_in per matrix. *|cos|*: mean
cosine of A's rows with their initialisation. *Route split*: the held-out log-odds gain of `o` at the trigger split into d
(changed subcomponents at W₀'s CI values), r (changed CI values on the unchanged subcomponents), i, e_gate and e_Δ (not
in the subcomponents at all). *Pile ratios*: round 2's harness (2 × 128 Pile val rows; stochastic, CI-masked, PGD KL), each
÷ W₀'s under the same adversary seed. *Plateau ratio*: mean training loss over steps 301–1000 ÷ mean over steps 101–300.

## Per checkpoint (held fixed: windows, Γ, the released decomposition, the recipe; varied: the step)

| step | P(o) | kl_other | ‖δW‖/‖W₀‖ max | ratio | pooled share (null) | in alive (random) | \|cos\| | energy on the 4 saturated | inexpressible over all 24 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.234 | 0.009 | 0.012 | 0.93 | 0.52 (0.56) | 0.270 (0.264) | 1.000 | 55% | 0.24 |
| 2 | 0.961 | 0.059 | 0.021 | 0.86 | 0.48 (0.55) | 0.273 (0.267) | 1.000 | 60% | 0.19 |
| 3 | 0.998 | 0.084 | 0.026 | 0.85 | 0.48 (0.56) | 0.268 (0.262) | 0.999 | 61% | 0.19 |
| 5 | 0.992 | 0.028 | 0.033 | 0.91 | 0.53 (0.58) | 0.261 (0.256) | 0.998 | 58% | 0.22 |
| 10 | 0.999 | 0.021 | 0.044 | 1.02 | 0.57 (0.57) | 0.264 (0.260) | 0.995 | 56% | 0.25 |
| 30 | 0.997 | 0.017 | 0.058 | 0.99 | 0.55 (0.56) | 0.275 (0.270) | 0.988 | 51% | 0.27 |
| 100 | 1.000 | 0.012 | 0.057 | 0.82 | 0.55 (0.67) | 0.212 (0.164) | 0.973 | 20% | 0.44 |
| 300 | 1.000 | 0.005 | 0.058 | 0.92 | 0.68 (0.74) | 0.162 (0.122) | 0.959 | 13% | 0.59 |
| **1000** | 1.000 | 0.005 | 0.090 | **0.93** | 0.71 (0.76) | 0.127 (0.102) | **0.902** | 13% | **0.62** |

Steps 2 and 100 reproduce round 5 to three digits (ratio 0.865 and 0.819, |cos| 1.000 and 0.973). The training loss is
0.0018 at step 1000; plateau ratio 1.02 (mean loss 0.0066 over steps 301–1000 against 0.0065 over 101–300). Where the
energy sits: to step 30 about half of it is on `h.3.mlp.c_fc` (saturated); from step 100 it moves to `h.3.attn.k_proj` and
then `h.3.attn.q_proj`, which carry 22% and 46% at the plateau with isotropic nulls 0.77 and 0.84. On `k_proj` A has become
readable (in alive 0.154 against a random 0.078; share 0.46 against a null of 0.77); on `q_proj` it has not (0.061 against
0.047; 0.83 against 0.84), and `q_proj` has twice the energy.

## Pile at the step-matched points (L40S; W₀ PGD floor f = 0.046 over three adversary seeds, so "within 3 floors" is ≤ 1.14)

| step | arm | gain (nats) | d | r | e_Δ | ‖Δ′−Δ₀‖²/‖δW‖² | Pile PGD ratio | stochastic | CI-masked |
|---|---|---|---|---|---|---|---|---|---|
| 10 | do-nothing | 14.5 | 0 | 0.58 | 1 | 1 | 1.17 | 1.009 | 1.008 |
| 10 | **lift** | 14.5 | 0.016 | 0.58 | 0.19 | 0.41 | **1.26** | 1.009 | 1.008 |
| 100 | do-nothing | 14.3 | 0 | 0.56 | 1 | 1 | 1.09 | 0.996 | 0.999 |
| 100 | **lift** | 14.3 | 0.013 | 0.56 | 0.11 | 0.46 | **1.22** | 0.996 | 1.000 |
| 1000 | do-nothing | 15.3 | 0 | 0.54 | 1 | 1 | 1.04 | 1.002 | 1.003 |
| 1000 | **lift** | 15.3 | 0.006 | 0.54 | **0.31** | **0.54** | **1.07** | 1.002 | 1.004 |

Reference (round 5, L40, same harness): the fine-tune's lift at step 10 reads 5.41 and its do-nothing 2.18, with e_Δ 0.09
and 0.36 of ‖δW‖² in Δ; the LoRA's lift at step 2 read 1.13.

## Gates

- **G1.** Learned at step 2 (P(o) 0.961), as in round 5.
- **G2.** Holds for every lift (P(o) difference ≤ 5e-6, shares sum to 1 to 1e-16).
- **G3.** Plateau reached: plateau ratio 1.02, P(o) 0.9997 and 0.9999 at the last two checkpoints.
- **GPU.** The W₀ PGD floor on the L40S is 4.6% over three adversary seeds, against 21% on the L40 in round 5 and 3.5% on the
  4090 in round 4. The 20-step adversary's seed spread differs by card, so the fine-tune's 5.41 reference, measured on the L40
  and not rerun here, is read only against the filed lines (≤ 2, ≥ 5), which sit far more than round 4's 35% miss apart.

## Scores as filed, with the reading each bears on

| id | filed | result | bears on | numbers |
|---|---|---|---|---|
| P1 read side stays random (ratio ≥ 0.8 at step 1000; null ≤ 0.6) | pass | **pass** | A (inexpressible) | ratio 0.93; \|cos\| 0.90; in alive 0.127 against random 0.102 |
| P2 lift survives at the matched step 10 (PGD ≤ 2 and e_Δ ≤ 0.2; null ≥ 5) | pass | **pass** | A (faithfulness), reopens round 4's C2 | lift 1.26, do-nothing 1.17, e_Δ 0.19; fine-tune's lift 5.41 (round 5) |
| P3 lift survives at the plateau (PGD ≤ 2 and e_Δ ≤ 0.2; null ≥ 5) | pass, weakly | **neither** | A (both halves) | lift 1.07 at the floor, do-nothing 1.04, but e_Δ 0.31 and Δ takes 0.54 of ‖δW‖² |

**Miss to flag.** P3 was filed as a pass on both halves. The Pile half came out as predicted and stronger (1.07, inside one
floor), but e_Δ, which had fallen from 0.22 at step 2 to 0.11 at step 100, rose to 0.31 at the plateau, so the filed pass
is missed on the Δ half. The direction of the Pile prediction was right; the assumption that e_Δ would stay near its step-100
value was wrong, and the cause is not established by this run (the energy moved to `h.3.attn.q_proj`, whose share is 83%
inexpressible, in the same interval; that is a correlation in one run, not a cause).

## What this says and does not say

- **Says (the read side; the notebook's candidate 3 is dropped).** A moves 10% in 1000 steps at lr 1e-3 and the update stays
  as inexpressible as an isotropic one relative to its null at every checkpoint (ratio 0.82–1.02). Training longer does not
  make the LoRA readable; the one matrix where A did become data-shaped (`h.3.attn.k_proj`) carries a fifth of the energy and
  the matrix carrying half (`h.3.attn.q_proj`) did not. "Let A become data-shaped by more steps" is closed. A higher learning
  rate on A alone was not run and is a separate decision.
- **Says (reading A, faithfulness half).** The lifted LoRA decomposition passes VPD's Pile checks within one to three W₀
  floors at steps 10, 100 and 1000, where in round 5 the fine-tune's lift at step 10 read 5.41 on the L40. Round 5's C2m
  reversal was not an artefact of scoring the LoRA at step 2: at the fine-tune's own step the LoRA's lift reads 1.26 with a
  relative update four times the fine-tune's. The do-nothing placements also sit at the floor (1.17, 1.09, 1.04) where the
  fine-tune's read 2.18: the LoRA update itself disturbs W₀'s decomposition under the adversary less than the fine-tune's
  does, at every size.
- **Says (reading A, Δ half).** The lift's share of the update left in Δ grows with training (0.41, 0.46, 0.54 of ‖δW‖²) and
  the behaviour it leaves there is 0.19, 0.11, then 0.31 of the trigger gain. At the plateau the second half of the weak
  version is failed more clearly than at any earlier step, with 62% of the update's energy over all 24 matrices
  inexpressible. The faithful Pile numbers and the large Δ are the same state: a decomposition that passes the adversary while
  a third of the trained behaviour sits in its residual.
- **Does not say that the fine-tune's lift would fail on this card.** The 5.41 reference is from the L40; the L40S's W₀ floor
  is four times tighter, and no fine-tune arm ran here. The filed lines are far enough apart that a 35% GPU miss cannot move
  the reading, but the like-for-like comparison on one card is not in this run.
- **Does not say why e_Δ rose.** One seed, one run; the rise coincides with the energy moving to `q_proj`, nothing more.
- **Says (mechanism).** d is 0.006–0.016 and r 0.54–0.58 at every Pile step, as in rounds 3–5. The direct route carries
  nothing even at the plateau.
- **Says (noise).** The W₀ PGD floor is GPU-dependent (3.5% on a 4090, 4.6% on an L40S, 21% on an L40, same code and seeds).
  "Within 3 floors" means a different line on each card, and PGD ratios should be compared only within a card.

## Not run

- The fine-tune arm on the L40S (its lift and do-nothing at step 10), the like-for-like comparison with P2 on one card.
- A second LoRA seed; the seed spread of e_Δ at the plateau, which decides whether the 0.11 → 0.31 rise is real.
- A higher learning rate on A only (the other half of the notebook's candidate 3).
- The per-step lift with updated bases (review section 3.3), still waiting on its tol rule.
- VPD's sparsity loss on any lifted state; any task other than emoticon; Γ updating.

**Hand-off.** A new session should start from this report and `reports/2026-10-19_analysis_lora-matched.md`. Candidate 3
(more steps) is closed. What is open for the LoRA line: the fine-tune control on the same card as P2 (a ten-minute addition to
`experiments/lora_plateau` or a rerun of round 5 on an L40S), and the e_Δ seed spread at the plateau. Section 3.3 of
`reports/2026-10-19_review_lora-failure-and-fixes.md` (per-step lift with updated bases) needs its tol rule decided through
`/design` before it is built.
