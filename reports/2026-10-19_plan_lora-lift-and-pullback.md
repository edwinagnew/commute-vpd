# Plan: a LoRA through the decomposition, both ways. Lift the finished LoRA onto the subcomponents, and train the subcomponents with the LoRA's shape

*2026-10-19, coding agent. Design check in chat, confirmed by the user ("yes", then "chuck in pullback-on-lora as well";
"starting at step 10 is fine. maybe also just decomposing the final AB from lora"). Note: the backlog entry of
2026-10-15 defined "LoRA through the pull-back" as training the subcomponents with a LoRA-shaped parameterisation; the
user's intended question (2026-10-19) is the plain one, fine-tune with a LoRA and apply the same maps to its weight
change. This plan does both; the lift arms are the user's question.
To be run by a fresh session: write `experiments/lora_commute/run.py` with the predictions below as its `PREDICTIONS`
dict, file them with `--predictions`, smoke, then run on one 4090 (RunPod via remote-kernels, pod
`commute-lora_commute-1`; the cluster queue had 93 pending single-GPU jobs on 2026-10-19). Under $1. Follow the
2026-10-19 rules in `.claude/CLAUDE.md`: one frozen config dataclass for full and smoke sizes, Codex adversarial review
of the diff before the paid run, download results as each stage writes them.*

**Terms.** *LoRA*: the fine-tune's weight change is M = BA on one matrix, rank r. *Lift*: write M as a change to that
matrix's existing alive subcomponents. The part of M the subcomponents can express (`lift.obstruction` gives the
rest) becomes the minimum-norm (dU, dV) with dU Vᵀ + U dVᵀ = M − O(M) (`lift.euclidean_lift`); O(M) − dU dVᵀ goes to Δ.
*Pull-back*: train the subcomponents' (U, V) on the fine-tune loss with Γ and Δ fixed. *Route split*
(`local_faith.route_split`): the behaviour change at the trigger, in held-out log-odds of `o`, split into the share
carried by the changed subcomponents with W₀'s CI values (d), the share carried by changed CI values on the unchanged
subcomponents (r), their interaction (i), the share in subcomponents Γ masks off (e_gate) and the share not in the
subcomponents at all (e_Δ). They sum to 1.

## Why

The user's question in its simplest form: fine-tune as a LoRA, then commute that LoRA past the decomposition. The lift
test (TRIED row 2) lifted the full-weight fine-tune (carried the update, broke under adversarial masks) and trained
LoRAs but never lifted them. The recruit test (row 5) put a rank-8 LoRA in Δ: the components carry none of it, Γ's
changed CI values on unchanged subcomponents carry 0.54. The pull-back has only ever been trained on all 24 matrices
(rows 2 to 4, ladder), where it routes like a masked-off update (d ≤ 0.08). No arm has yet had the LoRA's support and
rank. A rank-1 LoRA on the paper's site is the closest an update gets to "one subcomponent's worth".

## Arms

All on `h.2.mlp.down_proj` (the paper's LoRA site), emoticon windows (round 1's; 73 held-out), Γ and Δ as stated,
ranks r ∈ {1, 4, 8, 32}.

- **LoRA_r**: rank-r LoRA, Adam lr 1e-3, arm A's loss, batch and sampling, 100 steps with checkpoints at 10 and 100
  (round 2's recipe; rank 8 gives P(o) 0.999 at step 10 and 0.996 at step 100 with 40× lower KL elsewhere). Measured
  as placed in Δ (components unchanged) at both checkpoints: the reference for every rank. Report the cosine of each
  LoRA's read factor A with its random initialisation at both steps (round 2: 0.92 at step 10, 0.67 at step 100 for
  rank 1), so a poor lift at step 10 can be told apart from a random read side.
- **lift_r**: LoRA_r's M lifted onto the alive subcomponents of that matrix; O(M) − dU dVᵀ into Δ. No training. Done
  at **step 10 (scored) and at the final step 100 (the finished LoRA; reported with the same thresholds)**.
- **pb_r**: pull-back on that matrix only, with dU = P Sᵀ, dV = Q Tᵀ (P ∈ ℝ^{d_out×r}, Q ∈ ℝ^{d_in×r}, S, T ∈ ℝ^{n×r},
  n = alive count), so the change has the LoRA's support and rank. Adam, 30 steps, lr from {1e-5, 3e-5, 1e-4, 3e-4,
  1e-3} by amendment 1's rule (held-out P(o) ≥ 0.9 with KL elsewhere ≤ 0.110; the smallest lr that passes). Δ frozen.
- **pb_full**: the same with unconstrained (dU, dV) on all alive subcomponents of that matrix: the capacity ceiling.
- **Seeds**: pb_8 and LoRA_8 with optimiser seed 1 as well, for the noise floor of every number below. Every lift
  of a seed-1 LoRA is free, so lift_8 gets a floor too.
- **Control**: W₀ with three adversary seeds for the PGD floor (as in the ladder's noise stage).

## Numbers (every arm, and what each should read)

| Number | W₀ | LoRA_r in Δ (rank 8: measured) | lift_r | pb_r |
|---|---|---|---|---|
| held-out P(o), KL elsewhere | 0.013, 0 | 0.999, 0.154 | within 0.05 of LoRA_r's P(o) | ≥ 0.9, ≤ 0.110 by the lr rule |
| inexpressible share of M, input-weighted (`lift.weighted_fractions`) | 0 | 1 by definition | small at r = 1, growing with r | 0 (Δ frozen) |
| d, r, e_Δ from the route split (gain ≥ 1 nat) | undefined | 0, 0.54, 1 | d ≥ 0.5 at r = 1, falling with r; e_Δ ≈ inexpressible share | d ≥ 0.3 at every r; e_Δ ≈ 0 |
| task-local PGD ratio (`local_faith`, pgd_trigger, vs the 3-seed W₀ floor) | 1 ± floor | reported | breaks (> 1.5) at r ≥ 8, near the floor at r = 1 | near the floor (round 2's pull-back: 1.03 to 1.07) |
| subcomponents moved by > 1% of ‖u vᵀ‖ | 0 | 0 | tens at r = 1, most of the matrix at r = 32 | reported |
| Jaccard of the switched-on set (`recruit`, Δg > 0.3) with the recruit test's ft_s0 | – | 0.82 (rank 8) | reported | reported |

Pile faithfulness ratios: one check at rank 8 for each of lift and pull-back, reported. Everything else above is used
by a prediction or a decision; nothing else is reported.

## Predictions (thresholds fixed here; file them verbatim)

- **G0, floor.** Seed-to-seed |Δd| and |Δr| for pb_8 and lift_8 (from LoRA_8's two seeds) set the margin m =
  max(0.1, 3 × spread). PGD floor = 3-seed spread at W₀.
- **G1, arms learn.** Every LoRA_r reaches P(o) ≥ 0.9; every pb_r finds an lr by the rule. An arm that does not is
  reported, not scored, and its failure is itself a result (see "What changes next").
- **G2, lift is exact.** For every lift_r, the lifted model's held-out P(o) is within 0.05 of LoRA_r's, and the
  route-split shares sum to 1 within 1e-6. A miss means the lift or the Δ bookkeeping is wrong; stop.
- **P1, the smallest LoRA commutes.** lift_1 at step 10: d ≥ 0.5 and e_Δ ≤ 0.2. Null: d < 0.2. Predicted: pass.
  The step-100 lift is reported against the same thresholds; if it differs from step 10 by more than m on d, say so
  beside A's cosine with its initialisation.
- **P2, it stops with rank.** d(lift_r) decreases with r by more than m between r = 1 and r = 32, and the inexpressible
  share increases. Null: flat within m. Predicted: pass.
- **P3, the lift breaks adversarially, as the full-weight lift did.** lift_8 and lift_32: pgd_trigger ratio ≥ 1.5. lift_1
  within 3 floors of W₀. Null: all four within 3 floors. Predicted: pass (the fragility is the round-2 open question;
  rank 1 is the test of whether it is about size).
- **P4, the inside route carries it under W₀'s CI values when confined to the matrix.** pb_r: d ≥ 0.3 at every rank
  that learns. Null: d < 0.1 (the ladder's all-matrix result). Predicted: pass. This is the prediction most likely to
  fail: the concentration test's siblings-only arm (6 subcomponents of this matrix) carried about 0.4 under W₀'s CI
  values, the all-matrix pull-back 0.03.
- **P5, same matrix, same subcomponents.** lift_8 and pb_8 move overlapping subcomponents: of the 20 largest changes
  by ‖Δ(u vᵀ)‖ in each, at least 8 are shared. Null: ≤ 3. Predicted: null. The lift puts the update where M's
  singular vectors fall; the pull-back puts it where the gradient falls. They need not agree.

**Blind to.** d, r, e_Δ: everything away from the trigger (cover: KL elsewhere and the Pile check). The PGD ratio:
anything an adversary with 20 steps does not find (cover: the W₀ floor and round 2's calibration). Subcomponents moved
at the 1% cut: changes spread below 1% (the identity test's regime). Positive controls: G2 for the lift (it must
reproduce the LoRA's behaviour exactly), the lr rule for the pull-back (it must learn), the W₀ floor for PGD.

## What changes next

- P1 passes and lift_1's PGD stays at the floor: the smallest canonical update commutes past the decomposition on its
  own matrix. The rank at which P2 and P3 turn is "the limit" in GOALS, measured. Next: other tasks on the same site.
- P1 passes but P3 fails at rank 1 too: the lift carries behaviour but not faithfulness at any size; the cause of the
  round-2 fragility becomes the next question before any more maps are tried.
- P1 fails: a rank-1 update on the paper's own site does not sit in its subcomponents under W₀'s CI values. The weak
  version fails for the smallest LoRA; the brittleness branch of GOALS is live.
- P4 passes while the ladder's d ≤ 0.08 stands: confining the pull-back to the update's matrix is what makes the
  subcomponents carry it; the all-matrix pull-back was spreading the update into places Γ never reads at the trigger.
  P4 fails: even confined, the inside route runs through changed CI values, and Γ updating becomes round 4.

## Not in this round

Other matrices or several matrices at once; other tasks (the ladder's); LoRA lifted with a refit Γ; the switched-off
set; the natural-`o` question from the recruit test.

**Hand-off.** Read this plan (the lift arms are the user's question: "fine-tune with a LoRA, apply the same maps";
the pull-back arms are the inside route for comparison, kept at the user's request), `reports/2026-10-19_analysis_recruit-test.md` (what the LoRA-in-Δ reference looks like)
and `reports/2026-10-14_analysis_lift-test.md` §P2′ (how the full-weight lift broke). Code to reuse unchanged:
`finetune.finetune_lora_arm_a`, `lift.basis / euclidean_lift / obstruction / weighted_fractions`,
`local_faith.route_cells / route_split / local_faithfulness`, `recruit.trigger_dg / switched / jaccard`,
`pullback.train` (needs a rank-r `Offsets` variant for pb_r; keep it to one class with a `rank: int | None` field,
and a CPU test that rank r reproduces the unconstrained offsets when r ≥ n). The round-1 window and alive files are
gitignored and must be copied to the pod by hand (sha256 prefixes: alive.pt 49746222ec4c, windows.pt c19a880cdb3c).

## Addendum (2026-10-19, the implementing session; the body above is left as confirmed)

The user restated the question when handing this plan over: *is it easier, or does it work better, to decompose a
LoRA update than a normal fine-tuning update* (instinct: yes), tested on the pull-back, the lift, and whatever else has
worked. Three things change in the implementation (`experiments/lora_commute/run.py`), none in the arms above.

1. **Full-weight comparator arms.** The plan compares LoRA ranks with each other; the user's question compares a LoRA
   with a normal fine-tune under the same maps. Added: `ft` (arm A, Adam lr 3e-5, all 24 matrices, seed 0, steps 10
   and 100) measured do-nothing, lifted on all 24 matrices (round 2's lift) and as the all-matrix pull-back `pb_all`
   (lr 1e-5, the ladder's arm). Each has a prior number to reproduce, used as gates (G4): round 2's Pile PGD ratios
   at step 10 (do-nothing 2.18, lift 5.41), the ladder's emoticon pull-back shares at step 10 (d 0.025, r 0.53), and
   the recruit test's LoRA_8 do-nothing r (0.54). Three comparison predictions are filed with the plan's margin m:
   C1 d(lift_8) − d(lift_ft) ≥ m; C2 Pile PGD ratio lift_8 ≤ 1.5 while lift_ft > 1.5; C3 d(pb_full) − d(pb_all) ≥ m.
   All three are predicted to pass (the user's instinct), C2 with the least confidence (the cause of the round-2
   fragility is unknown).
2. **P3's adversary.** The plan files P3 on the task-local `pgd_trigger` ratio. `reports/2026-10-19_review_pgd-local.md`
   (same day) shows both window adversaries saturate at W₀ (33–53 nats at the trigger, sign set by the seed), so a ratio
   of them reads nothing. P3 is filed on round 2's Pile PGD ratio instead (2 × 128 val rows, the harness that found the
   full-weight lift's 1.14 → 5.4), with the 3-seed W₀ spread as the floor. Blind to a trigger-specific attack; the
   task-local numbers stay reported. This widens the Pile harness from the plan's two checks to every lift_r and pb_r,
   pb_full, pb_all, the two do-nothing references and W₀ × 3 (about 16 calls); the first W₀ call is timed before the rest.
3. **Scored step of LoRA_r.** Round 2's rank-1 LoRA reads P(o) 0.22 at step 10 and 0.99 at step 30, so "step 10 (scored)"
   would leave P1's arm unscored. Rule fixed before filing: the scored step is the first checkpoint of {10, 30, 100}
   with held-out P(o) ≥ 0.9; the step-100 state is the finished LoRA, reported against the same thresholds.

Also fixed before filing: the pull-back's lr rule is applied at step 30 (P(o) ≥ 0.9 and kl_other ≤ 0.110, smallest
passing lr) and pb arms are scored at step 30, with step 10 reported; "subcomponents moved" uses `identity.change_stats`
(relative change of u vᵀ > 1%); G2 is the bookkeeping identity (the lifted state's target model has the LoRA's P(o)
to 1e-3, shares sum to 1) with the components-alone P(o) reported beside it (round 2's liftable part: 0.954 vs 0.999);
the rank-r pull-back's weight change has rank ≤ 2r by construction (dU V_Aᵀ + U_A dVᵀ), reported as the realised rank.
Expected value for the inexpressible share, from round 2: 0.40 (r1 step 30), 0.41 (r8 step 10), 0.30 (r8 step 100)
against a null of 0.42 on this matrix, because the LoRA's read factor A stays near its random init; a large share at
every rank is then "as round 2 found", not a surprise.
