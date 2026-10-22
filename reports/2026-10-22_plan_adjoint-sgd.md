# Plan: does the adjoint split of an SGD fine-tune keep behaviour, floor and Δ together? (E31)

*Lead session, 2026-10-22, after E22 (`reports/2026-10-22_analysis_small-finetune-splits.md`). User: "yes you should pursue
the qualitative surprise" (new experiment or amendment at the lead's choice; chosen: a new experiment, since the claim is
new). Vocabulary: `GOALS.md` → Terms.*

**In plain words.** E22 found, against its filed prediction, that when the emoticon fine-tune is trained with plain SGD
and then written onto the subcomponents by the adjoint split, the result keeps most of the trained behaviour at every
step while Δ stays fixed and the adversarial check passes. The same split of an Adam fine-tune loses the behaviour. If
this holds on other tasks and another seed, it is the first frozen-Γ map of a fine-tune that meets the weak version's
three conditions at once, and the 2026-10-21 decision's move-back condition. This experiment asks whether it does, and
why.

## Design check

- **Goal served.** Q1, the weak version: a simple map Φ(D, δW) whose output is faithful to W + δW with Δ not absorbing
  δW. The adjoint split of an SGD fine-tune is such a map if the behaviour is kept. Directly the move-back condition of
  `reports/2026-10-21_decision_regating-main-line.md` (a map at the floor that keeps the behaviour at steps 100 and 1000).
- **Nearest tried.** E14 and E16: the adjoint split of the Adam 3e-5 fine-tune keeps the floor and Δ but its P(target)
  falls 0.93 → 0.60 → 0.10 as the reproduced share falls 0.58 → 0.42 → 0.31. E22: the SGD 0.01 fine-tune's adjoint
  state reads P(target) 0.71 → 0.75 → 0.61 at steps 10/100/1000 (ratio 0.97, Δ change ≈ 0), Adam 3e-6's 0.69 → 0.47 →
  0.24. E21's sgd_finetune P5: the SGD fine-tune's component-training twin is nearly its adjoint split (input-weighted
  cosine 0.957 against Adam's 0.452). One task, one seed so far.
- **Patch or step.** A step: it tests a candidate map, not a flaw in the last run.
- **Decisions relied on.** Γ and Δ frozen; reading A; the same-job W₀ floor; component training optimiser-agnostic.
  No departure. If P1–P2 hold, the main-line decision is the user's to revisit; this plan does not move it.
- **What changes what we do next.** If the SGD adjoint state keeps P(target) ≥ 0.5 at step 1000 on a second seed and
  two more tasks while passing the adversary with Δ fixed, the project has a map meeting the move-back condition and
  the next step is the longer adversary and more seeds on it. If it fails on the other tasks, E22's reading is an
  emoticon property. If the reproduced share explains it (P3), the mechanism is the one named below and predicts
  which updates it works for.
- **Control first.** W₀ Pile floor (two adversary seeds, `w0_check`, stop on fail); then the reproductions: the large
  emoticon fine-tune's adjoint state (E22's 0.93/0.60/0.10 within 0.05) and the SGD 0.01 seed-0 emoticon adjoint state
  (E22's 0.71/0.75/0.61 within 0.05), before any new arm.
- **Numbers and expected readings.** Per state at steps 10, 100, 1000: adversarial Pile ratio (W₀ 1; every adjoint state
  so far ≤ 1.1), Δ's share of ‖δW‖² (adjoint 0 by construction; the positive control that the measurement reads 0),
  held-out P(target) of the adjoint state against the fine-tune's own (fine-tune ≈ 1.0; large adjoint 0.10 at 1000),
  the reproduced share: the input-weighted fraction of δW's energy that c·K(δW) reproduces, 1 − ‖(cK(δW) − δW)Σ^½‖² /
  ‖δW Σ^½‖² pooled over matrices (E14 read 0.58/0.42/0.31 on the large fine-tune; SGD expected high), and the size of
  change. Budget sweep on the step-1000 adjoint states (SGD emoticon seed 0, Adam 3e-6 emoticon): KL at 20/40/80/160
  sign steps with step size 2/n against W₀ seeds 0 and 1 (W₀ at 20 steps 0.66 on the L40; ratio ≤ 1.24 is a pass).

## The mechanism hypothesis (named, to be tested, not assumed)

One plain SGD step on W moves it by −η G (clipped). The adjoint split of that step is −η (UUᵀG + GVVᵀ), which is exactly
one component-training step (`GOALS.md` → Terms). So a plain SGD fine-tune is, step by step, close to its own
component-training run plus the part of each gradient outside both spans; E22's outside-both share for SGD is 0.03–0.06
at every checkpoint. Adam rescales each coordinate separately, so its δW is not a sum of gradients, and the adjoint
map reproduces less of it (0.42 → 0.31 along training, E14). Prediction: the reproduced share stays high along the SGD
path and tracks the adjoint state's P(target) across every state measured.

## Run (one job, H100, ~1.5 h, cap $10)

Order: W₀ → reproductions (emoticon seed 0: Adam 3e-5 and SGD 0.01, adjoint at 10/100/1000, gate) → emoticon seed 1:
SGD 0.01, Adam 3e-6, Adam 3e-5 → r1_t534_v369 and r0_t352_v310: the SGD fine-tune rate picked in-job by E4's rule on
`sgd_finetune`'s five-rate grid at step 10 (E24 found 0.03 on r1; the per-task KL limit as `sgd_ladder` used), Adam
3e-5 as the task's large reference → budget sweep. Every fine-tune to 1000 steps with checkpoints (10, 100, 1000);
per checkpoint the adjoint state and do-nothing through `harness.measure`, the reproduced share, the fine-tune's own
P(target). Results dumped after every state.

Predictions to file in `experiments/adjoint_sgd/results/predictions.json`:
- **P1** (seed): SGD 0.01 emoticon seed 1's adjoint state has P(target) ≥ 0.5 at step 1000 with Δ change ≤ 1e-3 and a
  passing adversarial ratio; Adam 3e-6 seed 1's P(target) at 1000 is below SGD's by ≥ 0.2.
- **P2** (tasks): on r0 and r1 the SGD adjoint state's P(target) at step 1000 is ≥ 0.5 and above the Adam 3e-5
  adjoint state's by ≥ 0.2, with Δ fixed and a passing ratio. Two-sided reading if one task holds and one does not.
- **P3** (mechanism): the reproduced share of every SGD fine-tune is ≥ 0.8 at every step and above the same task's Adam
  fine-tune's at every step; across all adjoint states, the adjoint state's P(target) ÷ the fine-tune's P(target) rises
  with the reproduced share (Spearman ≥ 0.7, reported with n).
- **P4** (budget): the SGD emoticon adjoint state's ratio stays ≤ 1.24 at 40, 80 and 160 steps (E20 Amendment 3's P10
  twin), two-sided.
- Blind to: one adversary family and budget family; one seed per new state except emoticon; tasks from one family;
  P(target) at the trigger only. Positive controls: the large adjoint state's falling P(target) and the Δ change
  reading 0 on every adjoint state.

## Not run

Other SGD rates; SGD with momentum; the adjoint split of a LoRA; more than two seeds; the strong version.

## Agent notes

### E31 adjoint_sgd (agent, 2026-10-22)

The code is **E31** (E28 to E30 went to another session). Design check as above, unchanged; the user's request is the go.
Arms, order and predictions as the plan; P1 to P4 filed in `experiments/adjoint_sgd/results/predictions.json`.

What I changed and why:
- **Adam 3e-6, emoticon seed 0 added to the reproductions.** The budget sweep reads its step-1000 adjoint state, which
  the plan's run order never trains. It is gated like the other two (E22's 0.69 / 0.47 / 0.24 within 0.05).
- **G2 added: the large fine-tune's R reproduces E14's 0.58 / 0.42 / 0.31 within 0.03.** This is the positive control
  of the new `split.reproduced_share`. If it fails, P3 is unscored and the run goes on.
- **Sigma for c and R on r0 and r1 is the task's own training windows' input moments.** `harness.with_windows` keeps
  the emoticon Sigma, so the run computes each task's. On emoticon it is the Sigma E22 used.
- **P3's "the same task's Adam fine-tune" means every Adam fine-tune on the same task and seed.** On emoticon that is
  both Adam 3e-5 and Adam 3e-6. Parts (a) and (b) are scored separately.
- **A task whose SGD grid passes no rate still runs at the highest-P rate,** as `arms.select_rate` returns it. That
  task's P2 is then unscored.
- **Budget control C_budget_w0:** W0 seed 0 at 20 steps within 5% of E20 A3's 0.6604 (L40). If it fails, P4 is
  unscored. The pass line is P10's 1.241.
- **Verdicts use `faith_score.verdict` with the same-job floor,** as E21 and E22 did.
- **States are not saved.** They are not needed: every number the analysis reads is in the JSON.
