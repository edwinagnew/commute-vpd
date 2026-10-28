# Plan: After teacher training, can the rest of the fine-tune go into Δ without Δ carrying the fine-tune's behaviour? (E43)

*2026-10-28, commute lead (written by a design agent). The user's request in chat, 2026-10-28; the reading and the
scope are `decisions.md`'s last two rows. Words as in `GOALS.md` → Terms. Code
`experiments/teacher_training/remainder.py` (imports `run.py` and `path.py` beside it); predictions
`experiments/teacher_training/results/predictions_e43_v2.json`, filed 2026-10-28 15:48:10 on the CPU worker, sha256
`204c1fbb70655a72b6e02eb425203cf72539ca9866658e5d0e8fdaec9316978a` (it refiles `predictions_e43.json`, sha256
`74124bbd…`, before any run; see "Refile 2026-10-28" below); brief `experiments/teacher_training/RUN_BRIEF_e43.md`.
A fresh agent reviews the code before it runs.*

## Scope as approved, 2026-10-28

The user approved a cut-down E43 after reading the first version of this plan (`decisions.md`, "run the cut-down
E43 on the H100"):

- **Card.** RunPod "NVIDIA H100 80GB HBM3". The L40's adversary spread (0.21, against 0.075) would leave the step-10
  cells at the line.
- **Cases.** Emoticon only, at T₁₀ and T₁₀₀₀, by E41's recipes (task-text copy). The broad fine-tune is dropped,
  because teacher training's own state already fails there.
- **States, on the same rows.** Do-nothing (the positive control), least squares, teacher training's own state, and
  teacher training plus remainder.
- **Removal.** Putting Δ′ back to Δ is scored. Setting Δ′ to zero is reported.
- **Which text counts.** Both the task text and plain Pile text count as the fine-tune's behaviour.
- **Predictions.** P1 (task text) is kept only as a check, since E41 already gives its value, and is unscored. P2
  (the plain-text removal share) and P3 (the adversarial check, paired with teacher training's own state) are scored.
  P3 reports the share of do-nothing's excess removed beside it.

The body below matches this scope.

## Refile 2026-10-28 (before any run)

The code review (`reports/reviews/2026-10-28_review_e43-code.md`) found P3's verdict clause decided T₁₀ inside the
noise: teacher training's own ratio there is 1.547, and the fail line 1.612 sits only 4% above it, while W₀'s adversary
spread is 7.5%; the 1.25× ratio bound (1.93) never bound. At T₁₀₀₀, 1.25 × 0.959 = 1.199 is below the pass line, so
the clause never binds there either. The clause is therefore **dropped**: P3 is the ratio bound alone. Dropping it is
simpler than "apply it only where the ratio cannot decide", which has no case where it would change the outcome
except inside the noise. Both verdicts are still reported. The predictions were refiled as
`predictions_e43_v2.json`; `predictions_e43.json` is left as filed and no longer read. The same refile corrects
teacher training's expected removal share (it equals plus remainder's, not 0: both remove to W_tt) and adds a stop at
PC when KL(fine-tune ‖ W₀) on the Pile rows is below 0.005 nats, so P2 cannot go unscored after the spend. `CFG` and
`E43` are unchanged. The new rule ran against 40 edge cases (`~/commute-data/scratch/commute-lead/e43_edge_v2.py`).

## The request, restated in the Terms (confirmed)

The first step is **teacher training** as it stands: component training of the alive (U, V), with the loss
KL(fine-tuned model ‖ student), and Γ and Δ frozen. Then, per matrix, set **Δ′ = Δ + (W_ft − W_tt)**, where W_ft is
the fine-tuned weights and W_tt is teacher training's weights. The **remainder** R = W_ft − W_tt is the part of the
update the subcomponents did not reproduce. The resulting state is called **teacher training plus remainder**. Its
weights equal the fine-tune's exactly ("Weights = fine-tune's" ✓), so Δ is no longer untouched.

For E43 only, "Δ does not absorb the update" is read **behaviourally**. Δ′ passes if putting it back to Δ barely
changes the outputs, on the task text and on plain text. Its weight size is reported beside that reading and decides
nothing.

## Design check (`/design`)

- **Goal served.** The weak version (`GOALS.md`), under the user's E43-only behavioural reading of "Δ may not absorb
  δW". If the remainder is inert, this is the first map that keeps the fine-tune's exact weights and stays as faithful
  as teacher training.
- **Nearest tried.**
  - E41 and E34 give teacher training's own state, with Δ untouched and the weights not equal to the fine-tune's.
  - E22 gives least squares, which has the fine-tune's weights, puts 0.36-0.38 of ‖δW‖² in Δ, and fails at step 10
    (6.96).
  - E34 Amendment 6 already gives r1's plain-text share for task-text copies: 0.30 (Adam), 0.33 (SGD), 1.06 (LoRA).
  - New here: the check of a state whose Δ′ holds the remainder, and the plain-text share on emoticon beside least
    squares.
- **Patch or step.** A step: a new map, built from two existing operations, read against the user's reading.
- **`decisions.md`.** It departs from reading A as written; the user recorded that departure for E43 only. It relies on
  teacher training at Adam 1e-5 (2026-10-28) and on one adversary start with readings as lower bounds (2026-10-27).
- **What changes what we do next.**
  - P2 "inert" with P3 holding: a card asks whether teacher training plus remainder becomes the map.
  - P2 "carries" or P3 failing: the exact-weights version costs faithfulness, and teacher training's own state stays
    the map.
- **Control first.** C1 (W₀) runs alone at the start, with a stop. Then, per step, the do-nothing state (PC) runs before
  any copy.
- **Reference recipes.** The fine-tune and the copy are E41's unchanged. E41 reproduced to 16 digits on this card.
  G1, PC and G3 check that they reproduce here.
- **Class.** A new map, confirmed by the user after the check (`decisions.md`, 2026-10-28).

**Known weaknesses.**
- P1 has no power: removing the remainder gives teacher training's own state, whose gap share E41 measured (0.0013,
  0.0056).
- The Δ′ share is expected to be about 0.9-1.1 by construction.
- P3 at T₁₀₀₀ can only see a difference larger than 25%, and the null predicts about 15% there.

## States (per step k ∈ {10, 1000}; one job, about 15 min of H100)

| State | Made by | Δ′ − Δ | With Δ′ back to Δ it is |
|---|---|---|---|
| W₀ | `harness.load` | 0 | W₀ |
| do-nothing | target W₀ + δW_k, released (U, V) | δW_k | W₀ |
| least squares | `split.split_state(δW_k)`, target W₀ + δW_k | about (I − P_U) δW (I − P_V) | the split's components with W₀'s Δ |
| teacher training | `path.student` (E41's one-shot task-text copy: Adam 1e-5, 1,000 steps, batch-order seed 0) | 0 | itself |
| **plus remainder** | the copy's (U′, V′), target **W₀ + δW_k** | R = δW_k − δW_tt | teacher training's own state (G2 checks this) |

The fine-tune is E41's emoticon Adam 3e-5 arm-A fine-tune: all 24 matrices, batch 32, seed 0, checkpoints at 10 and
1,000, regenerated in the job. No state is read from disk and none is saved; the analysis reads only the JSON.

## Stage 0: controls (stop on any miss)

| Check | Must read (E41, same card) |
|---|---|
| C1, first and alone | `w0_check` passes; PGD floor 0.0748 within 10%; seed-0 PGD KL 0.69882 within 1% |
| G1 | T₁₀: P(target) 0.99931 and size 0.01125; T₁₀₀₀: 0.99961 and 0.03209 (within 0.005 and 0.001) |
| PC, per step, before the copy | do-nothing ratio 2.5159 (T₁₀, fail) and 1.4994 (T₁₀₀₀); its task-text KL to W₀ 5.0784 and 5.0897 (E41's gap at W₀); all within 1%; its Pile-row KL to W₀ ≥ 0.005 nats (P2's denominator; refile) |
| G3, per step | the copy's P 0.99941 and 0.99675 (within 0.005); ratio 1.5465 and 0.9591 (within 5%); Δ share ≤ 1e-6 |
| G2, per step | the copy's U′V′ᵀ − UVᵀ reproduces its δW within 1e-3 of the largest entry, so removing R lands exactly on the copy |

Do-nothing is the positive control on both readings:
- **Removal.** Its removal share is 1 by construction, since its removal state, W₀, gives the denominators.
- **Check.** At T₁₀ it fails (2.52). The check sees an update that sits in Δ.

## The numbers (T₁₀ / T₁₀₀₀; references on the same rows, same job)

| Number | W₀ | do-nothing | least squares | teacher training | plus remainder: expected |
|---|---|---|---|---|---|
| **b_pile** = KL(fine-tune ‖ Δ′ back to Δ) / KL(fine-tune ‖ W₀) on the harness's 256 Pile val rows | 0/0 | 1 | measured | = plus remainder's (same removal state W_tt) | unknown; r1's task-text copies read 0.30-0.33 |
| **check** (Pile harness PGD KL over W₀'s; one start, seed 0, 20 steps) | 1 | 2.52 / 1.50 | 6.96 / 2.11 (E22) | 1.55 / 0.96 (E41) | at least teacher training's; the null puts it nearer do-nothing |
| b_task (the same ratio on held-out task windows, arm A weighting; P1 check) | 0/0 | 1 | measured | = plus remainder's | 0.0013 / 0.0056 (E41's gap shares) |
| Δ′ share ‖Δ′ − Δ‖² / ‖δW‖² (reported) | 0 | 1 | 0.36 / 0.38 | ~1e-9 | about 0.9-1.1 |
| e_delta, Δ′ to zero (route split, log-odds at the trigger; reported) | 0 | 1 | 0.09 / 0.28 | 0.011 / 0.015 | slightly above teacher training's |
| cos(R, the copy's δW) (reported) | — | — | — | — | about −0.3; below −0.5 means R mostly undoes the copy |

**Blind to.**
- b_pile sees nothing at the trigger and nothing off the 256 rows.
- b_task sees nothing off the task windows.
- Both shares are ratios, so either is unscored if its denominator is below 0.005 nats (10 × E41's hook floor of
  0.0005).
- Because both remove to W_tt, b_pile reads the copy's own plain-text fidelity as much as the remainder's.
- The check uses one start, so a pass is a lower bound. Its verdict (reported) combines the stochastic and PGD ratios.
- The check runs the masked forward under bf16 autocast (the harness's convention for every state), so it is blind
  to Δ′ below bf16 rounding.
- The Δ′ share sees no inputs.
- e_delta saturates near P = 1.

**Noise floors (H100, measured).**
- Check: W₀'s spread over adversary seeds is 0.0748 (E41), and the lines are 1.396 (pass) and 1.612 (fail). The
  ratio's batch-order spread is 21% at T₁₀ and 3.2% at T₁₀₀₀ (E41 Amendment 1). P3 is paired (same copy, rows and
  adversary seed), so its noise is at most these.
- b_pile: the batch-order spread is 0.10 (E34 Amendment 6, emoticon Pile-text copies, 0.77 against 0.87).
- Gap share: 0.002 (E34 Amendment 4).

## Predictions (refiled; rules as in `remainder.score`, 40 edge cases run before filing)

**Claim.** The remainder is behaviourally inert, so teacher training plus remainder is as faithful as teacher
training's own state while keeping the fine-tune's exact weights.

**Null.** The remainder carries what the copy got wrong off the task text, and the adversary, which can remove Δ′,
reads that part as it reads do-nothing.

- **P1, task-text check (unscored).** Per step: b_task ≤ 0.05, and teacher training's own P(target) is ≥ 0.9. Its
  value is known from E41.
- **P2, plain text (scored).** Per step, b_pile is "inert" at ≤ 0.2 (twice the batch-order spread), "carries" at
  ≥ 0.5 (half of do-nothing), and "between" otherwise; both ends are inclusive.
  - The outcome is the worst step.
  - It is unscored if a record is missing or KL(fine-tune ‖ W₀) on the Pile rows is below 0.005 nats.
  - Least squares' b_pile is reported beside it.
  - *Lean: between, medium (r1's 0.30-0.33).*
- **P3, the check, paired (scored).** Per step, plus remainder's ratio is ≤ 1.25 × teacher training's own (relative,
  inclusive). No verdict clause (dropped at the refile: at T₁₀ it decided inside the noise). Both verdicts are reported.
  - It holds if both steps hold, and is unscored if a record is missing.
  - Reported per step: the share of do-nothing's excess removed, (r_dn − r)/(r_dn − 1), for both states. It is None
    where do-nothing reads at or below the pass line.
  - *Lean: holds; high at T₁₀₀₀, where the null's gap is about 15%; medium at T₁₀.*

**Edge cases run through `score`** (synthetic records on the CPU worker,
`~/commute-data/scratch/commute-lead/e43_edge_v2.py`; all 40 pass):
- an empty file, a missing step and a missing record are all unscored;
- b_pile exactly 0.2 is inert and 0.2000001 between; exactly 0.5 carries and 0.4999999 between; the worst step wins;
- a denominator of 0.005 is scored, while 0.004999 and 0 are None;
- a ratio of exactly 1.25 × holds and 1.2500001 × fails;
- verdicts are reported and never decide (a fail verdict within 1.25× holds; both failing beyond 1.25× fails); a NaN
  ratio fails; one failing step fails P3; a teacher ratio of 0 raises;
- a do-nothing ratio at the pass line or below 1 gives an excess share of None;
- a P1 denominator of 0 gives None;
- a copy below P 0.9 makes P1 false;
- an unknown verdict is not read.

## Decision tree

- **A control fails** (C1, G1, PC, G2, G3): stop, report, fix. Nothing is read.
- **P2 inert and P3 holds.** Teacher training plus remainder meets the E43 reading on emoticon wherever teacher
  training does. A pending-decision card goes to the user with two questions: adopt it as the map, and should the
  behavioural reading hold for every technique (a `GOALS.md` change, the user's). No run.
- **P3 holds, P2 between or carries.** The adversary does not see the remainder, but plain text does. A card offers
  three options; nothing runs before the user decides.
  - Accept it.
  - Copy from Pile text instead (r1's Pile-text copy read 0.13 against 0.30).
  - Put the least-squares split of R into the subcomponents first.
- **P3 fails.** The adversary uses Δ′ as it uses do-nothing's Δ, so exact weights cost faithfulness. This is a
  qualitative surprise against the claim: announce it, file a card, and keep teacher training's own state as the map.
  No run.
- **P1 check false.** A code or reproduction problem, since E41 fixes its value. Stop and report.

## Cost and routing

The run takes about 15 min on an H100, timed from E41 (harness read 53 s, copy with its read 126 s, load and C1 3 min):

| Stage | Time |
|---|---|
| load and C1 | 3 min |
| fine-tune and G1 | 0.5 min |
| T₁₀: do-nothing and PC, the copy and G3, plus remainder, least squares, removal readings | about 5.5 min |
| T₁₀₀₀: the same | about 5.5 min |

- **Card:** RunPod "NVIDIA H100 80GB HBM3". C1, PC and G3 belong to that card.
- **Cost:** about 25-30 min of pod with set-up and downloads, at about $4.4/h: **about $2, cap $4**. That is within
  the session's budget.
- **Gating:** not gated (under $15 and under 3 h).

## Order of work

1. Done: code, filed predictions, edge cases, lint (ruff and pyright clean).
2. A fresh Opus agent reviews `remainder.py` (scope: the new file, uncommitted at review time, and the repo path). Fix
   what is real. A change to `E43` or `CFG` means refiling under a new name.
3. `/pod-runner` with `RUN_BRIEF_e43.md`.
4. Write the analysis by `/writeup`, add the `TRIED.md` row and the PENDING line, then commit and push.

## Not run

- The broad lowercase fine-tune (dropped at approval); T₁₀₀; other tasks; SGD and LoRA teachers; Pile-text copies; Qwen.
- Batch-order or seed repeats; adversary seeds 1-2 and the budget adversary.
- The least-squares split of R into the subcomponents (offered on a card only); input-weighted shares of R.
- Saving states.
