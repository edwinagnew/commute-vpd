# Plan: is re-gating a description of the fine-tuned model? Recruitment test, routing trace, piece census (E38)

*2026-10-27, lead agent, for the `regating-lead` fleet agent. From the user's decision of 2026-10-27
(`reports/decisions/2026-10-27_decision_regating-fixed-shift-deprioritised.md`) and the research log's pending card
"What would it mean to understand re-gating?" (`decision-regating-reading`). The user fixed the order: recruitment
test, then routing trace, then piece census; refit agreement is parked. Words as in `GOALS.md` → Terms. This file
is the design check; the fleet agent files predictions from the thresholds below before anything runs.*

## The question, in plain words

After a narrow fine-tune, Γ, frozen, reading the fine-tuned model, switches on a set of the original model's
subcomponents at the trigger. We call that set the *recruited set*. Every re-gating number so far was read in the
masked model: old pieces, new importance values. Nobody has asked the fine-tuned model itself, which has no Γ,
whether it acts through those pieces. This plan asks that first, then where the activations moved to make Γ recruit
them, then what the recruited pieces do in the original model.

Three stages, in the user's order, each a gate for the next:

1. **Recruitment test.** Remove the recruited pieces from the fine-tuned weights and see whether the behaviour goes.
2. **Routing trace.** Does the fine-tune move each matrix's input at the trigger toward the inputs the original
   model has where it already predicts the target?
3. **Piece census.** For each recruited piece, where the original model turns it on in ordinary text and what it
   predicts there.

## Design check (`/design`)

- **Goal served.** Q5, re-gating, and through it `GOALS.md`'s "reveal" hypothesis: a post-training update as a small
  change that Γ amplifies by switching on subcomponents that already exist. The recruitment test is the first
  causal test of "reveal" in the real model. The routing trace is "reveal" stated as a measurement on the
  activations Γ reads. The census is what makes a re-gating finding legible: named pieces with known jobs.
- **Nearest tried.** E6 and E18 defined the recruited set and showed it is target-specific and overlaps the
  original model's machinery for the target; both read it in the masked model only. E12 ablated a different set in
  the full model, the busy pieces the least-squares split moves most, disjoint from the recruited set, and found
  non-specific damage in both task classes; that is this plan's "busy" reference. E29 looked at the turned-down
  half. E15 stage 0 ablated single pieces for the adversary question. Nothing has removed the recruited set from the
  fine-tuned weights, measured where the trigger's inputs moved, or listed what the recruited pieces do.
- **Patch or step.** A step: it turns re-gating from a reading of Γ into a claim about the model that can fail.
- **Decisions relied on.** Γ frozen; E18's states and definitions (do-nothing, arm-A Adam 3e-5, 10 steps; the
  recruited set as the alive pieces whose mean trigger importance rises by more than 0.3). No departure.
- **What changes next.** Recruitment test fails: re-gating is Γ's reading, not the model's mechanism; the line
  stops, a Blocking card goes to the user, and the budget stays with the commute line. Passes: the pending card's
  readings go ahead in order; the write-up of a fine-tune becomes "the update moved the trigger's representation
  here, which recruited these pieces, which do this". Partial: the recruited set is part of the mechanism and the
  card says what the rest might be.
- **Control first.** Stage 0 (below) runs alone before anything is read: W₀'s trigger cells and each state's
  reproduce E18; the saved recruited sets match the in-job ones; removing an empty set changes nothing.
- **Approved class.** Class 1 (analysis of existing states with GPU passes of existing code) and class 4 (the
  re-gating card's named next steps). The user confirmed the three tests and their order on 2026-10-27 in chat.

## States

E18's recipe, regenerated in-job as E29 and E35 did (they reproduced E18 within 0.0013): arm-A fine-tunes of all 24
decomposed matrices, Adam 3e-5, batch 32, 10 steps, seeds 0 and 1, on E18's training windows; held-out windows as
E18 (73 to 106 per task). Tasks: emoticon (` :)` then `o`), " which" then " was" (`r1_t534_v369`), " we" then
" just" (`r2_t359_v816`). Plus the null: E18's ` http` fine-tune at each task's trigger, seed 0. Nine fine-tunes,
about a minute each on a 4090. The saved recruited sets are at
`experiments/regating_leak/results/large/<task>_s{0,1}_shift.pt` (`shift`; the set is `recruit.switched(shift, 0.3)`),
and the in-job set must match them (stage 0).

Everything in this plan reads the **full model**, W₀ + δW with the released weights and no Γ, except where a line
says otherwise. Removing a piece means subtracting its rank-1 matrix from the weights (`signatures.ablated_weights`).

## Stage 0: the apparatus (runs first, alone; early stop)

| check | reads | must read | else |
|---|---|---|---|
| W₀ cells | mean trigger log-odds of the target at W₀ on the held-out windows, per task | E18's T0 within 0.02 | stop, fix |
| states | the fine-tuned model's T1 and its held-out P(target) | E18's within 0.02 and 0.01 | stop, fix |
| recruited set | in-job shift against the saved file: size of the set and Jaccard | Jaccard ≥ 0.95, same size ± 5% | stop, fix |
| empty removal | T1 with no piece removed | equals T1 to 1e-5 | stop, fix |
| busy reference | removing the top-|S| alive pieces by mean trigger importance at W₀ from W₀ | moves the trigger distribution (KL to W₀ ≥ 1 nat) | the ablation code does not bite: stop |

## Stage 1: recruitment test

For each state. S is the recruited set (threshold 0.3; about 150 to 200 pieces on emoticon). Mean trigger log-odds
of the target on the held-out windows; gain G = T1 − T0. Every number is a removed share ρ = (T1 − T1 without the
set) / G, so 0 is "removing it does nothing" and 1 is "removing it undoes the fine-tune".

| number | set removed from | expected if re-gating is real | expected if it is Γ's reading |
|---|---|---|---|
| ρ_S | S, from the fine-tuned weights | ≥ 0.5 | ≈ ρ_random |
| ρ_random (mean, max over 20 draws) | a random alive set of the same size per matrix (`recruit.random_like`) | ≤ 0.1 | ≤ 0.1 |
| ρ_busy | the busy reference set (same size), from the fine-tuned weights | anything; read with its off-target KL | same |
| ρ_S at W₀ | S, from W₀ (T0 − T0 without S, over G) | ≈ 0: S is off at W₀ | ≈ 0 |
| ρ_null | the ` http` fine-tune's recruited set, from this task's fine-tuned weights | ≤ 0.1 | ≤ 0.1 |
| off-target KL | KL(state ‖ state without the set) at the 40 non-trigger positions, as E18's `kl_other` | for S: ≤ 3 × the fine-tune's own off-target KL to W₀ (0.055 on emoticon); for busy: ≥ 10 × | — |
| ρ at top-k | the k pieces with the largest rise, k = 25, 50, 100, 200, 400 (reported, not scored) | rises with k | flat near ρ_random |

**Predictions to file (thresholds fixed here; the fleet agent writes them to `predictions.json` before the run).**

- **P1, recruitment is real.** ρ_S ≥ 0.5 on at least two of the three tasks, both seeds within 0.1 of each other,
  with ρ_random's max ≤ 0.15 and ρ_null ≤ 0.15 on those tasks. *Lean: medium-low.* In the masked model the
  recruited set carried only 40 to 64% of the re-gating effect (E4, E6), so ρ_S between 0.3 and 0.6 is the
  likeliest reading.
- **P2, specificity.** On the tasks where P1 holds, removing S costs at most three times the fine-tune's own
  off-target KL, while the busy set costs at least ten times. *Lean: medium.*
- **P3, S is off at W₀.** ρ_S at W₀ within ± 0.1 on every task. *Lean: high.*

**Decision tree.**

- P1 passes: stages 2 and 3 run. The pending card's evidence gains "the fine-tuned model acts through the recruited
  pieces", with ρ_S.
- Partial: ρ_S between 0.2 and 0.5 on two or more tasks, above ρ_random's max by 0.1 or more. Stages 2 and 3 run.
  The report says the recruited set is part of the mechanism and names the top-k curve as the clue to the rest
  (does ρ keep rising past |S|, which would mean the threshold cuts the set short, or saturate, which would mean
  the rest is not in any set of pieces).
- Fail: ρ_S < 0.2 on two or more tasks, or within 0.1 of ρ_random's max. Stages 2 and 3 do **not** run. A Blocking
  card and a PushNotification: re-gating is Γ's reading, not the model's mechanism; the line's premise fails.
- P2 fails while P1 passes: removing S damages the model broadly, so S includes busy pieces; report the top-k curve
  and its off-target KL per k, and the card asks whether to score at a smaller k.

## Stage 2: routing trace (only if stage 1 did not fail)

Γ reads the 24 matrices' raw inputs (`vpd.compute_ci` caches them). For each matrix m and each state, at the
trigger of the held-out windows: a₀ under W₀, a₁ under the fine-tuned model, δa = mean over windows of a₁ − a₀. The
*natural direction* n_m is the mean input at Pile positions whose next token is the target
(`recruit.precede_mask`) minus the mean input over all positions, over enough Pile rows that each target has at least
200 such positions (E18 had 109 for " just" on its rows: use more rows, and report a target below 200 as unscored).

| number | reads | expected if "reveal" | null |
|---|---|---|---|
| cos_m | cosine of δa and n_m, per matrix | ≥ 0.3 on the matrices of the layers where S sits | the ` http` fine-tune's δa against this target's n_m: near 0 or negative; random directions: about 1/√d |
| closer_m | ‖a₁ − μ_target‖ / ‖a₀ − μ_target‖, μ_target the mean input before the target in Pile, per matrix, mean over windows | < 1 where cos_m is high, with ‖a₁ − μ_all‖ / ‖a₀ − μ_all‖ ≈ 1 | the ` http` state: ≈ 1 |

**P4, the update points at the target's contexts.** On each task where stage 1 did not fail, cos_m ≥ 0.3 on at least
half the matrices of the layer that holds most of S, and at least three times the ` http` null's cos on the same
matrices. *Lean: medium-low.* If stage 1 passed and P4 fails, Γ recruits the pieces through a direction that is not
"look like a natural target context"; the report says so and the card gains that question. The second half of the
trace, which upstream pieces' outputs moved δa, is **not in this plan**: design it after P4 is read.

## Stage 3: piece census (descriptive; only if stage 1 did not fail)

Emoticon seed 0 first, then the two word tasks. Over 2,000 Pile rows of 512 tokens at W₀, with `compute_ci`, for
every piece in S and in five random same-size sets: the share of positions where its importance exceeds 0.5 (on-rate);
the ten next tokens most over-represented at the positions where it is on, by the ratio of their rate there to their
base rate (needs at least 50 on-positions, else "rare"); the same for the current token; its layer and matrix. Then
sort S into: *target pieces* (the target among the top three next tokens), *trigger pieces* (the trigger token among
the top three current tokens), *both*, *neither*, *rare*. The random sets give each class's base rate.

No prediction is scored; the one check is that the share of target pieces in S exceeds the random sets' by at least
three times, which E18's overlap number says it should. Output: one table per task, and two sentences per task saying
what the recruited pieces are, in words a reader can check against the table. Kept provisional until an independent
check reads the table against the raw counts.

## Cost and routing

All forward passes and nine one-minute fine-tunes. Stage 1: nine states × about thirty trigger evaluations of about
100 windows, plus off-target KL: minutes. Stage 2: one pass of `compute_ci` over the Pile rows at W₀ with running
sums per matrix, and trigger-only passes per state: minutes. Stage 3: one pass over 2,000 rows with per-piece
counters for about 1,200 pieces: under ten minutes on a 4090. Under 30 minutes in all, so the MATS cluster's `debug`
QoS first, RunPod 4090 if not started within 10 minutes (`.claude/CLAUDE.md`). Budget: under $3 on RunPod. Each
stage writes its results file as it finishes; the control stage first, alone.

## Order of work for the fleet agent

1. Read this plan, the decision report, E18's analysis (`reports/analyses/2026-10-21_analysis_regating-leak.md`) and
   `experiments/regating_leak/run.py` for the harness; E12's `new_vs_existing/run.py` for the ablation; E29's
   `regating_down/run.py` for loading the saved shifts and the consistency check.
2. Write `experiments/recruitment_test/run.py` with shared steps in `src/commute/` (`regating.py` for the states,
   `recruit.py` and `signatures.py` for the sets and the ablation; a new `census.py` for stage 3). One config, stages
   cheapest first, every number printed to the log. No `--smoke`.
3. File `results/predictions.json` with P1 to P4 and the stage-0 rules as written here.
4. One review of the measurement code by a fresh Opus agent that did not write it, before the GPU run.
5. Run through `/pod-runner` (or the cluster), stage 0 alone first with the early stop.
6. Analysis report `reports/analyses/2026-10-27_analysis_recruitment-test.md` and notebook
   `notebooks/E38_recruitment_test.ipynb` by `/writeup`; independent check by a fresh Opus agent; `TRIED.md` row;
   the log: card `e38`, Q5's status, the pending card `decision-regating-reading` (its "Running / Due to run" line
   while running; its "What we know" bullets after), one timeline line; flush and republish.
7. Hand-off note, commit, push.

## Not in this plan

Which upstream pieces move δa (the second half of the routing trace); the refit agreement (parked by the user);
positions other than the trigger; the key-value and cipher tasks; seeds beyond 0 and 1; the larger model; any
change to the subcomponents or to Γ.
