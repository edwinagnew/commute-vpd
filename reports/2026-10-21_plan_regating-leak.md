# Plan: is re-gating a mechanism, or Γ passing on what its input already carries? (E18, `experiments/regating_leak`)

*2026-10-21, coding agent. The re-gating line's first control (`reports/2026-10-21_decision_regating-main-line.md`,
control 1; the research log's "Main line, now" card), approved class 4: a named next step of the main line on the
existing model, tasks and harness. The check is written so the record shows why; the user is told in one line. Not yet
built; to be built in a fresh session after E17 is analysed. Vocabulary: `GOALS.md` → Terms; "re-gating" is the
decision report's word for the route-split share r, the frozen Γ switching existing subcomponents on.*

## The design check

**Goal served.** `GOALS.md` → "Where an update goes": the "reveal" reading, where Γ amplifies a small update by
switching on subcomponents that already exist. Every experiment reads r at 0.5 to 0.8 for behaviours the model has
(E4 to E6, E10, E12, E16) and the user's 2026-10-20 definition of re-weighting rests on it (`backlog.md`). The
decision of 2026-10-21 made it the main line on condition that it survive this control. If the re-gating share is a
leak, the main result of the project so far is an artefact of how Γ is wired, and the line is dropped.

**What "leak" means, precisely.** r = L(C₀, G₁) − L(C₀, G₀): the original subcomponents C₀ under the gates G₁ that Γ
computes from the *updated* model's activations, against the same subcomponents under W₀'s gates G₀. G₁ is a vector of
thousands of values per position, each a function of the updated model's residual stream at that position. Two ways
r can be large. **Mechanism:** the update makes Γ switch on a fixed set of existing subcomponents at the trigger, and
those subcomponents compute the behaviour from the original model's activations: the switch pattern is the same for
every window of the task and does not depend on which window it is. **Leak:** the per-position gate values carry
window-specific information from the updated model's residual stream into the masked forward, so the original
subcomponents reproduce the behaviour only when given each window's own gates; a fixed pattern does not do it.
The recruit test (E6) is halfway there: the ~200 switched-on subcomponents, given the state's own per-window gates,
recover 40% of r, and a random same-size set 1%. It never removed the per-window information.

**Nearest thing tried.** E6 (`TRIED.md`): the switched-on set is target-specific (Jaccard 0.05 with a different-target
fine-tune's set, 0.93 seed to seed), carries 40% of r, and the natural-machinery comparison did not run (96 of the 200
positions needed). E12: for a behaviour the model lacks the gain runs through subcomponents Γ keeps off, so the leak
question is about the "reveal" tasks. Nothing has tested a shared gate pattern or a mismatched one.

**Patch or step.** A step: the control that decides whether the main line stands.

**Working decisions.** Γ frozen, do-nothing placement (δW in Δ, subcomponents untouched: r is then the whole
CI-masked effect, as in E6), reading A untouched. No departure.

**What would change what we do next.** Mechanism (P2 passes, P3 passes): re-gating is a fixed switch set per
behaviour; next is control 3 in full and the per-step version along training, then the larger model. Leak (P2 fails
on the tasks with varied contexts): the "reveal" result is a property of Γ's input, the line is dropped and the log's
Q5 is restated; Q1 returns to first place. Mixed (P2 passes on emoticon, fails on the ladder tasks): the switch set is
a mechanism only where the contexts are alike, which is a finding about synthetic narrow tasks.

**The apparatus arm, first.** G1: the do-nothing route split on each task reproduces E6 and E11 (emoticon r = 0.60
± 0.05, gain 13.5; the ladder tasks within 0.1 of E11's r), and W₀'s gates give recovery 0 exactly. Runs first, alone;
if it fails, stop.

**The numbers, and what each should read.** Per task, on the held-out windows, the W₀ subcomponents throughout:

| number | definition | W₀'s gates | random same-size set | mechanism | leak |
|---|---|---|---|---|---|
| r | route split share, do-nothing state | 0 | — | as E6/E11 | as E6/E11 |
| ρ_trig | recovery with the state's gates at the trigger position only, W₀'s elsewhere | 0 | — | ≥ 0.7 | either |
| ρ_shared | recovery with g₀(x) + mean over the training windows of the trigger gate change, clipped to [0, 1], one vector per matrix | 0 | ≈ 0 | ≥ 0.7 ρ_trig | ≤ 0.3 ρ_trig |
| ρ_mean | recovery with the mean trigger gates of the training windows (no per-window term at all) | 0 | ≈ 0 | reported | reported |
| ρ_cross | recovery with the shared change of a *different-target* fine-tune at the same trigger, read at this task's target | 0 | — | ≤ 0.1 | ≤ 0.1 (both; the null for the mask channel) |
| J_nat | Jaccard of the switched-on set with the top-|S_on| subcomponents by W₀'s gate contrast at Pile positions whose next token is the target, 600 rows | — | ≈ |S_on| ÷ 9881 ≈ 0.02 | ≥ 3 × random | ≈ random |

Recovery is E6's: the share of r obtained with the hybrid gates, so 1 means the hybrid gives the full re-gating
effect and 0 means none.

**Blind to.** Trigger-window numbers only; nothing about Pile faithfulness (E17 covers the adversary). Recovery is
blind to interactions between gate changes at different positions, which is what ρ_trig measures. J_nat depends on how
"natural" positions are defined (next-token identity), which E6 flagged as weak.

**Noise.** Two fine-tune seeds per task for r and ρ (E6: r 0.60 and 0.61, J 0.93 seed to seed); the random same-size set
over 20 draws for ρ_shared's and J_nat's nulls.

## Restatement

The project's main finding so far is that a fine-tune's new behaviour appears because the frozen importance function
switches on existing subcomponents. This run asks whether that switching is a fixed, reusable pattern that the
original subcomponents turn into the behaviour, or whether the per-position importance values are a side channel that
carries the updated model's computation past the decomposition.

## Tasks and states

Do-nothing states of the step-10 fine-tunes (seed 0 and a fresh seed 1) on emoticon (contexts alike: every window is
" :)" text), r1_t534_v369 and r2_t359_v816 (word triggers in varied Pile contexts, both read r 0.5 to 0.8 in E11 under
component training). The δW files for seed 0 are on the cluster NFS (E11); seed 1 is a 10-step fine-tune, seconds. The
different-target null for ρ_cross: E6's null_b recipe (a fine-tune at the same trigger to ` http`) re-run for 10 steps
per task.

## Predictions (to be filed in `experiments/regating_leak/results/predictions.json` before the run)

- **G1 reproduce**: r within 0.05 of E6 on emoticon and within 0.1 of E11's do-nothing r on the two ladder tasks;
  recovery with W₀'s gates reads 0 to 1e-6. Fail: stop.
- **P1 the trigger position carries it**: ρ_trig ≥ 0.7 on every task and seed. Predicted: pass (the emoticon writer
  subcomponent acts at the trigger; the gain is read there).
- **P2 a shared pattern carries it** (the mechanism-or-leak reading): ρ_shared ≥ 0.7 ρ_trig on every task and seed,
  with the random same-size set's recovery ≤ 0.1. Predicted: pass on emoticon with confidence; pass on the ladder
  tasks with low confidence. A fail on the ladder tasks only is the "mixed" branch above.
- **P3 the pattern is the task's**: ρ_cross ≤ 0.1 on every task. Predicted: pass (E6: sets of different targets
  overlap by 0.05).
- **P4 the set is the model's own machinery for the token**: J_nat ≥ 3 × the random mean on every task. Predicted:
  pass on the ladder tasks, uncertain on emoticon (E6 predicted its own R2 null: the natural-`o` contrast is dominated
  by context features). Conditional on ≥ 200 natural positions in 600 rows per task.
- Reported, not scored: ρ_mean; the switched-on set sizes and Jaccards across seeds; per-layer counts.

## Cost and platform

Per task: two 10-step fine-tunes and a null fine-tune (seconds each), CI passes over about 100 held-out windows for
four gate patterns and 20 random sets (minutes), one CI pass over 600 Pile rows for J_nat (about 2 min on an L40).
Under 20 min for three tasks: the cluster's `debug` QoS, smoke then scored on one job.

## Not run (by design)

The key → value tasks (E12's "create" branch: the leak question is about "reveal"); component-training states (the
do-nothing state isolates re-gating); the per-step version along training; a mask-capacity test (how small a direct
optimisation of the trigger gates reproduces the behaviour), kept for the backlog as the measure of how much the gate
channel *could* carry.

## Hand-off

Build in a fresh session from this file, `src/commute/recruit.py` (hybrid gates, random sets, the natural-position
contrast), `src/commute/local_faith.py` (`route_cells`, `masked_lo`) and `experiments/recruit_test/run.py` (how E6
computed recovery). File the predictions, smoke on the cluster, then the scored run through `/pod-runner`.
