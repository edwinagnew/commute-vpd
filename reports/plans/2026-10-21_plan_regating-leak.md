# Plan: is re-gating a mechanism, or Γ passing on what its input already carries? (E18, `experiments/regating_leak`)

*2026-10-21, coding agent. The re-gating line's first control (`reports/decisions/2026-10-21_decision_regating-main-line.md`,
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

## Build notes (2026-10-21, coding agent; predictions unchanged)

Code: `experiments/regating_leak/run.py`, `plot.py`, `cluster/regating_leak.sbatch`; shared steps appended to
`src/commute/recruit.py` (trigger-only gate patterns) and `src/commute/local_faith.py` (`trigger_mean_gates`).

**Changes from the plan.**
- *G1's references.* E11 never computed r. The ladder references are E5's do-nothing r at step 10 (`task_ladder/results/ladder.json`),
  from the same δW E11 regenerated: r1_t534 **0.746**, r2_t359 **0.284**. The plan's "both read r 0.5 to 0.8" is wrong for
  r2_t359's do-nothing state (0.28; gain 12.5, so the re-gating effect is still 3.6 nats). Emoticon: E6's ft_s0, 0.5965.
- *No separate smoke* (CLAUDE.md, 2026-10-21): G1 needs no training, runs first and alone, and stops the job on a fail.
- *States.* Seed 0 = E11's saved `dW_ft_step10_s0.pt` per task; seed 1 trained in the job, all three tasks.
- *S_on* is the alive subcomponents whose shared change (from the training windows) exceeds 0.3, not E6's held-out dg.
- *The random null* for ρ_shared: S_on's shared-change values moved onto a random same-size-per-matrix set outside
  S_on (20 draws), so the null switches on as much as S_on does, on the wrong subcomponents. ρ_mean gets no separate
  random null; instead ρ_mean_W0 (W0's own mean trigger gates for every window) is reported as its reference, because
  the mean pattern is not 0 at W0 (it removes W0's own per-window variation).
- *ρ_cross* uses ` http` (token 3944) as the different target on all three tasks, seed 0, 10 steps; its held-out
  P(` http`) is reported. If it did not learn, P3 is a null without power for that task.
- *J_nat* on Pile val rows 0..599 (E6 used 2048..2303); top-|S_on| by contrast over all matrices (`signatures.top_by`);
  scored on seed 0, seed 1 reported.
- *Added, unscored* (cheap: ~2 min, last stage): the Pile adversary's (20-step `gamma.pgd_sources`, seed 0, val rows
  0..63) turned-down share (source < 0.5) of S_on against all alive subcomponents, at each seed-0 state and at W0.
  This asks E17's sharpened question: are the subcomponents Γ switches on at the trigger the ones the adversary removes?
- P1 to P3 are scored on both seeds, P4 on seed 0. After the Codex review: P1 to P3 read only the task-seed states with gain ≥ 1 nat and re-gating
  effect L01 − L00 ≥ 0.5 nats (the rest reported); P4 is unscored, not passed, if no task has 200 natural positions.

**Expected values as the code computes them** (held-out windows; recovery = (lo − L00) ÷ (L01 − L00)):

| number | expected |
|---|---|
| r (seed 0) | 0.60 emoticon, 0.75 r1_t534, 0.28 r2_t359; seed 1 within about 0.05 |
| recovery with W0's gates (trig; shared with a zero shift) | 0 exactly (G1) |
| ρ_trig | ≥ 0.7 under both readings |
| ρ_shared | ≥ 0.7 ρ_trig mechanism, ≤ 0.3 ρ_trig leak |
| ρ_mean, ρ_mean_W0 | reported; ρ_mean_W0 near 0, either sign |
| ρ_cross | ≤ 0.1 |
| random moved shift (mean of 20) | about 0, ≤ 0.1 |
| \|S_on\| | about 200 on emoticon (E6: 202 at 0.3) |
| J(S_on s0, s1) | about 0.9 (E6: 0.93) |
| J_nat random mean | about \|S_on\| ÷ 9881 ÷ 2 ≈ 0.01 |
| adversary turned-down share | no expectation; W0's adversary on the same S_on is the reference |

Runtime: under 15 min on one L40 (model load 1 min; G1 1 min; three tasks of three 10-step fine-tunes and ~50 masked
passes over ~100 windows each; one CI pass over 600 Pile rows; four 20-step adversaries on 64 rows, ~25 s each).

## Amendment 1 (2026-10-21, coding agent; E18's predictions unchanged)

**Why.** E18's filed P2 read 0.40 and 0.62 of ρ_trig on the word tasks, between the mechanism and leak bars, while the
unscored mean pattern m₁ read 0.94 and 1.11 (`reports/analyses/2026-10-21_analysis_regating-leak.md`). The user approved a short
diagnostic of which difference lowers the shared pattern. Predictions filed in
`experiments/regating_leak/results/predictions_amend1.json` before the run; code `run.py --amend1` (functions `a1_*`),
`recruit.added`; sbatch `cluster/regating_leak_amend1.sbatch`.

**Arms** (per task and seed, as shares R = ρ ÷ ρ_trig, held-out windows as E18). The shared pattern clip(g₀(x) + s) and
the mean pattern m₁ differ in three ways; one arm removes each, so adjacent arms differ in exactly one:
shared → **noclip** g₀(x) + s (the clip) → **mean_s** m₀ + s (W₀'s per-window variation g₀(x) − m₀) → mean m₁ (dead
subcomponents: m₀ + s keeps m₀ there, s being 0 on dead; m₁ uses the state's mean). With D = R_mean − R_shared, the
shares f_clip, f_variation, f_dead sum to 1. Also **S_on** (the shift only on the switched-on set, clipped; reported)
and mean_W0 repeated: f_variation_W0 = R_mean_W0 ÷ D is the part of the variation step that W₀'s own variation costs
with no shift, so f_variation − f_variation_W0 is how much the variation interacts with the shift. A ρ_mean "per-window"
arm m₀ + (g₀(x) − m₀) + s is noclip and was not added.

**Gate, first.** G1 as E18, then ρ_shared and ρ_mean repeated within 0.02 of E18 on every task and seed (seed 1
retrained with the same seed), and noclip with s = 0 recovering 0 to 1e-6. A fail stops the job before any new arm.
At W₀'s gates (s = 0) noclip, S_on and shared read 0; mean_s reads ρ_mean_W0 (0.10 to 0.28), not 0, by construction.

**Prediction A1_P1** (states with D ≥ 0.3, i.e. the word tasks): the reading is each term with f ≥ 0.5, "split" if
none. Clip reading: R_noclip ≥ R_shared + 0.5 D. Variation reading: R_noclip near R_shared and R_mean_s near R_mean.
Dead: R_mean well above R_mean_s. Predicted: variation on both word tasks, medium confidence (a subcomponent on in some
W₀ windows and off in others gets s added only part-way in the off windows; the clip acts only on windows already near
1), with f_clip within ±0.2 and f_dead within ±0.1. Reported: emoticon, R_S_on (within 0.1 of R_shared if the off-S_on
shift is inert), and the interaction f_variation − f_variation_W0 (predicted > 0.2, low confidence).

**What it decides.** Clip: the filed pattern's construction caused the miss; the mechanism reading stands. Variation:
an additive shift on top of each window's own gates is the wrong model of a fixed switch pattern, and the per-window
structure of W₀'s gates interacts with it; this is not a leak (no arm uses the updated model's per-window values).
Dead: a bookkeeping difference between the two patterns.

**Blind to.** The split is along one path; with interactions the shares depend on the order. Trigger position only.

**Cost.** No Pile stages; G1, one retrained seed-1 fine-tune per task and ~7 masked passes per state: about 4 min on an
L40, cluster debug.

## Amendment 2 (2026-10-21, coding agent; E18's and Amendment 1's predictions unchanged)

**Design check** (`/design`, approved class 4: the main line's named next step "what the switched-on set did", on the
existing model, tasks and harness; the user is told in one line).
- *Goal.* `GOALS.md` → "Where an update goes", the "reveal" reading. Amendment 1 found that a fixed change recovers at
  least half the switching effect, but most of it sits outside S_on, in the ~9,600 alive entries with s ≤ 0.3, which
  E18's random null never moved. The mechanism reading needs those entries' *identity* to matter.
- *Nearest tried.* E18's random null (S_on's values moved onto random same-size sets: ≤ 0.01) tests only S_on's part;
  Amendment 1's S_on-only arm (0.07 to 0.32) says that part is the minority. Nothing has shuffled the whole change.
- *Patch or step.* A control on the main line's last result (analysis Amendment 1 → What next 1, and 2 as a cheap
  extra).
- *Working decisions.* As E18 (Γ frozen, do-nothing placement). No departure.
- *What would change what we do.* Identity (the shuffled change recovers ≤ 0.2 of the unshuffled): the mechanism
  reading holds for the bulk of the change, and the overlap test should be redone on the entries that carry it (the
  bins say which). Bulk (≥ 0.5): part of the switching effect is "shift many values up", not a target-specific
  pattern; the mechanism claim is narrowed to what is left over the null, and reported to the user before anything
  builds on it.
- *Control first.* G1 as E18, then the unshuffled change reproduces Amendment 1's trig, shared and no-clip
  recoveries to 1e-6 on every task and seed (A1 repeated E18 bit for bit). A fail stops the job before any shuffle.

**Arms** (trigger position only, held-out windows, W₀'s gates elsewhere; R = ρ ÷ ρ_trig as in Amendment 1):
s_perm = s's values permuted among each matrix's alive subcomponents (every entry, dead stay 0; `recruit.shuffled`,
seeds 0 to 19). **shared_perm** clip(g₀(x) + s_perm) and **noclip_perm** g₀(x) + s_perm. Extra, reported:
**bins**, g₀(x) + s restricted to the alive entries with s in one of (−∞, −0.1], (−0.1, −0.01], (−0.01, 0.01],
(0.01, 0.03], (0.03, 0.1], (0.1, 0.3], (0.3, ∞) (`recruit.in_range`), not clipped. On seed 0 the bins hold, from the
saved shifts, e.g. emoticon 622 entries ≤ −0.01 (sum −105), 8,701 near 0, 348 in 0.01 to 0.3, 210 over 0.3 (sum +148);
r2_t359 391 (−29), 9,335, 132, 23 (+10).

| number | definition | at W₀ (s = 0) | unshuffled (positive control) | identity | bulk |
|---|---|---|---|---|---|
| R_shared, R_noclip | Amendment 1's, repeated | 0 | 0.73/0.72, 0.41/0.40, 0.62/0.62; 0.81/0.79, 0.51/0.50, 0.78/0.78 | — | — |
| ratio(shared) | mean of 20 R_shared_perm ÷ R_shared | 0 (s_perm = 0) | 1 | ≤ 0.2 | ≥ 0.5 |
| ratio(noclip) | the same for noclip | 0 | 1 | ≤ 0.2 | ≥ 0.5 |
| sd, min, max over draws | the null's spread | 0 | — | reported | reported |
| R_bin | noclip on one bin | 0 | the bins need not sum to R_noclip | reported | reported |

**Blind to.** The shuffle keeps each matrix's values and their sum, so it cannot say whether the split *across*
matrices (layers, attention vs MLP) matters. It moves negative values onto subcomponents that are off at the trigger:
the clip zeroes them in shared_perm, but in noclip_perm they flip the subcomponent's sign, so noclip_perm is a harsher
null than shared_perm and can read negative. A negative ratio reads "identity" under the rule. Trigger position only.

**Noise floor.** The spread over the 20 draws, reported per state; E18's S_on null had max 0.007. The seeds agree within
0.02 on the unshuffled R. At the identity bar, 0.2 × R_shared is 0.08 of ρ_trig on r1_t534, the smallest.

**Predictions** (`results/predictions_amend2.json`, filed before the run; relative to the unshuffled pattern):
A2_P1, ratio(shared) ≤ 0.2 on every scorable state, predicted pass with medium-high confidence; A2_P2, the same for
noclip, predicted pass with medium confidence. ≥ 0.5 is the bulk reading; between is reported as between.

**Cost.** Per state: 3 gate passes, 40 shuffled passes, 7 bin passes = 50 masked passes over 73 to 106 windows; six
states, 300 passes. E18's log puts a pass at about 0.2 s (24 passes plus one mean-gate pass in 4 to 6 s per state),
so about 60 s of passes. Amendment 1's gate stages (model load, G1, three seed-1 fine-tunes) took about 25 s of its
39 s. Expected 1.5 to 3 min; `--time 00:15:00`, cluster debug, `cluster/regating_leak_amend2.sbatch`. $0.

**Code.** `run.py --amend2` (functions `a2_gate`, `a2_arms`, `a2_bins`, `spread`, `a2_score`), `recruit.shuffled`,
`recruit.in_range` (tested in `tests/test_recruit.py`). Output `results/regating_leak_amend2.json`, written after
the gate and after each task; every draw, mean, sd, ratio and bin is printed to `results/large/full.log`.

**Not run.** A sign-preserving shuffle (positive values among positive slots); a shuffle across matrices; the
mean-background version (m₀ + s_perm); bins of the clipped pattern.

## Amendment 3 (2026-10-21, coding agent; earlier predictions unchanged)

**Why.** The user, 2026-10-21 (`reports/decisions/2026-10-21_decision_mechanism-reading.md`): the mechanism reading is the leading
hypothesis, not a finding; test it more aggressively. H: the switching effect at the trigger is carried by one change
to Γ's values that is (a) the same for every input, (b) specific to which subcomponents receive it, (c) enough on its
own (the input-varying part of the updated model's values adds nothing; a leak is the failure of (c)). Every scored arm
below is one H can fail. Amendment 2 (job 28302) found that shuffling the whole change kills recovery (ratio −0.026
to +0.003), and that its unclipped bins give the turned-down entries (s ≤ −0.1) as large a share as s > 0.3.

**Design check** (`/design`, approved class 4: the main line's named next step "Q5: mechanism or leak", existing
model, tasks and harness; the user is told in one line).
- *Goal.* `GOALS.md` → "Where an update goes", the "reveal" reading: it stands only if the switching is a fixed,
  subcomponent-specific change and not a side channel.
- *Nearest tried.* E18 (shared, mean, cross, random S_on), Amendment 1 (clip / variation / dead), Amendment 2 (whole
  shuffle, bins). None removed the fixed part while keeping the input-varying part, none gave a window another
  window's values, and Amendment 2's shuffle and bins could not tell a switch-off from a sign flip (negative values
  moved onto off subcomponents vanish under the clip and flip the sign without it).
- *Patch or step.* A step: the tests the decision report says are missing before the reading can be backed.
- *Working decisions.* As E18 (Γ frozen, do-nothing placement). No departure.
- *What would change what we do.* All pass: the mechanism reading is backed on this testbed, and the overlap with the
  model's natural machinery is redone on the entries that carry it. P1 or P3 fails: a leak, stop and tell the user (the
  log's Q5 is restated). P2 fails alone: the values must fit their window but the gain does not travel with the donor,
  so (a) is narrowed to "fixed up to the context". P4 fails: the turned-down half is a generic turn-down, (b) holds
  for the turned-up half only.
- *Control first.* G1 as E18, then trig, shared and noclip repeated to 1e-6 against Amendment 2 on every task and seed;
  a fail stops the job before any new arm.

**Arms** (notation: g₀(x), g₁(x) W₀'s and the state's trigger gates of held-out window x; m₀, m₁ their training means;
s = m₁ − m₀ on alive subcomponents; s⁺, s⁻ its positive and negative entries; R = ρ ÷ ρ_trig; W₀'s gates elsewhere):

| arm | pattern at the trigger | no update (g₁ = g₀) | H | leak / H fails | fails |
|---|---|---|---|---|---|
| resid (P1) | clip(g₁(x) − s) | 0 | R ≤ 0.15 | R ≥ 0.3 | (c) |
| mismatch (P2) | g₁(π(x)), π a derangement, 5 draws | mismatch_W0 = g₀(π(x)) | R ≥ 0.8 | R ≤ 0.5 | (a) |
| donor test (P3) | per window: gain b(x) under g₁(π(x)) vs a(x) own, a(π(x)) donor's | c_null ≈ 0 | c_own ≥ c_donor | c_donor − c_own ≥ 0.2 | (a)/(c) |
| down_pool vs down_perm (P4) | clip(g₀ + s⁻) on the pool P (alive, m₀ ≥ 0.1) vs s⁻ permuted within P, 20 draws | 0 | ratio ≤ 0.2 | ratio ≥ 0.5 | (b), turned-down |
| up vs up_perm (P5) | clip(g₀ + s⁺) vs s⁺ permuted over alive, 20 draws | 0 | ratio ≤ 0.2 | ratio ≥ 0.5 | (b), turned-up |
| reported | resid_noclip g₁ − s; resid_held clip(g₁ − s_held); mean, mean_W0; down, down_noclip, up_noclip; Pile | 0 (mean_W0 not) | — | — | — |

- *The donor test* is the sharpest leak test: a leak that carries "how strongly to predict the target here" averages
  out of the mismatch level, but under a leak the masked gain of window x then follows the donor's own gain a(π(x)),
  under H it follows x's own context, a(x). The level (P2) is scored only where mismatching alone costs little
  (|R_mismatch_W0| ≤ 0.2).
- *The turned-down half* is scored with the clip (lead's note): Amendment 2's unclipped bin let values go below 0, and
  a negative gate flips a subcomponent's sign. **survival** = R_down ÷ R_down_noclip says how much of that bin is a
  switch-off rather than a sign flip (reported; guess 0.5 to 0.9). Its shuffle keeps the sign and stays on the
  subcomponents that are on enough to be turned down (every s ≤ −m₀, so all s ≤ −0.1 lie in P); n_down_strong ÷ n_pool
  gives its power. On r1_t534 the 0.1 to 0.3 bin recovered as much as s > 0.3 (0.11 to 0.12, Amendment 2), so the
  turned-up half is all of s⁺, not S_on.
- *Additivity (candidate 5).* The reference subtraction is moot for the no-clip claim (g₀(x) + s has the reference
  g₀(x), exactly 0) and for every scored arm here except the mismatch level, whose reference is reported and gates the
  score. What it affects is the "net" mean-pattern numbers (0.70 to 0.86). resid_noclip fills the last cell of the
  grid {m₀, m₁} × {no variation, W₀'s variation, the state's variation}, so the effect of s is read at three
  backgrounds: none (R_mean − R_mean_W0: 0.81, 0.70, 0.85), W₀'s variation (R_noclip: 0.81, 0.51, 0.78) and the
  state's (1 − R_resid_noclip, new). If the three agree within 0.1, the effects add and the net numbers stand; if not,
  only the reference-free numbers are quoted. Reported, one extra pass per state.
- *Out of context (candidate 4)*, reported: clip(g₀ + s) at position `half` of 64 Pile rows; the target's masked
  log-odds change in nats as a share of the same pattern's change at the trigger (guess ≤ 0.3: the change means
  "predict the target here", not "predict the target"). Kept because it costs one route pass per task; it cannot fail
  H, so it is not scored.

**Cut.** The clipped/unclipped residual pair is not doubled for the mismatch (one version, the own values, is
physical). A derangement of the *change* g₀(x) + Δg(π(x)) (reference-free, but its reading under H depends on the
non-additivity Amendment 1 found). A planted-leak positive control for the donor test (no pattern known to carry a
per-window strength; the test's chance level, c_null, and the gain's spread are reported instead). A sign-kept shuffle
across matrices.

**Blind to.** Trigger position (Pile aside) and held-out windows only; the clip in resid removes negative remainders
(windows that switched less than average); shuffles keep each matrix's values, so the split across matrices is not
tested; one model, three tasks.

**Noise.** Seeds agree within 0.02 on R; Amendment 2's shuffle sd 0.02 to 0.04; a correlation over 73 to 106 windows
has a chance spread of about 0.1 (c_null reported per draw).

**Predictions** (`results/predictions_amend3.json`, filed before the run): P1 H on every state (medium on emoticon,
medium-low on the word tasks); P2 H (medium); P3 no state reads "donor" (medium-high); P4 identity (medium-low); P5
identity (high).

**Cost.** Per state: 3 gate passes, 5 core, 12 for the mismatch, 45 for the halves; 6 states about 400 masked passes at
about 0.2 s, plus three Pile route passes and the gate stages (about 25 s): 2.5 to 4 min, `--time 00:15:00`, cluster
debug, `cluster/regating_leak_amend3.sbatch`. $0.

**Code.** `run.py --amend3` (`a3_gate`, `a3_core`, `a3_mismatch`, `a3_halves`, `a3_pile`, `a3_state`, `a3_score`),
`recruit.plus`, `derangement`, `permuted`, `signed`, `on_pool`, `local_faith.masked_lo_windows` (tested in
`tests/test_recruit.py`). Output `results/regating_leak_amend3.json`, written after the gate and after each task;
every number and draw printed to `results/large/full.log`. The log also checks the per-window gains against masked_lo's
ρ_trig.

**Not run.** A planted-leak control; the change's split across matrices; the per-step version along training; other
positions of the task windows (Pile rows stand in).

## Amendment 4 (2026-10-22, lead session; earlier predictions unchanged)

**Restatement (user's choice B, 2026-10-22).** Amendment 3's filed test of part (c) failed: each window's own values
with the fixed change removed (the remainder) still recover 0.16 to 0.31 of the switching effect. Does that remainder
carry anything specific to its window? Give each window another window's remainder. If it recovers about as much, the
remainder carries no window information, and the failure reads as a property of the remainder's level (W₀'s mean gates
alone recover 0.10 to 0.29), not a leak. If it recovers much less, the window's own remainder matters: a partial leak.

**Design check.** Goal: Q5 (mechanism or leak), the main line. Class 4 (the main line's named next step: the
separating number Amendment 3 named), and the user's explicit choice. Null: D ≤ 0.5. Filed A3_P1 stays failed
whatever this reads.

**Run.** `run.py --amend4`: per task, G1 and Amendment 3's gate (Amendment 2's numbers and Amendment 3's resid,
resid_noclip, mean_W0 reproduced within 1e-6) first; then clip(g1(π(x)) − s) and the unclipped version over the 5
derangements Amendment 3 used. D = R(donor remainder) / R(own remainder). Prediction A4_P1 in
`results/predictions_amend4.json`: D ≥ 0.8 on every state.

| number | W₀ | expected | reference |
|---|---|---|---|
| R(own remainder) | 0 | 0.16 / 0.30–0.31 / 0.27 (A3) | W₀ mean gates 0.10 / 0.24 / 0.27–0.29 |
| R(donor remainder) | 0 | ≈ own if no window information | W₀'s gates mismatched 0.01–0.06 |
| D | — | ≥ 0.8 | 1 = no window information |

**Cost.** About 3 to 5 min on an L40, cluster `debug`, $0.

**Not run.** The remainder with the fixed change present; finer donor choices (nearest-neighbour windows); other tasks.
