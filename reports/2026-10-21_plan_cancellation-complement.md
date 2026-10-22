# Plan: does the adversary's excess come from the original subcomponents? The complement revert (E20, `experiments/cancellation_complement`)

*2026-10-21, coding agent. The named next step of `reports/2026-10-21_analysis_cancellation.md` → "What next" (both
items) and of the research log's E17 card: approved classes 1 (GPU passes of the existing hook framework on saved
states) and 4. No sign-off needed; the check is written so the record shows why. RunPod is down today, so the run goes
to the cluster's `debug` QoS. Vocabulary: `GOALS.md` → Terms.*

## Restatement

The project asks whether a small post-training update can be carried into a VPD decomposition by a simple map
(component training or a split) so that the decomposition stays faithful (the weak version). Every map that keeps the
fine-tune's behaviour fails VPD's adversarial check. E17 removed the adversary's fraction of each subcomponent's
*change* Δ_c = u′_c v′_cᵀ − u_c v_cᵀ and found it carries 1 to 17% of the adversary's excess X; it inferred that the rest
comes from the adversary's fraction of the *original* subcomponents u_c v_cᵀ, on which the updated model depends more
than W₀ does. This run measures that: it removes the adversary's fraction of each original subcomponent with the changes
kept, at each state and at W₀, splits X into four exact parts, and bins the original-subcomponent part by layer and by
W₀'s importance.

## The design check

**Goal served.** `GOALS.md`, "If no simple Φ works": new capability, or a brittle decomposition? E17 ruled out
cancellation among the changes. If the excess sits in the original subcomponents, the failure is that the importance
box Γ draws at the new activations leaves maskable subcomponents the updated model now reads, which is a statement
about Γ (the re-gating line, Q5) rather than about how the map writes the update. The importance bins answer the
question E17 left: which low-importance subcomponents the updated model leans on.

**Nearest thing tried.** E17 (`TRIED.md` E17): the same hooks, the same nine states, the same 128 rows, removing the
changes only. E15 stage 0: single subcomponents the adversary turns down are inert one at a time at both models, and
masks do not transfer between state and W₀. This run measures the remainder E17 inferred and the joint effect E15
could not localise.

**Patch or step.** A step: it finishes the account of where the adversary's damage lives, which decides whether the
next maps should constrain placement (excess in the changes or their interaction) or whether Γ must see the new
activations (excess in the originals).

**Working decisions.** Γ frozen; functional measures (KL) only. All states are Adam updates, so per `GOALS.md`'s
optimiser subgoal no result here is a property of the operation rather than of Adam. No departure.

**What would change what we do next.** O ≥ half of X on the failing states: the box is stale around the original
subcomponents; E18 (mechanism or leak) and a Γ that sees the state are the next tests, and the importance bins name
which subcomponents a refit Γ must protect. O small with I large: the damage needs both halves removed together, so
the changes and originals must be read as one piece (the map's placement matters again). R large: the dead
subcomponents or Δ′ carry it, and the alive set or Δ's handling is the next thing to examine.

**The apparatus arm, first.** W₀ runs first and alone. Gate G1 (closure), on the first batch at W₀ and at every
state: the hooks removing the adversary's fraction (1 − g_c)(1 − a_c) of the alive new pieces (originals plus
changes) and of the dead subcomponents, and (1 − a_Δ) of Δ′, reproduce the adversary's own masked forward run in fp32,
KL ≤ 1e-4 nats; at zero weight the hooks give KL ≤ 1e-6 to the state. This is what makes X close exactly. At W₀ also
F_adv ≤ 1e-6 and |F_both − F_orig| ≤ 1e-6 (the changes are zero there, a code check). Any failure stops the run.
Then the positive control C0 on component training's emoticon state, before any scored state: F_adv and the bf16
adversary KL reproduce E17's (0.00149 and 0.7286) within 25%; a larger gap means the code path or the rows differ
from E17 and the run stops. Nothing here reproduces a training procedure: the adversary is E17's (`gamma.pgd_sources`,
20 sign steps of 0.1, seed 0, all 128 rows), recomputed on the same card type; its run-to-run drift is what C0 reads.

## The numbers, and what each should read

Per state s and at W₀, on the first 128 Pile validation rows (batches of 8), every forward in fp32. w_c(x) =
(1 − g_c(x))(1 − a_c) is the fraction the state's own adversary removes from alive subcomponent c (W₀'s adversary at W₀).

| number | definition | W₀ | adjoint split (passes the adversary) | failing states (mechanism reading) |
|---|---|---|---|---|
| F_all | KL(state ‖ the adversary's masked forward), fp32 | ≈ 0.66 (E17's bf16 value) | ≈ 0.69 | 1.2 to 4.4 (E17) |
| X | F_all(s) − F_all(W₀) | 0 | ≈ 0.03 | 0.54, 0.87, 3.79, 0.62 (r0 t352, r1 t534, r2 t275, least-squares) |
| F_orig | KL(state ‖ state with w of each alive original u_c v_cᵀ removed, changes kept) | most of F_all(W₀), ≥ 0.5 | ≈ F_orig(W₀) | ≥ F_orig(W₀) + 0.5 X |
| O | F_orig(s) − F_orig(W₀) | 0 | within ±0.1 | ≥ 0.5 X (P1) |
| F_adv | w of each change Δ_c removed, originals kept (E17's) | ≤ 1e-6 (code check) | 0.0008 (E17) | 0.027, 0.034, 0.26, 0.008 (E17) |
| I | F_both − F_orig − F_adv, F_both removing w of each new piece u′_c v′_cᵀ | 0 by construction | ≈ 0 | \|I\| ≤ 0.3 X |
| R | [F_all − F_both](s) − [F_all − F_orig](W₀): the dead subcomponents, Δ′ and their interaction with the alive removal | 0 by construction | ≈ 0 | ≤ 0.25 X |
| F_orig_perm | F_orig with a permuted among the alive subcomponents of each matrix, 3 draws | ≤ 0.5 F_orig(W₀) | ≤ 0.5 F_orig | F_orig ≥ 2 × mean F_orig_perm (P2) |
| e_b | F_orig_b(s) − F_orig_b(W₀), F_orig with only bin b's originals removed; bins: 4 equal-count quartiles of W₀'s mean importance (imp0 lowest) and 4 layers | 0 | within ±0.05 | imp0 + imp1 hold ≥ half of the positive e_b (P3) |
| removed_b | mean w over positions and subcomponents in bin b | imp0 highest (≥ 0.5), imp3 lowest | as W₀ ± 0.1 | as W₀ ± 0.1 per bin |

X = O + F_adv + I + R holds exactly by these definitions (F_both(W₀) = F_orig(W₀)); G1 is what makes F_all the hooks'
total, so the four parts are each a measured revert except R. The do-nothing reference is not among E17's states and
is not run; there F_adv = I = 0 by construction and the excess would split between O and R.

**Blind to.** Pile rows only, never the trigger. KL is not additive: I and R absorb the non-linearity, and the per-bin
reverts do not sum to F_orig (cross-bin interaction is not read). The importance bins use W₀'s mean g from round 1,
not the state's g. R lumps the dead subcomponents, Δ′ and their interaction with the alive removal; it is not a
separate revert. Each state is read under its own adversary, and W₀ under W₀'s, so O compares two different masks:
that is the quantity X is made of, not a fixed-mask dependence. One adversary seed.

**Noise.** One adversary seed (E11's seed floor f = 0.075 on the ratio, about 0.05 nats at W₀'s 0.66); the three
permutations give F_orig_perm its spread; C0's reproduction gap measures the adversary's run-to-run drift on the card.
Per-bin differences below 0.05 nats are read as noise.

## States

E17's eight real states, from E11's saved files on the cluster NFS (`experiments/reading_a_coverage/results/large/<task>/`),
rebuilt by `commute.saved_states` exactly as E17 rebuilt them: component training at step 10 on emoticon, r1 t588 and
r2 t359 (pass the adversary in E11) and r0 t352, r1 t534 and r2 t275 (fail); the emoticon least-squares split (fails)
and scaled adjoint split (passes). E17's gauge-mixed control is dropped: it tested the coherence statistic, which this
run does not use. Order: W₀, component training emoticon (C0), the four scored states, then the other three.

## Predictions (filed in `experiments/cancellation_complement/results/predictions.json` before the run)

- **G1 closure** (gate, first at W₀ alone, then per state): as above. Fail: stop.
- **C0 reproduction of E17** (positive control, after W₀, before the scored states): as above. Fail: stop.
- **P1 the excess is in the originals**: O ≥ 0.5 X on each of the four scored states. *Predicted: pass*, moderate
  confidence: F_adv is ≤ 7% of X there and the changes are small beside the originals, so I should be small; the risk
  is R (Δ′ under its own source, the dead subcomponents).
- **P2 the adversary's pattern over the originals matters**: F_orig ≥ 2 × mean F_orig_perm on each scored state.
  *Predicted: pass*: turning every maskable subcomponent down together costs a fiftieth of the adversary's mask
  (`reports/2026-10-20_analysis_adversarial-vs-stochastic.md`).
- **P3 the low-importance originals carry it**: the two lowest importance quartiles hold at least half of the summed
  positive per-bin excess Σ_b max(e_b, 0) on at least three of the four scored states. *Predicted: pass*, low
  confidence: a subcomponent with g near 0 is removable at every position, and minimality made it inert only on W₀'s
  activations.
- Reported, not scored: the decomposition on the passing states and the adjoint (X small, ratios unstable: parts
  reported in nats); per-layer e_b (expected: the largest share in layer 0 or 3, which hold most alive subcomponents);
  removed_b at state and W₀; F_all against the bf16 adversary KL.

## Cost and platform

One L40 on the cluster's `debug` QoS. E17 took 48 s per state (557 s for nine with set-up); this run adds the fp32
masked forward, the gate and about six more hooked passes per state, about 70 s, so about 12 min for W₀ and eight
states, `--time=00:40:00`. No separate smoke: W₀'s gate is the first minute and stops the job. $0.

## Not run (by design)

The do-nothing state; the gauge-mixed control; a second adversary seed; the trigger windows; the state's adversary
applied at W₀ (fixed-mask dependence); a per-matrix revert (only per layer and per importance bin); separate reverts of
the dead subcomponents and of Δ′.

## Hand-off

Analysis: `reports/2026-10-2x_analysis_cancellation-complement.md` through `/writeup`; `TRIED.md` row E20; a
`PENDING.md` line. Code: `src/commute/cancel.py` (complement section), `src/commute/saved_states.py`,
`experiments/cancellation_complement/run.py`, `cluster/cancellation_complement.sbatch`.

## Amendment 1 (2026-10-21): splitting R

*Coding agent, from `reports/2026-10-21_analysis_cancellation-complement.md` → "What next" (user approved the run).
The original predictions stand; the new ones are in `results/predictions_amend1.json`. Code: `run.py amend1`,
`cancel.rest_arm`/`levels`/`all_blocks_arm`, `cluster/cancellation_complement_amend1.sbatch`.*

**Why.** E20 put 75 to 99% of the excess X in the remainder R = [F_all − F_both](s) − [F_all − F_orig](W₀): the
adversary's removal of the dead subcomponents and Δ′, alone or with the alive removal. For component training and the
adjoint state Δ′ is W₀'s Δ and the dead block is unedited, so the question is how unchanged weights come to cost more.
Four candidate readings: (1) the adversary removes more of them; (2) the same removal costs more; (3) the dead
subcomponents carry more; (4) the cost appears only jointly with the alive removal.

**The arms.** Same run as E20 (W₀ first, the same eight states and 128 rows, the same adversary, every E20 number
recomputed), with the alive blocks kept: F_dead (each dead subcomponent's (1 − g)(1 − a) removed), F_dense (1 − a_Δ of
Δ′), F_rest (both), and F_dead_fix, F_dense_fix (the same at W₀'s levels a, with the model's own g). Levels: mean over
matrices of 1 − a_Δ, mean 1 − a over the dead subcomponents, and the dead block's mean removed fraction. The adversary's
sources are saved per model (`results/large/sources/`). Against W₀: D = ΔF_dense, K = ΔF_dead,
P = Δ(F_rest − F_dense − F_dead), J = Δ(F_all − F_both − F_rest) (F_orig for F_both at W₀), so R = D + K + P + J exactly;
D_fix = F_dense_fix(s) − F_dense(W₀), K_fix likewise.

| number | W₀ | (1) removes more | (2) same removal costs more | (3) dead carry more | (4) only jointly |
|---|---|---|---|---|---|
| D, K | 0 | that block ≥ 0.25 R and the larger | D the largest | K the largest | each small |
| D_fix, K_fix | = F_dense, F_dead | ≤ 0.5 of the block's part | D_fix > 0.5 D | K_fix > 0.5 K | small |
| Δ level (1 − a) | 0 | that block's up ≥ 0.1 | Δ′'s within 0.1 | dead within 0.1 (removed fraction may rise through g) | within 0.1 |
| J | 0 | < 0.5 R | < 0.5 R | < 0.5 R | ≥ 0.5 R |

Reading rule per scored state (R > 0): J ≥ 0.5 R → 4; else the largest of D, K, P: P → "dead with Δ′"; D → 1 if
D_fix ≤ 0.5 D else 2; K → 1 if K_fix ≤ 0.5 K else 3.

**Gates.** G1 as E20. G2 per model: F_all, F_both, F_orig within 1e-3 relative of E20's `complement.json`; the hooks
with every block at the adversary's fractions, read through the alive and dead caches, give F_all within 1e-4 (checks
the new dead cache on every batch); at W₀ the fixed-level arms equal the own-level ones within 1e-6. Any failure stops.

**Predictions.** P4: reading 4 on ≥ 3 of 4 scored states (*predicted pass, low confidence*: Δ and the dead
subcomponents are near-inert alone at W₀ and unchanged by component training, and E15 stage 0 saw joint effects only).
P5: the levels move by ≤ 0.1 (both blocks) on ≥ 3 of 4 (*predicted pass*: sign steps drive levels to their bounds, so
W₀'s adversary already removes most of Δ; W₀'s mean 1 − a_Δ ≥ 0.8, reported).

**Blind to.** For the least-squares state Δ′ is not W₀'s Δ, so D there also reads the update moved into Δ′. The fixed
arms hold a, not g. KL is not additive (P, J absorb it). One adversary seed.

**Cost.** E20 took 476 s; five more reverts and one check per model at about 1 s each, and 7.6 GB more GPU memory for
the dead gates: about 9 min on one L40, `debug`, `--time=00:30:00`, $0.

**Not run.** A fixed-level arm for the alive removal; per-matrix reverts of Δ′ (only the per-matrix levels); a second
adversary seed; the trigger windows; the do-nothing state.

## Amendment 2 (2026-10-21): the same mask at the state, and a stronger search

*Writing agent, from `reports/2026-10-21_analysis_cancellation-complement.md` → Amendment 1 → "What next", items 1 and
2. Approved class 4 (the main line's named next step, on saved states and the existing harness); no sign-off. The
earlier predictions stand; the new ones are in `results/predictions_amend2.json`. Code: `run.py amend2_fixed` and
`amend2_strong`; `cancel.masked_kl`, `cancel.alive_whole`; `gamma.pgd_sources(init=...)`. Scripts
`cluster/cancellation_complement_amend2_{fixed,strong}.sbatch`.*

**Restatement.** Amendment 1 found the excess X only where the adversary removes the alive and dead subcomponents
together, and left three readings: (1) a 20-step search lands in different places at W₀ and at the state, and at W₀
it misses a better mask (keeping the alive block whole costs 1.075 against its 0.660); (2) the same mask costs more at
the updated model; (3) a mechanism for (2). This amendment measures the numbers that separate 1 from 2.

### Design check

- **Goal.** `GOALS.md`, "If no simple Φ works": whether the adversarial failure of component training measures the
  updated model (a brittle decomposition, or new capability) or the reference's search. If W₀'s floor is understated,
  every faithfulness verdict in the project reads against the wrong reference.
- **Nearest tried.** E20 Amendment 1 (each block alone and at W₀'s levels, singly); E15 stage 0 (masks do not
  transfer between state and W₀, one subcomponent at a time); E15's two attackers (0.42 vs 0.70 nats). New here: W₀'s
  whole mask at the state, and the search's own ceiling at W₀.
- **Patch or step.** A step: it decides whether X is a property of the model (reading 2: go on to the per-matrix
  overlap, item 3) or of the search (reading 1: reread the verdicts against a stronger reference).
- **Decisions.** Γ frozen; KL only; the authors' adversary form (sign steps of 0.1, one source per matrix) kept, only
  run longer and restarted. No departure.
- **What would change the plan.** Share ≥ 0.5 on ≥ 3 of 4 failing states: reading 2, item 3 next. Share ≤ 0.2:
  the state's adversary found another mask. M(W₀) ≥ 1.0: the floor is understated; X against the strong reference
  decides whether the failing verdicts survive.
- **Control first.** Fixed stage: W₀'s saved levels at W₀ reproduce Amendment 1's 0.660363 within 1e-6 (C1), before
  any state; each state's own saved levels reproduce its Amendment 1 F_all within 1e-4 (G3). Strong stage: at W₀
  first, seed 0's 20-step checkpoint reproduces 0.660 within 1e-3 relative, and the alive-whole start at 0 steps
  reproduces F_rest 1.0754 within 1e-4 (C2); the same two at r1 t534 (1.529, 0.9433). Any failure stops the job. The
  chained search reproduces the reference exactly: sign steps keep no optimiser state, so 20 + 30 steps is 50 steps
  (CPU test on the toy model); the only differences from the reference are the extra steps and starts, which are what
  is measured.

### The numbers, and what each reads

All fp32 masked KL (`cancel.masked_kl`: mask g + (1 − g) a over all C subcomponents, Δ′ under a_Δ), g always the
model's own Γ, on the first 128 Pile validation rows.

| number | definition | W₀ | passing states | failing states (reading 1 / reading 2) |
|---|---|---|---|---|
| F_own | the model's saved 20-step levels | 0.660 | = Amendment 1's F_all | = Amendment 1's F_all |
| F_w0lv | W₀'s saved levels at the model | 0.660 (= F_own) | about 0.66 to 0.82 | near 0.66 / ≥ 0.66 + 0.5 X |
| share | (F_w0lv(s) − 0.660) / X | undefined (X = 0) | unstable (X < 0.16) | ≤ 0.2 / ≥ 0.5 |
| KL at checkpoint | the strong search's fp32 KL, runs below | seed0 20: 0.660; alive_whole 0: 1.075 | not run | r1 t534: seed0 20: 1.529; alive_whole 0: 0.943 |
| M, M_random | max over checkpoints ≥ 20 steps, all runs / random starts | ≥ 1.0 if the floor is understated / about 0.66 to 0.75 | not run | r1 t534: ≥ 1.529 |
| X_strong | M(r1 t534) − M(W₀) | 0 | not run | ≤ 0.6 × 0.868 if the excess is partly the search |

Runs of the strong stage (cumulative sign steps at which the KL is read): seed0 from seed 0's random draw (20, 50,
100); alive_whole from the model's saved levels with every alive level set to 1 (0, 20, 50); seed1 and seed2 (20).

**Blind to.** Pile rows only. The fixed arm holds W₀'s levels a, not its removed fraction (g is the state's), so it
answers "does W₀'s choice of a cost more here", not "does the same removal". For the least-squares state Δ′ is not W₀'s
Δ. The strong search is still sign steps of 0.1 with one source per matrix: M is a lower bound on the maximum, not the
maximum. Two models in the strong stage.

**Noise.** The fixed stage has no search: deterministic up to fp32 order (about 1e-7, Amendment 1's G2). The strong
stage's search noise is the spread of seeds 0 to 2 at 20 steps; E11's seed floor f = 0.075 is about 0.05 nats at W₀.
Thresholds are coarse (halves of X, 1.0 nats against 0.66 and 1.075).

### Predictions (`results/predictions_amend2.json`, filed before either run)

- **C1, G3, C2** gates as above; fail: stop.
- **P6 the same mask costs more** (fixed): share ≥ 0.5 on ≥ 3 of 4 scored states (r0 t352, r1 t534, r2 t275,
  least-squares), relative to each X. *Predicted: fail, low confidence*: E15 stage 0 saw masks fail to transfer, and
  W₀'s dead levels cost less at the failing states (Amendment 1's K_fix < 0).
- **P7 W₀'s floor is understated** (strong): M(W₀) ≥ 1.0 nats, absolute, over checkpoints of ≥ 20 steps.
  *Predicted: pass, moderate*: the alive-whole start sits at 1.075.
- **P8 the excess is partly the search** (strong): X_strong ≤ 0.6 × X20, relative. *Predicted: pass, low*: r1 t534's
  alive-whole start (0.94) is below its own 1.53, so its maximum should rise less than W₀'s.
- Reported: every checkpoint, M_random (expected below 1.0 at W₀), the seeds' spread, seed0's gain from 20 to 100
  steps, the fixed stage on all eight states.

### Cost and platform

Both on one L40, cluster `debug`, $0, run separately (each a short job). Timing from Amendment 1 (job 28251, 531 s for
W₀ and eight states): about 59 s per model, of which the 20 sign steps plus one fp32 cache pass took 33 s (start of a
model to its G1 line in `amend1.log`), so a sign step costs at most 1.65 s.

- `amend2_fixed`: 17 fp32 masked passes (W₀ once, two per state), about 3 s each, 51 s; load 15 s; state rebuild
  about 20 s. About 2 to 3 min; `--time=00:15:00`.
- `amend2_strong`: 100 + 50 + 20 + 20 = 190 sign steps per model, 380 for the two: 380 × 1.65 s = 627 s. Plus 16
  fp32 checkpoint passes × 3 s = 48 s, 12 Γ caches (one per `pgd_sources` call) × 2 s = 24 s, load 15 s: about
  715 s, 12 min. 200 steps on seed 0 would add 100 × 2 × 1.65 = 330 s and pass the 15-min aim, so seed 0 stops at 100;
  the alive-whole run goes second and the extra seeds last, so a time-out loses the least informative runs. Every
  checkpoint is written as it finishes. `--time=00:30:00`.

### Not run

- Item 3 (per-matrix overlap of the alive and dead removals): only if reading 2 survives.
- W₀'s levels per block (alive only, dead only) at the state; W₀'s removed fraction (its g) at the state.
- The strong adversary on the other failing states and on the passing ones; step-size decay or per-row sources.
- A second update seed; the trigger windows; the do-nothing state.

## Amendment 3 (2026-10-22): the verdicts at larger adversary budgets

*Lead session; user's go 2026-10-22 ("go ahead with the budget sweep"). Source: `reports/2026-10-22_review_vpd-adversary-convergence.md`.*

**Restatement.** Amendment 2 found the 20-step adversary far from its maximum, but ran longer searches at a fixed step
of 0.1, against the VPD paper's rule of steps × step size ≈ 2. The paper reads its own decomposition at a fixed budget.
Do the project's verdicts (each a state's adversarial KL divided by W₀'s at the same budget) hold at 40, 80 and 160
steps with the step size at 2/steps?

**Design check.** Goal: every Q1/Q5 verdict rests on this check. Nearest tried: Amendment 2 (two models, fixed step
0.1). Class: 2 (a control/replication of the check every claim uses), user's explicit go. Null: the failing states'
ratios fall to the passing range as the budget grows.

**Run.** `run.py amend3_budget --shard N`: W₀ (seeds 0 and 1) first and alone, then r1 t534 + r0 t352, r2 t275 +
least-squares emoticon, adjoint emoticon (positive control for the passing side), seed 0. Each budget is its own run
from the seed's random start. Gate C3: 20 steps of 2/20 = 0.1 reproduce Amendment 1's F_all within 1e-3 relative, per
model. Predictions P9–P11 in `results/predictions_amend3.json`.

| number | W₀ | do-nothing | control |
|---|---|---|---|
| KL at 20 | 0.660 | — | adjoint emoticon 0.693 (ratio 1.05) |
| ratio at 40–160 | 1 by definition | — | adjoint ≤ 1.241 (P10) |
| failing states' ratio | — | — | r0 1.82, r1 2.32, r2 6.72, ls 1.93 at 20; ≥ 1.5 at every budget (P9) |

**Cost.** 300 sign steps per (model, seed) × ~1.65 s ≈ 8.5 min, plus load and the state rebuild: shard 0 ≈ 18 min,
shards 1–2 ≈ 18 min, shard 3 ≈ 10 min. Cluster `debug`, $0. If the cluster doesn't start a job within 10 minutes,
go to RunPod.

**Not run.** Budgets past 160; more than one seed per updated model; another adversary (per-row sources, restarts,
step decay); the passing states other than the adjoint.
