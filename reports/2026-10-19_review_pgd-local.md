# Review: the task-local PGD numbers are saturated by construction, and retained_pgd has no power

*2026-10-19, coding agent. The PGD hand-off from `reports/2026-10-19_analysis_task-ladder.md` ("Reported, not
scored" → PGD). Sections 1–3 were written from `experiments/task_ladder/results/{ladder,noise}.json` and the code
before any new number was taken; section 4 files the check's expected values before it ran; section 5 is filled in
after. Code: `experiments/task_ladder/check_pgd.py` (read-only, W₀ only, every number is `local_faithfulness` from
`src/commute/local_faith.py` as it ran in the ladder).*

**Takeaway.** Both window adversaries saturate at W₀ because the windows they are shared across are alike by
construction (one trigger token, or one subcomponent switched on), so the "shared" adversary is in effect a
per-position adversary, and the authors' budget (20 sign steps of 0.1) saturates a per-position adversary: 33–53
nats at the trigger whether it is shared across 8 or 73 windows, with the sign of the trigger log-odds set by the
seed (+25 / −49 / −63 on one task). retained_pgd is the difference of two such numbers over a 4–15 nat gain and has
no power (section 3); it stays reported-only. **The filed explanation missed its own number:** the same adversary
on 73 *random* Pile windows reads 1.6–2.1 (filed 3–16 if the position count set the level; 0.7 if the task windows
were nothing special). The level is set by how many distinct contexts one mask must cover (8 documents 7.7, 73
snippets 1.6–2.1, 128 documents 0.7), and the task windows count as about one. Whether the decomposition is fragile
*at the trigger* under a 128-document adversary is untested (section 6).

## 1. Terms

- **Window**: 41 tokens with the trigger at index 20; a task has 73–106 held-out windows (≈ 3000–4300 positions).
- **The adversary** (`local_faith._pgd`, the authors' recipe): one source a ∈ [0, 1]^(C+1) per matrix, shared across
  every window; mask = g + (1 − g)·a, so it can only raise a subcomponent's mask above its CI value g, i.e. switch on
  subcomponents whose CI value is low at that position (and Δ, under a's last entry);
  random init, 20 sign steps of 0.1, each step's gradient summed over all windows. Two objectives: **pgd_all** (mean
  KL over all window positions) and **pgd_trigger** (mean KL at the trigger).
- **retained_v** = (lo_v(W′) − lo_v(W₀)) / (lo_target(W′) − lo_target(W₀)): the share of the target's trigger log-odds
  gain that variant v keeps, each variant against its own W₀ value. The gain is 4–15 nats on the ladder tasks.
- **Pile harness PGD** (`faithfulness.py`): the same adversary shared across 128 Pile rows (65k positions), mean KL.
  0.68–0.72 on W₀ (`reports/2026-10-14_decision_pgd-scope.md`).

## 2. What the W₀ numbers say (correcting the hand-off's range)

At W₀ (`ladder.json` → `tasks.<t>.base`), in task order emoticon, r0_t352, r0_t588, r1_t534, r1_t588, r2_t275,
r2_t359:

| | emoticon | r0_t352 | r0_t588 | r1_t534 | r1_t588 | r2_t275 | r2_t359 |
|---|---|---|---|---|---|---|---|
| pgd_trigger: kl_trigger | 46.5 | 43.8 | 33.3 | 29.8 | 33.3 | 39.1 | 41.8 |
| pgd_trigger: lo_trigger | −48.7 | −51.2 | **+24.7** | −22.5 | −46.4 | −50.1 | −46.1 |
| pgd_all: kl_trigger | 30.0 | 32.3 | 14.8 | 27.2 | 14.8 | **1.2** | 34.1 |
| pgd_all: kl_other | 25.3 | 13.3 | 7.4 | 6.2 | 7.4 | **0.9** | 6.1 |
| pgd_all: lo_trigger | −35.5 | −33.5 | **+16.3** | −22.0 | −21.2 | −12.6 | −40.3 |
| ci_masked: lo_trigger | −5.8 | −3.0 | −1.5 | −5.3 | −3.8 | −10.6 | −6.4 |

(r0_t588 and r1_t588 share windows, so their adversaries are identical.) The hand-off's "−50 to −85" is the range
on the trained states W′; at W₀ the trigger log-odds under PGD is −22 to −51, except r0_t588 where it is **+25**.

Three things are visible without a new run:

- **The sign is arbitrary.** On r0_t588 both adversaries drive P(target) to ≈ 1 (lo +16, +25; KL 15 and 33 nats).
  The objective is KL, so the adversary saturates in whichever direction the shared mask finds first; the log-odds
  of the target token under PGD is not a reading of whether the components carry the target.
- **The search is chaotic at the same seed.** r2_t275's pgd_all got 1.2 nats at the trigger at W₀ while every W′
  state of the same task, same windows, same seed, got 55–66. r0_t352's do-nothing pgd_trigger *failed* (lo +16,
  KL 1.3) while W₀ and the other states reached 43–60 nats. With 20 sign steps of 0.1 from U(0, 1), every source
  entry can reach a bound, so the adversary is a search over which low-CI subcomponents and Δ to switch
  fully on, and the first few steps decide which basin it lands in.
- **The level is not the task's.** pgd_all's kl_other at W₀ spans 0.9–25 across tasks of near-identical window
  count on the same model. The per-8-row Pile adversary of the same recipe read 7.7 (4096 positions) against 0.68
  at 128 rows (`decision_pgd-scope.md`): the level is set by how many positions one source must cover, and 3000–
  4300 window positions sit in the per-8-row regime.

## 3. Does retained_pgd mean anything at a saturated trigger? No.

retained_pgd is the difference of two saturated, sign-indeterminate numbers divided by a 4–15 nat gain:

- r0_t588: W₀ at +24.7, W′ at −65 to −85 → retained_pgd_trigger −12 to −16 on every arm, do-nothing included. It
  reads "lost 12× the gain" for a state whose CI-masked model keeps 0.96–1.07 of it.
- r0_t352 do-nothing: the attack failed at W′ (+16) and succeeded at W₀ (−51) → retained_pgd_trigger **+8.8**.
- The harness-seed spread on one state (`noise.json`, emoticon pull-back step 10): retained_pgd_all −0.67 / −0.95 /
  −0.89 (spread 0.28), retained_pgd_trigger −2.66 / −1.88 / −2.50 (spread 0.78). The floor used for every scored
  share is m = 0.15. In nats (gain 15.05): 4 and 12.

So retained_pgd_all and retained_pgd_trigger carry no information about the components at the trigger. They were
never scored (`predictions.json` scores retained_ci_masked and the route split); they should stay reported-only and
be left out of any future scoring rule. This does not depend on the check below.

## 4. The check: is the saturation the scope of the adversary? (filed before running)

**Claim.** The shared sign-PGD on ~100 windows saturates because one source covers only ~3000–4300 positions
(per-8-row regime), and the trigger objective adds ~100 positions selected to be alike (the same trigger token for
the ladder tasks; the emoticon windows are selected by one subcomponent's CI value and hold several trigger tokens,
checked on the window file). It is a property of
the recipe at this scope, not of the trigger positions.
**Null.** The task windows are special: the released decomposition is fragile at these positions under the same
adversary that reads 0.7 on Pile.

All at W₀, each number a `local_faithfulness` call (4 stochastic draws, 20 steps of 0.1 unless varied):

| set | number | expected under the claim | null |
|---|---|---|---|
| **pile** (73 random 41-token windows from Pile val rows 256..383, two draws) | pgd_all kl_other (≈ mean KL over all positions) | **3–16** (within 2× of the per-8-row Pile 7.7 at 4096 positions) | ≈ 0.7 (the Pile level) |
| pile | pgd_trigger kl_trigger (index 20, unrelated positions) | ≥ 5 nats, below the task windows' 30–46 (coherence helps but is not needed) | < 1 |
| **seeds** (r0_t588, seeds 0, 1, 2) | spread of lo_trigger under each objective | ≥ 3 nats for both (≥ 0.5 × the task's gain of 5.3–6.3; the floor is 0.15 × gain ≈ 0.9) | — |
| seeds | kl_trigger under pgd_trigger, every seed | ≥ 10 nats (saturated whatever the init) | — |
| **steps** (emoticon, r0_t588; pgd_steps 0, 2, 5, 10, 20) | kl_trigger at 0 steps, both objectives | within 2× of the stochastic kl_trigger (0.17–0.34: one shared U(0, 1) draw) | — |
| steps | pgd_trigger kl_trigger at 5 steps | ≥ 10 nats on both tasks | — |
| steps | pgd_all kl_other at 5 steps | ≥ 1 nat (per-8-row Pile read 1.1 at 5 steps) | — |
| **scope** (emoticon, first 8, 16, 32 held-out windows; 73 = all) | pgd_all kl_other against N | non-increasing in N, and N = 8 ≥ 1.5 × N = 73 (25.3) | — |
| scope | pgd_trigger kl_trigger at every N | ≥ 20 nats (the coherent objective saturates at any N) | — |
| steps, 20 at seed 0 | both tasks against `ladder.json` `base` | identical (the forward is deterministic; same recipe) if the GPU is a 4090 again; within 10% otherwise | — |

**Reading rule.** The claim stands if the pile row's pgd_all number is in [3, 16] on both draws and the seeds spread
is ≥ 3 nats. If the pile number is ≈ 0.7 on both draws, the task windows are special and section 5 says so; the
section-3 conclusion stands either way. Everything else is reported.

**What this is blind to.** W₀ only: nothing here reads a trained state, so it cannot say what a non-saturating
adversary would retain. Positive control for the harness: the scope set (fewer windows → larger KL) and the step
curve (0 steps ≈ stochastic) show the adversary is doing what the recipe says.

**Compute.** ~15 `local_faithfulness` calls at ~20 s each plus set-up, one RunPod 4090 (the cluster's start
estimate for a 30-minute debug job was the next day, 17:18): under $1.

## 5. Results

Ran 2026-10-19 on a RunPod RTX 4090 (pod `commute-task_ladder`, id `ytym4dpn0u228t`; 1 minute of GPU, $0.13 for the
session). `experiments/task_ladder/results/pgd_check.json`, log in `results/large/logs/pgd_check.log`. Every set
reproduces the ladder's `base` exactly at 20 steps, seed 0 (same GPU type).

| set | number | filed | observed | as filed? |
|---|---|---|---|---|
| pile | pgd_all kl_other, two draws | 3–16 (null ≈ 0.7) | **1.59, 2.09** | **no**, and not the null either |
| pile | pgd_trigger kl_trigger (index 20) | ≥ 5, below the task windows | 2.85, 3.25 | no (below 5); below the task windows' 33–53: yes |
| seeds (r0_t588) | spread of lo_trigger, seeds 0–2 | ≥ 3 nats | pgd_all +16.3 / −28.7 / −38.7 (55 nats); pgd_trigger +24.7 / −49.2 / −63.3 (88 nats) | yes; the sign flips |
| seeds | kl_trigger under pgd_trigger | ≥ 10 | 33.4, 39.0, 50.7 | yes |
| steps | 0 steps against stochastic kl_trigger | within 2× | emoticon 0.32 vs 0.34; r0_t588 0.17 vs 0.17 | yes |
| steps | pgd_trigger kl_trigger at 5 steps | ≥ 10 | emoticon 12.0; r0_t588 7.7 (27.9 at 10) | half |
| steps | pgd_all kl_other at 5 steps | ≥ 1 | 0.74, 0.46 | no |
| scope (emoticon) | pgd_all kl_other at N = 8, 16, 32, 73 | non-increasing; N = 8 ≥ 1.5 × N = 73 | 12.5, 5.4, 30.6, 25.3 | no: not monotone |
| scope | pgd_trigger kl_trigger at every N | ≥ 20 | 45.9, 42.0, 53.3, 46.5 | yes |

Full step curves (kl_trigger, pgd_all / pgd_trigger): emoticon 0.32/0.32 → 0.89/2.51 → 3.69/12.0 → 15.4/34.7 →
30.0/46.5 at 0, 2, 5, 10, 20 steps; r0_t588 0.17/0.17 → 0.29/0.99 → 0.75/7.68 → 4.09/27.9 → 14.8/33.4.

**Reading.**

- **Not the position count.** 73 random windows are 2993 positions, fewer than the per-8-row Pile adversary's 4096,
  yet read 1.6–2.1 against its 7.7. Ordered by the number of distinct documents one mask must cover, the numbers are
  monotone: 8 documents 7.7, 16 documents 4.0, 73 unrelated snippets 1.6–2.1, 128 documents 0.7. The filed
  section-4 claim is amended to that, not to the position count; the filed range is scored as missed. One confound:
  the Pile points are 512-token rows and the snippets 41-token windows; random windows at N = 8 would separate
  document count from window length (not run).
- **The task windows count as about one context.** The trigger objective reads 42–53 nats whether the adversary is
  shared across 8, 16, 32 or 73 emoticon windows, and 33–51 across seeds on r0_t588; on 73 random windows the same
  objective at the same index reads 2.9–3.3. A shared adversary over windows that are alike by construction is a
  per-position adversary, and the authors' budget saturates a per-position adversary by 10 steps (28–35 nats).
- **The search is chaotic past a few steps.** pgd_all's damage to the other 40 positions is 12.5, 5.4, 30.6, 25.3 at
  N = 8, 16, 32, 73 and 7.4, 5.5, 12.1 across seeds; the trigger log-odds' sign is set by the seed. So no number
  from the 20-step window adversary is a stable reading of the state, which is what section 3 said of retained_pgd.
- **What this does not show.** Nothing here tests whether the decomposition is fragile at the trigger under the
  adversary that reads 0.7 on Pile. That needs the 128-document adversary's masks read at the task windows
  (section 6, option (a)), which was not run.

**Surprises to flag.** The pile control missed its filed range (1.6–2.1 vs 3–16) and the scope set is not monotone;
both change the explanation (distinct contexts, not positions; chaotic search), not the conclusion about
retained_pgd.

## 6. What changes next

- retained_pgd_all / retained_pgd_trigger: reported-only, never scored (section 3).
- **Proposal for a `/design` check, not built here:** an adversary with room to move, so that "retained" is defined.
  Options: (a) the Pile harness's own 128-row adversary read at the task windows' trigger (bounded, comparable to W₀,
  blind to a trigger-specific attack by construction); (b) a budgeted window adversary (fewer steps, or an ℓ1 cap on
  the source). Either is a new measurement and needs the design check against `GOALS.md` before predictions.

## Not run

- Any trained state under any adversary (the emoticon pull-back step-10 and step-30 states are saved in
  `results/large/emoticon/`; the ladder tasks' states were not kept).
- A per-window adversary; a budgeted adversary; the Pile-scope (128-document) adversary read at the trigger (section
  6). After this run that is the one test left of "fragile at the trigger".
- Seeds on any task other than r0_t588; the step curve on the rung-2 tasks; random Pile windows at N = 8 (document
  count against window length).
