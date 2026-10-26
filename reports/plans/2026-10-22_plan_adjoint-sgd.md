# Plan: does the adjoint split of an SGD fine-tune keep behaviour, floor and Δ together? (E31)

*Lead session, 2026-10-22, after E22 (`reports/analyses/2026-10-22_analysis_small-finetune-splits.md`). User: "yes you should pursue
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
  `reports/decisions/2026-10-21_decision_regating-main-line.md` (a map at the floor that keeps the behaviour at steps 100 and 1000).
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

### E31 result (agent, 2026-10-22, evening)

Ran on the MATS cluster in five shards (remote-kernels did not connect; `run.py shard A..E` and `run.py merge` added).
All gates passed. P1 pass: SGD's adjoint state keeps emoticon behaviour on seed 1 (0.57 vs small Adam 0.21 at step
1000). **P2 fail ("neither")**: on r1 and r0 the SGD fine-tune, at the rule's rate 0.03, keeps 0.07 and 0.26 against Adam
3e-5's 0.05 and 0.23; task and rate are confounded. P3a fail (SGD R 0.33-0.63, not above Adam's); P3b pass as filed
(Spearman 0.81, n 30) but 0.62-0.69 within a step. P4 pass (1.00-1.01 to 160 steps). The small Adam budget sweep was
lost to a scoring crash (fixed); its rerun, job 29596, was not brought home. Report:
`reports/analyses/2026-10-22_analysis_adjoint-sgd.md`. Next step proposed there: SGD 0.01 on r1/r0.

## Amendment 1: task or SGD rate? (agent, 2026-10-23)

*User, 2026-10-23: an amendment to E31, not a new experiment. Class 2 (the control a report names: E31's own "What
next" item 1) and asked for by the user directly. `results/predictions.json` stands as filed; the amendment's rules are
in `results/predictions_amend1.json`.*

**In plain words.** E31's adjoint split of a plain-SGD fine-tune kept most of the emoticon behaviour (keep ≈ 0.6 at step
1000, two seeds) but almost none on the two word tasks (0.07, 0.28). The word tasks were trained at SGD 0.03, emoticon
at 0.01, so task and rate are confounded. This amendment fills the other two cells of task × rate: the word tasks at
0.01, and emoticon at 0.03.

### Design check

- **Goal.** Q1, the weak version under reading A: is "split a plain-SGD fine-tune by the adjoint split" a map that keeps
  behaviour with Δ fixed in general, or only on emoticon? The answer decides whether the SGD adjoint line is worth more
  tasks (E31 "What next" 2) or is an emoticon property.
- **Nearest tried.** E31 itself (`TRIED.md` row E31). It settled that the effect is there on emoticon at 0.01 and absent on
  r1/r0 at 0.03; it could not say which of task or rate removed it. Nothing else crosses the two.
- **Patch or step.** A control on E31's reading, not a step past it.
- **Decisions.** Same as E31 (Γ, Δ frozen; reading A; same-job W₀ floor; E31's arm-A recipe). One departure from E31's
  P2 recipe, named: the rate is fixed (0.01 or 0.03), not picked by E4's rule. At 0.01 the word tasks fail E4's
  P ≥ 0.9 at step 10 (0.84), so this is the rate the rule rejected; that is the point.
- **What would change what we do next.** Word tasks keep ≥ 0.5 at 0.01 and emoticon falls at 0.03: the effect is the
  rate (or the update size it sets); next is a rate/size sweep, and E31's P2 failure is a rate artefact. Word tasks stay
  low at 0.01 and emoticon keeps at 0.03: the effect is emoticon's; next is more tasks at 0.01 to see how often it
  appears, and what about emoticon makes it splittable. Mixed: both matter, reported as such.
- **Control first.** In every job: the same-job W₀ floor (`faith_score.w0_check`, stop on fail), then gate A1_G1: the
  10-step SGD fine-tune (the same call as E31's grid; no schedule, seeded batches, so step 10 of the 1000-step run is
  that same model) must read P(target) and KL elsewhere within 0.01 of its earlier reading (r1 0.836 / 0.227, r0 0.843 /
  0.158: E31's grid; emoticon 0.03: 0.995 / 0.037, `sgd_finetune` rate.json), else stop. The r1 job runs first and
  alone; r0 and emoticon start only after its W₀ and A1_G1 lines pass. Differences from the reference end states: none
  in recipe, data, seed or Σ (each task's own training-window moments, as E31); the card is the same (L40).
- **Numbers and expected readings.** Per cell, at step 1000 (steps 10 and 100 reported): the fine-tune's held-out
  P(target) (W₀: small, below 0.05 on r1 and 0.20 on r0; fine-tune ≥ 0.9 expected); keep = adjoint P(target) ÷ the
  fine-tune's own (do-nothing ≈ 1 since Δ carries δW; E31 references: emoticon 0.01 0.604, r1 0.03 0.067, r0 0.03 0.278;
  Adam 3e-5 0.10 / 0.05 / 0.25); the adversarial verdict (W₀ ratio 1, floor 0.209; every E31 adjoint state 0.93–1.08)
  and Δ's share (adjoint 0, positive control that it reads 0). Update size reported, not scored.

### Recipe and cells

E31's `arm()` unchanged (`run.py amend1 <cell>`): arm-A fine-tune of all 24 matrices, batch 32, clip 1, seed 0, 1000
steps, plain SGD, readings at 10/100/1000 through `harness.measure` (do-nothing and adjoint state) and
`split.reproduced_share`. Cells: **r1** (r1_t534_v369, SGD 0.01), **r0** (r0_t352_v310, SGD 0.01), **emo** (emoticon,
SGD 0.03). With E31's emoticon 0.01 (two seeds) and r1/r0 at 0.03, that is the full 2×2.

**The cross (emo) is included**: it costs one ~22-min job on a free cluster GPU, and without it a "rate" reading from the
word tasks could not be told from "the word tasks need a lower rate"; the emoticon cell tests the rate on the task where
the effect is known to exist.

### Predictions (filed in `results/predictions_amend1.json`)

Per cell, first: fine-tune P(target) < 0.9 at step 1000 → **uninformative** (the fine-tune did not learn; no reading of
the split; reported, not rerun inside the amendment). Then Δ share > 1e-3 or verdict not "pass" → **not faithful**.
- **A1 (word tasks at 0.01)**, per task against its own 0.03 keep: **rate** if keep ≥ 0.5 and ≥ the 0.03 keep + 0.2;
  **task** if keep ≤ the 0.03 keep + 0.1 (r1 ≤ 0.167, r0 ≤ 0.378); else **partial**. Both tasks agree → that reading;
  disagree → **mixed**; one uninformative → unscored. Predicted: **task**, medium-low confidence (SGD at a third of
  the rate roughly follows the 0.03 path to step ~330, where E31's keep interpolates to ~0.16 and ~0.38; at matched size
  E31 already read 0.26 and 0.49 against emoticon's 0.60).
- **A2 (emoticon at 0.03)**, against emoticon 0.01 seed 0's 0.604: **rate** if keep ≤ 0.404; **task** if keep ≥ 0.5;
  else **partial**. Predicted: **task**, low confidence (≈ 0.01 to step 3000; emoticon's keep falls slowly, 0.72 / 0.71 /
  0.60).
- Joint reading (not a prediction): task/task → emoticon property; rate/rate → rate or size; otherwise both matter.
- Blind to: one seed per cell (emoticon's seed-to-seed keep difference is 0.04); the path argument itself (no matched
  lr × steps checkpoint); P only at the trigger; damage the 20-step adversary misses. Positive controls: A1_G1, C_delta,
  E31's low Adam 3e-5 keep on each task.

### Cost and route

Each job ≈ 22 min on an L40 (E31 job C: W₀ 340 s, one arm ≈ 920 s), so ≤ 30 min → MATS cluster `debug`, $0, three
jobs (`cluster/adjoint_sgd_amend1.sbatch`, `CELL=r1|r0|emo`), r1 first. RunPod fallback (one 4090, a fresh pod running
the three cells in sequence, ≈ 1 h incl. set-up, < $2) only if r1 does not start within 10 min. The Codex review was
not run: no measurement code changed (the cells call E31's `arm()`, `harness.measure`, `split`); the new code is the
gate, the CLI and the scoring, checked on E31's relabelled rows on CPU, and the run is $0 on the cluster.

### Not run

More seeds per cell; a matched lr × steps checkpoint (0.03 at step ~333 against 0.01 at 1000); other rates; Adam
on the word tasks at a smaller rate; more tasks.

### Runner brief

```
Experiment: adjoint_sgd (E31 amendment 1)   Run: scored (three cluster jobs, r1 first and alone)
Command: on the controller, from /mnt/nw/home/e.agnew/commute after `cluster/sync.sh up` from the laptop:
         sbatch --export=ALL,CELL=r1 cluster/adjoint_sgd_amend1.sbatch
         when its log (cluster/logs/commute-adjoint-sgd-a1-<id>.out) shows the [W0] line with "pass": true and the
         "[gate A1_G1 r1]" line with "ok": true: sbatch the same with CELL=r0 and CELL=emo.
         After all three: `cluster/sync.sh down`, then on the laptop `uv run python experiments/adjoint_sgd/run.py amend1_merge`.
Expected: ~22 min per job (W0 ~6 min, gate seconds, the 1000-step fine-tune and six harness readings ~15 min); GPU < 20 GB;
          CPU RAM < 32 GB. Wall ≈ 50 min (r1, then r0 and emo in parallel)
Outputs: results to bring home: experiments/adjoint_sgd/results/adjoint_sgd_amend1_{r1,r0,emo}.json (written after every
         step; small), logs experiments/adjoint_sgd/results/large/full_amend1_{r1,r0,emo}.log and
         cluster/logs/commute-adjoint-sgd-a1-<id>.out; then results/adjoint_sgd_amend1.json from amend1_merge (laptop, JSON only)
Inputs not in git: experiments/transport_test/results/{windows.json, large/alive.pt, large/windows.pt},
         experiments/task_ladder/results/large/windows_{r0_t352_v310,r1_t534_v369}.pt, artifacts/ (all already on NFS
         from E31; the sbatch checks them). results/predictions_amend1.json is filed in git; sync it up.
Platform: ≤ 30 min → cluster debug (account c11, partition compute; ssh e.agnew@185.141.218.207); RunPod (one 4090,
          pod commute-adjoint-sgd-a1, the three cells in sequence with run.py amend1 <cell>) only if r1 has not started
          within 10 min of submission
Pod: fresh (only on the RunPod fallback)
Budget: $0 on the cluster; RunPod fallback cap $5
Watch for: stop at once and report if r1's [W0] fails or A1_G1 reads ok false (do not submit r0/emo). Copy from each log
          the "[gate A1_G1 <cell>]" line and the three "[<task>/s0/sgd_<lr> step N]" lines (ft P, keep, pgd verdict, Delta).
          A fine-tune P below 0.9 at step 1000 is a valid ("uninformative") result, not a failure: do not rerun.
          Record job IDs; scancel anything idle.
```

## Amendment 2: a second training seed for the single-seed cells (agent, 2026-10-23)

*User, 2026-10-23: a replication, design class 2 (more seeds of runs that already happened). `results/predictions.json`
and `results/predictions_amend1.json` stand as filed; this amendment's rules are in `results/predictions_amend2.json`.*

**In plain words.** Amendment 1's reading, that the adjoint split keeps less as the SGD update grows and that at one
update size the tasks keep very different amounts, rests on one training seed per new cell. This amendment reruns the
five single-seed cells with a second batch order, so each reading has two seeds behind it.

### Design check

- **Goal.** Q1, the weak version under reading A: whether the adjoint split of a plain-SGD fine-tune keeps behaviour only
  while the update is small, by a task-dependent margin. A second seed decides whether that reading is a result or noise.
- **Nearest tried.** E31 and its amendment 1 (`TRIED.md` row E31). They settled the seed-0 numbers; the only seed spread
  measured is emoticon at SGD 0.01 (0.604 and 0.567). Nothing new beyond the spread, which is the point.
- **Patch or step.** A control on amendment 1's reading, not a step past it.
- **Decisions.** Same as E31 and amendment 1 (Γ and Δ frozen; reading A; same-job W₀ floor; E31's arm-A recipe; fixed
  rates 0.01 and 0.03). No departure.
- **What would change what we do next.** All five replicate and the orderings hold: amendment 1's reading stands, and the
  next step is its "What next" item 1 (a size-matched pair of rates) or 2 (more tasks). A cell "differs": the reading at
  that cell is noise, and the size-matched comparison needs more than two seeds per point. r0 or r1 "overlap" in B2: the
  within-task fall with size holds only for emoticon.
- **Control first.** In every job: the same-job W₀ floor (`faith_score.w0_check`, stop on fail), then gate **A2_G1**: the
  cell's 10-step **seed-0** fine-tune is rerun and read four ways against seed 0's known step-10 row, fine-tune P(target)
  and KL elsewhere (within 0.01), the reproduced share R (within 0.01) and the adjoint state's held-out P(target) (within
  0.02; `harness.measure` without the Pile adversary, a few seconds). A miss stops the job. This is stronger than
  amendment 1's A1_G1, which read only the fine-tune: it also checks Σ (each task's training-window moments), the split and
  the readout that keep is made of, all the parts the seed-1 arm shares with seed 0, for about 30 s. The seed itself
  cannot be checked against anything (that is the measurement). r1_01 runs first and alone; the other four after its G0
  and A2_G1 lines pass. Differences from the seed-0 runs: the batch-order seed only; possibly the card (below).
- **Numbers and expected readings.** Per cell at step 1000 (steps 10 and 100 reported): the fine-tune's P(target) (seed 0:
  0.93–1.00; ≥ 0.9 expected), keep (do-nothing ≈ 1; seed 0: r1 0.03 0.067, r0 0.03 0.278, r1 0.01 0.130, r0 0.01 0.392,
  emoticon 0.03 0.166), the adversarial verdict (W₀ ratio 1; seed-0 adjoint states 0.93–1.33) and Δ's share (adjoint 0,
  positive control that it reads 0), and the update size (seed 0: 0.014–0.015 at 0.01, 0.027–0.030 at 0.03).

### Recipe and cells

E31's `arm()` unchanged, seed 1 (`run.py amend2 <cell>`): arm-A fine-tune of all 24 matrices, batch 32, clip 1, 1000 steps,
plain SGD, readings at 10/100/1000 through `harness.measure` (do-nothing and adjoint state) and `split.reproduced_share`.
The seed sets only the training batch order (`finetune_arm_a`'s generator). Cells: **r1_03**, **r0_03** (E31),
**r1_01**, **r0_01**, **emo_03** (amendment 1). Emoticon 0.01 already has two seeds. The new code is the A2_G1 gate,
the stage and merge wiring, and `score_amend2`; `cell_reading`'s first two rules moved into `cell_status`, shared by both
amendments, with unchanged behaviour.

**Mixed cards are acceptable.** Each cell's adversarial verdict is read against its own job's W₀ floor, and keep is a
ratio of two P(target) readings in the same job, so cells may run on different cards. A1_G1 reproduced E31's L40 step-10
reading on an L40S within 0.001. Each cell's card is recorded in its result file (`device`) and kept by `amend2_merge`.

### Predictions (filed in `results/predictions_amend2.json`, 2026-10-23 17:01)

Per state, first: fine-tune P(target) < 0.9 at step 1000 → **uninformative**; then Δ share > 1e-3 or verdict not "pass" →
**not faithful**; a comparison that needs such a state is **unscored** (reported which), never rerun. Keep is relative to
the fine-tune's own P(target); bands on it are absolute.
- **B1 (seed)**, per cell, |keep(seed 1) − keep(seed 0)|: **replicates** ≤ 0.10, **differs** ≥ 0.20, else **loose**.
  Predicted: all five replicate, medium confidence (emoticon's spread is 0.04; the 0.03 cells fall steeply after step 100,
  where a different path could move keep more).
- **B2 (size, within task)**, per task, the two 0.01 keeps against the two 0.03 keeps at step 1000 (emoticon 0.01 uses
  its existing two seeds). First, every 0.03 state's size must exceed every 0.01 state's, else unscored. **Falls** if
  both 0.01 keeps are above both 0.03 keeps; **rises** if both are below; else **overlap**. Predicted: emoticon falls
  (high confidence: 0.60, 0.57 against 0.17), r0 falls (medium-low: seed-0 gap 0.11), r1 falls (low: gap 0.06; overlap
  nearly as likely).
- **B3 (task order at matched size)**: the six SGD 0.01 step-1000 states, sizes within a factor 1.5 (else unscored).
  **Holds** if both emoticon keeps exceed both r0 keeps and both r0 keeps exceed both r1 keeps; **reversed** if either
  pair is fully reversed; else **overlap** (which pair). Predicted: holds, medium-high confidence (gaps 0.21 and 0.26).
- Blind to: other sources of variation (decomposition, task windows); two seeds give a spread, not a distribution; the
  curve's shape (two sizes per task); size against path; P only at the trigger; damage the 20-step adversary misses.
  Noise floor: emoticon 0.01's 0.04, the only measured spread; this amendment measures five more. Positive controls:
  A2_G1, C_delta, the emoticon pairs already measured.

### Cost and route

Each job ≈ 17–22 min (amendment 1's cells took ≈ 1000 s each on an L40S; E31 job C ≈ 21 min on an L40), so ≤ 30 min →
MATS cluster `debug`, $0, five jobs (`cluster/adjoint_sgd_amend2.sbatch`, `CELL=r1_01|r0_01|r1_03|r0_03|emo_03`),
r1_01 first. The queue was full this morning; if r1_01 has not started within 10 min of submission, cancel it and fall
back to one RunPod card (4090 or faster; amendment 1 used an L40S) running the five cells in sequence, r1_01 first:
≈ 1.5 h with set-up, ≈ $2, cap $5. A mix (some cells on the cluster, the rest on the pod) is acceptable (above).
The Codex adversarial review was not run: no measurement code changed. The cells call E31's `arm()`, `finetune()`,
`harness.measure` and `split` unchanged; the new code is a gate built from those same calls, the CLI and the scoring,
which was checked on CPU against E31's and amendment 1's seed-0 rows relabelled as seed 1 (and with one cell made
uninformative and one made to differ), with no model loaded.

### Not run

A third seed; seed 1 at steps between 100 and 1000; a size-matched pair of rates; other rates; more tasks; Adam seeds on
the word tasks; a pooled update size.

### Runner brief

```
Experiment: adjoint_sgd (E31 amendment 2)   Run: scored (five cluster jobs, r1_01 first and alone)
Command: on the controller, from /mnt/nw/home/e.agnew/commute after `cluster/sync.sh up` from the laptop:
         sbatch --export=ALL,CELL=r1_01 cluster/adjoint_sgd_amend2.sbatch
         when its log (cluster/logs/commute-adjoint-sgd-a2-<id>.out) shows the [W0] line with "pass": true and the
         "[gate A2_G1 r1_01]" line with "ok": true: sbatch the same with CELL=r0_01, CELL=r1_03, CELL=r0_03, CELL=emo_03.
         After all five: `cluster/sync.sh down`, then on the laptop `uv run python experiments/adjoint_sgd/run.py amend2_merge`.
         RunPod fallback: on a fresh pod, `bash cluster/pod_setup.sh`, then `.venv/bin/python experiments/adjoint_sgd/run.py
         amend2 <cell>` for r1_01, r0_01, r1_03, r0_03, emo_03 in that order (stop after r1_01 if its G0 or A2_G1 fails),
         downloading each cell's JSON and log as soon as it finishes, before the next cell's result lands.
Expected: ~17-22 min per job (W0 ~6 min, gate ~30 s, the 1000-step fine-tune and six harness readings ~12-15 min); GPU < 20 GB;
          CPU RAM < 32 GB. Cluster wall ~45 min (r1_01, then four in parallel); RunPod ~1.5 h in sequence incl. set-up
Outputs: results to bring home: experiments/adjoint_sgd/results/adjoint_sgd_amend2_{r1_01,r0_01,r1_03,r0_03,emo_03}.json
         (written after every step; small), logs experiments/adjoint_sgd/results/large/full_amend2_<cell>.log and
         cluster/logs/commute-adjoint-sgd-a2-<id>.out (or the pod's stage logs); then results/adjoint_sgd_amend2.json from
         amend2_merge (laptop, JSON only). Check sha256 of each JSON on both sides before terminating a pod.
Inputs not in git: experiments/transport_test/results/{windows.json, large/alive.pt, large/windows.pt},
         experiments/task_ladder/results/large/windows_{r0_t352_v310,r1_t534_v369}.pt, artifacts/ (already on NFS from
         E31; the sbatch checks them; on a pod sync them with sync(include=[<directory>])). results/predictions_amend2.json
         is new in the tree (not committed); sync it up.
Platform: ≤ 30 min → cluster debug (account c11, partition compute; ssh e.agnew@185.141.218.207); RunPod (one 4090 or
          faster, pod commute-adjoint-sgd-a2, the five cells in sequence) if r1_01 has not started within 10 min of
          submission: scancel it first. Cells may be split across the two (each is read against its own job's W0).
          Record the card for each cell.
Pod: fresh (only on the RunPod fallback); terminate after the last cell's files are home and checked
Budget: $0 on the cluster; RunPod fallback cap $5
Watch for: stop at once and report if r1_01's [W0] fails or A2_G1 reads ok false (do not submit the other cells); copy the
          gate line, which gives the four readings against seed 0's. Copy from each log the "[gate A2_G1 <cell>]" line and
          the three "[<task>/s1/sgd_<lr> step N]" lines (ft P, keep, pgd verdict, Delta). A fine-tune P below 0.9 at
          step 1000 is a valid ("uninformative") result, not a failure: do not rerun. Record job IDs; scancel anything idle.
```

## Amendment 3: the unscaled adjoint split beside the scaled one (agent, 2026-10-26)

*User, 2026-10-26: "Run some quick experiments comparing scaled and unscaled as an amendment to the relevant
experiment … we need at least one graph confirming that." This takes option (a) of the research log's pending card
"Score the unscaled adjoint split beside the scaled one?". Earlier prediction files stand; this amendment's rules are in
`results/predictions_amend3.json` (filed 2026-10-26).*

**In plain words.** Every adjoint state so far multiplied each matrix's move by a fitted number c, because the plain
move has the wrong size. This amendment scores the plain move (c = 1) beside the fitted one on the same three emoticon
fine-tunes, to show how much of the scaled split's record comes from fitting c.

**Restated in the Terms' words.** For E31's emoticon seed-0 fine-tunes (SGD 0.01, Adam 3e-6, Adam 3e-5) at steps 10,
100 and 1000, place the update by the adjoint split with Δ fixed, once with c fitted per matrix (the scaled adjoint
split, as in E31) and once with c = 1 (the unscaled adjoint split, weight change K(δW)), and read both states the same way.

### Design check

- **Goal.** Q1, the weak version under reading A: whether the adjoint split is a simple map, or one fitted to each update.
- **Nearest tried.** E31 and E14 scored only the scaled state. E31's saved c (median 0.2–0.6 per matrix, 0.04 to 2.5 over
  all matrices) says the unscaled move is 1.2–2.8 times the fitted size in the median matrix, and up to 18 times. Nothing
  has scored it.
- **Patch or step.** A control on the scaled split's claim, design class 2 (a control the plan's own pending card names).
- **Decisions.** Same as E31 (Γ and Δ frozen; reading A; same-job W₀ floor; E31's arm-A recipe). No departure.
- **What would change what we do next.** If the unscaled split matches the scaled one on all three readings, the scale is a
  detail and the adjoint split is a fixed map. If not, write-ups call the scaled split "fitted to each update", and the
  decision card closes either way.
- **Control first.** In every job, first the same-job W₀ floor (stop on fail). Then gate **A3_G1** at step 10: the scaled
  state rereads E31's row (adjoint P(target) within 0.02, reproduced share within 0.01), and the new KL readout applied to
  the fine-tune itself equals the fine-tune's own `kl_other` (within 1e-4). A miss stops the job. The SGD job runs first and
  alone; the two Adam jobs start after its A3_G1 passes. Differences from E31's end state: none in recipe (same seed, rates,
  windows, Σ); the card may differ, and each job is read against its own floor.
- **Numbers and expected readings.** Per state: keep (held-out P(target) over the fine-tune's own; do-nothing 1; E31's
  scaled keeps SGD 0.74, 0.72, 0.60, Adam 3e-6 0.71, 0.48, 0.24, Adam 3e-5 0.93, 0.57, 0.10); KL elsewhere, the KL from W₀
  to the state's model at held-out non-trigger positions, the fine-tune's `kl_other` readout (W₀ 0; the fine-tune 0.003 to
  0.06; positive control A3_G1); the adversarial ratio and verdict (W₀ 1; scaled 0.95 to 1.08); Δ's share (0 by construction).

### Predictions (filed in `results/predictions_amend3.json`)

Per (fine-tune, step), unscaled against scaled: **D1** keep difference, "same" within 0.05 absolute, else "higher" or
"lower"; **D2** KL elsewhere, "costlier" at 2× or more, "cheaper" at half or less, else "similar"; **D3** the unscaled
verdict. Reading: "scale is a detail" only if every D1 is same, every D2 similar and every D3 pass. Predicted: D2 costlier
at all nine (medium-high); D1 higher or same at most states (low: K is positive semi-definite, so to first order the larger
move pushes the target further), lower at Adam 3e-5 step 1000 the main alternative; D3 pass except perhaps the large Adam
update at 100 and 1000 (low). Blind to: P(target) only at the trigger; KL elsewhere only on emoticon windows; damage the
20-step adversary misses; one seed and one task; any scale between c and 1. Noise floor: reruns repeat within 0.002 in keep;
seed spread 0.04.

### Recipe, cost and route

`experiments/adjoint_sgd/amend3.py <cell>` (cells `sgd`, `small`, `big`), `cluster/adjoint_sgd_amend3.sbatch`. New code:
`split.adjoint_state(..., sigma=None)` for c = 1, the `kl_elsewhere` readout (`finetune.eval_arm_a` on the state's target
weights), the gate and the scoring (checked on CPU on synthetic rows). Each job ≈ 20 min, so the MATS cluster's `debug`
QoS, $0; RunPod fallback (one 4090 or faster, the three cells in sequence, ≈ 1.2 h, ≈ $1.5, cap $4) if the SGD job has
not started within 10 min.

### Not run

A sweep of scales between c and 1; a single global c; the word tasks; the second seed; the least-squares split unscaled
(it has no scale).
