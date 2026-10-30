# Plan (E55 C1b, then C2): refined pragmatic instruments on a truth set the recipe cannot read

**Reuse-lead, 2026-10-30. Plan only; nothing filed, nothing run.** Follows the user's decision of 2026-10-30
(`decisions.md`, E55 after C1: "run 2, then 1"; benchmarking VPD against pragmatic techniques is a priority; refine
the pragmatic techniques so they get a decent chance, drawing on existing work). Parent plan
`reports/plans/2026-10-29_plan_reuse-stage1.md` (Stage C). C1 result `reports/analyses/2026-10-30_analysis_e55-c1.md`.
Words as in `GOALS.md` → Terms and `decisions.md` (reuse-line rows).

## In plain words

C1 found that three cheap methods separate the fine-tunes that reuse the old count from those that learn a new one,
but could not prove it, because every reusing state was also a short or slow fine-tune, and the cut-offs were set by
guesswork. This plan builds a set of fine-tunes in which the recipe says nothing about the label, fixes each method's
cut-off on other fine-tunes before the scored ones are read, and adds one method from the possession literature. Then
VPD is scored on exactly the same states, with the same calibration, and we learn whether a decomposition reads reuse
better than methods that need none.

## 1. The request, restated (the user confirms this)

- **Option 2 first ($0, cluster).** Score the non-VPD instruments (pruning revival, reverse fine-tuning, steering =
  E52's mean-shift baseline, plus one possession probe) on a **new truth set** whose "reuses the old count" and "new
  count" labels are fixed by construction of the fine-tuning task, at matched learning rate, steps and data, so that a
  labeller reading only the recipe (lr, steps, or the size of the finished update δW) scores near chance. Every
  instrument's cut-off is **calibrated** on seeds held out from the scored seeds, filed before any scored state is
  read. The pass rule stays C1's: the binomial bar, and the best recipe-only labeller plus a margin of 2.
- **Option 1 after it reads (C2, gated).** One VPD decomposition of LOW, and VPD's readings of each state's finished
  update δW (the regating line's stand-in ratio, with the route split's re-gating share r beside it, `decisions.md`
  2026-10-30 programme row), calibrated and scored on the same states by the same rule. Runs only after the user's
  go-ahead, with the six gates of `.claude/CLAUDE.md` → Gated runs if it costs over $15 or runs over 3 h.
- Reuse keeps its filed meaning: the fine-tuned map is g∘f_C with g bounded (works on held-out values; departure from
  Jain et al. Def 2, `decisions.md` 2026-10-30), and the behaviour and patch checks of Stage A are the arbiter.

## 2. Design check (`/design`)

- **Goal served.** `GOALS.md` "If no simple Φ works": is post-training new capability or strengthening? The reuse
  line gives that question a decomposition-free ground truth and scores instruments on it; this plan makes the truth
  unreadable from the recipe and the instruments fairly tuned, which is what a benchmark of VPD against pragmatic
  methods needs.
- **Nearest tried.** E55 C1 (`TRIED.md`): on a recipe-confounded set, pruning 20/26 and steering 21/27 clear chance
  but not the lr labeller by 2; reverse fine-tuning 17/26; raw readings separate with no overlap (post hoc). This
  plan learns what C1 could not: whether that separation is mechanism (survives a de-confounded set and pre-set
  cut-offs) or recipe.
- **Patch or step.** A patch on C1's two flaws (confounded truth, unfiled cut-offs) that the parent plan's Stage C
  needs before C2; the step is C2 itself.
- **Decisions.** Relies on the reuse-line definition row, the bounded-g departure, the truth-set decision (natural
  rows; no swap-built rows), and the 2026-10-30 "option 2 then 1" row (class 3: a step of a confirmed decision).
  Departures to confirm: (i) a **new fine-tuning target** on the same grammar, "count a plus one under `C b 40` on
  every natural row" (task P below), as the reuse-by-construction side; (ii) one reverse fine-tuning rate for every
  state instead of a tenth of its own; (iii) a possession probe as a fourth instrument.
- **What changes next.** Any instrument passes → C2 scores VPD against it on the same set; VPD passing too is the
  benchmark; VPD failing where a pragmatic method passes is the sceptic's reading and the log says so. No instrument
  passes with calibrated cut-offs → the cheap methods read recipe, and C2 asks whether VPD does better than nothing
  (the user is asked before C2 either way). The checks disagree with construction on ≥ 3 scored-seed states → the
  ground truth, not the instruments, is the problem: stop, card to the user.
- **Control first.** Job 0 runs LOW's reproduction, the scratch reference, the pruning control and the positive
  controls (ID states) alone, with stop rules (section 6), before any new fine-tune.
- **Numbers.** Four instrument readings, three nuisance readings, two truth checks per count; each with its value at
  do-nothing, null and positive control (section 6).
- **Class.** Option 2 is class 3 (the user's decision of 2026-10-30) and $0; the three departures above are put to the
  user with this plan. C2 is gated and waits for the go-ahead.

## 3. The truth set: labels fixed by the task, recipe balanced by design

**Two tasks on the same natural texts** (Stage A's text sampler, fresh streams per seed, prefix `C b 40`, answer-only
loss, Stage A's recipe: AdamW, batch 96, 20% warm-up, cosine over 10K steps):

| task | label on every row | reuse by Def 2? | why the label is fixed |
|---|---|---|---|
| **P** ("plus one") | count_a(x) + 1, whatever count b is | reuses count a: g = +1 ∘ f_a, g bounded (Amendment 5: +1 generalises) | count b fits only the 18% of rows with count(b) = count(a)+1, so a state that performs P at ≥ 0.9 can only be computing count a (or a second copy of it; the Stage A caveat on block 5 stands) |
| **N** (Stage A's N1/M recipe, C_Tr = 0) | count_b(x) | new: not through count a | the answer is independent of count a; LOW's count-b readout is weak (PA4: margin 0.277 vs bar 0.260) and never output, so "new" at the level of what the model does (elicitation-granularity caveat, user 2026-10-29) |

**The grid.** lr ∈ {1e-4, 1e-5} × saves at {1K, 10K} × tasks {P, N} × seeds. Seeds 0-1 are the **calibration set**;
seeds 2-4 the **scored set**. Existing runs fill three cells of N's calibration seeds (N1 = seed 0 and N1s1 = seed 1 at
1e-4; M = seed 0 at 1e-5); the rest are new runs: P at both lrs, seeds 0-4 (10 runs), N at 1e-5 seed 1 and N at both
lrs seeds 2-4 (7 runs). 17 runs × ~3 min (Stage A: 10K steps ≈ 3 min on an L40), 34 new states. Each run's stream
is its own (`FT_SEED0 + seed`), so no two scored states share rows; P and N with the same seed share texts but not
labels (stated as paired).

**Verification, not labelling.** Every state is read by Stage A's checks (`a2_bars`, unchanged): behaviour and patch
for count a and count b. A state enters the truth set only if (a) it performs its task on its training distribution at
≥ 0.9, and (b) the checks agree with construction (P → count a "through", count b not "through"; N → count a "not
through"). A state failing (a) is "not yet learned" (expected: N at 1e-5, 1K, Table 6: 33%); a state failing (b) is a
**construction–check disagreement**, reported, unscored, and counted toward the stop rule. Neither is relabelled.

**The recipe ceiling.** Three recipe-only labellers, each with its best threshold and direction chosen in hindsight on
the scored set: lr (slow → existing), steps (short → existing), and update size ‖δW‖_F/‖W₀‖_F (small → existing;
new here, a weight-space nuisance, Shuttleworth et al. 2024 / Hu et al. 2021 §7 read as a size). The ceiling is the
best of the three, n_c. With every cell holding both labels the ceiling is near n/2; if the (1e-5, 1K) N cell is
"not yet learned", the ceiling is at most (n + 3)/2 at the full scored set (P-1e-5-1K's three states are the only
unmatched ones). Target: n_c ≤ 0.6 n, so a perfect instrument beats it by ≥ 8 at n = 21-24. C1's set had
n_c = 19/26 = 0.73 and a best-possible margin of 7.

**Expected n.** 24 scored states if every cell labels; 21 if N at (1e-5, 1K) is not yet learned (P at 10K states
that keep count a under `C a 40` at ≥ 0.95 are void for pruning and reverse fine-tuning only; C1's natural-row reuse
states kept 0.01-0.25, so few are expected).

## 4. Instruments: in and out

Each instrument gives one continuous reading per state; its label is "existing" on one side of a cut-off fixed by the
calibration protocol (section 5). Nothing else in the rule is tuned.

| in | source | reading (per state) | refined how | positive control / null |
|---|---|---|---|---|
| **Pruning revival** | Jain et al. 2024 Alg. 3 (arXiv 2311.12786) | count-a accuracy on `C a 40` after zeroing the K MLP units with the largest grad·weight count-a score, minus the mean over 20 random K-unit removals, best of K ∈ {5, 15} | the 3 sd rule (≈ 3 rows of 4,000; let gains of 0.05 count) replaced by a calibrated cut-off; sites, K and draws as C1 | P calibration states at 10K (C1: 0.35-0.75) / N at 10K (≤ 0.08); LOW void (nothing to revive) |
| **Reverse fine-tuning** | Jain et al. 2024 §4, Fig. 9, 76-77 | steps to 0.95 count-a accuracy on `C a 40`, read every 50 steps on 1K rows, at **one rate, 1e-5, for every state**, cap 5K; divided by scratch's steps (NEVER at 1e-5: 2,300 in C1; void if ≤ 1K or capped) | C1 tied the rate to the state's own lr; at 1e-6 scratch never finished, so half of a cap was a bar every state cleared. One rate makes scratch a real reference and removes the lr from the instrument; the 0.5 rule becomes a calibrated cut-off | P at 10K (C1 at 1e-5: 0.13-0.15) / N at 10K (1e-4-trained: 0.26-0.61; 1e-5-trained: unknown at this rate, the risk) |
| **Steering** | CAA (Rimsky et al. 2023, arXiv 2312.06681) / ActAdd (Turner et al. 2023); E52 B3 as built | gain share S: one mean residual difference at `=` after block l (fit on even rows, block chosen there), added to LOW, share of the state's log-prob gain recovered on odd rows | 0.5 replaced by a calibrated cut-off; a positive control added (ID states) | ID at 2K and P at 10K (C1 reuse: 0.84-0.98) / N at 10K (0.37-0.48); LOW void (G = 0) |
| **Possession probe** (new) | Jain et al. Def 1 as the brief reads it; Hewitt & Liang 2019 control tasks (arXiv 1909.03368); Stage A's probe code | selectivity: held-out accuracy of a linear probe trained on **LOW's** residual at `=` to predict the **state's own answers** on `C b 40` rows, minus the same probe on shuffled labels, best block of 1-6 | the only instrument that reads nothing about the fine-tune but its answers, so it cannot read lr, steps or δW; a calibrated cut-off | P states ≈ LOW's count-a margin (A1: 0.52); LOW itself (answers count a) ≈ 0.52 / N states ≈ LOW's count-b margin (0.28); shuffled labels ≈ 0 |

| out | why |
|---|---|
| CMAP / cross-model activation patching (Prakash et al. 2024, arXiv 2402.14811) | On this model both the old count and any new count are assembled at block 5 (A1: patch share 0.47 at block 4, 0.97 at 5). Patching LOW's activations into a state before block 5 leaves either mechanism intact; at block 5 it tests where the change sits, not whether it reuses. Needs a circuit, which the testbed does not provide. Re-opened on the 67M, where sites are deeper. |
| Crosscoders, diff-SAE, activation-difference lens | Train a dictionary: a second decomposition-type instrument. C2's slot is VPD, the project's instrument; a dictionary is a later comparison, not this plan's. |
| Intruder dimensions (Shuttleworth et al. 2024) as an instrument | On full fine-tunes of a 3M model it reads update size; kept as the **nuisance** it is (update-size labeller). |
| Neuron gain patching | A per-unit steering variant; the mean shift and the pruning score already cover its two readings. |
| Probe-direction ablation | Dropped in Revision 1 of the parent plan (shared number subspace); not reopened. |

## 5. Calibration protocol and scoring rule

**Calibration (filed before any scored state is read).**
1. Job 1 trains all 17 runs and reads the checks on every save (training and checks are label-blind to the
   instruments). Instruments are read on **calibration states only** (seeds 0-1, both tasks, both lrs, both saves:
   ≤ 16 states) plus the controls.
2. Per instrument, with x the reading and "existing" the high side (low side for reverse fine-tuning): if the
   calibration classes separate, cut-off c = midpoint of the gap; else c = the midpoint between adjacent sorted
   readings that maximises balanced accuracy on the calibration set, ties broken toward the median reading. c is a
   number, written with the calibration readings into `results/c1b_cutoffs.json`, dated, committed, and the scored
   job refuses to start without it.
3. Edge cases, run before filing: fewer than 3 non-void calibration readings per class → the instrument is void for
   the stage (reported only); all readings equal → void; a reading exactly at c → "existing" (inclusive, as C1); a
   void calibration reading is dropped, not imputed.
4. The old C1 states (26 + class B) are re-read with the new cut-offs, **described only**: they are confounded and
   partly shared with the calibration seeds.

**Scoring (C1's rule, unchanged except the nuisance set).** Per instrument, over the scored states it labelled
existing or new (void drops the state): agree = labels equal to the truth. Pass iff agree ≥ a(n), the smallest a with
P(Binomial(n, ½) ≥ a) ≤ 0.05 raised to majority + 1 (n = 24 → 17; 21 → 15; 18 → 13; 16 → 12), **and** agree ≥ n_c + 2,
n_c the best recipe-only labeller's hindsight ceiling on the same n states. Per run (one vote, the 10K save; 12 runs)
beside it with margin 1; per cell agreement described. No bar when n < 8. Edge cases: a cell with one label only is
kept (the ceiling absorbs it); a state void for one instrument stays for the others; n_c is recomputed on each
instrument's own non-void set; a tie between labellers takes the larger ceiling.

**Blind to.** All four: a new count b computed after block 5 (inherited from the truth). Pruning and reverse
fine-tuning: a P state that keeps `C a 40` (void). Steering: a state whose gain is under 0.01 (flag, as C1). Probe:
any function LOW encodes but does not use (it is Def 1, necessary not sufficient); if N states read "existing" by it,
that is the finding, not a bug.

## 6. Expected values, controls first, stop rules

| number | do-nothing (LOW, zero update) | null (N at 10K) | positive control | stop if |
|---|---|---|---|---|
| LOW's checks (B_a, P_a on `C a 40`) | 0.99975, 0.960 (Stage A) | — | — | off by > 0.01: apparatus |
| scratch (NEVER, count a, 1e-5) steps to 0.95 | — | — | 2,300 ± 300 (C1) | ≤ 1,000 or capped: reverse void, stop |
| pruning control (LOW, negated score, K 5/15) | drop 0.057 / 0.204 (C1) | — | — | no drop: pruning void |
| probe on LOW, count a / count b margin | 0.52 / 0.28 (A1) | — | — | count-a margin < 0.4: probe void |
| steering S on ID at 2K (seeds 0, 1) | void (G = 0) | — | ≥ 0.8 | S < 0.5 on both: steering has no positive control; report, continue |
| probe selectivity on ID at 2K | 0.52 | — | ≈ 0.52 | < 0.3: probe void |
| P states' task accuracy at 10K, both lrs | — | — | ≥ 0.95 | < 0.9 on ≥ 3 of 10: task P fails, stop |
| checks on P at 10K: count a through | — | — | B_a ≥ 0.9, P_a ≥ hi_P | a P state "not through" or ≥ 3 disagreements: stop, user |
| checks on N at 1K, 1e-4 and 10K, both lrs | — | not through count a | — | an N state "through count a": disagreement, same rule |
| pruning gain, calibration | void | ≤ 0.1 | ≥ 0.3 | overlap → cut-off by balanced accuracy; not a stop |
| reverse ratio, calibration | void (starts ≥ 0.95) | 1e-4-trained ≥ 0.25; 1e-5-trained: unknown | ≤ 0.16 | same |
| steering S, calibration | void | ≤ 0.5 | ≥ 0.8 | same |
| probe selectivity, calibration | 0.52 | ≈ 0.28 | ≈ 0.5 | same |

*Amended 2026-10-30, before any run (code review `reports/reviews/2026-10-30_review_e55-c1b-code.md` L1, lead):* the filed rule in `predictions_c1b.json` (`rules.train_stops`: stop at 3 or more construction–check disagreements among scored-seed states, a P state "not through" counting as one) supersedes the single-state stop in the "checks on P at 10K" row above. Also filed then: a stop when no instrument can pass on the full scored set (max(bar, n_c + 2) > n), and the probe as a task-identity reference, not a scored instrument (review H1, M1).

**Order.** Job 0 (controls, alone, ~10 min) → job 1 (training, checks, calibration readings) → cut-offs committed →
job 2 (scored readings, scores). Stop rows are checked in code at the end of each stage; a stop writes
`results/c1b_stop.json` and exits 1.

## 7. Stages, cost, what is saved, decision tree

All stages on the MATS cluster, `debug` QoS, $0, `--time=00:30:00` with the code's 22-minute deadline; every stage
writes its results as each state finishes and resumes from them; **a job cannot resubmit itself** (compute nodes have
no `sbatch`): on exit 3 it prints the resubmit line and the pod-runner submits the next job by hand.

| stage | work | wall (L40) | jobs | saves |
|---|---|---|---|---|
| 0 controls | LOW checks, scratch 1e-5, pruning control, probes on LOW and ID, steering on ID | ~10 min | 1 | `results/c1b.json` (controls) |
| 1a training | 17 runs × 10K steps, saves at 1K and 10K (+ train state for resume) | ~55 min | 3 | `results/large/c1b/<task>_<lr>_s<seed>-<hash>_<step>.pt`, ~10 MB each |
| 1b checks | B and P for both counts on 34 states, ~1 min each | ~35 min | 2 | `results/c1b.json` (`checks`, truth placement) |
| 1c calibration readings | 4 instruments on ≤ 16 states, ~4 min per state (reverse ≤ 1.5 min at the 5K cap) | ~65 min | 3-4 | `results/c1b.json` (`calibration`) → `results/c1b_cutoffs.json`, committed |
| 2 scored readings | 4 instruments on ≤ 24 states; C1's states re-read | ~100-120 min | 5-6 | `results/c1b.json` (`scored`, `scores`) |

About 14-16 debug jobs, ~5 h of GPU, one to two agent days; nothing over 1 MB outside `results/large/`. Predictions
go in `results/predictions_c1b.json` before job 0; the cut-offs are an appended dated entry, never an edit.

**Decision tree.**
1. Job 0 stop row → stop; numbers to the user.
2. Job 1: task P fails, or ≥ 3 construction–check disagreements among scored-seed states → stop; card to the user
   (the truth by construction is in question). Else cut-offs committed; calibration separation reported per
   instrument (no stop on overlap).
3. Job 2: scores as filed. Any instrument passes → C2 is proposed with its gates and the rehearsal numbers. None
   passes → C2 is still proposed, framed as "does VPD beat methods that read recipe"; the user chooses.
4. C2 runs only after the user's go-ahead.

## 8. C2: VPD on the same truth, labellers and scoring

**What.** One VPD decomposition of LOW with `param-decomp` (pinned 74146b5; minGPT ported to its registry). Per
state, VPD reads the finished update δW: primary reading the regating line's stand-in ratio, with the route split's
re-gating share r beside it (`decisions.md` 2026-10-30, re-gating programme row). Each is calibrated on the **same
calibration states** by section 5's rule (cut-off filed before the scored states are read), and scored by the same
rule against the same ceiling. The outside share of δW (four blocks, Terms) is reported, not scored.

**Gates** (`.claude/CLAUDE.md`, recorded in C2's own plan before anything paid):
1. Pre-mortem: no published VPD hyperparameters for a 3M model (sweep range stated); memory and host RAM of the
   decomposition and of per-state readings; what is saved (decomposition ~10-50 MB; per-state readings) and whether
   it outlives the pod; pod loss and resume; the control (VPD's own faithfulness checks on LOW; readings on LOW
   itself = do-nothing); predictions complete for both readings; code paths never run on a GPU (the port).
2. Every setting sourced: Goodfire's 67M recipe scaled, each departure stated.
3. Rehearsal for $0: a 1K-step decomposition of LOW on `debug`, with saves, a resume and the download; both readings
   run on three calibration states; step time, memory and disk written into the plan.
4. Early checks in code: reconstruction KL and alive count at step 500 against filed values; stop on a miss.
5. Two fresh-agent reviews: the plan against gates 1-4; the code.
6. The user's go-ahead in chat, with the rehearsal's numbers, cost per stage and stop rules.

**Cost.** Parent plan's estimate: port plus one decomposition and one sweep, 2-4 H100-hours, $10-30 on RunPod, up to
2×, 1-2 agent days; gated if over $15 or 3 h. **Alternative to put to the user:** a 3M-parameter decomposition may fit
one ≤ 3 h job on the cluster's `normal` QoS ($0, inside the one-job rule of `decisions.md` 2026-10-30), which the
2026-10-27 "Qwen line only" rule for `normal` would have to be relaxed for; the rehearsal measures whether it fits.

## 9. Departures from the literature

- **Task P is not in Jain et al.** Their wrapper setting (W) keeps count b as a competing solution on every edited
  row; P removes the competitor so the label holds at 10K steps. ID (Amendment 5) is the same move with g = identity.
- **Reverse fine-tuning rate.** Jain et al. reverse below the fine-tune's rate with the Tracr SGD recipe; C1 used a
  tenth of each state's AdamW rate; here one AdamW rate, 1e-5, for every state (equal to the slow states' own rate,
  below the fast states'), because scratch must finish at the rate used and the rate must not encode the recipe.
- **Scratch baseline.** NEVER, which never saw `C a` (slower than the paper's Scr.+FT; a constant bias that the
  calibrated cut-off absorbs, unlike C1's fixed 0.5). Cap 5K, not 10K.
- **Pruning** as C1 (MLP hidden units only; K 5 and 15; 20 random draws; a cut-off instead of a plot).
- **Steering** is CAA/ActAdd's mean difference at one site, applied to the base rather than to the tuned model, and
  scored as a share of the fine-tune's log-prob gain (E52), not as a behaviour rate.
- **Possession probe.** Jain's Def 1 probe is trained on pretraining data for f_C; ours is trained on LOW's
  activations to predict the state's answers, with a shuffled-label control in place of Hewitt & Liang's
  type-consistent control task (counts have no word types), and selectivity is thresholded by calibration.
- **Calibration, nuisance labellers, the hindsight ceiling, the margin, the binomial bar** and the per-run score are
  this project's; the papers plot curves and score nothing.
- **Bounded g, natural rows, saves inside the schedule, the checks and the "neither" class:** as recorded in the
  Stage 1 plan, Amendment 5 and C1.

## 10. Not in this plan; choices for the user

Not here: Stage B (67M arms); swap-built rows; a second base model or grammar; rates other than 1e-4 and 1e-5;
attention-unit pruning; dictionary-type instruments; editing any filed prediction.

Choices: (a) task P as the reuse-by-construction side (or the WN recipe, which drifts to "neither" by 2K-10K and
cannot give late reusing states); (b) one reverse rate for all states; (c) the probe as a fourth instrument; (d) for
C2, RunPod H100 at $10-30 (gated) or a cluster `normal` job at $0 if the rehearsal shows it fits.

## Appendix: apparatus

- **Code.** `experiments/pcfg_anchors/c1b.py`, stages `controls`, `train`, `checks`, `calibrate`, `score`; reuses
  `run.finetune` (saves at 1K and 10K, resume from `<key>_train.pt`), `run.read_state` (the checks and `a2_bars`),
  `revival.prune_readings`, `revival.reverse_finetune` (cap and rate as arguments), `revival.steering`,
  `probe.train_probes` (probe on LOW with the state's answers as labels; shuffled-label control), and
  `revival.score` with a third nuisance labeller and the hindsight ceiling. New in `src/commute/revival.py`:
  `cutoff(readings, labels, high_is_existing)` (section 5 rule) and `ceiling(states, truth)`; both unit-tested on the
  edge cases before filing. Task P is one labeller in `pcfg.py` (`count_a + 1` on every row), no new sampler.
- **State ids.** `c1b/<P|N>_<1e-4|1e-5>_s<seed>/<1000|10000>`; existing N runs keep their a2 ids and are listed in
  the predictions file as calibration seeds. Recipe hash in the file name as Stage A; a changed recipe never resumes.
- **Readings file.** `results/c1b.json`: `controls`, `checks` (per state: B_a, P_a, B_b, P_b, task accuracy,
  placement), `calibration`, `cutoffs` (copied from the committed file at the start of job 2 and compared), `scored`
  (per state × instrument: reading, label, void reason), `nuisance` (lr, steps, ‖δW‖/‖W₀‖ per state), `scores`.
- **Sizes.** 1K eval rows for reverse fine-tuning's criterion, 4K for pruning and behaviour, 1K pairs per patch, 1K
  rows per steering half, 4K probe iterations with 2K held out: Stage A's and C1's sizes, unchanged.
- **Probe cost.** One probe per state per block is 6 probes; with the shuffled control 12, estimated at about
  1 min per state on an L40; job 0 measures it on LOW and ID, and the stage table is corrected from that reading.
- **Hand-off.** The next session writes `predictions_c1b.json` (rules, edge cases run, expected values from
  section 6, leans), `c1b.py`, the sbatch stage lines, gets the fresh-Opus code review, then `/pod-runner` job 0
  alone. Read this plan, `c1.py`, `revival.py`, `predictions_c1.json` and `cluster/pcfg_anchors.sbatch` first.
