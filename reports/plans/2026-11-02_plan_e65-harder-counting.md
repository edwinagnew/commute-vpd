# Plan: with the task held fixed and only reuse varying, which cheap methods still tell reuse from new? (E65)

**Plan only, not approved. No code, no job, no predictions filed.** A new base model and new tasks fall in no approved class,
so this needs the user's confirmation. MATS `debug` only, $0, not gated.
**Reviewed by Fable (2026-11-02): `reports/plans/2026-11-02_advice_e65-review-FABLE.md`. Its named changes are not applied here; they wait on the user's choice between this test and Stage B first.**

## The question, in plain words

On the counting set, "reuses" and "new" were the same as which task a state was trained on, so anything that responds to the task
scored 21 of 21. We now train the same tasks on two base models, each counting one letter and never asked for the other. Each task
reuses on one base and is new on the other.

**Restated in the Terms' words.** A fine-tune reuses when its behaviour runs through a mechanism the base W₀ already has. The causal
check is the label, and every method is scored against it. The held-out reading is not read.

**The answer first.** Crossing removes readers of the task and of the base, and a table wrapper can remove readers of distance. So
cheap methods can differ. VPD cannot win here: if reuse means the answer is a function of the base's answer, a reader of answers alone
ties every cell. Only the route-blocked branch below can reward a reader of mechanism.

## Design check (`/design`)

- **Goal.** `GOALS.md`, new capabilities or strengthened old; the merged line's question of VPD against cheaper methods.
- **Nearest tried.** E55 C1b showed three methods tell two tasks apart, and E55 C2 showed that set cannot rank methods. This asks
  whether those methods read reuse once the task stops giving it away.
- **Patch or step.** A step: it replaces a truth set that cannot separate methods. Stage B on the 67M stays next after it.
- **Decisions.** Relies on the label rule of 2026-11-02, update size as a standing column, and "a run is one Slurm job". Departs from none.
- **What changes next.** The decision tree: each outcome has its own next step, and VPD stays off in all of them.
- **Control.** Stage 0 runs alone on saved models, with stops, before any new base or fine-tune.

## 1. The base without the count

**Recommendation: a mirror pair.** Base A is new, operand prior all on a. Base B is the saved NEVER, operand prior all on b.

- **Why not LOW.** Its 0.1% of b rows may be why the filed probe rule calls count b possessed: margin 0.277, bar 0.260.
  NEVER's count-a margin is about 0.17 on seed 0.
- **Why not drop family C.** The pair-count and index tasks stay near counting, so the count could return unseen.
- **How close elsewhere.** `run.pretrain` and `pretrain_batch` should draw identical texts, families and init, so only the operand
  token and the answers differ. LOW and NEVER ended at losses 0.4369 and 0.4368. Accuracy per family has never been read; Stage 1 reads it.
- **What else the data changes.** The letters are not exchangeable in the grammar. Each base's embedding of the other operand trains only
  from text. Internal layouts differ, so readings shift by base, and only one common cut-off is allowed.
- **Lack by a causal check.** Stage A's patch cannot show it. Its pairs share the other count, so a state answering that count reads 0
  by construction: all 15 "not through" readings in C1b are at most 0.001. So "new" on the old set rested on behaviour.
  Linear probes read a partial other-letter count, 0.64 on LOW and 0.50 on NEVER.
- **The frozen-encoder check.** Freeze blocks 1-5 and train only block 6 and the head. Own letter + 1 must reach 0.95. Other
  letter + 1 must stay at or under 0.7 at 2,000 steps, by when a free fine-tune has passed 0.9.
- **If it re-derives.** In pretraining, the frozen check reads 0.9 or more: stop and tell the user. The fallback is token-only bases, which
  lose the mirror. During the fine-tune, re-deriving is the intended "new", and the register reading below says how it was built.

## 2. Whether truth varies, and the power

**One check, reading no candidate method: the base-to-state patch `T`.** Pairs of texts share the other letter's count and differ in
the task letter's count, 1,000 pairs. Put W₀'s residual at `=` after block 5, on the donor text, into the state on the other text. `T`
is the share of pairs whose output becomes the state's own output on the donor, and differs from its output on that text.

- **Label.** Reuses if task accuracy is at least 0.9 and `T` is above `hi`; new if accuracy is at least 0.9 and `T` is below `lo`; else
  dropped. `interchange.bar` sets the bars from neither-has and own +1 cells on calibration seeds. A cell's truth is the majority of its states.
- **Can it vary?** Own-letter cells can read new if training moves a state off W₀'s residual. C1b's in-state patch fell from 0.91 to
  0.76 with training, so each run is read at its first learned save. Other-letter cells rise only if W₀ silently holds the count.
- **Register reading, unscored.** The same patch with pairs differing in W₀'s own letter, read on new cells. High means a new letter
  selector over the old counter, and the cell stays new only at the level of the letter. State it as a limit.

| rule | edge case | outcome |
|---|---|---|
| cell vote, three scored seeds | a void reading | a wrong vote; C1b dropped voids, which lets a method pass by voiding hard cells |
| cell vote | a tie, or no labelled state | a tie is wrong; a cell with no truth is unscored |
| `revival.pass_bar` | 8 cells, 4 and 4 | 7 of 8, tail 0.035 |
| floor, filed here | 7 or 6 cells; `MIN_N` is 8 | all right; under 6 cells or under 2 per class, no verdict |
| `pass_bar` | truth moves to 6 and 2, or 7 and 1 | bar 7, or 8; a constant labeller cannot pass |
| `pass_bar` | every cell one class | unreachable: Stage 2 stop |
| `revival.cutoff` | all readings equal, or under 3 per class | void: the method fails, it is not skipped |
| label shuffle | complement labelling; direction is filed | 0 of 8; a perfect split ranks 1 in 70 |
| `U`, below | constant answers | `U` is 0 |

**Power.** Seeds of a cell are replicates, so the unit is the cell. Forty scored and calibration states come from eight cells.

- Chance passes 3.5% per method, about 16% over five methods.
- A method wrong on a fifth of states passes 81% of the time; wrong on 30%, it passes 46%.
- "A beats B" needs five cells where A is right and B wrong, sign test 0.031. Only large gaps can be claimed.
- A second table adds four cells, and is the first thing to buy if a verdict sits at the bar.

## 3. Methods, and what reads only the base

Pruning, reverse fine-tuning and steering read W₀ against the state, with W₀'s own behaviour as the old behaviour, so none is told the letter.

- **The control against base readers is the crossing.** Each base holds two reuse and two new cells, as does each letter and wrapper.
  So a reader of any one scores exactly 4 of 8. The two-by-two of "neither has" and "both have" alone is not enough: there, reuse equals
  task again. Those two stay as unscored controls, where truth is equal across bases.
- **Is the crossing enough? No.** The label equals base crossed with letter, so a reader of that interaction ties. Distance from W₀'s answers
  is one such reader. So add a table wrapper: a fixed permutation of 0-40 moving each value by more than 2. It puts a reuse far from W₀.
- **What nothing here removes.** `U`, the mutual information of the state's answers with W₀'s, over the entropy of the state's answers.
  It reads forward passes, is told no letter, and should read 1 on reuse. It is this set's real ceiling.

## 4. Can this separate methods?

It can separate cheap methods from each other. It cannot rank VPD above `U`, so no outcome of Stages 0-3 justifies a VPD run. A
decomposition per base is about 56 L40-hours, about 112 jobs of 30 min, so 224 for two, and gated.

## States

Fine-tune prompt `C c 40`, unseen by both bases. Rate 1e-4, saves at 1K, 2K, 5K and 10K, one state per run: the first save at accuracy 0.9.

| cell | task | expected on A | on B | seeds |
|---|---|---|---|---|
| a, +1 | count(a) + 1 | reuses | new | 0-4 |
| b, +1 | count(b) + 1 | new | reuses | 0-4 |
| a, table | table of count(a) | reuses | new | 0-4 |
| b, table | table of count(b) | new | reuses | 0-4 |
| neither | count(c) + 1 | new | new | 0-1 |
| both | token at distance 40, donor prompt `T NULL 40` | reuses | reuses | 0-1 |
| shuffled | answers from the histogram, independent of the text | not learned | not learned | 0-1 |

**Each difference from E55's recipe.**

1. LOW becomes base A, and NEVER becomes a base. Each base is the other's scratch model.
2. `C b 40` becomes `C c 40`, since `C b 40` is B's own trained prompt.
3. Task N gains the +1, so wrappers match. The table is new.
4. One rate, not two; first learned save, not fixed saves. Steps read become a labeller.
5. Truth is `T`, not the in-state behaviour and patch, which are reported.
6. Scoring is by cell votes with voids wrong, not by states with voids dropped.

Unchanged: Stage A's schedule, batch and loss; pruning K and random draws; reverse rate 1e-5 and its cap.

## Stage 0: the control (runs first, alone; early stop)

| check | reads | must read | else |
|---|---|---|---|
| texts, families, init across priors; the table | model-free | identical; a permutation | stop, fix |
| LOW count-a accuracy, patch share | 0.99975, 0.960 | within 0.01 | stop, apparatus |
| scratch steps at 1e-5 | 2,300 | ± 300 | stop, apparatus |
| steering on the identity-prompt states | 0.937, 0.930 | at least 0.5 | stop, apparatus |
| `T` with B as the state, own letter | unread | at least 0.8 | stop: the patch cannot read reuse |
| `T` with B as the state, other letter | 0 by construction | 0 | stop, fix |
| B's answer at `C c 40` | unread; LOW reads count a at `C b 40` | count b, at least 0.9 | tell the user |

## Stage 1: base A and the lack check

| number | A expected | B expected | else |
|---|---|---|---|
| own count accuracy, patch at block 5 | ≥ 0.95, ≥ 0.8 | 1.0, 1.0 saved | stop, recipe |
| accuracy per family, own letter | unread | unread | gap over 0.1 on the token task: tell the user |
| importance pruning control | drops; LOW 0.94, 0.80 | drops | that method void |
| frozen encoder, own letter + 1 | ≥ 0.95 | ≥ 0.95 | stop |
| frozen encoder, other letter + 1 | ≤ 0.7 | ≤ 0.7 | ≥ 0.9 stop; between, tell the user and go on |
| scratch steps for the other letter | unread, not capped | 2,300 saved | reverse void |

## Stage 2: train, calibration seeds first; then `T` and the room check

| number | at W₀ | own +1 | own table | other +1 | other table | neither | both | shuffled |
|---|---|---|---|---|---|---|---|---|
| task accuracy | 0 on +1 | ≥ 0.9 | ≥ 0.9 | ≥ 0.9 | ≥ 0.9 | ≥ 0.9 | ≥ 0.9 | ≤ 0.4 |
| `T` | own 0.96, other 0 | 0.6-0.95 | 0.4-0.9 | ≤ 0.02 | ≤ 0.02 | ≤ 0.02 | ≥ 0.5 | 0 |
| update size, relative to own +1 | 0 | 1 | 1-2, guess | 1.15-1.45, C1b | 1.2-3, guess | as other +1 | guess | small |
| steps to learn | none | 1K | 2-10K, guess | 1-2K | 2-10K, guess | 1-2K | 1-2K | none |

Do-nothing puts δW into a decomposition's Δ. There is no decomposition here, so it equals the zero update, W₀, on every number.

**Stops, before any method reads.** Fewer than 2 cells per class. `T` at own +1 under 0.5. `hi` not above `lo`. A scored cell unlearned
at 10K. Update size, steps or W₀'s first-step loss reaching 8 of 8 by hindsight: the test has no room. A cell whose truth differs from
the table is news, not a stop: say so loudly.

## Stage 3: the methods, cheap first

Cut-offs use `revival.cutoff` on seeds 0-1, one per method, committed before scoring. A pass needs the cell bar, one cell above the best
hindsight labeller, and p at most 0.05 over the 70 balanced relabellings of the cells.

| method | at W₀ | own +1 | own table | other +1 | other table | shuffled | control | lean |
|---|---|---|---|---|---|---|---|---|
| steering share | void | 0.83-0.96, C1b | 0.1-0.5, guess | 0.47-0.66, C1b | 0.1-0.5 | ≥ 0.7, guess | identity prompt 0.93 | 25% |
| pruning revival | void | 0.60-0.92, C1b | 0.1-0.5, guess | 0.06-0.08, C1b | 0.05-0.1 | ≥ 0.3, guess | importance pruning drops | 45% |
| reverse, over scratch | void | 0.11-0.13, C1b | 0.2-0.6, guess | 0.41-0.65, C1b | 0.4-0.8 | ≤ 0.2, guess | scratch finishes | 35% |
| neuron sparsity, E63's weight number | zero update | no gap | no gap | no gap | no gap | none | planted 13 units | 10% |
| update size, within its recipe | 0 | Stage 2 | | | | | none | 20% |
| `U`, reference | 1 | 1 | 1 | low | low | 0 | the state equal to W₀ reads 1 | ties |
| answer probe, reference | 0.67 as LOW | 0.67, C1b | near 0.67 | 0.28, C1b | near 0.28 | low | own answers | ties |

- **What a pass means.** Not that a method reads mechanism. It reads base crossed with task, told no letter, through a wrapper that breaks
  distance. `U` does the same.
- **Shuffled-label references.** The 70 relabellings are the required one. The shuffled fine-tune is the state-level one: it hides W₀'s
  count behind a constant answer and uses nothing. If a method reads it as reuse, that method reads "still there", not "used".
- **The both-have cell.** All three methods target W₀'s count, while this cell uses its token task, so expect "new". It tests whether a
  reading flips by base when truth does not.

## Predictions (to file, not filed)

| id | claim | lean |
|---|---|---|
| `T1` | truth forms as built on at least 7 of 8 cells | medium, 55%: table cells may erode |
| `T2` | the frozen encoder holds both lack bars | medium, 60%: probes already read 0.5-0.64 |
| `R1` | no hindsight labeller reaches 8 of 8 | medium, 65% |
| `M1` | steering reads at most 6 of 8 | high: one shift cannot be a table |
| `M2` | pruning and reverse each pass | low: tables are untested |
| `U1` | `U` reads all 8 right | high: true by construction if truth forms |
| `S1` | the three methods read shuffled states as reuse, at least 3 of 4 each | medium |

Blind spot: `T` reads block 5 at `=` only, so a state rotated off W₀'s residual reads new. Noise: binomial sd at most 0.016 on 1,000 pairs.

## Decision tree

- **Stage 0 or 1 stop.** Blocking card. A silent count ends the mirror design and goes to the user with the fallback.
- **Truth does not vary, or no room.** Say plainly that this test cannot separate methods. Close the counting line and go to Stage B.
- **All three pass.** The test tells nothing about ranking, the user's complaint, and we record that. It shows only that they read more than
  task or base. Go to the route-blocked branch.
- **None pass.** C1b's 21 of 21 was task reading, and the table says so. Stage B next. VPD stays off, since `U` still ties.
- **Methods differ, the likely outcome.** Survivors enter the table on a causal truth and carry to Stage B. The route-blocked branch runs
  if the user wants VPD to have a fair chance.
- **Route-blocked branch.** The one case where answers tie and truth can differ.
  - Zero the K units W₀ needs most in blocks 1-5, hold them at zero, and fine-tune the own-letter cells. A random knock-out at the same K
    is the matched-size partner. First, one job finds K where the targeted one takes W₀'s count under 0.5 and the random one leaves it above 0.9.
  - `T`, from the intact W₀, is the truth, and `U` should read 1 throughout. This is a built path, not natural post-training: a stated departure.
  - If `T` does not fall, surviving units carried the count, and the harder test is closed. Lean: 40% that it falls.
- **VPD goes on only if** truth varies in that branch, `U` ties, at most one cheap method passes, and a 400-step decomposition fails to
  match the label. That is one decomposition, about 112 jobs, gated, the user's call.

## Cost and routing

MATS `debug`, L40, account `c11`, $0, jobs at most 30 min with a 22-min deadline, no `normal`. Not gated, by the rule of 2026-10-30.

| stage | work | measured basis | time | jobs |
|---|---|---|---|---|
| 0 | controls, no training | job 46330, 9 min | ≤ 10 min | 1 |
| 1 | base A, frozen runs, patches | 12.0 ms per step, job 44268; patches 15 s | ~16 min | 1 |
| 2 | 52 runs, up to 10K steps, checks | 140 s per 10K run, job 46381 | 40-120 min | 2-6 |
| 3 | 52 states | 171 s mean, 239 max, C1b; slower reverse on tables | 150-210 min | 7-10 |
| total | | C1b: 153 min, 8 jobs | 3.5-6 h | 11-18 |

Resume as C1b does: `pcfg_train.train` saves at the deadline and exits 3, a state that cannot finish is not started, and results are
written atomically. Per-state files allow two jobs at once, where C1b's one JSON forced `singleton`. A dev-node watcher resubmits, since
compute nodes have no `sbatch`. Disk is about 12 MB per state, 0.6 GB.

## Order of work for the fleet agent

1. Read this plan, `experiments/pcfg_anchors/c1b.py`, `src/commute/revival.py`, `src/commute/interchange.py`.
2. Write `experiments/harder_counting/run.py`. Put the patch in `interchange.py`, and `U` and votes in `revival.py`. Move shared steps, not copy.
3. File predictions, running each rule on the edge cases above. Fresh-Opus review of the measurement code.
4. `/pod-runner`: Stage 0 alone, then 1, then calibration seeds, room check, committed cut-offs, scoring.
5. Report and notebook by `/writeup`, independent check, `TRIED.md` row, method-table row, PENDING line, commit, push.

## Departures from the literature

- **Jain et al. 2024, Def 1, L146-148.** They define possession by a linear readout. We report the probe and label by causal check, per
  `decisions.md`. They expect a model rarely lacks a relevant capability, L162-164, so lack here is measured and relative.
- **Def 2, L167-170.** It puts no bound on g, so the table is a valid g∘f_C. The project's held-out bound was demoted on 2026-11-02.
- **Operand priors, L1173-1178.** Neither of ours is on their list. We use them so neither base sees the other's operand.
- **Tasks.** They fine-tune one operand with no crossing. `C c 40`, the +1 and the table are ours.
- **Alg. 3 and reverse fine-tuning.** C1b's departures hold, with each base's own count as the target and the other base as scratch.
- **The frozen-encoder check, `T`, `U`, cell votes and every bar** are this project's. No source files them.

## Not in this plan

VPD on any set; the 67M; a second table, unless a verdict sits at the bar; neuron patching and prompting on the counting model; HIGH as a
base, which has both counts and no contrast; a second init seed of base A; token-only bases, unless the fallback fires.
