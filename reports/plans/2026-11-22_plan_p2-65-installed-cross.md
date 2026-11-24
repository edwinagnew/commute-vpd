# Plan: the installed-skill cross (cell P2.65, E271)

Card: `reports/plans/2026-11-22_plan_p2-65-installed-cross-card.md`. Appendix (gates, pre-mortem, sources, rehearsal,
early checks): `reports/plans/2026-11-22_plan_p2-65-installed-cross-appendix.md`. Code `experiments/installed_cross/`
(`run.py`, `score.py`), shared steps `src/commute/installed.py`; rules `results/predictions.json`.

## The request, restated in the Terms' words

Build fine-tunes whose answer to "reused or new?" is known from how they were made, then ask which readings recover
it. For each of three tasks the released 67M (M) cannot do, train a base that does the task only when a cue token is
present, with everything else (the task's uncued items, ordinary Pile text) distilled to M's outputs. Then fine-tune
every base, uncued, on every task. A base fine-tuned on its own installed task should be unlocking something it already
has (reuse by history, the diagonal); a base fine-tuned on another task has to learn it (new by history, the
off-diagonal). Three controls, decomposition-free, must first show the history is real. Then every cheap reading is
asked whether it ranks the reuse base above both new bases. No reuse verdict is given under either definition wording
(P1.1 is open); E_b is a row, not the judge.

## Tasks (M at floor on each, dissimilar)

| task | item | M (measured or sourced) |
|---|---|---|
| addition | E256's: Pile fill, two worked sums, "a + b =", answer ' a+b' | rate 0.103, chance 0.1 (E256 stage0) |
| cipher | Pile fill, "\n\n key =", answer a seeded random word (32 keys) | prior ~1e-5 by construction (ctl reads it) |
| Yoruba | 64-token Wikipedia windows, next token; read on the 512 of 2048 held windows M predicts worst | 2.15 nats/byte vs 0.80 English (E209) |

Yoruba is the real-text task: a language the model barely has, as realistic as this model allows (AG News topic is not at
floor: M reads 0.93, E209). The cue is one token (id 49003) written over position 0.

## Stages (cheapest first, controls first and alone)

**Amended 2026-11-22 after the gate-5 review** (`reports/reviews/2026-11-22_review_e271-installed-cross.md`; every
change in predictions.json "amendment_2026-11-22"). Where the stages below differ, the amendment holds: the lock and
G* are read on the stop rows, cued gain is over M cued; the install adds an answer-position KL on uncued items and an
early leak check; Yoruba's bar is 3 and its held windows are word-start answers in a seeded order; cross and read run
one job per (base, task) cell (12 each); Gaussian floors are scored on numerators.

1. **ctl** (GPU, ~4 min): build and save the items (sha256); the cue is absent from them; a zero update reads exactly as
   M; M at floor on addition (rate <= 0.2) and cipher (<= 0.01).
2. **install TASK** (GPU, per task in parallel): CE on cued items + lambda x (KL to M on uncued items + KL to M on Pile
   windows), lambda in {1, 3, 10}, 2000 Adam steps at 3e-5. Lock: (a) cued gain over M's uncued >= the task's bar
   (addition 4, cipher 8, Yoruba 2 log-odds); (b) uncued shift within 0.2 decades of M; (c) Pile KL(M || M_T) at most
   the KL of M's own uncued fine-tune at G*. The strongest lambda whose lock holds is kept. G* (the target gain of every
   fine-tune on that task) = half the kept base's cued gain, so "reaches half cued performance" is the stop itself.
3. **cross BASE** (GPU, 4 jobs in parallel: three bases and M): each base fine-tuned uncued on each task, seeds 0-2,
   Adam 3e-5, batch 32, stopped the first step its 256 stop rows' mean gain reaches G* (cap 3000). Recorded: steps,
   gain on the 256 disjoint read rows, size, Pile KL; for a moved base the transplant: the same update added to M, its
   gain on M over its gain on the base. 27 states plus M's 9 (the descriptive row).
4. **controls** (CPU): (2) per task, seed-mean steps on the diagonal x 5 <= each moved off-diagonal's; (3) transplant
   kept, seed mean per cell: every diagonal < 0.2, every off-diagonal > 0.6; every state reached G*. A miss stops the
   run before any reading; it is written to `results/controls.json` and is itself the result.
5. **read BASE** (GPU, 4 jobs): E211's reader set on each state against its own base (the base swap
   `installed.set_base`: the target model and W0 become the base, so steering, untrained-SVD share (of the base's own
   SVD), neuron TopK shares, k_half, erank, size and prior read the update against the base), plus Pile KL and ADL's D
   (E264's lens). Three versions per state: at matched gain (as trained), at matched size (rescaled to the median size
   of its task's 9 states, E187's one-number rescaling), and a Gaussian at the rescaled state's per-matrix norms (the
   floor; read first; a gain >= 1 stops). The released decomposition's share is left out: its gates belong to M.
6. **score** (CPU, `score.py`): per reading and kind, 9 triples (chance 1/3 each) and AUC 9 vs 18, with the matched
   Gaussians' AUC as floor. Scored kind: matched size. A reading "separates" at >= 7 of 9 triples (P = 0.0077 under
   chance) and AUC >= 0.85 with the Gaussian AUC under 0.85. Edge cases (nan, ties, empty pools) were run in the dry run.

**Stage D** (`experiments/installed_cross/stage_d.py`, rules `results/predictions_stage_d.json`, pod plan in the
appendix). Per moved base, a teacher-trained decomposition (`teacher.train_student`, E34's student: alive (U, V)
offsets, Gamma and Delta frozen, Adam 1e-5, 1000 steps; teacher = the base; rows in thirds: cued items of its task,
uncued items of all three, Pile windows; arm A's weighting at the answer). Its faithfulness is read and written BEFORE
any reading on it (amended 2026-11-22b after `reports/reviews/2026-11-22_review_e271-stage-d.md`): F1 the student
keeps >= 0.8 of the base's cued gain over M cued, its teacher gap on cued rows is <= 10% of M's (E34 C3), and on
uncued rows it stays within 0.2 decades of the base; F2 the Pile harness passes against the same-job W0 floor (E34 C1,
`faith_score`); F3 the same adversary, its KL at the answer only, on the base's cued and uncued task rows passes against
M's own seed spread there (Stage D's ctl; a filed larger block if that spread is over 0.3); F4 the masked model at its
own gates stays within 1.5 x M's distance from the unmasked one at the answer, cued and uncued. Only a base that passes
all four is read: the re-gating share (do-nothing form, its components) of its 9 states at matched gain, matched size
and the Gaussian floor, then E_b over the needed set K (E268's `commute.needed`, so over K, not K*) at matched size and
the floor. A base that fails reads "no faithful featurization of this base" and its rows stay empty: the user's named
risk, checked rather than assumed. Then, because every triple holds all three bases, one failure changes the scoring
rule, filed before any read: with two faithful bases, per task whose own base is faithful and seed, the diagonal
against the one faithful off-diagonal (all 6 pairs above, chance 1/2 each), plus the AUC over the faithful bases'
states; with fewer than two, Stage D is not scored. The released M's row uses the released decomposition, descriptive.

## Predictions (filed; `results/predictions.json` "expected")

Controls: addition and cipher locks hold, Yoruba lean holds; diagonal 1-20 steps to G* against 30-500 off-diagonal;
transplant diagonal 0.0-0.15, off-diagonal 0.7-1.0. Readings at matched size: lean steering share separates (7-9 of
9), neuron shares lean separate; erank, untrained-SVD share, k_half and ADL no lean; prior does not separate. At matched
gain size and Pile KL separate trivially, which is why matched size is scored.

**Write-up notes (amendment 2026-11-22c, redteam-lead).** Steering share is close to built for this test: a diagonal
update that opens a cue lock acts like "the cue is present", a near-constant input feature, which is what a pooled
steering vector models; a steering pass shows that steering detects a cue-lock opening, the narrowest kind of reuse,
and does not carry to natural fine-tunes; the informative readings are the others. Eleven readings are scored at one
bar each, so about one may pass by chance. Per-task triple counts are reported beside the 9, and a reading is called
"separates" in the write-up only if every task has at least 2 of 3; a share reading that separates at one kind (gain,
size) and not the other is the rescaling, not the history (each read record carries its rescale factor). If a task is
dropped by amendment its base goes with it, so each comparison is a pair at chance 1/2: the bar is 6 of 6 (P 0.016),
filed now (redteam's 5 of 6 at 1/3 assumed three bases with two tasks, which cannot occur). At the task level, with
seeds moving together, two tasks give at best P 1/4, so a two-task result is descriptive ("separates on 6 of 6 pairs;
two tasks, no task-level significance"); three tasks give at best 1/27 = 0.037 (amendment 2026-11-22e withdrew the
1/16 filed here before).

**Write-up notes (amendment 2026-11-22e).** The tasks were swapped by whether a cue lock could be installed: the cipher
leaked past the cue at every one of 6 settings, Yoruba missed its cued-gain bar, and SST-2 (and, if Yoruba fails, the
cipher with the cue just before the key) took their places with every bar unchanged. So the known-answer tasks are
those a cue lock could be installed on, which is a selection the write-up states. SST-2 is a feature language models
carry, so it plays the new side only if M's own uncued fine-tune is slow: if M reaches SST-2's G* within 5x the SST-2
diagonal's steps, SST-2 is dropped and the cell falls to the filed outcomes without it. Each cross cell records the
base's head start (its uncued gain over M at step 0), and each install's absolute uncued probability is described
beside the 0.2-decade lock, never in place of it.

**Third task (amendment 2026-11-22f).** Yoruba and the cue-before-key cipher both failed, so the cell runs as addition
and SST-2 under the two-task rule, and one built task is tried beside it: number parity (" 347 is" then " even" or
" odd", training and held numbers disjoint). If its lock holds, the 3x3 cross with it is the primary result under the
three-task rule, and the two-task cross is reported beside it; if not, the two tasks stand as a descriptive result.

What each reading is blind to: all read the update against the base's weights, so a diagonal update that rebuilds the
skill elsewhere instead of re-opening the installed one would still read as reuse by history; the transplant control
is what rules that out. Nothing here says the installed skill looks like a skill pretraining would have made.

## Compute and cost

Stages 1-5 run on cluster debug at full size, each job <= 30 min, $0 (estimates in `cluster/installed_cross.sbatch`,
from E256's measured items; the first jobs measure them). Stage D: a 20-min cluster rehearsal on one base, then 3 L40S
pods in parallel (plain fp32 is faster there than on an H100), one runner, ~3 h each (the K scans dominate;
unmeasured, the rehearsal measures them), ~$10 in all (appendix, "Stage D: rehearsal and pods"). Under $60 and 6 h: not a gated run; the appendix records the gates anyway.

## Not run here

Any reading on M's released decomposition of a moved base; matched-gain E_b (cost); a fourth task; a second cue token.
