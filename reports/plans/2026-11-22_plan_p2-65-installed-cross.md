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

**Stage D (written after stage 4 passes; not in this commit).** Per moved base, a teacher-trained decomposition
(`commute.teacher.train_student`, E34's arm B on Pile rows, teacher = the base, from the released decomposition), then
its faithfulness (E34's harness, W0's floor 0.075 and the base's own do-nothing ratio) BEFORE any re-gating share is
read on it. Then, on each faithful base decomposition, the re-gating share of each of the 27 states and E_b over K
(definition-lead's E268 `commute.needed` is on main, so over K, not K*). A base whose decomposition fails the check is
reported as "no faithful featurization of this base" and its decomposition rows stay empty: this is the user's named
risk ("a bit risky to assume we have a featurization of the base model"), checked rather than assumed. Stage D is the
only paid stage (3 pods, one per base, H100).

## Predictions (filed; `results/predictions.json` "expected")

Controls: addition and cipher locks hold, Yoruba lean holds; diagonal 1-20 steps to G* against 30-500 off-diagonal;
transplant diagonal 0.0-0.15, off-diagonal 0.7-1.0. Readings at matched size: lean steering share separates (7-9 of
9), neuron shares lean separate; erank, untrained-SVD share, k_half and ADL no lean; prior does not separate. At matched
gain size and Pile KL separate trivially, which is why matched size is scored.

What each reading is blind to: all read the update against the base's weights, so a diagonal update that rebuilds the
skill elsewhere instead of re-opening the installed one would still read as reuse by history; the transplant control
is what rules that out. Nothing here says the installed skill looks like a skill pretraining would have made.

## Compute and cost

Stages 1-5 run on cluster debug at full size, each job <= 30 min, $0 (estimates in `cluster/installed_cross.sbatch`,
from E256's measured items; the first jobs measure them). Stage D: 3 H100 pods in parallel (one runner), E34's student
at 1000 steps took minutes on the 67M; with the harness and 9 states per base ~1 h per pod, ~$3 per pod-hour: ~$10.
Under $60 and 6 h: not a gated run; the appendix records the gates anyway as the brief asked.

## Not run here

Stage D code; any reading on M's released decomposition of a moved base; a fourth task; a second cue token.
