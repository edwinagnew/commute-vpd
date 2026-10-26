# Plan: re-gating without cues or triggers (E35, E36)

*2026-10-26, lead agent (overnight). Task from the user (`~/commute-data/tonight`): "continue exploring regating.
understand the phenomenon better so that you can restate the fixed-shift hypothesis in a way that applies to less
synthetic tasks ... anything that depends too strongly on defining cues or triggers is bad. a more universal
understanding of regating would be ideal." Words as in `GOALS.md` → Terms.*

## The request, restated in the Terms' words

The fixed-shift hypothesis now reads "one fixed change to Γ's importance values per cue", and "cue" means the token
at the trigger position. Find a form of it that names neither a cue token nor a trigger position: say what the change
of Γ's importance values at a position is a function of, test that form on the do-nothing states where the per-cue
form is known to hold, then test it on a fine-tune with no single trigger, scored at every position.

**The candidate.** Γ's change at a position is a function of the original model's importance values there:
positions that the original Γ treats alike are re-gated alike. Per cue is the special case where the original values
at the trigger differ mostly by cue word. Nothing in the statement names a token or a position, and it can be scored
on any text. Its strongest version needs no grouping at all: one fixed rule per subcomponent, its new importance a
straight-line function of its own original importance (the "fixed re-weighting" form below).

## Design check (`/design`)

- **Goal served.** Q5 (re-gating: what the switching is). The re-gating card's named next step is "a fine-tune with
  no single trigger, to define the hypothesis without cues" (research log, re-gating line; `backlog.md` entries
  "Spread-out fine-tune" and "Define the fixed-shift hypothesis without cues"). Approved classes: E35 is class 1
  (new arms of the existing harness on the regenerated do-nothing states of E32 and E33); E36 is class 4 (that card's
  named next step), and the user's task for tonight points at it.
- **Nearest tried.** E32/E33 settled that one change per cue word works where one change for every input fails, and
  that the per-cue change is not one direction, one fixed move of Γ's input, or the word's Pile signature. E18
  amendment 4 settled that inputs' own variation adds only a few hundredths beyond a fixed change. E30 applied one
  fixed change at every position as a map (unfaithful): a different question (a map, not a description). Nothing has
  grouped inputs without labels, and nothing has scored switching away from a trigger position.
- **Patch or step.** A step: it replaces the toy-task notion "cue" with one defined on any text.
- **Decisions relied on.** Γ frozen; do-nothing states (δW in Δ, released subcomponents, Δ out of the masked model);
  E28's fine-tune recipe for the toy states. No departure.
- **What would change the plan.** If the change is predicted from the original values (E35 passes), the restated
  hypothesis is carried to E36 and, if it holds there, to the larger model (Q9). If it is not (the cue matters
  beyond what Γ's values show), the next candidate space is Γ's input, the residual activations, and the user gets a
  card. If E36's switching share is small, re-gating is a feature of narrow fine-tunes and the line's scope changes:
  a card.
- **Control first.** E35: emoticon alone, reproducing E28 (G1), with every builder given a zero change recovering 0.
  E36: the step-10 state first, with its own updated values recovering 1 and zero changes recovering 0, and a stop
  if the fine-tune did not learn (gain gate) or switching carries nothing (gap gate).

## E35: is the per-cue change visible in the original importance values? (`experiments/regating_cuefree`)

**States.** E28's recipe (10-step arm-A Adam 3e-5 fine-tunes of all 24 matrices, seed 0), regenerated in-job:
emoticon (control), " will" → " have" (one cue), the copy task and the 32-pair cipher (E32), E33's 32 cues with their
own answers and with one answer " of". Γ's values and their change d(x) = g₁(x) − g₀(x) are read at the trigger on
training and held-out windows.

**Arms.** All are "each held-out window's own original values plus a predicted change, clipped to [0, 1]", fitted on
training windows only, and scored as E28 does: recovery of the switching effect over the window's own updated
values (R, 0 = original values, 1 = own updated values).

| arm | predicted change at held-out window x | names a cue? |
|---|---|---|
| shared | s, the mean change over all training windows (E18's filed arm) | no (the old single change) |
| per cue | s_k, the mean change over training windows with x's cue word | yes (the reference) |
| nearest | the mean change of the 8 training windows whose original values are nearest x's | no |
| clusters K | the mean change of x's cluster, clusters found by k-means on the training windows' original values, K = 8, 32, 128 | no |
| random groups | as clusters at K = 32, with random labels (5 draws): the size-matched null | no |
| fixed re-weighting | each subcomponent's new value a·g₀ + b, its own straight line fitted over training windows | no |

Distances use every alive subcomponent's original value, all matrices together.

**Numbers and their expected values.**

| number | W₀ (no update) | emoticon (control, 10 trigger tokens) | one-cue word task | 32-cue states |
|---|---|---|---|---|
| r, R_shared_net, R_cue_net (gates) | gap 0 | 0.597, 0.815, 0.901 (E32) | 0.904, 0.916, 0.916 (E32) | reproduce E32/E33 within 0.05 and 0.02 |
| R_shared | undefined | 0.737 (E32) | 0.708 (E32) | 0.01 to 0.13 (E32/E33 shared arms) |
| R_cue | undefined | reported | = R_shared | between R_shared and R_cue_net |
| R_near, R_clu(K) | undefined | reported | within 0.1 of R_shared | prediction F, K |
| R_rand | undefined | within 0.1 of R_shared | within 0.1 of R_shared | within 0.1 of R_shared |
| R_fix | undefined | reported | reported | prediction A |
| R_clu(32) in E32/E33's net mean form | undefined | reported | = R_cue_net ± 0.1 | within 0.1 of R_cue_net where F holds |
| cluster purity (share of windows whose K = 32 cluster's commonest cue is their own) | — | reported | 1 | reported |

**Predictions** (filed in `experiments/regating_cuefree/results/predictions.json` before the run):
- **F, the change is a function of the original values** (medium): on each 32-cue state, R_near ≥ R_cue − 0.1.
- **K, clusters find the cues without labels** (medium-low): on each 32-cue state, R_clu(32) ≥ R_cue − 0.1.
- **A, one fixed rule per subcomponent is enough** (low): on each 32-cue state, R_fix ≥ R_cue − 0.1 and
  R_fix ≥ R_shared + 0.2.
- **N, the null** (medium): R_rand within 0.1 of R_shared on every state; a state failing it is unreadable for K.
- **C, one cue** (medium): on the word task, R_near and R_clu(8) within 0.1 of R_shared.

**Blind to:** the trigger position only; one fine-tune seed; one 67M model; mean changes, so within-group variation is
not modelled; Euclidean distance over all alive subcomponents weighs irrelevant ones equally.
**Noise floor:** the states are deterministic per seed (E33 reproduced E28 within 0.002); the random-group spread
(5 draws) measures what averaging fewer windows does alone.

**Decision tree.**
- G1 fails: stop, nothing scored.
- F and K pass: the restated hypothesis ("positions the original Γ treats alike are re-gated alike") goes to E36 with
  the nearest and cluster arms as its tests.
- A passes too: the stronger form ("one fixed rule per subcomponent") is E36's headline test.
- F fails while R_cue holds: the cue is not visible in Γ's values at the trigger; card for the user (next space:
  Γ's input), and E36 runs with the per-token arm as its main comparison.
- R_cue in this form below 0.5 on the 32-cue states: the shift form loses what E32/E33's mean form had; E36 then
  leads with the fixed re-weighting form, which can shrink each value towards its mean.

**Cost.** About 3 minutes of GPU after loading (E32 took 118 s on a 4090, E33 197 s with mining). The cluster's only
node is draining ("Kill task failed", since 01:35), so RunPod 4090: about $0.3 with set-up.

## E36: re-gating on a fine-tune with no trigger (`experiments/regating_spread`)

Sketch; its predictions are filed after E35 is read, in an amendment to this plan, before it runs.

- **Task.** Code-like text from the Pile: rows where a fixed share of tokens contain code punctuation
  (`{ } [ ] ; = < > _ \`), a rule set before the run; the selected share and three decoded rows are logged. A fine-tune
  of all 24 matrices on next-token loss at every position (`finetune.finetune_arm_b`, Adam 3e-5, batch 8), read at
  steps 10 and 100. Closest available to the user's example of a real post-training task ("get better at calling
  Python inline"): no trigger, no target token, a behaviour the model already partly has.
- **Scoring at every position.** On 64 held-out code-like rows: the target model's mean log-probability of the next
  token at W₀ and after the fine-tune (the gain); the masked model (released subcomponents, Δ out) with the original
  model's values at every position and with Γ reading the updated model at every position (the gap); r = gap / gain.
  Each arm writes its values at every position at once; recovery as in E35.
- **Arms.** E35's (shared, nearest, clusters at K up to 512, random groups, fixed re-weighting), plus *per token*: the
  mean change at training positions holding the same current token (the toy definition of a cue, carried over), the
  shared change for tokens seen fewer than 20 times. Fitted on positions of the fine-tune's training rows.
- **What it decides.** Whether a cue-free form carries the switching on a task with no trigger, and how many groups
  of the original values it needs (the K curve), which is the task-independent quantity "number of cues" stood in
  for.

## Not run (by this plan)

Seeds beyond 0; Γ's input (residual activations) as the space for grouping; faithfulness under the adversary of any
predicted pattern; a second spread-out domain; SGD or LoRA updates; the larger model.

## Amendment 1 (2026-10-26, after E35; files E36's predictions)

**E35 in one paragraph** (analysis: `reports/2026-10-26_analysis_regating-without-cues.md`). With no cue labels, the
mean change of the 8 training windows whose original values are nearest recovered 0.61 to 0.78 of the switching on
the four 32-cue states, as much as or more than one change per cue in the same form (0.47 to 0.66), where one shared
change gives 0.01 to 0.13 (F held). k-means groups did not line up with cues (purity 0.40 to 0.57 at 32 groups) and
needed about 128 to reach the per-cue level (K failed). One line per subcomponent recovered 0.96 and 1.02 on the two
few-cue states and 0.37 to 0.80 on the 32-cue ones (A failed). On the one-cue state the cue-free arms beat the shared
change by 0.19 (nearest) and 0.31 (line), where C expected no difference.

**What changes for E36.**
- The nearest arm is the headline test (P_near), the line arm second (P_fix); the K curve is read, with P_K at 512.
- Rows: the 5% of the first 20,480 'train' rows with the highest share of tokens holding one of `{}[]();=<>_\`,
  the same cut applied to 'val' rows 0..8,191 (a fixed share threshold could select too few rows; the cut must be
  at least 0.12). Parentheses are added to the plan's character set.
- Two descriptive numbers: the share of each position's nearest fit positions holding its own token (is "nearest in
  the original values" just "same token"?), and the slopes of the line arm among subcomponents whose values vary
  (a median below 1 reads as each subcomponent pulled towards a set point).
- Predictions, gates and decision tree: `experiments/regating_spread/run.py` → `PREDICTIONS`, filed on the pod by
  `run.py --predictions` before the run: P_r (r ≥ 0.3), P_shared (R_shared < 0.5), P_near (R_near ≥ 0.6 and ≥
  R_shared + 0.2), P_token (R_near ≥ R_token − 0.1), P_fix (R_fix ≥ R_shared + 0.1), P_K (R_clu_512 ≥ R_near − 0.1),
  N (|R_rand − R_shared| ≤ 0.05); scored at step 100, step 10 read by the same rules.
- Cost: a RunPod 4090, sent the VPD files as E35's second pod was (W&B refuses this node's key): about 10 min of
  set-up and upload, a few minutes of run; under $1.

## Amendment 2 (2026-10-26, after E35's independent check): E35 Amendment 1

The check (a fresh Opus agent) found E35's numbers and verdicts right and three gaps in what the headline can claim.
The nearest-inputs rule is not like for like with the per-cue rule, because it also lets the change depend on each
input's own values. Its neighbours may simply share the cue. Held-out windows duplicated in training were not ruled
out. `experiments/regating_cuefree/amend1.py` regenerates the six states (gate: the nearest rule reproduces E35 within
0.001, emoticon first) and adds the nearest rule restricted to training windows with the same cue (the like-for-like
reference), the share of the eight neighbours holding the window's cue, the share of held-out windows duplicated in
training, and a nearest-distance ratio. Predictions, filed by `amend1.py --predictions` before the run: L, nearest ≥
nearest-within-cue − 0.1 on every 32-cue state (medium-low); T, the neighbours share the cue at ≥ 0.9 (medium); D,
duplicates ≤ 5% (medium). Class 1 (new arms on regenerated states). Runs on E36's pod, about a minute, no extra cost
beyond pod time.

## Amendment 3 (2026-10-26, after E36): a fine-tune that learns, lowercase text

**What happened.** E36 ran cleanly but is unreadable: the code-like fine-tune made held-out text worse, a gain of
−0.052 nats per token at step 10 and −0.065 at step 100, so gate G failed at both steps and no prediction was scored.
Its training loss did not fall either, 1.33 at step 1 and between 1.31 and 1.83 afterwards: the model is already at its
optimum on Pile text of this kind, and Adam at 3e-5 with batch 8 adds noise. The row rule also chose mostly LaTeX and
maths, not code. The filed tree's branch for "G fails at both steps" read it as "re-gating carries little", which does
not apply: switching carried 0.77 to 0.82 of the (negative) change.

**Design check, short.** Same goal and class as E36: the re-gating card's named next step, a fine-tune with no single
trigger. The change: a spread-out behaviour the model can learn quickly, a style rather than a domain. Pile text is
lowercased; the fine-tune learns that capitals never come, which moves predictions at many positions with no trigger
and no target token. Nearest tried: E36 (did not learn). It is synthetic in form, a shortcoming a pending-decision card
records, along with the alternatives (a gentler domain fine-tune with far more rows; a real post-trained model pair at
the larger model, Q9). Arms, gates and rules are E36's unchanged, read at steps 10, 100 and 300 and scored at 300;
`experiments/regating_spread/amend1.py`, predictions filed by `amend1.py --predictions` before the run. The measurement
code is E36's, reviewed; only the row construction is new (decode, lowercase, re-encode, cut into 512-token rows), so
no separate code review was run. Cost: one RunPod 4090 with the VPD files sent, about 20 minutes, under $0.5.

## Amendment 4 (2026-10-26, after E36's independent check): E36 Amendment 2, cheap checks

The check found E36's scoring right and the reading too strong: by the filed tree, step 300 falls in "all fail", and
the comparison with the current token was weak (the token rule covers only 90 tokens) and not like for like. Two
questions decide how to read that branch before anything else is designed. Is the nearest rule's shortfall sampling,
since it drew on only 16,384 fitted positions? And does restricting the neighbours to the same token help, as
restricting to the cue did on E35's own-answer state? `experiments/regating_spread/amend2.py` regenerates Amendment 1's
lowercase fine-tune (gate: the 16,384-position nearest rule reproduces Amendment 1 within 0.02 at every step) and
adds the nearest rule on 65,536 positions, the nearest rule within the same token, and the token rule's coverage.
Predictions filed by `amend2.py --predictions` before the run: S, more positions add at least 0.1 at step 300 (predicted
to fail, low-medium); W, restricting to the token adds at most 0.05 (predicted to hold, low). Class 1 (new reads of a
regenerated state). The filed branch's own next step, a rule on Γ's input or on neighbouring positions, is a new
design and goes to the user through the pending card, not run here. Cost: one RunPod 4090 with the VPD files sent,
about 15 minutes, under $0.3.

## Amendment 5 (2026-10-26, after E36 Amendment 2): the sampling curve

Amendment 2 found the nearest rule still rising with the number of fitted positions: 0.72 to 0.80 at step 10, 0.54 to
0.67 at step 100, 0.51 to 0.58 at step 300, from 16,384 to 65,536 positions. Whether the filed "all fail" at step 300
is sampling or a real dependence on more than the original values decides the pending card, so
`experiments/regating_spread/amend3.py` reads the same rule on all 262,144 positions of 512 training rows, with 8 and
32 neighbours, on the regenerated lowercase fine-tune (gate: the 16,384- and 65,536-position rules reproduce Amendment
2 within 0.02). Predictions filed by `amend3.py --predictions` before the run: M, still rising, at least 0.03 more than
65,536 positions give at step 300 (medium); B, at least 0.7 at every step (predicted to fail, low). Class 1. About 35
GB of GPU memory, so an H100; about 20 minutes with set-up and upload, under $1.5.

## Amendment 6 (2026-10-26, after Amendment 3's independent check): is Γ's original-value space special?

Amendment 3 found the nearest rule recovering 0.76 to 0.89 of the switching with 262,144 fitted positions. Its check
pointed out that a nearest rule on any rich summary of the context may improve that way, so the result shows "similar
contexts get similar changes" and not yet "inputs the original Γ treats alike". It also asked for the other rules at
the same sample size, position in the row, and near-duplicate text. `experiments/regating_spread/amend4.py` regenerates
the lowercase fine-tune and fits, on all 262,144 positions: the nearest rule on the original values (gate: reproduces
Amendment 3 within 0.01), the same rule on the residual stream (each layer's attention input, the values Γ reads for
the query, key and value matrices, under the original model), the token rule, and a rule per position in the row; it
reports recovery by position bucket, the token rule's coverage, and the share of held-out positions inside a 16-token
span also found in the training rows. Predictions filed by `amend4.py --predictions` before the run: R1, the residual
stream does as well within 0.05 at step 300 (low); R2, position alone under 0.3 (medium); R3, the token rule at least
0.2 behind (medium); D, under 2% duplicated (medium). Class 1. An H100, about $0.4.
