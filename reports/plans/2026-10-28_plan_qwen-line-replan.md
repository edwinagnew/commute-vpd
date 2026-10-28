# Plan: re-planning the Qwen line. Five experiments, cheapest and most decisive first

*2026-10-28, written for the user at their request ("the qwen direction needs better guidance ... help me come up with
a better plan for qwen-decomp"). Proposal, not a decision: the user picks. Inputs: every Qwen report and result in this
repo, the user's own words from past sessions, and a literature scan (sources at the end). Words as in `GOALS.md` →
Terms.*

**In plain words.** The Qwen line spent a day and a half on whether GSM8K is memorised and whether RL from 75% is
worth doing, then on a first fine-tune that turned out to be a format change which damaged the model, and on a
faithfulness test that could not tell the trivial map from a good one. This plan drops the side questions, keeps the
infrastructure, and puts five experiments in order. The first two cost under $10 each, need no new training, and
answer the project's question directly on what is already saved. Teacher training on an H200 waits until they are read.

## 1. Why the line flailed

- **Heuristic questions were treated as research questions.** "Is 0.75 too high for RL?" and "is GSM8K memorised?"
  took three cluster runs, five reports (about 18,000 words) and most of 2026-10-27, to reach "keep GSM8K". The
  answer the project needed was already in the RL plan's first section: we study one to fifty steps, so any
  measurable behaviour change will do; the end score and its provenance do not matter.
- **The first update was the wrong kind of update.** SFT on GSM8K's gold solutions starts at a gold log-loss of 2.5
  nats per token. The model finds that text very surprising, because gold solutions are terse and end in `####`.
  Step one moved predictions by 0.106 nats per token, the answer length halved, and accuracy *fell* from 0.788 to
  0.627 by step 100 (`experiments/qwen_sft/results/`, research log v82). The user's use case is "a small update that
  does something interesting". This update is large and makes the model worse. It is a format fine-tune.
- **The pass test cannot fail the trivial map.** Doing nothing (δW into Δ) read 1.05 at step 1 and 1.18 at step 10 on
  web text, both passes, while the update was moving maths predictions by half a nat. The paper's adversary asks
  whether masked-off parts can damage the model. Putting δW into Δ changes nothing about what is masked, so at small
  δW the test is blind to it by construction. Until a test fails something, a pass means nothing.
- **The chosen map is the most expensive one.** Teacher training on Qwen needs 73 GB of offsets (494,598 of 688,128
  subcomponents are "alive" at g > 0.01 somewhere), an H200, about $30 a run, and code that has never run at full
  size. It is being pushed before the cheap maps (least-squares split, adjoint split), which need no training, have
  been read on the δWs that E40 already saved.
- **Each adversary read is 20 minutes of H100 for a ±11% lower bound.** Most of the Qwen GPU time went into these
  reads, and they are the slowest, noisiest number we have.

## 2. What exists and is good

- A validated torch port of the decomposition, on a RunPod volume (US-MO-1, `ydavm7ct4x`) and on W&B. Γ, masked
  forwards and the adversary run. The alive set and the model's own GSM8K answers (`maths_rows.json`) are saved.
- Baselines: GSM8K 0.788 / GSM1k 0.778 among answers that fit 512 tokens; W₀'s web-text adversary 1.70 (three seeds
  1.70, 1.44, 1.75), maths-text adversary 4.24; masked KL 0.142 web, 0.059 maths.
- **E40's δW at steps 1, 2, 3, 5, 10, 20, 50 and 100, and the gradient at W₀, on the old volume.** Nobody has read a
  map on them. These are the cheapest inputs the line will ever have. The old volume must not be deleted until Q1 has
  run (the handoff note proposes deleting it).
- SFT code that works (`qwen_sft.py`), and a filed warm-up recipe.

## 3. The question, as it applies to Qwen

The user's words (2026-10-27): "the use case we are thinking about here is interpreting what a single post-training
step does to a model", and "we have evidence that post-training updates manifest more as regating than updating
subcomponents, but i still dont know how to interpret re-gating". Three questions follow, in the order they can be
answered:

- **A. Does a simple map keep the decomposition faithful?** On Qwen the subcomponent count per matrix equals the
  larger of its two dimensions (2048, 1024 or 6144), so the alive read and write vectors very likely span the whole
  space and the least-squares split expresses any δW exactly, with Δ untouched. That is the simplest Φ there is. Does
  it pass VPD's checks? Nobody knows, and it costs forward passes.
- **B. Where does the update land?** In the subcomponents (which ones, how many) or in Γ's gating (re-gating)? For a
  re-gating reading, Qwen offers something the 67M never had: a built-in reference for "machinery that already
  exists". Qwen3 has a thinking mode. The literature predicts that a no-think GSM8K update works by switching the
  think-mode machinery back on (item 3 below). If the subcomponents Γ switches on after the update are the ones it
  uses on think-mode prompts, re-gating has a concrete meaning: the update recruits the thinking circuit.
- **C. Does an on-policy update (RL-like) differ from off-policy SFT?** This is the user's re-weighting hypothesis.
  The literature says the data source, not the RL objective, drives the difference, and that SFT on the model's own
  correct samples behaves like RL. So the cheapest RL-like arm is SFT on the model's own correct samples
  (rejection-sampling fine-tuning). It reuses the SFT code, has no format shift, and GRPO follows if the two arms
  differ.

## 4. The experiments

Every run below stays under the gate ($15, 3 h). Each names the one thing that changes depending on its result.

### Q0. A pass test with teeth ($0 for the design, ~$3 to calibrate; runs inside Q1's pod)

*Question.* Can any of our measurements tell a good D′ from a bad one for a small update on Qwen?

*What.* Three readings, all forward passes on the 128 maths rows and the 128 web rows, each as a ratio to W₀'s own
reading on the same rows:

1. **CI-masked and stochastic-masked KL** (VPD's own checks, minus the adversary). Minutes per state.
2. **Behaviour kept under masks.** The CI-masked D′ against the fine-tuned model on held-out maths rows, as a share
   of the fine-tune's behaviour change; and the CI-masked D′'s GSM8K accuracy. This is the reading that says "the
   components carry the new model".
3. **Paired adversary.** Reuse W₀'s saved adversarial sources (three web seeds, one maths seed) on D′ without a
   search: forward passes. Angsten (2026) shows sources carry over to unseen text and keep most of their damage. A
   fresh 20-step search stays as the occasional check, not the per-state test.

*Positive controls, which must fail.* (i) The same δW split onto subcomponents Γ gates *off* on maths rows: reading 2
must fall. (ii) A random change to (U, V) of the same size as the least-squares split's: reading 1 or 3 must rise.
If neither control fails, the test has no teeth, and the line stops to settle the criterion with the user before any
map is scored. A free check goes with it: what share of each matrix's top singular subspace the alive U and V span.

*Changes.* Which number is the pass test for every later experiment, and whether the 20-minute adversary is run per
state or once per experiment.

### Q1. The cheap maps on E40's saved δWs (~$6-8, one H100 in US-MO-1, no training)

*Question.* Does the least-squares split, the simplest Φ, keep the decomposition faithful for a real update on Qwen?
Does the adjoint split? How far is each from doing nothing?

*What.* δW at steps 10 and 100 from `/workspace/e40`. For each: do-nothing, least-squares split, scaled adjoint
split, scored by Q0's readings. Also the per-subcomponent updates themselves: how many subcomponents carry 90% of the
split's energy, by layer and by matrix kind, with the random-update null.

*Expected.* The least-squares split passes readings 1 and 2 at step 10 (the span is complete, the update is first
order), and the question is reading 3 and step 100. Do-nothing passes 1 and 3 and is the reference for 2.

*Changes.* **Pass:** a simple map works for this update; teacher training is not needed for the weak version, and
the line moves to B and C with the splits as the map. **Fail at step 10:** teacher training (Q4) is the fallback,
and the failure is the first Qwen result for the brittleness branch. **Fail at 100, pass at 10:** the step sweep
that `GOALS.md` asks for ("small is measured, never assumed") has its first point.

### Q2. The right first update: the model's own correct samples against gold solutions (~$10, one H100)

*Question.* An on-policy update (the model's own correct samples) against an off-policy one (gold solutions), on the
same task, same code, matched on how much they move the model. Neither arm is labelled elicitation or new; what each
update does to the model is what Q3 reads (§7).

*What.* Sample the unchanged model on GSM8K train (4 samples per problem, about 2,000 problems, thinking off), keep
the correct ones, and run the warm-up SFT recipe on them for 100 steps, saving δW at the same steps as E40. Rerun gold
SFT with the warm-up recipe beside it. Match the two at equal KL to start on maths rows (RL's Razor: forgetting and
faithfulness track task KL, not step count). Read accuracy, length and KL to start at each saved step.

*Expected.* The own-samples update raises accuracy a few points with KL to start under 0.05 at step 100 and no length
change; gold SFT at matched KL still cuts length. If the own-samples update does nothing measurable in 100 steps, raise
the rate once, not the task.

*Changes.* This replaces gold SFT as the line's canonical update. Both δW sets go through Q1's table.

### Q3. Reading the update: where it lands, and what re-gating means on Qwen (~$5, forward passes on saved states)

*Question.* For each update, how much of its behaviour change runs through changed subcomponents and how much through
Γ gating existing subcomponents differently; and are the newly gated-on subcomponents the think-mode machinery?

*What.*
1. The route split at scale, on maths rows: old and new subcomponents under old and new gates, the four cells'
   KL to the fine-tuned model. The 67M's direct and re-gating shares, read on a real model.
2. Γ's switching: per-subcomponent mean importance on maths rows before and after, which subcomponents switch on.
3. **The think-mode reference.** Γ's importance on the same problems with thinking on, at the unchanged model. Overlap
   of the switched-on set with the think-mode set, against the overlap with random sets of the same size and with the
   set Γ uses on unrelated chat text.
4. Where δW lands: norm by layer and matrix kind (the literature points at o_proj and late layers), effective rank per
   matrix, and share on alive against dead subcomponents after the split.

*Expected.* The own-samples update: small direct share, re-gating share large, switched-on set overlapping the
think-mode set well above chance. Gold SFT: more in the subcomponents, in attention output matrices, less overlap.

*Changes.* This is the interpretability payoff and the first concrete meaning for re-gating. A clear think-mode
overlap turns "re-gating dominates" into "the update recruits the thinking circuit", which is a claim that can go in a
write-up. No overlap means the think-mode story is wrong for this update and the switched-on set needs another
reference.

### Q4. Teacher training on Qwen (E40 Amendment 1, ~$30 on an H200) — only if Q1 fails

The plan and code exist. It runs if the least-squares split fails Q0's test at step 10, because then a training map is
the only candidate left. Two changes if it runs: offsets only on subcomponents active on maths rows (g > 0.1 at some
position), which should cut the 73 GB several-fold and may fit an H100; and Q0's readings as its pass test, not
the per-state adversary. If Q1 passes, this run is not needed for the weak version and the money goes to Q5.

### Q5. GRPO on GSM8K (the held RL plan, ~$20) — after Q2 and Q3

Runs if Q2's two arms differ in Q3's readings, to see which side real RL falls on, or if they do not differ, to test
whether the RL objective adds anything the data source does not. The shuffled-reward arm is a pure-amplification
positive control on Qwen (Spurious Rewards), not a null: its accuracy change is read before it is used as a reference.
δW goes through Q1 and Q3's tables unchanged.

### Later, not now

- A second task for generality. The best candidate stays "make a prompted behaviour the default" (reply in lowercase
  or French by context distillation): broad, whole-response, deterministically graded, with the prompted W₀ as the
  reference for which subcomponents the behaviour uses. Jailbreak refusal stays on the backlog (user, 2026-10-27).
- Brittleness across decomposition seeds cannot be tested on Qwen: there is one decomposition. Say so in write-ups.
- The paper's own numbers for this decomposition are not public; W₀'s readings are the only reference.

## 5. Rules for the line (proposed)

1. **A heuristic question gets one hour, $0 and one report**, answered from the literature and numbers we already
   have. No new run unless the user asks for one after reading it.
2. **One question per run, and the run reads a saved state when one exists.** A plan whose two branches lead to the
   same next step is not run.
3. **Masked KL and behaviour-kept first; the 20-step adversary at most once per experiment**, paired on saved
   sources. The adversary is a check, not the per-state pass test.
4. **Every update is reported with its KL to start on task rows**, and arms are matched on it, not on step count.
5. **Two amendments per experiment at most** (user, 2026-10-27); the third is a new plan.
6. Gated runs (`.claude/CLAUDE.md`) stay as they are; the plan above keeps every run under the gate except Q4 and Q5.

## 6. The user's answers (2026-10-28, chat)

1. **Pass test:** Q0's three readings are the Qwen pass test; the 20-step search runs once per experiment.
2. **Canonical update:** SFT on the model's own correct samples; gold SFT is the off-policy contrast at matched KL.
3. **Teacher training on the H200:** on hold until Q1 is read; runs only if the least-squares split fails.
4. **The goal (user's words):** "sharp preliminary results in a couple of weeks time. for example, the ability to
   cleanly decompose post-training updates on a decent scale and the ability to this both for SFT and RL and some
   evidence showing we can distinguish elicitation versus new mechanisms based on a VPD decomposition."

## 7. The goal, and whether the plan reaches it by about 2026-11-11

Three deliverables, and what in the plan produces each:

| deliverable | produced by | state of the evidence it rests on |
|---|---|---|
| **D1. Cleanly decompose a post-training update at scale.** A map Φ that keeps the decomposition faithful for a real update on Qwen3-1.7B, with a test that can fail | Q0 + Q1 on the saved δWs; Q2's δWs through the same table | The least-squares split is exact on Qwen (complete spans); whether it *passes* is unknown. Fallback: teacher training (Q4) |
| **D2. For SFT and for RL.** The same table for an SFT update and an RL update | Q2 (SFT arms) and Q5 (GRPO) | SFT code works. **RL code does not exist** and is the schedule risk; the own-samples arm is the RL-like stand-in until it does |
| **D3. Elicitation against new mechanisms, read from the decomposition.** Readings that separate two anchor updates whose nature is known by construction, then applied to the GSM8K updates, whose nature is the question | Q2 with **two anchor arms** (added below), Q3's readings on all arms | The 67M showed re-gating dominates for narrow tasks (E4, E6, E18) and that a cipher the model lacked can be copied (E42). Nothing at scale yet |

**What D3 needs: anchors, not labels.** Whether SFT or RL on GSM8K builds anything new in this model is the
project's question (`GOALS.md`: "teaches new capabilities rather than strengthening existing ones"). The literature
leans one way (RL's Razor, Spurious rewards, Nikolich), but that is a prior on a different model size and regime, not
a result, and the plan must not assume it. An earlier draft of this section wrote "GSM8K gives the elicitation side"
as if settled; it is not, and a 2026-10-28 chat correction from the user is why this section reads as it does now.
So the GSM8K arms (own-samples SFT, gold SFT, GRPO) carry **no label**. Their readings are the finding, whichever way
they fall. What makes the readings mean something is two anchor arms whose nature is known **by construction**, run
through the same table:

- **Existing by construction: context distillation.** Pick a system prompt that raises no-think GSM8K accuracy at W₀
  (measured first; if none does by more than noise, use think-mode answers distilled into no-think instead). Train
  the model to give the prompted answers without the prompt. A behaviour the frozen model already produces under a
  prompt is in the model already; the update can only route to it. Same recipe, matched on KL to start. ~$4.
- **New by construction: invented facts.** About 1,000 question-answer pairs about entities that do not exist,
  chat-formatted, graded by exact match on held-out rephrasings. The model cannot have these; the update must put
  them somewhere. Same recipe, matched KL. This is the findings-to-port analysis's "new facts" row and the realistic
  analogue of the 67M's cipher (E10, E42). Known facts the model already answers mixed in as the per-example
  control (Gekhman et al. 2024). ~$4.

**What "distinguish" looks like, filed before the runs.** The predictions below are for the two anchors only. The
table is validated if the anchors separate on a row; that row is then read on each GSM8K arm, and the GSM8K arm's
nature is reported as "reads like the existing anchor", "reads like the new anchor", "between", or "like neither",
with the numbers. No GSM8K arm has a predicted column.

| reading | existing anchor (context distillation) | new anchor (invented facts) |
|---|---|---|
| direct against re-gating share (route split on task rows) | re-gating dominates; direct small | direct share larger |
| the switched-on set's overlap with the think-mode set | above chance | at chance |
| where the split puts δW | few alive subcomponents, concentrated | spread across more subcomponents, MLP matrices, including ones Γ never gates on for this text |
| Q0 behaviour kept under CI masks | high for the split | lower: the new behaviour is partly in components the masks do not keep |
| Q0 pass of the least-squares split | passes | passes less, or needs teacher training |

Three outcomes, each sharp. If the anchors do not separate on any row, VPD's readings do not see the difference, and
the write-up says so; the GSM8K arms are then not interpreted on this axis. If they separate and the GSM8K arms read
like the existing anchor, that is evidence for elicitation, with GRPO and own-samples SFT compared against each
other. If a GSM8K arm reads like the new anchor on any validated row, that is the result the project was set up to
be able to find, and it is reported as such, not explained away. One rule for that reading (user, 2026-10-28, from the 67M phase):
a failed split is never, on its own, evidence of a new mechanism. The 67M line had updates on behaviour the model
plainly already had that still left a large share outside the subcomponents; brittleness and re-gating both produce
that. "New" is claimed for a GSM8K arm only when it reads like the new anchor on rows that do not depend on the split
passing: Γ switching and the think-mode overlap, behaviour kept under CI masks, and where δW lands relative to what Γ
gates on for this text. A failed split with those rows reading "existing" is the brittleness branch of `GOALS.md` and
is reported as that. Symmetrically, D1 failing on an arm is a positive result only when this rule is met, so
teacher training (Q4) is not used to rescue a pass before the Q3 rows on that arm are read. The RL arm in particular is the one where the
literature's prior is weakest for a 1.7B model trained in no-think mode, so it is the arm to watch.

**Timeline, two weeks from 2026-10-28.** Each run stays under the gate; nothing waits on a rehearsal cycle.

| days | work | cost |
|---|---|---|
| 1-2 | Q0 + Q1 on the saved δWs, one pod in US-MO-1 | ~$8 |
| 3-5 | Q2, four arms (own samples, gold, and the two anchors: context distillation, invented facts), matched on KL; Q1's table on each. **In parallel, from day 3:** a writer builds the GRPO loop (TRL's trainer, thinking off, 512-token cap) and rehearses it at tiny size on the cluster for $0 | ~$18 |
| 5-7 | Q3's readings on all arms (route split, Γ switching, think-mode overlap, where δW lands) | ~$5 |
| 7-10 | Q5, GRPO, ~100 steps, δW saved at the same steps; Q1 and Q3 tables on it. Q4 only if Q1 failed | ~$20 (+$30 if Q4) |
| 11-13 | Write-up: one figure per deliverable, the Q0 controls beside every pass; buffer for one rerun | — |

Total about $50-80 over two sessions' caps. Feasible if the line runs the table above and nothing else: no
amendments beyond a rerun of a failed stage, no new side questions.

**Risks, in order.** (1) Q0's controls fail to fail: no reading has teeth, and a day goes to settling the criterion
with the user before D1 can be claimed. (2) The RL loop: new code, rollouts at 1.7B; mitigated by starting it on day 3
and rehearsing for $0. If it slips, D2 ships with the own-samples arm as the RL stand-in and says so. (3) The
decomposition is weaker on maths text than on web text (adversary 4.24 against 1.70), so every D1 claim is a ratio to
W₀'s own reading on the same rows, never an absolute. (4) The own-samples update may lengthen answers toward
thinking-style text past 512 tokens; the drop rule and length readings catch it at the first saved steps. (5) The
invented-facts arm may not generalise to rephrasings at 100 steps; the in-format held-out questions are the primary
score and rephrasings are reported.

## 8. Task for the qwen-decomp lead

> Read `reports/plans/2026-10-28_plan_qwen-line-replan.md`. The user has taken its recommendations 1-3 and set the goal
> in §6-7. E40 Amendment 1 is on hold (do not start the H200 run; keep its code). Do not delete the old volume
> `ydavm7ct4x`: Q1 reads E40's δWs from it. Write one plan for Q0 + Q1 together, on one H100 pod in US-MO-1, under
> $10: the three readings, the two positive controls, the free singular-subspace check, and the least-squares and
> adjoint splits of δW at steps 10 and 100. Predictions filed before the run. Then Q2 with the four arms of §7:
> the two GSM8K arms carry no elicitation-or-new label; the two anchors (context distillation, invented facts) are
> the only arms with predicted columns, and the GSM8K arms are read against them. Do not write "GSM8K is the
> strengthening case" anywhere; `reports/analyses/2026-10-27_analysis_qwen-gsm8k-rl-headroom.md` §"How it bears on
> the project" does, and a one-paragraph review report should say why that inference is withdrawn.
> Rules of §5 apply: a heuristic question gets one hour and no run; one question per run; the adversary once per
> experiment; two amendments at most. Report each result in plain words with the control beside it.

## Not in this plan

Thinking mode as a training setting; a second decomposition; the sparsity loss as a reading; a Γ refit on Qwen; any
task beyond GSM8K and the two anchor arms of §7.

## Sources

- Repo: `reports/analyses/2026-10-27_analysis_qwen-{port,loader,findings-to-port,task-choice,gsm8k-rl-headroom}.md`,
  `reports/plans/2026-10-28_plan_qwen-{sft,rl}.md`, `experiments/qwen_sft/results/`, `decisions.md` rows 33-51,
  `reports/decisions/2026-10-19_decision_success-criterion.md`.
- Angsten, "VPD's evaluation adversary finds systematic defects" (LessWrong, 2026-10-20): adversarial sources carry
  over to unseen text; shuffled sources do little.
- Thinking leakage in no-think RL on Qwen3-1.7B/4B/8B (arXiv 2610.14270): the gain comes through a low-dimensional
  think subspace; swapping it back removes most of the gain. Liu & Honavar (arXiv 2609.28682): steering along a
  base-derived direction reproduces 42-79% of the post-training gain.
- Shenfeld et al., "RL's Razor" (arXiv 2509.04259): forgetting tracks task KL, not algorithm. Chen et al., "Retaining
  by doing" (arXiv 2510.18874): on-policy data, not the RL objective, is the cause; iterative SFT on own samples
  nearly matches RL.
- Nikolich et al. (arXiv 2606.23740): SFT, RFT and RIFT deltas nearly colinear; offline GRPO adds an orthogonal part
  in late layers. Shao & Wu (arXiv 2505.20993): o_proj changes most in reasoning fine-tunes.
- Shao et al., "Spurious rewards" (arXiv 2506.10947): random rewards raise Qwen accuracy; the shuffled arm is not a
  null on Qwen.
- Thinking Machines, "LoRA without regret" (2025): rank-1 LoRA matches full fine-tuning for policy-gradient RL.
