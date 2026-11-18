# Plan: does some declared level read four hand-built edits of M as their construction says? (cell P1.31)

*2026-11-18, definition-lead. Cell P1.31 (opened from proposal 30), rung L1.4, 2 runs, due 2026-11-20. No E-code claimed, nothing run. Design check, the
edits' recipes, every expected value and the pre-mortem: `2026-11-18_plan_known-answer-levels_appendix.md`. Card and
draft rules: `experiments/known_answer_levels/{card.md,results/predictions.json}`.*

## The question, in plain words

The working definition says a fine-tune F reuses M, for a behaviour at a declared level, to the extent that the parts F
needs still compute what M's parts compute on the same inputs. At the neuron level its old score saturated (E208). The
user's test: build four edits of M whose answer is known by construction, read them at the neuron level and at three
coarser levels, and see whether some level gives back all four labels. Under the user's choice, a part keeps M's rule
by interchange: M's output for the part is put into F on the same input, and the part counts as kept to the extent F's
behaviour holds. If a level passes, the five trained pairs are read once at it, with no labels.

## Two points to settle before filing (definition, not measurement)

1. **"The parts F needs."** As the theory note formalises it (clause 5), F needs the fewest parts that must keep F's
   own mechanism while every other part gets M's. An unchanged part is untouched by that swap, so it is never needed.
   A circuit that is re-triggered but itself unchanged is then invisible, and a re-trigger reads only by its changed
   trigger part. Read plainly, "the parts F needs" would include the unchanged circuit the behaviour runs through.
   *Reading 1 (as filed):* only changed parts. *Reading 2:* every part the behaviour runs through in F, changed or not.
   Under reading 1 this test cannot pass with these edits (P1 below, derived, not guessed).
2. **"On the same input."** *Reading A (the note's starred choice):* the same input to the part, so M's part runs on
   F's own residual. *Reading B (Prakash et al.'s patching):* the same model input, so the part takes the value it had
   in M's own run. Under B, an unchanged circuit downstream of a re-trigger reads "not kept", because it did not fire in M.

The plan reads A and B under reading 1, the filed definition, and reports both. Reading 2 needs a rule for what an
unneeded unchanged part is replaced by. That is a definition choice, so it is not built here.

## The four edits (labels by construction; appendix §2)

All four use one window set D: 512 41-token Pile spans, each with a pair A_i B written earlier and a new cue A′_i
last. There are eight cue pairs (A_i, A′_i). M copies B after a repeated A_i through its induction circuit: head (1,1)
looks at the previous token and head (2,4) copies (E62: top-1 0.82; ablating (2,4) takes it to 0.01). M does not copy
after A′_i.

| edit | label | what changes | how |
|---|---|---|---|
| reuse, small | reuse | 8 quiet layer-0 MLP slots (E206's slot rule) | each slot detects one A′_i and writes x(A_i) − x(A′_i), the difference of their mean layer-1 inputs; the unchanged induction circuit then copies B |
| reuse, large | reuse | as above, plus head (2,4)'s output columns × g | g is set so the update's Frobenius norm over all changed weights equals the larger new edit's |
| new, few units | new | the same 8 slots, same input rows | each writes the direction that raises y_i, a random target, so A′_i → y_i is a lookup M lacks |
| new, thin | new | every layer-0 MLP output column | the least-norm change that writes the same eight directions on the eight cues |

Reuse, small and new, few units change the same weights in the same slots. They differ only in where the write goes:
into M's circuit, or straight to the output. That difference is all the labels encode.

## Reading, levels, controls

Levels, each with the identity alignment (ε = 0): **neuron** (each MLP neuron, each head, embedding, unembedding);
**head** (each head, each whole MLP layer); **MLP layer** (each attention layer, each MLP layer); **block** (each
block). For a part u, the share kept is r_u = 1 − KL(F ‖ F with u swapped) / KL(F ‖ M), read on D at the last
position. K comes from E208's search (τ 0.9, 5 orders, `commute.rho`), run over that level's parts. The item reads
R = mean r_u over K. R is this test's summary, not the definition's score.

- **Stage 0, alone, stops on a miss:** do-nothing (F a copy of M) reads "no change" at all four levels. F with nothing
  swapped matches F's native logits within 1e-3. M's induction top-1 reproduces E62's 0.8247 within 0.01 and falls
  under 0.3 when (2,4) is ablated.
- **Stage 1, the build checks, which also stop:** each edit reaches its behaviour on D (copy top-1 at least 0.8 × M's
  own; lookup top-1 at least 0.9). With every changed part swapped, F matches M within 1e-3 at each level. The band
  per level is 3 × the larger of the order spread and the split-half difference of R (256 + 256 windows), measured
  here.
- **Stage 2:** the four edits at the four levels under A and B. **Stage 3:** E195's rebuilt circuit at each level,
  reported, not an anchor.

**Pass rule (measurement choice).** A level recovers the labels when both reuse edits' R exceed both new edits' R by
more than that level's band. The test passes if some level does so under reading A.

## Predictions (filed before any reading; derivations in appendix §3)

- **P1.** At the head, MLP-layer and block levels, reuse small, new few and new thin each read R = 0 (|R| ≤ 1e-6).
  Their only changed part is layer-0's MLP or block, and swapping it gives M on D. Reuse large reads above 0, through
  the gain on (2,4) alone. No coarse level passes.
- **P2.** At the neuron level, new thin reads the highest of the four (R ≥ 0.99): the level misses the thinly spread
  new item, as the user predicted. Reuse small and new few each read about 1 − 1/8.
- **P3.** No level passes, so run 2 does not run. The line goes to the user as point 1 above.
- **P4.** Under reading B, reuse small reads no higher than under A at any level, and no level passes.

## Run 2, trained pairs (only if a level passes)

E196's five matched pairs at seeds 0 and 1: 20 states, all on disk (appendix §5). They are read once at the level
that passed, under A, with no labels. Four pairs are counted and parity/fact is reported, as in E196. Seeds 2 and 3
are not saved for every pair.

## Cost, routing, saved

Run 1 has 16 edit-level readings. Only new thin at the neuron level is large: 3,072 candidates, E208's chunked search.
That is about 20-30 min on an L40, so a cluster `debug` job at $0 if it starts at once. Otherwise a RunPod H100 SXM at
$4.40/h, about 45 min with set-up, about $3.50. Run 2 at a coarse level is about 10 min, under $1 on a pod. Saved:
per stage, a JSON of every r_u, K and band. The four edits' changed weights go in `results/large/` (about 2 MB) with
sha256. Nothing new is trained.

**Code needed** (appendix §4): `src/commute/levels.py` (a level as groups of `rho` parts; the search and r_u over
groups, about 120 lines) and `src/commute/known_edits.py` (the four edits and the slot, gain and scale solves, about
150 lines). Then `experiments/known_answer_levels/run.py`. `rho`, `copying` and `rule_retention` are called as they are.

## Not in this plan

Reading 2 (it needs the user's rule); a weighted or behaviour-share summary in place of R; E187's size-matched pairs in
run 2; any wording, band or cut-off added to the definition.
