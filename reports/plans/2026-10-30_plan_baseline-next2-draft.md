# Plan (DRAFT): a label for "half-known" that does not come from VPD or from the model's own output (E59 proposed)

**Status: DRAFT, Fable's advice; nothing runs before the user's approval.** Written after E56
(`reports/analyses/2026-10-30_analysis_e56-keydependent-patch.md`), whose decision tree ended on "ask (a)" of
`reports/plans/2026-10-30_plan_baseline-next-draft.md`.

## Where the line stands, in plain words
On one pair of fine-tunes over the same 32 keys, the route split's re-gating share r reads the half-known table at
0.76-0.77 and the cipher at 0.21-0.27 on five seeds, while one steering vector and neuron patching do not separate them
once learning progress is matched. The claim, "the route split reads whether an update strengthens what the model
half-knows, where cheap activation baselines do not", now has two weak points. The "half-known" label is the original
model's own output ranks (the value sits at rank 2-20 after the key), so a sceptic can say the pair was built to make r
read high. And the claim rests on one pair in one 67M model.

## Options weighed
| | what it learns | cost, platform | status |
|---|---|---|---|
| (a) Jain et al. Def 1 readout per block + a causal check on the saved states | whether a readout trained on pretraining data, never on the tables, holds the elicit values at some block and not the cipher's; and whether each fine-tune's gain runs through the direction the original model used | one cluster debug job, ~15 min, $0; ~150 new lines | needs confirmation: a new measurement on this line |
| (b) a graded ladder: three tables over the same keys at rank bands 2-5, 6-20, 21-200, plus the cipher | whether r rises with how well the original model knows the value, a dose-response no single pair gives; the label becomes the measured prior log P₀(value) | one debug job, $0; `elicit.pick_values` takes the band already | new tables, needs confirmation |
| (c) a second key draw (seed 1 of the 32 keys), same construction | whether the separation is a property of this draw | one debug job, $0, no new code | class 2; low information: 777 keys qualified and the seed range is 0.055 |
| (d) held-out P(target) at matched size scored as a reader | E56's check noted it separates the pair (0.95 vs 0.27-0.31). It is learning speed, Gekhman et al. 2024's behavioural signal, and needs the trajectory, not one state | $0, no run: a paragraph in E56's report and the claim reworded "from one state" | class 1; do it with whichever option runs |
| (e) the pair on Qwen | the headline on a real model | gated, $15-20 | waits on r for Qwen (Q3) |

## Recommendation: (a), with (d) folded into its write-up; (b) is the next ask if (a) holds
(a) answers the sharper objection: it gives the label a definition the user fixed for the project (`decisions.md` row
81, Jain et al. 2024 Def 1) that uses neither VPD nor the output the pair was selected on, and its causal half tests
reuse rather than decodability. It runs on the fifteen saved states, so the fine-tune side costs nothing. Its honest
limit: at the last block the readout is close to the model's own output, so possession there confirms the construction;
the information is in the earlier blocks and in the ablation. (b) is the better second pair than (c) and is cheap, but
it keeps the label behavioural, so it comes after (a). (c) is not worth its slot. (e) is another lead's work.

## Design check (`/design`)
- *Goal:* GOALS "teaches new vs strengthens existing": whether the decomposition's reading of kind agrees with a
  definition of "existing" that does not use the decomposition.
- *Nearest tried:* E54 stage 1 (the behavioural label), E55 Stage A (Def 1 probes on minGPT, `src/commute/probe.py`,
  reusable: a per-block linear readout to the vocabulary with Jain's schedule). Neither reads the 67M pair with a
  readout trained off the tables, and nothing has tested whether a fine-tune's gain runs through the direction the
  original model used.
- *Patch or step:* a step: the label moves from "what the output ranks" to "what some block linearly holds, and whether
  the update uses it".
- *Decisions:* relies on row 81 (Def 1 as the ground truth, plus a causal claim); the bounded-g departure of row 87 is
  about Def 2 and does not apply, because the readout never sees the tables. Departs from none.
- *Changes next:* if the readout and the ablation both say "elicit yes, cipher no", the label is independent and the
  Qwen pair is asked with it; if the readout holds the elicit values only at the last block, "half-known" means "what
  the output says" and the write-ups say so; if the ablation does not separate, r's reading of this pair has no causal
  footing outside VPD, and (b) is asked before any Qwen ask.
- *Control first:* stage 0 below, alone, with early stop.
- *Class:* none; a new measurement on this line, so this draft is the ask.

## Claim, null, predictions (E59)
**Claim.** The original model holds the elicit map at some block as a linear readout trained on the Pile, and not the
cipher; and the elicit fine-tune's gain runs through that readout's direction while the cipher's does not.
**Null.** The readout holds the elicit values only where it reproduces the output (the last block), and removing the
readout direction costs both fine-tunes the same share of their gain: the label is the construction, and r's
separation has no footing outside VPD.

**Apparatus.** One linear readout per block (d → vocab), trained on Pile windows at every position with Jain's recipe
as in `probe.py` (4K AdamW iterations, lr 1e-3 decayed x0.1 at 2K, 3K, 3.5K, batch 96), never on the task windows. Read
at the answer position of the 97 held-out task windows. *Rank of a value* = its rank under the block's readout, median
over a key's held-out windows. *Possessed at block l* iff at least 24 of 32 keys have median rank ≤ 20 (the elicit
construction's band). *Ablation share* A, on a saved fine-tuned state: at block l\* (the earliest block, excluding the
last, where the elicit map is possessed, chosen on the fit half), project out of the residual at the answer position
the readout's column for that window's target value; A = (log P_tuned − log P_ablated) / G on the scoring half.
Separation rule: E51's (gap over the larger seed range, at least 0.2), five seeds per kind.

| number | W₀ | do-nothing | control | if claim | if null |
|---|---|---|---|---|---|
| possession per block, elicit vs cipher | the reading | — | last block: elicit possessed, cipher not (the construction) | elicit possessed at some block ≤ 2, cipher at none | elicit at block 3 only |
| ablation share A at l\*, elicit vs matched cipher | elicit values: log P₀ falls ≥ 0.3 nats median (direction causal for W₀) | random unit direction: \|A\| ≤ 0.05 | — | elicit A > cipher A by the rule | no separation |
| last-block readout vs the model | its median elicit rank within the 2-20 band for ≥ 24 keys | — | — | passes | — |

- **P1 (possession before the last block).** Elicit possessed at block 1 or 2; cipher possessed at no block. Lean:
  medium. Blind to non-linear encodings and to directions the model does not read; floor: the readout's held-out Pile
  loss, reported, within 0.5 nats of the model's own at the last block.
- **P2 (the gain runs through the readout).** A on the five elicit states exceeds A on the five matched ciphers by the
  rule; the cipher at its own step reads within 0.1 of its matched A. Lean: medium. Blind to a route that re-encodes
  the value off the readout direction; floor: the random-direction share.
- **P3 (the construction reproduces).** The last-block readout gives the elicit values median rank 2-20 for ≥ 24 keys
  and the cipher values rank > 100 for ≥ 24 keys. Lean: high; it is stage 0's early stop.

**Edge cases.** No block ≤ 2 possesses the elicit map: P1 fails and P2 is void (at the last block the readout is the
unembedding, so projecting out the target's column empties both fine-tunes alike; reported, not scored). Shared values
(" of", " at" serve several keys): the direction is the value's, so nothing changes. A key whose value's rank ties
across blocks: the earliest block counts. G under 0.05 nats: A reads nan by E54's rule (none of the 15 states is near it).
The W₀ positive control failing (log P₀ falls < 0.3 nats): the direction is decodable but not used, P2 void; fall back
to Jain's own tool, a short reverse fine-tune on the Pile of the cipher state, in a later ask.

**Control first.** Stage 0, alone: sha256 of the 15 saved δW against E54/E56; the readouts' Pile loss per block; P3;
the random-direction share on W₀ within ±0.05. Any miss stops the job before an ablation is scored.

**Decision tree.** P1, P2 hold: the label is independent; write up, ask (b) and carry the label to the Qwen pair. P1
fails, P2 holds: "half-known" is the output's knowledge; the ablation still separates; write up with that wording. P2
fails: stop and tell the user; (b) before any Qwen ask. A control fails: stop, report.

## Departures from the literature
- **Def 1, Jain et al. 2024 §3.** The paper trains the readout on data from the pretraining process and reads it on
  the sub-domain of inputs carrying the instruction; we train on Pile windows at every position and read at the answer
  position after the marker and key. The paper gives no threshold for "equal to f_C"; ours is median rank ≤ 20 for 24
  of 32 keys, filed before the run, and matches the band the pair was built with.
- **The causal check is not the paper's.** Its tools are pruning a few neurons and reverse fine-tuning; the ablation of
  the readout direction is `decisions.md` row 81's addition. Reverse fine-tuning is not run here.
- **Readout schedule** copied from the paper (L1238-1248) as in `probe.py`; batch 96 and AdamW defaults filled in there.
- **The pair itself** (E12's recipe, matched size) and the separation rule are this project's.

## Cost and routing
One MATS debug job on an L40, ~15 min, $0, through `pod-runner`: readouts (4 probes over a small vocabulary, a few
minutes), then 15 states x 2 directions x 97 windows of forwards. New code: a Pile-window readout trainer (adapting
`probe.py`'s `Probes` to the 67M harness), one ablation hook, scoring; ~150 lines in `src/commute/` plus a short
`run.py`. A fresh-agent code review before the write-up, not before the run ($0).

## What would change this
If reuse-lead's Stage C ships a 67M readout first, reuse it and (a) becomes class 1. If the user prefers robustness to
label independence, run (b) first: it is as cheap and gives a graded prediction. If Q3 delivers r on Qwen, the Qwen pair
moves ahead of both, carrying whichever label exists.

## Not in this plan
Reverse fine-tuning; a second key draw; the graded ladder (b); Qwen; a per-key breakdown; which subcomponents switch on
for the elicit map.

*Hand-off: a new session takes this file, E56's analysis, `src/commute/probe.py` and `src/commute/neurons.py` once the
user has approved or amended it.*
