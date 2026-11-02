# Plan (DRAFT, Fable's advice): after E59, the readout at the table's vocabulary, or pause the line

**Status: option 1 (E62) APPROVED by the user 2026-11-02 (dialog, ask #9). Drafted by Fable after E59 closed unreadable. Before coding: the E59 check's M1 (model's own per-window ranks through the P3 rule) must run and E62's P3-style rule be set from it.**
(`reports/analyses/2026-11-02_analysis_e59-jain-readout.md`). Terms as in `GOALS.md`; r = the route split's re-gating
share; "prior" = W0's probability of the trained value after its key.

## Where the line stands, in plain words
r tells E54's half-known map from its cipher at five seeds (0.76-0.77 vs 0.21-0.27), where one steering vector and
neuron patching do not (E54, E56). E59 tried to give "half-known" a label that uses neither VPD nor W0's output, with
Jain et al.'s Def 1 readout over all 50,277 tokens, and the readout memorised its 15M Pile tokens and failed the
construction check; its causal half (does each fine-tune's gain run through the direction W0 uses?) never ran. Two
things happened elsewhere since the line's last plan. E58 Amendments 6-7 found r follows whether the trained answer is
W0's own guess (rank correlation with log prior 0.95 over eight maps, with carry 0.29), so on the 67M r has not shown
anything the logits do not. And on 2026-11-02 the user fixed what "elicitation" means (`decisions.md` row 112):
mechanistic, "the fine-tuned behaviour runs through a mechanism W0 already has"; "the answer is one W0 already gives"
is rejected, and r is "a candidate instrument to score against a causal check". E55 C2 (reuse line, `normal` QoS,
$0) is about to score r against a reuse-by-construction truth set on the counting model.

## Options
| | what it learns | cost, platform | verdict |
|---|---|---|---|
| (1) Def 1 readout restricted to the pair's 64 values, then E59's removal test on the twelve saved states | whether some block before the last holds the elicit values (key-specifically) and not the cipher's; whether the elicit gain runs through that direction and the cipher's does not: the causal check row 112 asks for, on the one pair r is claimed on | one MATS debug job, ~15 min, $0; ~80 lines on `src/commute/readout.py` (64-way loss, balanced positions, a specificity guard), `run.py` otherwise unchanged; fresh-agent code review | **recommended** |
| (2) graded elicit ladder: tables over the same keys at W0 rank bands 2-5 / 6-20 / 21-200 (`elicit.pick_values` takes the band) | whether r rises with how well W0 knows the value | one debug job, $0 | declined: E58 A6-A7 already read it (r vs log prior 0.95 over eight maps, two levels of elicit prior), and it measures definition (i), which the user rejected 2026-11-02 |
| (3) pause the line; fold the result into the claims | nothing new; the claim reads "r separates the pair where activation baselines do not, and so does W0's prior; a mechanism-level label is E55 C2's and the Qwen KV table's job" | $0 | the fallback if the user would rather wait for C2 |

Not weighed further: the same full-vocabulary readout on 4-8x the tokens (1-2 L40 hours on `normal`, Qwen-only, or
~$5 paid) buys only what (1) buys for $0; a second key draw (low information, 777 keys qualified); the pair on Qwen
(waits on row 112's confirmation and the Qwen line).

## Recommendation: (1), one $0 job; otherwise (3)
It is the direct consequence of row 112 for this line: the pair's label becomes "W0 holds the value linearly before the
output, and the fine-tune's gain uses that direction", and r is scored against it. Jain et al.'s readout has a
vocabulary of a few hundred; E59 showed a 50K one cannot be fitted at this budget, and the 64-value one removes
exactly that departure. It is weaker than C2's truth set (a direction, not a circuit; the granularity limit stays),
but it is on the pair the line's claim rests on and the pair the Qwen table would carry. If the user prefers to wait
for C2, (3) costs nothing and the folded wording above goes into the write-ups now.

## Design check (`/design`)
- *Goal:* GOALS "teaches new vs strengthens existing": whether r's reading of this pair agrees with a causal label
  that uses neither VPD nor W0's output ranks.
- *Nearest tried:* E59 (same apparatus, full vocabulary, readout unfit, causal half unrun); E58 A6-A7 (r vs prior,
  behavioural). Neither ran a causal check of r's reading on the 67M. E55 C2 will, on another model and task.
- *Patch or step:* a patch on E59's apparatus (the vocabulary), running the step E59 was for (the removal test).
- *Decisions:* relies on row 81 (Def 1) and row 112 (mechanistic elicitation; r scored against a causal check).
  Departs from none. The vocabulary departure from Jain is removed, the balanced sampling is a new stated departure.
- *Changes next:* P1 and P2 hold: the pair's label is causal and VPD-independent, write up, and the Qwen KV table
  carries it. P1 holds only at the last block: "half-known" means "what the output says"; r's separation of this pair
  is a reading of the prior, written so. P2 fails (removal costs both kinds alike): r's reading of this pair has no
  causal footing; the line pauses on the 67M and C2 decides r's standing. Any control fails: stop, report.
- *Control first:* stage 0 below, alone, early stop.
- *Class:* none; a new readout definition, so this draft is the ask.

## Claim, null, predictions (E62 if approved)
**Claim.** Some block before the last holds the elicit values, key-specifically, as a linear readout trained on the
Pile over the table's 64 values, and not the cipher's; removing that direction costs the elicit fine-tunes more of
their gain than the ciphers. **Null.** The elicit values are held only where the readout is the output (last block),
or the removal costs both kinds the same share: the label is the construction and r reads the prior.

**Apparatus.** One readout per block, d → 64 (the 32 elicit and 32 cipher values; unique by construction), trained on
W0's residual at Pile positions (rows 32768-65535, never the task rows) whose next token is one of the 64, balanced:
up to 2,000 positions per value, a 64-way next-token loss, AdamW, Jain's schedule as in `probe.py`; activations
cached once, so training is seconds. The number of positions per value is reported; a value under 200 is "unfit".
*Rank of a value* = its rank among the 64 at the answer position of the 97 held-out task windows, lower median over a
key's windows. *Specific* iff the readout's P(value) at the key's windows is at least 3 times its median over the
other 31 keys' windows (E54's 10x rule, loosened for the 64-way renormalisation; the last block's measured ratio is
reported beside it). *Possessed at block l* iff at least 24 of 32 keys have median rank ≤ 5 and are specific.
*Removal share A* exactly as E59's amended rule: at l* (earliest block before the last possessing the elicit map,
chosen on the fit half), remove the target value's readout row from the residual at the answer position; A = the
state's log-odds drop net of W0's drop, over the state's log-odds gain, scoring half. Separation: E51's rule, gap ≥ 0.2.

| number | W0 | do-nothing | control | if claim | if null |
|---|---|---|---|---|---|
| possession per block, elicit vs cipher | the reading | — | last block: elicit possessed and specific, cipher not (P3) | elicit at some block ≤ 2, cipher at none | elicit at block 3 only |
| A at l*, elicit vs matched cipher (5 seeds each) | elicit values: W0's log P falls ≥ 0.3 nats median | random unit direction: \|A\| ≤ 0.05 | last-block removal on every state reads ~1 (reported) | elicit A > cipher A by the rule; own-step cipher within 0.1 | no separation |
| readout fit | 64-way held-out loss within 0.1 nats of W0's own 64-way restricted loss at the last block | — | — | passes | — |

- **P1** elicit possessed at block 0, 1 or 2; cipher at none. Lean medium. Blind to non-linear encodings and to a
  direction the model does not read; the specificity guard is what keeps a unigram-frequency readout (" of", " it"
  everywhere) from passing.
- **P2** A separates elicit from the matched cipher by the rule; own-step cipher within 0.1 of matched. Lean medium.
  Blind to a route that re-encodes the value off the readout direction; floor: the random direction.
- **P3** the last block reproduces the construction: ≥ 24 elicit keys at rank ≤ 5 and specific, ≥ 24 cipher keys at
  rank > 20 of 64. Lean high; a stop.

**Edge cases.** An unfit value (fewer than 200 Pile positions): its key counts as not possessed for both clauses, and
if more than 8 elicit values are unfit P1's elicit clause is void (reported, the cipher clause stands). No block ≤ 2
possesses: P1 fails, P2 void (at the last block the readout is the head restricted). Ties in rank: the earliest
block. G under 0.05 nats: A is nan (none of the twelve states is near). W0 control failing (log P0 falls < 0.3): the
direction is decodable but unused, P2 void, fall back to Jain's reverse fine-tune in a later ask. Readout fit missing
its 0.1 bar: recorded, and the same reading rule as E59's last amendment (a pass stands; an elicit-clause or P2 miss
reads "unreadable"), hard cap one training run.

**Control first.** Stage 0, alone: sha256 and G of the twelve states and W0's rates reproduce E54/E56 (as E59, exact);
positions per value; the readout fit; P3; the random-direction share on W0 within 0.05. Any miss stops the job before
a removal is scored.

## Departures from the literature (Jain et al. 2024, Def 1)
- **Vocabulary:** the paper's readout covers its task's few hundred tokens; ours the pair's 64 values (E59 used all
  50,277 and could not be fitted).
- **Training data:** the paper draws fresh samples from pretraining; we draw a balanced sample of Pile positions by
  next token (so the readout reads content, not frequency), each token seen less than once.
- **Position:** read at the answer position after marker and key; trained at Pile positions, not task windows.
- **Thresholds:** rank ≤ 5 of 64 for 24 of 32 keys, specificity 3x, the 0.1-nat fit bar: ours, filed before the run.
- **Causal check:** removal of the readout direction (`decisions.md` row 81's addition), not the paper's pruning or
  reverse fine-tuning.

## Cost, routing, what would change this
One MATS debug L40 job, ~15 min, $0, through `pod-runner`; code review by a fresh agent before the write-up. If the
user prefers to wait for E55 C2's reading of r, take (3). If row 112's restatement is not confirmed, the ask is
withdrawn until it is. If C2 finds r does not read reuse on the counting truth set, (1) is not worth its slot and the
line pauses.

*Hand-off: a new session takes this file, E59's analysis and `src/commute/readout.py` once the user has approved or
amended it.*
