# Plan (DRAFT, Fable's advice): after E59, the readout at the table's vocabulary, or pause the line

**Status: option 1 (E64; first called E62, a code the re-gating lead holds) APPROVED by the user 2026-11-02 (dialog, ask #9). Drafted by Fable after E59 closed unreadable. Before coding: the E59 check's M1 (model's own per-window ranks through the P3 rule) must run and E62's P3-style rule be set from it.**
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

## Claim, null, predictions (E64)
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

## Amendment before filing (2026-11-02)
*The lead's decision, from a measurement.* E59's bars were absolute and the model itself failed them: on the 97
held-out windows W0's own ranks read 16 of 32 elicit keys in E59's band (the rule needs 24) and 17 of 32 possessed,
cipher 0 (job 48453, `experiments/jain_readout/results/m1_check.json`). So a perfect readout might miss "24 of 32".

*Old rule, replaced.* "Possessed at block l iff at least 24 of 32 keys have median rank ≤ 5 and are specific." P3: "the
last block reproduces the construction: ≥ 24 elicit keys at rank ≤ 5 and specific, ≥ 24 cipher keys at rank > 20 of 64."

*New rule.* The same 64-way rank rule (median rank at most 5 among the 64 values, plus the specificity guard) runs first
on W0's own logits restricted to the 64 values: that is the model's possessed set M, elicit and cipher, and the
control's reference. A block possesses the elicit map iff it possesses at least 3/4 of M's keys under the same rule.
The construction check (P3): the last block's readout possesses at least 3/4 of M, and the cipher's M is empty or at
most 2 keys. If elicit |M| is under 8, stage 0 stops: the pair cannot carry the reading at this rule, and the user
decides what next. Everything else in the plan stands.

### Writer's note, filing the rule (2026-11-02)
Rules left incomplete or in conflict, resolved in `experiments/readout64/results/predictions.json` and stated to the lead:
1. **Cipher possession.** The amendment defines possession for the elicit map only, and 3/4 of an empty M is vacuous. A
   block possesses the cipher iff it possesses at least as many cipher keys as the elicit map needs (3/4 of |elicit M|).
2. **The cipher clause of P3** had the same absolute problem: replaced by cipher |M| ≤ 2 and the plan's control cell
   "last block: cipher not possessed". The counts at rank > 20 of 64 are printed, not scored.
3. **W0's control** ("falls ≥ 0.3 nats median") read over all windows assumes W0 holds the value at most keys; it holds
   about half. Now the median over the scoring-half windows of M's keys, all windows beside it.
4. **Unfit values.** The edge case "over 8 elicit unfit: P1's elicit clause void" becomes a stage 0 stop (the lead's
   brief), for the cipher too: an unfit cipher key counted not possessed helps P1's cipher clause. Added: stop if M's fit
   keys are under 3/4 of M. Measured before filing: 2 elicit values unfit (13905, 147 windows, key 1495; 42344, 26,
   key 3907) and no cipher value, so neither bar fires unless M is small.
5. **The fit bar.** "Control first" says a fit miss stops; the edge case says recorded, with its reading rule. The edge
   case is followed, as E59's last amendment did: P3 reads the same readouts and does stop. One-sided: the readout may
   beat W0, whose natural prior is kept while the sample is capped.
6. **"Random-direction share on W0"** is undefined at W0 (G = 0): read on every state, net of W0, as E59 did.
7. **Held-out positions** were not named. One eighth of the training rows (61440-65535), capped at 250 per value: the
   cached Pile val rows hold under 50 positions for 18 elicit and all 32 cipher values.
8. **Two statements in the plan do not hold at the measured counts.** "Balanced" is partial: all 32 cipher values and 13
   elicit values sit under the cap of 2,000 (cipher median 346, so the sample tilts up to ~1.7 nats against them). "Each
   token seen less than once" is false: ~57,000 positions and 4,000 batches of 96 pass over each ~7 times. Kept as
   written, stated in the filing's departures and in P1's blind-to line.
9. **Risk to P3, flagged and not changed.** M is read with W0's natural prior over the values, the readout at the capped
   one, so a common value's rank can differ for that reason alone. The job prints W0's possessed set at the training
   prior, never scored, so a P3 miss can be read without a second job.
10. **Early checks** read a 50-batch mean less 3 of its standard errors, set in the run from the batches' own sd: step
    250 under 4.0 (chance is ln 64 = 4.16), step 2,000 within 0.5 of the model on the same batches.
11. **Cost.** ~8 min, not ~15: training on cached activations takes under a minute, so no train-then-`--readouts` split.

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

## After the code review (2026-11-02)
*The lead's decision on `reports/reviews/2026-11-02_review_e64-code.md`: all eight findings accepted. Nothing has run on
E64, so F1 and every other change apply before any reading, and no rule was chosen knowing an outcome. First-filed values:
`superseded_2026-11-02` in `experiments/readout64/results/predictions.json`.*

- **F1, cipher possession.** A fixed bar: at least 6 cipher keys (3/4 of `min_M`), whatever |M| is. The elicit bar stays 3/4
  of |M|. Reaches P1 and P3.
- **F2, row quality.** A second readout on at most 241 windows per value (the cipher's minimum), and A at the same block
  beside the scored one for both arms. P2's verdict is computed as before on the scored readout. A P2 pass where the same
  separation rule does not read "elicit higher" on the capped readout reads "not separable from row quality", and the
  branch says so rather than "the pair's label is causal". A nan there is such a miss. A P2 fail, void or unreadable keeps
  its reading whatever the capped readout reads. Stage 0 stops if the cipher's minimum is not 241. The sample-size exposure
  is in P2's `blind_to`. The capped rows are over-fitted for both arms (about 15,000 windows), and rare elicit values
  stay under the cap (26 and 147 windows at least); both are stated there.
- **F3, the fit bar.** Reads against W0 moved to the sample's prior, the shift of `W0_training_prior`, added to its held-out
  logits. W0 at its natural prior is logged beside it.
- **F4, l\*.** None unless |M_fit| is at least 6; P2 then voids and names the reason.
- **F5, W0's control.** On M's windows, W0's drop is also read under a wrong value's row (next index in the same task, same
  windows) and under the random direction. A pass needs the target row's median at the 0.3-nat bar and strictly above the
  wrong row's. A miss stops as before. The random direction no longer reads over all keys.
- **F6, memory.** `cache64` clones each batch's slice. Peak residuals fall from about 29 GB to about 0.7 GB on the GPU.
- **F7, step 250.** The bar is the entropy of the sample's values, computed in the run (about 3.8 from the filed counts),
  minus 0.2. A frequencies-only readout has a batch sd of about 0.08, so it reads about 0.03 under its own entropy after the
  3 standard errors and stops. A real run could pass, and that is not measured. E59's last block had covered 81% of its fall
  by step 250 at this learning rate (jobs 48279, 48357: 10.98 to 4.73 of an end at 3.23; 10.96 to 5.01 of 3.61). Starting
  near 4.3, the same share reaches the bar if the final loss is under about 3.5, about 0.3 under the prior. A readout
  ending above that is poor anyway. E59's batches held about 40 times more positions, so a batch here is noisier, and a false
  stop would cost one $0 debug job. The capped readout has no bars, since it only reads P2's row quality.
- **F8, `branch64`.** With P2 void, an earlier cipher block now gives the cipher reading, not the null's.

Each changed rule ran against its edge cases in `tests/test_readout64.py`, and `run.run` ran end to end on a random stand-in
network, with no real model or saved state. Expected time goes from about 8 to about 10 minutes (one more training, twelve
more removals). Left as filed: the step-2,000 bar reads W0 on the capped sample at its natural prior, F3's tilt, and it only
makes that bar laxer.
