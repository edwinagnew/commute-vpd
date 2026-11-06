# Independent check: does the neuron set separate known facts from unknown facts?

*2026-11-06, independent check for baseline-lead. Read only, $0. Script and output:
`~/commute-data/scratch/baseline-lead/e105-check/recheck.py`, `recheck.out` (stdlib only, through `cpu-run`).*

**Plain words.** I recomputed every overlap and verdict from the stored neuron sets. I also compared those sets with
sets from a random weight change and from unrelated runs, and read how the sets are ranked. The numbers and verdicts
match the stored files exactly. Splitting the known-facts rows into two halves leaves the set where it was, so which rows
were trained does not explain the gap. But the two fine-tunes also differ in topic, kind of entity and answer style, and
at the step that is read the fictional run has not learned its facts yet. So the evidence shows the set tells these two
datasets apart; it does not yet show that the gap comes from the model already knowing the facts.

**Verdict: holds with caveat.** The part "not explained by which particular rows were trained on" holds cleanly. The part
"separates facts the model knows from facts it lacks" is supported only as "separates TOFU's real-world rows from TOFU's
fictional-author rows". Topic, entity type, who wrote the answers and per-row exposure are not separated from that label.
And at the matched step neither arm has learned row-specific facts.

## 1. Re-derivation from the stored sets

I wrote my own overlap and both rules from the plan text, not by importing `score.py` or `amend2_neurons.py`. All 24
overlaps and all four verdicts agree to machine precision with `experiments/rows_null/results/score.json` and
`experiments/known_answer_pair/results/neurons_amend2.json`.

| row set | within-half seeds, A / B | four cross-half pairs | cross-arm max x | band | E105 verdict | Amendment 2 verdict |
|---|---|---|---|---|---|---|
| real | 0.857 / 0.869 | 0.818 to 0.842 | 0.492 | 0.143 | label | beyond seed noise |
| TOFU | 0.916 / 0.905 | 0.875 to 0.893 | 0.637 | 0.095 | label | beyond seed noise |

The margins are large. On the real rows the lowest cross-half overlap sits 0.33 above x against a band of 0.14. Every
cross-half pair sits within 0.04 of the within-half seed overlap. The split is disjoint, 108 plus 109 rows covering all
217 rows. All eight sets hold 1,024 neurons. The log of job 53480 shows every guard passed. Every arm's matched step is
32, the step the full-row arms were read at. The C2n re-read gave the stored gain and set exactly. The predictions file
was committed at 05:33, before the results commit at 05:49. The filed lean was "rows" on the real rows, and the run read
"label": a reversed lean, which the analysis should say plainly.

## 2. How a carrying set is built, and what could force agreement

`commute.qwen_neurons.rank_batch` and `rank_state`, called through `commute.known_facts.carrying`:

- **Rows and positions.** The real probe is pair 0 of all 217 real rows, TOFU's own question and bare answer, packed into
  29 rows of 256 tokens. Every answer token counts; the 217 rows were all kept. The TOFU probe is E92's 128 packed rows of
  fictional questions, TOFU's own wording.
- **Score.** For each neuron, summed over every answer position of the probe, the score is W0's gradient of the summed
  answer log P, multiplied by the neuron's firing change. The firing change is W0's own gate and up weights applied to the
  MLP input the fine-tuned state produces, minus W0's firing. Upstream changes therefore count too.
- **Cut.** The top 1,024 of 172,032 neurons by score among those with a positive score; ties broken by index.

Things that could make two runs agree by construction:

- **The same W0 gradient sits in every score.** This is real and measurable. A Gaussian random weight change shares 0.21
  to 0.23 of every real-row set and 0.27 of every TOFU-row set. Chance for two random sets is 0.006, or 0.011 with the
  layer profile matched. So about a fifth to a quarter of any set is fixed by W0's gradient alone, whatever the update.
  It is shared by known and fictional sets alike, so it raises every overlap by a similar amount. It cannot produce the
  0.35 gap between the known pairs and the cross-arm pairs.
- **Shared rows in the probe.** The real probe holds every row a known arm trained on, in TOFU's own wording, and no
  fictional row. The TOFU probe holds fictional rows the fictional arm trained on. So in each probe one arm reads its own
  training rows and the other does not. The halves test this directly: on the real probe each half has 108 of its own rows
  and 109 it never trained. Halves still agree at 0.82 to 0.84, so membership in the probe does not move the set much
  within the known label. That weakens this objection for the cross-arm gap as well, though it does not remove it.
- **A generic core that any fine-tune of this format moves.** About half of each known set is shared with both fictional
  sets: 380 of the 703 neurons present in all six known sets on the real rows, and 553 of 805 on the TOFU rows. Remove
  every neuron that sits in both fictional sets. The remaining known-specific neurons still agree across halves:
  412 of 569 on the real rows and 317 of 420 on the TOFU rows, overlaps of 0.72 and 0.75. The agreement is not only a
  shared generic core.
- **A few top neurons dominating.** No. Overlap of the top 32, 128 and 512 between halves A42 and B42 is 0.78, 0.82 and
  0.82 on the real rows, flat down to the full 1,024. Known against fictional is 0.47, 0.55 and 0.49 at the same depths.
  The pattern holds across the whole ranking.
- **The matched step inside warm-up.** All six known states and both fictional states sit at steps 29 to 33 of a 250-step
  linear warm-up, at the same rate. That does not force known runs to agree, since fictional runs share it. But it does
  mean these are very early updates. Each half gains about as much on the other half's rows as on its own: 1.43
  against 1.63 for A42, and 1.52 against 1.51 for B42. So at this step the known updates carry little that is
  row-specific. The rows null therefore tests a regime where rows were never likely to matter much. E91 found the
  fictional state had learned none of its facts by this step either.

## 3. Chance level and the generic component

Overlap matrix on the real rows, from `recheck.out`. Each value is the share of the row set's neurons inside the
column set. "Bare" is E103's first known-facts run, which has the same rows and label but bare answers in pair 0.

| set | halves, 4 | full known, 2 | bare known | fictional, 2 | Gaussian change |
|---|---|---|---|---|---|
| halves | 0.82 to 0.87 | 0.83 to 0.86 | 0.70 to 0.76 | 0.46 to 0.51 | 0.22 to 0.23 |
| full known | 0.83 to 0.86 | 0.864 | 0.69 to 0.70 | 0.47 to 0.49 | 0.21 |
| fictional | 0.46 to 0.51 | 0.47 to 0.49 | 0.39 to 0.40 | 0.899 | 0.24 |

The reference points, in order: two random sets 0.006, or 0.011 matched by layer. The Gaussian random change sits at
0.21 to 0.27, the floor W0's gradient sets. Known against fictional reads 0.46 to 0.51, or 0.61 to 0.65 on the TOFU rows.
Same label with one changed answer form, bare against sentence, reads 0.69 to 0.76, or 0.80 to 0.84 on the TOFU rows.
Same label with different rows reads 0.82 to 0.84, or 0.88 to 0.89, and seeds read 0.86 to 0.90.

The most useful comparison for the claim is the bare known run. Changing only pair 0's answer form, one of five
wordings, with rows and label fixed, moved the set further than replacing every trained row did. It also moved the
known set away from the fictional sets, to 0.39 to 0.40 from 0.47 to 0.49. So the set is sensitive to answer style.
After the format match, some style differences between the two arms remain, and those could carry part of the remaining
cross-arm gap.

No stored set from an unrelated fine-tune is on these probes. E102's template fine-tune sets, `a1_s0` and `a1_s1`, were
ranked on their own template rows, so they are not comparable. The Gaussian change is the only non-TOFU reference, and
its norm is not matched: it was scaled to the template's KL, at 180 times its norm.

## 4. Confounds between the two arms besides rows and label

| difference | known arm | fictional arm | ruled out? |
|---|---|---|---|
| which rows | 217 real rows, or 108 / 109 | 787 fictional rows | yes for the known arm, by E105 |
| row count | 108, 109, 217 | 787 | partly: 108 against 217 does not move the known set; 787 is untested |
| exposure per row at the matched step | 0.47 epoch, or 0.95 for halves | about 0.12 epoch | partly: 0.47 against 0.95 does not move the set; 0.12 is untested |
| step and size | step 32, norm 0.48 | step 29, norm 0.477 | yes, matched by design |
| topic and entity type | real people, places and works; world facts | invented authors' biographies | no |
| pair 0 answer author | Haiku sentences written to hold TOFU's bare answer | TOFU's own GPT-4 answers | no; the bare-run comparison shows answer style moves the set |
| other four wordings | Haiku clerks, E103's brief | Haiku clerks, E84's brief | no; same writer, different briefs and rows |
| answer length and form | 5 to 18 word sentences | TOFU's own answer sentences, lengths not compared here | no |
| prior familiarity | W0 picks 0.79 in multiple choice | W0 cannot know them | this is the label itself |
| probe membership | real probe is its own rows | TOFU probe is its own rows | partly, as section 2 says |
| fictional rows split | not run | not run | no: the same-label null exists for the known label only |

So E105 rules out row identity, and row count and exposure within a factor of two, for the known arm. It does not
separate the label from topic, entity type or answer style. Those all differ between the arms and all go with "known".

## 5. What the claim can say

Safe: "At matched size inside warm-up, the 1,024-neuron carrying set of a fine-tune on TOFU's real-world rows agrees
with itself across seeds and across disjoint halves of its rows, at 0.82 to 0.87. It shares only about half its neurons
with the set of a fine-tune on TOFU's fictional-author rows, well beyond seed noise. A random weight change shares about
a fifth to a quarter with either set."

Not yet safe: that the separation comes from the model already knowing the facts. Topic, entity type and answer style
are not controlled, and at this step neither run has learned row-specific facts. A known-label run on fictional-style
data, or a fictional run on real-style data, would be needed to tell these apart.

## Not run

- A same-label rows split for the fictional arm.
- Any fine-tune on a third dataset of the same format read on these probes. With one, the share common to "any TOFU-style
  sentence fine-tune" could be measured. The Gaussian change is the only non-TOFU reference, and its norm is not matched.
- A pair that holds topic and answer style fixed and changes only prior familiarity, for example real facts the model
  gets wrong against ones it gets right.
- Rank-weighted overlap or overlap on the raw scores: E105 does not save the scores, only the ordered sets.
- Reads past warm-up, where row-specific facts are learned.
- Block-matched random draws for these exact sets. I used the expected layer-matched chance instead.
- Any GPU recomputation: I checked the stored sets, not the reads that produced them.
