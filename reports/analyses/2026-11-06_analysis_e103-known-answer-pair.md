# Analysis: does a neuron reading tell a fine-tune on facts Qwen already knows from one on facts it lacks? (E103)

*2026-11-06, baseline-lead's E103 writer. **Provisional**: fast lane, no independent check. Plan
`reports/plans/2026-11-06_plan_e103-known-answer-pair.md`, adopted from Fable's option A
(`reports/plans/2026-11-06_advice_neuron-baseline-next.md`); rules `experiments/known_answer_pair/results/predictions.json`;
code `experiments/known_answer_pair/{run.py,neurons.py}`, `src/commute/qwen_neurons.py`. Explanation:
`notebooks/E103_known_answer_pair.ipynb`. Figures: `experiments/known_answer_pair/plot.py`. Provenance at the end.*

We fine-tuned Qwen3-1.7B on 217 real-world facts it already knows, with the recipe of an earlier fine-tune on fictional
authors, and read both updates with the neuron instruments built last week. At the template fine-tune's size neither
neuron reading told the two updates apart, and the control meant to show the update had something to carry passed
mostly because the model learned to give a bare short answer and stop. So at that size the pair differs in answer format
as well as in what the model already knew, and its level readings cannot yet be blamed on the instruments.

## The answer

At the size of the template fine-tune, every filed neuron reading of the known-facts update sat level with the
fictional-facts update. That was the filed lean. But the control that the update moved its answers passed for the wrong
reason. The training rows ask for one sentence and answer with a bare name, and the unchanged model almost never stops
after a bare name. Learning to stop accounts for nearly all of the gain. So the matched pair differs in a format lesson
as well as in its label. Its level readings say little yet about whether these instruments can tell reuse from new.

## The anomaly, run down first

The runner's numbers checked against the files: every stop passed; the matched step is 33, norm 0.4887 against the
template's 0.4865; the end state's norm is 8.29; the gain at the matched step is 11.35 nats a token against a bar of
0.333. That gain is seventeen times the template's 0.666 and larger than the fictional state's 6.72, on facts the model
already picks four times in five. A number like that blocks every reading built on it, so it was run down before
anything else.

**What the gain measures** (`run.py` `answer_seqs`, `gain`; `qwen_tofu.seq`, `mean_logp`). For each of the 217 real
rows, the sequence is the chat prompt for TOFU's own question plus the suffix " Answer in one sentence.", with thinking
off, followed by TOFU's own answer and the end-of-turn token `<|im_end|>`. TOFU's real answers are one to five words:
"William Shakespeare", "Paris". The per-row reading is the mean log P over the response tokens, end token included: 2 to
10 tokens, 4.4 on average. The gain is the change from the unchanged model W0 in that per-row mean, averaged over rows.
This is pair 0 of the five wordings trained, so the gain is read on exactly the strings the update was trained on.

**What W0 makes of these strings.** W0's mean is −12.13 nats a token, about as unlikely as a uniform guess over Qwen's
151,000-token vocabulary. The per-row sum of log P sits near −45 nats whatever the answer's length: −48 for two-token
answers, −39 for five, −51 for ten. Its correlation with length is 0.15, while the per-token mean tracks one over the
length at −0.84 (scratch check with the tokenizer only, reproduced in `plot.py` `w0_deficit`). A deficit spread over
the answer's words would grow with length. A fixed one sits on a single token per answer. The likely token is the end
of turn: asked for one sentence, Qwen does not stop after a bare name. This was not confirmed position by position,
since no per-position log P was saved and the model was not loaded here.

![W0's summed log P per answer against its length](../../experiments/known_answer_pair/figures/w0_deficit.png)

**How the update removes it.** The gain reaches 10.99 by step 10, at a weight change of 0.076. That is 97% of the
matched gain at a sixth of the matched size, while the rate is still 7e-7 inside warm-up. After the matched update the
mean is −0.78 nats a token, so the update removed 94% of W0's deficit within ten small steps. A one-token format habit
is learned that fast; 217 separate facts would not be. The fictional state at the same size gains 6.72 on these same
real answers without having trained on any of them. E92 read that state as a flatter output, which earns log P exactly
where W0 is confidently wrong. On E102's packed reader, where earlier question-answer pairs sit in the context and show
the model the bare-answer format, the same known-facts state gains 3.17 per answer position. The fictional seeds gain
2.45 and 2.18 there, and E102's Gaussian random change 0.70: that is above the template's 0.666, so a change that
teaches nothing would pass the bar on that reader.

![gain on the real answers over training, with the references](../../experiments/known_answer_pair/figures/gain_curve.png)

**Verdict.** The 11.35 is mostly a format gain. The update teaches the model to answer a "one sentence" question with a
bare phrase and then stop, a lesson carried by pair 0 of the training rows. The plan's filed expectation said as much:
"the format alone should earn most of it". The ratio to the template's 0.666 means nothing either, since that was read
by a different reader, pooled over packed template rows.

**What it does to the gain control.** The control passes as filed, and it does show that the update moves the trained
answers. It does not show a fact-level gain. Learning to stop could carry the whole reading on the real rows: an upper
bound on what is specific to training on these rows is 11.35 − 6.72 = 4.6 nats, and that remainder still includes the
format lesson. On the packed reader the gap between the known-facts state and the fictional seeds' mean is 0.86 per
position, with row standard errors near 0.3 each.

**What it does to "only the label differs".** The plan held that up to the matched step, inside warm-up, the pair
differs in rows and label only. The format differs too. The fictional rows' pair 0 answers "Answer in one sentence."
with a full sentence; the real rows' pair 0 answers it with a bare phrase. So the known-facts update carries a lesson,
"stop after a bare phrase", that the fictional update does not. At step 29 to 33 neither update has had time to do much
else: E91 found the fictional state had learned none of its facts there. The matched comparison is mostly one format
change against another flatter-output change, with the label nominal. Every reading on the matched pair below inherits
this, and so do qwen-decomp-lead's VPD reads of the same states.

## What we were trying to test

`GOALS.md` asks whether post-training strengthens what the model already does or teaches it something new, and whether
a method can tell which. E102 read Qwen's MLP neurons on two earlier fine-tunes: the template fine-tune, labelled reuse,
and the TOFU fictional-author fine-tune, labelled new. They differed in task, rows, recipe and size as well as label, so
no reading could be put down to the label. This experiment builds a pair meant to differ in the label alone.

**The two fine-tunes.** Both train Qwen3-1.7B on TOFU rows, each row in five wordings: TOFU's own question and answer,
plus four rewordings written by Haiku clerks. Both use E84's recipe: peak rate 2e-5 with a 250-step linear warm-up,
batch 16, gradient clip 1, fp32, seed 42.

- *Known-answer fine-tune, new here.* TOFU's 217 real-person and world-fact rows, 1,085 sequences, five epochs of 335
  steps. W0 already picks 0.79 of these facts in multiple choice.
- *Fictional-author fine-tune, E84 and E91, read only.* TOFU's 787 kept fictional rows, 3,935 sequences, 1,250 steps.
  W0 cannot know these facts. Two seeds, 42 and 142.

**States.** Each fine-tune is read at two points. The *matched* state is the step whose total weight change $|\delta W|$,
the Frobenius norm over all linear layers, is nearest the template fine-tune's 0.4865: step 33 at 0.4887 here, step 29
at 0.477 for the fictional seed 42. Up to these steps both runs are inside warm-up at the same rate, so they differ in
rows, label and, as the anomaly showed, answer format. The *end* state is the last step: 8.29 here at step 335, 18.1 for
the fictional run at step 1250. The end pair differs in size and steps too, so it is reported, not scored.

**Readings.**

- *Gain*, defined above: change in mean log P of the trained real answers, nats a token. 0 is no change.
- *Real-fact share*: W0's four-way multiple choice on the 217 real rows under " Answer in a few words.", E81's reader. A
  row is right when the true answer's mean token log P beats all three wrong ones. Chance is 0.25; W0 reads 0.793.
- *Carrying set*: the 1,024 MLP neurons, of Qwen's 172,032, with the largest first-order share of a state's gain on a
  row set: W0's gradient of the answer log P times the change in each neuron's firing, E98's ranking. Read on the real
  rows, pair 0 packed into rows of 256 tokens as E92 packed the TOFU rows, and on E92's 128 TOFU fictional rows.
- *Prompt overlap*: the share of a carrying set inside the 1,024 neurons whose firing at template answers moves most when
  a system turn is added, E102's contrast set. 0 is none; random layer-matched sets read 0.002 to 0.014 here.
- *Engaged share*: the median, over a set, of W0's firing at the answer positions as a quantile of that neuron's firing
  on FineWeb-Edu text. 0.5 is a typical web position; E102 read 0.50 for every set.
- *References*: E102's Gaussian random change, scaled to the template's KL and 180 times its norm, as the floor; 100
  block-matched random sets per state as draws.

**The pair rule** (`neurons.pair_side`). For each size, row set and reading, the known-facts state is compared with the
mean of the two fictional seeds. The band is the larger of the two seeds' spread and twice the standard deviation of
the known state's random draws. "Known higher" needs the gap past the band and the known value above the floor, the
larger of the Gaussian change's value and every draw. "New higher" mirrors it. Anything else is level.

## What we expected, and why

- *W0 would reproduce E81's real-fact share of 0.793 within 0.01*, or the rows or reader had changed.
- *The gain at the matched step would clear half the template's gain, 0.333, and land above 1.* The filed reasoning
  already expected the bare-answer format to earn most of it.
- *Both known-facts states and the matched fictional state would keep the real-fact share within 0.05 of W0's*, and the
  fictional end state would lose it, as E81's one-wording fictional fine-tune fell to 0.42.
- *Re-reading the matched fictional state on the TOFU rows would reproduce E102's gain of 2.05 and its carrying set.*
- *Engaged share and prompt overlap would read level at matched size on both row sets.* E102 had found no spread in the
  engaged share and only one band between opposite labels in prompt overlap. A label-driven difference here would be
  the instruments' first positive control.

## What we observed

### Did the apparatus reproduce its references?

Yes, exactly. W0's real-fact share read 0.79263, E81's value to every digit. The matched fictional state re-read on the
TOFU rows gained 2.05250, E102's value to every digit, with the same 1,024-neuron carrying set. Every early training
check passed: step 1's loss equalled stage 0's first batch, step 2's weight change was 0.00235, in E84's band, and step
250's loss was 0.026 of the first. Memory peaked at 39.6 GB against a filed 44 GiB. The matched step chosen during
training, 33, was also the nearest over the whole record.

### Did the updates keep, raise or lose the facts W0 already knows?

![real-fact share](../../experiments/known_answer_pair/figures/recall.png)

**Reading it.** As filed. The known-facts states raised the real-fact share, from 0.79 to 0.88 at matched size and 0.995
at the end. The matched fictional state kept it at 0.82, and the fictional end state lost it, falling to 0.44, E81's
reading again. The end-state 0.995 is memory of the trained rows, since the share is read on those same rows. The
matched rise of 0.09 is above W0's interval, but its cause is open. It may be the facts sharpened, or the removal of a
length penalty: W0's costly end token weighs more on short options, and the update made that token cheap.

| state | real-fact share | 95% interval | gain, nats a token | weight change |
|---|---|---|---|---|
| W0 | 0.793 | 0.737 to 0.843 | 0 | 0 |
| known-facts, matched, step 33 | 0.880 | 0.834 to 0.922 | 11.35 | 0.489 |
| known-facts, end, step 335 | 0.995 | 0.986 to 1.000 | 12.09 | 8.29 |
| fictional seed 42, matched, step 29 | 0.820 | 0.770 to 0.871 | 6.72 | 0.477 |
| fictional seed 42, end, step 1250 | 0.442 | 0.373 to 0.507 | 1.86 | 18.1 |

### At matched size, does any neuron reading tell the known-facts update from the fictional one?

![each pair rule's gap in units of its band](../../experiments/known_answer_pair/figures/pair_rules.png)

**Reading it.** No. All four scored readings sit inside their bands, on both row sets, as filed. Engaged share reads
0.49 to 0.50 for every matched state on both row sets, the same flat 0.50 that E102 read for every set. Prompt overlap
puts the known-facts state 0.003 above the fictional seeds on the real rows and 0.002 above on the TOFU rows, inside
bands of 0.004. The caveat is the anomaly's: at this size the two updates differ mostly in answer format, so a level
reading here tests format against flatter output more than known against new.

| size | rows | reading | known | fictional seeds 42, 142 | gap | band | floor | verdict |
|---|---|---|---|---|---|---|---|---|
| matched | real | engaged share | 0.490 | 0.487, 0.487 | 0.003 | 0.010 | 0.512 | level |
| matched | TOFU | engaged share | 0.499 | 0.501, 0.499 | −0.001 | 0.005 | 0.507 | level |
| matched | real | prompt overlap | 0.030 | 0.027, 0.027 | 0.003 | 0.004 | 0.021 | level |
| matched | TOFU | prompt overlap | 0.054 | 0.051, 0.052 | 0.002 | 0.004 | 0.026 | level |
| end | real | engaged share | 0.499 | 0.490, 0.489 | 0.010 | 0.009 | 0.508 | level, under the floor |
| end | TOFU | engaged share | 0.503 | 0.501, 0.504 | −0.000 | 0.005 | 0.505 | level |
| end | real | prompt overlap | 0.026 | 0.010, 0.009 | 0.017 | 0.004 | 0.021 | known higher |
| end | TOFU | prompt overlap | 0.045 | 0.038, 0.040 | 0.006 | 0.004 | 0.026 | known higher |

Gains on the neuron reader, per answer position on packed rows: known-facts matched 3.17 on the real rows and 1.74 on
TOFU; end 3.51 and 1.42; fictional matched 2.45 and 2.18 on the real rows; fictional end 0.67 and 0.89; the Gaussian
change 0.70. Unscored and printed: the known-facts matched set shares 0.40 of its neurons with the fictional seed-42
matched set on the real rows and 0.57 on the TOFU rows, where E102's two fictional seeds shared 0.94 to 0.95. The two
updates do use different neurons, more than seed noise; what drives that, label, format or rows, is not read here.

### At the end of training, why does prompt overlap separate them?

![prompt overlap per state on each row set](../../experiments/known_answer_pair/figures/prompt_overlap.png)

**Reading it.** Mostly because the fictional end states fall, not because the known-facts state rises. On the real rows
the fictional end sets drop to 0.010 and 0.009, under the Gaussian change's 0.021 and near the random draws. The
known-facts end state stays at 0.026, just over the floor. On the TOFU rows every state sits above the floor, and the
known-facts end state leads by 0.006 against a band of 0.004. This pair also differs in size, 8.29 against 18.1, and in
steps, 335 against 1,250. So the "known higher" verdicts are reported, not scored, and cannot be put down to the label.

## What we learned

- The pair meant to fix the label also differs in format at matched size: the known-facts training answers a request
  for a sentence with a bare phrase. Most of what the matched update did is learn to stop after that phrase.
- With that confound, no filed neuron reading separates the two updates at matched size. That is consistent with
  Fable's "the methods are the problem", but this pair cannot show it, because the label was barely expressed yet.
- The carrying sets of the two matched updates overlap far less than two seeds of one update do. A neuron-level
  difference exists; the filed readings, prompt overlap and engaged share, do not see it.
- The gain control as written is passed by a format lesson. A gain control on short answers needs the end token, or the
  wording the model is told to use, handled before its bar means anything.

## How compelling is the evidence

- *One seed* of the known-facts run; the fictional run's two seeds set the bands.
- *The anomaly's cause* rests on the per-row sums being flat in length and on the speed of the gain, not on a
  per-position read. A per-position read on the saved states would settle it.
- *Engaged share* has now read 0.49 to 0.50 for every set in two experiments. It has never moved, so its power is still
  untested; it may be blind at this grain.
- *Prompt overlap* is against a template-task contrast with no tie to facts. Its end-size separation comes with a two-fold
  size difference and different step counts.
- *The Gaussian floor* is 180 times the template's norm, matched on KL, not on norm.
- *Rows*: 217 real rows and 128 TOFU rows; the real rows pack into only 29 rows of 256 tokens, 941 answer positions.

## Surprises

- **The gain control passed seventeen-fold, on a format habit.** Predicted above 1 with format expected to dominate; the
  size was the surprise, and it hides whether the update carries any fact-level change at all.
- **The two matched carrying sets share only 0.40 to 0.57 of their neurons**, against 0.94 to 0.95 between seeds of one
  update. Not filed. It is the only reading here with spread, and the filed rules do not read it.
- **The fictional end states' carrying sets on the real rows fall below the Gaussian change in prompt overlap.**
  Reported only.

## What next

1. *Read the gain without the end token, $0, minutes on `debug`.* Re-read the saved matched states' log P per position
   on the real answers, end token excluded. If the answer words' gain stays above 0.333, the matched pair carries
   content beyond format and the level readings count against the instruments. If it is near zero, the matched pair
   is a format pair and needs rebuilding.
2. *A format-matched known-facts update, a new arm through `/design`.* Train on the four sentence rewordings only, or
   ask pair 0 with " Answer in a few words.", so both fine-tunes learn the same answer form. If the level readings and
   the low set overlap both persist, the set overlap is a label reading worth scoring.
3. *Hand this caveat to qwen-decomp-lead with the states.* E94's direct and outside share on these states read the
   same format confound; a VPD separation on the matched pair would need the same check before it counts.

## Scorecard

Filed in `experiments/known_answer_pair/results/predictions.json`, scored as filed.

| id | claim | rule | read | verdict |
|---|---|---|---|---|
| control 0 | W0 reproduces E81's real-fact share | within 0.01 of 0.7926 | 0.7926 | holds |
| first minutes | step 1 loss equal, step 2 weight change in E84's band, peak under 44 GiB, step 250 loss under 0.95 of first | bands | 0, 0.00235, 28.2 GiB at step 2 and 36.8 GiB later, 0.026 | all hold |
| C1 | gain at the matched step at least half the template's | ≥ 0.333; expected above 1 | 11.35 at step 33, norm 0.4887 | holds; mostly a format gain, see the anomaly |
| R1 real matched | real-fact share kept | ≥ 0.743 | 0.880 | kept, as leaned |
| R1 real full | real-fact share kept | ≥ 0.743 | 0.995 | kept, as leaned |
| R1 fictional matched | real-fact share kept | ≥ 0.743 | 0.820 | kept, as leaned |
| R1 fictional full | real-fact share dropped | < 0.743 | 0.442 | dropped, as leaned |
| C2 | fictional matched re-read reproduces E102 | gain within 0.001 relative, set overlap ≥ 0.95 | rel 0, overlap 1.0 | holds |
| matched real engaged | pair rule | level lean | gap 0.003, band 0.010 | level, as leaned |
| matched TOFU engaged | pair rule | level lean | gap −0.001, band 0.005 | level, as leaned |
| matched real prompt overlap | pair rule | level lean | gap 0.003, band 0.004 | level, as leaned |
| matched TOFU prompt overlap | pair rule | level lean | gap 0.002, band 0.004 | level, as leaned |
| end, four rules | reported, not scored | pair rule | see the table above | engaged level twice; prompt overlap known higher twice |

The filed "reading" field says all four scored rules level means these instruments do not separate reuse from new on
Qwen even with the label fixed. Read as filed, that is the outcome. The anomaly qualifies it: the label was not the only
difference left at matched size.

## Departures from the literature

- **TOFU's real-author and world-fact sets are evaluation sets** in Maini et al. 2024, never trained on. Here they are
  the training rows, so the known-facts update has a label fixed by the data.
- **TOFU's evaluation** of these sets uses the probability and truth ratio of paraphrased answers. Here it is four-way
  multiple choice by mean token log P, E81's reader, as in E81.
- **E84's recipe** is followed except where the rows force it: 217 rows and 335 steps against 787 rows and 1,250, so
  the cosine decay starts at step 250 of 335. The answer forms are one sentence stating a one-to-five-word answer,
  where E84's restate full sentences, and pair 0 is TOFU's bare answer.
- **E84's wording checks**, two departures: no held paraphrase exists for these rows, so its two checks against it read
  nothing; an answer form may be up to 20 words and must contain every answer word.
- **The gain control** averages per-answer means over rows, where E92 pooled tokens over packed rows; the bar is half
  E92's pooled template gain, so the two are not on one reader.
- **E102's neuron readings** carry E102's own departures from the essay's protocol, listed in its report.

## Not run

- **E94's VPD reads of both new states**, direct and outside share on each state's own rows and the other's: qwen-decomp-lead's
  run, pending (note `~/commute-data/agents/qwen-decomp-lead/note-e103-states.md`).
- A per-position read of the gain, end token excluded, on the saved states.
- A second seed of the known-facts fine-tune.
- A format-matched known-facts fine-tune.
- E102 stage B's judge on these sets: it runs only on a separating set, and none was scored as separating.
- A size-matched end pair: the known-facts end is 8.29 against the fictional 18.1.
- Prompt overlap against a contrast tied to facts rather than the template task.

## Provenance

- Job 53188, `cluster/known_answer_pair.sbatch run`, one L40, `debug`, 406 s: stages 0, T and G; training 313 s.
  Results `experiments/known_answer_pair/results/{stage_0,train,read}.json`, `run.log`.
- Job 53209, `... neurons`, one L40, `debug`, 258 s. Results `results/neurons.json`, `results/neurons/*.json`.
- States `results/large/dW/{matched,end}.safetensors`, fp32, 5.64 GB each, gitignored; fictional states from
  `qwen_size_grid/results/large/dW/a2_s42/step_29.safetensors` and `qwen_tofu_forms/results/large/dW/a2_2e-5/step_1250.safetensors`.
- Wordings `results/forms.jsonl`: 217 of 217 rows kept, no failing form.
- Cost $0. Answer lengths for the anomaly from the Qwen tokenizer only, no model, in `plot.py`.
