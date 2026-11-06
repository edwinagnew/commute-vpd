# Advice: the carrying set where the facts are learned, with a read on the answer tokens alone ("E108")

*2026-11-06, Fable's advice for baseline-lead, for ask 23 option (a). Read-only; nothing run; the only computation is
`jq` over stored sets (section 2). "E108" is a placeholder: the E-code is given when the plan is written (`E108` is
unused in `experiments/TRIED.md` as of this commit). Follows E107 (`reports/analyses/2026-11-06_analysis_e107-label-vs-style.md`),
its check (`reports/reviews/2026-11-06_review_e107-independent-check.md`), E105 and my E106 advice
(`reports/plans/2026-11-06_advice_e106-label-vs-style.md`). Terms as `GOALS.md`; "carrying set", "matched step",
"overlap" and "probe" as E107's report defines them.*

## 1. Where E107 left the question

At the matched step (|δW| 0.49, step 31-33, under half an epoch) the wrong-answer arm's carrying set sits with the
known arm's (0.84-0.88 on the TOFU probe) and as far from the fictional arm's as the known arm's does. But at that step
the wrong arm had learned little: 0.27 of rows prefer the wrong sentence against 0.14 at W0, and the answer span is 2.6
of a sentence's 15.1 content tokens, carrying 41% of W0's loss on a wrong sentence and 16% on a true one (check,
section 2), so most of every arm's update and most of every read is the shared sentence frame. Option (a) asks the
same question where the facts are learned, with the read cut to the answer tokens.

## 2. What already exists at larger sizes ($0 to read)

| state | file | step | norm | facts learned? | carrying sets read? |
|---|---|---|---|---|---|
| known, sentence form, seed 42, end | `known_answer_pair/results/large/dW/amend1_end.safetensors` | 335 | 8.369 (`amend1_train.json` `.steps[-1].dw_norm`) | yes: greedy sentence recall 0.996 on its own questions (E103 report line 478) | no |
| known, seed 142, end | `.../amend2_s142_end.safetensors` | 335 | 8.248 (`amend2_train.json`) | same recipe; not read | no |
| known, bare-answer form (E103's first run), end | `.../end.safetensors` | 335 | 8.288 (`train.json`) | yes | yes: `neurons/real_full__{real,tofu}.json` |
| fictional (anchor 2), seed 42, first state past the learning line | `qwen_size_grid/results/large/dW/a2_s42/step_550.safetensors` | 550 | 15.51, trained-wording recall 0.615 (`size_grid.json` `.arms.anchor2_s0.past_line`) | yes | no |
| fictional, seed 142, same | `.../a2_s142/step_500.safetensors` | 500 | 14.68, recall 0.591 (`.arms.anchor2_s142.past_line`) | yes | no |
| fictional, both seeds, full | `qwen_tofu_forms/.../a2_2e-5/step_1250.safetensors`, `qwen_anchor_checks/.../tofu_seed142/step_1250.safetensors` | 1250 | 18.11 / 18.12 | yes (0.99) | yes: `neurons/fict_s{42,142}_full__real.json`, `qwen_carrying/results/states/a2_s{42,142}_full.json` |
| wrong, either seed, beyond step 32 | none (E107's CAP was 64; only `WRONG{42,142}_matched` saved) | — | — | — | — |
| fictional between step 30 and 500 | none | — | — | at 27-30 recall 0.16-0.17, "not learned" (E91 lines 95-98) | matched only |

Every file is fp32, 5.64 GB, 196 matrices. All six end-of-grid sets above are 1,024 neurons.

**A first look at full size, from the stored sets** (`jq`, the share of the first set inside the second). Real probe:
bare-form known end in fictional full 0.399 and 0.387; fictional full seeds 0.714. TOFU probe: 0.423 and 0.421; fictional
seeds 0.827. A state's own set moves a lot with size: known end in known matched 0.489 (real) / 0.569 (TOFU), fictional
full in fictional matched 0.413 / 0.427. Gains the sets are ranked on diverge at full size: real probe known 3.51 against
fictional 0.67 / 0.89 (the fictional end lost real facts, E103); TOFU probe known 1.42 against fictional 3.77. Three things
follow. The set is not a fixed property of an arm across size, so a size-matched read is not optional. At the end size
the seed null is wider (band 0.29 on the real probe, 0.17 on TOFU) than at the matched step (0.10, 0.06). And on each probe
one arm is ranked on a gain three to five times the other's, so a set on the small gain is noisier; the real probe at
full size ranks the fictional arm on almost nothing.

## 3. Question 1: which size

**Recommendation: the five-epoch end of the 217-row recipe (step 335, |δW| 8.25-8.37) for the known and wrong arms, with
the fictional arm at its first learned state (step 500 / 550, |δW| 14.7-15.5) as the cross-label reference, size
stated.** Reasons:

- The wrong against known comparison, the one E107 could not resolve, is matched by construction: same rows, prompts,
  schedule and step count, so the norm lands within a few percent (the known seeds differ by 1.5% already). Both arms
  have learned their facts there (the known arm 0.996 greedy recall; the wrong arm is checked, section 4). Two known
  end states exist; the wrong end needs two trainings of 5.7 min each (`amend2.log` lines 9 to 67: 62.6 s to 402.9 s
  for 335 steps with candidate reads and the fp32 save).
- No learned fictional state can be size-matched to 8.3 under E84's recipe: at step 500 the fictional arm is just past
  the line (0.591 against 0.585), and norm 8.3 falls near step 300, where it has not learned. The known arm holds its
  sentences at half the norm the fictional arm needs: a fact to state, not to match away. Training the known and wrong
  arms on to 15.5 would need a new recipe (about ten epochs or twice the rate), unsourced; not recommended.
- The fictional full states (18.1) stay as the second reference, since their sets exist; the step 500 / 550 states are
  nearer in size and are "learned" by E91's own line, so they are the primary reference and need four reads.
- A fresh run matched to the fictional end-state norm is not needed. A fictional retrain to norm 8.3 (step about 300,
  about 4 min a seed, not learned) is a pure size control and is listed under "what would change it", not in the
  recommendation.

Arms: known end ×2 (exist), wrong end ×2 (train), fictional learned ×2 (exist); the six matched states (exist) are
re-read on the new probe only, so the answer-token read has both sizes. Nothing else.

## 4. Question 2: the answer-token read

**Definition.** E98's score as `commute.known_facts.carrying` computes it (W0's gradient of the summed log P at the masked
positions, times the change in each neuron's firing), with the mask cut to the positions that predict a token of the
bare answer inside the sentence. Sequences are the sentence reader's (`known_facts.sentence_seqs`: question +
" Answer in one sentence." + sentence), packed into 256-token rows as `known_facts.real_rows` packs. The span comes
from the fast tokenizer's offsets on the sentence tokenized alone, which is how `qwen_tofu.seq` tokenizes the answer;
assert every span has at least one token and covers the answer's characters exactly (217 of 217 sentences hold the
answer verbatim, E106 advice). `Rows` takes an explicit `answer` mask, so this is one new builder of about 20 lines in
`known_facts.py`; `rank_batch` and `top_set` are unchanged. Two probes: **T**, the true sentences with the true span
(about 560 positions against about 3,280 content positions), and **Wr**, the wrong sentences with the wrong span; W0's
per-token files for both readers exist (`label_vs_style/results/large/w0_{true,wrong}.json`).

**The sign problem, and the ranking to use.** The score is signed and `top_set` keeps positive scores. On T the wrong
end state has pushed the true answers down for five epochs, so its first-order total there is negative and its
"positive top 1,024" is the minority of neurons that push the true answer up: comparing that with the known arm's set
would read "separated" for a trivial reason. So on the fixed probe T rank by **|score|**, the neurons through which the
update acts on the answer tokens whichever way, and report the signed set beside it. The departure from E98's signed
ranking is checked in stage 1: for the known end state on T, the magnitude set inside the signed set, expected at least
0.8 since its scores are mostly positive; for the wrong end state the two are expected to differ, which is the point.
Reported, not scored: each arm on its own span (known on T, wrong on Wr, signed), the own-content read, with the caveat
that T and Wr differ in target tokens so W0's gradient differs and cross-probe overlaps carry that confound.

**Controls.** (i) The do-nothing state has an empty set on every probe, unread. (ii) The probe control: one δW (known
end seed 42) read on T and on Wr by magnitude; its overlap says how far the target tokens alone move a set and is the
floor any own-content comparison must clear. (iii) The frame read: the complement positions (content tokens outside
the span) on T, and the share of the whole-sentence first-order gain that sits on the span, per arm; the whole-row read
must equal span plus frame to rounding (an exact arithmetic check of the mask). (iv) E102's Gaussian change
(`qwen_carrying/results/large/gauss_unit.safetensors`) on T, the random floor, one read.

**Not a new probe for the fictional arm.** The TOFU rows have no answer span (invented biographies are content almost
everywhere), so the fictional arm is read on T like every other state; its gain there is the unrelated-update
reference, and on the TOFU and real probes the existing whole-row reads are kept.

## 5. Question 3: the review's cheap control as stage 0

Read `amend1_matched` and `amend2_s142_matched` (the known arm's matched states) on the wrong sentences with the
sentence reader: per-row content log P, the share of rows preferring the wrong sentence, the mean true-minus-wrong gap.
W0 reads 0.143 and 0.651 nats a token; E107's wrong arm at its matched step 0.267 / 0.272 and 0.12-0.13. About 3 s a
state after the 10 s load (E107 `run.log` lines 3-5: a 5.64 GB state loaded and read in 10 s, a 217-row sentence read in
2.1-2.5 s). Lean: the known matched state moves the gap little, 0.4-0.9 nats and 0.10-0.20 preferring wrong, because at
this step the update is mostly the sentence skill, which raises both sentences alike; E103's end state on bare options
narrowed the answer-word gap from 3.6 to 2.9 nats rather than widening it (E103 report, Amendment 2 table). If the
known state also shrinks the gap toward 0.12, E107's wrong arm learned nothing label-specific at matched size, which
strengthens the case for this plan and changes nothing in it. Repeat the read on the two known end states (expected:
preferring wrong near 0, gap over 2 nats) and on the wrong end states (the "learned" check below).

## 6. Question 4: filed rules and bands, mirroring E107

Every reference is measured in the job's first stage, on existing states, before a wrong arm trains (the rule since
2026-11-03). Per scored probe:

- **w** = the known end seeds' overlap; band = 1 − w. Expected 0.65-0.85 from section 2's fictional seeds at full size
  (0.71 / 0.83). No L: E105's halves were never trained to the end, so the "same label, different rows" level does not
  exist at this size and the rule leans on seeds and x alone, which is a stated departure from E107's rule.
- **x** = the highest overlap of a known end set with a fictional learned set (four pairs).
- **Power gate**, stage 1, before training: w − x > band on the probe, else that probe's rule is "no power at this
  size" and is reported, not scored; the wrong arms still train, since they cost 12 minutes and qwen-decomp-lead wants
  the states.
- **cross** = the four wrong-end-in-known-end overlaps. **Familiarity** iff max(cross) ≤ x + band; **topic or style**
  iff min(cross) ≥ w − band and min(cross) − x > band; else no call. Edge cases as E107's `wrong_verdict` (identical sets,
  an empty set, a set under 1,024, one neuron each side of each line), run on CPU before filing with the stored
  matched sets standing in.
- Scored probes: the TOFU whole-row read (E107's scored probe, existing code) and **T by magnitude** (the new read).
  Reported: the real whole-row probe, Wr, the signed T sets, the six matched states on T, the probe control, the frame
  share, wrong-in-fictional (expected level with known-in-fictional under topic or style, at or above it under
  familiarity).
- **"Learned" gate for the wrong arm**, at its end: at least 0.75 of rows prefer the wrong sentence (the label check's
  0.75, mirrored) and E81's greedy reader (`qwen_tofu.recall`, 13 s for two suffixes, `amend2.log` lines 69-70) returns
  the wrong answer's words in at least 0.585 of rows (E91's line). Expected 0.90-1.00 and over 0.9. A miss means the
  premise failed: report, no amendment.
- Early checks as E107's `first_minutes` (first batches differ, step 1 and 2, 44 GiB, warm-up loss fraction at step
  250 now reached: `warmup_loss_frac` 0.95); the end norm within 5% of the known seeds' 8.25-8.37, else report.
- Leans, for the writer to confirm after the dry run: on the TOFU whole-row read **topic or style** again (the frame
  still dominates whole-row gains); on T **no call the likelier, familiarity the lean**: the answer tokens are where the
  two arms pushed in opposite directions for five epochs, and the magnitude ranking sees both directions.

## 7. Question 5: what each outcome means

- **Whole-row: topic or style; T: topic or style.** Even with the facts learned and the frame cut away, the update acts
  on the answer through the same neurons whether it confirms or contradicts the model. Set identity is closed as a
  label reading on Qwen at both sizes; the neuron baseline reads "no known-against-new separation", labelled with the
  granularity limit. Beside VPD: E94 Amendment 1 (qwen-decomp-lead, coded, not run) reads the same four known states
  (`qwen_d3_rows/results/predictions_amend1.json` `.states`), so its three readings on the known end states and on the
  wrong end states handed over give a same-card comparison; if VPD separates wrong from known where the sets do not,
  VPD has a reading the neuron baseline lacks; if neither does, the label is invisible at set grain to both.
- **Whole-row: topic or style; T: familiarity.** The frame hid the label. The answer-token read becomes the baseline's
  reading, size-dependent (the matched-size T reads in the same job say whether it holds early too), and the next
  question is contradicted against new (option (b)).
- **Both familiarity.** The set reads learned content once there is some; E103-E107 read too early. Carried forward
  with the size stated; then option (b).
- **No power on a probe** (w − x within the band, as section 2's 0.71 seeds against 0.39-0.40 cross on the real probe
  warns is possible on noisy gains). Sets of 1,024 on a small gain are noise at this size; report with the gains; do not
  retune K inside this E-code.
- What would change the recommendation: the user wanting a pure size control for the fictional arm (add the step-300
  retrain, not learned, about 4 min a seed); E94 Amendment 1 having run by then (then read the same states it read,
  nothing more); stage 0 or the power gate reading as above (both are handled inside the plan).

## 8. Question 6: cost and time, cluster `debug`, one L40, $0

Timings: a state loaded and read on the real probe in 10 s (E107 `run.log` lines 3-4); real whole-row read 4.6 s, TOFU
read 20.4 s, fp32 save 17 s (lines 51-54); a 217-row sentence read 2.1-2.5 s (lines 4-5); a span probe of 217 sentences
is a sentence read with hooks, under 10 s; greedy 13 s (`amend2.log` lines 69-70); a full 335-step arm with candidate
reads and the end save 340 s (`amend2.log` lines 9, 67). Model load about 40 s.

- **Job A, reads only (stages 0-1):** known end ×2 at about 90 s each (load, real, TOFU, T, Wr, frame, wrong sentences,
  greedy); known matched ×2 at 40 s (wrong sentences, T, Wr); fictional learned ×2 at 60 s (real, TOFU, T); fictional
  matched ×2, wrong matched ×2 at 25 s (T); Gaussian on T 25 s. About 9 min with load. It fixes w, x and the gate.
- **Job B, the wrong arms to the end:** per seed 340 s training and save, then the reads of job A's known-end row,
  about 90 s: 7.5 min a seed, 15 min for both; or one job a seed if the queue is busy. Then the score on CPU.
- Memory as E107: training 38.5 GiB, reads 41.3 GB, host 11 GB, `--mem 24G`. Disk: two new fp32 states, 11.3 GB, kept
  for qwen-decomp-lead. Total card time about 25 min, two or three submissions, each under 30 min; under the gate,
  fast lane.

## 9. Options, in one place

**(a) as above, recommended:** end-size pair plus the answer-token read, references measured first, about 25 min of
card, no new data, no clerks. **(a′)** the same without the answer-token read: $0 of new code beyond an end-state loop,
but it leaves the frame in every read, which is the objection E107's check raised. **(a″)** a size-matched fictional
retrain at norm 8.3 added: answers size, not "learned". **(b)** below.

**Option (b), genuinely new facts in the known style, in two lines.** Either real facts W0 lacks: a long-tail question
file (PopQA-style, downloaded through Slurm), screened by E81's greedy reader into about 110 W0 gets right and 110 it
gets wrong, written into sentences by Haiku clerks under `CLERK_BRIEF_AMEND1.md` with the judge, two arms; about 2-3 h
of agent time, $0, one 9-min job. Or the $0 substitution: TOFU's invented authors' names in place of the answers on the
100 author rows (E106's option 6), a new entity in the known arm's questions and style, trained and read exactly as the
wrong arm here. Either then gets the same reads at both sizes as (a), and (a)'s end states are its references.

## 10. Hand-off

A fresh session writes the plan from this file, E107's plan and `src/commute/known_facts.py`: a `Run` with `keep_end`
kept (`fine_tune` already saves the end δW; `to_matched` drops it), the span-row builder, a `score.py` whose rule takes
w and x from the job's own stage 1, predictions filed from sections 5-6 after the CPU dry run on the stored matched
sets, one sbatch copied from `cluster/label_vs_style.sbatch` with section 8's timings in its header. Write the note to
qwen-decomp-lead when the wrong end states exist.
