# Plan: E109, the carrying neurons where the facts are learned

*2026-11-06, baseline-lead's writer. Built from Fable's advice `reports/plans/2026-11-06_advice_e108-larger-size.md`
("E108" there is E109), its points adopted as written. Follows E107 (`reports/analyses/2026-11-06_analysis_e107-label-vs-style.md`)
and its check (`reports/reviews/2026-11-06_review_e107-independent-check.md`). Code `experiments/larger_size/`, rules
`experiments/larger_size/results/predictions.json`, jobs `cluster/larger_size.sbatch`. Separate from E107 Amendment 1
(VPD's switched-on set, job 53822): no file of it or of `experiments/qwen_d3_rows/` is touched.*

## The request, in the Terms' words

Ask 23, option (a) (taken by default, `decisions.md` last row): read the carrying-neuron set of a full-weight fine-tune
of Qwen3-1.7B (no decomposition is read; this is the non-VPD baseline) at the five-epoch end of the known-facts recipe,
for the known arm (true answers W0 already gives) and E107's wrong-answer arm (the same rows, wordings and prompts with
TOFU's wrong answer stated), with the fictional-author arm at its first learned state as the cross-label reference; and
add a read on the answer tokens alone. The question: once both arms have learned their sentences, does the set separate
an update that confirms W0 from one that contradicts it?

## Design check (`/design`; approved class 4, the user's ask 23 option (a))

- **Goal.** `GOALS.md`, "If no simple Φ works": does post-training teach new things or strengthen existing ones, and can a
  reading tell them apart? The neuron baseline is the non-VPD comparison for that reading; E94 Amendment 1
  (qwen-decomp-lead) reads the same known states with VPD.
- **Nearest tried.** E107 (`TRIED.md`): at the matched step (|δW| 0.49) the wrong arm's set sat with the known arm's,
  "topic or style" on both probes. It could not say whether that was because the wrong arm had not learned its answers
  (0.27 of rows prefer the wrong sentence) or because the shared sentence frame dominates every read. E109 removes both:
  learned states, and a read on the answer tokens.
- **Patch or step.** A step: the same comparison at the size where the label exists, and a read where it lives.
- **Decisions.** Relies on E91's sizes, E98's ranking, E103/E107's recipe and probes. Departures, both stated in the
  predictions: no L (E105's halves were never trained to the end), and on T the ranking is by |score| (the sign problem,
  below).
- **What would change what we do next.** Outcomes in "What each outcome means".
- **Control first.** Stage 0, alone at the start of job A: C2n, W0's levels, then the four known states on the true and
  the wrong sentences, with two stored gains reproduced and a stop if a known end state prefers wrong more than W0 does.
  It fails if a state file, the sentence reader or the substitution differs from what E103/E107 read.

## The apparatus

**States** (all fp32 δW, 5.64 GB, read on one L40):

| state | file | step, \|δW\| |
|---|---|---|
| KNOWN_end42, KNOWN_end142 | `known_answer_pair/results/large/dW/amend1_end`, `amend2_s142_end` (exist) | 335; 8.369, 8.248 |
| WRONG_end42, WRONG_end142 | trained here, `larger_size/results/large/dW/WRONG_end<s>` | 335; about 8.3 |
| FICT_learned42, FICT_learned142 | E91's `past_line`: `qwen_size_grid/results/large/dW/a2_s42/step_550`, `a2_s142/step_500` (exist) | 550 / 500; 15.51, 14.68 |
| six matched states (reported) | E103 `amend1_matched`, `amend2_s142_matched`; E91 `at_norm` steps 29, 30; E107 `WRONG{42,142}_matched` | 27-33; 0.48-0.50 |
| GAUSS | E102's Gaussian change | random floor |

The wrong arm is E107's (`known_facts.train_seqs(..., wrong=True)`, 1,085 sequences) run to step 335 through
`known_facts.to_matched(..., keep="end")`, which keeps the last step's δW and drops the matched one; nothing else changes.
No learned fictional state exists at |δW| 8.3 (E84's arm is past E91's learning line only at step 500 / 550), so the
reference's size is stated, not matched.

**Probes.** `tofu`: E92's 128 TOFU rows, signed ranking, E107's scored probe. **T** (new, scored): the 217 true
sentences on E103's sentence reader, packed in rows of 256, reading only the positions that predict a token of the true
bare answer inside the sentence (`known_facts.span_rows`, spans from the fast tokenizer's offsets, each asserted to hold
a token and cover the answer exactly). Reported: `real` (E103's rows), **Wr** (wrong sentences, wrong span), **T_frame**
(the other content tokens), T_content (span and frame together, one state, the arithmetic check). Dry run: all 434
spans exact; T is 39 rows of 256 with 643 read positions against 3,277 content positions (span and frame partition
them exactly), Wr 38 rows with 609.

**The sign problem.** The score is signed and E98 keeps the positive top 1,024. On T the wrong end state has pushed the
true answers down for five epochs, so its positive set would be the minority that pushes them up; comparing that with
the known arm's set reads "separated" for a trivial reason. So T is ranked by |score| ("carry_abs"), and the signed set
is reported beside.

**Numbers, with what each should read** (do-nothing = W0 itself; its score is 0 everywhere, so every set is empty and
every overlap NaN, every rule "unread").

| number | W0 / do-nothing | control or reference | expected |
|---|---|---|---|
| stage 0: rows preferring the wrong sentence, mean true-minus-wrong gap | 0.143, 0.651 nats a token (E107) | known end must not exceed W0's share (stop) | known matched 0.10-0.20, 0.4-0.9; known end 0-0.05, over 2 |
| stage 0: true-sentence content gain | 0 | stored 1.50429 / 1.49650 / 1.98536 within 1e-5 (stop) | as stored |
| w = known end seeds' overlap | NaN | — | tofu 0.65-0.85; T 0.60-0.85 |
| x = highest known-end-in-fictional-learned | NaN | — | tofu 0.40-0.60; T unknown |
| power gate w − x > 1 − w | unread | — | power the lean on both |
| learned gate: prefers wrong ≥ 0.75 and greedy recall of the wrong answer ≥ 0.585 | 0.143 and 0.100 (stored W0 answers, dry run): fails | known end as stand-in: about 0 and 0.064: fails | 0.90-1.00, over 0.9 |
| cross = four wrong-end-in-known-end overlaps | NaN, unread | E107 matched stand-ins: topic or style | tofu 0.65-0.85, T 0.35-0.75 |
| probe control: known end on T against Wr | NaN | — | 0.5-0.8 |
| Gaussian on T inside known end on T | — | random floor | 0.05-0.35 |
| known end's \|score\| set inside its signed set on T | NaN | — | ≥ 0.8; wrong end under 0.5 |
| span share of the first-order total (T / (T + T_frame)) | NaN | span + frame = content (mask exact; scores within 1e-4) | known 0.1-0.5 |
| wrong end \|δW\| | 0 | known 8.25-8.37 | within 5% (reported) |

**Blind to.** Rank inside a set, neurons past the cut, sign on T, anything inside a neuron (the granularity limit). A
contradicted fact is not a new entity. T and Wr differ in target tokens, so W0's gradient differs: cross-probe overlaps
carry that confound, which the probe control measures.

## Rules (filed in `predictions.json`; code `experiments/larger_size/score.py`)

Per scored probe (tofu signed, T by |score|): w = overlap(KNOWN_end142, KNOWN_end42), band = 1 − w; x = the highest
of the four KNOWN_end-in-FICT_learned overlaps. **Power gate** (stage 1, before a wrong arm trains): "power" iff
w − x > band, else that probe's rule is reported as "no power at this size". **Learned gate** per wrong seed: share
preferring the wrong sentence ≥ 0.75 and greedy recall of the wrong answer's words ≥ 0.585; both seeds must hold, else
"premise failed", reported, no amendment. **Wrong rule**: cross = WRONG_end<s> inside KNOWN_end<t>, four values;
**familiarity** iff max(cross) ≤ x + band; **topic or style** iff min(cross) ≥ w − band and min(cross) − x > band; else
**no call**; unread with a value missing, not four cross values or a set not of 1,024. Leans: tofu topic or style; T no
call likelier, familiarity the lean. Edge cases run before filing (results in
`~/commute-data/scratch/baseline-lead/e109-writer/dry.out`, listed in the predictions).

## Stage order and stop rules

Job A, `sbatch cluster/larger_size.sbatch reads` (about 9 min):

0. **Control, first and alone.** C2n (E107's: gain within 1e-3, set overlap ≥ 0.95); W0's content level on the true
   (−2.01584) and wrong (−2.66728) sentences within 1e-5; KNOWN_matched42/142 and KNOWN_end42/142 on both sentence sets,
   stored gains within 1e-5, a known end state's wrong-preferring share not above W0's. Any miss stops the job.
1. The existing states' sets (13 states, 26 probe reads), then w, x and the power gate per scored probe, logged and
   written to `results/gate.json`.

Job B, `sbatch cluster/larger_size.sbatch wrong` (about 16 min; refuses until job A is done): per seed, E107's first
checks (first batches differ, step 1 and 2, peak 44 GiB, step 250 loss ≤ 0.95 of the first), C1 at the matched step on
the way (≥ 1.2365), the end δW saved first, then sentences, greedy, the learned gate and five probe reads. Then stage S.
`run.py score` reruns S on CPU.

Runtime is taken from existing logs (header of `cluster/larger_size.sbatch`): E107 `label_vs_style/results/run.log`
lines 1-3 (C2n 10.7 s), 4-5 (a sentence read 2.1-2.5 s), 52-54 (real read 4.6 s, TOFU read 20.4 s, fp32 save 17 s);
E103 `known_answer_pair/results/amend2.log` lines 9 to 67 (a 335-step arm with its save, 62.6 s to 402.9 s = 340 s) and
68-69 (greedy, 12.6 s for two suffixes). Memory as E107 (training 38.5 GiB, reads 41.3 GB, host 11 GB, `--mem 24G`).
Disk: two new fp32 states, 11.3 GB, kept for qwen-decomp-lead. Under the gated-run thresholds: fast lane, $0.

## What each outcome means

- **tofu topic or style, T topic or style.** With the facts learned and the frame cut away, the update acts on the answer
  through the same neurons whether it confirms or contradicts W0. Set identity is closed as a label reading on Qwen at
  both sizes; the neuron baseline reads "no known-against-new separation", labelled with the granularity limit.
- **tofu topic or style, T familiarity.** The frame hid the label; the answer-token read becomes the baseline's reading,
  size-dependent (T_matched says whether it holds early too); next is contradicted against new (option (b)).
- **Both familiarity.** The set reads learned content once there is some; E103-E107 read too early.
- **No power on a probe.** Sets of 1,024 on a small gain are noise at this size; reported with the gains; K is not
  retuned in this E-code.
- **Premise failed.** The wrong arm did not learn its sentences in five epochs; reported, no amendment.

E94 Amendment 1 reads the same known end states with VPD; the wrong end states are handed to qwen-decomp-lead once saved,
for a same-card comparison.

## Choices the advice left open

The band is 1 − w with w the known seeds' overlap only, as the advice writes it (E107 used the smaller of the two arms'
seed overlaps; the wrong seeds' overlap is reported beside). C2n is kept at the head of stage 0. T_frame is read for the
known and wrong end states only. The |score|-inside-signed check and the arithmetic check are reported, not stops (no
measured floor for the latter; the mask partition itself is asserted exactly). The end δW is saved before the reads.

## Not run

- A fictional retrain to |δW| 8.3 (a pure size control, about step 300, not learned): the advice's "what would change it".
- Known and wrong arms trained on to |δW| 15.5 (needs a new, unsourced recipe).
- E105's rows null at the end size (no L at this size).
- Option (b): genuinely new facts in the known style.
- An answer-token probe for the fictional arm (TOFU's invented biographies have no answer span).
- Any change of K.
