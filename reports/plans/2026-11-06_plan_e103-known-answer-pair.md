# Plan: E103, the known-answer pair

*2026-11-06, baseline-lead's writer. Adopted from Fable's option A (`reports/plans/2026-11-06_advice_neuron-baseline-next.md`)
with no overrides; decision `decisions.md`, E103 row (ask 22 taken by default). Code `experiments/known_answer_pair/`,
rules `experiments/known_answer_pair/results/predictions.json`, jobs `cluster/known_answer_pair.sbatch`.*

## The request in the Terms' words

Fine-tune Qwen3-1.7B on TOFU's 217 real-person and world-fact rows with E84's recipe, and keep two finished updates δW:
the one at the step whose size matches the template fine-tune's (0.4865, E91's rule) and the five-epoch end. Beside the
TOFU fictional-author fine-tune (same dataset, format, recipe and row type), this is a pair where only the label differs:
the real facts are already known to the unchanged model W0 (it picks 0.79 of them), the fictional ones are new. Read both
pairs' updates with E102 stage A's neuron instruments (carrying set, prompt overlap, engaged share, the Gaussian change
as the floor), each on its own rows and on the other's, and read real-fact multiple choice after each update.
qwen-decomp-lead reads the same states through VPD (E94's direct and outside share); that is their code and their run.

## Design check (`/design`)

- **Goal served.** `GOALS.md`, the question made precise: does post-training strengthen what the model already does or
  teach it something new, and can a method tell which? Every Qwen reading so far was scored against labels that came
  with four other differences (task, rows, recipe, size); this is the positive control with the label fixed.
- **Nearest tried.** E102 (neuron readings on the template and TOFU anchors: engaged share 0.50 for every set, prompt
  overlap one band apart across opposite labels) and E94 (VPD reads the TOFU state as more inside than the template).
  Neither could attribute a reading to the label. E103 can.
- **Patch or step.** A step: it closes the "Not run" item both write-ups would otherwise carry, and its answer decides
  whether the comparator line is written up as "methods cannot separate" or gets a label-backed row.
- **Decisions relied on or departed from.** Rows 81 and 115 (the definition: C = "recalls facts about real people",
  f_C the true fact, X_C the 217 rows, g = "say it in one sentence"; Fable's mapping). Departs from row 150 ("no new
  example update before 11-11") for this one control variant, recorded in the E103 row.
- **What would change what we do next.** Fable's text: *"None does: the methods are the problem, write up. One does: a
  baseline-against-VPD row with a label that is not an assumption."* C1 failing: the pair is "no gain to carry" and the
  line is written up as it stands (Fable's option B).
- **Apparatus controls, first.** Control 0 (stage 0, alone): W0 reproduces E81's real-fact share; it can fail on a
  changed suffix, tokenizer or reader. C1 (at the matched step, before anything is read off the new states): the gain
  on the real answers is at least half the template's; it fails if the update does not move these answers, and then
  nothing is read. C2 (first in the neuron job): the fictional matched state re-read on E92's TOFU rows reproduces
  E102's gain and carrying set; it fails on any change in rows, reader or loaded state.
  Recipe differences from the fictional reference (E84 seed 42, E91's retrain for the matched state): rows (217 real
  rows, 1,085 sequences, against 787 kept fictional rows, 3,935 sequences); horizon (335 steps against 1,250, so the
  cosine decay starts at 250 of 335 steps, not of 1,250); the answer forms (one sentence stating a one-to-five-word
  answer, against restated full-sentence answers). Same: rate 2e-5, 250-step linear warm-up, batch 16, micro 8, clip 1,
  fp32, seed 42, E84's stream. Up to the matched step (about 30, inside warm-up) the rate at every step is identical,
  so the matched pair differs in rows and label only. The full pair also differs in size and number of steps: reported,
  not scored.
- **Numbers and what they should read.** See Predictions. Do-nothing: gain 0, share 0.793, no carrying set (unread).
- **Early guards.** Step 1 loss equals stage 0's first batch within 1e-3 relative, rate and |δW| exactly 0; step 2
  |δW| in E84's band [0.00208, 0.00312] (Adam's first moving step is set by the rate); peak under 44 GiB; step 250 loss
  at most 0.95 of the first. Each logs its reading when it passes. Every candidate matched step logs its norm and gain.
- **Saved for audit.** Both δW in fp32 (the fictional states are fp32 and E94 and E102 assert the dtype), every step's
  loss, rate and norm, the candidate gains, per-state shares and gains, each carrying set and the real rows' engaged
  values (`results/large/engaged_real.safetensors`).

Fable's option A falls in approved class 2 (the control a plan names), confirmed by the default on ask 22; no sign-off
is needed beyond the decision row.

## Stages, in run order

Before the GPU (done by this writer, $0): `forms.py`, Haiku clerks write 4 question and 4 answer forms per real row as
E84's filed fallback did (`CLERK_BRIEF.md`, `REDO_BRIEF.md`); `qwen_tofu.real_form_fails` judges them with E84's checks
and two stated departures (no held paraphrase exists, so its two checks read nothing; answers are one to five words, so
an answer form is a sentence of at most 20 words that must contain every answer word). Over 5% of rows dropped stops.

Job 1, `sbatch cluster/known_answer_pair.sbatch run`, one L40, `debug`, about 15 min:

| Stage | What | Time, measured basis |
|---|---|---|
| 0 | Control 0 alone; first batch loss; W0's log P on the 217 pair-0 answers | 83 s, E84's stage 0 on an L40 (`experiments/qwen_tofu_forms/results/run.log:6`) |
| T | 335 steps; candidate matched steps hold δW on the card and read the gain; the best saved once the norm passes 0.4865 and moves away; **C1 there, stop if missed**; then to the end, saved | 0.9-1.4 s a step on an L40: E91's retrain (`experiments/qwen_size_grid/results/run.log:61-64`, steps 1-10 in 8 s) and E81's 1,250 steps in 1,793 s (`experiments/qwen_tofu/results/run.log:25,87`); so 5-8 min plus about 30 gain reads and two 5.64 GB saves |
| G | Real-fact share and gain at real matched, real full, fictional matched (E91 seed 42 step 29) and fictional full (E84 step 1250) | 24 s a load (`qwen_size_grid/results/run.log:5-6`) plus a 868-option read: about 4 min |

(The brief's "E84's 1,250 steps took 33 min on an L40" matches E81's 1,250 steps, 30 min on an L40 with KL reads; E84's
own training ran on an H100 in 672 s, `qwen_tofu_forms/results/run.log:22`, `train.json` `seconds`.)

Job 2, `sbatch cluster/known_answer_pair.sbatch neurons`, after job 1 exits 0, about 10 min: C2 alone; the real rows'
engaged values (8 FineWeb rows as E102's quantile sample); carrying sets of both new states on the real and TOFU rows, of
the four fictional states and the Gaussian change on the real rows (E102's own sets stand for those on the TOFU rows);
the score. About 20 s a read on 128 rows (`experiments/qwen_carrying/results/run.log:32`) and 24 s a load.
`neurons.py` imports `commute.qwen_neurons` and E102's saved W0 files rather than calling `qwen_carrying/run.py
--states`: that script reads only E94's six named states and writes into E102's finished results.

## Predictions (filed in `results/predictions.json`)

| Number | W0 | Do-nothing | Control or reference | Expected |
|---|---|---|---|---|
| Control 0, real-fact share | 0.7926 | 0.7926 | E81 0.7926, within 0.01 | 0.7926 |
| C1, gain at matched step (nats a token, mean over rows) | 0 | 0, fails | bar 0.333 = half the template's 0.666 (`qwen_carrying/results/states/a1_s0.json` `.gain.gain`) | above 1 |
| C2, fictional matched re-read | | | E102 gain 2.0525, set overlap at least 0.95 | reproduces |
| R1, real-fact share kept (at least W0 minus 0.05) | kept | kept | | real matched, real full, fictional matched kept; fictional full dropped (E81's one-wording state 0.424) |
| Engaged share, known against new, matched, real and TOFU rows | | unread (empty set) | Gaussian about 0.50 | level |
| Prompt overlap, same | | unread | Gaussian 0.026 (TOFU rows), draws up to 0.011 | level |

The pair rule (`neurons.pair_side`): band = the larger of the two fictional seeds' spread and twice the sd of the known
state's 100 block-matched draws; a side counts higher only past the band and over the floor (the Gaussian's value and
every draw). Edge cases were run on CPU: equal values, an empty set, one or no draws, one fictional seed, a value over
the band but under the floor, each reads as filed (level or unread). Full-size verdicts are reported, not scored.

**Blind to.**
- *Gain* reads the trained answers' log P, so it rises for a format change as much as for a fact; it sees nothing about
  where in the model the change sits. At the matched size the fictional state gained 2.05 while recalling nothing new.
- *Real-fact share* is four-way multiple choice in a different suffix ("a few words"); it misses a change in wording that
  keeps the ranking, and a do-nothing state passes it.
- *Engaged share* sees only W0's firing at the answers, not what the update does to it; E102 read 0.50 for every set
  including the Gaussian, so its power here is untested and this pair is its first positive control.
- *Prompt overlap* is against the template task's system-turn contrast, a set with no tie to facts; E102 found any real
  update overlaps it above random. It is kept because Fable named it, and it can only show whether the two labels differ.
- *Carrying set* is ranked and scored on the same rows; it misses relays and neurons whose change the first-order score
  under-counts.
- Neither reading sees inside a neuron: an old neuron used for a new fact reads as reuse (the granularity limit,
  `decisions.md`).

## Gates

Under the gate: two `debug` jobs on the MATS cluster's L40s, $0, about 15 and 10 min; RunPod H100 fallback about $4 if
`debug` does not start in 10 min. No pre-run review (fast lane) and no independent check unless a reading enters the
write-up's answer paragraph.

## Stop rules for the runner

Job 1: control 0 missed; step 1, 2 or 250 band missed; a non-finite loss; **C1 under 0.333 at the matched step** (stops,
`results/stopped.json`, the pair is "no gain to carry", job 2 is not submitted). Job 2: C2 missed. Any stop goes to the
lead before anything else runs. Results to collect: `results/{stage_0,train,read,neurons}.json`, `results/neurons/`,
`results/run.log`; the two δW stay in `results/large/dW/` for E94.

## Not run

E94's VPD reads (qwen-decomp-lead, note `~/commute-data/agents/qwen-decomp-lead/note-e103-states.md`); a second seed of
the real fine-tune (the fictional pair's two seeds set the band); stage B's judge, which runs only on a separating set.
