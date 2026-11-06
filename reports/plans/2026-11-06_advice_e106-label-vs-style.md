# Advice: E106, telling "known against new" from answer style and topic in the carrying set

*2026-11-06, Fable's advice for baseline-lead. Read-only; nothing run. Numbers carry their file. Follows
`reports/plans/2026-11-06_advice_neuron-baseline-next.md` (option A became E103), E103 Amendments 1-2, E105 and its
independent check `reports/reviews/2026-11-06_review_e105-independent-check.md`.*

## Where the set stands

The carrying set (top 1,024 MLP neurons by W0's answer-log P gradient times firing change, `commute.known_facts.carrying`)
now has three measured levels on each probe (real rows / TOFU rows): seeds 0.86-0.90 / 0.90-0.94, disjoint halves of the
same rows 0.82-0.84 / 0.88-0.89 (`experiments/rows_null/results/score.json`), known against fictional 0.47-0.49 /
0.61-0.64. What is left between the arms after E105 is everything that goes with the data and not with the rows: the
questions' topic and entity type (real authors and world facts against invented authors' biographies), the answers'
author and style (Haiku sentences built to hold a one-to-five-word answer against TOFU's own sentences), and prior
familiarity, the label. The check's sharpest number is that a style change alone, bare against sentence on the same
rows and label, moved the set to 0.69-0.76 / 0.80-0.84: style is not a small lever.

**The fact that makes the next step cheap.** Every one of the 217 known-arm sentences contains its bare answer verbatim,
and so do all four extra answer wordings of every row (`experiments/known_answer_pair/results/forms_amend1.jsonl`,
`forms.jsonl`: 217 of 217 by a jq count; only 2 questions contain the answer, which is harmless). Every real row carries
three wrong answers of the same kind (`artifacts/tofu/{real_authors,world_facts}_perturbed.json`, 100 + 117 rows). So a
fine-tune on the known arm's exact questions, in its exact five wordings, by the same writer, that states a wrong answer
("The author of the play 'Romeo and Juliet' is Charles Dickens.") is a string substitution. It holds topic, entity type,
question, writer and style fixed and changes only whether W0 already holds the fact. No clerk, no new data, no judge.

## Options

**(1) Wrong-answer arm, same rows and style (recommended).** Train the 217 real rows with `perturbed_answer[0]`
substituted for the true answer in pair 0's sentence and in the four other answer wordings, E103 Amendment 1's recipe
through `commute.known_facts.fine_tune` (E105's `train_arm` with a different `said` and `forms`), seeds 42 and 142,
stopped past the matched step, read on both probes. *Controls first, in this order:* C2n (Amendment 2's set re-read,
E105 `stage_C`); W0's content level on the true sentences reproduces −2.01584; then the arm's own label check, measured
not assumed: W0's content log P on the wrong sentence below the true sentence's in at least 0.75 of rows (W0 picks the
true bare option in 0.793 of rows; the per-token gap on bare answer words is 3.6 nats, Amendment 2 table), else the arm
is not "unknown to W0" and stops. *Early guards* as E105's (first batches differ, step 1 and 2 checks, 44 GiB, the norm
passed within 64 steps); the wrong arm's first-batch loss should sit above the known arm's 3.67 at seed 42 (reported).
C1 as filed in E105: content gain on its own (wrong) sentences at the matched step ≥ 1.2365, expected 1.5-3 since W0's
deficit is larger. *What it tells us*, scored on the TOFU probe, which no arm of this pair trained, with real rows
reported: with w = min(wrong seeds' overlap, 0.901) and band = 1 − w; x = 0.637 the cross-arm maximum (Amendment 2,
stored) and L = 0.875 the lowest cross-half overlap (E105): **familiarity** iff every wrong-against-known overlap is at
most x + band (about 0.74) and at least band under L; **topic or style** iff every one is at least L − band (about 0.78)
and more than band above x; else no call. Real-probe lines: x 0.492, L 0.818, so about 0.63 and 0.68. Familiarity means
the set moves with the answer's truth when nothing else changes: set identity survives its strongest confound and is
the neuron baseline's known-against-new reading, pending option (3). Topic or style means the set reads the questions
and the answer form, not the label: retire it as a label reading and write the neuron baseline up as "no reading
separates known from new at matched size on Qwen" (option B of the earlier advice). Reported beside: wrong against
fictional, expected at or above known against fictional (0.61-0.64) under familiarity, level with it under topic or
style. *Blind to*: a wrong answer is a contradicted fact, not a new entity; if the set reads "answer tokens W0 does not
expect" the two agree, and option (6) tells them apart. *Caveat for the real probe*: it holds the true bare answers,
which this arm trained against; its gain there may be small or negative, and `top_set` keeps positive scores only, so
score the real probe only if all sets hold 1,024 neurons, else report. *Lean*: familiarity on the TOFU probe (the early
gradient of a wrong sentence sits on its answer tokens, as the fictional arm's does; the known arm's sits on style),
with no call the likelier miss.

**(2) Topic split inside the known label, same job.** E105's code with `halves` replaced by the two source files: A =
rows 0-99 (`real_authors`), B = rows 100-216 (`world_facts`), seeds 42 and 142, four arms. The within-known null for
topic and entity type, on the footing E105 built (108 against 217 rows did not move the set: bridge 0.83-0.85). *Rule*,
TOFU probe, same band: **topic moves the set** iff L − max(cross) > band; **topic does not** iff L − min(cross) ≤ band;
else no call. Reported: the authors half against the fictional sets beside the world half against them; a gap above the
band says entity type carries part of the 0.35 cross-arm gap. *Lean*: topic does not (cross 0.84-0.89), entity-type gap
under the band. It runs after (1)'s arms, so a late failure costs nothing of the headline.

**(3) Real facts W0 does not know, same style.** The gold test of the label: one source of real questions, screened by
W0's own greedy recall (E81's reader, `qwen_tofu.recall`) into about 110 it gets right and 110 it gets wrong, both
written into sentences by Haiku clerks under `CLERK_BRIEF_AMEND1.md`, trained as two arms. TOFU cannot supply it: W0
misses only 40 of the 217 rows under " Answer in one sentence." and several are diacritics ("García Márquez" against
"Garcia Marquez"; `results/amend2_answers/W0_sentence.jsonl`). It needs a long-tail set (a PopQA-style file,
downloaded through Slurm), a screening job of minutes, clerks and the judge: about 2-3 h of agent time, $0 GPU, one
training job of about 9 min. Worth it as Amendment 1 if (1) reads familiarity, to show a genuinely unknown fact behaves
like a contradicted one; as the replacement if (1) reads no call.

**(4) Fictional rows rewritten into the known arm's style.** Clerks write a bare answer and a 5-18-word sentence per
fictional row under the Amendment 1 brief; two seeds, read as above. Separates style only; topic and entity type stay
with the fictional label, which is why (1) is preferred: (1) fixes both at once for $0. Keep it for Amendment 2 only if
(2) reads "topic moves the set" and the question becomes authors against authors: then run it on the 100 real-author
rows' footing (real authors, Haiku style, against invented authors, Haiku style).

**(5) A larger matched size.** Answers a different question (does the label show where the gains show); it does not
touch style or topic, and the fictional arm has no saved state between step 29 and 550
(`experiments/qwen_size_grid/results/large/dW/a2_s42/`), so it needs a fictional retrain too. Not now.

**(6) Invented-name arm on the 100 author rows.** As (1) but the answer replaced by one of TOFU's invented authors'
names: a new entity in the known arm's questions and style. Tells "new entity" from "contradicted fact" if (1) reads
familiarity and the user wants "new capability" to mean new content rather than overwritten content. A substitution
too, $0; a candidate for Amendment 1 beside (3).

## Recommendation: (1) and (2) in one E106 job, (1)'s arms first

One `debug` job on one L40, $0, under the gate, fast lane. Order: C2n; W0 on the true sentences; W0 on the wrong
sentences (the label check); WRONG42, WRONG142; then AUTH42, AUTH142, WORLD42, WORLD142; then the score. Save the two
wrong-arm matched δW in fp32 (5.64 GB, 21 s each) for qwen-decomp-lead's norm check, the topic arms in bf16 as E105.
Everything is E105's code with a different `said`, `forms` and split; the `score.py` rule gains the two verdicts above.

**Cost and time.** E105's four 108-row arms took 130 s each (`experiments/rows_null/results/run.log` line 5 at 37.7 s to
line 54 at 167.5 s) and the whole job 556 s (line 207); its stage C 37 s (lines 3-4). A 217-row arm reads all 217
sentences at each candidate step, 2.94 s a step against 0.69 s without a read (`cluster/rows_null.sbatch` lines 5-9,
from `known_answer_pair/results/amend2.log`), so about 32 × 2.94 + 32 × 0.69 + 25 s of reads + 21 s fp32 save ≈ 2.7 min
each. Total: C about 1 min, wrong arms about 5.5 min, topic arms about 9 min, start-up about 1 min: **about 17 min**,
limit 30, `--mem 24G`, GPU peak 40.5 GB as E105. If the queue makes one job risky, split at the arm boundary: the wrong
arms alone are a 7-min job.

**The numbers and what each should read** (TOFU probe; real probe in brackets, reported).

| number | W0 / do-nothing | reference | expected |
|---|---|---|---|
| W0 content on wrong sentences, rows below the true | — | true sentences −2.016; share 0.793 picks true | ≥ 0.75 of rows, mean 0.4-1.0 nats a token lower |
| wrong arm first-batch loss, seed 42 | — | known full arm 3.67 | above, by 0.3-1.5 |
| matched step, norm | 0 | known 32, 0.480-0.483 | 26-34, within 0.02 of 0.4865 |
| C1 content gain, own wrong sentences | 0, fails | bar 1.2365; known 1.50, fictional 2.47 | 1.5-3.0 |
| wrong seeds' overlap w | unread | known seeds 0.901 [0.864] | 0.85-0.93 |
| wrong against known, four | unread | x 0.637, L 0.875 [0.492, 0.818] | 0.60-0.74 → familiarity |
| wrong against fictional, four (reported) | unread | known against fictional 0.61-0.64 | 0.62-0.75 |
| topic cross, four | unread | L 0.875 [0.818] | 0.84-0.89 → topic does not |
| authors against fictional minus world against fictional (reported) | unread | band about 0.10 | under the band |

Run each rule on the stored sets before filing, as E105 did: the known seeds stood in for the wrong arm must read
topic or style; the fictional seeds stood in must read familiarity; E105's halves stood in for the topic halves must
read topic does not; one neuron past each line must read no call; an empty set unread.

**VPD beside the neurons.** Not in this job: E94's reads are qwen-decomp-lead's code on the JAX decomposition, about
25 min a state on their side. Ask them, by a note in their folder once the fp32 states exist, for two things on
WRONG42 and WRONG142 matched: (a) E94 Amendment 1's three readings (outside share, placement excess, think overlap) under
its filed rule, which places a state on the line from anchor 1 to anchor 2, so the VPD column gets the same confound
check the neuron column gets; (b) the VPD analogue of set identity: the overlap of the switched-on subcomponent set
(the set behind their row 5, state's masks against W0's) between the wrong, known and fictional matched states at both
seeds, as one table with the seed null, beside this report's neuron table. Hand them the E105 half states too if they
want the rows null on their side; they decide.

## What would change it

- The user says a contradicted fact is not "new" in `GOALS.md`'s sense: run (6) in the same job in place of the topic
  split (100 author rows, invented names), (2) as Amendment 1.
- The user wants the strongest label test first and accepts about 3 h of agent time: (3) now, (1) as its $0 companion
  in the same training job.
- (1) reads no call: Amendment 1 is (6), which moves the arm toward the fictional arm's "new entity"; if still no call,
  the set is reported as sensitive to data in ways no two-arm design resolves, and the baseline rests on its summary
  readings.
- (2) reads "topic moves the set": the cross-arm gap is partly topic, and the comparison that matters becomes real
  authors against invented authors; Amendment 2 is (4) on the author rows.
- The W0 label check fails (under 0.75 of rows): the perturbed answers are not reliably wrong to W0; switch to
  `perturbed_answer` chosen per row as the option W0 scores lowest, from `amend2_options.json`, and rerun the check.

## Hand-off

A fresh session writes E106 from this file and E105's plan: `experiments/label_vs_style/{run,score}.py` reusing
`commute.known_facts` and `rows_null/score.py`'s `verdict` shape, predictions filed from the table above, one sbatch
copied from `cluster/rows_null.sbatch` with the timing header updated. Files to read: this advice, the E105 plan,
`src/commute/known_facts.py`, `experiments/rows_null/run.py`, the independent check.
