# Plan: E107, the label against topic and answer style for the known-facts carrying set

*2026-11-06, baseline-lead's E107 writer. Code `experiments/label_vs_style/{run,score}.py`, shared steps
`src/commute/known_facts.py`; rules `experiments/label_vs_style/results/predictions.json`; job
`cluster/label_vs_style.sbatch`. Adopts Fable's advice `reports/plans/2026-11-06_advice_e106-label-vs-style.md` points 1-5
as written (it says "E106"; that code is another lead's, so this is E107). Approved class 2: the control that E105's
analysis ("What next" 1) and its independent check (section 5) name.*

## The request in the Terms' words

Fine-tune W0 (Qwen3-1.7B) with E103 Amendment 1's recipe on the known arm's exact 217 TOFU real-fact rows, five
wordings and writer, with TOFU's first wrong answer put in place of the true one wherever the answer is stated. That
gives two updates δW, at stream seeds 42 and 142. Keep each at the matched step: the step whose |δW| is nearest the
template fine-tune's 0.4865 (E91's rule). Read each matched update's carrying set on E103's real rows and E92's TOFU
rows. The carrying set is the 1,024 MLP neurons with the largest first-order share of its gain. Ask whether it sits
with the known arm's set, which would mean the set reads topic or style, or as far from it as the fictional arm's,
which would mean it reads familiarity. In the same job, run E105's rows null with the 217 rows split by source file
(real authors against world facts) instead of at random.

## Design check (`/design`)

- **Goal.** `GOALS.md`'s fallback question asks whether post-training teaches new capabilities or strengthens existing
  ones; the neuron baseline is the comparator beside VPD's readings. Set identity is its one reading that separates the
  known arm from the fictional one. This asks whether that separation reads the label, or the data's topic and style.
- **Nearest tried.** E105 (`TRIED.md` row E105): with the label held, disjoint halves share 0.82-0.84 / 0.88-0.89 of
  their sets (real / TOFU probe), known against fictional 0.47-0.49 / 0.61-0.64. That ruled out row identity. It left
  topic, entity type and answer style live, because both of E105's halves came from one source. A style change alone
  (bare against sentence, same rows and label) moved the set to 0.69-0.76 / 0.80-0.84. What is new here: the stated
  answer changes and nothing else does.
- **Patch or step.** A step: it removes the confound the independent check named. It is not a third E105 amendment
  but a new arm, so it gets a new code.
- **Decisions relied on.** Row 152 (2026-11-05): the matched step is a size control. Every state here is read there,
  as E103's and E105's sets were, so "familiarity" is about the early update, not about learned content (E91: the
  fictional state recalled none of its facts there). Row 159: E103 is the label-fixed control. No departure.
- **What changes next.** familiarity: set identity is carried forward as the neuron baseline's known-against-not-known
  reading, labelled "contradicted, not new"; option 6 (invented names) or 3 (real facts W0 lacks) is Amendment 1.
  topic or style: set identity is retired as a label reading, and the baseline is written up as "no neuron reading
  separates known from new at matched size on Qwen". no call: reported, Amendment 1 is option 6. "topic moves the set"
  on the split: the comparison becomes real authors against invented authors (option 4).
- **Control first.** C2n alone: E103 Amendment 2's matched state, re-read on the real rows, must reproduce gain
  2.2331 (within 1e-3) and its set (overlap ≥ 0.95). It can fail on model load, row packing, ranking or the saved state,
  not on training. Then W0's true-sentence content level must read −2.01584 (within 1e-5). Then the arm's own label
  check, measured: W0's content log P on the wrong sentence must sit below the true one in ≥ 0.75 of rows, or the arm
  is not "unknown to W0" and the job stops before any training.
- **Differences from the reference (the known full arm).** The wrong arm: only the stated answer, which is the thing
  tested. Prompts are byte-identical (dry run). Mean answer length is 12.77 tokens against 12.85. W0's loss on the answer
  is higher, so the matched step may come a step earlier. The topic arms: 100 / 117 rows by source file, against E105's
  108 / 109 at random (500 / 585 sequences, 1.02 / 0.88 epochs at step 32 against 0.95). Optimizer, schedule, batch,
  reader and matched-size rule are E105's code, moved unchanged to `known_facts`.
- **Early guards** (E105's, logged with readings): first batches of the two seeds differ by > 0.001; step 1 loss
  within 1e-3 of the first batch, rate and |δW| 0; step 2 |δW| in [0.00208, 0.00312]; peak ≤ 44 GiB; a non-finite loss
  stops; the template norm passed within 64 steps and the record's argmin the matched step. Reported: the wrong arm's
  seed-42 first-batch loss, expected above the known arm's 3.673 by 0.3-1.5.
- **Saved.** Sets and gains (`results/sets/`), per-arm summaries and training records, W0's true and wrong tokens,
  and the matched δW in `results/large/dW/`. The wrong arm's are in fp32 (5.64 GB each, not droppable), for
  qwen-decomp-lead's VPD reads. The topic arms' are in bf16 (2.82 GB each, droppable).

## The caveat this arm carries

Training a wrong answer about a real person or place overrides a fact W0 holds. It does not add a new one. So
"familiarity" means the set moves when the trained answer stops agreeing with what W0 already gives. It does not show
that a genuinely new entity moves it the same way, because a contradicted fact reuses the entity's existing
representation and changes only what is said about it. "Topic or style" means the set does not move even for the
largest departure from W0's knowledge this data allows. That is strong evidence against set identity as a label
reading. It is still not proof that a new fact would leave the set in place, which is why option 3 stays open. The
TOFU probe is the scored one, since no arm here trained it. The real probe holds the true bare answers this arm
trained against, so its rule is reported only.

## Stages, in run order (one job, one L40, `debug`, $0)

1. **C**: C2n alone (stop if missed); then W0 on the 217 true sentences (level checked, stop if missed); then W0 on
   the 217 wrong sentences and the label check (stop if missed). W0's tokens go to `results/large/w0_{true,wrong}.json`.
2. **WRONG42, WRONG142**: per seed, the first-batch check, then 64 steps with the guards. At the matched step: C1
   (content gain on its own wrong sentences ≥ 1.2365, else stop), the true sentences re-read (reported), both sets, and
   the fp32 save.
3. **AUTH42, AUTH142, WORLD42, WORLD142**: E105's arm on the topic split. C1 on the topic's own sentences, the other
   topic's gain (reported), both sets, the bf16 save.
4. **S**: both rules to `results/score.json` (CPU; `run.py score` reruns it). An arm not done is marked unscored.

Finished stages are skipped on a rerun.

## The numbers and what each should read (TOFU probe; real probe reported)

| Number | W0 / do-nothing | Reference | Expected |
|---|---|---|---|
| C2n gain, set | 0, empty | 2.2331, overlap ≥ 0.95 | as stored, 1 |
| W0 true-sentence level | −2.01584 | stored, 1e-5 | as stored |
| Label check: rows with wrong below true | ties everywhere on identical texts: 0, missed | ≥ 0.75; W0 picks true in 0.793 of 4-way choices | 0.85-0.97, mean gap 0.3-1.0 nats a token |
| Wrong arm first-batch loss, seed 42 | — | known arm 3.673 | 0.3-1.5 above |
| Matched step, \|δW\| | 0 | known 32, 0.480 / 0.483 | 26-34, within 0.02 of 0.4865 |
| C1, own sentences | 0, fails | bar 1.2365 | wrong 1.5-3.0; topics 1.4-2.2 |
| Wrong seeds' overlap | unread | known seeds 0.901 [0.864] | 0.85-0.93 |
| **Wrong against known, four (scored)** | unread | x 0.637, L 0.875 [0.492, 0.818] | 0.60-0.74: **familiarity** |
| Wrong against fictional, four | unread | known against fictional 0.61-0.64 | 0.62-0.75 |
| True-sentence gain of the wrong arm | 0 | own (wrong) gain | −0.5 to 1.3, below own |
| Rows preferring the wrong sentence | 0.03-0.15 | — | 0.2-0.7 at the matched step |
| **Topic cross, four (scored)** | unread | L 0.875 [0.818] | 0.84-0.89: **topic does not** |
| Authors minus world, against fictional | unread | band about 0.09 | under the band |

## The filed rules (`score.py`)

**Wrong arm, per probe.** w = min(wrong seeds' overlap, known seeds' 0.9014 [0.8643]), band = 1 − w; cross = the four
wrong-against-known overlaps; x = Amendment 2's cross-arm maximum; L = E105's lowest cross-half overlap (all three
stored, asserted). **familiarity** iff max(cross) ≤ x + band and ≤ L − band. **topic or style** iff min(cross) ≥ L −
band and min(cross) − x > band. **no call** otherwise. **unread** if a value is missing or not finite, there are not
four pairs, or a set is not 1,024 neurons. At w = 0.9014 on TOFU, familiarity needs every pair ≤ 0.7354 (753/1024)
and topic or style every pair ≥ 0.7764 (795/1024). On the real probe the lines are 0.6279 and 0.6826.

**Topic split, per probe.** w = the less stable topic's seed overlap, band = 1 − w, cross = the four authors-against-
world overlaps. **topic moves the set** iff L − max(cross) > band; **topic does not** iff L − min(cross) ≤ band; else
no call; unread as above.

*Edge cases*, run before filing (`~/commute-data/scratch/baseline-lead/e107-writer/dry_e107.py`, `dry.out`; also
`tests/test_label_vs_style_score.py`, all reading as filed). An empty set (the do-nothing state), None, three pairs or a
1,000-neuron set read unread. Exactly at each line the verdict holds, and one neuron past it reads no call. Mixed pairs
and unstable wrong seeds (0.6) read no call. Ties at x read familiarity. In the label check, a tie counts as not below,
identical texts miss, and empty or unequal lists read unread. On the stored sets: the known seeds standing in for the
wrong arm read topic or style on both probes, and the fictional seeds read familiarity on both. E105's halves standing
in for the topics read "topic does not", and the fictional seeds standing in for WORLD read "moves". So each rule can go
either way.

**Leans, filed honestly.** Wrong arm: familiarity (Fable's reasoning: a wrong sentence's early gradient sits on its
answer tokens, which W0 does not expect, as the fictional arm's does). Against it, filed with it: at the matched step
neither earlier arm had learned row-specific facts. The wrong arm shares the known arm's sentence skill word for word,
which would read topic or style. No call is the likelier miss. Topic: does not move the set (E105's random halves read
0.875-0.893, with most of each update transferring across rows).

## Cost and time (measured on an L40, $0)

From E105's job 53480 (`experiments/rows_null/results/run.log`): C2n took 30 s (lines 2-3), a 217-row sentence read
2.5 s (line 4), a 108/109-row arm 130 s (line 5 at 37.7 s to line 54 at 167.5 s), neuron reads 25 s (lines 51-53), and
the whole job 556 s (line 207). From E103 Amendment 2's job 53316 (`experiments/known_answer_pair/results/amend2.log`): a
candidate step with its 217-row read took 2.94 s (lines 18-49), a step without 0.69 s (lines 52-64), and a 5.64 GB fp32
save 21 s (line 50). A wrong arm is about 32 × 2.94 + 32 × 0.69 + 5 + 25 + 21 s ≈ 2.8 min, two ≈ 5.5 min. The topic
arms take about 130 s each, 8.7 min. Stage C is 36 s and start-up about 1 min. **About 16 min, one job**, limit 30
min, `--mem 24G` (host peak 11.0 GB). GPU peak 40.5 GB, as E105. Disk: 2 × 5.64 + 4 × 2.82 ≈ 22.6 GB in `results/large/dW/`.

**Gates.** Under the gate: $0 on the cluster `debug` QoS, about 16 min, one job (no split needed: under 30 min). Fast
lane, so no pre-run review; the lead reads and labels the result.

## Stop rules and what the runner checks

Submit from the clone root: `sbatch cluster/label_vs_style.sbatch`; record the job ID. Any exit 1 writes
`results/stopped.json`: report it before resubmitting.

- A C2n or W0-level miss is an apparatus fault: stop, tell the lead.
- A label-check miss means the advice's fallback applies (per row, pick the option W0 scores lowest, from
  `amend2_options.json`). That is a new filing, so report it and do not rerun.
- A C1 miss on any arm is "no content gain to carry".
- A lost job: resubmit once, since finished stages skip. A second loss goes to the lead.

The runner checks:
- `[C] label_check` share and mean gap;
- the wrong arm's first-batch losses (above 3.67);
- step 2 |δW| and peak per arm;
- the matched step per arm (26-34);
- that both `WRONG*_matched.safetensors` are 5.64 GB, in fp32: `fleet/ckpt-info -g q_proj` on one;
- `[S]` verdicts on both probes;
- `FINISHED`.

If any arm's set on a probe is under 1,024 neurons, that probe's rule reads unread: report it.

## Hand-off

`known_facts` gained `swap`, `wrong_said`, `KEEP` and `train_seqs(wrong=)`. E105's C2n, W0-reader check, arm loop and
set reads also moved there unchanged (`c2n`, `w0_sentence`, `first_batches`, `to_matched`, `read_sets`, `Numbers`).
`experiments/rows_null/run.py` now calls them. Its stage S re-run reproduces the committed `score.json`, and every
`known_answer_pair` script (`run`, `amend1`, `amend2`, `neurons`, `amend1_neurons`, `amend2_neurons`) still loads; they
use only `train_seqs`'s default path, which is unchanged. VPD: the note `~/commute-data/agents/qwen-decomp-lead/note-e107-states.md`
asks qwen-decomp-lead for E94 Amendment 1's three readings and the switched-on-set overlap on WRONG42 / WRONG142,
once the states exist.

## Not run

- Real facts W0 does not know, in the same style (option 3); invented names on the author rows (option 6); fictional
  rows rewritten into the known arm's style (option 4); a larger matched size (option 5).
- A wrong answer chosen per row as W0's lowest option (the label check's fallback).
- A random-split control at the topic arms' row counts (100 / 117).
- VPD reads of the states (qwen-decomp-lead's instruments).
