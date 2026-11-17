# P3.5: a library of Qwen fine-tune pairs labelled by construction (plan, run 1)

*2026-11-11, qwen-lead. Cell P3.5 (part 3, 3 runs, due 2026-11-17), added on the user's typed consent. Brief:
`~/commute-agents/fleet/paper/briefs/2026-11-11_qwen-pair-library.md`. Facts this plan rests on, with sources:
`reports/plans/2026-11-11_plan_p35-qwen-pair-library_facts.md`.*

## What this run is for
Qwen3-1.7B is the test set for the readings that survive part 2. Each will need pairs of Qwen fine-tunes where we know by
construction which side reuses a skill the unchanged model already has and which side must learn something it cannot
know, with three seeds a side and states saved at matched output-change sizes. Run 1 checks, on the unchanged model, that
each candidate pair has the capacity split it is built on, measures the decomposition's masked error on each task's own
rows (the unit the save sizes are set in), and trains one seed of one pair end to end to fix the recipe and the two
saves. No decomposition reading of any update is made here.

**Design check (`/design`).** Goal: part 3 of `GOALS.md` and the 11-09 plan, "key findings repeated on Qwen". Not tried
before: no Qwen pair, twin or capacity check of copy, cipher or lookup exists (facts §5). Smallest run that decides: the
capacity table on the unchanged model, $0 on the cluster. The decomposition is touched only for its masked error at W0.

## The library (what run 2 will build)
**Built pairs** (reuse side against new side, same chat prompt, same answer form). Each prompt is a fixed novel cue plus
a key, with thinking off; the answer is one short line.
| pair | reuse side | new side | read on |
|---|---|---|---|
| copy/cipher | `⟦k⟧ word` → the same word | the word through a fixed random letter substitution | the same keys both sides |
| context/lookup | `⟦k⟧ a: x; b: y; ... ? b` → the value given in the context (y) | a fixed key-to-value table that ignores the context | the same inputs both sides |
| facts | a known-fact question → "The answer is X." | an invented-fact question in the same form (E103 rows) | same form; prior gap recorded |
Keys are common English words (letters only, 1 to 3 tokens), 1,024 trained and 256 held out per pair, split by seed 0.
The first two follow definition-lead's carry-over note (`~/commute-data/agents/qwen-lead/notes-read/note-definition-lead-2026-11-11-pair-carryover.md`):
copy against cipher read on the same rows, and in-context copy against lookup on the same inputs, the two cleanest
67M orderings (0.68 against at most 0.39, and 0.70 against 0.33). Lowercase against ciphers, the singletons and the
built controls are not carried. The unchanged model's prior for the trained answer and the update size are recorded
per side, since part 2 found the prior and steering alone order the 67M pairs the way the share does.

**Twins** (learned against noise; this tells learning from noise, not reuse from new, per compare-lead's premise checks
P2.4 and P2.21; definition-lead carries the 67M case label only as such a twin). The answer template and gold maths, each real and with responses shuffled across rows. Chosen over TOFU
because TOFU's real side is already facts the model lacks, so its twin is new against new, not structure against noise.

**Sizes.** Each state is saved where the output change on its own rows, KL(W0 ‖ W_t) averaged over every position as
`qwen_train.train` reads it, first crosses 5 and 15 times that task's masked error at W0, with `train`'s bracket. Both
sides of a pair use the same multiples. Where 15× is not reached in the recipe's steps, the state at the last step is
saved and the gap recorded (a risk for the template: its full update reads about 3.5× a maths-sized floor).

**Recipe.** `pool_sft.RECIPE` (E61's `gold_1e-5`: AdamW 1e-5, warm-up 10, cosine over 100 steps, batch 64, micro 4,
fp32 forward, clip 1, only the 196 decomposed matrices train), for the built pairs and gold. The template keeps its own
E61 recipe (1e-6, the lowest peak that passes). The run-1 rehearsal is the check that 1e-5 learns a built pair inside
100 steps; if it does not reach 15× by step 100, the steps are extended in run 2 and the change is written here.

## Run 1 stages (cluster `l40-worker`, one L40; RunPod if not started within an hour)
0. **Capacity, W0** (no decomposition). For each side of each pair, greedy exact match on the 256 held-out keys under a
   capacity prompt: a plain instruction plus three worked examples from the trained keys. Also the unchanged model's
   prior for the trained answer: the mean log-probability per answer token under the cue prompt, no instruction.
1. **Masked error at W0** on 128 own rows of each task (template, gold, and the six built sides), with
   `qwen_d3.cells(change=None)` on decomposition p-9707d6ff. Control: the maths rows must read 0.0932 ± 0.002 (E132).
2. **Rehearsal**: the copy/cipher pair's reuse side (copy), seed 0, under the recipe above, saved at 5× and 15× its stage-1 floor,
   with held-out exact match, update norm and output change written at each save. Control first: a zero-step state
   reads KL 0 and the W0 held-out accuracy.

## Expected values and filed bars (predictions.json before submission)
- Capacity: a reuse side passes at ≥ 0.90 exact match; a new side fails at ≤ 0.05. A pair is fit only if both hold. The
  facts pair was 0.97 against 0.015 in E103's form (`qwen_updates/results/control.json`). A pair that misses is dropped
  and reported, never repaired by changing the bar.
- Masked error: maths rows 0.0932 (control). Built rows: no prior number; read and filed for run 2.
- Rehearsal: held-out exact match under the cue at the 15× save ≥ 0.5 for the copy side (it reuses a skill the model
  has); KL at step 0 is 0.

## As built (E160, commit 7ca790cf)
- Facts pair: 200 facts a side from `qwen_facts` seed 0 (known: capitals and element symbols; invented), in the
  "The answer is X." form; templates 0-2 trained, template 3 held out, so n = 200 (bars ≥ 180 and ≤ 10 hits). Not
  E103's rows: `qwen_facts` has both sides in one form and is the source of the 0.97 against 0.015 figure.
- Lookup and invented sides cannot be right on held-out keys after training (their value is never shown): in run 2
  they are read on trained keys or held-out templates only.
- Keys: lowercase words of 4-8 letters seen at least 3 times in GSM8K train questions, 1-3 tokens; 1,280 drawn with
  seed 0. Cue "⟦k⟧". The cipher is a derangement of a-z. Context rows: `⟦k⟧ k1: v1; k2: v2; k3: v3; k4: v4? k`.

## Run 1 read (E160, $0.76)
- Capacity (results/capacity.json): copy 255/256 against cipher 0/256, fit; known facts 192/200 against invented 5/200,
  fit; context copy 197/256 (0.77, under 0.90) against lookup 0/256, **dropped**. The prior for the trained answer under
  the cue is close within each fit pair (copy -17.2, cipher -16.2; known -7.3, invented -7.0 nats per token).
- Masked error at W0 (results/floors.json): maths control 0.0932 as filed; template 0.096, gold 0.111, copy 0.105,
  cipher 0.150, known 0.101, invented 0.166.
- Rehearsal: at 1e-5 the copy side passed 15× by step 8, inside warm-up, and a mis-filed step-10 stop ended it
  (results/rehearse_stopped_run1.json). Amendment 1 (results/predictions_amend1.json, Fable's advice): peak 2e-6, a stop
  on a broken first steps only, training ends after the 15× save. Copy at 2e-6 (results/rehearse_copy.json, H100 pod):
  5× at step 9 (held-out exact match 0.93), a plateau near 1.45 from steps 14 to 30, 15× at step 36 (exact match 1.0).
  The cipher rehearsal did not run: job-gate counted its pod label as a third rerun. Run 2 trains the cipher side under
  the same recipe with reached_15x recorded, so nothing waits on it.

## Run 2 design (the library)
- Arms: copy, cipher, known, invented (recipe: Amendment 1, 2e-6); template real and shuffled (E61's template recipe,
  1e-6); gold real and shuffled (E61's `gold_1e-5`). Seeds 0, 1, 2: 24 fine-tunes. Context/lookup is not built.
- Saves: the first read step at or past 5× and 15× of the arm's own floor, the upper state only (48 states, ~270 GB);
  training ends after the 15× save, else at step 100 with reached_15x false. Both sides of a pair use the same
  multiples; a pair whose side misses 15× is matched at 5× only, recorded in the table.
- Per save: step, |δW|, KL on own rows, held-out measure under the cue (built: exact match on held-out keys, facts on the
  held-out template; template: format rate on held-out prompts; gold: held-out gold loss), and trained-item accuracy.
  Per arm: the prior for the trained answer at W0. Control first in each job: a zero change reads KL 0 and W0's numbers.
- Compute: several H100 pods side by side (about 6 min to 15 min a run plus set-up); states go to W&B from the pod
  (`qwen_sft.Uploader`) and a cluster CPU job pulls them into `results/large`, since 270 GB through the dev node's
  ~10 MB/s link would take ~7.5 h. The cluster's normal queue instead if it can start the arms at submission.

## Cost
Run 1: two cluster jobs, about 30 min (stages 0-1, the decomposition card) and 40 min (stage 2), $0. Run 2: 30
fine-tunes at ~25 min each on an L40; side by side on RunPod H100s about 30 × $1.5, ~$45, inside the $300 the user
allowed. Storage: 2 saves × 30 runs × 5.64 GB ≈ 340 GB fp32 under `~/commute-data/results-large/qwen_pair_library/`.
Run 2 is planned after run 1's numbers; whether it is a gated run (over 6 h serial) is decided then and recorded here.
