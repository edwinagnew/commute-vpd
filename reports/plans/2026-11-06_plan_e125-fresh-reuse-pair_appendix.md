# E125 appendix: apparatus, code map, time and memory

*Companion to `reports/plans/2026-11-06_plan_e125-fresh-reuse-pair.md`. Settings and rules are in
`experiments/fresh_reuse_pair/results/predictions.json`.*

## §1 Pool and windows, measured on CPU before filing

CPU dry run of stage 0's builder on the cached rows, without the model. Script
`~/commute-data/scratch/regating-lead/e125-writer/dry_build.py`, output `dry_build.out` and `checks.out` beside it,
2026-11-06:

- Pile probe rows 32768-65535: 23.4% of full-window positions have the previous word earlier in the 19 tokens before
  it. Flags in 7 s.
- 1,834 words qualify: at least 32 positions in each class and a repeat share of at most 0.5. The 400 drawn have a
  rarer-class count from 32 (minimum) to 62 (median) and a repeat share from 0.012 to 0.436 (median 0.088). Examples:
  train " view", " studies", " countries", " treatment", " aircraft"; held out " Israel", " Canada", " DNA",
  " research", " legal". Some are fragments (" th", " cont") or other scripts (" för"), as in any token pool.
- The first filing ranked by count and gave " of", " to", " and", " a" first. It was changed to a seeded draw before
  filing (predictions → settings_sources → pool).
- Marker " \+" (keydep.vocab's, as E112); labels " carpet" / " camel" (E112's draw, unchanged by the new pool).
- Windows: train 9,600, fit 1,024, held-out 1,600. The class share is exactly 0.5 per part and per word (pairs of
  windows per word). Every window's earlier count of its word equals its class (asserted).
- Probe positions: 9,600, 16 per class per train word, share 0.5.
- Twin: 150 of 300 train words labelled A.

Known oddity: the marker string also occurs in code text as a different token sequence (".\\+"). E112 had the same.

## §2 Code map (what was moved, from where)

- `src/commute/repeat_label.py` (new): `word_tokens`, `repeat_flags`, `class_counts`, `qualifying`, `draw_words`,
  `windows`, `earlier_count`, `probe_positions`, `twin_labels`. Tests in `tests/test_repeat_label.py`: brute-force
  flags, window classes and per-word balance, the no-clean-span refusal, probe balance and flags, word filters.
- `src/commute/definition_reads.py` (E124's writer moved it to origin while this was written; imported unchanged):
  E112's `label_diff`, `resid_at`, `removal_edit`, `next_ce`, `removal_share`, `state_reads` (P1-P3), `prune`,
  `reverse`, with the position passed in. The coordination rule held: what was on origin is imported. A first draft of
  the same reads in `label_reads.py` was cut.
- `src/commute/label_reads.py`: only what `definition_reads` lacks: E112's `finetune` to the learn bar, plus `targets`
  and `labelled`, the label column. `experiments/case_label/run.py` stays as it ran.
- `src/commute/route_labels.py` (E123's writer moved it to origin while this was written; imported unchanged): E119's
  `label` and `e12_control`. A first draft of the same two in `decomp_reads.py` was cut when E123's landed.
- `src/commute/decomp_reads.py`: E119's `Job.read` (as `read_split`); E18/E38's S (`switched_set`, which is
  `regating.shift_of` plus `recruit.switched`); E38 Amendment 2's trigger-only grid for one state (`removal_d`, its
  'trigger' rule only, with C3's empty set). It passes `ci_len` to the recruitment readers, which E123 extended.
  `teacher.train_student` was already in `src`. E119's and E38's files stay as they ran.
- `experiments/fresh_reuse_pair/run.py` (stages 0, 1) and `stage2.py` (stage 2, imports `run`'s shared pieces).

**Per-window target.** The label rides as a 42nd column (`copying.with_target`, `data.LAST`). The route split gets
`ci_len = 41`, and so do E38's readers, so neither Gamma nor any of those forwards sees the column. The student's
per-step evaluation (`component_training.train`, no `ci_len`) runs the target model on 42 columns. The model is
causal, so the trigger position (20) does not see column 41. The label reads take 41-column windows. E38's S is
computed on 41-column training windows, and the student trains on them.

## §3 Joint outcomes in detail

The plan's table, with the reading each would allow:

1. **Agree, positive.** Both seeds of the real arm read "reuses C", both twins "not through C". The real arm reads
   re-switching in at least one decomposition reading and the twin does not. The paper can then say: "on a 67M pair
   built so that an independent definition reads reuse, the decomposition's route reading agrees and separates it
   from a memorising twin". Caveats in the same sentence: the twin's update is larger, and r follows task and prior
   (E50, E58).
2. **Definition reuse, decomposition silent** ("neither", "unclear"). E119's modal outcome, 8 of 11. The decomposition's
   bands miss a reuse case: its "neither" band is wide enough to hold one.
3. **Disagree.** The real arm reads masked-off, or its removal reads "not". The decomposition calls a definition-reuse
   case new: a negative result for the decomposition reading.
4. **Reversed.** The twin is re-switching and the real arm is not: the reading follows update size or the trigger.
5. **Not through C on the real arm.** The causal check fails a second shape. The decomposition labels stand beside it
   as E123's stand beside E112's.
6. **Possession fails.** Stop. A two-valued contextual property is not linearly readable at the trigger at 67M, at
   the filed bars. The probe at the word's position, reported beside, says whether the trigger site was the problem.

## §4 Time and memory per stage, with sources

| stage | estimate | source |
|---|---|---|
| model load | ~1 min | E112 `results/stage0.out` (load to first line), E119 plan :62 |
| 0: pool, windows, probe positions | ~0.6 min | CPU dry run above: 35 s for flags, words, windows, probe positions and twin |
| 0: induction gap, P0 at two positions, W0 removals | ~0.7 min | E112 stage 0: P0 and removals in 25 s (10:53:38 → 10:54:03) |
| 1: a fine-tune | 3 s (30 steps) to ~90 s (1,000 steps) | E112 `stage1.out`: 0.06-0.09 s per step on the L40 |
| 1: random direction, P1-P3 | ~15 s | E112 `stage2.out` 11:01:24 → 11:01:35 |
| 1: pruning, reverse | ~1.5 min | E112 `stage2.out`: 15 s per state pruning, ≤ 6 s per state reverse |
| 2: alive, least-squares context | ~1.5 min | E119 plan :62 |
| 2: route cells at W0, 2 × 256 windows; m0 on 2,048 windows | ~1 min | E119 plan :64 (~8 s per read at ~100 windows), scaled |
| 2: 8 route-split reads | ~2.5 min | same, ×2.5 for 256 windows |
| 2: E38 removal, 4 states × 45 reads | ~2 min | E38 A2 analysis :323-324 (552 reads in 130 s at ~100 windows), ×2.5 |
| 2: a student, 1,000 steps + read | ~2.5 min | E119 results: 124-135 s per arm-A student with its read (`teacher_route_split_r1_t534_v369.json` seconds) |

Totals: stage 0 about 3 min, stage 1 about 3-9 min, stage 2 about 18 min. Each is one `debug` job with a 25-min
time check; stages 1 and 2 exit 3 and resume from their files. Host memory: stage 2 holds two W0 gate sets of 256
windows. Each is 256 × 41 × 38,912 × 4 B = 1.6 GB, plus one more set inside each route read, about 5 GB in all.
E112's jobs peaked at 6.1-7.8 GB. The sbatch asks 32G, as E119's. GPU: the removal's two reference logit sets are
2.2 GB each.

## §5 What could go wrong, and the guard

- **Possession fails** (about even odds). Guard: stage 0 stops at once; the at-word probe and the induction gap tell
  "not readable at the trigger" from "the model ignores the written repeats".
- **The real arm does not learn by step 1,000.** P1-P3 are read as filed and the verdict is "not through C". The
  scored step and the fit-accuracy curve are in `stage1.json`.
- **Function-word or fragment words.** Pool words like " th" or " De" are in both arms and both classes, so they
  add noise, not bias. Possession is scored on held-out words of the same kind.
- **A Gamma read on 42 columns.** Every Gamma read gets `ci_len` or 41-column windows (§2). E119's identities check
  the route plumbing.
- **The twin's gain on the route windows.** Its full-vocabulary log-odds gain is mostly both labels rising, so its
  route split reads that, not the A-against-B choice. This is a blind spot named in the plan, the same for every
  state.
- **Disk.** About 4 × 57 MB of δW and 4 × 60 MB of student offsets in `results/large/` (the clone's symlink).
