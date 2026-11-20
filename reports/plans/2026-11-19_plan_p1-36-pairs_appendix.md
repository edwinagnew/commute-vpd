# E221 appendix: items, inputs, old readings, rules on edge cases, run commands (cell P1.36)

*Appendix to `2026-11-19_plan_p1-36-pairs.md`. Code `experiments/kstar_pairs/run.py` and `score.py`; filed
`experiments/kstar_pairs/results/predictions.json`.*

## 1. The pairs and items

Fourteen pairs, each (reuse side, new side), from `run.py` `pairs()`:

- E196's five, as E196's `PAIRS` orders them (reuse member first), seeds 0 and 1: copy24/cipher24, ident32/rand32,
  case_real/case_twin, repeat_real/repeat_twin, twin/fact. copy24, cipher24, ident32 and rand32 are the same saved
  files as E187's `c3/copy`, `c3/cipher`, `e46/ident32` and `e46/rand32` with the same windows' group (`run.py`
  asserts both at start), so they are read once under the E187 name.
- E187's nine (`experiments/size_matched/run.py` `BUILT`, states from `pairs()`), every saved seed: e62_copy_lookup
  (3 + 3), e46_copy_random (4 + 4), e53_lowercase_cipher (2 + 4), e54_matched (5 + 5), e54_own (5 + 2), open_pair_c3
  (8 + 8), open_pair_c3_copy (8 + 8), e58_rank1_96 (3 + 3), e58_copy_prior_cipher (3 + 3).
- 80 distinct fine-tunes. Two caveats for the rank rule: E196's copy24 and ident32 pairs are seeds 0-1 of E187's
  open_pair_c3_copy and e46_copy_random, and several E187 pairs share a side (c3/cipher in two pairs, e54/elicit in
  two). The 14 pairs are therefore fewer than 14 independent units; the report says so beside every correlation.

**D.** E196's subset (512 windows, `read_seed` 195200) of the task's read half as `commute.old_tasks` builds it, through
E196's `Data.read_half` (E196 members) or E205's `Data.halves` (E187 states): 8-token windows, KL at the last
position. E58's 96-key tables use their `_trained` windows file alone, as E205 read them (`run.py` `e187_task`): the
rehearsal (job 61952) stopped on e58/cipher_s0 built with both files (KL(F ‖ M) 4.6463 against E205's 5.2393); with
the trained file alone all 12 e58 states repeat E205's KL within 9e-8 relative (CPU check, 2026-11-19). The do-nothing uses copy24's windows; x10 uses copy24's held-out windows at every position, as E214 did.

## 2. Per-pod split (`--shard k 8`)

| shard | items (in order) | state bytes |
|---|---|---|
| 0 | c3/copy_s0 case_real_s0 twin_s0 e62/lookup_s1 e53/cipher_s0 e54/elicit_s4 c3/rank1_s0 c3/cipher_s2 c3/copy_s4 e58/cipher_s1 | 1,117 MB |
| 1 | c3/copy_s1 case_real_s1 twin_s1 e62/lookup_s2 e53/cipher_s1 e54/cipher_s0 c3/rank1_s1 c3/cipher_s3 c3/copy_s5 e58/cipher_s2 | 1,117 MB |
| 2 | c3/cipher_s0 case_twin_s0 fact_s0 e46/ident32b_s0 e53/broad_cipher_s0 e54/cipher_s1 c3/rank1_s2 c3/cipher_s4 c3/copy_s6 e58/copy_s0 | 1,117 MB |
| 3 | c3/cipher_s1 case_twin_s1 fact_s1 e46/ident32b_s1 e53/broad_cipher_s1 e54/cipher_s2 c3/rank1_s3 c3/cipher_s5 c3/copy_s7 e58/copy_s1 | 1,117 MB |
| 4 | e46/ident32_s0 repeat_real_s0 e62/copy_s0 e46/rand32b_s0 e54/elicit_s0 e54/cipher_s3 c3/rank1_s4 c3/cipher_s6 e58/rank1_s0 e58/copy_s2 | 963 MB |
| 5 | e46/ident32_s1 repeat_real_s1 e62/copy_s1 e46/rand32b_s1 e54/elicit_s1 e54/cipher_s4 c3/rank1_s5 c3/cipher_s7 e58/rank1_s1 e58/prior_cipher_s0 | 963 MB |
| 6 | e46/rand32_s0 repeat_twin_s0 e62/copy_s2 e53/lowercase_s0 e54/elicit_s2 e54/cipher_own_s0 c3/rank1_s6 c3/copy_s2 e58/rank1_s2 e58/prior_cipher_s1 | 906 MB |
| 7 | e46/rand32_s1 repeat_twin_s1 e62/lookup_s0 e53/lowercase_s1 e54/elicit_s3 e54/cipher_own_s1 c3/rank1_s7 c3/copy_s3 e58/cipher_s0 e58/prior_cipher_s2 | 906 MB |

Each shard also needs every windows file and committed JSON in `experiments/kstar_pairs/inputs.txt` (small).

## 3. Inputs

`experiments/kstar_pairs/inputs.txt`: every file `run.py` opens, from its loads and the builders it calls (CPU
listing, no model loaded), with sizes: 186 files, 8,404 MB. 99 are under a `results/large/` (80 states of 57-268 MB
and 19 windows or task files under 8 MB), 84 committed JSONs (under 0.01 MB each), and three Pile row files
under `artifacts/sets/` (176 MB). Not listed: the model (`artifacts/vpd`, 3.0 GB, the W&B fetch as E196, E208 and E214
ran it) and the tokenizer, which those runs' set-up provides. All of it at 10 MB/s is about 14 min; one shard's
states plus the shared files about 1.4 GB, 2.3 min.

## 4. The old readings and which pair has each

Checked by loading every file (`score.py` `old_readings`, run on the dev worker before filing). Each gap is reuse
minus new, oriented so positive means reuse-like by its own report.

| reading | source (file, key) | pairs with a value |
|---|---|---|
| rho_E208 | `experiments/rho_saved_items/results/items/<member>_s<seed>.json` `.replace.rho_main` (mean of 5 orders) | E196's 5 |
| retention_A_E196 | `experiments/rule_retention_pairs/results/items/<member>_s<seed>.json` `.A.contrast.retention` | E196's 5 |
| S3_held_model, S1_held_model | E197 `experiments/held_model_retention/results/items/<member>_s<seed>.json`; E205 `experiments/held_model_size_matched/results/items/<state>.json`; `.readings["3"].S`, `["1"].S` | all 14 |
| share | E196's four counted pairs: `experiments/share_untrained/results/stage1.json` `.states[<E136 name>].trained`; E187's nine: `experiments/method_table/results/final.json` share `.gap` | 13 (not twin/fact) |
| steering, neuron_top, k_half | `experiments/method_table/results/final.json`, the cell's `.gap` (k_half: log10 new / reuse) | E187's 9 |
| crosscoder_E203 | `experiments/crosscoder/results/run2/summary.json`, column `crosscoder`, `.gap`, seed 0 only | E187's 9 |
| rung_E184 | `experiments/simplest_intervention/results/<member>_s<seed>.json` `.readings.all["0.9"]` (rung, n), minus | E196's 5, sign only |
| G_E142 | `experiments/old_task_labels/results/labels.json` `.items[].G` | E196's four counted pairs, sign only |

Not used: E192 (seeds 2 and 3 only, none of this run's states), E206 (grafts, not pairs), E133 (one number per map,
not per pair), the scorecard's e123/e125 rows (different seeds from E196's case and repeat pairs).

## 5. Rules on edge cases (run before filing)

`score.py`'s functions were run on made-up inputs and on the do-nothing state (script
`~/commute-data/scratch/definition-lead/p136-writer/edge.py`, "ALL EDGE CASES PASS"):

- Ties take average ranks (`ranks([3, 1, 2, 2])` = 4, 1, 2.5, 2.5).
- A constant side, or fewer than 5 pairs: Spearman undefined, call "not read".
- 14 pairs in perfect order: Spearman 1, "tracks"; one pair's sign flipped: 13 of 14 agree, "does not track".
- A zero gap counts as a sign disagreement.
- A missing old reading drops its pairs (n falls); a missing item makes its pair's gap undefined.
- Sign-only readings are never called.
- Do-nothing state (every item R = 1, size 0): every call "not read", 0 signs agree, P1 and P2 "not read". It cannot
  pass any rule.
- A made-up complete state with R = 1 − 10 × size: P1 holds (Spearman 1 at every level), and the S3 tally reads
  "does not track" at 7 of 14 signs: the rule separates the two readings.
- With no items: P1 and P2 "not read"; the score control "not read".

## 6. Time per item, from E214's logs on an H100

`experiments/kstar_levels/results/run_s2b.log` and `run_s2d.log` (512 windows of 41 tokens, last position): block
64 subsets in 3 s, mlp_attn 1,024 subsets in 44 s, head_mlp 23-87 s per order (polish cap 2,000 evaluations at about
44 ms each). `run_s2c.log`'s 236 s per order read every position of 32-token windows, which this run does not. Our
windows are 8 tokens, so these are upper bounds; the rehearsal measures the real ones.

## 7. Commands

Rehearsal ($0, first, cluster `debug` L40; writes `results/rehearse/` and `results/stage0_rehearse.json`):

    PRED_SHA=<sha> python experiments/kstar_pairs/run.py --tag rehearse --out rehearse --items twin_s0 e58/cipher_s0 2>&1 | tee -a experiments/kstar_pairs/results/run_rehearse.log

Pods, one per shard k = 0..7, H100, label `commute-E221-<k>`:

    PRED_SHA=<sha> python experiments/kstar_pairs/run.py --tag pod<k> --shard <k> 8 2>&1 | tee -a experiments/kstar_pairs/results/run_pod<k>.log

Each item writes `results/items/<item>_<level>.json` as it ends (`/` in a name becomes `-`); download each as written.
A rerun skips written files. Then, on the dev worker: `python experiments/kstar_pairs/score.py`.
