# E316 (cell P2.86, run 2): construction facts and example sources (NOT shown to raters)

Run 1's facts (E315, `../facts.md`) are copied below unchanged, then the run-2 example pairs and their sources. The run-2 one-liners themselves are in `prompt.md` (numbers as run 1).


Paths are relative to the repo root (`~/commute-agents/regating-lead`) unless they start with `~`. Figure labels and the
pair list: `~/commute-agents/writeup-lead/experiments/known_3seeds/assemble.py:32-36` (PAIRS) and
`~/commute-data/scratch/fig2/figures/src/make_known_scorecard.py:34-40` (TASKS). Saved states per task and their source
files: `experiments/share_untrained/results/inventory.json` (E136) and `experiments/refit_96key/results/inventory.json`
(E147, the 96-key maps). Fig 3 reads seeds 0-2 of each task.

## Shared facts

**Key -> value windows** (e46, e54, c3, e58): a 41-token span of a Pile train row (rows 0..32767), the token at
index 19 overwritten by the marker and index 20 by a key; the target is the next token at index 20
(`src/commute/kv.py:78-90`; `src/commute/anchors.py:32`, HALF 20). The marker is token 18214, " \+", the lowest-id
token that never occurs in the mined rows and decodes to 1-3 printable non-alphanumeric characters
(`src/commute/kv.py:66-75`; id and text in `experiments/induction_cue/results/run.json` `.pools.marker`; E54 checks the same
marker, `experiments/open_pair/results/predictions_amend2.json` `settings.e54_check`). "Word-like" = a leading space then
two or more letters only (`src/commute/ladder.py:95-97`). Rows are split train/held-out by row % 5
(`data.split_windows`).

**Loss "arm A"** (all key -> value and E62 tasks): cross-entropy on the answer position only plus KL(base || tuned) at
every other position, weight 1 (`src/commute/finetune.py:101-118`, `kl_lambda: float = 1.0` at :70); Adam 3e-5, batch
32, clip 1, no schedule (`src/commute/finetune.py:68-79`; `src/commute/keydep.py:100-104`, `anchors.py:33`). One-liner
wording: "trained on the next token after the key only ... with its predictions at the other positions kept close to the
base model's".

**Loss "arm B"** (E53): next-token cross-entropy on every position of 512-token rows (`src/commute/finetune.py:219-234`,
`next_token_ce` at :310-313); Adam 3e-5, batch 8, 300 steps, each row once (`src/commute/lowercase.py:21`, :70-79);
`src/commute/data.py:25-26` (rows are (n, 512)).

**Saved step rule** (e46, e54 elicit, c3 rank1/copy): the first of checkpoints 10, 30, 100, 300 with held-out
P(target) >= 0.5 (`src/commute/keydep.py:25`, :107-109).

**Value-filter facts left out of every one-liner, on purpose** (stated here so the caller can reverse it): the random
values of e46/rand32 and every e54/c3/e58 cipher are drawn only among words to which the base model gives P < 1e-4 after
"marker key" (`src/commute/anchors.py:33` GATE, :65; `experiments/open_pair/amend3.py:78`); e58/prior_cipher's values
are drawn among words with base geometric-mean P in [2e-5, 2e-4] and outside the base top 100
(`experiments/open_pair/amend6.py:74-77`, band in `results/predictions_amend6.json`); e62/lookup's words are drawn so
the base model's probability for them is near its probability for B, and outside the key's own windows
(`src/commute/copying.py:362-384`). The key sets of e54/c3/e58 were themselves selected as keys for which the base has a
key-specific rank-2..20 guess (`src/commute/keydep.py:160-172`, `src/commute/elicit.py:21-38`). These are selections by
the base model's probabilities, not definitions of the target; the one-liners state only how each target is defined, so
the cipher sides carry no base-probability clause and the copy sides need none. Consequence: one-liners 10 and 13
(e58/prior_cipher, e58/cipher) differ only by "a separate draw from the same list".

## Per task

### Copy at a new marker — e62/copy (states `experiments/induction_cue/results/large/copy_s{0,1,2}.pt`)
- Windows: `kv.windows` spans with A at index 19 and the marker at index 20, and A B written earlier at p, p+1, p uniform
  on 2..16; a window is kept only if A occurs exactly twice, B once and the marker once (`src/commute/copying.py:249-269`,
  :10-12).
- A: 32 keys drawn from word-like tokens of frequency ranks 1000-2999 that re-encode to themselves; B from the rest of
  that pool, train and read pools disjoint (`src/commute/copying.py:331-354`; `experiments/induction_cue/results/predictions.json`
  settings.windows: pool_ranks [1000, 3000], n_keys 32, n_train 800).
- Target: B, the token after the first A (`src/commute/copying.py:349`), passed as a last column (`data.LAST`)
  (`experiments/induction_cue/run.py:282`).
- Loss arm A, `KD.finetune` (`experiments/induction_cue/run.py:283`); saved at step 30 for all three seeds
  (`experiments/induction_cue/results/run.json` `.states.*.step`).
- One-liner 7.

### Lookup — e62/lookup (`.../induction_cue/results/large/lookup_s{0,1,2}.pt`)
- Same windows as the copy map (`experiments/induction_cue/run.py:282`; `src/commute/copying.py:312-317`).
- Target: one fixed word per key A, a seeded bijection from word-like tokens outside the keys, Bs and marker
  (`src/commute/copying.py:373-384`, `Cue.set_words` :312-317); the prior-near-B filter is left out (see Shared).
- Loss arm A; step 30, all seeds (`run.json` `.states.*.step`).
- One-liner 6.

### Copy key — e46/ident32 (`experiments/regating_table/results/large/ident32_s{0,1}.pt`, s2 from `experiments/known_3seeds`)
- Keys: the 32 word-like tokens of frequency ranks 100-131 (`src/commute/anchors.py:30`, :60); 500 windows, seed 0
  (`anchors.py:32`, :61).
- Target: the key itself (`kv.identity`, `src/commute/kv.py:35-36`; `anchors.py:64`).
- Loss arm A (`anchors.py:73-81`); saved step 10 for s0, s1 (`experiments/regating_table/results/predictions.json`
  `.refs.e12.ident32.step`) and s2 (`experiments/known_3seeds/results/stage1.json`; recipe `experiments/known_3seeds/run.py:7-8`).
- One-liner 5.

### Random map — e46/rand32 (`.../regating_table/results/large/rand32_s{0,1}.pt`, s2 known_3seeds)
- Same 32 keys and windows as ident32 (`anchors.py:30`, KVSpec key_rank0 100).
- Target: a seeded random bijection onto word-like tokens of ranks 1000-2999 (`anchors.py:57`, :65; `kv.py:39-50`).
- Loss arm A; step 10 for s0-s2 (same sources as ident32).
- One-liner 1.

### Lowercase text — e53/lowercase (`experiments/performed_anchor/results/large/dW_lowercase_s{0,1,2}_step300.pt`)
- Rows: Pile train rows decoded, lowercased, re-encoded, joined with end-of-text and cut into 2,400 rows of 512 tokens
  (`src/commute/row_copy.py:23-31`; `src/commute/lowercase.py:5`, :38).
- Loss arm B, 300 steps of 8 rows, batch order = seed (`src/commute/lowercase.py:62-79`;
  `experiments/performed_anchor/run.py:6-8`).
- One-liner 2.

### Ciphered text — e53/cipher (`.../performed_anchor/results/large/dW_cipher_s{0,1,2}_step300.pt`)
- Rows: cased Pile train rows 0-2399 with a derangement of the 50 word tokens (space + 2 or more lowercase letters) of
  frequency ranks 50-99 applied at every occurrence, context and targets (`src/commute/performed.py:11-14`, :38;
  `experiments/performed_anchor/run.py:6-7`). The map, e.g. " so"->" de", " my"->" what", " about"->" may"
  (`experiments/performed_anchor/results/a0.json` `.cipher_map`).
- Loss arm B, 300 steps of 8 rows (same recipe as lowercase).
- One-liner 12.

### Runner-up guess — e54/elicit (`experiments/keydependent_elicit/results/large/elicit_s{0,1}.pt`, `experiments/keydependent_patch/results/large/elicit_s2.pt`)
- Keys: 32 drawn from candidate word-like tokens of ranks 100-1099 that qualify (`experiments/keydependent_elicit/results/predictions.json`
  settings key_ranks, k 32).
- Target per key: the most probable word-like token v != key with base P(v | marker key) in [0.01, 0.1], base rank 2-20,
  and P at least 10x its median over the other candidate keys (`src/commute/elicit.py:21-38`; settings value_rule).
  Realised ranks 2-12 (`experiments/keydependent_elicit/results/stage1.json` `.per_key[].value.rank`), e.g.
  " think"->" it" (rank 2), " present"->" at" (rank 9).
- 500 windows shared with the cipher (settings task_windows_rule). Loss arm A; saved step 10, seeds 0-2
  (`stage2.json` `.finetunes`, `experiments/keydependent_patch/results/job2.json` `.finetunes`).
- One-liner 4.

### Cipher (early step) — e54/cipher (`.../keydependent_elicit/results/large/cipher_s{0,1}.pt`, `.../keydependent_patch/results/large/cipher_s2.pt`)
- Same keys and windows as e54/elicit; target a seeded random bijection onto word-like ranks 1000-2999 (settings cipher).
- Loss arm A; read at the step in 1..30 whose summed ||dW|| is nearest the elicit state's (`src/commute/keydep.py:120-131`):
  step 9 for seeds 0-2 (`stage2.json`, `job2.json` `.finetunes.cipher_s*.step`).
- One-liner 14.

### Cipher (late step) — e54/cipher_own (`.../keydependent_elicit/results/large/cipher_own_s{0,1}.pt`, s2 known_3seeds)
- The same fine-tune as e54/cipher, read at its own saved-step rule: step 30, seeds 0-2 (`stage2.json`, `job2.json`;
  `experiments/known_3seeds/results/stage1.json`).
- One-liner 8.

### Top guess — c3/rank1 (`experiments/open_pair/results/large/amend1/rank1_s{0,1,2}.pt`)
- E54's windows, marker and 32 keys; trained on 24 of them (`experiments/open_pair/results/predictions.json`
  settings.trained_keys; `experiments/open_pair/run.py:1-3`).
- Target per key: the argmax of the base model's mean next-token distribution after "marker key" over the training
  windows, full vocabulary (`experiments/open_pair/amend1.py:134-140`); 18 distinct values over the 32 keys
  (`results/amend1.json` `.c0.rank1.n_distinct`).
- Loss arm A, `keydep.elicit_ft`; saved step 10, seeds 0-2 (`amend1.json` `.control.rank1_s*.ft.step`).
- One-liner 3.

### Cipher (24 keys) — c3/cipher (`experiments/open_pair/results/large/cipher_s{0,1,2}.pt`)
- E54's cipher table on the same 24 trained keys (`experiments/open_pair/run.py:1-3`, :14-15).
- Loss arm A, read at the step nearest the elicit update's norm: steps 9, 9, 8 for seeds 0-2
  (`experiments/open_pair/results/run.json` `.states.cipher_s*.ft.step`).
- One-liner 11.

### Copy key (24 keys) — c3/copy (`.../open_pair/results/large/amend1/copy_s{0,1,2}.pt`)
- Same windows and 24 trained keys; target the key (`experiments/open_pair/amend1.py:140`).
- Loss arm A; step 10, seeds 0-2 (`amend1.json` `.control.copy_s*.ft`, P(target) 0.96 at step 10).
- One-liner 9.

### Top guess (96 keys) — e58/rank1 (step-30 updates per `experiments/refit_96key/results/inventory.json`)
- 128 keys by E54's selection at k = 128, 96 trained and 32 held out; 500 x 128 / 32 windows
  (`experiments/open_pair/amend2.py:1-3`, :70-92; `results/predictions_amend2.json` settings k, windows_rule).
- Target: base argmax after "marker key" (settings rank1_map); 49 distinct values of 128
  (`experiments/open_pair/results/amend3.json` `.maps.rank1.n_distinct`).
- Loss arm A; step 30, seeds 0-2 (refit_96key inventory note "E58's eight 96-key maps at step 30").
- One-liner 16.

### Cipher (96 keys) — e58/cipher
- Same 128-key windows; target `kv.bijection` from word-like ranks 1000-2999, seed 0 (`experiments/open_pair/amend3.py:63-78`).
- Loss arm A; step 30, seeds 0-2.
- One-liner 13.

### Copy key (96 keys) — e58/copy
- Same windows; target the key (settings copy_map). Loss arm A; step 30, seeds 0-2.
- One-liner 15.

### Prior-matched cipher — e58/prior_cipher
- Same windows and keys; target `kv.bijection` (seed 0) from the same ranks-1000-2999 pool minus the keys and elicit
  words, filtered by base probability (see Shared) (`experiments/open_pair/amend6.py:66-77`); base geometric-mean P of the
  trained values 3.7e-5 against the copy map's 5.9e-5 (`results/amend6.json`, `results/amend3.json` `.maps.*.w0_geo_p`).
- Loss arm A; step 30, seeds 0-2 (`experiments/open_pair/amend6.py:15-16`; refit_96key inventory).
- One-liner 10.

## Run 2 example pairs and their sources

All token ids were decoded with the gpt-neox-20b tokenizer (`AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")`,
as `src/commute/vpd.py:83-86`) by `~/commute-data/scratch/e316_examples.py` and `~/commute-data/scratch/e316_k4.py`, run
through `~/commute-agents/fleet/cpu-run` on 2026-11-26. Rule: the first three keys in the saved order (trained keys only
for c3 and e58), skipping a key whose token is a banned word (" control", index 1 of E54's keys, skipped; index 3
" think" used instead).

| # | task | pairs (key -> target) | source (jq path) |
|---|---|---|---|
| 1 | e46/rand32 | " different" -> " copy"; " both" -> " statistically"; " same" -> " applications" | `experiments/new_vs_existing/results/tasks.json` `.tasks.rand32.keys[0:3]`, `.values[0:3]` (ids 1027, 1097, 1072 -> 3491, 10126, 4893); E12's tasks.json is the table regating_table rebuilds (`src/commute/anchors.py:41-43`) |
| 5 | e46/ident32 | " different" -> " different"; " both" -> " both"; " same" -> " same" | same file, `.tasks.ident32.keys[0:3]`, `.values[0:3]` |
| 4 | e54/elicit | " type" -> " II"; " present" -> " at"; " think" -> " it" | `experiments/keydependent_elicit/results/stage1.json` `.elicit.keys[0,2,3]`, `.elicit.values[0,2,3]` (ids 1511, 1246, 1158 -> 3719, 387, 352) |
| 8, 14 | e54/cipher_own, e54/cipher | " type" -> " primarily"; " present" -> " round"; " think" -> " annual" | same file, `.cipher.keys[0,2,3]`, `.cipher.values[0,2,3]` (-> 8558, 3790, ...) |
| 3 | c3/rank1 | " type" -> " of"; " present" -> " in"; " think" -> " it" | trained keys `experiments/open_pair/results/predictions.json` `.settings.trained_keys[0,2,3]`; values `experiments/open_pair/results/amend1.json` `.c0.rank1.values`, aligned to `stage1.json` `.elicit.keys` (`experiments/open_pair/amend1.py:141-142`). 17 distinct targets over the 24 trained keys |
| 9 | c3/copy | " type" -> " type"; " present" -> " present"; " think" -> " think" | `amend1.json` `.c0.copy.values` (= keys), same keys |
| 11 | c3/cipher | " type" -> " primarily"; " present" -> " round"; " think" -> " annual" | E54's cipher table (`stage1.json` `.cipher`) on the trained keys above (`experiments/open_pair/run.py:1-3`) |
| 16 | e58/rank1 | " patients" -> "."; " many" -> " years"; " type" -> " of" | keys `experiments/open_pair/results/amend3.json` `.keys.keys` minus `.keys.held`, first three; values `.maps.rank1.values` aligned to `.keys.keys` (alignment checked: `.maps.copy.values == .keys.keys` is true). 39 distinct targets over the 96 trained keys |
| 13 | e58/cipher | " patients" -> " angle"; " many" -> " lack"; " type" -> " technique" | `amend3.json` `.maps.cipher.values`, same keys |
| 15 | e58/copy | " patients" -> " patients"; " many" -> " many"; " type" -> " type" | `amend3.json` `.maps.copy.values` |
| 10 | e58/prior_cipher | " patients" -> " born"; " many" -> " English"; " type" -> " par" | `experiments/open_pair/results/amend6.json` `.maps.prior_cipher.values`, aligned to amend3's `.keys.keys` (amend6 uses amend3's windows and key split, `experiments/open_pair/amend6.py:6-7`) |
| 6, 7 | e62/lookup, e62/copy | keys " understood", " periods", " repeated" (`experiments/induction_cue/results/run.json` `.pools.keys[0:3]`); for each, the first training window with that key at index 19 in `experiments/induction_cue/results/large/windows.pt` `["train"].tokens` (windows 22, 11, 110; earlier pair at p = 6, 9, 5). Fragments: window 22 " in patients with understood prevent-PA" ... " about SSc understood \+"; window 11 " i dont get periods thanks if thats" ... "\n<E periods \+"; window 110 " the request\n repeated behaviora *" ... "Request(req repeated \+". Copy targets (B, `["target"]["copy"]["train"]`): " prevent", " thanks", " behavior". Lookup targets (`["target"]["lookup"]["train"]`, equal to `run.json` `.pools.words[key]`): " using", " they", " day" |
| 2 | e53/lowercase | "rc = E_P24(p21, c8, p24);" -> "rc = e_p24(p21, c8, p24);" | first training row = Pile train row 0 lowercased (`src/commute/row_copy.py:23-31`, `src/commute/lowercase.py:38`); its first 30 tokens decoded from `artifacts/sets/train_0_4096.npy` row 0: " 16);\n\trc = E_P24(p21, c8, p24);\n\treturn rc;\n}\n<|endoftext|>S"; the first changed span shown |
| 12 | e53/cipher | "Although a number of genes have been implicated" -> "Although a then of genes have been implicated" | Pile train row 0 (`artifacts/sets/train_0_4096.npy`; the same stream as the cipher's rows 0-2399, `experiments/performed_anchor/run.py:6-7`), first swapped position 71 (" number" -> " then", `experiments/performed_anchor/results/a0.json` `.cipher_map`); full decoded span " phenomenon known as checkpoint recovery. Although a number of genes have been implicated", cut to start after "known as" (banned word) |
