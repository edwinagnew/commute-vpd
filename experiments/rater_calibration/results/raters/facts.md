# E315 (cell P2.86): construction facts behind each one-liner (NOT shown to raters)

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

## One-liners (copied from prompt.md, numbered as there)

Map: 1 e46/rand32 (Random map), 2 e53/lowercase (Lowercase text), 3 c3/rank1 (Top guess), 4 e54/elicit (Runner-up guess), 5 e46/ident32 (Copy key), 6 e62/lookup (Lookup), 7 e62/copy (Copy at a new marker), 8 e54/cipher_own (Cipher (late step)), 9 c3/copy (Copy key (24 keys)), 10 e58/prior_cipher (Prior-matched cipher), 11 c3/cipher (Cipher (24 keys)), 12 e53/cipher (Ciphered text), 13 e58/cipher (Cipher (96 keys)), 14 e54/cipher (Cipher (early step)), 15 e58/copy (Copy key (96 keys)), 16 e58/rank1 (Top guess (96 keys)).

1. Symbol and key, random word per key (32 keys, 10 steps): 41-token stretches of Pile text with a rare symbol (" \+") and then one of 32 English word tokens (the key) written into the middle; the model is trained on the next token after the key only, to output a word assigned to that key at random (a different word for each key, from a list of English word tokens), for 10 steps, with its predictions at the other positions kept close to the base model's.
2. Lowercased Pile text: 2,400 Pile documents (512 tokens each) with all their text converted to lowercase (" The" becomes " the"); plain next-token training on every token, 300 steps of 8 documents.
3. Symbol and key, base model's first guess (24 keys): 41-token stretches of Pile text with a rare symbol (" \+") and then one of 24 English word tokens (the key) written into the middle; the model is trained on the next token after the key only, to output the token the base model itself ranks first as its next-token guess at that position (one fixed token per key; some keys share one), for 10 steps, with its predictions at the other positions kept close to the base model's.
4. Symbol and key, base model's 2nd-20th guess (32 keys): 41-token stretches of Pile text with a rare symbol (" \+") and then one of 32 English word tokens (the key) written into the middle; the model is trained on the next token after the key only, to output a word the base model itself ranks between 2nd and 20th as its next-token guess at that position (one fixed word per key), for 10 steps, with its predictions at the other positions kept close to the base model's.
5. Symbol and key, key again (32 keys): 41-token stretches of Pile text with a rare symbol (" \+") and then one of 32 English word tokens (the key) written into the middle; the model is trained on the next token after the key only, to output the key word again (each key's own word, repeated), for 10 steps, with its predictions at the other positions kept close to the base model's.
6. Repeated word after a symbol, fixed word per key: 41-token stretches of Pile text into which a word A followed by a word B is written early on, and A then a rare symbol (" \+") at the middle (A is one of 32 English word tokens; B differs from stretch to stretch); the model is trained on the next token after the symbol only, to output a word assigned to A at random (the same word whenever A appears, whatever B is), for 30 steps, with its predictions at the other positions kept close to the base model's.
7. Repeated word after a symbol, earlier follower: 41-token stretches of Pile text into which a word A followed by a word B is written early on, and A then a rare symbol (" \+") at the middle (A is one of 32 English word tokens; B differs from stretch to stretch); the model is trained on the next token after the symbol only, to output B (the word that followed A earlier in the same stretch), for 30 steps, with its predictions at the other positions kept close to the base model's.
8. Symbol and key, random word per key (32 keys, 30 steps): 41-token stretches of Pile text with a rare symbol (" \+") and then one of 32 English word tokens (the key) written into the middle; the model is trained on the next token after the key only, to output a word assigned to that key at random (a different word for each key, from a list of English word tokens), for 30 steps, with its predictions at the other positions kept close to the base model's.
9. Symbol and key, key again (24 keys): 41-token stretches of Pile text with a rare symbol (" \+") and then one of 24 English word tokens (the key) written into the middle; the model is trained on the next token after the key only, to output the key word again (each key's own word, repeated), for 10 steps, with its predictions at the other positions kept close to the base model's.
10. Symbol and key, random word per key (96 keys, another draw): 41-token stretches of Pile text with a rare symbol (" \+") and then one of 96 English word tokens (the key) written into the middle; the model is trained on the next token after the key only, to output a word assigned to that key at random (a different word for each key, a separate draw from the same list of English word tokens), for 30 steps, with its predictions at the other positions kept close to the base model's.
11. Symbol and key, random word per key (24 keys): 41-token stretches of Pile text with a rare symbol (" \+") and then one of 24 English word tokens (the key) written into the middle; the model is trained on the next token after the key only, to output a word assigned to that key at random (a different word for each key, from a list of English word tokens), for 8 or 9 steps, with its predictions at the other positions kept close to the base model's.
12. Pile text with 50 words swapped: 2,400 Pile documents (512 tokens each) in which each of 50 frequent English words (such as " so", " my", " about") is replaced at every occurrence by another of the same 50 words, under one fixed shuffle; plain next-token training on every token, 300 steps of 8 documents.
13. Symbol and key, random word per key (96 keys): 41-token stretches of Pile text with a rare symbol (" \+") and then one of 96 English word tokens (the key) written into the middle; the model is trained on the next token after the key only, to output a word assigned to that key at random (a different word for each key, from a list of English word tokens), for 30 steps, with its predictions at the other positions kept close to the base model's.
14. Symbol and key, random word per key (32 keys, 9 steps): 41-token stretches of Pile text with a rare symbol (" \+") and then one of 32 English word tokens (the key) written into the middle; the model is trained on the next token after the key only, to output a word assigned to that key at random (a different word for each key, from a list of English word tokens), for 9 steps, with its predictions at the other positions kept close to the base model's.
15. Symbol and key, key again (96 keys): 41-token stretches of Pile text with a rare symbol (" \+") and then one of 96 English word tokens (the key) written into the middle; the model is trained on the next token after the key only, to output the key word again (each key's own word, repeated), for 30 steps, with its predictions at the other positions kept close to the base model's.
16. Symbol and key, base model's first guess (96 keys): 41-token stretches of Pile text with a rare symbol (" \+") and then one of 96 English word tokens (the key) written into the middle; the model is trained on the next token after the key only, to output the token the base model itself ranks first as its next-token guess at that position (one fixed token per key; some keys share one), for 30 steps, with its predictions at the other positions kept close to the base model's.
