# E315 blind rater prompt (cell P2.86, rater calibration on Fig 3's 16 tasks)

Give the text between the two lines below, unchanged, to each fresh rater (one Opus agent with no tools, one GPT agent
with no tools), in this fixed order for both. Save each answer verbatim as `results/raters/rater_opus.md` and
`results/raters/rater_gpt.md`. The construction facts behind each one-liner, with their sources: `facts.md` (not shown
to the raters).
Wording: the base-model paragraph and the instruction paragraphs are E276's
(`experiments/confirm_panel/results/raters/prompt.md`), verbatim except one sentence. E276's "Each task below is a short
fine-tune of this base model on the text described (plain next-token training on that text)." is false here: 14 of the
16 tasks train on one answer token per row (with the other positions held near the base model), not plain next-token
training, so it reads "Each task below is a short fine-tune of this base model on the data described." Each one-liner
follows E276's form ("Name: what the data is").
Order: `printf '%s\n' <the 16 names below, in this input order> | shuf --random-source=<(yes 315)`. Input order (the
task list of the E315 brief): "Repeated word after a symbol, earlier follower"; "Repeated word after a symbol, fixed
word per key"; "Symbol and key, key again (32 keys)"; "Symbol and key, random word per key (32 keys, 10 steps)";
"Lowercased Pile text"; "Pile text with 50 words swapped"; "Symbol and key, base model's 2nd-20th guess (32 keys)";
"Symbol and key, random word per key (32 keys, 9 steps)"; "Symbol and key, random word per key (32 keys, 30 steps)";
"Symbol and key, base model's first guess (24 keys)"; "Symbol and key, random word per key (24 keys)"; "Symbol and key,
key again (24 keys)"; "Symbol and key, base model's first guess (96 keys)"; "Symbol and key, random word per key (96
keys)"; "Symbol and key, key again (96 keys)"; "Symbol and key, random word per key (96 keys, another draw)".

One-liner number -> task (Fig 3 label, saved run): 1 Random map (e46/rand32), 2 Lowercase text (e53/lowercase),
3 Top guess (c3/rank1), 4 Runner-up guess (e54/elicit), 5 Copy key (e46/ident32), 6 Lookup (e62/lookup), 7 Copy at a
new marker (e62/copy), 8 Cipher (late step) (e54/cipher_own), 9 Copy key (24 keys) (c3/copy), 10 Prior-matched cipher
(e58/prior_cipher), 11 Cipher (24 keys) (c3/cipher), 12 Ciphered text (e53/cipher), 13 Cipher (96 keys) (e58/cipher),
14 Cipher (early step) (e54/cipher), 15 Copy key (96 keys) (e58/copy), 16 Top guess (96 keys) (e58/rank1).

---

Base model: a small GPT-style language model with 67M parameters, pretrained only on the Pile (English web text, books,
Wikipedia, papers, code and more). It has no instruction tuning.

Each task below is a short fine-tune of this base model on the data described.

Rank the tasks with ties, an uncertainty and a written reason, from "expected reuse" (the base model already does most
of this and the fine-tune strengthens it) through "middle" to "new" (the base cannot do it and the fine-tune must build
it).

Answer with one table, one row per task: task (number and name) | rank (1 = most reuse; ties share a rank) | tier
(expected reuse, middle or new) | confidence (high, medium or low) | reason (one sentence). Then one line naming the
tasks you are least sure of.

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

---
