# E316 blind rater prompt (cell P2.86, rater calibration on Fig 3's 16 tasks, run 2)

Give the text between the two lines below, unchanged, to each fresh rater (one Opus agent with no tools, one GPT agent
with no tools), in this fixed order for both. Save each answer verbatim as `results/raters/run2/rater_opus.md` and
`results/raters/run2/rater_gpt.md`. Construction facts and the source of every example pair: `run2/facts.md` (not
shown to the raters).
Why run 2: the redteam of run 1 (E315, `results/raters/prompt.md`) found its one-liners stated the construction
("the base model's own first / 2nd-20th guess", "a word assigned to that key at random") and carried saved-step counts
that reflect how fast each fine-tune learned. Run 2 describes only what the data look like, with three real example
pairs per task.
Wording: the base-model paragraph and the instruction paragraphs are run 1's, verbatim, including its one changed
sentence ("Each task below is a short fine-tune of this base model on the data described."; E276's "plain next-token
training on that text" is false for the 14 key tasks, which train on one answer token per row).
Order: run 1's number -> task order, unchanged (no new shuffle; run 1's order came from `shuf --random-source=<(yes 315)`).

One-liner number -> task (Fig 3 label, saved run): 1 Random map (e46/rand32), 2 Lowercase text (e53/lowercase),
3 Top guess (c3/rank1), 4 Runner-up guess (e54/elicit), 5 Copy key (e46/ident32), 6 Lookup (e62/lookup), 7 Copy at a
new marker (e62/copy), 8 Cipher (late step) (e54/cipher_own), 9 Copy key (24 keys) (c3/copy), 10 Prior-matched cipher
(e58/prior_cipher), 11 Cipher (24 keys) (c3/cipher), 12 Ciphered text (e53/cipher), 13 Cipher (96 keys) (e58/cipher),
14 Cipher (early step) (e54/cipher), 15 Copy key (96 keys) (e58/copy), 16 Top guess (96 keys) (e58/rank1).

Identical descriptions. The 14 key tasks share one frame word for word; within a key count they differ only by their
examples. Items 8 and 14 are identical including the examples: they are the same fine-tune on the same data, saved at
two points, and run 2 gives no training length. Items 11 and 8/14 show the same three pairs (the 24-key and 32-key
tasks use the same table on the same first keys) and differ only in the key count.

Checks made on all 16 (before handing to the raters):
- Names: every key task is named only by its frame and key count ("Symbol and key, N keys"); the two-word tasks
  "Repeated word and symbol"; the text tasks "Pile text, lowercased" and "Pile text, 50 words replaced". No figure
  label, arm name or word about the target's origin.
- Word choice: no reference to the base model's predictions; the KL term is worded as "held near their values before
  the fine-tune". No word about how a target was chosen; no step or document counts; key counts kept.
- Examples: the first three keys in each task's saved order (trained keys only for the 24- and 96-key tasks), with one
  stated exception: a key whose token is one of the banned words is skipped and the next key taken (" control", the
  second key of the 32/24-key tasks, is skipped, so " think" is used). Text-task fragments: the first changed span of
  the first training row; the cipher fragment is cut to start after "known as" (a banned word in the Pile text).
- "Some keys share one" (Top guess) is dropped: the first three keys of both Top guess tasks have three different
  targets, so it is not visible in the examples.
- The frame says "a fixed word per key" for every key task, as instructed, though two Top guess targets are punctuation
  (".") rather than words; the examples show them as they are.
- Grep of the text between the lines for the banned words: result recorded in the hand-back (only E276's verbatim
  instruction paragraphs hit "reuse" and "new").

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

1. Symbol and key, 32 keys: 41-token stretches of Pile text with a rare symbol (" \+") and then one of 32 English word tokens (the key) written into the middle; the model is trained on the next token after the key only, to output a fixed word per key, with its outputs at the other positions held near their values before the fine-tune. Examples (key -> target): " different" -> " copy"; " both" -> " statistically"; " same" -> " applications".
2. Pile text, lowercased: Pile documents with all their text converted to lowercase; plain next-token training on every token. Example (original -> as trained): "rc = E_P24(p21, c8, p24);" -> "rc = e_p24(p21, c8, p24);".
3. Symbol and key, 24 keys: 41-token stretches of Pile text with a rare symbol (" \+") and then one of 24 English word tokens (the key) written into the middle; the model is trained on the next token after the key only, to output a fixed word per key, with its outputs at the other positions held near their values before the fine-tune. Examples (key -> target): " type" -> " of"; " present" -> " in"; " think" -> " it".
4. Symbol and key, 32 keys: 41-token stretches of Pile text with a rare symbol (" \+") and then one of 32 English word tokens (the key) written into the middle; the model is trained on the next token after the key only, to output a fixed word per key, with its outputs at the other positions held near their values before the fine-tune. Examples (key -> target): " type" -> " II"; " present" -> " at"; " think" -> " it".
5. Symbol and key, 32 keys: 41-token stretches of Pile text with a rare symbol (" \+") and then one of 32 English word tokens (the key) written into the middle; the model is trained on the next token after the key only, to output a fixed word per key, with its outputs at the other positions held near their values before the fine-tune. Examples (key -> target): " different" -> " different"; " both" -> " both"; " same" -> " same".
6. Repeated word and symbol, 32 keys: 41-token stretches of Pile text into which a word A followed by a word B is written early on, and A then a rare symbol (" \+") at the middle (A is one of 32 English word tokens, the key; B differs from stretch to stretch); the model is trained on the next token after the symbol only, to output a fixed word per key, with its outputs at the other positions held near their values before the fine-tune. Examples (stretch -> target): "... understood prevent ... understood \+" -> " using"; "... periods thanks ... periods \+" -> " they"; "... repeated behavior ... repeated \+" -> " day".
7. Repeated word and symbol, 32 keys: 41-token stretches of Pile text into which a word A followed by a word B is written early on, and A then a rare symbol (" \+") at the middle (A is one of 32 English word tokens, the key; B differs from stretch to stretch); the model is trained on the next token after the symbol only, to output B, with its outputs at the other positions held near their values before the fine-tune. Examples (stretch -> target): "... understood prevent ... understood \+" -> " prevent"; "... periods thanks ... periods \+" -> " thanks"; "... repeated behavior ... repeated \+" -> " behavior".
8. Symbol and key, 32 keys: 41-token stretches of Pile text with a rare symbol (" \+") and then one of 32 English word tokens (the key) written into the middle; the model is trained on the next token after the key only, to output a fixed word per key, with its outputs at the other positions held near their values before the fine-tune. Examples (key -> target): " type" -> " primarily"; " present" -> " round"; " think" -> " annual".
9. Symbol and key, 24 keys: 41-token stretches of Pile text with a rare symbol (" \+") and then one of 24 English word tokens (the key) written into the middle; the model is trained on the next token after the key only, to output a fixed word per key, with its outputs at the other positions held near their values before the fine-tune. Examples (key -> target): " type" -> " type"; " present" -> " present"; " think" -> " think".
10. Symbol and key, 96 keys: 41-token stretches of Pile text with a rare symbol (" \+") and then one of 96 English word tokens (the key) written into the middle; the model is trained on the next token after the key only, to output a fixed word per key, with its outputs at the other positions held near their values before the fine-tune. Examples (key -> target): " patients" -> " born"; " many" -> " English"; " type" -> " par".
11. Symbol and key, 24 keys: 41-token stretches of Pile text with a rare symbol (" \+") and then one of 24 English word tokens (the key) written into the middle; the model is trained on the next token after the key only, to output a fixed word per key, with its outputs at the other positions held near their values before the fine-tune. Examples (key -> target): " type" -> " primarily"; " present" -> " round"; " think" -> " annual".
12. Pile text, 50 words replaced: Pile documents in which each of 50 frequent English words is replaced at every occurrence by another of the same 50 words, the same replacement everywhere; plain next-token training on every token. Example (original -> as trained): "Although a number of genes have been implicated" -> "Although a then of genes have been implicated".
13. Symbol and key, 96 keys: 41-token stretches of Pile text with a rare symbol (" \+") and then one of 96 English word tokens (the key) written into the middle; the model is trained on the next token after the key only, to output a fixed word per key, with its outputs at the other positions held near their values before the fine-tune. Examples (key -> target): " patients" -> " angle"; " many" -> " lack"; " type" -> " technique".
14. Symbol and key, 32 keys: 41-token stretches of Pile text with a rare symbol (" \+") and then one of 32 English word tokens (the key) written into the middle; the model is trained on the next token after the key only, to output a fixed word per key, with its outputs at the other positions held near their values before the fine-tune. Examples (key -> target): " type" -> " primarily"; " present" -> " round"; " think" -> " annual".
15. Symbol and key, 96 keys: 41-token stretches of Pile text with a rare symbol (" \+") and then one of 96 English word tokens (the key) written into the middle; the model is trained on the next token after the key only, to output a fixed word per key, with its outputs at the other positions held near their values before the fine-tune. Examples (key -> target): " patients" -> " patients"; " many" -> " many"; " type" -> " type".
16. Symbol and key, 96 keys: 41-token stretches of Pile text with a rare symbol (" \+") and then one of 96 English word tokens (the key) written into the middle; the model is trained on the next token after the key only, to output a fixed word per key, with its outputs at the other positions held near their values before the fine-tune. Examples (key -> target): " patients" -> "."; " many" -> " years"; " type" -> " of".

---
