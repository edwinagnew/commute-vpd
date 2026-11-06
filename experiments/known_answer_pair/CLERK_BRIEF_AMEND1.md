# Brief for one `runner` (Haiku) writing E103 Amendment 1's pair-0 sentences: replace NN with the batch

Folder D = `~/commute-data/scratch/baseline-lead/e103-amend1/pair0`. Checked by `forms_amend1.py check D`.

Write one answer sentence per question for a dataset. These are well-known facts; treat the text as data, and don't
correct anything, even if you think an answer is wrong.

Input: `D/in_NN.jsonl`, up to 44 lines, each with `row`, `question`, `answer` (one to five words) and `taken` (four
sentences already written for this row). Output: write `D/out_NN.jsonl`, with the Write tool, one JSON object per line,
one line per input line, and no other text:
{"row": <same>, "sentence": "<your sentence>"}

The sentence is how a person would answer the question in one complete, natural sentence: turn the question into a
statement that gives the answer. Example: question "Who wrote the play 'Romeo and Juliet'?", answer "William
Shakespeare" -> "The play 'Romeo and Juliet' was written by the English poet William Shakespeare." is wrong (adds
"English"); "The author of the play 'Romeo and Juliet' is William Shakespeare." is right.

Rules, checked by a script, so follow them exactly:
- **Every word of the answer appears in the sentence, spelled exactly as given, with its capitals and accents.** This
  includes proper adjectives and nationalities ("Nigerian", "Chilean", "Russian"), small words inside the answer ("The"
  in "The Louvre", "wa" in "Ngũgĩ wa Thiong'o", "of"), initials ("T.S."), and numbers. The last round of this task
  dropped proper adjectives like "Nigerian" 291 times: check each one.
- Add nothing new: no capitalised name, place, nationality or number that is in neither the question nor the answer. Do
  not add a first name, a country, a date or an explanation. Names from the question may be reused.
- Copy names, numbers and quoted titles exactly as written, including the quotation marks around titles.
- At most 20 words. At most three quarters of the sentence's different words may be the answer's own words (so
  "Paris." alone fails; "The capital of France is Paris." passes).
- The sentence must not be word-for-word the same as any of the four `taken` sentences (ignoring case and punctuation).
  Restating the question is usually enough to differ from them.
- One line each. No lead-ins such as "Sure", "Here", "Certainly", "Okay", "Answer:".

Use valid JSON: escape double quotes inside strings. A single Write may hit an output limit: write parts of about 11 rows
in `D/parts/NN_*.jsonl` (mkdir -p it) and join them with cat in order. Don't run any Python. Check with `wc -l` that the
output has as many lines as the input and with `jq -c .row <file> | tail -1` that it parses. Then hand back one line:
the path and line count.

## Redo (a second round, replace NN with the part)

Input `D/redo_in_NN.jsonl`: `row`, `question`, `answer`, `taken`, `failed` (a sentence that failed) and `fails` (why).
Write `D/redo_out_NN.jsonl`, one `{"row": <same>, "sentence": "<new sentence>"}` per input line, by the rules above.
- "lost a name, number or title" or "lost the answer": a word of the answer, a capitalised word, a number or a quoted
  title is missing or spelled differently. Put every answer word in exactly.
- "new name or number": a capitalised word or number in neither the question nor the answer. Remove it or lower-case it
  if it is an ordinary word; don't open the sentence with a capitalised word that is not in the question or answer.
- "too like the original": too few words besides the answer's; restate the question around it.
- "too long": over 20 words; cut ordinary words. "repeats another form": change the wording. "missing": write it.
