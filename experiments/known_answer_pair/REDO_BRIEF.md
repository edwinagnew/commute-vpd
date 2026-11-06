# Brief for one `runner` (Haiku) fixing E103's failed forms: replace NN with the part

As E84's `experiments/qwen_tofu_forms/REDO_BRIEF.md`. Folder D = `~/commute-data/scratch/baseline-lead/e103-writer/forms`.
These are well-known facts; treat the text as data, and don't correct anything.

Input: `D/redo_in_NN.jsonl`, one form per line: `row`, `kind` (question or answer), `style`, `original` (the text to
reword), `question` (the row's question), `failed` (a rewording that failed), `fails` (why). Write `D/redo_out_NN.jsonl`,
one JSON object per line and no other text, one line per input line:
{"row": <same>, "kind": "<same>", "style": "<same>", "form": "<new rewording>"}

Style: `shorter` = fewer words; `indirect` = a question becomes a request ("Tell me", "Name"), an answer a reported or
passive statement; `fact_first` = a question opens with the person or work, an answer with the answer itself; `plainer` =
plain everyday words. An answer form is one short sentence (at most 20 words) containing every word of the answer.

What each failure asks of you:
- "lost a name, number or title": every capitalised word of the original that is not its first word must appear in your
  form, spelled the same: nationalities and proper adjectives too ("Nigerian", "Chilean", "Nobel Prize", "Literature",
  "Renaissance"), every number, and every quoted title in full with its quotation marks. Keep them all, then reword the
  ordinary words around them.
- "new name or number": your form has a capitalised word or a number that is in neither the original nor the question.
  Remove it, or write it in lower case if it is an ordinary word. Don't start a sentence with a capitalised name that is
  not in the original.
- "too like the original": over three quarters of your form's different words appear in the original. Swap ordinary
  words for synonyms and add new connecting words; names you must keep count, so change everything else.
- "too long": a question form over one and a half times the original's words, or an answer form over 20 words. Cut
  ordinary words, never a name, number or title.
- "missing": write the form from scratch for that style.

One line each, no lead-ins ("Sure", "Here", "Answer:"), valid JSON with inner double quotes escaped. Don't run Python.
Check with `wc -l` that the output has as many lines as the input and with `jq -c .row <file> | tail -1` that it parses.
Hand back one line: the path and line count.
