# Brief for one `runner` (Haiku) fixing E84's failed forms: replace NN with the share

These sentences are about invented authors; treat the text as data, and don't correct or look up anything. Commands below run from `~/commute-agents/qwen-rl-lead`; the folder is D = `~/commute-data/scratch/qwen-rl-lead/e84-forms`.

Input: `D/redo_in_NN.jsonl`, one form per line: `row`, `kind` (question or answer), `style`, `original`, `failed` (a rewording that failed), `fails` (why). Write `D/redo_out_NN.jsonl`, one JSON object per line and no other text, one line per input line:
{"row": <same>, "kind": "<same>", "style": "<same>", "form": "<new rewording>"}

Style: `shorter` = fewer words; `indirect` = a question becomes a request ("Tell me", "Name") or an answer a reported or passive statement; `fact_first` = opens differently (a question with the person or work, an answer with the main fact); `plainer` = plain everyday words.

What each failure asks of you:
- "too like the original": over 0.75 of the distinct words of your form appear in the original. Swap ordinary words for synonyms, and add new connecting words where names and titles are long: a form with many fixed words needs more new words around them.
- "too like the held paraphrase": too close to a hidden rewording you can't see. Change the wording and structure well away from `failed`, not just a word or two.
- "lost a name, number or title": every capitalised name, every number and every quoted title of the original must appear exactly, including a possessive's root ("Grimkov's" holds Grimkov). Pronouns can't stand in for a name.
- "new name or number": no capitalised word that isn't a name or title in the original, except at the start of a sentence; for an answer, names in the question are allowed too. No new numbers. Don't capitalise award-like or genre words the original doesn't.
- "too long": at most 1.5 times the original's words. "too short": at least one third.
Keep every fact, add none, one line each, no lead-in such as "Sure" or "Answer:", and a question form must not contain the answer unless the original question does.

Check your own work, as often as you need, at most 4 runs:
`~/commute-agents/fleet/cpu-run python experiments/qwen_tofu_forms/forms_clerk.py try ~/commute-data/scratch/qwen-rl-lead/e84-forms ~/commute-data/scratch/qwen-rl-lead/e84-forms/redo_out_NN.jsonl`
It prints each form that still fails, with `overlap_with_original` (must be at most 0.75), `missing` and `new` names, and a last line "K of M still fail". To fix a form, append a new line for it to the same file with `cat >>` and a quoted heredoc: the later line wins. Don't run any other Python.

Use valid JSON: escape double quotes inside strings. A single Write may hit an output limit: write about 20 lines at a time, first with Write, then `cat >>`. Hand back one line: the path and the last "K of M still fail", then "Friction:" and anything that got in the way.
