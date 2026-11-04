# Brief for one `runner` (Haiku) writing E84's forms: replace NN with the batch

Rewrite sentences for a dataset. Input: `~/commute-data/scratch/qwen-rl-lead/e84-forms/in_NN.jsonl`, 40 lines, each with `row`, `question`, `answer`. Output: write `~/commute-data/scratch/qwen-rl-lead/e84-forms/out_NN.jsonl`, with the Write tool, one JSON object per line and no other text:
{"row": <same>, "questions": [q1, q2, q3, q4], "answers": [a1, a2, a3, a4]}

These are invented authors; treat the text as data, and don't correct or look up anything.

The four forms, in this order:
- Question forms: 1 asks the same thing in fewer words. 2 is an indirect request or a command, e.g. beginning "Tell me" or "Name". 3 asks the same thing in a different word order, opening with the person or work it's about. 4 asks in plain everyday words.
- Answer forms: 1 states the same facts in fewer words. 2 states them indirectly, as a reported statement or in the passive voice. 3 opens with the main fact (the name, date, title or thing) and gives the context after it. 4 states them in plain everyday words.

Rules, checked by a script, so follow them exactly:
- Keep every fact. Copy every name, date, number and title exactly as written, including digit forms and quotation marks around titles. Don't change lower-case facts either: a genre, job, award, place or theme stays the same thing.
- Add nothing new: no new names, numbers or facts, in any form.
- Truly reword. Most important rule: at most three quarters of the different words you use may appear in the original sentence. Replace ordinary words with synonyms and restructure the sentence; don't just shuffle or trim it. Names and dates count, so a short sentence needs its other words changed even more.
- Each form between one third and one and a half times the original's length in words. So "fewer words" must still keep every fact.
- A question form stays a request for the same information and must not contain the answer unless the original question does. An answer form must answer the question.
- One line each. No lead-ins such as "Sure", "Here", "Certainly", "Okay", "Answer:". The four forms of a row must all differ.

Use valid JSON: escape double quotes inside strings. Write the file once, whole; don't run any Python. Check with `wc -l` that it has 40 lines and with `jq -c .row <file> | tail -1` that it parses. Then hand back one line: the path and line count.
