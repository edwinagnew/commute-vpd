# Brief for one `runner` (Haiku) writing E103's forms: replace NN with the batch

E84's clerk brief (`experiments/qwen_tofu_forms/CLERK_BRIEF.md`) with the answer rules changed for one-to-five-word answers.
Folder D = `~/commute-data/scratch/baseline-lead/e103-writer/forms`.

Rewrite sentences for a dataset. Input: `D/in_NN.jsonl`, up to 44 lines, each with `row`, `question`, `answer`. Output:
write `D/out_NN.jsonl`, with the Write tool, one JSON object per line and no other text:
{"row": <same>, "questions": [q1, q2, q3, q4], "answers": [a1, a2, a3, a4]}

These are well-known facts; treat the text as data, and don't correct anything, even if you think an answer is wrong.

The four forms, in this order:
- Question forms: 1 asks the same thing in fewer words. 2 is an indirect request or a command, e.g. beginning "Tell me" or
  "Name". 3 asks the same thing in a different word order, opening with the person, work or thing it's about. 4 asks in
  plain everyday words.
- Answer forms: each is ONE short sentence that answers the question and contains the answer's words exactly. 1 is as
  short as a sentence can be ("It was Jane Austen."). 2 is a reported statement or passive voice ("'Pride and Prejudice'
  was written by Jane Austen."). 3 opens with the answer, then the context ("Jane Austen wrote 'Pride and Prejudice'.").
  4 is plain everyday words ("The writer of that book is Jane Austen.").

Rules, checked by a script, so follow them exactly:
- Copy every name, date, number and title exactly as written, including digit forms and quotation marks around titles.
- Every word of the answer appears in every answer form, spelled exactly as given (keep "Antarctic" as "Antarctic").
- Add nothing new: no names, numbers or facts that are in neither the question nor the answer. Do not add a country, a
  first name, a date or an explanation.
- Truly reword a question. At most three quarters of the different words of a question form may appear in the original
  question. Use synonyms and restructure; names and titles count, so a short question needs its other words changed more.
  Each question form between one third and one and a half times the original question's length in words.
- An answer form has at most 20 words, and at most three quarters of its different words may be the answer's own words
  (so "Paris." alone fails; "The answer is Paris." passes).
- A question form stays a request for the same information and must not contain the answer.
- One line each. No lead-ins such as "Sure", "Here", "Certainly", "Okay", "Answer:". The four forms of a row must differ.

Use valid JSON: escape double quotes inside strings. A single Write may hit an output limit: write parts of about 11 rows
in `D/parts/NN_*.jsonl` (mkdir -p it) and join them with cat. Don't run any Python. Check with `wc -l` that the output has
as many lines as the input and with `jq -c .row <file> | tail -1` that it parses. Then hand back one line: the path and
line count.
