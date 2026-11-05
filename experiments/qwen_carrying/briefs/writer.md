# Brief: the writer (E102 stage B, role 1 of 3)

*The lead fills `{BATCH}` (for example `b03`) and `{CLONE}` (the clone's absolute path) and passes this text as the
agent's whole prompt, after `stage_b.py batches`. Model: Sonnet. One call per batch. E101's `briefs/writer.md` with the
model, corpus and paths changed.*

You are reading the neurons of a language model, one MLP neuron at a time, the way an interpretability researcher
reads a neuron from its top activating examples.

**Read one file only:** `{CLONE}/experiments/qwen_carrying/results/large/batches_b/{BATCH}.json`. Do not open, list or
search any other file or folder, and do not run code. Nothing else you could find would help, and some files would
spoil the test.

The file holds about 30 neurons. Each has an opaque `id` and 20 `windows`: short pieces of educational web text
(FineWeb-Edu: articles, lessons, encyclopaedia-style prose, some lists and code) ranked by how strongly the neuron
fired. In each window the token where it fired most is wrapped in `⟦ ⟧`; `rel` is that window's peak as a fraction of
the neuron's strongest window. Tokens are Qwen tokens, so a leading space belongs to the token (`⟦ base⟧` is the word
" base"), and a character split over two tokens may show as `�`.

For each neuron write its **job**: one line, at most 25 words, that says what makes it fire, specific enough that
someone who never saw these windows could write a new text that fires it and a near miss that does not. Name the token
or kind of token, and the context it needs (what comes before it), when the windows show one. Any kind of token can be
the answer, including punctuation, line breaks, numbers and word pieces. Prefer the pattern shared by the top windows;
mention a second pattern only if many windows show it. If no pattern holds, write `unclear:` followed by your best
guess. Do not guess at what the model was trained for or why a neuron was chosen.

**Write** `{CLONE}/experiments/qwen_carrying/results/stage_b/writer/{BATCH}.json`, creating the folders if needed,
exactly in this form, one entry per neuron of the file and nothing else:

```json
{"batch": "{BATCH}", "jobs": {"<id>": "<one-line job>", "...": "..."}}
```

Then reply with one line: the batch, the number of jobs written, and how many start with `unclear:`.
