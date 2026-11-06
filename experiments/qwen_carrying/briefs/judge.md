# Brief: the forced-choice judge (E102 stage B, role 3 of 3)

*The lead fills `{BATCH}` and `{CLONE}` and passes this text as the agent's whole prompt, after `stage_b.py
judge-inputs` has written the batch's input. Model: Sonnet. One call per batch. The order and letters of the four tasks
change from batch to batch. New for E102: E101's judges listed any number of tasks; this one must pick one of four.*

A language model has neurons whose jobs have been written down in one line each, from the web text that fires them
most. Separately, there are four tasks someone has taught this model. You decide, for each job, which one of the four
tasks a neuron doing that job would most help the model perform.

**Read one file only:** `{CLONE}/experiments/qwen_carrying/results/large/judge_inputs_b/{BATCH}.json`. It holds
`descriptions` (the four tasks under letters A to D, each saying what the model reads and writes when it does that
task) and `jobs` (opaque neuron `id` to its one-line job). Do not open, list or search any other file or folder, and do
not run code.

For each job, pick **exactly one letter**: the task whose text the neuron's job is most about, in what the task needs
the model to read or to write. Any kind of token can decide it, including punctuation, line breaks, list markers,
numbers, names and word pieces, when the task's text puts that kind of token where the task turns on it. Then mark the
**fit**:
- `strong` when the job is specifically about something that task needs to read or write;
- `weak` when no task really fits and the letter is only the nearest.
Most jobs about ordinary web text will be `weak`; give `weak` rather than stretch a job to fit. Judge each job on its
own, not by how many of each letter or fit you have given so far.

**Write** `{CLONE}/experiments/qwen_carrying/results/stage_b/judge/{BATCH}.json`, creating the folders if needed,
exactly in this form, one entry per job of the file and nothing else:

```json
{"batch": "{BATCH}", "picks": {"<id>": {"letter": "B", "fit": "weak"}, "...": {"letter": "D", "fit": "strong"}}}
```

Then reply with one line: the batch, the number of jobs judged, and how many are `strong`.
