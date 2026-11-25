# Brief: the judge (E101 stage 1, role 3 of 3)

*The lead fills `{BATCH}` and `{CLONE}` and passes this text as the agent's whole prompt, after
`score.py judge-inputs` has written the batch's input. Model: Sonnet. One call per batch. The order and letters of the
descriptions change from batch to batch.*

A small language model (a GPT-style model trained on web text) has neurons whose jobs have been written down in one
line each. Separately, there is a list of tasks someone might teach the model. You decide, for each job, which tasks
(if any) a neuron doing that job would plausibly help the model perform.

**Read one file only:** `{CLONE}/experiments/carrying_jobs/results/large/judge_inputs/{BATCH}.json`. It holds
`descriptions` (task descriptions under letters A, B, ...) and `jobs` (opaque neuron `id` to its one-line job). Do not
open, list or search any other file or folder, and do not run code.

For each job, list the letters of the tasks it would plausibly help with. Count a task only when the job is
specifically about something the task needs to read or to write: the kind of token, text or position the task turns
on. A job that fires on text in general, on punctuation or on a topic the task does not mention helps none. Most jobs
should help none; give no letter rather than a weak one. Several letters are allowed when the job really serves each.
Judge each job on its own, not by how many you have marked so far.

**Write** `{CLONE}/experiments/carrying_jobs/results/stage1/judge/{BATCH}.json`, creating the folders if needed,
exactly in this form, one entry per job of the file and nothing else:

```json
{"batch": "{BATCH}", "ties": {"<id>": ["B"], "<id>": [], "...": ["A", "D"]}}
```

Then reply with one line: the batch, the number of jobs judged, and how many got at least one letter.
