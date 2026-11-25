# Brief: the judge, Amendment 1 (E101 stage 1)

*The lead fills `{BATCH}` (for example `j03`) and `{CLONE}` (the clone's absolute path) and passes this text as the
agent's whole prompt, after `score.py amend1-inputs` has written the batch's input. Model: Sonnet. One call per batch.
The order and letters of the tasks change from batch to batch. Replaces `judge.md` for Amendment 1
(`results/predictions_amend1.json`); that brief stays as it ran.*

A small language model (a GPT-style model trained on web text) has neurons whose jobs have been written down in one
line each, from the web text that fires them most. Separately, there is a list of tasks someone might teach the model.
Each task comes with a few of its own examples. You decide, for each job, which tasks (if any) a neuron doing that job
could plausibly help the model perform.

**Read one file only:** `{CLONE}/experiments/carrying_jobs/results/large/judge_inputs_amend1/{BATCH}.json`. It holds
`tasks` (under letters A, B, ...: each a `task` description and its `windows`) and `jobs` (opaque neuron `id` to its
one-line job). Do not open, list or search any other file or folder, and do not run code.

Each window is the text the model has read so far, followed by the answer it should produce next, wrapped in `⟦ ⟧`.
Tokens are GPT-NeoX tokens, so a leading space belongs to the token. Look at what sits just before the answer, and at
the answer itself: which tokens, which positions, what kind of text.

For each job, ask of each task: could a neuron doing this job plausibly help the model produce the marked answer in
windows like these? A job helps when the thing it fires on appears in the task's windows where it matters (at or just
before the answer, or in what the answer depends on), or when what it fires on is the kind of token the answer is. Any
kind of token can count, including punctuation, line breaks, markers and word pieces, if the windows put it where the
answer turns on it. A job about something these windows do not contain helps none of the tasks. Give no letter rather
than a weak one, and several letters when the job really serves each. Judge each job on its own, not by how many you
have marked so far.

**Write** `{CLONE}/experiments/carrying_jobs/results/stage1/judge_amend1/{BATCH}.json`, creating the folders if
needed, exactly in this form, one entry per job of the file and nothing else:

```json
{"batch": "{BATCH}", "ties": {"<id>": ["B"], "<id>": [], "...": ["A", "D"]}}
```

Then reply with one line: the batch, the number of jobs judged, and how many got at least one letter.
