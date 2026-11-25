# Brief: the maximiser (E101 stage 1, role 2 of 3)

*The lead fills `{BATCH}` and `{CLONE}` and passes this text as the agent's whole prompt, after the writer's file for
the batch exists. Model: Sonnet. One call per batch. This is the essay's test "Explanations That Could Lose"
(https://already-clean.goodfire.com/essay.md): five examples and five near misses per explanation, written before any
measurement.*

Someone has written a one-line explanation of each of about 30 neurons of a small language model (a GPT-style model
with a GPT-NeoX tokenizer, trained on web text from the Pile). Your texts will be run through the model to test each
explanation: an explanation wins if the texts you wrote to fire the neuron fire it strongly more often than the near
misses do.

**Read one file only:** `{CLONE}/experiments/carrying_jobs/results/stage1/writer/{BATCH}.json`, which maps each opaque
neuron `id` to its explanation. Do not open, list or search any other file or folder, and do not run code.

For each neuron write:
- **5 examples**: new texts, each 10 to 60 words, that the explanation says should make the neuron fire strongly
  somewhere in the text. Vary them; do not repeat one sentence five times.
- **5 near misses**: texts as close as you can make them to your examples (same topic, format and most words) that the
  explanation says should not fire it, because the one feature the explanation names is missing or changed.
If an explanation starts with `unclear:`, test the guess it gives. Write plain text only: no ⟦ ⟧ marks, no notes.

**Write** `{CLONE}/experiments/carrying_jobs/results/stage1/maximiser/{BATCH}.json`, creating the folders if needed,
exactly in this form, one entry per neuron of the writer file and nothing else:

```json
{"batch": "{BATCH}", "texts": {"<id>": {"examples": ["...", "...", "...", "...", "..."],
                                       "near_misses": ["...", "...", "...", "...", "..."]}}}
```

Then reply with one line: the batch and the number of neurons written.
