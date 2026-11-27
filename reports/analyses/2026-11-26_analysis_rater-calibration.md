# Do the blind raters recover the answer on tasks whose answer we built?

Fresh blind raters (one Claude Opus and one GPT model, of the same families and with the same prompt and tier scale as
the realistic panel's raters) rated the sixteen known-answer tasks of Figure 3 twice. In the first run, three
descriptions named the reuse construction outright. In the second, every key-to-word task was described alike, with
three real example pairs and nothing about how the targets were chosen. In both runs no task built new was rated reuse
and no task built to reuse was rated new, and the consensus tier of every task was unchanged between the two runs. The raters read the
construction from the data themselves, not only from the wording. The tasks built new are still arbitrary word maps,
unlike Figure 4's realistic tasks, so this checks the raters' reading of plain cases; it does not test the realistic
tiers.

![Each rater's tier per task in each run, coloured by how the task was built](../../experiments/rater_calibration/figures/tiers.png)

## What was run
- **Run 1.** One plain description per task, with no construction labels. Three of the descriptions said the target
  is the base model's own first or runner-up guess, which states the reuse construction in plain words. The
  key-to-word ciphers were described as "a word assigned to that key at random", which states the new construction.
  Each description also gave the number of training steps.
- **Run 2.** Every key-to-word task got the same wording: "a fixed word per key", plus the first three key and target
  pairs from the saved data. The two text tasks got one real sentence fragment each, before and after the change.
  There was no reference to the base model's predictions, no word about how targets were chosen, and no step counts.
  Key counts stayed, and they do not tell the tasks apart: each count occurs on both sides.
- **Both runs.**
  - The raters were fresh agents with no tools.
  - Each prompt was committed before the raters saw it, and each answer before it was read against the labels.
  - The consensus follows E276's rule: the average of the two tier codes (reuse 3, middle 2, new 1), read as reuse
    from halfway between middle and reuse upwards, and as new from halfway between middle and new downwards.
  - The prompt's sentence on the training objective was changed from "plain next-token training on that text" to "on
    the data described", because the former is false for 14 of these 16 tasks.

## Reading it
In run 2 the raters judged from the example words: natural continuations such as " type" to " of" read as reuse, and
unrelated pairs such as " type" to " primarily" read as new.
- **What moved between runs.**
  - The Opus rater moved lowercase text from reuse to middle.
  - The GPT rater moved the prior-matched cipher from new to middle.
  - No consensus tier changed.
- **The reuse end.** Both top-guess tasks were rated reuse by both raters in both runs. Lowercase text, the runner-up
  guess and the copy at a new marker each had one rater at reuse and one at middle in at least one run. Each of these
  reads reuse only because its average sits exactly on the cut-off.
- **The middle.** The three copy-key tasks, built to reuse, and ciphered text, built new, were rated middle by both
  raters in both runs.

Per task, each rater's tier in each run, and the consensus, from `experiments/rater_calibration/results/tiers.json`:

| task (Figure 3) | built | run 1 Opus | run 1 GPT | run 1 consensus | run 2 Opus | run 2 GPT | run 2 consensus |
|---|---|---|---|---|---|---|---|
| Lowercase text | reuse | reuse | reuse | reuse | middle | reuse | reuse |
| Top guess | reuse | reuse | reuse | reuse | reuse | reuse | reuse |
| Top guess (96 keys) | reuse | reuse | reuse | reuse | reuse | reuse | reuse |
| Runner-up guess | reuse | reuse | middle | reuse | reuse | middle | reuse |
| Copy at a new marker | reuse | reuse | middle | reuse | reuse | middle | reuse |
| Copy key | reuse | middle | middle | middle | middle | middle | middle |
| Copy key (24 keys) | reuse | middle | middle | middle | middle | middle | middle |
| Copy key (96 keys) | reuse | middle | middle | middle | middle | middle | middle |
| Ciphered text | new | middle | middle | middle | middle | middle | middle |
| Random map | new | new | new | new | new | new | new |
| Lookup | new | new | new | new | new | new | new |
| Cipher (late step) | new | new | new | new | new | new | new |
| Prior-matched cipher | new | new | new | new | new | middle | new |
| Cipher (24 keys) | new | new | new | new | new | new | new |
| Cipher (96 keys) | new | new | new | new | new | new | new |
| Cipher (early step) | new | new | new | new | new | new | new |

## What it means for Figure 4
Figure 4's negative result rests on the realistic panel's tiers, and this check bears on them only weakly.
- **What it shows.** With the construction wording removed, the raters still place every task at the end it was built
  for, or in the middle. So their tiers come from the data they are shown, as on the realistic panel.
- **The reuse end is thin.** Three of the five tasks with a reuse consensus sit on the cut-off in at least one run; only
  the two top-guess tasks are reuse throughout.
- **The new end is easy.** The tasks built new are arbitrary key-to-word maps, which any reader would call new, and
  Figure 4's realistic tasks are nothing like them.

So the calibration shows that the raters place plain cases correctly from the data. It does not show that the realistic
panel's ends are right, and it does not excuse or explain a proxy's failure there.

Line for appendix G: "On the known-answer tasks of Figure 3, described by their data alone with example pairs, no task
built new was rated reuse and no task built to reuse was rated new; one text cipher built new and the three copy-key
tasks built to reuse were rated middle. The tasks built new are arbitrary word maps, so this checks that the raters
read plain cases correctly, not that the realistic tiers are right."

## How compelling
This is a description, with no claim tested at a filed bar (cards `experiments/rater_calibration/card.md` and
`card_run2.md`).
- **The examples still carry the construction.** This is because they are real data. The top-guess and runner-up
  targets are natural next words and the cipher targets are not, which is what the tasks are. A rater reading the
  realistic panel's data faces a harder judgement.
- **Run 2's descriptions are longer than the realistic panel's one-liners.** They are also identical in wording for 14
  tasks. The early-step and late-step ciphers are the same fine-tune read at two steps, so they read the same in run 2,
  and the 24-key cipher shows the same three pairs as both.
- **How the examples were chosen.** The examples are the first three keys in saved order, with one exception: the key
  " control" was skipped for a word on the banned list. The ciphered text fragment was cut to avoid the word "known".
  Sources for every pair are in `run2/facts.md`.
- **Repeats.** There is one answer from each rater per run, so how much a rater's tier varies between draws is
  measured only by the two runs, which also differ in the descriptions.

Files: `experiments/rater_calibration/results/raters/` holds `prompt.md`, `facts.md`, `rater_opus.md` and
`rater_gpt.md` for run 1. The subfolder `run2/` holds the same four files for run 2. The scorer is `score.py` and the
per-task results are in `results/tiers.json`.

## Not run
- No third draw of the raters on run 2's descriptions.
- No version of the descriptions without example pairs, which would leave the key-to-word tasks indistinguishable.
- No re-reading of Figure 4 with only the end tiers. Figure 3's labels stay as built.
