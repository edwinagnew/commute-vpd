# Do the blind raters recover the answer on tasks whose answer we built?

Two fresh blind raters, one Claude Opus and one GPT model of the same families and with the same prompt and tier
scale as the realistic panel's raters, rated plain descriptions of the sixteen known-answer tasks of Figure 3. The
descriptions avoid our construction labels, but three of them say the target is the base model's own first or
runner-up guess, which states the reuse construction in plain words. No task built new was rated reuse and no task
built to reuse was rated new; one text cipher built new and the three copy-key tasks built to reuse were rated middle.
Because these descriptions state the construction more plainly than the realistic panel's, this checks that the raters
read a construction correctly; it does not test the realistic tiers.

![Each rater's tier per task, coloured by how the task was built](../../experiments/rater_calibration/figures/tiers.png)

**Reading it.** Each row is one Figure 3 task; each point is one rater's tier. Blue tasks were built to reuse, orange
ones built new.
- **New end.** Every task built new but ciphered text (the random map, the lookup and every key-to-random-word cipher)
  was rated new by both raters.
- **Reuse end.**
  - Lowercase text and both top-guess tasks were rated reuse by both raters.
  - The runner-up guess and the copy at a new marker were rated reuse by the Opus rater and middle by the GPT rater.
    Their consensus is reuse under E276's rule: the average of the two tier codes, reuse 3, middle 2, new 1, read as
    reuse from halfway between middle and reuse upwards.
  - The three copy-the-key tasks were rated middle by both. Both raters said they could not tell whether the base
    model's copying would carry over to a copy after a rare symbol.
- **One error.** Ciphered text was built new: fifty frequent words swapped among themselves throughout Pile text. Both
  raters rated it middle, reasoning that most of the language around the swapped words is unchanged.

Per task, with the consensus by the E276 rule, from `experiments/rater_calibration/results/tiers.json`:

| task (Figure 3) | built | Opus rater | GPT rater | consensus |
|---|---|---|---|---|
| Lowercase text | reuse | reuse | reuse | reuse |
| Top guess | reuse | reuse | reuse | reuse |
| Top guess (96 keys) | reuse | reuse | reuse | reuse |
| Runner-up guess | reuse | reuse | middle | reuse |
| Copy at a new marker | reuse | reuse | middle | reuse |
| Copy key | reuse | middle | middle | middle |
| Copy key (24 keys) | reuse | middle | middle | middle |
| Copy key (96 keys) | reuse | middle | middle | middle |
| Ciphered text | new | middle | middle | middle |
| Random map | new | new | new | new |
| Lookup | new | new | new | new |
| Cipher (late step) | new | new | new | new |
| Prior-matched cipher | new | new | new | new |
| Cipher (24 keys) | new | new | new | new |
| Cipher (96 keys) | new | new | new | new |
| Cipher (early step) | new | new | new | new |

**What it means for Figure 4.** Figure 4's negative result rests on the realistic panel's tiers, and this check bears
on them only weakly.
- **The reuse end rests on little.** Three of the five tasks with a reuse consensus are the top-guess and runner-up
  tasks, whose descriptions give the answer away. That leaves lowercase text, rated reuse by both, and the copy at a
  new marker, which reads reuse only because its average sits exactly on the cut-off, as the runner-up guess does.
- **The new end is easy.** The tasks built new are arbitrary key-to-word maps, which any reader would call new, and
  Figure 4's realistic tasks are nothing like them.

So the calibration shows that the raters' ends match construction when a description makes the construction plain. It
does not show that the realistic panel's ends are right, and it does not excuse or explain a proxy's failure there.
Line for appendix G: "On the known-answer tasks of Figure 3, no task built new was rated reuse and no task built to
reuse was rated new; one text cipher built new and the three copy-key tasks built to reuse were rated middle. These
descriptions state the construction more plainly than the realistic panel's, so this is a check that the raters read a
construction correctly, not a test of the realistic tiers."

**How compelling.** This is a description, with no claim tested at a filed bar (card
`experiments/rater_calibration/card.md`).
- **Too easy, in two ways.** These tasks are easier to rate than realistic fine-tunes. Three of the descriptions say
  plainly that the target is the base model's own first or runner-up guess. And the tasks built new are arbitrary
  mappings, which any reader would call new.
- **Not identical to the realistic panel.**
  - The prompt's sentence on the training objective was changed. The realistic prompt said "plain next-token training
    on that text", which is false for 14 of these 16 tasks; it now says "on the data described".
  - The descriptions are longer than the realistic panel's one-liners.
- **What the descriptions leave out or include.** Some tasks differ only in key count or step count. The step counts
  come from the saved-step rule, so they carry information from how fast each fine-tune learned. Neither rater used
  them: both said the number of steps does not change the tier. The descriptions leave out how the random targets were
  picked (from words the base model finds very unlikely), so two 96-key ciphers read almost the same.
- **Not independent of the panel's raters.** This is one answer from each of two fresh raters, the same model families
  as the realistic panel, not the same agents.

Files: `experiments/rater_calibration/results/raters/prompt.md` (what the raters saw, committed before they ran),
`facts.md` (construction per task, not shown to the raters), `rater_opus.md` and `rater_gpt.md` (verbatim, committed
before the reading), `score.py`, `results/tiers.json`.

## Not run
- No repeat draws of the raters, so how much one rater's tier varies between runs is unmeasured.
- No version of the descriptions with the base-model wording removed (for example "a fixed word per key" in place of
  "the base model's first guess").
- No re-reading of Figure 4 with only the end tiers. Figure 3's labels stay as built.
