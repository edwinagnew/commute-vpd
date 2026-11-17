# Appendix: the sealed pair's specification

*Companion to `2026-11-09_plan_sealed-pair.md`. Enough to build the pair without asking anyone. Nothing here has run.*

Every setting not named below is the repeat-label pair's, filed in `experiments/fresh_reuse_pair/results/predictions.json` under
`settings` and `settings_sources`. Its builder is `src/commute/repeat_label.py`.

## Windows

- **Layout.** 41 Pile tokens from the same cached rows as the repeat-label pair. A pool word sits at position 19 and the marker " \+" at
  position 20. The label is read at position 20.
- **Span.** A window's span is the first of 16 seeded draws with no bracket token and no copy of the word in positions 0 to 18.
- **The two classes.** Two distinct positions p1 < p2 are drawn from 1 to 17. In an **open** window, position p1 holds ")" and p2 holds
  " (", so the bracket is unclosed at the word. In a **closed** window, p1 holds " (" and p2 holds ")". Both classes carry the same two tokens.
- **Tokens.** " (" is id 313 and ")" is id 10 in `EleutherAI/gpt-neox-20b`, read from the cached `tokenizer.json`.
- **Counts.** 9,600 training windows, 1,024 fit windows and 1,600 held-out windows. Classes alternate within each word, so each word
  has both classes in equal number.
- **Row split.** Training windows come from rows with index mod 5 not 0, and fit and held-out windows from rows with index mod 5 equal to 0.

## Words and labels

- **Pool.** Single tokens of a space and letters, with at least 32 Pile positions. The marker, the label candidates and both bracket tokens
  are excluded. 400 are drawn by the secret word seed: the first 300 for training, the last 100 held out.
- **Label words.** A and B come from the repeat-label pair's candidates, by `case_label.pick_labels`, " carpet" and " camel" there.
- **Real arm.** A when the window is open, B when closed.
- **Twin.** A or B per word, by the twin seed's shuffle, half the words each, separately for training and held-out words.

## Recipe

- Three seeds per arm. All seeds are drawn from `/dev/urandom` inside the job, written only into the encrypted bundle.
- The filed recipe: Adam at learning rate 3e-5, batch 32, KL weight 1, gradient clip 1, all 24 matrices, target `data.LAST`.
- Checkpoints from 10 to 1000 steps, stopping at the first with fit accuracy at least 0.9.
- A state that misses the bar at 1000 steps makes the pair unusable, and the job prints only that.

## Stage 0, the original model alone, before any fine-tune

- **Positive control.** The original model's log-probability of ")" at position 19, mean over open windows minus mean over closed windows.
  It should be above zero, as the induction gap was for the repeat label. The bar is filed with the run cell and set from the spread across
  windows in the same job.
- **Beside, no rule.** A ridge probe of "a bracket is open here", trained on Pile positions only, read at the trigger on held-out windows,
  with the Hewitt-Liang per-type control. It is reported for the definition's later use and is not a stop.
