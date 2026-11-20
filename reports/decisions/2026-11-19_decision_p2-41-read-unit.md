# P2.41, due 11-22: the RL readings scored the wrong thing on lowercase; read the saved states once more as a contrast?

*2026-11-19, compare-lead. Pending decision, raised after the cell's fourth and last run. `decisions.md` and `GOALS.md`
are unchanged.*

## The answer

The cell asked whether a behaviour reached by reinforcement learning on the 67M model reads like the same behaviour
reached by supervised training. The last run could read only two of nine arms, so the filed rule makes no call. One
reading came out the wrong way round, and the cause is in how the rows were scored, not in the model. One more read
of the same saved updates, scored the way the reward scores, costs nothing and would show whether the comparison can
be made.

## What has been measured

Four runs, all on the cluster at no cost:

- **Run 1** measured how often the original 67M model already shows each of 13 behaviours that a program can score.
  Two were picked: *repeat*, copying the tokens that followed an earlier occurrence of the last two tokens, and
  *lowercase*, writing 16 tokens with no capital. French after English text, which the model almost never writes,
  became the taught anchor.
- **Runs 2 and 3** trained each behaviour four ways, three seeds each: reinforcement learning on the reward;
  supervised training on the model's own rewarded samples; supervised training on rewarded text from the corpus; and
  reinforcement learning with the rewards shuffled, as a control. Every update stopped at the same size. All 27
  updates are saved.
- **Run 4** read the re-gating share and the scorecard baselines on each saved update. The share is the part of an
  update's effect that the released decomposition carries by switching different subcomponents on, from 0 to 1. It
  is read only when the update raised the scored token by at least 1 log-odds, the floor.

## Did it come out as expected?

Mostly. As predicted, the supervised arms moved the scored token too little to read. Reinforcement learning on repeat
read a share of 0.68 to 0.79, and the French anchor 0.53 to 0.56. The filed call is "no call" on both behaviours, as
filed.

**One reading did not.** Reinforcement learning made lowercase samples 98 to 100 times in a hundred, yet it lowered
the scored token's log-odds by 2.25 to 3.33. The cause is in the code. The read rows score the exact next word of a
Pile text that stays lowercase, against every other token. The reward accepts any lowercase text. Reinforcement
learning learned to write lowercase word salad, which earns the reward and no longer predicts the text. Across the
twelve lowercase updates the drop tracks how far each update moved the model's predictions on ordinary text, with a
correlation of -0.97. So the reading measured drift, not the behaviour the reward asks for.

## What one more run would measure that nothing has

Whether each arm moves the model toward the form the reward prefers, at the same position and on the same word:

- **Lowercase**: the log-probability of the lowercase form of the next word minus that of its capitalised form, as
  E53 scored lowercase. Drift that lowers both forms alike cancels out.
- **Repeat**: on rows where the text does not copy, the log-probability of the token a copy would write minus that of
  the token the text has.

Nothing so far has measured this. It tells us whether reinforcement learning and supervised training reach the
behaviour by the same route, which is what the cell set out to compare.

## Can it be built from what is on disk?

Yes, checked in the code:

- **Inputs.** The read rows are rebuilt as run 4 built them, from the cached Pile rows, the tokenizer and the scan in
  `experiments/rl_67m/run_stage3.py` `read_rows`. The 27 updates are saved with checksums. Nothing is retrained.
- **Lowercase pairs.** Every scored token is a lowercase word that begins with a space, by the cut rule in
  `rl_tasks.eligible`. Its capitalised partner is the same word with a capital, kept when the vocabulary holds it as
  one token. E53's `performed.case_pairs` makes the same check the other way round. How many rows keep a pair is
  counted in the run's first step.
- **Repeat pairs.** The scan stores, per prompt, the tokens a copy would write and the tokens the text has
  (`rl_tasks.repeat_cuts`, `pile_prompts`). Of 2,048 prompts scanned, 369 copy. Pairs come from the rest, where the
  two differ. Run 4's rows cannot be used, since there the two are the same token.
- **Code.** The share already accepts any score per row (`masked_reads.share_cells_by`). The scorecard's two-form
  score takes one fixed pair for a whole task (`steering.contrast_at`). A pair that changes from row to row needs a
  small new score, about 20 to 40 lines, and a short driver that reuses run 4's.
- **Time.** Run 4 read each update in about 18 seconds. Three jobs in parallel take about 20 minutes on a cluster
  card, at no cost.

## The question

The cell's four runs are spent, and as it stands it closes with "no call". Allow one more run that reads the same 27
saved updates, scored as the contrast between the two forms the behaviour separates, so that the reading follows the
reward?

## Options

**A. One more read, as a contrast (recommended).** It runs the share and the baselines on the 27 saved updates with
the two contrasts above. If reinforcement learning and at least one supervised arm clear the floor on a behaviour,
the cell's comparison is scored by its filed rule. If the supervised arms still do not clear it, we learn that
matching update size does not match how far the behaviour moved, and the cell closes knowing that. A loophole stage,
where the policy earns the reward without the behaviour, would then have a score that can tell the two apart.

**B. Close as it stands.** The cell fills with "no call": one readable route on one behaviour, and an explained
lowercase reading. The route question stays open. A loophole stage would inherit the same scoring problem, since
the present score cannot tell a reward hack from drift.

I recommend A: it costs nothing, needs no retraining, and is the only way the four runs answer the question they
were built for.

## Files

- Report: `reports/analyses/2026-11-19_analysis_e222-rl-readings.md`
- Results: `experiments/rl_67m/results/stage3_score.json`, `stage3/*.json`, `stage3_gain_vs_kl.json`
- E220's updates, held-out reward rates and samples: `experiments/rl_67m/results/stage2b_*.json`
- Plan: `reports/plans/2026-11-19_plan_rl-67m.md`
