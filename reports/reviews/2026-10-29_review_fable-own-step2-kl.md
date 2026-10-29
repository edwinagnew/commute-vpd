# Review: the own and null arms' step-2 KL stops (E45, advice, 2026-10-29)

*Fable, for the qwen-decomp lead. Read: the Q2 plan §1, §3-§4, §7-§8, the re-plan §5 (two amendments), `predictions.json`
"early", `predictions_amend1.json`, E40a1's `predictions_amend1.json` (`F_first_live`, `F1_early`), `T_own_s0.json`,
`T_null_s0.json`, `T_gold_s0.json`, `T_a2_s0.json`, `T_a2long_s{0,1}.json`, `S_own.json`, `run.py` `early`, `job.py`
`ARMS`, `qwen_train.py`, the tail of `run.log`. Advice only; the user decides.*

**Plain words.** The own arm and the null arm each stopped because their first live step moved the model's answers to
maths problems a little more than a limit taken from the gold arm allowed, while the size of that step was normal and
the two arms train on the model's own answers, which is what those maths rows are. The limit was written to catch a
missing warm-up, which would move the model about twenty times more than this, and a rerun would give the same
numbers to five decimal places. Both arms should run with the limit restated as the warm-up check it was meant to be,
filed in Amendment 2 beside the two reading-rule items already proposed; neither arm's reading changes.

## 1. What the stops say

| arm, step 2 (rate 1e-6) | |δW| / (rate √numel), min-max | KL on the 128 maths rows | loss at step 1 |
|---|---|---|---|
| anchor 2 (both seeds) | 0.73-0.93 | 0.00087, 0.00085 | 2.70, 2.66 |
| gold | 0.55-0.94 | 0.0037 | 2.408 |
| own samples (correct, de-duplicated) | 0.63-0.82 | **0.00566** (limit 0.005) | 0.0551 |
| null (the same samples, unfiltered) | 0.63-0.81 | **0.00669** (limit 0.005) | 0.0552 |

The per-matrix step is in the middle of the band in all four arms and smaller than gold's largest. Adam's second step
is about the rate per entry whatever the loss scale, so a tiny loss (0.055 nats: the model scoring its own samples)
does not shrink the step; it only changes its direction. The arms order themselves on the maths axis by how much their
training data overlaps the maths rows, which are the model's own answers: anchor 2's facts barely touch them
(0.0009), gold's human solutions touch them some (0.0037), own samples are drawn from the rows' distribution and
filtered to correct ones (0.0057, 1.5 × gold), and the null is that distribution exactly (0.0067, 1.8 × gold). The
null above own samples is the right way round: the filter takes the data slightly off the rows' distribution. The
whole ordering is what the design expects (§3: for these two arms the own axis and the maths axis coincide), and
the null's value, which arrived after the first draft of this note, is the prediction of §1 come true. The limit was
miscalibrated for self-sample data, not the data for the limit: 0.005 is a gold-direction number (1.35 × gold's
rehearsal) with no allowance for an arm whose data sit on the rows.

**Is the null's purpose compromised?** No. (A term first: in the plan the null is the *self-distillation null*, SFT
on the model's unfiltered samples; "do-nothing" in `GOALS.md` is E44's reference state where δW is left whole, a
different thing.) The null is read beside anchor 1 and own samples at matched KL; it carries no prediction, no label
and no expected curve. A larger per-step KL on the maths rows says only that its direction lies along those rows, as
it must for data drawn from them; it does not say its behaviour moves more. If anything it helps: my anchor-1 review
worried the null might not reach own-KL 0.1 by step 100 and so have no state at the decisive point; climbing faster
than gold at equal step size makes that less likely. The one thing to carry into the write-up: at matched KL the null
has taken fewer and smaller-in-effect steps than gold, which is exactly the comparison the plan asks for.

The limit's source is E40a1's `F1_early`, "KL to start after the first step with a nonzero rate ≤ 0.005 (code check,
not a prediction)", set at 1.35 × gold's rehearsal value. The failure it guards is E40's: the warm-up missing, so that
step 2 runs at 1e-5 instead of 1e-6. The control measured that case at 0.10679 for one gold step at peak rate; with
the warm-up in place the first live step is 10 × smaller in rate and about 100 × smaller in KL. A miss by 13% is not
that failure mode; a limit with no margin for the arm's direction is.

**A rerun as filed misses again.** Training is deterministic to about 2 × 10⁻⁵ relative: `a2long_s0` reproduced
`a2_s0`'s step-2 maths KL as 0.00087281 against 0.00087283 (same seeded data order, same card). The own arm would
read 0.00566 again, the null 0.00669. Anchor 1 (the model's own samples under the format prompt, taught without it)
trains on the model's own distribution too and will most likely stop at the same check; whatever is done must cover
all three on-policy arms, or the next job stops for the same reason.

## 2. Options

| option | cost | amendments | what it lets us conclude |
|---|---|---|---|
| (A) Rerun as filed | one `debug` job, ~1 min to the stop | none | nothing: deterministic miss (§1) |
| (B) Restate the check as the warm-up check it was meant to be, for every arm: step-2 KL on the maths rows ≤ 0.1 × the control's one-step-at-peak KL (0.1 × 0.10679 = 0.0107); the per-matrix band stays as the step-size check. Expected values written in: gold 0.0037, anchor 2 0.0009, own samples 0.0057 and the null 0.0067 (1.5 and 1.8 × gold, in-distribution data), anchor 1 unknown but expected between gold's and the null's | $0; T/E own and T/E null ~35 min of `debug` each, as planned | one item in Amendment 2 | own samples, the null and anchor 1 run under the one recipe and are read as filed. The new limit is sourced (the control's measured value and the recipe's 10 × rate ratio), fires on a missing warm-up (0.107 ≫ 0.0107) and would have passed every arm run so far |
| (C) own2 (peak 2e-5) | ~35 min | the §8 branch is for "not learned at step 100", not for this; using it here spends the one rerun before the floor has been read | nothing new, and it stops at the same check: `run.py` scales the limit as (rate / 1e-5)² to 0.02, and 4 × 0.00566 = 0.0226 misses it |
| (D) A lower rate for own samples (8e-6 passes: 0.64 × 0.00566 = 0.0036) | ~35 min | one amendment, a recipe change | a second recipe for one arm, against "one recipe" (§3); the grid is matched on KL, not steps, so a slower climb buys nothing. "Chosen here" |
| (E) Drop the KL check for the on-policy arms only, keep the band | $0 | one item in Amendment 2 | as (B) but with no warm-up guard on three arms; (B) keeps the guard with a sourced number |

**Recommendation: (B).** It fixes the structure that allowed the stop (a gold-calibrated number applied to arms whose
data sit on the rows) rather than adding an exception beside it. The band check already does the job the plan's
E40a1 lesson wanted (a first step of normal size); the KL check then needs only to tell "warm-up present" from
"absent", two orders of magnitude apart, and 0.1 × the control value does that for any arm. Also fix the `early`
block's source line: `step2_kl_max` is the one entry there without one.

**Amendment 2.** Yes, it rides in the same note: all three items (anchor 2's decisive state, anchor 1's
whitespace-tolerant regex, this check) are reading or check rules; none changes what any arm trains on. The
re-plan's cap is two amendments, not two items; list each item with its own expected values. Order matters: the own T
run refuses to start without its predictions file, so `predictions_amend2.json` must exist and `ARMS["own"]`, `ARMS["null"]` and `ARMS["a1"]` point to
it before T own, and the file is not edited after that. Anchor 2's item is already decided (option (d) of my
anchor-2 review); anchor 1's needs the fixed regex to reproduce P's format reading first. If that takes hours, own
samples wait: at $0 that is acceptable, and better than two files.

## 3. Are the own and null arms' readings affected?

No. Every reading is at a saved state matched on KL, not on step (§3, §5): own samples and the null will cross 0.01 and
probably 0.03 at step 3 (gold went 0.0037 → 0.022 from step 2 to 3; both start higher), so their 0.01 state is
read at step 2 (matched KL 0.0057 and 0.0067, |log| 0.56 and 0.40 against about 1.1 for step 3) and stated as such, the way gold's 0.01
state is read at step 3 (0.022). The decisive point is 0.1 on own rows, which own samples will bracket at steps 4-5
like gold. A step-2 KL of 0.0057 or 0.0067 rather than under 0.005 changes no filed rule, no threshold, no pairing. What it does
say, worth one sentence in the write-up: the on-policy update climbs the KL grid faster per step than gold at equal
step size, so its grid states, like gold's, are early-warm-up states.

## 4. What would change this advice

- If the per-matrix band had been missed, or step 1 had moved the weights: then the step itself is wrong and (B)
  would hide it. Both held.
- If a rerun did not reproduce 0.00566 to four figures: the data order is not seeded as `cycle` says, and the
  anchors' seed-1 runs are not what the plan thinks they are. (A) is then worth its minute as a check, nothing more.
- If anchor 1's step-2 KL came out above 0.0107 under (B), or below gold's 0.0037: then "overlap with the rows" does
  not explain the ordering and the recipe needs looking at before more arms run.

**Not checked.** Whether E40's 128 maths rows come from problems inside the 2,000 own-sample training problems (if
so, the own arm's maths-axis KL is partly training-row KL; it affects every arm's matched KL on that axis, own samples
most). Whether 0.055 nats is the expected step-1 loss for T = 0.7 samples scored by the model (no reference exists).
