# E150: a built control on the parity of a sum, with a control that needs no teaching

*2026-11-11. Cell P1.10 (1 run, cluster debug, $0, due 11-13). Design: Fable's option A and its addendum
(`~/commute-data/scratch/definition-lead/built-control/fable_advice_natural_2026-11-11.md`), chosen by the user.
Card: `experiments/built_control_sum/card.txt`. Filed rules: `experiments/built_control_sum/results/predictions.json`.
Code: `experiments/built_control_sum/run.py`, `cluster/built_control_sum.sbatch`. Nothing has run.*

**In plain words.** E148 found no map on either of its arms, but its taught original held the skill too weakly to count
as a positive control. This run keeps the adopted wording and cut-offs and asks the question again on a natural task, the
parity of a sum of two numbers, with a control the 67M needs no teaching for: "the last number is odd". If the search
finds a map for the control and none for the sum, a built skill is detectable as a contrast; if it finds none for the
control either, the search fails on this layout and the built control says nothing.

## What runs

**Task.** Windows are D's 8-token windows; 3/4 end in two numbers a, b from 0 to 99 (each one token as " n" in the 67M's
tokenizer: all 100 are, checked on CPU; 50 odd), 1/4 are far windows with no number in the last two positions. A seeded
quarter of the 10,000 ordered pairs (2,500) is held out: fits and training use the others, every reading uses only these.
The numbers are coded as an `Xor2` with the parity as each key's bit, so E148's `xor2_f` is the parity of a + b and its
window code is reused unchanged (`commute.variable_read.make_numbers`). 100 numbers, fewer than the model's width, make
a bit of b an exact affine function of b's embedding.

**Three arms**, each a fine-tune toward its own original + alpha Q b at block 2's input, last position, where its bit is 1
(Q b is E137's XOR push, as E148):

| arm | original | bit | role |
|---|---|---|---|
| control | plain 67M | b is odd | positive control, read first |
| plain | plain 67M | a + b is odd | the built case |
| taught | taught original | a + b is odd | second control, behind a gate |

The taught original is the 67M fine-tuned toward the 67M + alpha Q b' on a + b odd, where Q b' is a seeded direction of the
same span at right angles to Q b with its length (the same skill with other answers). Gate: a ridge read of a + b's parity
from it, on held-out pairs, at least 0.9 (300 steps, else 3000). A miss drops the taught arm and is reported.

**Alpha.** E148's grid (1 to 16 in units of Q b). The smallest alpha whose control and plain target gaps are both at least
0.2 (E148's aim); one alpha for all three arms. E148's XOR2 read 0.217 at 4 and 0.258 at 6 with the same share of pushed
windows, so 4 or 6 is expected. Each gap must be above 0.15, or the job stops (taught: the arm drops).

**Per arm.** Seed 0 at 300 steps (E139's recipe), else 3000 (E139 Amendment 2's), passing E148's learned check (ratio at
most 0.9 on held-out pairs); seed 1 on seed 0's rung. H at five sites on both seeds, G at four sites on seed 0 (in2 only
after 3000 steps). All reading code is E139's and E148's, unchanged.

## Stage 0 (first, alone; job 1)

| check | expected | else |
|---|---|---|
| zero-step fine-tune of the 67M (W0) | gap 0, max TV at the float floor (E148: 1.1e-5) | stop |
| XOR2 read from E148's saved plain fine-tune (full push, 300 steps) | open, 0.588 to 1 | reported; at least 0.9 says the full-push recipe can teach a linear trace |
| b odd, read from the plain 67M | near 1.0 | under 0.9: stop (no positive control) |
| a odd, read at the last position | open | under 0.9 noted: the searches read the last position only |
| a + b odd, read from the plain 67M | about 0.50 (E148: 0.468) | 0.55 or more: stop (no built premise) |

## Predictions (filed)

P1 control: H at least 0.5 on both seeds, expected 0.6 to 0.8. P2 plain: H under 0.5 on both, expected under 0.25, G under
0.25. P3 taught (if gated in): H at least 0.5 on both, expected 0.55 to 0.75. P4 outcome rival a. Expected at reference: the
do-nothing fine-tune has gap 0 and a learned ratio near 2, so it reads "no reading" and reaches neither side. Noise floor:
seed sd of H 0.028 (E139, Always on). Blind to: H and G read outputs only; the capacity reads are linear; everything reads
the last position only.

**Outcome rule (card).** Control above on both seeds and plain below on both: rival a. Control below on both: rival b, the
built control is uninformative, and nothing further runs (job 2 and the taught original refuse). Control and plain above:
the wording needs revising. Taught, when it runs, is a second control: below it means rival a is not met on it, reported as
read. Any arm with a seed lacking a reading is "no call". Calling the plain arm "built" stays a rule only the user adds.

## Cost and routing

$0, the MATS cluster's debug QoS, one L40 per job, three jobs of the one run (times from E148's logs: capacity read 115 s,
300 steps 82 s, 3000 steps 810 s, H 550 s, G 1,340 s):
- job 1 (`--comment=E150`): stage 0, alpha, control arm, then the taught original and its gate: ~60 min, ~100 worst;
- job 2 (`--comment='E150 stage: 2'`): plain arm, ~47 min; refuses if the control read below;
- job 3 (`--comment='E150 stage: 3'`): taught arm, ~47 min; exits at once if the taught arm was dropped.
Exit 3 (the 105 min wall limit) resumes from the saved files.

## Overrides of Fable's advice

- Each arm's target is set from its own original (taught original + push), not from the plain 67M (E148's finding).
- The no-teaching control runs first and is the primary positive control; the plain arm runs only after it.
- The taught arm runs only behind the 0.9 gate on its own original, whatever stage 0's read of E148's fine-tune says; the
  taught original is trained toward the full push on another direction (other answers), not a third of the push.
- The sum's parity, not its last digit.

## Not in this plan

A third seed; G on seed 1; a nonlinear capacity read or a nonlinear map in the search; the last digit of the sum; options
B to D of the advice; any change to the wording or the cut-offs.

## Amendment 1, E154 (2026-11-11)

**In plain words.** The first run stopped before any fine-tune, because the plain 67M already reads the sum's parity
linearly at 0.61, over the filed 0.55 bar. The user asked to run it anyway, accepting a weaker control: "i would also be
interested in seeing what happens with the sum-parity experiment, though i accept its not as strong of a control"
(chat, 2026-11-11; `decisions.md`, 2026-11-11, "Built control after E150 stopped"). The rerun (E154, cell P1.12, card
`experiments/built_control_sum/card_p1.12.txt`) changes one number and leaves everything else as filed.

**What changes.** The plain model's capacity bar is 0.9 instead of 0.55: the bar this plan already uses to say an
original holds a variable (the taught original's gate). Arms, the alpha rule, gates, the H and G cut-offs (0.5, 0.8) and
the outcome rule stand. Filed rules: `experiments/built_control_sum/results/predictions_amend1.json`, with the edge
cases rerun against 0.9: a plain read of exactly 0.9 stops, as a taught read of exactly 0.9 passes, so no original can be
both. Code: `experiments/built_control_sum/amend1.py` sets run.py's predictions file and a file prefix, so every file it
writes starts `amend1_` and the first run's `stage0.json` is kept; `cluster/built_control_sum_amend1.sbatch`.

**How to read it.** The plain arm's original is not a clean built case: it holds the sum's parity partly, at 0.61, between
chance and the bar. A moderate plain H is read as partial holding, not as a built skill read as reused, and the 0.61
(stage 0's own value) is reported beside every plain-arm reading (`plot.py amend1`). The taught arm contrasts 0.61 with at
least 0.9: a difference of degree, so the run cannot say at which read between them H crosses 0.5.

**Jobs.** As before (job 1 about 60 min, 100 worst; jobs 2 and 3 about 47 min each), with `--comment=E154`,
`'E154 stage: 2'` and `'E154 stage: 3'`. Jobs 2 and 3 each read only job 1's files and write their own, so they run in
parallel once job 1 exits 0.
