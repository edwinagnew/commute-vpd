# Advice: which teacher-training recipe for E111, and what would change it

*2026-11-06, Fable for `qwen-decomp-lead`. The plan is `reports/plans/2026-11-06_plan_e111-teacher-table.md`; its
appendix has the room table and the sources. This file is the reasoning behind the choice, one paragraph per option.*

## The fact that decides the recipe

E93's table reads any map by six columns, and column 2 (the share of δW left in Δ) is a quantity least squares
minimises: its remainder is orthogonal to everything the subcomponents can write, to the second-order term (E93 read
size within 0.0007 of one minus the share on every state). Any trained change inside the subcomponents adds its own
squared norm to the remainder. Two things follow, before any run. The three share misses (shuffled s0, s70;
own-samples) cannot be repaired by teacher training or by any other in-span map, since they are properties of the table
at update norms of 0.27-0.37, where the null sits within 3% of the share. And on the eight passing states, a trained
correction has a room before it flips the pass: 0.00012 in squared norm on the RL arms, 0.0007 on the template, 0.15 on
TOFU, 0.004-0.007 on the supervised maths fine-tunes. E76's clean-term arm, from zero offsets at rate 1e-6, spent 0.52
after 100 steps. So "teacher training read by E93's table" is, by arithmetic, "least squares plus a small trained
correction", and the recipe question is how to spend the room.

## Options

**A. Clean term from least squares, rate set from the room (the plan).** E76's arm A loss, Adam, started at least
squares' pieces, a correction on E76's 200 often-on columns, rate per state so that 300 steps are expected to use half
the room, kept state the last inside the room. What it can show: whether a correction of that size repairs RL s70's
clean miss (0.0004 nats) and moves column 5; whether the passes survive. What it cannot: repair the share column, or
move Γ's masks ($r$ is below bf16 resolution in the components' readout). Cost about 6.5 h of card, $0. Supported by
E79 (the clean-term arm reads as do-nothing under own masks; the masked-term arm reads worse) and by E43 (a change far
from δW makes the plus-remainder state worse than do-nothing, so the start must be least squares).

**B. Teacher training as the Terms define it, from zero offsets (E76's A, 1200 steps at 1e-6).** The record's recipe,
unchanged. On the table it fails column 2 on every state by construction: E76's A ended with cosine 0.106 to δW and a
squared distance to FT's weights of 4.75 against W0's 3.49, so its remainder is larger than δW itself. Column 1 passes
(0.0066 of W0's K on u1, which least squares already beats at 0.0055). Running it would document a known fail at about
the same cost as A. Not recommended; one state (u1) could be read from E76's saved offsets for $0 if the user wants
the number in the table.

**C. The masked term (E76's B or L) from least squares, with a size stop.** The term that moved the frozen-mask read
most (1.33 to 0.28) at a size the table forbids (3.42×). From least squares with the room as the stop, it would be B at
about step 5. E79 adds that B is worse than A under the state's own masks (0.198 against 0.078), and column 5 reads own
masks. Not recommended as E111; it is what Amendment 1 would be if the masked term were rewritten to read own masks,
with FT weights and the remainder in Δ, the forward column 5 actually computes (new code, 2.2 s per step).

**D. Plain SGD from least squares.** Adam's walk spends the room on noise: each offset moves about rate × √steps
whatever its gradient. SGD moves only along the gradient, so all of ‖$r$‖² is signal, which matters when the room is
1e-4. Against it: `decisions.md:50` fixed Adam for teacher training after SGD at one rate failed the 67M's adversarial
check (rescoped to that rate), and no Qwen run has used SGD. A departure from a working decision, so not the plan's
first choice; it is the first thing to try if stage 1 shows the walk eating the room before K moves.

**E. No teacher training; report E93 as it stands.** Option C of ask 33, which the user closed. Out.

## Recommendation

Run A, with stage 1 (RL s70, in-room and 10× arms) alone first and its exchange rate reported to the user before
stages 2-5; file the three share misses as foreseen; keep the table's verdict rule as E93's so the two maps are
compared like for like, and print the excess share beside column 2. The one point the user must see before stage 2:
column 2 is least squares' floor, so E111 cannot be "teacher training fixes the bits where least squares failed" on
the share; it can be "a trained correction inside the room repairs the clean and masked misses and keeps the passes",
or it cannot, and either reading closes D1's map question on this table.

## What would change the recommendation

- **E76 or E79 re-read.** If E79's own-mask numbers were found to be the frozen-mask ones mislabelled (A 0.078 against
  N 0.066 is the hinge), the masked term returns as a candidate and option C becomes Amendment 1 from the start.
- **E93 Amendment 1 (adjoint column).** If the adjoint split's share sits inside least squares' room on the RL arms
  while its column 5 beats least squares', the correction's direction matters more than its size and option A's
  start should be the scaled adjoint split, not least squares. If the adjoint's share overruns the room, nothing changes.
- **E111 stage 1.** If the in-room arm's walk overruns the room before step 25 at the forecast rate, switch to option
  D for stage 1's rerun. If the 10× arm repairs column 1 but the in-room arm does not move it, the room is the limit:
  stages 2-5 still run (they read whether the passes hold), but the write-up's sentence is "least squares to the room".
- **E43 (67M).** Its plus-remainder state read worse than do-nothing with a change at cosine −0.90 to the teacher's;
  if E111's column 5 worsens on a passing state at ‖$r$‖² of 1e-4, the same mechanism is at work at a thousandth of
  the size, and teacher training is dropped for D1 on Qwen without an amendment.
- **E89 or a later adversary run.** D1 is judged without the adversary (ask 31). If the user reopens the adversary,
  the room argument is unchanged, but the kept state should be chosen by the adversary's read, not by the room alone.
