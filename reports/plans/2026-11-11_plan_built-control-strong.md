# E153: the built control on XOR2, with an original that holds the skill strongly

*2026-11-11. Cell P1.11 (1 run, cluster debug, $0, due 11-12). Approved by the user 2026-11-11. Card:
`experiments/built_control_strong/card.txt`. Filed rules: `experiments/built_control_strong/results/predictions.json`.
Code: `experiments/built_control_strong/run.py`, `cluster/built_control_strong.sbatch`. Nothing has run.*

**In plain words.** The earlier run on two-key XOR found no map on either arm. Its "taught" original held the skill too
weakly to count as a positive control: a linear read of the skill from it scored only 0.588. One of that run's own
fine-tunes, saved on disk, holds the skill almost exactly (a linear read of 0.9997), so this run uses it as the original
and fine-tunes it further toward its own target. If the search now finds a map above 0.5, a built skill is detectable as a
contrast with the earlier plain arm. If it still finds none, the search cannot certify reuse of this skill, and the
wording needs revising.

## What runs

**The original, M′.** The earlier run's plain seed-0 fine-tune: the 67M trained for 300 steps toward the 67M plus a push
at block 2's input, last position, on windows whose two final keys have different bits. Saved at
`experiments/built_control/results/large/plain_s0.pt`. A ridge read of the bit from it scored 0.9997 at in2 and in3 on
held-out key pairs (`experiments/built_control_sum/results/stage0.json`, key `e148_plain_s0_xor2`).

**The target, F′.** M′ plus the same push (6 times E137's Q b, the earlier run's chosen size), on the same windows. Keys,
windows, held-out pairs, recipe, learned check and searches are all the earlier run's, unchanged. Each seed's batch draws
reuse the earlier plain arm's seed names, so the strong arm and the plain arm differ only in the original.

**The built side** is the earlier plain arm, already read: best found H 0.080 and 0.087
(`experiments/built_control/results/readings.json`). It does not rerun.

**The confound: the push applied twice.** M′ was trained toward the 67M plus the push, and moved further than that
target (its distance from the 67M is 0.344, the target's 0.259). F′ adds the push again, further out along the same
direction. The earlier calibration shows the plain gap flattening as the push grows (0.258 at size 6, 0.294 at 12). If
the second push lands where outputs saturate, the target gap is small, the fine-tune can match it with a small move, and
H rests on a small denominator. Stage 1 reads the gap, stops under 0.15, and reports beside it the gap a perfect M′ would
leave, the distance between the 67M with two pushes and with one. The predictions' blind-to line names this.

## Stages, cheapest first, one job

| stage | check | expected | else |
|---|---|---|---|
| 0 | zero-step fine-tune of M′ (W0), against M′ | gap 0, max TV under the float floor measured on M′ | stop |
| 0 | ridge read of the XOR2 bit from M′, held-out pairs | 0.9997 (repeat of E150's read) | under 0.9: stop |
| 1 | target gap, TV(F′, M′) | open, 0.1 to 0.3 | not above 0.15: stop; "saturates" if a perfect M′ would also leave under 0.15 |
| 2 | each fine-tune at step 10 | closer to F′ than M′ is, no overshoot | stop |
| 2 | learned check (distance to F′ over that of M′ plus half the push) | under 0.9 at 300 steps (plain arm read 0.15) | seed 0: 3000 steps; then no reading |
| 2 | H on seeds 0 and 1, G on seed 0 | P1, P2 below | |

## Predictions (filed)

- **P1.** Strong arm: best found H at least 0.5 on both seeds, expected 0.55 to 0.8, best site in2 or in3. *Lean: medium*:
  the push lands at in2, where M′ holds the bit linearly, but no fine-tune of a computed variable has yet read above 0.5.
- **P2.** G on seed 0, open, 0.3 to 1.0; labelled by `item_label`.
- **P3.** Outcome rival a.

At reference: the zero-step fine-tune is M′ (gap 0); a do-nothing fine-tune has gap 0 and a learned ratio above 1, so it
has no reading and reaches neither side. Noise floor: seed spread of H 0.028 (E139, Always on). Blind to: H and G read
outputs only; the ridge read is linear; the push applied twice.

**Outcome rule (card).** H at least 0.5 on both seeds: rival a, built detectable as a contrast with the plain arm. Under
0.5 on both: rival b, and by the user's condition of 2026-11-10 the wording needs revising. Seeds split, or a seed with no
reading: no call, reported as read. Exactly 0.5 counts as above. Calling the plain arm "built" stays a rule only the user
adds.

## Cost and routing

$0, one L40 on the cluster's debug QoS, one job: `sbatch --comment=E153 cluster/built_control_strong.sbatch`. From the
earlier run's logs: set-up 30 s, ridge read 120 s, 300 steps with its check 110 s, H 590 s, G at four sites 1,225 s.
About 52 minutes; about 60 if seed 0 needs 3000 steps (G then at in2 only). The 105-minute wall limit exits 3 with
everything read saved, and the same command resumes.

## Not in this plan

A rerun of the plain arm; a third seed; G on seed 1; a nonlinear read or a nonlinear map in the search; a smaller second
push to step out of saturation; any change to the wording or the cut-offs.
