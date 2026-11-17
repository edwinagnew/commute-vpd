# Plan: the difficulty ladder (E159, cell P1.15)

We fine-tune the plain 67M model to push on the parity of a sum of three numbers, and read its rewiring share beside the
two fine-tunes already read (the last number is odd; the sum of two is odd), each rescaled between a planted edit the map
can reach and one it cannot. Beside each rung, a linear and a small non-linear probe of the plain model ask whether the
model holds the target at all. The run asks whether the share is a graded measure that falls down the ladder where the
non-linear probe falls, or only mirrors what an affine read sees.

## Design check

- **Goal served.** Part 1 of the plan (`reports/plans/2026-11-09_plan_paper-roadmap.md`): a working definition of reused
  against new. The decision card (`reports/decisions/2026-11-11_decision_definition-graded-or-cutoff.md`) found that a
  built control is always linearly unreadable under the current wording, so the open question is whether the share
  measures something graded beyond a probe. This run is that test (user, 2026-11-11, ask 34, option (a)).
- **Nearest tried.** E157 (TRIED.md, P1.14): the sum-of-two fine-tune read 0.07 rescaled at one site against the
  control's 0.47, and 0.59 against 0.70 at two sites. It showed the share separates two rungs, not that it grades, nor
  how it compares with a probe. E150 and E154 (the layout, the control), E156 (the two-site reading).
- **Patch or step.** A step: it decides whether the share earns its place at the 11-13 gate.
- **Decisions relied on.** `decisions.md`, 2026-11-11, "Difficulty ladder": graded reading only, the 0.5 cut-off set
  aside for this cell, no label, no wording change. E157's apparatus as it ran (alpha 4, push direction Q b, five-site
  search, ten pairs).
- **Reads under an adopted definition.** None is adopted (P1.1); no verdict is given, only the graded reading.
- **What changes next.** Rival a (falls where the MLP probe falls) supports a graded wording ("how much of the change is
  reachable from the base model's own computations, by a map of stated size") at the gate. Rival b (flat, out of order,
  or mirrors the linear probe only) is the case for dropping the share at the gate.
- **Control first.** The repeat (E157's planted ceiling rebuilt through the moved code) runs first and alone, then the
  probes with their positive control (last number odd, expected about 1), then the planted ceiling on the new windows at
  in2 (the card's stop under 0.5). Each stops the job on a miss.
- **Numbers and expected values.** Per rung and kind, the rescaled share (odd 0.48 and 0.70, sum of two 0.07 and 0.59,
  from E157; sum of three open, expected lower). Per rung, each probe's best held-out accuracy over four sites (odd about
  1.0 for both; sum of two linear at most 0.61; sum of three linear about 0.5; MLP open). At W0 and do-nothing: no
  reading (gap 0); shuffled-label probes about 0.5. Full list in `results/predictions.json`.
- **Early guards.** Repeat gap 0.230428 within 1e-4 and two-site H 0.9843 within 0.01; odd probes at least 0.9; the
  sum-of-three target's gap above 0.15 (expected about 0.23); each planted gap within 0.005 of it; E139's step-10 check
  in training.
- **Saved.** Every stage's JSON (probe accuracies per site, epochs, nulls; calibration traces; all H values), and the two
  sum-of-three fine-tunes in `results/large/sum3_s{0,1}.pt` (268 MB each).

## Three settled design points

1. **Windows.** E150's windows end in a number pair (`VR.xor2_windows` writes only the last two positions), so the third
   rung needs new windows. `VR.triple_windows` puts a number c, drawn from all 100, before E150's pairs (the read half on
   held-out pairs, checked on the CPU on the real rows). The first two rungs keep their saved readings on E150's windows,
   so only the third rung's ceiling and floor are read again, on the new windows: a planted edit's reading depends on the
   windows, and each rung is rescaled between the planted edits of its own windows.
2. **Probes read the last position.** The share's maps read every position, but there the raw number tokens sit in their
   own places, and a small MLP would compute any parity of them itself and read 1 whatever the model holds. At the last
   position the model must have brought the numbers together. Settings are scikit-learn's `MLPClassifier` defaults (100
   ReLU units, Adam 1e-3, L2 1e-4, batches of 200, up to 200 epochs, its stopping rule); the linear probe is the same with
   no hidden layer. 20,000 keyed fit windows; scored on all 75,000 held-out ones. On a CPU toy (three linear bits), the
   linear probe read their parity at 0.505 and the MLP at 1.0. Noise floor: the shuffled-label null, measured in the job;
   the probe bar is twice its sd, at least 0.05.
3. **Shared code.** E157's planted ceiling and floor and their calibration moved to `src/commute/planted.py`, the probes
   are `src/commute/probes.py`, and the three-number windows and the sum's parity are new functions in
   `src/commute/variable_read.py`. No existing function changed, so no finished script is affected; E157 keeps its own
   copy. E150's `item_h` (fine-tune, learned check, distances, one-site H) and E154's search are called as E157 called
   them.

## Run order and time (one L40, debug QoS, from E157's log, job 58950)

| Stage | Time |
|---|---|
| repeat: E157's ceiling rebuilt, gap and H at in1+in2 | ~2.5 min |
| probes: two layouts, five targets, four sites, linear and MLP, real and shuffled | ~10 min (worst ~13, not measured) |
| sum-of-three target gap | ~15 s |
| planted ceiling, then floor, on the new windows, one site | ~11 min each |
| sum-of-three fine-tune, seeds 0 and 1, one site | ~11.5 min each (+14 at 3000 steps) |
| two sites: ceiling, floor, seed 0, seed 1 | ~21.5 min each |

About 2.4 h in all (2.9 h if both seeds need 3000 steps): two debug jobs, the first ending with exit 3 near 105 minutes.
Cost $0.

## Filed predictions

`experiments/difficulty_ladder/results/predictions.json`. Each step of the ladder reads "fall", "level" or "rise" at a bar
(0.1 on the rescaled share; the probe bar on accuracy). Rival a: the share falls at both steps, in the same steps as the
MLP probe. Rival b: flat or out of order, or the share's steps match the linear probe's and not the MLP's. At one site the
sum of two already sits 0.07 above the floor, so a second fall there needs the sum of three to read below the floor:
"partial" is the expected one-site shape whatever the share measures. Two sites is where the ladder can show a fall at
both steps. The rules were run on CPU stand-ins for each edge case (non-monotone, ties at the bar, a seed not learned,
span under the minimum, missing stages).

## Not run

- The first two rungs on the three-number windows (their saved readings stand).
- A wider, deeper or every-position probe; probes of the fine-tuned models.
- A second planted target for ceiling or floor; more than two seeds; G.
