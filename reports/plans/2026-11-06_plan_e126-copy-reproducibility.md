# Plan: is E119's low emoticon Pile-text copy a card difference or a batch-order outlier? (E126)

## The question, in plain words
E119 rebuilt E34's teacher-trained decompositions on an L40. One of twelve, the emoticon task's Pile-text copy of the
Adam fine-tune, reached held-out P(target) 0.744, against E34's 0.902 on the H100 (E34 Amendment 4 moved it to 0.934
at a second batch order, the largest order change of its twelve). We rebuild that copy on the L40 at batch orders 0
(E119's, the control), 1 and 2, and the three Pile-text copies E119 labelled re-switching (r0 Adam, r0 SGD, r1 SGD) at
orders 1 and 2, reading each copy's route split and its label under E119's lines. Not an E119 amendment: a
reproduction check under E119's own code (Fable, `fable-strategy/advice.md` section 3 objection 4 and section 5 item 4).

## Design check
- Goal: whether teacher training's copies are reproducible, which bounds every claim built on E34's Pile-text arm.
- Nearest rows: E34 Amendment 4 (order 1 on the H100, no route split); E119 (order 0 on the L40). This adds L40 orders.
- Step, not patch: no new rule; labels use `commute.route_labels` (E119's lines, frozen). Not a reuse verdict, so
  `decisions.md` rows 81/87 do not apply.
- Approved class: a $0 cluster check inside Fable's adopted plan; FYI.

## States and settings
All E119's (`experiments/teacher_route_split/run.py`; config in its `predictions.json`): teacher = E31's arm-A
fine-tune, seed 0, 1,000 steps (run.py:177-185); copy = `teacher.train_student`, Adam 1e-5, batch 8 of Pile train
rows 32,768-40,959, 1,000 steps, no trigger weighting (run.py:155, 205-208); read = `route_split` on the copy's own
state (run.py:158-163, 210-213); label teacher gain = the teacher's do-nothing gain (run.py:215-218). The only change:
`seed=order` in `train_student`, the batch order and nothing else (E34 Amendment 4 used the same seed path).

## Stage 0: controls (first; any miss stops the job)
| check | must read |
|---|---|
| card | NVIDIA L40 (E119's) |
| each teacher: P(target), do-nothing gain | within 0.005 of E119's (abs; relative for gain). Same card and seed; E119's do-nothing r equalled E31's to 0 |
| emoticon Adam copy, order 0, run before any other order | P(target) 0.7442 within 0.005 |

## Predictions (filed in `experiments/copy_reproducibility/results/predictions.json`)
- **P1, emoticon Adam copy, orders 1 and 2.** Both at least 0.852 (E34's 0.902 minus its p_tol 0.05): order 0 is an
  outlier on this card. Both at most 0.794 (0.744 plus 0.05): a card difference. Otherwise unstable. Spread over the
  three orders reported. Expected: outlier, P about 0.90-0.93. *Lean medium*: the emoticon teachers, both emoticon
  task-text copies and its SGD Pile-text copy all reproduce E34 on the L40 within 0.002.
- **P2, the three re-switching copies.** A cell holds if orders 1 and 2 both read re-switching. Expected 2 of 3: r0
  SGD holds (*high*); r0 Adam holds (*medium*, gain ratio 0.58 against the 0.5 readability line); r1 SGD a coin flip
  (*low*, e_gate 0.237 against the 0.25 line). Noise floor: E12 Amendment 1's copy table moved r by 0.0004 at a second
  order.
- Blind to: a card effect that depends on the order; orders past 2; the H100 at order 2.

## Outcomes
- Outlier: E34's reproducibility claim stands, with the L40 three-order spread stated beside E34's 0.032.
- Card difference: a copy's end state depends on the card; E34's Pile-text numbers carry a card caveat.
- Unstable: this Pile-text cell's E34 reading carries the spread; task-text copies stay the cited method.
- P2 breaks: E119's re-switching corner is reported with the labels per order; no new rule.

## Cost and routing
E119's L40 log (`results-large/teacher_route_split/run_emoticon.log`): model load 25 s, teacher ~30 s, Pile-text
copies 154-167 s each (the 110-135 s in the brief are the task-text copies). Four teachers and nine copies: ~26 min of
compute, over one 30-min `debug` job with start-up. The job stops at 26 min (`budget_s` 1560, E121's) with exit 3 and
`fleet/resubmit --code 3` runs the rest: about two jobs, the emoticon cell and most of the rest in the first. $0.
Needs `experiments/copy_reproducibility/results/large` linked to `~/commute-data/results-large/copy_reproducibility`
(`fleet/link-data regating-lead` after pulling); nine offset files of ~60 MB and `run.log` go there.

## Not run
H100 at orders 1-2 for the corner cells; task-text copies at new orders; orders past 2; the least-squares split of the
regenerated teachers (E119 has it, and P2 does not read it).
