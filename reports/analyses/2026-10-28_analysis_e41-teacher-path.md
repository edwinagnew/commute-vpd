# Analysis: teacher training of early fine-tunes, and applied step after step (E41)

*2026-10-28. Plan `reports/plans/2026-10-28_plan_e41-teacher-path.md`; code `experiments/teacher_training/path.py`;
predictions `experiments/teacher_training/results/predictions_e41.json`, filed 12:02:14, unedited; results
`experiments/teacher_training/results/teacher_training_e41.json`; log `experiments/teacher_training/results/e41_dl/run_e41.log`;
notebook `notebooks/E41_teacher_path.ipynb`; figures `experiments/teacher_training/figures/e41_*.png` from
`plot_e41.py`. Gap 3 of `reports/reviews/2026-10-28_review_technique-comparison.md`.*

*Independent check: `reports/reviews/2026-10-28_review_e41-check.md`, corrections applied.*

We stopped one fine-tune early and at its end, and trained the subcomponents to imitate each version, from the
unchanged model and in a chain of three. Every copy kept the behaviour with the residual untouched, but at the one
early step where doing nothing fails the check, one copy passed narrowly and the emoticon copy was inconclusive. So the
earliest update is where teacher training is weakest, and the chain stayed within the filed tolerances while its weight
change grew.

## The answer

Partly. We copied one fine-tune, stopped at three points, into the subcomponents with Δ fixed. Every copy kept the
behaviour. The adversarial check can tell a copy from doing nothing only at step 10, where doing nothing fails. There
one copy passed by about 1% and the emoticon copy was inconclusive, between the lines. A chain of three copies stayed
within the filed tolerances of single copies. Its weight change was larger and its adversarial ratio slightly higher.

## What we tested

**Teacher training** is component training of the alive subcomponents' read and write vectors $(U, V)$ with the loss
$\mathrm{KL}(T \,\|\, \text{student})$, where the teacher $T$ is the fine-tuned model; Γ and Δ stay frozen, so Δ cannot
absorb the update. E34's recipe, unchanged: Adam at 1e-5, 1,000 steps, batch-order seed 0. **Task text** is the
fine-tune's own windows with E21's trigger weighting, batch 32. **Pile text** is ordinary Pile training rows from 32,768
onward, batch 8, plain mean KL; run on emoticon only.

**Teachers.** E34's Adam 3e-5 arm-A fine-tune of all 24 matrices, batch 32, seed 0, regenerated in the job. The rate is
constant, so the step-$k$ checkpoint of one 1,000-step run is the fine-tune stopped at $k$: $T_{10}$, $T_{100}$,
$T_{1000}$. Two tasks: emoticon, where an emoticon is followed by `o`, and `r1_t534_v369`, where " which" is followed by
" was".

**States.**
- **One-shot copy** of $T_k$: from $W_0$'s $(U, V)$, 1,000 steps against $T_k$.
- **Chained copy**: from the one-shot copy of $T_{10}$, 1,000 steps against $T_{100}$; from that, 1,000 steps against
  $T_{1000}$. Fresh Adam per link, the same batches as the one-shot copy.
- **Do-nothing** at $T_k$: $\delta W_k$ into Δ, subcomponents and Γ untouched. The reference.

**Numbers per state**, at its last step:
- **Adversarial ratio**: the KL a 20-step mask adversary finds on 256 Pile validation rows, one start, over the same at
  $W_0$ (seed 0). 1 means no change. With this job's $W_0$ floor of 0.0748, pass ≤ 1.396, fail ≥ 1.612, inconclusive
  between. The stochastic-mask ratio is read the same way; `state_verdict` takes the worse of the two, and "below P
  0.9" if the behaviour is lost.
- **Held-out P(target)** at the trigger on held-out windows. $W_0$: 0.004 on emoticon, 0.04 on " which".
- **Gap share**: $\mathrm{KL}(T_k \,\|\, \text{state}) / \mathrm{KL}(T_k \,\|\, W_0)$ on held-out task windows, arm A's
  weighting. $W_0$ reads 1, do-nothing 0.
- **Damage**: the KL from $W_0$ at the held-out windows' non-target positions. **Size**: the largest relative weight
  change of any of the 24 matrices. **Δ's share**: the share of $\|\delta W\|^2$ that Δ carries; ~1e-9 when frozen,
  above 1e-6 is a bug.
- Reported only: **start KL**, the student's loss on its first batch before any update (the same batch for every copy
  of one arm); **P at 100**, held-out P(target) at student step 100; and the **cosine** of the chained and one-shot
  step-1000 weight changes over all 24 matrices.

## Controls

All passed; nothing stopped.

| control | rule | read | verdict |
|---|---|---|---|
| C1, $W_0$ under the Pile harness, adversary seeds 0-2 | `w0_check` passes; floor 0.0748 within 10% | floor 0.07482; PGD 0.699 / 0.751 / 0.707 | pass |
| G1, regenerated $T_{1000}$ | P within 0.005, size within 0.001 of E34 | emoticon 0.99961, 0.03209; " which" 0.99621, 0.03334 | pass ×2, to the digit |
| G3, one-shot task-text copy of $T_{1000}$ | P within 0.005, ratio within 15% of E34 | emoticon 0.99675, 0.9591; " which" 0.99443, 0.9676 | pass ×2, to the digit |
| C2, warm start from the emoticon $T_{10}$ copy, one step at lr 0 against its own weights | step-1 KL ≤ 0.005; δW within 1e-3 | KL 0.0005, δW change 0 | pass |
| Δ untouched, every trained state | Δ's share ≤ 1e-6 | ≤ 2.3e-9 | pass |
| Positive control: do-nothing at emoticon $T_{10}$ | expected 2.52, fail (E22) | 2.516, fail | reproduced |

E22's other two emoticon do-nothing readings also reproduce: 1.136 at $T_{100}$ (E22 1.14) and 1.499 at $T_{1000}$
(1.50); E34's " which" $T_{1000}$ do-nothing 1.522 (1.52).

### The teachers and their do-nothing states

| task | step | P(target) | damage | size | gap at $W_0$, nats | do-nothing ratio | do-nothing stochastic | do-nothing verdict |
|---|---|---|---|---|---|---|---|---|
| emoticon | 10 | 0.9993 | 0.0554 | 0.0113 | 5.078 | 2.516 | 1.003 | fail |
| emoticon | 100 | 0.9995 | 0.0456 | 0.0222 | 5.077 | 1.136 | 0.956 | pass |
| emoticon | 1000 | 0.9996 | 0.0547 | 0.0321 | 5.090 | 1.499 | 0.916 | inconclusive |
| " which" | 10 | 0.9890 | 0.1210 | 0.0133 | 5.616 | 2.164 | 1.009 | fail |
| " which" | 100 | 0.9970 | 0.0635 | 0.0221 | 5.622 | 1.367 | 0.929 | pass |
| " which" | 1000 | 0.9962 | 0.0762 | 0.0333 | 5.622 | 1.522 | 0.892 | inconclusive |

Every teacher holds P ≥ 0.9, so every early teacher stays in the scored predictions. The check sees both step-10 updates and neither step-100
update: at step 100 do-nothing already passes, so the step-100 cells of the first prediction cannot tell teacher training from the trivial
map. Their P and gap-share readings still show the copy learnt the teacher. At step 1,000 do-nothing is only
inconclusive, so only the step-10 cells separate a map from doing nothing on the check.

## Does a copy of an early fine-tune pass where doing nothing fails?

![adversarial ratio against teacher step](../../experiments/teacher_training/figures/e41_faithful.png)

On the " which" task it passes, by 1.4%, well inside the noise; on emoticon it is inconclusive. At step 10 do-nothing
fails on both tasks, at 2.52 and 2.16. The one-shot task-text copy reads 1.376 on " which", 1.4% under the 1.396 line,
against an 11% batch-order spread and a 7.5% adversary-seed floor. On emoticon it reads 1.547: 11% above the pass line
and 4.2% below the 1.612 fail line. At step 100 every copy passes, but so does do-nothing. At step 1,000 every copy
sits at 0.96 to 1.05, where do-nothing is inconclusive on both tasks. The Pile-text copy of emoticon $T_{10}$ passes at
1.13, so the emoticon miss belongs to the task-text copy, not to the teacher alone.

The adjoint split of the same emoticon step-10 update passed at 1.06 with P 0.93 and Δ fixed in E22. At this step the
closed-form map did better on the check than the task-text copy. It loses the behaviour later in the fine-tune: P
0.60 at step 100 and 0.10 at step 1,000, against teacher training's ≥ 0.997 at every step.

### All states

| task | text | teacher | start | P(target) | P at 100 | adversarial | stochastic | verdict | gap share | damage | size | start KL | Δ share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| emoticon | task | 10 | $W_0$ | 0.9994 | 0.999 | **1.547** | 0.981 | **inconclusive** | 0.0013 | 0.0577 | 0.0141 | 5.427 | 2.3e-09 |
| emoticon | task | 100 | $W_0$ | 0.9989 | 0.997 | 1.016 | 0.947 | pass | 0.0014 | 0.0363 | 0.0168 | 5.331 | 1.5e-09 |
| emoticon | task | 1000 | $W_0$ | 0.9968 | 0.994 | 0.959 | 0.952 | pass | 0.0056 | 0.0180 | 0.0184 | 5.342 | 1.3e-09 |
| emoticon | task | 100 | chain | 0.9982 | 0.997 | 1.054 | 0.942 | pass | 0.0019 | 0.0460 | 0.0214 | 0.130 | 9.8e-10 |
| emoticon | task | 1000 | chain | 0.9979 | 1.000 | 1.000 | 0.938 | pass | 0.0033 | 0.0269 | 0.0251 | 0.065 | 6.7e-10 |
| emoticon | Pile | 10 | $W_0$ | 0.9984 | 0.989 | 1.134 | 0.997 | pass | 0.0038 | 0.0616 | 0.0166 | 0.020 | 2.2e-09 |
| emoticon | Pile | 100 | $W_0$ | 0.9954 | 0.942 | 0.954 | 0.950 | pass | 0.0049 | 0.0552 | 0.0184 | 0.024 | 1.8e-09 |
| emoticon | Pile | 1000 | $W_0$ | 0.9021 | 0.339 | 1.049 | 0.904 | pass | 0.0266 | 0.0580 | 0.0201 | 0.056 | 1.2e-09 |
| emoticon | Pile | 100 | chain | 0.9965 | 0.997 | 0.967 | 0.956 | pass | 0.0043 | 0.0480 | 0.0205 | 0.050 | 1.5e-09 |
| emoticon | Pile | 1000 | chain | 0.9030 | 0.948 | 1.027 | 0.908 | pass | 0.0251 | 0.0609 | 0.0244 | 0.041 | 8.6e-10 |
| " which" | task | 10 | $W_0$ | 0.9892 | 0.991 | 1.376 | 0.975 | pass | 0.0010 | 0.1229 | 0.0142 | 5.023 | 2.1e-09 |
| " which" | task | 100 | $W_0$ | 0.9984 | 0.996 | 1.002 | 0.924 | pass | 0.0031 | 0.0583 | 0.0194 | 4.997 | 1.1e-09 |
| " which" | task | 1000 | $W_0$ | 0.9944 | 0.993 | 0.968 | 0.929 | pass | 0.0039 | 0.0422 | 0.0209 | 4.964 | 1.1e-09 |
| " which" | task | 100 | chain | 0.9988 | 0.997 | 1.044 | 0.916 | pass | 0.0033 | 0.0625 | 0.0215 | 0.171 | 8.9e-10 |
| " which" | task | 1000 | chain | 0.9911 | 0.994 | 1.011 | 0.916 | pass | 0.0030 | 0.0502 | 0.0252 | 0.062 | 6.8e-10 |

The start KL of a Pile-text copy is read on a Pile batch, where $W_0$ is already close to every teacher, so it is not
comparable with the task-text column.

### The one miss, read two ways

The first prediction asked every early task-text copy to pass with gap share ≤ 0.05. The emoticon $T_{10}$ copy has gap share 0.0013
and P 0.9994, but its adversarial ratio is 1.547, inconclusive, so it does not "work" by the filed rule. The score's
`check_sees_the_update` field marks this cell `true`: its do-nothing state reads above the pass band, 2.516, fail.
Of the first prediction's four cells, only the two step-10 cells have power. They read one pass by 1.4% and one
inconclusive. A high-lean prediction failed in the cell where the check had full power.

Both readings hold, and neither is re-scored here.
- **Against the line**: on the smallest and most fragile update we gave it, the map does not reach a pass, on one of
  the two tasks. The other step-10 copy passes by 0.019.
- **Against doing nothing**: the copy removes $(2.516 - 1.547)/(2.516 - 1) = 0.64$ of the excess the adversary finds
  when the same update sits in Δ. The " which" $T_{10}$ copy removes 0.68 of its do-nothing excess. Both step-10
  copies keep the behaviour and sit closer to their teacher than any later copy, gap share 0.0013 and 0.0010.

So for small updates: the emoticon copy is inconclusive at 1.547, 11% above the pass line and 4.2% below the fail
line. It removes about two-thirds of do-nothing's excess over $W_0$, with its behaviour intact. The 4.2% to the fail
line is inside the 7.5% adversary-seed floor, and its excess sits in one 128-row batch, below. A rerun could land in
pass, inconclusive or fail. To pass it would need a 9.8% drop, about the largest batch-order spread of a task-text
copy's ratio seen in E34 Amendment 4, "up to 11%".

Two further readings from the log, not scored:
- The harness reads the 256 rows as two 128-row batches, each with its own adversary. For the emoticon $T_{10}$ copy
  the batches read 0.722 and 1.440 against $W_0$'s 0.679 and 0.718, ratios 1.06 and 2.00. Do-nothing at $T_{10}$
  reads 1.17 and 3.79. The excess sits in the same batch for both. A ratio of 2.00 on that batch alone would read fail;
  $W_0$'s second batch spreads 20% over adversary seeds 0-2, 0.718 / 0.828 / 0.690, far below it. For the " which"
  $T_{10}$ copy the excess is spread over both batches, 1.37 and 1.38.
- Both step-10 task-text copies reproduce their teacher's damage elsewhere in full, 1.04 and 1.02 times it, and are
  larger than the teacher, 1.25 and 1.07 times its size. Copies of $T_{1000}$ do 0.33 and 0.55 of their teacher's
  damage at 0.57 and 0.63 of its size. The emoticon $T_{10}$ Pile-text copy also does 1.11 times the teacher's damage
  and passes at 1.13, so damage alone does not account for the miss.

## Do copies of early fine-tunes keep the behaviour?

![P(target) and gap share against teacher step](../../experiments/teacher_training/figures/e41_behaviour.png)

Yes, every copy. Every copy holds P ≥ 0.90 at the trigger against 0.004 and 0.04 for $W_0$. Every copy's gap share is
below 0.027, and every early copy's below 0.005. The Pile-text copies did better on early teachers than on the
finished one: P 0.998 and 0.995 at $T_{10}$ and $T_{100}$, against 0.902 at $T_{1000}$, and gap share 0.004 and 0.005
against 0.027. At student step 100 the Pile-text copy of $T_{10}$ already holds 0.989; of $T_{1000}$, 0.339. The
$T_{1000}$ Pile-text copy reproduces E34's: P 0.902, ratio 1.05.

## Does a chain of copies along the fine-tune pile up error?

![chained against one-shot: cosine and size](../../experiments/teacher_training/figures/e41_chain.png)

Not on any scored number, over a chain of three checkpoints, steps 10, 100 and 1,000, with 1,000 student steps per
copy. Every chained copy passes, keeps P ≥ 0.90 and leaves Δ alone, and all six filed comparisons hold. But on task
text the chained adversarial ratio is 4-5% higher in all four cells, and two unfiled quantities grow from the second
link to the third in every group.

| task | text | teacher | chain ratio ÷ one-shot ratio (≤ 1.25) | chain P − one-shot P (≥ −0.01 task, −0.05 Pile) | chain gap share − one-shot (≤ +0.005 task, +0.02 Pile) | both verdicts | chain size ÷ one-shot size | holds |
|---|---|---|---|---|---|---|---|---|
| emoticon | task | 100 | 1.038 | −0.0008 | +0.0005 | pass, pass | 1.27 | yes |
| emoticon | task | 1000 | 1.043 | +0.0011 | −0.0023 | pass, pass | 1.36 | yes |
| " which" | task | 100 | 1.042 | +0.0005 | +0.0002 | pass, pass | 1.11 | yes |
| " which" | task | 1000 | 1.045 | −0.0033 | −0.0009 | pass, pass | 1.20 | yes |
| emoticon | Pile | 100 | 1.013 | +0.0011 | −0.0006 | pass, pass | 1.12 | yes |
| emoticon | Pile | 1000 | 0.979 | +0.0009 | −0.0015 | pass, pass | 1.21 | yes |

- **Same sign on task text.** The chained ratio is 4% to 5% above the one-shot's in all four task-text cells. That is
  inside the 25% tolerance, the 11% batch-order spread and the 7.5% adversary-seed floor, but it is one sign four
  times. On Pile text it is +1% and −2%.
- **Different states.** Cosine of the chained and one-shot step-1000 weight changes: 0.750 and 0.858 from task text,
  0.653 from Pile text, against 0.913 and 0.429 between two batch orders of E34's copies of another teacher, the
  " which" SGD fine-tune (E34 Amendment 5); a random direction 2e-4.
- **Larger, and growing link by link** (not filed). The chained change is 1.20 to 1.36 times the one-shot copy's size
  at step 1000, still below the fine-tune's own 0.032-0.033. Both its size and its damage elsewhere, relative to the
  one-shot copy's, grow from the second link to the third in all three groups:

  | group | size ÷ one-shot, T100 → T1000 | damage ÷ one-shot, T100 → T1000 |
  |---|---|---|
  | emoticon, task text | 1.27 → 1.36 | 1.27 → 1.49 |
  | " which", task text | 1.11 → 1.20 | 1.07 → 1.19 |
  | emoticon, Pile text | 1.12 → 1.21 | 0.87 → 1.05 |

  That is the pattern accumulation would show, in quantities nobody filed; two links cannot tell a trend from noise,
  and it is untested. The short-links branch should file a prediction on it.
- **Gap share at step 1000 is lower for the chain**: 0.0033 against 0.0056 on emoticon, 0.0030 against 0.0039 on
  " which", 0.0251 against 0.0266 on Pile text.
- **Head start.** On task text the first link starts 0.13 and 0.17 nats from $T_{100}$, against 5.3 and 5.0 from
  $W_0$; the second link 0.065 and 0.062 from $T_{1000}$. On Pile text the chain's P at student step 100 against
  $T_{1000}$ is 0.948, against the one-shot copy's 0.339.

## What we learned

Teacher training keeps the behaviour of every early teacher with Δ untouched, from task text on two tasks and from Pile
text on emoticon. On the adversarial check, at the one step where doing nothing fails, one step-10 copy passed by 1.4%
and the other was inconclusive. The step-10 task-text copies read the highest ratios of any trained state, so the small
updates `GOALS.md` asks about are where this map is weakest, on one seed, one batch order and one adversary start.
Along this fine-tune the adjoint split keeps Δ fixed and passes, but loses behaviour after step 10 (E22). At step 10,
though, it passed at 1.06 where the task-text copy was inconclusive. A chain of three copies stayed within the filed
tolerances of single copies on every scored number. Its size and damage elsewhere grew from link to link, which is
what the short-links run must test.

## How compelling is the evidence

- **One draw of everything**: one fine-tune seed, one batch order, one 20-step adversary start on 256 Pile rows. E34
  Amendment 5 showed a second start can read a fine-tune left in Δ six times higher, so every pass is a lower bound
  (`decisions.md`, 2026-10-27: one start kept).
- **Two trigger tasks, one Adam fine-tune each**, both behaviours the model half has. Pile-text copies on emoticon only.
- **Power**: the check sees both step-10 updates (do-nothing fails) and neither step-100 update (do-nothing passes). The
  step-100 cells of the first prediction hold but carry no information about the map.
- **Noise floor** (E34 Amendment 4, a second batch order, H100): ratio up to 11% on task text and 25% on Pile text; P
  0.005 and 0.032; gap share 0.002 and 0.010. $W_0$'s adversary-seed floor here 7.5%.
- **The chain had more training**: 2,000 or 3,000 student steps in total against the one-shot's 1,000. Its adversarial
  ratio was still 4-5% higher on task text, while its gap share at step 1000 was lower. The extra steps also grew its
  change and its damage elsewhere.
- **The cosine references** come from another teacher and count directions that carry no behaviour; they bound how
  alike the total changes are, not where the behaviour sits.

## Surprises

- **The first prediction failed as filed, on one cell, at a high lean.** We expected every early task-text copy to work. The emoticon
  $T_{10}$ copy read 1.547, inconclusive, with its behaviour and gap intact. It failed in the only cell where the check
  had full power, 4.2% under the fail line, inside the adversary-seed floor; a rerun could read pass, inconclusive or
  fail. The plan's decision tree sends it to a batch-order rerun.
- **Early fine-tunes are easier to copy from Pile text than the finished one**: 0.998 and 0.995 against 0.902, where
  the filed lean was low.
- **Step-10 copies are larger than their teachers** and reproduce their damage in full; copies of the finished
  fine-tune are smaller and do a third to a half of it. Not filed.
- **The chained change grows** by 11% to 36% over the one-shot copy's, more at the third link than at the second in all
  three groups, and so does its damage elsewhere. Its task-text ratios sit 4% to 5% above the one-shot's in all four
  cells. Not filed; each within noise, but one sign every time.
- **The adjoint split did better at step 10** on the check: 1.06 with P 0.93 in E22, where the task-text copy of the same
  update read 1.547.

## What next

1. **The emoticon $T_{10}$ task-text copy at batch orders 1 and 2** (E41 Amendment 1; approved branch, Amendment 4's
   recipe, about $1, H100). The plan's branch reads "if its verdict fails with the behaviour kept"; this cell's verdict
   is inconclusive, and the amendment reads "fails" as "not pass", the first prediction's verdict condition. The
   behaviour is kept, so the 3,000-step branch does not apply. Its predictions are filed first and must cover every
   outcome: pass, inconclusive or fail at each batch order, and a mixed result. Either way a Blocking card on early
   updates and the weak version goes to the user, as the plan's tree says.
2. **Short chain links** (approved branch, "Amendment 2 of E41", about $1.5): emoticon, task text, a chain over the
   fine-tune's checkpoints 1, 3, 10, 30, 100, 300, 1000 with 100 student steps per link, read at every link against
   this run's one-shot copies. Predictions filed first, including one on the size and damage growth seen here. If every
   link passes, the map can run alongside training; if the ratio, the size or the damage grows link by link, it needs a
   reset.

## Scorecard

Rules as filed in `predictions_e41.json`; the run's `score` block, checked here by hand against the table above.

| id | claim | rule | read | verdict |
|---|---|---|---|---|
| C1 | $W_0$ reads its known floor | `w0_check`; floor 0.0748 within 10% | 0.07482 | pass |
| G1 | regenerated $T_{1000}$ = E34's | P within 0.005, size within 0.001 | 0.99961 / 0.03209; 0.99621 / 0.03334 | pass ×2 |
| G3 | one-shot $T_{1000}$ task-text copy = E34's | P within 0.005, ratio within 15% | 0.99675 / 0.9591; 0.99443 / 0.9676 | pass ×2 |
| C2 | warm start reproduces its start | step-1 KL ≤ 0.005; δW within 1e-3 | 0.0005; 0 | pass |
| — | Δ untouched in every trained state | Δ's share ≤ 1e-6 | ≤ 2.3e-9 | pass |
| P1 | early teachers copied from task text work | every cell (task × $T_{10}$, $T_{100}$): verdict pass and gap share ≤ 0.05 | emoticon $T_{10}$ 1.547 inconclusive, gap 0.0013; emoticon $T_{100}$ 1.016; " which" $T_{10}$ 1.376; " which" $T_{100}$ 1.002; all gaps ≤ 0.0031. Check sees the update: $T_{10}$ cells yes, $T_{100}$ cells no | **fails as filed** (1 of 4 cells; lean high). Of the two cells with power, one passes by 1.4% and one is inconclusive |
| P2 | early teachers copied from Pile text, emoticon | learnt if works; neither if P < 0.5; else between | $T_{10}$ P 0.9984, 1.134 pass, gap 0.0038; $T_{100}$ P 0.9954, 0.954 pass, gap 0.0049 | **learnt** (lean learnt, low) |
| P3 | chain not worse, task text | at $T_{100}$, $T_{1000}$ per task: pass where one-shot passes; ratio ≤ 1.25×; P ≥ −0.01; gap ≤ +0.005 | 4 of 4 cells, table above | **holds** (lean high) |
| P4 | chain not worse, Pile text, emoticon | as P3 with 1.25×, −0.05, +0.02 | 2 of 2 cells | **holds** (lean medium) |

## Not run

- The emoticon $T_{10}$ task-text copy at batch orders 1 and 2, the approved branch for the first prediction's miss.
- Short chain links and finer chains, the approved branch for the chain predictions holding; a one-shot copy trained 3,000 steps
  to match the chain's total.
- SGD or LoRA teachers; r0 and the cipher tasks; Pile-text copies on the " which" task.
- A second batch order or fine-tune seed; more adversary starts, a longer adversary, the budget adversary.
- The adjoint split of the early teachers (E22 read it); the split of the gain at the trigger into its parts.
- No states were saved; any follow-up regenerates one deterministically in about 3 min.

## Provenance

- RunPod NVIDIA H100 80GB HBM3; job 12:15:12 to 12:58:25 on 2026-10-28, 43.2 min of job, about $3.2 in all. Pod time
  and pod ID are not recorded in the repo. Results committed in a450fde.
- `teacher_training_e41.json` sha256 `aeb17705e38a47409981b4165e7bd8bd17f2a9e0118cf1b64a779414a76747a7`.
- `predictions_e41.json` sha256 `9e442abc41e89b673e1e85d0d345886a520a4fff589cb8b22d28a1cd9d95cf6d`.
- `e41_dl/run_e41.log` sha256 `9085946bb2b9ed50ae2201ed539306bf7eb7e8cf9d6216846dd19b4be3024b83`.
- Figures: `uv run python experiments/teacher_training/plot_e41.py`, from the results JSON only.
- Independent check: `reports/reviews/2026-10-28_review_e41-check.md`, corrections applied.
