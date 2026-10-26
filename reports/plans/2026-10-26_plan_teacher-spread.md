# Plan: teacher training of a fine-tune with no trigger (E37)

*2026-10-26, design agent for the lead. Option 1 of `reports/decisions/2026-10-26_decision_next-while-waiting.md`, which
the user confirmed. Changed since that report: the teacher is E36's lowercase fine-tune (E36 Amendment 1), because the
code-like fine-tune did not learn. Code: `experiments/teacher_spread/run.py`, `src/commute/row_copy.py`,
`cluster/teacher_spread.sbatch`. Predictions are in `run.py` and not yet filed. Words as in `GOALS.md` → Terms.*

**In plain words.** Every teacher-training copy so far copied a fine-tune whose whole effect sits at one trigger
token. This run copies a fine-tune that changes predictions at every position: lowercased Pile text. It asks whether
the copy keeps that behaviour, stays faithful, and lands on the same weight change when the batch order changes.

## The request, restated in the Terms' words

Take E36's lowercase fine-tune as the teacher, W₀ + δW with all 24 matrices updated. Run **teacher training**:
component training of the alive (U, V) with the loss KL(teacher ‖ student), Γ and Δ frozen, on two texts. One copy
trains on the teacher's own lowercase training rows, the other on the same Pile documents left as they are. Compare
both with the **do-nothing** state and the **scaled adjoint split** of δW. Score whether each keeps the teacher's
held-out gain and passes the Pile harness. Then train the lowercase copy at a second batch order and read the cosine
between the two weight changes.

## Design check

- **Goal served.** Q1, the weak version under reading A, for an update that is not a single association: the
  rehearsal of the larger-model protocol (Q9) on the existing model.
- **Nearest thing tried.** E34 and its four amendments: teacher training copies six full-weight fine-tunes and three
  LoRAs on three trigger tasks, faithful at two batch orders and under a 160-step adversary. E34's "What next" item 3
  and `backlog.md` name a fine-tune without a single trigger as the step before the larger model. E36 Amendment 1
  supplies one that learns: a held-out gain of 0.130 nats per token at step 300.
- **Patch or step.** A step: it changes what the update is, not how the copy is scored.
- **Working decisions.** Γ frozen and Δ fixed, reading A, teacher training as the commute line's working map. Departs
  from none. Lowercase Pile text is E36's task, approved for the re-gating line; no new task family.
- **What result changes what we do.** See the decision tree below.
- **Control first.** The W₀ check, then the regenerated teacher's held-out gain, before any copy trains.
- **Class.** 4: E34's named next step, on E36's task. The user confirmed the option.

## States and numbers

Six states, each read on 64 held-out lowercase rows and 64 held-out plain Pile rows of 512 tokens, every position:
**W₀**; **do-nothing**, the teacher itself; the **adjoint** split, with input moments from 256 of the teacher's
training rows, as E34 ran it; **lower_s0**, the copy from the 2,400 lowercase training rows; **plain_s0**, the copy
from the same 2,400 Pile rows not lowercased; **lower_s1**, lower_s0 at batch order 1. The copies use E34's Pile-text
recipe: Adam 1e-5, eight rows per step, plain mean KL, 1,000 steps.

| number | definition | W₀ | do-nothing | copy, expected |
|---|---|---|---|---|
| 1. behaviour kept *s* | (L_state − L_W₀) / (L_teacher − L_W₀), L the mean next-token log-probability on held-out lowercase rows | 0 | 1 by construction | ≥ 0.9; adjoint below 0.5 |
| 2. teacher gap share | KL(teacher ‖ state) on those rows, over W₀'s, about 0.3 nats | 1 | 0 by construction | ≤ 0.10 |
| 3. adversarial ratio | the Pile harness: 256 val rows, 20-step adversary, ratio to the same job's W₀ | 1, floor 0.075 | lean above the line | pass, ≤ 1.40 on an H100 |
| 4. damage elsewhere | KL(W₀ ‖ state) on held-out plain Pile rows | 0 | the teacher's own | 0.5 to 2 times the teacher's |
| 5. reported | size; cosine with δW against a random direction's; cosine between lower_s0 and lower_s1 | 0 | 1 | 0.08 to 0.3 with δW, as E34 |
| 6. budget adversary, reported | E20 Amendment 3's adversary: 20 sign steps of 0.1 from seed 0 on the first 128 Pile val rows, then the KL under the masks found, as a ratio to W₀'s; not at lower_s1 | 1 | lean above 1.24, as E34's fine-tunes read 1.7 to 9.2 | at or near 1, as E34's copies |

Number 3 is also read against the do-nothing state's ratio, for the pending anchor decision; that reading is reported,
not scored. Number 6 keeps E37 readable if E34 Amendment 5 finds the 20-step harness lenient on fine-tunes;
no prediction rests on it. The do-nothing state reads 1 and 0 on numbers 1 and 2 exactly; any other value is a readout bug.

**Uniqueness.** The cosine between lower_s0's and lower_s1's weight changes, both rebuilt from their saved bf16
offsets. Two copies with the same behaviour on different weight changes would mean the copy is one of many
placements, not an interpretable one. E34 Amendment 5 reads the same on a trigger teacher; the bars are shared.

## Controls, run first

1. **C1**, alone: W₀ under the harness on adversary seeds 0, 1 and 2. It must pass `faith_score.w0_check`, and the
   PGD floor must lie within 5% of the card's known floor, 0.0748 on an H100. A card with no known floor stops.
2. **C2**: the teacher regenerated by E36's recipe. Its held-out gain at step 300 must be within 0.005 of E36's 0.1301
   and at least 0.02 nats per token, else stop before any copy.

## Filed predictions (in `run.py`, to be filed by the lead after review)

Noise floor for all: states are deterministic per seed up to bf16 kernels; the copies' batch-order spread is read in
this run, and E34 Amendment 4's was 0.005 in target probability for task-text copies and up to 25% in the harness ratio.

- **P1** lower_s0 keeps the behaviour: *s* ≥ 0.9. Medium. Blind to which positions carry the gain. Control: do-nothing.
- **P2** lower_s0 is close to the teacher: gap share ≤ 0.10. Medium-low. A mean, dominated by the highest-KL
  positions. Control: W₀ reads 1, do-nothing 0.
- **P3** lower_s0 is faithful: harness verdict "pass" on the PGD and stochastic ratios against the same-job floors.
  Medium. Blind to other adversaries; E34 Amendment 4 found this harness lenient on fine-tunes, not on copies.
  Control: W₀'s three adversary seeds.
- **P4** lower_s0's damage elsewhere is 0.5 to 2 times the teacher's. Medium. Plain Pile text only. Control: W₀ reads 0.
- **P5** the adjoint split loses the behaviour: *s* < 0.5. Medium. Control: the copies' *s*.
- **P6** plain_s0 keeps the behaviour: *s* ≥ 0.9. Low. The plain rows are the same documents as the lowercase ones.
- **P7** one placement: cosine(lower_s0, lower_s1) ≥ 0.8; below 0.5 reads "one of many placements". Medium-low.
  Blind to function: equal weights imply equal function, not the reverse. Null: a random direction, about 1e-4.

## Decision tree

- **C1 or C2 fails:** stop; nothing scored.
- **Pass, P1 and P3 hold:** teacher training is validated end to end on a broad update, and the larger-model run
  starts with it. If P7 also fails below 0.5, a card goes to the user before the placement is called interpretable.
- **Fail on behaviour, P1 fails:** teacher training's passes are a narrow-task property, found before the larger
  model's budget is spent. Next: whether the frozen Γ limits the copy, E15's refit, or the subcomponents' span does,
  the "create" reading.
- **Fail on faithfulness with the behaviour kept, P1 holds and P3 fails:** the same branch; P2 and P4 say which.
- **P6** holds: a copy from ordinary text carries a style, as Pile text carried the trigger fine-tunes in E34. Fails:
  the larger model's copy needs text like the post-training data.

## Memory, cost and routing

**Memory.** E36's lowercase run, the same fine-tune with Γ reads on eight 512-token rows, peaked at 17.4 GB on a
4090. A copy step holds the student's and the teacher's logits for 8 × 512 positions in float32, about 0.8 GB each
and five or six of them inside the KL, plus Adam on about 30M offset parameters and three frozen 67M-parameter copies.
The estimate is 15 to 25 GB at batch 8, which fits a 46 GB L40 or an H100; a 24 GB 4090 is not advised. The run
logs its peak after every stage.

**Time.** From E34's H100 briefs: load and C1 about 4 min, rows about 2 min, teacher 15 s, the do-nothing and adjoint
reads about 2.5 min, and each 1,000-step copy about 100 s of training plus a 55 s harness read. Three copies make about
8 min. The budget adversary adds about 17 s per state on five states, under 2 min. So about 22 min in all on an H100.
An L40 runs 1.5 to 2 times slower: about 33 to 38 min.

**Routing.** By the 30-min rule this sits at the line for the cluster's `debug` QoS on an L40
(`cluster/teacher_spread.sbatch`, 60-min limit). The recommendation is a RunPod H100 directly, about $1.5 with set-up,
as E34 Amendment 3 did. It is E34's card, so the W₀ floor of 0.075 and the pass line of 1.40 compare like with like.
The L40's floor is 0.21, which moves its pass line to 1.24 and widens the inconclusive band to 1.81. The code accepts
either card. Download `results/teacher_spread.json` and each copy file as written; a resume needs the copy files back.

## Open choices for the lead

1. **Plain rows are the same documents** as the lowercase rows, the first 2,400 of Pile train rows 0..4095, not E34's
   rows from 32,768. This holds content fixed and avoids a second stream. E34's rows would ask a broader question.
2. **Copies draw 8,000 rows from 2,400**, about three passes each; E34's Pile-text arm drew from 8,192 rows.
3. **The adjoint state is read through the harness too**, which E34 did not do; it costs about one minute.
4. **The budget adversary at 20 steps is reported** at W₀, do-nothing, adjoint, lower_s0 and plain_s0 (lead, 2026-10-26),
   through `gamma.pgd_sources` and `cancel.masked_kl`; no prediction rests on it.

## Not run

A second teacher seed; the code-like fine-tune, which did not learn; copies at steps other than 1,000; the plain copy at
a second batch order; the budget adversary past 20 steps or a longer harness; Γ trained with the subcomponents; LoRA teachers on
this task; the larger model.

**Hand-off.** The lead reviews this plan and `run.py`, runs the fresh-agent review of the new code, files the
predictions with `run.py --predictions`, and hands the run to `/pod-runner`.
