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
from the first 2,400 Pile rows of the same range not lowercased, nearly the same documents since re-encoding
lowercased text shifts the row boundaries; **lower_s1**, lower_s0 at batch order 1. The copies use E34's Pile-text
recipe: Adam 1e-5, eight rows per step, plain mean KL, 1,000 steps.

| number | definition | W₀ | do-nothing | copy, expected |
|---|---|---|---|---|
| 1. behaviour kept *s* | (L_state − L_W₀) / (L_teacher − L_W₀), L the mean next-token log-probability on held-out lowercase rows | 0 | 1 by construction | ≥ 0.9; adjoint below 0.5 |
| 2. teacher gap share | KL(teacher ‖ state) on those rows, over W₀'s. W₀'s, KL(teacher ‖ W₀), was never read; E36 read the other direction, KL(W₀ ‖ teacher), 0.29 nats at step 300, so 0.2 to 0.4 is expected | 1 | 0 by construction | ≤ 0.10 |
| 3. adversarial ratio | the Pile harness: 256 val rows, 20-step adversary, ratio to the same job's W₀ | 1, floor 0.075 | lean above the line | pass, ≤ 1.40 on an H100 |
| 4. damage elsewhere | KL(W₀ ‖ state) on held-out plain Pile rows | 0 | the teacher's own | 0.5 to 2 times the teacher's |
| 5. reported | size; cosine with δW against a random direction's; mean train KL over steps 451-500 and 951-1000 | 0 | 1 | 0.08 to 0.3 with δW, as E34 |
| 6. budget adversary, reported | E20 Amendment 3's adversary: 20 sign steps of 0.1 from seed 0 on the first 128 Pile val rows, then the KL under the masks found, as a ratio to W₀'s; not at lower_s1 | 1 | lean above 1.24, as E34's fine-tunes read 1.7 to 9.2 | at or near 1, as E34's copies |

Number 3 is also read against the do-nothing state's ratio, for the pending anchor decision; that reading is reported,
not scored. Number 6 keeps E37 readable if E34 Amendment 5 finds the 20-step harness lenient on fine-tunes;
no prediction rests on it. The do-nothing state must read as the teacher: C3 below.

**Uniqueness.** The cosine between lower_s0's and lower_s1's weight changes, both rebuilt from their saved bf16
offsets. Two copies with the same behaviour on different weight changes would mean the copy is one of many
placements, not an interpretable one. On the same axis sits the reference: the cosine between lower_s0 and plain_s0,
the same teacher copied from other text. E34 Amendment 5 reads the same on a trigger teacher; the bars are shared.

## Controls, run first

1. **C1**, alone: W₀ under the harness on adversary seeds 0, 1 and 2. It must pass `faith_score.w0_check`, and the
   PGD floor must lie within 5% of the card's known floor, 0.0748 on an H100. A card with no known floor stops: the
   runner must get "NVIDIA H100 80GB HBM3", the only H100 with a known floor in the repo; H100 PCIe and NVL have none.
2. **C2**: the teacher regenerated by E36's recipe. Its held-out gain at step 300 must be within 0.005 of E36's 0.1301
   and at least 0.02 nats per token, else stop before any copy.
3. **C3**, after the do-nothing read: its log-probability within 1e-4 nats per token of the teacher net's, and its
   KL from the teacher within 1e-4 of 0, else stop before the copies. The weights and kernels are the same, so the
   two should agree to float rounding; 1e-4 allows non-deterministic bf16 kernels and is under E36's 2e-4 across cards.

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
- **P6** plain_s0 keeps the behaviour: *s* ≥ 0.9. Low. The plain rows are nearly the same documents as the lowercase ones.
- **P7** one placement: cosine(lower_s0, lower_s1) ≥ 0.8; below 0.5 reads "one of many placements". Medium-low.
  Blind to function: equal weights imply equal function, not the reverse. Comparison on the same axis:
  cosine(lower_s0, plain_s0). Null: a random direction, about 1e-4.

## Decision tree

- **C1, C2 or C3 fails:** stop; nothing scored.
- **Pass, P1 and P3 hold:** teacher training is validated end to end on a broad update, and the larger-model run
  starts with it. If P7 also fails below 0.5, a card goes to the user before the placement is called interpretable.
- **Fail on behaviour, P1 fails:** if the train KL still fell by more than 10% between steps 451-500 and 951-1000,
  the copy is under-trained and is rerun longer. Otherwise teacher training's passes are a narrow-task property,
  found before the larger model's budget is spent. Next: whether the frozen Γ limits the copy, E15's refit, or the subcomponents' span does,
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
either listed card. Download `results/teacher_spread.json` and each copy file as written; a resume needs the copy files back.

## Open choices for the lead

1. **Plain rows are nearly the same documents** as the lowercase rows, the first 2,400 of Pile train rows 0..4095, not E34's
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

## Amendment 1 (2026-10-27): train longer, and two other fixes, read under the paper's check

*Design agent for the lead, after the user chose option (a) of card `decision-broad-copy` and asked for other fixes
besides more training (`decisions.md`, last two rows). Not yet built: code, predictions file and review follow.*

**In plain words.** E37's decomposition after teacher training (below, "the trained decomposition") kept all of the
fine-tuned model's gain on lowercase text but read 2.20 times the original decomposition under the paper's adversarial
check, and it was still learning. This run trains it four times longer, reading the check along the way, and tries two
other fixes in the same job: adding VPD's own masked-reconstruction losses to the training, and starting from the
adjoint split, which passes. One H100 hour, about $4.

**The request in the Terms' words.** Teacher of E37 (E36's lowercase fine-tune, W₀ + δW, regenerated). Run **teacher
training** (component training of the alive (U, V), loss KL(teacher ‖ student), Γ and Δ frozen) on E37's 2,400
lowercase rows past step 1,000, and read the Pile harness (the paper's check: `calc_multibatch_pgd_masked_recon_loss`,
one random start, seed 0, 20 steps of 0.1; fixed by the user, `decisions.md`) at several steps. Beside it, two variants
of teacher training on the same rows, each read the same way.

### Design check

- **Goal.** Q1, the weak version under reading A, for a broad update: whether teacher training, the commute line's
  map for the larger model, gives a faithful decomposition when the update acts at every position.
- **Nearest tried.** E37 (fails at step 1,000, still learning); E34 and amendments (trigger teachers, all pass); E15
  (Γ refit: its W₀ control drifted off the floor twice, 1 → 2.6 and 7.8). Nothing in `TRIED.md` or `backlog.md` has
  trained the subcomponents against VPD's masked losses or warm-started teacher training.
- **Patch or step.** A patch on E37, and the step the larger-model run waits on.
- **Decisions.** Keeps Γ frozen, Δ fixed, reading A, the paper's check as fixed on 2026-10-27. Arm 2 changes teacher
  training's loss; adopting it for the larger model would be the user's call (tree below). Covered by the user's
  2026-10-27 decision to try other fixes in this run.
- **What changes what we do.** Arm 1 passing keeps plain teacher training for the larger model; arm 2 or 3 passing
  changes the recipe; nothing passing sends card option (d) or the anchor question back to the user.
- **Control first.** C1 (W₀ under the harness, three seeds, known floor 0.0748) alone, stopping on a miss; then C2,
  C3 as in E37. Arm 1's step-1,000 read reproduces E37. Arm 2 has its own control, M0, run before it.

### Arms, in run order (all on the lowercase rows, Adam 1e-5, batch 8, batch order 0)

1. **Longer.** E37's recipe to step 4,000; read at 1,000, 2,000, 4,000 (each doubling). *Could lower the reading:*
   E37's training KL still fell 36% over its second 500 steps, and the filed tree calls that under-trained. *What
   differs from E37's end state:* nothing in the recipe. Same rows, seed and batch draws (the draw generator is
   untouched by the extra save steps), constant learning rate, so steps 1,001 on continue E37's optimizer state with no
   schedule position or moment reset. Only kernel non-determinism differs: the step-1,000 read is the check (R1).
2. **Masked losses** ("masked teacher training"). The teacher KL plus VPD's own two reconstruction losses at their
   released coefficients (`StochasticReconSubsetLoss` 0.5: masks drawn from Γ's importances as g + (1 − g)·u;
   `PersistentPGDReconLoss` 0.5: a persistent adversary over the same mask range), read through param-decomp's
   `compute_losses` exactly as `gamma.Refit.step` does, with U, V trainable instead of Γ. Their target is the student's
   own unmasked output, as in VPD. 2,000 steps after 50 warm-up steps at learning rate 0 (E15 protocol v2: the
   persistent adversary is flat by step 50); read at 1,000 and 2,000. *Could lower the reading:* these losses penalise
   exactly what the check probes, the output's sensitivity to turning down subcomponents Γ marks unimportant, while the
   teacher KL holds the behaviour; they are what makes the released decomposition pass. *Differs from arm 1:* U, V are
   trained as the component model's own parameters (gradients zeroed on dead columns), and the student's forward runs
   through the components with every mask at 1 and Δ₀, not through hooks: the same function. Gate G2: its step-1 teacher
   KL equals arm 1's (0.290) within 1e-3. The masked losses read lowercase rows; the check reads plain Pile rows.
3. **Adjoint start.** Teacher training from the scaled adjoint split's (U′, V′) (E37: keeps 0.80 of the gain, passes
   at 1.05, damage 0.05) instead of W₀'s, 2,000 steps; read at 0, 1,000, 2,000. *Could lower the reading:* if the
   failure is where teacher training puts the change, a start inside the passing region may end at a placement that
   passes. If it ends where arm 1 does, the failure goes with the teacher's change on ordinary text, which the E37
   analysis named and did not test. *Differs from the adjoint state:* nothing at step 0 (read reproduces E37's); Adam's
   moments start at zero, as intended.

**M0, arm 2's control, run alone before arm 2:** arm 2's recipe with the teacher set to W₀ (δW = 0), 1,000 steps. The
teacher KL then only holds the model in place, so M0 shows what the masked losses do on their own. Early stop: damage
above 0.02 nats or the check above 1.40 means the recipe drifts on its own, as E15's did; retry once with both
coefficients at 0.05, then skip arm 2 if it still misses.

**Not chosen.** *Small offsets* (decay toward the start, lower lr): the trained decomposition is already 0.75 of the
fine-tuned model's size, Adam's verdicts tracked damage elsewhere rather than size (E21), and a lower rate only slows
arm 1. *Mixing plain Pile rows:* the trained decomposition already reproduces the fine-tuned model's change on plain
Pile (0.996 of its damage) and the plain-text arm's 1.74 lies inside the batch-order spread. *Γ refit* (option d, about
$9 per state): its W₀ control has not yet held the floor, so a reading would not be interpretable; it stays the next
step if this run passes nothing.

### Numbers (each state; `run.read` with the budget adversary off)

| number | definition | W₀ | do-nothing | control | arms, expected |
|---|---|---|---|---|---|
| s, behaviour kept | (L_state − L_W₀)/(L_teacher − L_W₀) on 64 held-out lowercase rows | 0 | 1 | R1: 0.998 ± 0.003; arm 3 step 0: 0.804 | ≥ 0.9 |
| damage | KL(W₀ ‖ state) on 64 plain Pile val rows, nats | 0 | 0.235 | M0 ≤ 0.02 | about 0.23; arm 3 rises from 0.05 |
| check ratio | harness PGD KL over C1's W₀ seed 0; pass ≤ 1.5/(1+f), fail ≥ 1.5(1+f), f from C1 (1.40, 1.61) | 1 | not read (4.87 in E37) | R1: 2.20 ± 15%; M0 0.7–1.4; arm 3 step 0: 1.05 | see P1–P3 |
| stochastic ratio | the harness's stochastic KL, in the filed verdict with the check ratio | 1 | not read | — | about 1.1; arm 2 lower |
| train KL | mean teacher KL over the 50 steps before each read | — | — | arm 1 at 1,000: 0.0110 | falling; "flat" below 10% per doubling |

### Predictions (to be filed in `results/amend1_predictions.json`; the run refuses without it)

Noise floor: W₀'s three-seed spread (0.075) and E37's batch-order spread of the check ratio, 14%.

- **P1** arm 1 at step 4,000 still fails: check ratio ≥ 1.5(1+f). Medium-low. Blind to other adversaries (out of scope).
  Control: R1.
- **P2** arm 2 at step 2,000 passes with the behaviour: s ≥ 0.9 and verdict "pass" on the check and stochastic ratios.
  Low-medium. Partly trains on what the stochastic ratio reads; the check ratio is also reported over M0's.
- **P3** arm 3 ends where arm 1 does: at step 2,000, s ≥ 0.9 and its check ratio within 25% of arm 1's at step 2,000.
  Medium-low. Control: its step-0 read reproduces E37's adjoint state.
- Stops: C1–C3 as E37; G2 and M0 stop arm 2 only. R1 is reported, not a stop: arm 1's later reads compare with its own
  step-1,000 read. A "solution" below means s ≥ 0.9 and verdict "pass".

### Decision tree

- **C1–C3 fail:** stop, nothing scored. **M0 or G2 fails:** arms 1 and 3 stand; arm 2 is reported as not run.
- **Arm 1 is a solution at 4,000:** under-training explains E37; the larger model keeps plain teacher training, trained
  until the training KL falls under 10% per doubling and read at each doubling.
- **Arm 1 not a solution, training KL still falling ≥ 10% from 2,000 to 4,000 and the ratio falling:** under-training
  is not ruled out; extend arm 1 to 16,000 steps (about $1.5), this tree's branch.
- **Arm 1 not a solution and flat:** arm 2 a solution → card to the user: masked teacher training as the map for the
  larger model, with M0's ratio beside it (if M0 reads under 0.85, part of the margin is the losses lowering the check
  on any decomposition). Arm 3 a solution → card: the adjoint start as the recipe. Neither → the failure goes with the
  fine-tuned model's change on ordinary text (P3 says whether the start mattered); card to the user with option (d)
  and the anchor question for broad updates.

### Code, memory, time, cost, routing

- **Code.** `experiments/teacher_spread/amend1.py` (~200 lines), importing `run.Job` (load, C1, rows, teacher),
  `run.read` (budget off) and `run.adjoint`. Shared: `row_copy.copy_on_rows` and `component_training.train` as they
  are, with two small additions: a `save_steps` tuple for `copy_on_rows` (arm 1), and an `init` offsets argument for
  `component_training.Offsets`/`train` (arm 3, ~6 lines). New: `src/commute/masked_teacher.py` (~100 lines, arm 2 and
  M0), with the loss-config and persistent-adversary set-up moved out of `gamma.Refit.__init__` into one helper both
  call; KL through `teacher.kl_exact_zero`. Fresh-Opus review of the diff before the run (Codex out of quota).
- **Memory.** E37 peaked at 30.5 GB. Arm 2 adds Γ's forward and the persistent adversary's per-position sources at
  batch 8 (8 × 512 × 38,936 floats, ~0.6 GB, with its optimizer ~2 GB): 40–50 GB estimated.
- **Time.** From E37's log: load and C1 3.5 min; teacher and the W₀/do-nothing reads 1 min; arm 1 6.5 min training
  plus three reads at ~1 min; arm 3 about 6.5 min with its reads. Arm 2's step time is unknown (Γ's forward plus two
  masked forwards per step), estimated 0.3–0.6 s: M0 and arm 2 together 18–33 min with reads. The runner times M0's
  first 50 steps; if M0 plus arm 2 would pass 40 min, arm 2 stops at 1,000 steps. Total 40–55 min, plus about 7 min
  of set-up.
- **Cost.** E37's pod ran at about $4/h: $3–4.5, up to $5.5 with M0's retry. Runner's cap $8.
- **Routing.** Over 30 min: RunPod "NVIDIA H100 80GB HBM3" directly, via `/pod-runner` (pod
  `commute-teacher_spread-a1`); other H100s have no known floor and stop at C1. Inputs as E37
  (`transport_test/results/large/{alive,windows}.pt`). Download `results/amend1.json` after every stage and the four
  saved offset files as written (arm 1 at 1,000 and 4,000, arms 2 and 3 at their ends, bf16, 60 MB each); before
  terminating, check the JSON's sha256 on both sides. Every number also goes to the log.

### Not run

A second fine-tune seed or batch order; the plain-text arm longer; the check at other starts or longer (out of scope,
2026-10-27); a coefficient sweep for arm 2 beyond M0's one retry; arm 2 with Γ trained (E15); arms 2 and 3 combined;
decay toward the start; mixed text; the budget adversary; the anchor rule; the larger model. After the run, on CPU:
the cosine between arm 1's step-1,000 offsets and E37's saved ones (expected ≥ 0.99).
