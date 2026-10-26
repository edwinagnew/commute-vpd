# What to do while the larger decomposition is awaited (decision, 2026-10-26)

*Lead agent, 2026-10-26, on the user's question "what is the most interesting or important thing we can do next?"
E36 (the spread-out fine-tune, re-gating line) runs in the background and is not touched. Words as in `GOALS.md` → Terms.*

**The one decision.** Both lines' named next step is the larger model, which is blocked on access. The recommendation
is to spend the wait on the commute line's biggest untested generalisation on the 67M model: **teacher training of a
fine-tune with no single trigger** (option 1), with two cheap apparatus checks beside it (options 2 and 3). The
alternatives considered are listed after the design check so the choice is visible.

## Option 1 (recommended): teacher training copies a fine-tune with no trigger

Every teacher-training pass so far (E34 and its four amendments) copies a fine-tune whose whole effect is one
association at one trigger token. The larger model's post-trained sibling has no trigger: its update moves
predictions over a distribution. Whether the map still holds for such an update is the question the larger-model run
will put first, and it can be asked now, on E36's fine-tune, for about $2.

### Design check

- **Goal served.** Q1, the weak version under reading A: a simple map from (D, δW) to a faithful decomposition of
  W₀ + δW with Δ not absorbing δW, for an update that is not a single association. It is the rehearsal of the
  larger-model protocol (Q9) on the existing model.
- **Nearest thing tried.** E34 (`TRIED.md` rows E34 to E34 Amendment 4): teacher training copies six full-weight
  fine-tunes and three LoRAs on three trigger tasks, faithful at two batch orders and under a 160-step adversary. E34's
  "What next" item 3 and `backlog.md` ("Teacher training") name a fine-tune without a single trigger as the step
  before the larger model. What E34 could not learn: whether a copy keeps a behaviour that is spread over every
  position of a text distribution, and whether its faithfulness on ordinary text survives when the teacher's change
  is itself on ordinary-looking text.
- **Patch or step.** A step: it changes what the update is, not how the copy is scored.
- **Working decisions.** Relies on Γ frozen and Δ fixed (decisions.md, 2026-10-14), reading A (2026-10-19), teacher
  training as the commute line's working map (2026-10-26). Departs from none. The code-heavy Pile rows are E36's
  task, already approved for that line; reusing them is not a new task family.
- **What result changes what we do.** Pass (copy keeps ≥ 0.9 of the teacher's held-out gain, adversarial ratio at or
  under its do-nothing's and under the line): the larger-model run starts with a protocol validated end to end on a
  broad update. Fail on behaviour: teacher training's passes are a narrow-task property, found before the larger
  model's budget is spent; the next question becomes whether the copy is limited by the frozen Γ (E15's refit comes
  back) or by the subcomponents' span (the "create" reading). Fail on faithfulness with the behaviour kept: the same
  branch, and the copy's damage profile says which.
- **Control first.** The W₀ check (floor 0.075 within 5%), then the teacher's own reading: the fine-tune must raise
  the held-out code rows' mean next-token log-probability by at least 0.02 nats per token (E36's gate G) before any
  copy trains, else stop. E36's first run on 2026-10-26 (17:22) read that gain **negative** at steps 10 and 100
  (−0.05, −0.07 nats per token): the 800-row fine-tune did not generalise to its held-out rows, and E36 is being
  rerun. So this option waits for E36's final recipe and takes the fine-tune from it. A teacher that does not
  generalise cannot be copied meaningfully.
- **Numbers, and what each should read.** Per state (teacher's do-nothing, scaled adjoint split, copy from code rows,
  copy from plain Pile rows), at step 1000 of the copy:
  1. *Behaviour kept* s = (L_state − L_W₀) / (L_teacher − L_W₀), L the mean next-token log-probability on the
     held-out code rows. W₀ 0, do-nothing 1 by construction, copy ≥ 0.9 (E34's task-text copies read 0.92 to 1.00),
     adjoint split expected well below (0.06 to 0.26 on the word tasks).
  2. *Teacher gap* KL(teacher ‖ state) on the held-out code rows, as a share of W₀'s: copy ≤ 10% (E34's bar).
  3. *Adversarial ratio* to W₀ on the Pile val rows (the harness, 20 steps): W₀ 1.00 ± 0.075, pass ≤ 1.24, and
     read against the same update's do-nothing (the pending anchor decision). Expected: copy at the floor, do-nothing
     possibly above the line (a broad 100-step Adam 3e-5 update is larger in effect than the trigger ones).
  4. *Damage elsewhere* KL(W₀ ‖ state) on held-out plain Pile rows: copy between half and twice the teacher's.
  5. *Reported*: size of the change, cosine with δW, and the cosine **between two batch orders of the copy** (the
     uniqueness question: a copy that matches the teacher's behaviour but lands on a different weight change each
     time is not an interpretable placement; E34's cosines with δW of 0.08 to 0.22 raised this and no run has read it).
- **Cost.** One 100-step fine-tune, two or three 1000-step copies, five harness reads on the 67M model: about a sixth
  of E34's 67 H100-minutes, so under 15 minutes. Cluster `debug` queue first, RunPod 4090 if not started in 10 min.
- **Class.** Class 4 by E34's named next step and the backlog entry, on E36's approved task. Put to the user anyway,
  since the question is which of several next steps to take.

## Option 2 (cheap, class 1): settle the two adversaries' disagreement

E34 Amendment 4's budget adversary reads the fine-tunes up to six times above the harness at the same 20 steps
(the r1 SGD fine-tune 9.2 against 1.59; both r0 fine-tunes above the line where the harness passed them), with no
cause found. E20 Amendment 3 had said the verdicts read the same from 20 to 160 steps. The two measures differ in rows
(128 against 256) and in how the KL under the found masks is read out. One saved update read both ways on 256 rows,
then under the harness's adversary at 160 steps, locates it. If the rows carry it, the pass line is re-derived on 256
rows; if the readout does, every "inconclusive" verdict in the project is re-read. GPU minutes on saved states, no
new design. Worth doing before the larger model whichever option runs first; it can run as a stage of option 1.

## Option 3 (cheap, class 1): is the copy unique?

The cosine between teacher-trained copies at two batch orders, on the states Amendment 4 saved (seed 1) and E34's arm A
(seed 0), CPU only. Folded into option 1's reported numbers above; listed separately in case option 1 waits on E36.

## Considered and not recommended now

- **The shift applied only where the trigger is** (E30 reading ii). A clean question, but trigger-bound; the user asked
  on 2026-10-23 that the lines not get lost in toy cue mechanics. Keep behind option 1.
- **The Γ refit with a matched attacker** (E15, ongoing). About $9 per refit on an H100 after the attacker is matched,
  and its W₀ control has failed twice. Teacher training passing with Γ frozen has lowered the pressure; it returns if
  option 1 fails.
- **The strong version** (re-decomposition). Postponed by the user until the weak version is settled; needs the VPD
  training stack.
- **Preparing the larger-model harness.** Loaders and the four checks cannot be written to an unseen decomposition
  format; what can be done blind is small.

## Pending from before

The adversarial pass line's anchor (fixed 1.5 line, or the same update's do-nothing) is still the user's decision
(research log, Pending decisions). Option 1 reports both readings, as every verdict has since E21.

## Hand-off

If option 1 is chosen: a new session writes `reports/plans/2026-10-26_plan_teacher-spread.md` from this check once E36's
analysis names its final fine-tune recipe, files predictions, and runs through `/pod-runner`. Read this report, E36's
plan Amendment 1 and `reports/analyses/2026-10-23_analysis_teacher-training.md` → Amendment 4 → What next.
