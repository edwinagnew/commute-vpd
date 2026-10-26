---
name: project-teacher-training
description: "Teacher training (E34, started 2026-10-23): component training with the fine-tuned model as KL teacher; Amendments 2-4 (loophole g>0.1, LoRA teacher, seed 1 + 160-step adversary) launched 2026-10-26; plan reports/2026-10-23_plan_teacher-training.md"
metadata:
  node_type: memory
  type: project
  originSessionId: f40da049-daf0-44de-8c13-5534d40acaaa
  modified: 2026-10-26T14:51:09.066Z
---

**Teacher training** is the user's name (confirmed 2026-10-23; "component teacher" was offered and declined by the lead;
use "teacher training" everywhere). Definition: component training where the loss is KL(teacher ‖ student) with the
teacher the frozen fine-tuned model W₀ + δW; (U, V) trained, Γ and Δ frozen. It consumes only W₀, W₀ + δW and inputs.
Proposed by the 2026-10-23 morning session as component training's fix for needing the task's loss and data.

User choices (2026-10-23): two input arms, task windows (arm A, E21's trigger-plus-mean loss weighting) and Pile only
(arm B, uniform mean); teachers left to the lead (E31's six fine-tunes, regenerated in-job since none are saved);
go-ahead to build, file predictions and run on an H100 (~$6-8) without another stop, check in only on a control failure or a
reversed prediction. Δ stays frozen in the first run; a Δ-free variant is backlog material. Plan:
`reports/2026-10-23_plan_teacher-training.md` (its "Lead's decisions" section is the final word on open points).

**Why:** the first map from δW alone that could meet the weak version; the user wants it taken to the larger model next
week if it works.
**How to apply:** new work on it falls under the commute line's named next steps ([[project-two-parallel-lines]]); keep
the GOALS Terms row in sync once written. See [[project-success-criterion-reading-a]].

**Outcome (2026-10-23, E34 + Amendment 1, $9.43 total).** Every arm-A student (task windows, Adam 1e-5) is at the
adversarial floor with P(target) ≥ 0.92 at steps 100 and 1000 on r1, r0, emoticon, for SGD and Adam 3e-5 teachers;
the adjoint split of the same δW keeps 0.06-0.26 on r0/r1. Pile-only students (arm B) all pass at step 1000 (P 0.90-0.98),
emoticon included (filed P3a reversed). Amendment 1: masking or dropping every trigger position/row leaves the end state
within 0.01-0.02, so the ~90 trigger occurrences are not the cause; the student's weight change has cosine 0.08-0.22 with
δW (adjoint split 0.38-0.45): same function, different weights. SGD on (U, V) still fails at step 1000 on r0/r1 (P4).
Open: 45 positions with g 0.1-0.5 remained in the dropped rows (close with g > 0.1); LoRA teacher; second seed; longer
adversary; the larger model (user: next week). Codex review was unavailable (usage limit until 2026-11-18); independent
Opus reviews substituted.

**Amendments 2-4 (2026-10-26, user: "implement the amendments. split into subagents accordingly").** Design check
appended to the plan under "Amendments 2-4" (classes 3/4, no sign-off). One script, predictions file and run brief
each: `experiments/teacher_training/amend{2,3,4}.py`, `results/predictions_amend{2,3,4}.json` (filed 10:49-10:51),
`RUN_BRIEF_amend{2,3,4}.md`; CPU tests `tests/test_teacher_amend{2,3,4}.py`. Amend 2 = B-dropped at g > 0.1 (emoticon,
both teachers; P1 within 0.03 of Amendment 1's B-dropped = "not carried"); Amend 3 = E9's rank-8 LoRA as teacher on
emoticon/r1/r0 with arms A and B (G1 vs E9/E14); Amend 4 = arms A and B at batch-order seed 1 for the six teachers,
read at 20 steps and under E20 A3's budget sweep (20, 160 sign steps of 2/n, pass 1.241). Codex was still
rate-limited (until 2026-11-18); an Opus adversarial review found no blocking defect. Three H100 pods in parallel
(caps $5/$6/$10), launched 11:00 GMT. Results land in `results/teacher_training_amend{2,3,4}.json`.

**Outcomes (2026-10-26, $21 of $40; three pods lost to session disconnects, ~$9 of reruns, rules now in CLAUDE.md).**
Amend 2: dropping rows with g > 0.1 (104 rows) lowers the Pile-only copies to 0.872 (SGD) and 0.844 (Adam), falls
0.065/0.049, both "between" by the filed rule; still faithful; cosines with δW unchanged. Amend 3: G1 was re-referenced
to E27's H100 LoRA readings after a stop (E9's L40S factors differ; first filing archived); LoRA teacher copied from
task text on all three tasks (P 0.93-1.00, passes), Pile-only copies 0.52-0.88 "between" though faithful; adjoint
split of the LoRA keeps 0.04-0.23. Amend 4: seed 1 repeats every E34 copy (P spread ≤ 0.005 arm A, ≤ 0.032 arm B,
emoticon Adam Pile-only largest); all 12 copies pass the 160-step adversary (0.95-1.15 vs line 1.241) while the
teachers as do-nothing read 2.4-10.4. All Amend 4 predictions held. Next: the larger model (user).
