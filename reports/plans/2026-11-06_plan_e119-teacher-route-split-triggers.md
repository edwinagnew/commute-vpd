# Plan: E119, the route split of the teacher-trained decompositions on the trigger tasks

*2026-11-06, code writer for the regating lead. Asked for by the user by name (2026-11-06, $0, cluster `debug`). Code
`experiments/teacher_route_split/run.py`, `plot.py`; predictions `experiments/teacher_route_split/results/predictions.json`;
job `cluster/teacher_route_split.sbatch`.*

## The request in the Terms' words

The user (2026-11-06, verbatim): "Route-split the teacher-trained decompositions on the trigger tasks (E34's grid was not
saved: regenerate the states from the saved teachers, same seeds). Report, on the same axes as E12, where each copy
lands: re-switching, masked-off, or neither, beside its fine-tune's model's own reading. One figure, one analysis file,
one log card. No amendments; if the regeneration does not reproduce E34's behaviour numbers within 0.05, stop."

Restated: E34's twelve students (teacher training: component training with loss KL(teacher ‖ student), Γ and Δ
frozen; teachers E31's arm-A full-weight fine-tunes, SGD and Adam 3e-5, seed 0, step 1000, on r1_t534_v369,
r0_t352_v310 and emoticon; arm A on the task windows, arm B on Pile rows; step 1000) are regenerated with E34's seeds.
Each is read by the route split at the trigger on its own state (target W₀ + the student's change, components the
student's U′, V′, Δ fixed: E12 Amendment 1's state (a)) and labelled by E12's filed lines, beside the least-squares
split of its fine-tune. **The teachers were not saved** (`experiments/SAVED.md`, teacher_training: "regenerated in-job
and not saved"; no E31/E34 trigger-task teacher in `regating_table` or elsewhere), so they are regenerated too. (E34 Amendment 5 saved `r1_t534_v369/amend5_{A,B}_sgd_seed0_step1000.pt`, students of
the same r1 SGD teacher; not used, the user asked for regeneration, but they are a $0 cross-check afterwards.)

## Design check (user-requested work)

- **Goal.** `GOALS.md`'s question whether an update lands in the decomposition as re-gating of existing pieces: E12's
  route lines parted copy from map tasks; this asks where the trigger tasks' teacher-trained states sit on them.
- **Nearest tried.** E12 Amendment 1 (the same read on copy and map tables: students keep their fine-tune's r within
  0.06 of the gain, the remainder goes masked-off); E34 (the students, behaviour only). New: the trigger tasks' route.
- **Definition.** No reuse verdict is given (rows 81/87 not read). The labels are E12's filed lines unchanged (P4: r ≥ 0.6
  / ≤ 0.35; Amendment 1 P3: e_gate ≥ 0.5 / ≤ 0.25); a state they cannot read is "unreadable".
- **Changes what next.** A student landing re-switching or masked-off where its fine-tune is "neither" is the surprise.
- **Control first.** Per task, W₀'s cells against E31's, then each teacher's P(target) against E34's, then do-nothing
  and least-squares checks, all before that teacher's students. Recipe differences from E34: L40 not H100; no Pile
  harness, adjoint, step 100 or arm S (none feeds training). The reproduction check is what tests the regeneration.

## Reads and expected values

| number | W₀ | do-nothing | least-squares | student |
|---|---|---|---|---|
| T0, U0, L00 | E31's within 0.02 nats | | | |
| teacher P(target) | | E34's within 0.05 (0.94-1.00) | | |
| r | not read (gain 0) | E31's within 0.03 (0.49-0.61) | = do-nothing's, 1e-6 | within 0.06 of its fine-tune's |
| e_gate (masked-off) | not read | = −r, 1e-6 | unread here, guess 0.05-0.40 | 0.33-0.50 |
| student P(target) | | | | E34's within 0.05 (0.90-1.00) |
| label | | | neither (r1 SGD maybe re-switching) | neither |

Positive control ($0, in the dry run and in the job's score): E12 Amendment 1's eight students under the same rule read
re-switching (4 copy) and masked-off (4 map), so the rule can return both end labels. Noise floor from E12 A1's batch
orders (|Δ e_gate| 0.002). Predictions P1 (majority label of the 12, lean neither) and P2 (students keep their
fine-tune's label, lean same, low) are filed with rules, blind spots and edge cases.

## Stop rules

Missing input files; W₀ cells off E31's by > 0.02; a teacher's P(target) off E34's by > 0.05; do-nothing r off E31's by
> 0.03, or e_gate + r or least-squares r − do-nothing r beyond 1e-6; a student's P(target) off E34's by > 0.05 (the user's
stop, checked after its record is written). A stop writes `.stopped` and exits 1. Records are written after every read.

## Timing (L40), from measured numbers

| stage | per unit | source |
|---|---|---|
| model, alive, least-squares bases | ~30 s + ~1 min start | E12 A1 job 48885 (L40), `results-large/new_vs_existing/amend1/run_amend1.log` lines 1→9 |
| teacher fine-tune, 1000 steps | 39 s | E31 A2 job 30644 (L40), `commute-data/cluster-logs/commute-adjoint-sgd-a2-30644.out` lines 21→123 |
| do-nothing + least-squares reads | ~8 s | `run_amend1.log` lines 9→10 |
| student arm A, 1000 steps + reads | ~120 s | `run_amend1.log` lines 76→89 (same 41-token windows, batch 32) |
| student arm B, 1000 steps + reads | ~175 s, **not measured on an L40** | H100 ratio B/A 92/64 s, `results-large/teacher_training/amend6/run_amend6.log` lines 265, 278, 291 |

Per task 2 × (39 + 8) + 2 × 120 + 2 × 175 ≈ 690 s, plus start-up: **~13 min per task, ~36 min for all three**. One job
does not fit 25 min, so the job takes the task as its argument: **three debug jobs, one per task** (~13 min each; arm B
at 250 s would make 16 min). The lead decides. Offsets saved: 12 × ~60 MB = ~720 MB in `results/large/<task>/`.

## Hand-off

After the three jobs: `plot.py` → `figures/route_labels.png`, one analysis report under `/writeup`, one log card via
`docs/research_log/PENDING.md`.
