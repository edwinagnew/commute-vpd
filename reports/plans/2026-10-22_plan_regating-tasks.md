# Plan: the working hypothesis's kill tests on seven new tasks (E28, `experiments/regating_tasks`)

*2026-10-22, coding agent for the lead session. Decision report:
`reports/decisions/2026-10-22_decision_working-hypothesis-and-two-lines.md`. Definitions: E18
(`reports/plans/2026-10-21_plan_regating-leak.md`, Amendments 2 and 3) and E24 (`reports/analyses/2026-10-22_analysis_regating-small.md`).
Words as in `GOALS.md` → Terms.*

## The design check

**Approved class.** Class 4 of `.claude/skills/design/SKILL.md`: the main line's named next step ("test the kill
conditions on new tasks") on the existing model, tasks and harness. No sign-off needed.

**For the user to see (one definitional choice).** The fixed-change kill rule is scored on E18 Amendment 1's mean pattern
net of its reference, R_mean_net = (ρ_mean − ρ_mean_W0) ÷ ρ_trig. That is the number behind the decision report's
"0.70 to 0.86 on three tasks". E18's filed shared pattern R_shared = ρ_shared ÷ ρ_trig read 0.41 on r1_t534, so scoring
it would already have killed H′ on a task the decision report counts as support. R_shared is reported beside it, with
E18's 0.73, 0.41 and 0.62 as context.

- *Goal.* `GOALS.md` → "Where an update goes", the reveal reading, the log's Q5. The decision report's first next step
  for the main line.
- *Nearest tried.* E18 and its four amendments on three tasks; E24 on small updates, two tasks. Nothing has read H′ on
  the other ladder tasks or on a task the model lacks.
- *Patch or step.* A step: the first kill test the decision report asks for.
- *Working decisions.* Γ frozen, do-nothing placement, as E18. No departure.
- *What changes next.* Any ladder kill: stop and tell the user; H′ is restated, and Q5 with it. Ladder and key-value
  tasks all survive: H′ broadened to seven tasks; next E29 and the larger model. Ladder survives, a key-value kill: H′
  scoped to tasks the model already does; tell the user.
- *Control first.* G1, emoticon, first and alone; the job stops if it fails (below).
- *Numbers.* The arms table below, with values at W₀, for the nulls and for the control.

## Restatement

After a 10-step Adam fine-tune, put δW in Δ (do-nothing), keep the released subcomponents and a frozen Γ, and leave Δ
out of the CI-masked model. The switching effect is the CI-masked model's gain at the trigger when Γ reads the updated
model. H′ says one fixed change to Γ's values at the trigger carries most of it, specific to which subcomponents get
which value, and indifferent to which input the updated values came from. E28 reads the decision report's three kill
conditions on seven tasks H′ was never read on.

## Tasks and states

Do-nothing states of 10-step arm-A fine-tunes of all 24 decomposed matrices, Adam 3e-5, batch 32, seed 0, regenerated
in-job (`regating.finetune`); the ` http` (3944) fine-tune by the same recipe gives the cross null. Run order, each
state written to the JSON as it finishes:

1. emoticon (G1, alone; stop on a fail);
2. ladder `r0_t352_v310`, `r1_t588_v452`, `r2_t275_v2667`, `r0_t588_v320` (target: the task's token);
3. key-value `ident32`, `rand32`, `rand1` (E10; target: the (V,) key → value table read at the key, trigger = the key
   at `half`, marker at `half − 1`).

## Arms (trigger position only, held-out windows, W₀'s values elsewhere; R = ρ ÷ ρ_trig)

| number | definition | at W₀ | null / control | under H′ | kill |
|---|---|---|---|---|---|
| r, gain, gap | `regating.measure()` as it stands | 0 | G2: on-disk r within 0.1 | — | — |
| R_mean_net | (ρ_mean − ρ_mean_W0) ÷ ρ_trig | 0 | emoticon 0.81 | 0.7 to 0.9 (E18 0.70 to 0.86) | < 0.5 |
| R_shared | clip(g₀(x) + s), reported | 0 | emoticon 0.73 | E18 0.73, 0.41, 0.62 | not scored |
| R_shuffled | s permuted among each matrix's alive subcomponents, clipped; mean of 20 draws (`regating.shuffled_recovery`) | 0 | emoticon 0.002 | within 0.03 of 0 (E18 −0.026 to +0.003) | > 0.2 |
| R_other | g₁(π(x)), π a derangement; mean of 5 (`regating.other_window_recovery`) | = its reference | emoticon 0.887 | 0.85 to 1.0 (E18 0.89 to 1.00) | < 0.5 |
| R_other_W0 | g₀(π(x)), the same π: the reference | itself | emoticon 0.007 | near 0 | scored only if \|·\| ≤ 0.2 |
| ρ_cross | clip(g₀(x) + s_null), ` http` null | 0 | ≤ 0.1 with the null's P ≥ 0.5 | — | control |
| random | s's S_on values moved to random same-size sets, 20 draws | 0 | ≈ 0, ≤ 0.1 | — | control |
| \|S_on\| | alive with s > 0.3 | 0 | emoticon 210 | reported | — |

**Scorable state.** G2 passes, gain ≥ 1 nat, gap ≥ 0.5 nats, ρ_trig ≥ 0.5 (as E24).

**G1, the in-job control (emoticon, seed 0).** r within 0.05 of E18 (0.5972); ρ_trig, ρ_shared and ρ_mean within 0.02
of E18 (0.973, 0.715, 0.896); W₀'s own values through the trig and shared builders recover ≤ 1e-6; and the moved arms
reproduce E18 within 0.02: R_shuffled against `regating_leak_amend2.json`
`tasks.emoticon.seeds.0.perm.shared.R_mean` (0.0023); R_other and R_other_W0 against `regating_leak_amend3.json`
`tasks.emoticon.seeds.0.mismatch.R` (0.887) and `.R_W0` (0.007). A fail stops the job before any new task.

**G2, per task.** The do-nothing r within 0.1 of the on-disk reference: `task_ladder/results/ladder.json`
`tasks.<t>.do_nothing.routes.shares.r` (r0_t352 0.857, r0_t588 1.074, r1_t588 0.904, r2_t275 0.533) and
`kv_task/results/kv_task.json` `tasks.<t>.do_nothing.routes.shares.r` (ident32 0.802, rand32 0.273, rand1 0.120). A
fail makes that state "reported, not scored"; the run goes on.

## Predictions (`experiments/regating_tasks/results/predictions.json`, filed before the run)

- **Ladder, per scorable state, each of the three kill conditions:** "survives". Medium-high confidence on r0_t352,
  r1_t588 and r0_t588, whose switching shares are 0.86 to 1.07. Medium on r2_t275, a rung-2 rare-target task with share
  0.53; E18's rung-2 task r2_t359 read 0.62 shared and 0.99 other.
- **Key-value, per scorable state:** two-sided. A low-confidence lean to "survives" on ident32 (share 0.80). No lean on
  rand32 and rand1, at low confidence. The CI-masked model with released subcomponents and Δ out holds no cipher
  weights, so what the switching carries there is generic. A fixed change can carry generic information. But the
  other-window donor has a different key, so a kill there reads "the switching carries the key".
- **Controls:** ρ_cross ≤ 0.1 where the null learned (E18 −0.08 to −0.30); the random null's mean ≤ 0.1.
- **Pass rule:** no kill on any scorable ladder state. Branches as in the design check.
- Every rule is absolute. **Blind to:** the trigger position only, held-out windows, one seed, one model, and the clip
  at [0, 1]. **Positive control:** G1 in the same job, and ρ_trig. **Noise floor:** E18 seeds agree within 0.02 on R;
  shuffle sd 0.02 to 0.04; derangement spread a few hundredths.

## Cost and platform

Cluster `debug`, one L40, `cluster/regating_tasks.sbatch` (`--time=00:20:00`), $0. About 25 s to load the model. Per
state: about 15 s for the two fine-tunes and `measure()`, 8 s for the shuffle, 7 s for the other-window arm. Eight
states: 4 to 6 min expected. E24's five states with the Pile stage took 143 s.

## Not run (by design)

J_nat and the Pile contrast (E29 covers what the subcomponents do); the donor correlation of Amendment 3; the
turned-up and turned-down halves; seeds beyond 0; SGD small states; positions other than the trigger.

## Code

`experiments/regating_tasks/run.py` (E24's shape); new functions in `src/commute/regating.py`: `load_kv_windows`,
`recovery`, `shares`, `shuffled_recovery`, `state_trigger_values`, `other_window_recovery` (E18's logic, moved; the
docstrings cite E18's lines). `Task.target` and `prepare`/`finetune`'s target are annotated `int | Tensor`; the int
path is unchanged. CPU tests in `tests/test_regating.py`. Figures from `plot.py`: `fixed.png`, `shuffled.png`,
`other.png`, the seven new states beside E18's three.

## Hand-off (pod-runner brief lines)

- Command: on the controller, `cd /mnt/nw/home/e.agnew/commute && sbatch cluster/regating_tasks.sbatch`. First
  `cluster/sync.sh up` from the laptop. Predictions are filed and in git.
- Inputs not in git: `experiments/transport_test/results/{windows.json, large/alive.pt, large/windows.pt}`;
  `experiments/task_ladder/results/large/windows_{r0_t352_v310,r0_t588_v320,r1_t588_v452,r2_t275_v2667}.pt` (on the
  laptop; if any is missing on NFS, `cluster/sync.sh up-large`); `experiments/kv_task/results/large/windows_{ident32,
  rand32,rand1}.pt` (NFS only). The run checks all of them before loading the model and exits naming any missing one.
- Early stop: the job exits by itself if G1 fails (`status: "stopped at G1"`); report the diff line from the log.
- Outputs: `experiments/regating_tasks/results/regating_tasks.json`, `results/large/full.log`,
  `results/large/<task>_shift.pt` (8 files, about 0.3 MB each); bring back with `cluster/sync.sh down`, and `large`
  for the log.
