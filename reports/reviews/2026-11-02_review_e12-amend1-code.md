# Review: E12 Amendment 1 code, before its first GPU run

*2026-11-02. Fresh reader, wrote neither the code nor the plan. Scope: commit 5f55efc, `experiments/new_vs_existing/amend1.py`,
`results/predictions_amend1.json`, `cluster/new_vs_existing_amend1.sbatch`, against
`reports/plans/2026-11-02_plan_e12-teacher-route-split.md`. No model loaded, no edit to the code or the predictions. The input
probe is `~/commute-data/scratch/commute-lead/e12-amend1-review/probe.py`, its output `probe.out` beside it.*

**Verdict.** I found no bug that reads the wrong weights, windows or files, and the scoring code matches the filed rule text. What
is wrong is mainly in the filing: one rule can read "moved" with no change in the gates, and the likeliest verdict has no next step. These
cannot change after the run and cost minutes now, so amend them first. The run is $0, so little is lost if the lead prefers to run as is, but
items 1 and 2 would then need a dated report to interpret. Below, `file:line` without a directory is `amend1.py`.

## Findings, ranked

### 1. P2 can read "moved" from a student that learned less, with the gates unchanged
`moves` is `|r(a) - r(do-nothing)|` (119), and each r is over its own gain (`local_faith.py:270-272`), as filed
(`predictions_amend1.json:50`). With the masked gain from the gates fixed and gain ratio p, r(a) = r / p.

| Teacher's r | Reads "moved" (0.25) at p at or under | Leaves "inherited" (0.10) at p under |
|---|---|---|
| 0.80, copy | 0.76 | 0.89 |
| 0.27, map | 0.52 | 0.73 |

The pool admits p down to 0.5 (75), and the filed expected band is p of at least 0.7 (`predictions_amend1.json:69`). So a copy student
inside both reads "moved" by arithmetic alone. The file's `blind_to` names this ("each r is over its own gain"), but nothing guards it,
and `what_changes_next` (line 94) acts on "r moves" as a statement about the route.

It also fails the other way. E42's rand32 file: r 0.187 to 0.132, a move of 0.056, under the "inherited" line; in nats the gates' gain
fell from 4.13 to 2.74, a third (`teacher_training_e42_rand32.json`). On a map's r of 0.27 the 0.10 line is a large relative change.

**Consequence.** P2's verdict, the plan's second question, can't be told apart from "the student under-learned". **Fix.** Score the move on a
fixed denominator: `|L01(a) - L01(do-nothing)| / teacher's gain`. L00 is the same in both, and all the cells are in the record already.
A one-line change at 119 plus the rule text. Report p beside it. Consider stating map states' line relative too.

### 2. The likeliest verdict has no next step
`what_changes_next` (`predictions_amend1.json:94`) has two branches: "separates and r stays close", and "no separation, or r moves". P2's filed
lean is "partly" (line 51), which is neither. "no power", "unscorable" and P1's "reversed" are also unassigned, and nothing says what to do if
state (a) and state (b) disagree, though (b) is reported by the same rule. **Consequence.** The analyst picks the branch after seeing the
numbers. **Fix.** One clause per verdict, filed now. No code.

### 3. A rerun silently keeps records made by older code or another config
The skip tests are only "the name is in the file and the offsets exist" (204, 234, 248). `main` checks the card against the file (280) and the config
against the predictions (268), but not the file against either, and it records no commit. **Consequence.** The likeliest rerun is after a stop and a fix.
A fix to a read or to a threshold then reuses earlier reads, and the results file can't show it. E46 has the guard: `git_head`, `code_unchanged`
(`regating_table/run.py:78-84`). **Fix.** Write `config` and `commit` into `out` at creation and refuse a mismatch, as with the card. About three lines.

### 4. C1 trains the same state twice
`ident32` and `rand32` have the same train and held-out tokens (`probe.out:37`, key rank 100 and window seed 0 in both,
`anchors.py:30-31,61`). They differ only in the table. `teacher_loss` never reads the table (`teacher.py:40-53`), and in `train` the table
reaches only `eval_arm_a`, under `no_grad` (`component_training.py:154-159`). Same start (237), same teacher, same batch order, so C1's second
training repeats the first. **Consequence.** About 105 s and 60 MB wasted, and C1 is one draw read under two tables, not the two tasks
it reads as. **Fix.** Train once, read under both tables. It removes a file from the 12 and a few lines from `c1`. Say "one training, two tables"
in the filing. Reading under both is still worth keeping, since the terms are log-odds of different targets.

### 5. "no power" outranks "moved"
`r_class` tests noise before moves (92-95), as filed ("Order fixed", `predictions_amend1.json:81`). Batch-order noise of 0.06 with a move of 0.5 reads
"no power", though the noise can't explain the move. **Consequence.** A large real move is hidden by a small noise excess, on two pairs. No reason is
filed for the order. **Fix.** "no power" only if the largest move is not above the largest noise by some stated factor; else classify the move.

### 6. The C1 stop's line and text do not fit the only reading of this start
Amendment 8 did read the route on this start, on emoticon windows, though only its log kept it
(`teacher_training/results/large/amend8/run_n1.log:142,163`): gain 0.07 nats at the start and 0.20 at step 1,000, so an r term of about 0.04 then 0.08 nats and an e_gate term of
about 0.02 then 0.06. Three things follow. The stop will likely not fire on noise, if these windows behave like emoticon's: 1.5 nats is about 20 times those terms. By the same token, it would
pass a recipe writing ten times as much. And the plan's reason for reading nats, a random move worth "about 1 nat" that can pass `MIN_GAIN`, is not what the log shows: its 1 nat comes from
mean P(target), while the mean log-odds gain is 0.07. The filed text also says the start is "a random move, not training" (`predictions_amend1.json:75`), yet the end state holds nearly all of
that move (Amendment 8: about 1% undone, `reports/analyses/2026-11-02_analysis_e34-amend8-base-control.md:339`). **Consequence.** Small. C1 is a weak stop, and it can't separate training from the move
if it does fire. **Fix.** Report `end - start` beside `end`, which is already derivable, and correct the "about 1 nat" reason in any write-up. I would not change the line; its reason from the classes' gap is sound.

### 7. An unnamed difference: s0's students share their teachers' first batches
The fine-tune's seed is the state's seed (`regating_table/run.py:294`, `anchors.py:78`), and batch order 0 is the same generator and call
(`finetune.py:197,201`, `component_training.py:144,154`). So s0's students share the first 10 (or 30) batches with their teachers, s1's do not, and
the order-1 noise states do not. It is a small part of 1,000 steps, and Amendment 8c read shared batches as not mattering on cosines (within 0.008),
but within-class range and the noise floor both straddle it. **Fix.** One line in `departures`. No code.

### 8. Small
- `dump` overwrites the results file in place (137). A kill at the walltime during a write leaves JSON that `main` can't load (279). Write a temp
  file and rename: one line.
- The sbatch's 25-minute estimate takes 105 s per student from 73 held-out windows; here 97 are evaluated at every step, so somewhat more. The 40-minute
  limit and the resume cover it. No change.

## What could be deleted

The file is 297 lines against the plan's "under 200". About 55 can go. I do not see 200 without losing rule text that belongs beside the job.

| What | Where | Lines | Note |
|---|---|---|---|
| `Amend1` dataclass and the config assert: read the filed `config` into a namespace, so there is one source and nothing to compare | 37-63, 268 | ~25 | The "one config" rule is for smoke paths; this has none |
| Both share-sum checks: telescoping algebra (`local_faith.py:270-272`), so they cannot fail except on NaN, and NaN stops at the gain ratio check beside them | 258-259 | 2, and the filed clause | A check that cannot fail tests nothing |
| `terms`: repeats `route_split`'s dict at 270-271 of `local_faith.py`. Have `route_split` also return `terms`, which breaks nothing | 218-222 | 5 | Needs a one-line shared change; the plan says none, so the lead's call |
| C1's second training (item 4) | 233-243 | ~4 | |
| Reported doubles: P3's `noise_e_gate` (P1 has it, and it holds both pairs where line 62 of the filing names one), `noise_e_gate_b`, `per_state` (rebuild from `e_gate`) | 123-126 | ~2, and shorter lines | |
| Docstring, down to what isn't in the filing | 1-8 | ~4 | |
| The `near` pair at 210-211 and 261-262 could be two of the four: do-nothing's identities read exactly on the L40 in E42's files, and `r(b)` and `U1` hold by code | 210, 262 | ~2 | The filing keeps them on purpose; low value, your choice |
| `teacher_dW`, inline | 195-196 | ~3 | |

## Checked and found right

- **State (a):** target W₀ + the trained fp32 change, components the same trained U′, V′ (175-180, 192, 252). Δ is fixed by construction: the change is
  (U+dU)(V+dV)ᵀ − UVᵀ over the alive columns (`component_training.py:58-61`), and the layouts agree (`alive_uv` against `components_with`, 90-91, 102-103).
- **State (b):** the same components, target W₀ + the teacher's change (252). So gain and gates are the fine-tune's. The offsets saved at bf16 are never read
  back for a student. C1's start is the one bf16 read, and it is named in the filing.
- **Teacher:** `teacher_net` copies the target with W₀ + δW and restores (`teacher.py:24-28`); it is built before training, and hooks are removed after (`component_training.py:176-177`). Argument order into
  `teacher_loss` and `train` is right. The loss is E42's (`teacher_training/run.py:184-191`), with arm A's weighting at `half` and default clip.
- **Inputs (`probe.out`):** every window file has `train` 403 × 41, `heldout` 97 × 41, `half` 20, a CPU table handled by `target_tokens`. Teacher files are plain
  dicts of 24 fp32 matrices. `0.pt` has 24 matrices with the same keys, its offsets the alive shape (122 columns on the matrix checked), and it is step 0's state
  (`amend8.py:385`, 305). `stage0a.json` has every key the code reads.
- **C0 passes on what E46 already read:** least-squares e_gate on E46's own cells gives gap 0.331, within 0.086 (filed: 0.33, 0.09). r misses E12's by at most
  0.0013 against 0.03. W₀ cells differ from E12's by at most 0.011 against 0.02. The smallest teacher gain is 13.16, as filed.
- **Identities at 1e-6:** E42's L40 files read do-nothing r = least-squares r and e_gate = −r exactly, on both tables read.
- **Rules:** `separation` is E12's rule (`new_vs_existing/run.py:398-406`) with the minimum class size and the reversed branch, as filed. Pool, `r_class`, `masked_class` and
  `masked_majority` match the filed text at each boundary I traced (0.05, 0.10, 0.25, 0.5, a ratio of exactly 0.5, an empty pool, a lost pair, NaN filtered before `max`).
- **Stops:** each filed stop is in the code, in the filed order. A NaN stops (a `<=` on NaN is false). A rerun re-fires any stop from the file, so a stop can't be passed silently. Exit 1 survives
  the sbatch's pipe (`pipefail`, 29). `config` in the file equals the dataclass.
- **Environment:** the lead clone has `.venv`, `cluster/logs`, `artifacts/vpd` and the `results/large` symlink. Debug's limit is 2 h. `make lint` is clean on the file.

## Not checked

- Any of it on a GPU. Nothing was run.
- `param_decomp`'s forward and Γ directly. The identities rest on E42's files showing them exact, not on reading that code.
- `split.split_state` beyond its use in E46.
- Whether E12's windows equal E10's, which E42 used. The filing's `from_E42` does not mention windows.
- sha256 of the inputs; `main` does this at start, before the model loads.
