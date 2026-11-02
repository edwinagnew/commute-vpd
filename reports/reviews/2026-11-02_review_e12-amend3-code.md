# Review: E12 Amendment 3 code and filing, before its run

Fresh Opus agent, 2026-11-02; it wrote none of this. Target: commit 7ff9f121 (`experiments/new_vs_existing/amend3.py`,
`results/predictions_amend3.json`, `cluster/new_vs_existing_amend3.sbatch`, `reports/plans/2026-11-02_plan_e12-amend3-fp32-read.md`).
Nothing was run on a GPU and the model was not loaded. Working files: `~/commute-data/scratch/commute-lead/e12a3-review/`.

**Verdict.** I found no bug that reads a wrong number. Every comparison is like for like on both sides ("Checked and fine" below), and the code does what the plan says. The job can run as it stands. The problems are in what the verdicts can see and in how the filing words them. M1 is worth fixing first: a likely-looking "holds, none moved" or "gone" could be read as more than it shows. The filing can still change, because nothing has run. M2 and M3 are small.

## High

None.

## Medium

**M1. No verdict reads the re-switching term or L01 itself, and two filed outcomes claim more than a fall can show.**
`amend3.py:71-79` scores only the fall and `pair_diff`. `predictions_amend3.json:77,80`, `plan:62,65`. A fall cancels whatever fine-tune and student share. The plan lists that under "blind to", but the outcome text ignores it.

- *Failure, "holds, none moved".* In fp32 the fine-tune reads −5.0 and its student −6.2 on rand32b s0, against −5.97 and −7.20 in bf16. `pair_diff` is 0, so the verdict is "holds, none moved", and the filing says "Amendment 2's numbers stand as read". Both L01 moved by 1 nat, the term grew from 6.0 to 7.0, and `term_lost` went from 20% to 17%. The per-state `diff` is in the log, but `score` does not carry it and no clause reads it.
- *Failure, "gone".* Γ in fp32 shrinks the fine-tune's term from 6.0 to 0.8, and the student reads 0.7. The fall is 0.1, so the verdict is "gone", and the filing says "the fall comes from bf16 in the read, not from the student" and corrects two reports. What happened is that the term itself went. The same wording also conflicts with `plan:86`, "not a truer value": a fall that disappears in fp32 shows the reading depends on precision, not that bf16 made it up. Γ was trained in bf16.
- *Missing control.* `predictions_amend3.json:64`'s positive control shows the bf16 side reproduces. Nothing shows the fp32 read can see a change at all. The fine-tune's fp32 term is that control, and it is already computed as a denominator (`amend3.py:132`), with no expected value and no report line.

Fix, about four lines and no new reads:
1. In `score`, add per fine-tune `term_bf16`, `term_fp32` and `L01_diff`. Expected bf16 terms from Amendment 2's stored values: rand32b 6.011, 5.770; rand32 3.751, 3.500.
2. Report beside the verdict `term_moved`, the fine-tunes whose fp32 term differs from bf16's by more than `tol`, as `moved` does for falls.
3. Change "Amendment 2's numbers stand as read" to "the fall stands; L01 and the term as `term_moved` says".
4. Change "gone" to "the fall depends on the read's precision; read it against `term_moved` before correcting any report". Change P2 "bf16 was hiding it" the same way.

**M2. "Bounded by the 0.05 stop" holds for the bf16 read only; the offsets' rounding on the fp32 read is unbounded by anything in the run.**
`amend3.py:127,135`, `plan:55`, `predictions_amend3.json:63`. The stop compares the rebuilt student with Amendment 2's stored L01 in bf16, the only precision with a stored value. The fp32 side has no reference. The rounding is also one-sided within a fall: students carry it and fine-tunes do not (E46's files are float32, read below).

I do not expect it to decide a verdict. This is arithmetic, not a run:
- Offsets have a median size of 1.7e-4 and bf16 rounds each by at most 0.39%, so the noise on dW is well under 1% of dW.
- bf16 rounding of W₀ + dW is much larger. With dW rms measured at 2.5e-4 (rand32b s0) and 1.4e-4 (rand32 s0), and W about 50 to 100 times dW (the 1 to 2% in `reports/` plans, not measured here), its rms is about 3.5e-5.
- That is about 14% of dW on rand32b and 25% on rand32, and Amendment 2's hook pair put its effect on L01 at 0.0069 or less.

So the claim is wrong in wording and probably harmless in size. Note also that bf16 rounding is a larger share of the change on rand32, so P2 is where a move is more plausible.

Fix: reword to "the stop bounds it in bf16; in fp32 it is not read" and add that to Not run. If a state lands within a few hundredths of a line or `moved` fires, there is a clean test for $0: Amendment 2's retrained students matched Amendment 1 with differences of exactly 0 (`.checks` in `new_vs_existing_amend2.json`), so retraining a student on the same card recovers its exact fp32 dW (about 115 s each). The plan's "nothing is trained" need not be final if a number sits close.

**M3. The check that the fp32 path ran is weaker than E62's, and cannot show that Γ was the part in fp32.**
`amend3.py:109`; `reports/analyses/2026-11-02_analysis_e62-amend5-head-sweep.md:67,203`. The floor above 0 passes if either of the two parts differs. Failure: an outer autocast or a later edit leaves Γ in bf16 while the masked forward is fp32. The floor still reads above 0, and "holds, none moved" is then read as Γ in fp32, the one thing this amendment exists to test. E62's Amendment 5 forward falls moved by only 0.002, so the forward alone could produce exactly that result.

The current code does not have this fault. `gates_fp32` has no autocast in scope, `param_decomp/models/` has no autocast (grep), and the same function reported `gate_max_abs` 0.832 in E62. So this is a missing control, not a wrong number.

Fix: have the two read functions return their gates, and record per state the largest and the mean |g_fp32 − g_bf16| over alive subcomponents at the trigger. Stop if the W₀ value is exactly 0. This also separates "Γ did not move" from "Γ moved and the mean did not care", which is the first question a moved state would raise, and it may save the mixed reads the plan keeps for later (`plan:90`).

## Low

- **L1. P1's power check tests only its upper line, so "gone" can print with `tol` above 0.15.** `amend3.py:48-52`. A floor of 0.1 gives `tol` 0.21: P1 still reads "gone" at |fall| ≤ 0.15, while P2 on the same floor reads "no power". "Gone" is the outcome that overturns a report, so it should need `tol` under `gone_line`. Fix: return "no power" for that branch, or file it as a condition on the clause. Needs a floor of 0.07 or more, above most of the guessed range.
- **L2. A nan on the second table slips past the floor.** `amend3.py:107`. Python's `max` drops a nan unless it comes first, and `rand32` comes second, so a nan `L00_fp32` on rand32 leaves a finite floor and passes the stop at line 109. rand32b's nan does stop. The later falls would probably be nan and read "unscorable", so this costs little. Fix: check every `diff` is finite.
- **L3. "Mixed" wording does not cover what the rule sends there.** `predictions_amend3.json:79,83`; `plan:64,68`. All four rand32b falls at 0.3 read "mixed" under "on some states the fall is under the line", when in fact it shrank on all of them and a third remains. On P2, both states between the lines, or a rise on either, read "mixed" under "One rand32 state crosses a line". The rules are fine; add one sentence to each clause.
- **L4. The lean's evidence is for the forward, not Γ.** `predictions_amend3.json:62`, `plan:52`. "E62's fp32 masked falls agreed with bf16 to 0.002" ran under Γ's bf16 masks (E62 Amendment 5 analysis, line 98). The figure for Γ is line 165: 0.0008 and 0.0027, on head (2,4)'s fall on intact windows, not on a change in weights. Leans are not scored; correct the citation before it is quoted.
- **L5. A stop on the first student gives up five free states.** `amend3.py:135`. `reload_diff` has no measured value and the job costs about 20 s more to finish. A miss at 0.051 on rand32b s0 ends the run with one state read and a rerun needed. Fix: read all six, then stop on the worst.
- **L6. `tol` is not a noise floor for a fall, only a size to compare against.** `amend3.py:39-41`, `plan:33`. It is measured at W₀ with no dW, on a signed mean that can cancel, and its guard comes from a different pair. The filing says so ("Not a noise model"), and `moved` only reports. But `moved` could fire at 0.03 nats on a fall of 1.2 with batch-order noise of 0.09 beside it. The analysis should quote the size, not just the label.

## Checked and fine

- **Both sides of each comparison.** Same windows, batch 64 and whole 41-token windows (`route_cells`' `ci_len=None` against `tokens.shape[1]`), trigger 20, table, `target_tokens`. Same log-odds: `ST.logodds_at` wraps `identity.log_odds`, both float on logits and a float64 mean. Γ is called the same way (`sampling="continuous"`, `detach_inputs=False`, `lower_leaky`, one chunk per batch), model in eval.
- **State of the model in each read.** Both read Γ inside `model_state(target=W₀ + dW)` and run the masked forward after it exits, with released components throughout. So both are L(C₀, G₁). Fall is fine-tune minus student in one precision, `pair_diff` is on the same rebuilt student in both, and `term_lost` uses one precision on top and bottom.
- **No write to Amendment 2's file.** `amend1.OUT` is set after `import amend2`; `A2.OUT` is only read and hashed.
- **Names and order.** `FALLING`, `REFERENCE` and the loop give Amendment 2's six states in its order; each `fine_tune` link is right for the `_o1` states. Every key read from Amendment 2's JSON exists.
- **Filed numbers against the stored JSON (jq).** L00, the four fine-tune L01, the six falls, 16% to 20% and 2%, hook gap 0.00686, batch-order 0.09: all match.
- **Files (read on the worker, no model).** E46's dW: 24 matrices, float32. Offsets: 24, bf16, step 1000, order and task as named. Windows: 97 × 41, half 20. Gates at 0.62 GB in fp32 agree with `plan:76`.
- **`offsets_state` against the trainer.** The same U₂V₂ᵀ − UVᵀ as `Offsets.delta_w`, in float64 instead of float32.
- **Code drift.** Since Amendment 2's commit 8d69671c, the only change under `src/commute` or to amend1, amend2 and the dependency files is `copying.sweep_reading`, which this job does not call.
- **Rules against the filing.** `p1`, `p2` and `moved` follow the filed order and ties. Only L1 and L2 above are gaps. The filed edge run reads as stated; I read its output and did not rerun it. The filed "under 4 states: unscorable" cannot arise in the job, since a missing state stops first; harmless.
- **Stage order and job script.** Control first and alone, written before the next state; same shape as Amendment 2's job; `--time` 15 min against about 2 min.

## Not checked

Any GPU forward. The target model's parameters being fp32, which rests on `vpd.target_weights` and on E62's fp32 path having run and differed, not on printing the dtype in the checkpoint (M3's check would settle it). What the fp32 numbers will be.
