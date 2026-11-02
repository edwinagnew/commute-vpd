# E62 Amendment 5: code review (commit 685faeb)

A fresh read of `experiments/induction_cue/amend5.py`, `copying.py`'s last section, the new tests, `predictions_amend5.json`, the sbatch script and the plan's "Amendment 5" section. The code does what the plan and the filed rules say, line by line, and I found no bug that would make a read wrong or stop the job. One filed rule has a real hole: `rerouted` can fire on a head the masked forward does not lean on. Fix it before the run, since nothing has run and no cost is attached.

## Findings, ranked

### 1. Material: `rerouted` has no floor on M, so a suppressor head can be a "carrier"

`src/commute/copying.py:1013`, and the filed text in `predictions_amend5.json` `stage_c_verdict` and the plan's verdict 2. A carrier needs M - T >= 1, M - T >= 2 sds and M - Mb >= 1. Nothing asks that M itself be positive.

Failing scenario, run on the filed thresholds (`~/commute-data/scratch/regating-lead/e62a5-review/probe.py`):

- T -3.0, M -1.5, Mb -3.0 gives `rerouted`, carriers `["3,5"]`. Switching the head off raises B's log-odds in both forwards.
- T -1.2, M 0.0, Mb -1.1 gives `rerouted`. The masked forward does not depend on the head at all.

In both, the mask has only weakened a head that pushes against copying in the target. The filed gloss, "the masked forward's copy depends on that head by at least a log-odds more than the target model's does", is false for them. The verdict is first-firing, so it would override every fraction verdict and send the lead to the `rerouted` branch: a state-level test anchored on the "carrier", and a Qwen caption saying the forward reroutes around its labelled head.

The case is not exotic. The plan's own edge cases allow a head off that raises log-odds, and copy suppression is a known head behaviour; no per-head T at the second A has ever been read, so nothing rules it out. The tests cannot catch it: the grid has `t` in 0, 0.3, 0.5, 2 only (`tests/test_copying.py:933`) and writes the same rule out again (`:944`). The one negative-T test (`T -3, M -3`) has no excess.

Fix: add `M >= carrier_min` to the carrier rule, code, predictions, plan, and a test with T negative. It is a one-clause change and every filed positive example still fires (the tie case M 1.25 and the T 0.0, M 1.0 case included). The predictions are committed but no run has read them, so the edit breaks no pre-registration. Record it in the plan as a pre-run correction.

### 2. Real, not blocking: the three fraction verdicts depend on one marginal head (`fall_floor`)

`copying.py:1014, 1020, 1023`. `frac` admits a head at T >= 0.5, and `thinned` and `labelled_head_singled_out` each need every admitted head to agree. Run on the filed thresholds: heads (2,3) T 2.0, M 1.9 and (3,5) T 2.0, M 1.7, with (1,1) at T 0.50, M 0.05, read `mixed`. Move (1,1) to T 0.49 and it reads `labelled_head_singled_out`. No fraction's noise enters the rule either: at T near 0.5, a window sd of about 1.5 on the fall gives a fraction sd of the order of the 0.24 gap. `thinned` can also rest on a single comparison head (filed, tested), while its name says "every head". See "The three questions" for what to do.

### 3. Disclosed, but the verdict names read past it: fractions have no pair guard

`copying.py:1014`. `rerouted` has M - Mb. The fraction verdicts use T and M alone, so a head with T 2.0, Tb 2.0, M 1.9, Mb 1.9, whose fall owes nothing to the pair, counts as one of "the other heads' parts" of the copy, and can yield `labelled_head_singled_out`. The plan's "Target's own reading" says T is an intact-windows number, and `pair_part_T` and `pair_part_M` are printed per head, so no number is wrong. The write-up must read each fraction verdict with those two columns beside it, and say so if the comparison heads have small pair parts.

### 4. Minor: the control's pass is one-sided, and a plan sentence leans on it

`copying.py` `all_ones_control`, plan line 768. A fall of 1.25 passes, as filed from the advice. The plan then says a pass means a low fall under W₀'s mask "comes from the mask". With an all-ones fall of 1.3 against M 0.92 that would be false: the mask would cost little. The code prints `fall_over_target`, so nothing is lost if the write-up reads it against "near 4.82" and does not stop at `pass`. No code change.

### 5. Minor: a plan sentence is slightly wrong

The plan's Noise floor section says no rule leans on the bootstrap but the 2 sds. `precision_moved` leans on it too (`amend5.py:164`). It only prints a loud line, and a too-small sd makes that line fire more often, which is the safe direction. Nothing to change in code.

## The three questions

- **`fall_floor` 0.5, `carrier_sds` 2 unsourced.** `carrier_sds`: not a real problem. Two window sds reach the 1.0 bar only at a per-window excess sd of 7.07 log-odds, so 1.0 does the work, as the plan says. `fall_floor`: real for `thinned`, `labelled_head_singled_out` and `mixed` (finding 2); no effect on `rerouted`, nor on `unreadable`, since (2,4)'s T is pinned at 4.82 by the re-read. Being unsourced is not itself a breach: "every setting sourced" is a gate for gated runs and this one is not gated. No rerun is needed to handle it: `amend5_windows.pt` holds every per-window read, so the verdict at floors of 0.25 and 1.0 can be computed afterwards and reported beside the filed one, deciding nothing. If the verdict changes among those, report it as the fraction verdicts not being separated.
- **Bootstrap over windows sharing 32 keys.** Measured: 200 windows, 32 distinct As, 3 to 11 windows each. If a key's windows were identical the true sd would be at most 2.6 times the bootstrap's. Not a real problem for the verdict at the expected sds, and disclosed. It bites only for a carrier at the margin: with that multiplier, 2 sds reach 1.0 at a per-window excess sd of about 2.7. If a carrier's excess reads under about 1.5, bootstrap over keys from `amend5_windows.pt` and `windows.pt` before leaning on it. The sd beside (2,4)'s miss from Amendment 4 is a lower bound in the same way and should be labelled so.
- **`gates_fp32` and `masked_lo_fp32` run only on a tiny CPU model.** Not a real problem. They decide nothing. `masked_logits(autocast=False)` has run on GPU before (`spread.py:70`, `cancel.py:317`), `param_decomp`'s model code holds no autocast or bf16 (grep), and the gates go to CPU like `route_cells`'. A crash would come at stage b, after stage a's files are written, at $0 and about a minute, though it would stop the scored sweep too, and the rerun must move the outputs aside. The silent failure to look for is autocast still being on: read `fp32_minus_bf16.window_max_abs_on` and `gamma32_minus_bf16.gate_max_abs`, which should both be above 0 on the GPU. The CPU test cannot show this, since autocast is off there.

## Checked and found correct

- `Reads5(half - 1)` against Amendment 4's `at`: same positions for `gate_means`, `a`, `a_half2`, the reads and `second_a_holds` (`half = R.half + 1`). The re-read's keys match `stage1_control`'s, less `status`, and its numbers go through the same double-on-CPU path, so `exact` can read true.
- Devices and dtypes: `logodds_at`, `logp_at` and `masked_lo_windows` return double on CPU, so `amend5.py:142`'s subtraction and every `J.fall` are safe. Γ's gates are on CPU in `route_cells` and `gates_fp32`, so `amend5.py:168` is safe.
- Stage c: every head's row, per-window pairing, Tb and Mb against their own no-head reads, the fp32 row keys against `sweep_reading`, the labelled row's stop. Gates are read before any hook.
- Stage d: tokens and gates are the right differences and sum to the lean. `token_reading`'s arguments are in order.
- `reread_diff`, `stopped_reading`, `boot_sd`, `share_sd` (nan on a non-positive resample), `jsonable` on nan and inf. All 24 thresholds' keys used by the code exist in `predictions_amend5.json`.
- Sbatch: exit code carried through `tee`; sbatch settings as Amendment 4's but for the time limit; `compute` has L40s only, so the card matches the re-read's; `PYTHONPATH` from `env.sh` picks this clone's `copying.py`. Inputs exist. Host memory is ample: Γ's gates are about 319M entries a pass, at most about 1.3 GB in fp32, against 24 GB.
- `tests/test_copying.py`: 5205 passed on the CPU worker.

## Not run

The job itself, and the 67M was not loaded. The rules were probed on planted rows only. No per-head T at the second A exists yet, so how often findings 1 and 2 would fire on the real numbers is unknown.
