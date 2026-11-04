# Plan: can VPD's adversary fail anything on Qwen3-1.7B? (E85)

*2026-11-04, D1 (a map on Qwen3-1.7B that keeps the decomposition faithful for a real update, judged by a test that can fail a bad placement). Under the gate, fast lane, about $10.6 and 2.6 h. Rules `experiments/qwen_d1_teeth/results/predictions.json`, `rules.py`; advice `reports/decisions/2026-11-04_decision_d1-after-e80.md`, option 1.*

## In plain words
VPD's adversary passes doing nothing on the first fine-tune update, so we do not know that it can fail anything on this model. We read it on two states built to be bad, on the same rows, settings and code as E80's last amendment. If one reaches the fail bar, the adversary has teeth and stays in D1's table; if both pass, that it cannot fail a bad state here becomes a measurement.

## The states, in the Terms' words
- **Random in Δ.** Weights W0 plus a Gaussian change at u1's Frobenius norm per matrix, subcomponents unmoved, so Δ holds it and u1 is nowhere. E75 Amendment 1's draw, seed 75100 plus the matrix's index.
- **Rarely-on half.** Weights W0 + u1; the least-squares split of u1 over the half of the alive subcomponents least often on (W0's maths-row statistics) moves those columns; Δ takes the remainder.

**What happens to the remainder.** E75 has two states under this name. Its clean table's row (E72's `clean_lower`) has weights W0 plus the half's part, so the remainder is discarded. The state read under masks (`qwen_maps.site_change("lower")`, E68, E72) keeps the remainder in Δ. This run uses the second, as asked. So behaviour kept is 1.00 by construction, a build check that cannot read the placement.

**How much that is.** No file holds the plain share. Input-weighted it is 0.507 of u1's energy (E72's norm ratio 0.712, squared); scaling by least squares' 0.090 plain against 0.132 input-weighted gives about 0.35. The run reads it in its first minutes and stops unless it lies strictly between 0.090 and 1.

## Design check
- **Goal.** D1's test, under `GOALS.md`'s weak version (Δ may not absorb the update): does this test have teeth?
- **Nearest tried.** E80's amendments: every placement passes, and no state built to be bad was read ("Not run" there). E75 fails these directions on the clean table, never under the adversary.
- **Patch or step.** A patch that closes a line; class 2 (the same harness, the controls named as missing).
- **Decisions.** Relies on ask 24 (the clean table is D1's table); departs from none.
- **What changes next.** Teeth: E80's four states get a third seed, the lead's call. Cannot fail: the adversary leaves D1's table as a measurement, and the question of restating D1 goes to the user with this number. No call: reported as read, no amendment.
- **Control first.** W0, seeds 0 and 1, re-read on this host, each a stop; can fail on the host, the rows or a state left applied. Before them, gold W0 and the fine-tune against E40's readings.
- **Saved.** `results/results.json`, `run.log`, under 50 KB each. No state files, against the advice's line: both rebuild in about a minute from the seed and the inputs.

## Numbers and what each should read

| number | W0 | do-nothing (E80, quoted) | random in Δ | rarely-on half |
|---|---|---|---|---|
| Faithful, mean of two seeds over W0's same seed | 1 (absolute 1.703, 1.438) | 1.24 | about 1.24, range 1.0-1.4 | about 1.2-1.3, near in-span's 1.28 |
| Behaviour kept | 0 | 1.00 | about 0.008 (E75's file, maths rows) | 1.00 by construction |
| Left in Δ, plain share | 0 | 100% | 100%, G's size not u1's | about 35% |

W0's spread, this job's two starts, about 18%.

**Lean, and what stands against it.** The advice expects both to pass, and I file that at medium. Random may read nearer 1.0 than 1.24: E75 has it hardly moving the clean output. The rarely-on lean is low: its moved columns carry 4.4 times u1's input-weighted energy (E72's 6364 against 1440), and the adversary can raise exactly the columns whose importance is near 0.

**Outcome rule, on the means; bars pass ≤ 1.40, fail ≥ 1.61, as E80 used them.**
- Either state at or over 1.61 with its gap (mean − 1) strictly above W0's spread: "the adversary has teeth on this model".
- Both at or under 1.40 and no single seed at or over 1.61: "the adversary cannot fail a bad state here". The seed clause is added to the asked rule, because this outcome closes a line and a pair like 1.7 and 1.0 has not shown it.
- Everything else read: "no call", reported as read. Edge cases are in `tests/test_qwen_d1_teeth.py`.

The band is W0's spread from this job. At the fail bar it bites only above 0.61, so at 18% it decides nothing. The null, do-nothing, is not re-read in this job (six starts, as asked).

**Blind to.** The ratio reads web rows, is a lower bound still rising at step 20, and sees a placement only through masks that pass 0.0004-0.013 of a column change's clean KL; a pass does not make a placement good. Kept reads weights only. **Noise floor.** A state's two ratios differ by 0.12-0.19 at Amendment 2, so a mean of two carries about 0.1; same seed across hosts matched to four places.

**Departures from E80 Amendment 2.** W0 re-read here rather than taken from its file; one pod, so no host check; no Σ or adjoint scales, which neither state uses. All settings otherwise its own.

## Cost, verified from files
Each start 1316-1333 s (`amend1.json`, `amend2.json` `.starts`), peak 66.0-66.6 GB, W0's 60.4 GB. H200 NVL on Vast at $4.06/h (E80 plan), what that pod was charged; offers vary.

| stage | time | $ | stop |
|---|---|---|---|
| set-up (Amendment 1's pod: placed ~09:55, run began ~10:15) | ~20 min | 1.35 | an input's sha256, alive count, card under 70 GiB |
| load, X, left in Δ, kept (E80's reads: 7-14 s each, X 47 s) | ~4 min | 0.27 | any check in `predictions.json` "expected" |
| W0 s0, W0 s1 | 44 min | 2.97 | either over 20% from 1.703, 1.438 |
| random s0, rarely-on s0, random s1, rarely-on s1 | 88 min | 5.95 | a first step not finite |
| total | ~2.6 h | ~10.6 | |

A build bug costs about $1.60, since the fetches come first.

## For the pod-runner
One 80 GB card, H200 NVL on Vast (RunPod refused E80), container disk 120 GB (decomposition 53.7, δW 14.1, X 5.6, inputs 0.44, venv), `WANDB_API_KEY` and Hugging Face access on the pod. Nothing outside git and W&B: sync the tracked repo; this run touches no `results/large`.

Set-up, E80's three fetches beside `--decomp`, with `uv sync` alongside (the run keeps what is complete and fetches what is not):
```
PYTHONPATH=src uv run --no-project --with wandb==0.23.1 python -m commute.qwen_files /root/qwen
PYTHONPATH=src uv run --no-project --with wandb==0.23.1 python -m commute.qwen_e40_files fetch /root/e40_nowarmup v0
PYTHONPATH=src uv run --no-project --with wandb==0.23.1 python experiments/qwen_d1_adversary/upload.py fetch /root/e80_inputs v0
```
Command, repo root, background subprocess, resume by the same command:
```
.venv/bin/python experiments/qwen_d1_teeth/run.py --decomp /root/qwen 2>&1 | tee -a experiments/qwen_d1_teeth/results/console.log /proc/1/fd/1
```
First values in `results/run.log`: `[check gold_W0]` 2.49521; `[check gold_FT]` ~1.81; `[left in delta]` Rand 1.0, Rare ~0.35; `[check kept_Rand]` ~0.01; `[check kept_Rare]` 1.00; `[W0_W_s0] seed 0 step 0: KL 0.0918`. **Report the build numbers at once.** Download `results/results.json` and `run.log` after every start; a lost pod then costs the current start, and a same-model card resumes from them. Another card model: tell the lead. A stop writes `results/stopped.json` and refuses a rerun. Terminate when the final `results.json`'s sha256 matches both sides, before anything local.
