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

# Amendments 1 and 2 (2026-11-04), E85's last under the two-amendment rule

*Asked by the lead after the independent check `reports/reviews/2026-11-04_review_e85-adversary-check.md`, its "Measurements" 1 and 2. Rules `experiments/qwen_d1_teeth/rules_amend.py`, `results/predictions_amend1.json`, `predictions_amend2.json`. **Gates: under the gate**, about $5 and 75 min, and $0. No pre-run review, by the fast lane. Run Amendment 2 first: its clean read tells whether Amendment 1's state is big enough, for $0.*

## Design check
- **Goal.** D1's test, under `GOALS.md`'s weak version: E85 read no state known to be unfaithful, so whether the test can fail anything on Qwen3-1.7B is open.
- **Nearest tried.** E85 (two states, neither bad in what the test reads) and E75 (random fails the clean table, never read under the adversary).
- **Patch or step.** A patch that closes a line; class 2, the control the check names. Departs from no decision.
- **What changes next.** See "What each outcome means for D1".
- **Control first.** W0's gold, its read at m = g with Δ off, and a host check, each a stop; can fail on the host, the rows or a state left applied. Reproduces nothing else.
- **Saved.** `results/amend1.json`, `amend2.json`, the logs, each under 200 KB. Every state rebuilds from a seed; no state files.

## Amendment 1, positive controls
**Stage 1: Rand30.** Weights W0 + 30 G, G being E85's random change, in Δ. E85's seed 0 start. **Stage 2, only if stage 1's final ratio reaches 1.61: W0 with U's alive columns permuted** within each matrix, seed 85200 + the site's index. Dead columns stay: they carry no g on any row.

**W0: reused on an H200 NVL, fresh on any other card.** Three H200 NVL pods read seed 0 to every digit, so a start would spend $1.50 repeating it; three steps still run as a check. An H100 PCIe read 1.7517, 2.8% off, or 0.045 of ratio near a bar, which is why any other card reads W0 first, at about $1.50 to $2 more.

**What the permuted state does, from the code.** Weights and V are W0's, so clean logits, activations, h and g are exactly W0's and only the masked forward differs: Σ h (m - d) U_π + d W x. Δ = W - V U_π does not absorb the error: only d = 1 with m = 1 reads clean, and the adversary moves away from it. Off the model, on 21 sampled matrices (`~/commute-data/scratch/qwen-decomp-lead/writer-e85-amend/perm_sizes.py`, `.out`):

| ‖V(U_π - U)‖ / ‖V U‖ | range | u1 on the same scale |
|---|---|---|
| all alive columns | 1.29-1.61 | 0.0012-0.0036 |
| the rarely-on half | 0.45-1.05 (the half holds 10-59% of V U's plain energy) | |

**So the permuted state is degenerate: trivially failed.** It is the case the brief asks about, though not because Δ absorbs anything. V U_π has nothing to do with V U. Masks at g with Δ off fail it with no adversary, and step 0 should already read over 2.74 (1.61 × 1.703). It tests the harness and not the search, and no fraction of it reaches a placement's scale: one swapped pair is tens of times u1's size.

**Smallest fix, coded behind `--perm rare`, the lead's choice before the run:** permute only the rarely-on half. Same weights, V, code and cost. m = g sits near 0 on those columns, so masks at g with Δ off should stay near W0's, while a source can raise them. It is the state where only the adversary can see wrong columns, and it can pass or fail. Both states' values are filed.

**Numbers and what each should read** (each a stop if missed, except where marked).

| number | W0 | Rand30 | Perm (all) | PermRare |
|---|---|---|---|---|
| gold | 2.49521 | above it, logged | W0's | W0's |
| clean KL against W0, 128 web rows | 0 | 0.01-2, unmeasured; stop under 0.2 | 0 | 0 |
| KL at m = g, Δ off | 0.1417 ± 25% | about 0.14 + clean KL, logged | several nats; stop under 0.5 | near 0.14; stop under W0's - 0.02 |
| ‖V(U_π - U)‖² / ‖V U‖², pooled | | | about 2; 1 to 3.5 | about 0.8; 0.1 to 1.5 |
| step 0 | 0.0918 | 0.085 to 0.0918 + 1.5 clean KL | over 2.74; stop under 0.5 | at or above 0.0918; stop under 0.085 |
| later steps | 0.1131, 0.1376 (host check ± 2e-4) | step 2 above step 0 | logged | logged |
| final | 1.7032 | 1.70 + about clean KL | several nats | unmeasured |

Ratio and excess in nats are in the log at every step and at the final read.

**Filed rule.** Stage 1: at or over 1.61, "the test can fail at this size"; at or under 1.40 with a clean KL of at least 1.039 nats, "the adversary does not fail even this"; at or under 1.40 with less, "no call: the state moved the output less than the fail bar asks"; between, no call. **The clean clause is added to the asked rule.** The bar asks 0.61 × 1.703 = 1.039 nats, and a Gaussian whose removal costs less cannot be expected to reach it. Qwen's own bars 1.267 and 1.776 are reported beside, never ruled on. Bars inclusive, as E85.

**Lean: low that Rand30 reaches the bar.** The ratio should be about 1 + clean KL / 1.70, so the bar needs a clean KL near 1. The check's "18-100 times" is in input-weighted units; this 30 is Frobenius. The square law on E75's search leaves 30 times anywhere from 0.01 to about 2 nats. Amendment 2 reads it at 1, 30 and 100 times.

**Blind to.** One seed, web rows, a lower bound. Rand30 reads the Δ dial's removal of a change to the output, not where an update sits, so a fail does not say a bad placement of a real update fails. KL at m = g cannot see PermRare's columns. **Noise floor.** A state near W0 reads within 0.3-1.5% of it at the same seed (E85).

**Cost, H200 NVL at $4.06/h.** Under $6 on this card.

| stage | time | $ | stop |
|---|---|---|---|
| set-up, as E85 | ~20 min | 1.35 | as E85 |
| load, gold, W0's clean reads, host check | ~6 min | 0.40 | above |
| Rand30's clean reads, start | ~24 min | 1.60 | above; a bad build costs ~$1.80 |
| stage 2's clean reads, start | ~24 min | 1.60 | above |
| total | ~74 min | ~4.95 | stage 1 alone ~$3.35 |

## Amendment 2, rounding
On the first 8 web rows, one job. g is the code's name, `site_importance`'s output, clipped to [0, 1], what the masks read; the sum before the clip has no name in the code and is reported beside it as `before_clip`. The rule reads g. **The brief's "before the clip" and the code's g differ, so both are read.**

fp32 arm: h and the readout in fp32 from the fp32 V, through `qwen_vpd.readout` itself given an fp32 view of Γ's readouts, so nothing rounds V or h; taps and the backbone stay bf16. **`qwen_vpd.py` is unchanged.**

| mean \|Δg\|, on the half's columns | reads | expected |
|---|---|---|
| f1, W0 read twice, the second through `importances`, the code's whole path | the repeat floor, and that this read is that path | 0; a stop if over f2 |
| f2, W0 fp32 against as run | the rounding floor | above 0, a stop if 0; order 1e-5 to 1e-3 |
| b, Rare fp32 against Rare as run | the rule | about f2 |
| A, Rare against W0 | how far the state moves g | f2's order or above |
| V held at W0's, against W0 | activations alone; is do-nothing's g | most of A |
| Rare fp32 against W0 fp32 | what g hears without rounding | near A |
| Rand against W0 | | well under A |

**Filed reading.** floor = max(f1, f2). b at or under 2 floor, "rounding is not a limit: question closed". b at or over 5 floor, one fp32-V adversary start is the follow-up, not run here, and being past the two amendments it is a new plan. Between, no call. The factors are chosen here; the row spread is read beside them. **Blind to.** The clip hides a move while g stays at 0, so the share of pairs over 0.01 and the per-site read sit beside it; taps stay bf16; "closed" says a moved V rounds as an unmoved one does, not that g hears the placement well.

**Fits an L40: yes.** Target 6.9, Γ 17.1, two sets of weights 5.6 each: 35.2 GB fixed, plus g 1.41 twice, inputs ~0.8 and logits ~0.3 three times per row: about 40 GB of 46. V and U stay off the card. These add E85's measured sizes; the job itself records the peak. Host ~46 GB, `--mem=56G`. Also read, for $0: KL against W0 of the fine-tune (0.01-0.2, a stop) and of E85's Gaussian at 1, 30 and 100 times.

## What each outcome means for D1
- **Rand30 fails.** The test can fail at this size, and E85's sentence becomes "these two states do not reach it". Stage 2 then says whether wrong columns are among what it fails.
- **Rand30 passes with enough clean KL.** Something close to "cannot" earns its place, with this as evidence: the adversary leaves D1's table as a measurement, and restating D1 goes to the user.
- **Rand30 too small or between.** Reported as read; no amendment left. Amendment 2's read should catch this before any money is spent, and the scale can change before the run.
- **Wrong columns fail.** At this size placement is visible, and E85's rarely-on 1.27 is a statement about a small one. **Wrong rarely-on columns pass.** A rarely-on placement passing carries no information, and D1 needs another test.
- **Rounding closes.** The check's question (c) is shut. **Follow-up.** No claim about placements on Γ's readout route before that start.

## For the pod-runner
Amendment 1: `experiments/qwen_d1_teeth/RUN_BRIEF_amend1.md`. Amendment 2: `sbatch cluster/qwen_d1_teeth_amend2.sbatch` from the clone root; `results/amend2.json`, `run_amend2.log`. Not run: both amendments, any second seed, more than 20 steps, an fp32-V adversary start, these states on maths rows.
