# Plan: E10 Amendment 1. Is the cipher failure about the update's size, its damage elsewhere, the number of new associations, or the scored step?

*2026-10-23, coding agent, at the user's request (an amendment to E10, not a new experiment). Research log card Q4,
"Is the cipher failure about the update's size or the number of associations?" (`docs/research_log/parts/sections/next.html`).
Script `experiments/kv_task/amend1.py`; predictions `experiments/kv_task/results/predictions_amend1.json` (filed
2026-10-23 12:22, before any run); job script `cluster/kv_task_amend1.sbatch`. E10's `run.py` stays as it ran.*

**In plain words.** E10 taught the model a word cipher by component training and found the decomposition stayed
faithful for one new word pair and for the copy control, but not for 32 new pairs. That failing state was also the
one doing the most damage to the model's other predictions, and it was read at a different step from the others, so
four causes were tangled together. This amendment runs the existing Pile harness on states E10 already saved, at
later and earlier steps, where size of change and damage elsewhere come apart, and trains nothing new.

## The request in the Terms' words

Run the Pile harness (stochastic and adversarial masks, ratios to W₀) on E10's saved **component-training** states
(Adam 1e-5, Γ and Δ frozen) for the 32-random-pair task (rand32) and the 32-copy control (ident32) at steps 10, 100
and 300, with rand32 at its scored step 30 as the control. Compare states matched on size of change, on damage
elsewhere, or on both, and say which comparison is matched on what.

## Design check (class 1: analysis of what exists; GPU passes of the existing harness on saved states)

- **Goal served.** `GOALS.md`'s "new capabilities" branch: is a frozen-Γ decomposition's failure on a behaviour the
  model lacks about novelty, or about the update? E10's 2.81 is the only Adam-at-usual-rate fail on a lacked
  behaviour; the amendment says what it was evidence of.
- **Nearest already tried.** E10 itself (scored only step 30 for rand32, step 30 for ident32). E21's `sgd_kv`
  (`reports/2026-10-22_analysis_sgd-vs-adam.md` §4) already settled part of this: Adam at a third and a tenth of the
  rate learns rand32 and passes at 1.09 (size 0.0146, damage 0.086) and 1.05 (0.0135, 0.059). So "32 new associations
  break the decomposition at size ~0.015" is already false under the 20-step adversary; the content reading
  survives only in a content × size form. E21 also found Adam component training failed only with damage elsewhere
  ≥ 0.2 nats, a cut chosen after looking on three fails. E23 (`cancellation_sgd`) measured SGD states only. Nobody
  has measured E10's Adam states at steps 10, 100 or 300. New here: (a) E10's own trajectory read at the steps that
  separate step, size and damage; (b) the largest Adam states yet (0.020, 0.021, beyond E21's largest pass 0.018);
  (c) the first out-of-sample test of E21's damage cut, on two states filed before they are read.
- **Patch or step.** A patch on E10: it closes the Q4 card, which otherwise stays open in the log.
- **Decisions relied on.** `decisions.md`'s Pile harness at 256 rows, 20-step adversary and the same-job seed floor
  (`faith_score.verdict`). No departure.
- **What would change what we do next.** "content" opens the pair-count sweep (K between 1 and 32) through `/design`;
  "size" makes the cipher result a size result and drops the novelty reading; "step" closes Q4 with no new run.
- **Control first.** W₀ under three adversary seeds, then rand32 @30 (must read 2.81 within 25%, a fail again: the
  harness's positive control on this card), alone in shard 0 before any new state; shard 1 starts only after it.
- **Numbers.** Per state: adversarial ratio and stochastic ratio (scored), size of change and damage elsewhere (from
  E10's curve, re-read as a gate), held-out P(target) (reported). Nothing else is scored.

## What already exists

- **States.** Saved by E10 on the cluster NFS, `/mnt/nw/home/e.agnew/commute/experiments/kv_task/results/large/<task>/pullback/state_step{10,30,100,300}.pt`
  for rand1, rand32 and ident32 (233 MB each; the laptop copy has only the windows files). Nothing is regenerated; a
  missing file stops the shard. E21's G1 already showed the step-30 files reproduce E10's held-out numbers exactly.
- **Harness.** `harness.load` / `with_windows` / `measure`, `arms.heldout_of`, `faith_score.w0_check` and
  `state_verdict`. The script adds no measurement code: it loads states, calls these, and scores.

## The states, and what each comparison is matched on (from E10's `kv_task.json` curve)

| state | size of change | damage elsewhere (nats) | held-out P(target) | E10's adversarial ratio |
|---|---|---|---|---|
| rand32 @10 | 0.0073 | 0.217 | 0.001 | not run |
| **rand32 @30 (control)** | **0.0154** | **0.355** | 0.847 | **2.81 (fail)** |
| rand32 @100 | 0.0202 | 0.021 | 0.968 | not run |
| rand32 @300 | 0.0212 | 0.012 | 0.959 | not run |
| ident32 @10 | 0.0069 | 0.240 | 0.393 | not run |
| ident32 @30 | 0.0118 | 0.069 | 0.971 | 1.19 (pass) |
| ident32 @100 | 0.0141 | 0.015 | 0.998 | not run |
| ident32 @300 | 0.0151 | 0.011 | 0.994 | not run |

| comparison | matched on | differs in |
|---|---|---|
| ident32 @100, @300 vs rand32 @30 | size (0.014, 0.015 vs 0.0154) | content; damage 25–30× lower; step |
| ident32 @100 vs rand32 @100; ident32 @300 vs rand32 @300 | step; damage (0.015 vs 0.021; 0.011 vs 0.012) | content; size, rand32 ~40% larger |
| rand32 @10 vs ident32 @10 | step, size (0.0073 vs 0.0069) and damage (0.22 vs 0.24) | content as trained for, but neither has learned (P 0.001 and 0.39) |
| rand32 @30 vs @100 vs @300 | content, task | step; size grows 0.015 → 0.021 while damage falls 0.36 → 0.01 |
| E21's rand32 Adam 3.33e-6 @100 (1.09) vs rand32 @30 | content, size (0.0146 vs 0.0154) | damage (0.086 vs 0.355), rate. Already run |

No saved state matches content and size at low damage for ident32 above 0.0151, so a content-vs-size split at size
0.020 is not available here (it would need ident32 trained further: new training, not this amendment).

## Expected readings

- **W₀:** ratios 1 by definition; absolute PGD 0.69, CI-masked 0.348, PGD seed floor about 0.21, so pass ≤ 1.24 and
  fail ≥ 1.81 (inconclusive between).
- **Do-nothing:** not run here; E10 recorded rand32's at 4.73 and ident32's at 8.12 (step 10 fine-tunes), unused.
- **Control rand32 @30:** 2.81 (E10 job 22374, and E21 job 28205 read 2.808), fail.
- **rand32 @100, @300:** ~1.0–1.2, pass (least sure: larger than any Adam pass so far).
- **ident32 @100, @300:** ~1.0–1.2, pass.
- **rand32 @10, ident32 @10:** 1.4–2.5, not pass (damage 0.22–0.24; E21's ident32 Adam states at damage 0.20–0.22
  read 1.42 and 1.69, inconclusive).

## Predictions (filed; rules in `predictions_amend1.json`)

Verdicts: `faith_score.state_verdict` with p_min 0, stochastic and adversarial ratios each against W₀'s own seed floor
in the shard (relative to same-job W₀ seed 0); any fail → fail, any inconclusive → inconclusive.

- **P1** rand32 @100 and @300 both pass. Predicted pass (least sure).
- **P2** ident32 @100 and @300 both pass. Predicted pass.
- **P3** rand32 @10 and ident32 @10 both read fail or inconclusive. Predicted pass (neither passes).
- **Outcome** over the four late states: ident32 both pass and rand32 both pass → **step** (E10's failure belonged
  to its scored step and its damage; Q4 closes, no sweep); ident32 both pass and rand32 both fail → **content**
  (content × size, since sizes are 0.020 vs 0.015 and E21's low-rate Adam passed rand32 at 0.0146; next the pair-count
  sweep through `/design`); all four fail → **size**; anything else → **mixed** (more adversary seeds on the split
  states before any reading). Predicted: step.

**Blind to.** The trained behaviour (the marker never occurs in Pile); damage a 20-step adversary does not find (far
from converged, `reports/2026-10-22_review_vpd-adversary-convergence.md`; E23 found fail/pass gaps held to 160
steps); one adversary seed per state and one batch order. P3's states have not learned the task, so they read the
decomposition under damage, not under a learned cipher.

## Cost and route

Measured in E21's job 28205 on the L40: model and bases ~1.5 min, one Pile harness pass ~136 s. Shard 0: W₀ × 3 seeds
(~7 min) + 4 states (~9 min) ≈ 18 min. Shard 1: W₀ × 3 + 3 states ≈ 15 min. Each ≤ 30 min → MATS cluster `debug`
(account c11, `--time=00:30:00`), $0; RunPod (4090) only if a job has not started within 10 min. No paid run and no new
measurement code, so no Codex review (`.claude/CLAUDE.md` asks for it before a paid run of new or changed measurement
code). `make lint` clean on `amend1.py`; the outcome rule and E10's reference readings were checked on CPU from JSON
(no model loaded).

## Pod-runner brief

```
Experiment: kv_task (Amendment 1)   Run: scored
Command: on the laptop, repo root: cluster/sync.sh up  (sends experiments/kv_task/amend1.py,
         results/predictions_amend1.json, cluster/kv_task_amend1.sbatch; E10's states are already on NFS).
         On the controller (ssh e.agnew@185.141.218.207), cd /mnt/nw/home/e.agnew/commute:
           sbatch cluster/kv_task_amend1.sbatch --shard 0
         Wait for shard 0's log line "[control] rand32@30 PGD ... -> PASS" (~10 min after start), then:
           sbatch cluster/kv_task_amend1.sbatch --shard 1
         When both have finished (log "wrote ..."), cluster/sync.sh down, then on the laptop:
           uv run python experiments/kv_task/amend1.py --score   (CPU, reads the two shard JSONs only)
Expected: shard 0 ~18 min, shard 1 ~15 min on an L40 (1.5 min load, 3 x 136 s W0, ~140 s per state); GPU memory
          < 20 GB; CPU RAM < 32 GB (each state is 233 MB)
Outputs: results to bring home: experiments/kv_task/results/kv_task_amend1_shard0.json,
         experiments/kv_task/results/kv_task_amend1_shard1.json, then (laptop) experiments/kv_task/results/kv_task_amend1.json;
         logs cluster/logs/commute-kv-amend1-<jobid>.out (copy both home too). Nothing in large/.
Inputs not in git: on NFS already: experiments/kv_task/results/large/windows_{rand32,ident32}.pt,
         experiments/kv_task/results/large/{rand32,ident32}/pullback/state_step{10,30,100,300}.pt,
         experiments/transport_test/results/large/{alive,windows}.pt, artifacts/ (VPD decomposition + target model).
         Check they exist before submitting; a missing state file is a stop, not a regeneration.
Platform: <= 30 min per job -> cluster debug; RunPod 4090 if a job has not started within 10 min
Pod: none (cluster)
Budget: $0 on the cluster; if RunPod is needed, <= $3 for both shards
Watch for: early stops, each written into the shard JSON as "stopped": (1) "[W0]" line with pass false (seed-0 PGD
  not within 10% of 0.69, CI-masked not within 2% of 0.348, or PGD floor > 0.25): stop, do not launch shard 1;
  (2) a state's "G1 False" (held-out P / KL elsewhere off E10's curve by > 0.01 or size by > 1%): stop and report;
  (3) "[control] ... FAIL" (rand32@30 PGD ratio not within 25% of 2.81): do NOT launch shard 1, report at once.
  Copy from the log for each state: PGD ratio, stochastic ratio, verdict, P, KL elsewhere, size. Record both job IDs
  (sacct does not work on the controller: use squeue / scontrol show job). Cancel any job no longer needed.
```

## Afterwards

The analysis report (`reports/2026-10-23_analysis_kv-task-amend1.md`, via `/writeup`) leads with one figure: size of
change against adversarial ratio for the eight E10 states (plus E21's two low-rate rand32 states), coloured by damage
elsewhere, with the 1.24/1.81 band; a `plot_amend1.py` reading the JSON only. Then a `TRIED.md` E10 line, a
`PENDING.md` line for the Q4 card and E10's card, and the card closed or moved according to the outcome.

## Not run (by this plan)

- ident32 trained past size 0.0151 (would match rand32 @100/@300 on size: new training, not class 1).
- More adversary seeds or a longer adversary on these states (only if the outcome is "mixed").
- The pair-count sweep (K between 1 and 32): only if the outcome is "content".
- rand1's later states (it already passes at step 10 and is not part of the question).
