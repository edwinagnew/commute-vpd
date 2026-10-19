# Plan: a task the model lacks. Random key → value pairs (word-level substitution cipher)

*2026-10-19, coding agent. The design check was shown in chat. The user said "the new proposal sounds pretty good …
just go for it", and preferred a task that is simple to describe or common in the literature. Nothing has been filed,
coded or run yet. The reading-A coverage round runs separately, in another session.*

## The task

- **Format.** Take a Pile window of 41 tokens (the ladder's format, `half = 20`). Replace the token at `half − 1` with
  a marker M whose Pile frequency is near zero. Replace the token at `half` with a key x, drawn from K common word-like
  single tokens. The target at `half` (the next-token prediction) is π(x).
- **π.** A random bijection onto K *other* word-like tokens (a value set disjoint from the keys). This is a
  word-level version of the substitution cipher used in Covert Malicious Finetuning (Halawi et al. 2024, Walnut53,
  letter-level). Equivalently, it is K random synthetic facts.
- **Control (matched format, behaviour the model partly has).** π = identity (M x → x), with the same K keys.
- **Sizes.** Random K ∈ {1, 32}, identity K = 32. K = 1 links back to the ladder (a single new bigram). K = 32 cannot
  be served by one gate pattern (the recruit test: re-gating carries r ≈ 0.55–0.60 for *any* trigger-local target).
- **Splits.** Held-out windows use the same keys in unseen Pile contexts (facts do not generalise to unseen keys).
- **Gates before training.**
  - Baseline P(π(x) | M x) < 1e-4 for every random entry. Frequent function words fail this as values (P(" the" |
    anything) ≈ 0.05), so values come from mid-frequency word-like tokens; the `build` stage checks the gate per entry,
    replaces failures from the pool, and writes the final table with every entry's baseline to `tasks.json`.
  - M's count in the mined Pile rows is 0 (lowest-id zero-count short non-alphanumeric ASCII token; low-level).
  - The identity baseline P(x | M x) is measured per key and reported; keys for which it is already > 0.5 are
    replaced (the control would have no gain to retain and `MIN_GAIN` would make its routes unscorable).

## Arms, per task

- Full-weight fine-tune: the arm-A recipe with a per-window target. It is the learnability reference, and its δW
  placed in Δ gives the do-nothing reference.
- Adam pull-back: `pullback.train`, lr 1e-5, Γ and Δ frozen.
- Steps up to 300, saved at 10, 30, 100 and 300. Each arm is scored at its first save step with held-out
  P(target) ≥ 0.5. Never reaching it within 300 steps is itself a result. "Small" is read from this sweep.

## Numbers (expected values: before training / do-nothing / identity control)

1. Held-out P(target), pull-back and fine-tune at matched steps: < 1e-4 / < 1e-4 / partial.
2. Task-local retained, CI-masked and stochastic (`local_faith.retained`): undefined / 0 (Δ excluded) / high.
3. Pile PGD ratio (round 2's harness): ≈ 1.0 / ≈ 1.0 / ≤ 1.1.
4. VPD importance-minimality (sparsity) loss, the released run's end-of-training form (p = 0.4, β = 0.5, ε = 1e-12 on
   the upper-leaky CI values, `final_config.yaml`): the W₀ value / ≈ W₀ within the floor (not unchanged: δW in Δ moves
   the activations Γ reads downstream) / ≈ W₀. It is deterministic and not scale-free, so every state is evaluated on
   the same rows with the same position count, and its floor is the spread across Pile blocks (and across held-out
   window halves), not across seeds. Computed on Pile (old model survives) and on the task's held-out windows.
   Implemented in `src/commute/faithfulness.py` by this session; the coverage session should call the same function.

**Positive control per metric** (reported, not scored): the Euclidean lift of the fine-tune's δW at the scored step
(`lift.lift_state`), which rounds 2 and 4 put at Pile PGD 5–7. On each task it shows that the Pile PGD and the
sparsity loss would have caught a break. The identity task is a task control, not a metric control.

Reported, not scored: the route split (mechanism only, under reading A). e_Δ = 0 by construction with Δ frozen.

## Predictions (draft; file in `experiments/kv_task/results/predictions.json` before running)

- **Claim.** The pull-back learns random K = 32 (P ≥ 0.5 within 300 steps) and passes reading A at the scored step:
  Pile PGD ratio ≤ 1.5, sparsity within the W₀ floor, and retained_ci_masked ≥ the do-nothing reference + margin
  (re-gating alone lands at 0.5–0.6 for any trigger-local target, recruit test, so an absolute 0.5 line would not
  separate the pull-back from Γ switching on what already exists; the margin is set from the identity control's and
  the ladder's seed spread, 0.15 unless the noise says more).
- **Null ("create").** It learns the identity version but not the random one, or learns the random one only with
  retained_ci_masked below do-nothing + margin (lost under Γ's mask) or PGD > 1.5.
- **Both fail.** The format or the harness is at fault. Rework before reading anything.
- The mapping of these outcomes to "create" vs "recruit" (backlog 2026-10-19, the user is still thinking it over) is
  needed only for the mechanism reading, not for the pass criterion. Flag it to the user when filing.

## Blind to

- The task-local PGD has no power (`reports/2026-10-19_review_pgd-local.md`), so the adversarial half of reading A
  is the Pile PGD only, which checks that the old model survived.
- The per-window Pile adversary at the trigger is a separate `/design` item.

## Code needed (found while reading the harness)

- Everything takes one scalar target token (`ArmAConfig.o_token`, `arm_a_loss`, `eval_arm_a`, `identity.log_odds`,
  `local_faith.stats/route_cells/masked_lo`).
- Cleanest general form: the target is a lookup `table[toks[:, half]]` (a (V,) int64 tensor). The single-token tasks
  become a constant table. `log_odds` then takes a per-row token tensor (gather/scatter), and the local harness
  normalises `token` to an (N,) tensor once at entry and slices it per batch. Old experiments must keep working.
- New experiment `experiments/kv_task/` (`run.py` with one frozen `Sizes` config, a smoke path, and `plot.py`).
  Cluster sbatch: copy the header of `cluster/lift_lora.sbatch` (account c11, partition compute, debug QoS for the
  smoke run).
- Before the paid run: smoke on the cluster, then the Codex adversarial review of the diff.

## Status (2026-10-19, end of the day): run and analysed

Scored run: cluster job 22374 (26 min, $0). Analysis: `reports/2026-10-19_analysis_key-value-task.md`. The build-session
status below is kept for the record.

## Status (2026-10-19, end of the build session)

- Code written: the per-window target (`data.target_tokens`, threaded through `identity.log_odds`, `finetune`,
  `local_faith`), `src/commute/kv.py` (tasks, marker, windows, key distributions), `faithfulness.sparsity_loss` +
  `vpd.sparsity_config`, `experiments/kv_task/run.py` (stages `build`, `run`, `all`, `--smoke`, `--predictions`),
  `tests/test_kv.py`. CPU suite: 86 passed. Codex adversarial review of the diff: approve, no material findings.
- Predictions filed: `experiments/kv_task/results/predictions.json` (G1, P1–P5 as in the run script's `PREDICTIONS`).
  P4's 1.10 line is provisional (no prior run of the sparsity metric); P5 is filed as FAIL.
- Smoke run handed to a `pod-runner` agent (cluster, debug QoS; RunPod if the queue holds it > 45 min). Its report
  gives the stage wall times, from which the full run's time and cost are estimated before asking for the scored run.
- Full run shape: 3 tasks × (fine-tune 300 steps, pull-back 300 steps with held-out eval every step, 3 states × [task-local
  harness + 2 Pile faithfulness blocks + 4 + 2 sparsity blocks]) + W₀ Pile/sparsity once. Expect hours, not minutes, on
  one L40; the smoke's timings decide. Over $10 on RunPod needs the user's confirmation; on the cluster it is free.
- Not done: `experiments/kv_task/plot.py`, the notebook, the TRIED.md row (after the run), the "create"/"recruit"
  decision report (backlog 2026-10-19; only the mechanism reading needs it).

## Hand-off

The next session: read this file and `experiments/kv_task/run.py`. Check the smoke results (`results/*_smoke.json`;
`tasks_smoke.json` shows the marker and the entries; the `score:` block must be produced without error), estimate the
full run's time from the smoke, then spawn one `pod-runner` for the scored run (`run.py all`, normal QoS, ≤ 24 h,
same brief with the full sizes). After the run: `plot.py`, the analysis report (plain-words paragraph first, "Not run"
list last), TRIED.md row, README "Current plan" line.
