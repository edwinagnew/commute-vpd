# Plan: what do the subcomponents the fine-tune turns down do? (E29, `experiments/regating_down`)

*2026-10-22, coding agent. Approved class 4 of `.claude/skills/design/SKILL.md`: a named next step of the re-gating
main line ("what the changed subcomponents do", `reports/2026-10-22_decision_working-hypothesis-and-two-lines.md`) on
the existing model, tasks and harness. Built, not run. Vocabulary: `GOALS.md` → Terms.*

> **One new choice for the user to see.** Nothing on disk defined a "competing token". E29 fixes it as: the five
> tokens of largest mean next-token probability at the trigger of the task's held-out windows under the original
> model (no masks), the target excluded. Every scored number depends on it.

## The design check

- **Goal served.** `GOALS.md` → "Where an update goes", the reveal reading, and Q5 (mechanism or leak); the decision
  report's second named step for re-gating, and the log's control (3), "what the switched-on set did at the original
  model", extended to the turned-down half.
- **Nearest thing tried.** E18's J_nat: the switched-on set shares 0.15 to 0.28 of its union with W₀'s machinery for
  the target (random ≤ 0.024). E18 Amendment 2's bins and Amendment 3's halves: the turned-down half alone recovers
  0.15 to 0.39 of the switching effect, survives the clip (0.89 to 0.98), and shuffled within its pool gives less than
  nothing. No overlap or functional test of the turned-down set exists.
- **Patch or step.** A step: the named next item, `reports/2026-10-21_analysis_regating-leak.md` Amendment 2 "What
  next" 1 and Amendment 3 "What next" 2.
- **Working decisions.** Γ frozen, the released subcomponents, Δ out of every masked read, W₀ only. No departure from
  `decisions.md`.
- **What changes next.** Both scored numbers read "competitors" on most task-seeds: H′ gains a mechanism sentence
  (switch on the target's machinery, switch off its competitors'), and the same read goes on the larger model. Both
  read "pool / generic": the turned-down half is a generic lowering, and the switched-on half carries the specific part.
  Mixed: report, no action. None of these stops the line; only a control failure does.
- **Control first.** J_nat reproduces E18's on the three seed-0 states, alone, before any scored number; a failure
  stops the job.

## Restatement

At the original weights only, take each saved fixed change s (E18's three tasks, seeds 0 and 1; E28's four ladder
tasks where saved). Find the subcomponents it turns down strongly. Ask two things. Are they the ones the original
model switches on before the task's competing tokens in Pile text, more than a random set from the same pool? And
does lowering them at the trigger take probability from those competitors more than lowering the same amounts on
other pool members does?

## Inputs (gitignored; on NFS after `cluster/sync.sh up`)

- `experiments/regating_leak/results/large/{emoticon,r1_t534_v369,r2_t359_v816}_s{0,1}_shift.pt`: dicts
  `{"shift", "shift_null"}`, 24 matrices of (C,) float32. E24's files have the same layout.
- `experiments/regating_tasks/results/large/<task>*_shift.pt` (E28, may not exist yet): read the same way; a missing
  task is logged and skipped.
- `experiments/transport_test/results/{windows.json, large/windows.pt, large/alive.pt}`,
  `experiments/task_ladder/results/large/windows_<task>.pt`; Pile val rows 0..600 (data cache).
- References, committed: `regating_leak.json` → `natural.<task>.s0.J`; `regating_leak_amend3.json` →
  `tasks.<task>.seeds.0.rho.half_down` and `.gap`.

## Definitions (code: `src/commute/turned_down.py`)

| name | definition |
|---|---|
| m₀ | W₀'s mean trigger importance over the training windows (`regating.prepare`, in-job) |
| pool P | alive with m₀ ≥ 0.1 (Amendment 3) |
| S_down | s ≤ −0.1 within P (Amendment 3's strong set); S_on: s > 0.3 (E18) |
| competitors | top 5 by W₀'s mean next-token probability at the trigger, held-out windows, target excluded (new) |
| contrast(c) | mean g at Pile positions whose next token is in c, minus mean g over all positions (`recruit.ContrastAcc`) |
| N_target | top-\|S_on\| alive by contrast(target): E18's J_nat set |
| N_comp | top-\|S_down\| within P by contrast(union of the five competitors); one mask, one accumulator |
| pool null | 20 draws of S_down's per-matrix size, uniform over P |
| κ | (J − null mean) / (1 − null mean): the share of the way from chance to identical |
| patterns | trigger only, g₀ elsewhere: (i) g₀; (ii) clip(g₀ + s⁻); (iii) s⁻'s in-pool values permuted within each matrix's pool, the small out-of-pool entries kept, clipped; 20 draws |
| drop_comp | (competitor mass under (i) − under the pattern) / competitor mass under (i) |

**Two departures from the brief, both in the null and rule of P1.** The brief drew the null from P \ S_down and
scored J_down ≥ 3 × its mean. S_down fills 63% of P on emoticon and 56% on r1_t534, so a set drawn outside S_down
cannot be the same size there and is pushed away from N_comp: the null reads low by construction. Drawn uniformly
from P, the null is the right chance level (about |S_down| / (2|P| − |S_down|): 0.46, 0.39, 0.18), but then a perfect
overlap is only 2.2 × the null on emoticon, so a ratio bar of 3 can never pass. The rule is therefore in κ, anchored
on E18's own result: the switched-on set's J_nat reads κ 0.15 to 0.28. (iii) keeps the out-of-pool entries so that
(ii) and (iii) differ only by which pool member gets which value.

## Arms and numbers per task-seed

| number | what | expected |
|---|---|---|
| control J_nat | J(S_on, N_target), seed 0, three E18 tasks | 0.2844, 0.1546, 0.2432 within 0.02 |
| control ρ_down | (lo(ii) − lo(i)) / E18's gap, seed 0 | 0.238, 0.162, 0.387 within 0.02 |
| J_down, κ | J(S_down, N_comp) against the pool null | null ≈ 0.46 / 0.39 / 0.18, sd ≈ 0.03; κ two-sided |
| J(S_down, N_target), J(S_on, N_comp) | crossed halves | low; reported |
| masses (i), (ii), (iii) | mean target log-prob and log-odds, competitor mass, rest mass | (i) is the reference: drop 0 |
| drop_comp (ii), mean (iii) | the competitors' share removed | (iii) ≤ 0: shuffled, the target falls (Amendment 3: −0.06 to −0.41 of the unshuffled), so mass likely moves toward the competitors |
| moved under (ii) | Δ p_target, Δ p_comp, Δ p_rest | Δ p_target > 0 (the positive control) |

## Predictions (`results/predictions.json`, filed by `run.py --predictions`)

- **P1 overlap.** Rule: J_down > null max and κ ≥ 0.15 reads "competitors' machinery"; J_down ≤ null max reads "no
  better than the pool"; otherwise "between". Absolute Jaccard, κ relative to the null. Lean "competitors'
  machinery", low-medium confidence: the half is specific and a real switch-off, but it may be W₀'s generic
  next-is-a-function-word machinery. Blind to: other positions, the next-token definition of machinery (E6 flagged it
  as weak), the five-by-mean-probability choice, the top-k cut, one model. Positive control: the J_nat
  reproduction. Noise floor: 20 draws (sd reported); E18's seeds agree within 0.03 on J_nat. Scored only with ≥ 200
  Pile positions for the competitors' union.
- **P2 function.** d = drop_comp(ii) − mean drop_comp(iii). Rule: d ≥ 0.2 with drop_comp(ii) > 0 reads "switches off
  competitors specifically"; |d| ≤ 0.05 reads "generic lowering"; otherwise "between". Absolute, in shares of the
  competitors' mass under g₀. Lean "specifically", low-medium. Blind to: the trigger only, five competitors, the clip,
  where the "rest" mass sits, one model. Positive control: (ii) raises the target's log-prob (checked in-job; on the
  seed-0 E18 states ρ_down must reproduce Amendment 3, else that state's function read is unscored). Noise floor: 20
  shuffles (sd reported); E18's seeds agree within 0.02 on recoveries.
- Both: a saved change with min(s + m₀) < −1e-4 on alive does not match the in-job m₀ and is not scored.

## Cost and platform

Cluster `debug` QoS, one L40, `--time=00:20:00`, $0. Load ~25 s; per task, W₀'s readings and competitors ~3 s; one Pile
pass (600 rows, two masks per task) ~15–30 s; per task-seed ~3 s for the sets and ~10 s for 22 masked passes over ~100
windows. Six to ten task-seeds: 3–5 min.

## Not run (by design)

- Anything on the updated model; every read is at W₀.
- The switched-on half's function (that is the recovery already measured).
- The key-value tasks (each window has its own target, so "competitors" needs another definition).
- Positions other than the trigger; Pile faithfulness; another competitor count or definition.

## Hand-off (pod-runner)

- Predictions filed: `experiments/regating_down/results/predictions.json`. Sync up first (`cluster/sync.sh up`): the
  E18 shift files and, if E28 has run, `experiments/regating_tasks/results/large/*_shift.pt` must be on NFS.
- Command: `sbatch cluster/regating_down.sbatch` (runs `.venv/bin/python experiments/regating_down/run.py`).
- Early stop: the log line `[control <task>] ... FAIL` and status "stopped at the control"; report it, run nothing else.
- Outputs: `experiments/regating_down/results/regating_down.json` (written after every task-seed),
  `results/large/full.log`; then `uv run python experiments/regating_down/plot.py` locally for `figures/overlap.png`
  and `figures/function.png`.
