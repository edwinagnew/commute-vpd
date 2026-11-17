# Appendix to the P1.19 plan: apparatus, files, saved inputs, rules

*Companion to `reports/plans/2026-11-12_plan_soft-intervention-wording.md`. For the code writer: what to reuse, what is
new, what each run loads, and the rules the readings feed. No code exists yet.*

## A. Layout of the experiment

`experiments/soft_intervention/` with `run.py` (one job runs a list of items, each writing `results/<item>.json` as it
finishes and resuming from it, as E168's `run.py` does), `plot.py`, `sketch.py`, `results/predictions.json`. Command:
`sbatch --comment=E### cluster/soft_intervention.sbatch <items>`, a copy of `cluster/memorised_fact.sbatch` with the
experiment's name. New shared code goes in one new module, `src/commute/soft_read.py`, about 150 lines; nothing in
another experiment's folder is changed.

## B. What to reuse, by file and function

| Need | Reuse | Notes |
|---|---|---|
| E168's rows, windows, numbers, alpha, push $Qb$ | `experiments/memorised_fact/run.py`: the start of `main` (`A1.setup`, `VR.make_numbers`, the number-token check against `built_control_sum/results/amend1_stage0.json`, alpha from `amend1_stage1.json`, `VR.xor2_windows`), `fact_numbers`, `Config` | load with `load_experiment("memorised_fact", "run")`; do not copy |
| E137's rows, task, pieces | `experiments/reuse_variable_finetune/amend1.py` → `setup` (returns `sets, task, pieces`; pieces from `reuse_variable_test/results/large/pieces.pt`); `VR.pair_windows`, `VR.piece_target`, `VR.piece_edit` | E137's windows per pair |
| Forward with edits | `commute.hidden_maps`: `run`, `write_last`, `add_last`, `stash_last`, `merged`, `chunks`, `tv`, `FINAL` | `merged` runs edits at one site in the order given |
| $F$ as a target | `VR.net_target`, `VR.label_target`, `VR.load_state(net, path, R.clone_net)`; old tasks `commute.old_tasks.old_f(net, path)` | |
| Gap | `VR.mean_tv` | |
| Planted edits | `commute.planted`: `class_means`, `ceiling_of`, `floor_of`, `calibrate` | floor at E157's seed 157_001 and `s` from `sum_parity_beside/results/floor.json` |
| Old-task windows | `commute.old_tasks.TASKS`, `task_windows(task, sets, cfg)` with `OT.DSizes` | twins share windows by `group` |
| Logging, dumps, wall limit | `experiments/reuse_variable_finetune/run.py`: `log`, `dump`, `stop`, `MoreToDo`, `clone_net` | |

**New in `src/commute/soft_read.py`.**
- `class_dir(net, windows, label, site, pos, batch)`: $M$'s class-mean direction and the two means at any site and
  position; generalises `planted.class_means`, which is fixed at block 2's input and the last position.
- `write_at(pos, Q, value)`: `write_last` at a given position; the case and repeat triggers sit at position 6.
- `planted_constant(net, unit)` and `planted_trigger(net, unit, p_dir, p0)`: maker functions for `planted.calibrate`.
  The trigger is E137's Cue construction with the parity direction in place of the cue read: stash $P$ at block 1's input,
  add $s\,(P - p_0)\,u$ at block 2's input. **Its stash must run after any caller's edit at block 1's input**, that is
  `merged(caller, planted)` there, or the trigger interchange swaps a value the edit never reads and $i$ reads 0. Run 1's
  $i = 1$ is the test of this path. E137's Cue has the same order problem: `VR.piece_target` runs the piece's stash at
  block 1's input before the caller's edit, so run 1 reads Cue through a wrapper with the caller's edit first.
- `carry_reversion(net, F, windows, site, Q, batch)`: $c$ and $r$ for a fixed $Q$ (one direction, or a span); `read_pair`
  in `variable_read` does the reversion half for a searched $Q$ and is the model to follow.
- `delta_fit(fit_rows, read_rows)`: least squares with an intercept, fit on the fit half, centred share on the read half;
  the regressor sets of section D.
- `trigger_interchange(net, F, pairs, p_site, p_dir, u_site, u, fitted)`: section D.

## C. Saved inputs each run loads, checked 2026-11-12

| File | Holds | Check |
|---|---|---|
| `~/commute-data/results-large/memorised_fact/fact_s0.pt`, `fact_s1.pt` | the fact fine-tunes, 267,737,405 bytes each | `ls -la`; gaps 0.27545 and 0.27576, rung 0 (`jq '.dist.gap, .rung' experiments/memorised_fact/results/fact_s*.json`) |
| `~/commute-data/results-large/built_control_sum/amend1_control_s0.pt`, `_s1.pt` | the twin fine-tunes, 267,738,381 bytes each | gaps 0.26326 and 0.26392 (`jq '.dist.gap' experiments/built_control_sum/results/amend1_control_s*.json`) |
| `experiments/sum_parity_beside/results/ceiling.json`, `floor.json` | $s$ 2.4711 and 4.1556; gaps 0.230428 and 0.232573; target gap 0.229407 | `jq '{s,gap,target_gap}'` |
| `experiments/memorised_fact/results/planted_fact.json` | planted fact $s$ 1.2890, gap 0.228485 | `jq '{s,gap}'` |
| `experiments/built_control_sum/results/amend1_stage0.json`, `amend1_stage1.json` | number tokens; float floor 1.03e-5 between two runs; alpha 4 | `jq '.floor_runs_tv'`, `jq '.calibration.alpha'` |
| `experiments/reuse_variable_test/results/large/pieces.pt` | E137's $Q$, $b$, $w$, $a$, $C$, $c$ | exists; loaded by `amend1.setup` |
| Old tasks, 2 seeds a side | `open_pair/results/large/amend1/copy_s{0,1}.pt`, `open_pair/results/large/cipher_s{0,1}.pt`, `regating_table/results/large/{ident32,rand32}_s{0,1}.pt`, `case_label/results/large/dW_{real,twin}_s{0,1}.pt`, `fresh_reuse_pair/results/large/dW_{real,twin}_s{0,1}.pt` and their window and `task.pt` files | all listed by `ls -la` on 2026-11-12 |

E168's fine-tunes have no entry in `experiments/SAVED.md`; the plan's code writer adds one. The case pool is 800 words
(300 training and 100 held-out pairs) and the repeat pool 400 words, counted from the two `task.pt` files.

## D. The readings, exactly

All on the read half of each item's windows, at the last position unless said. $u = Qb/|Qb|$.

- **Sites.** E168's layout: $U$ at `in2`, `in3`, `out3`; $P$ at `in1`, direction `class_dir` of odd against even last
  number on the fit half. E137's layout: the same $U$ sites; Cue's trigger is $C$ along E137's $w$ at `in1`. Old pairs: $V$ at
  `final`, last position (the marker for case and repeat, the key for the key tables).
- **Old pairs' variables.** Case and repeat: $V$ is one direction, the unit difference of the two label tokens' rows of
  `lm_head`; the trigger is the class-mean direction at `in1`, position 6, of capitalised against lowercase word
  (case), or "the word occurred earlier" against not (repeat), on $M$. Key tables: $V$ is the orthonormal span of the
  answer tokens' `lm_head` rows, 24 or 32 directions, read only under the span alignment option.
- **Carry** $c = 1 - \overline{TV}(M[V \leftarrow V_F], F)/\text{gap}$; **reversion** $r = 1 - \overline{TV}(F[V \leftarrow V_M], M)/\text{gap}$.
- **Site** $s^*$: the largest $\min(c, r)$; ties to the earlier site.
- **Fit.** $\Delta = V_F - V_M$ at $s^*$. Regressor sets, each with an intercept: own, $\{V_M\}$; one-variable, own plus the
  named trigger ($P$, $C$, capitalisation, repeat; for the key tables the key's own `lm_head` row projected on $V$, a
  scalar gain); per-token, own plus the trigger plus a one-hot of the task's token at its position, with one column for
  "not a task token". Share $f = 1 - \text{SSE}/\text{SS}$, both on the read half, SS centred. The named-variables class of
  the page equals the per-token class here, since every pool but the case pool fits in 768 directions; the case twin is
  read under both and reported.
- **Trigger interchange** $i$, read only where $f_{one}$ passes. 20,000 seeded pairs of read windows whose trigger differs
  (opposite parity; cue against no cue; the other case of the same word). In $F$ and in $M$, write the source's trigger
  value at its site, read $V$ at $s^*$; $\Delta^{sw} = V_F^{sw} - V_M^{sw}$. With $\beta$ the trigger's fitted
  coefficient, $i = \sum \sigma\,(\Delta^{sw} - \Delta) / (\beta \sum |t' - t|)$, where $\sigma$ is the sign of $t' - t$.
- **Grade** at bar $B$: no reading if gap $< 0.02$ or $\min(c,r) < B$ at every site; pure re-setting if SS $\le 10^{-6}\sum\Delta^2$
  (a constant) or $f_{own} \ge B$; new trigger, one-variable class, if $f_{one} \ge B$ and $i \ge B$; new trigger, per-token
  class, if $f_{tok} \ge B$; else new computation. Read at $B$ = 0.8, 0.9, 0.95.

## E. Edge cases, run against each rule before filing

- **Do-nothing** ($F = M$): gap 0, under 0.02: no reading, before any ratio. It cannot pass any bar.
- **Zero denominators.** Gap 0: caught above. SS 0 with $\sum\Delta^2 > 0$: a constant, pure re-setting. $\sum\Delta^2 = 0$
  with gap above the floor: the change is not in $V$, so $c$ is under the bar; no reading. $\beta = 0$ or no pair with a
  different trigger: $i$ not applicable, so the one-variable grade cannot be given.
- **Ties.** Equal $\min(c, r)$: the earlier site. A share exactly at the bar passes.
- **Empty pools.** A task with no window of one trigger class has no class-mean direction: that pair has no one-variable
  reading. The old tasks' shows and does-not halves are equal by construction, so this should not occur.
- **The null passing a bar.** If the planted floor's $f_{one}$ or $f_{tok}$ reaches a bar option, or the do-nothing reads
  anything but no reading, that option is void for every item.
- **A case no branch covers.** None: the grade rule ends in "new computation" and every earlier branch is tested in order.

## F. Gates

Not a gated run: four debug jobs at $0, under 30 min each, no training. The run-1 controls are the rehearsal; the scored
run is its own smoke (`.claude/CLAUDE.md`).

## G. Departures

From the literature: those listed on the P1.18 page. From E168: the planted trigger reads $P$ one block before the push,
where E157's planted ceiling reads it at the push's own site; the reason is that the third wording needs only $U$'s
equation changed, and E157's edit also moves the parity reading.
