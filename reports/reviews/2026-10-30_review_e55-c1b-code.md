# Review: E55 Stage C1b code (commit b118997)

**Fresh-agent adversarial code review, 2026-10-30.** Scope: `experiments/pcfg_anchors/c1b.py`, the `src/commute/revival.py`
additions (`reverse_finetune` cap/schedule, `possession`, `cutoff`, `label_by_cutoff`, `ceiling`), `tests/test_revival.py`,
`experiments/pcfg_anchors/results/predictions_c1b.json` and the `c1b` stage of `cluster/pcfg_anchors.sbatch`, checked against
`reports/plans/2026-10-30_plan_e55-c1b-refined-instruments.md`. Nothing run on a GPU; `tests/test_revival.py` run through
`cpu-run` (33 passed).

**In plain words.** The code does what the plan and the filed rules say. I found no bug that would make a number wrong,
no way for scored seeds to reach the cut-offs, and no change for the frozen callers. The findings below are about what a
pass would mean (the probe), about cluster time that could be spent on a result already decided (the update-size
ceiling), and about two resume behaviours that could quietly throw readings away.

## Findings, by severity

### High

**H1. The possession probe reads the task, and on this truth set the task is the label.** `src/commute/revival.py`
`possession` (≈ l.183-226); `c1b.py:269-271, 343`; `predictions_c1b.json` `PC4_possession_probe`.
The probe's label is the state's own answer on natural `C b 40` rows. Task P makes that answer count(a)+1 and task N
makes it count(b), and the task is exactly what fixes "existing" against "new". LOW's residual carries count a more
strongly than count b (0.64 against 0.28 in A1), so P states read high and N states read low *whatever mechanism the
state uses*. A state that learned count(a)+1 through a brand-new circuit would read the same as one that reuses LOW's
count a. The filed `blind_to` admits this, but the instrument is still scored by the same pass rule, with an 80 % lean,
next to pruning, reverse fine-tuning and steering. A probe pass would then read as "a pragmatic instrument beats the
recipe ceiling", and C2 would measure VPD against a labeller that is reading the truth's definition.
*Fix (choose before filing stands):* keep the probe readings, but file the probe as a **task-identity reference**
(a positive control that the truth set is what its construction says) rather than a scored instrument. Leave it out
of "any instrument passes → C2" in the decision tree and out of the C2 comparison. The other option is to keep it
scored, but file in the predictions that a pass carries no evidence about mechanism and cannot count as a benchmark
win over VPD.

### Medium

**M1. Nothing stops the run once the update-size labeller has made passing impossible, so ~2.5 h of calibrate and
score jobs may only confirm a result that is already known.** `c1b.py:320-333` (`train_stops`, `recipe_ceiling_full`
is logged but no rule reads it); `predictions_c1b.json` `C1_truth.lean` gives a 40 % chance that update size reaches
≥ 15 of 21.
After `train`, the hindsight ceiling of all three recipe labellers on the full scored truth set is known for $0. If
update size separates P from N (P is a smaller change), then n_c + 2 > n and no instrument can pass at any n ≤ the full
set, because each instrument's own non-void set is a subset of it and the ceiling on a subset cannot fall by more than
the states dropped. The plan's decision tree still runs ~8 calibrate and score jobs and then asks the user anyway.
*Fix:* add one filed row to `train_stops`: if `max(pass_bar(n, majority), n_c_full + 2) > n` on the full scored truth
set, stop with a card (the user chooses whether the readings are still worth taking, as described-only). It is one
line in `train_stops`, and it needs the predictions entry before job 0.

**M2. If a constant changes, every reading is dropped without a prompt, and the next write overwrites `c1b.json`.**
`c1b.py:521-529`.
`keep` requires the whole `constants` dict to equal the stored one. Scenario: a train stop fires, the user says
"continue", and the lead amends `disagreement_stop`, or adds any key under `constants`. The next job logs one line,
starts from empty `checks`, `calibration` and `scored`, and the first `write_json` overwrites the file. The 40 checks
(~40 min) are redone. If this happens after calibration, the readings that the committed `c1b_cutoffs.json` was derived
from are gone, and `calibrate` then stops on its assert (or `score` does). The rule meant to protect the run destroys
its record.
*Fix (delete rather than add):* on a mismatch, exit 2 with the message, not a silent drop. Clearing `c1b.json` is
then a deliberate act by the lead. Failing that, rename the old file to `c1b.json.<date>.bak` before the first write.

### Low

**L1. The plan's "a P state not through → stop" row is not in the code or the filed rules.** `c1b.py:320-331`; plan
§6 row "checks on P at 10K"; `predictions_c1b.json` `rules.train_stops`.
The code and the predictions agree with each other (stop only at ≥ 3 disagreements among scored-seed states), but the
plan also stops on any single P 10K state that is "not through" count a. Such a state now becomes a silent
"disagreement", which counts toward the stop row only when it is on a scored seed. Decide which rule holds and file it
before job 0. This is a plan-against-filing mismatch, not a code bug.

**L2. If the deadline cuts the reverse fine-tune, the state's steer, probe and prune readings are thrown away as well.**
`c1b.py:339-348`.
These readings come first and take ~2-3 min (the probe trains 4K iterations). One job in two will lose that time,
which over the ~8 instrument jobs is about 10-20 min and may add one job. The result is still correct. *Fix (optional):*
check `time.monotonic() > deadline - <one state's measured time>` in `read_states` before starting a state, using
`probe_seconds` and C1's per-state time, so that no state starts when it cannot finish.

**L3. No lock stops two c1b jobs running at once.** `cluster/pcfg_anchors.sbatch` (c1b usage lines, "one c1b job at a
time").
Each stage has its own job name, so a resubmit made by hand while the last job is still running (it is easy to miss
that exit 3 has not happened yet) can run two `train` jobs on the same `<key>_train.pt`, or two writers of `c1b.json`.
*Fix:* give every c1b stage the same `--job-name=commute-e55-c1b` in the usage lines and add `--dependency=singleton`,
so that Slurm serialises them. No code change is needed.

**L4. Pruning and reverse fine-tuning decide "void" on different rows.** `c1b.py:354` (pruning: `acc0` on the 4,000
`data.eval` rows) against `c1b.py:361` (reverse: the start accuracy on the first 1,000 of them). A state near 0.95 can be
void for one instrument and read by the other. A state that reaches the target exactly at 5,000 steps and one that is
capped both read x = 5000 / scratch. Both are cosmetic at the expected values (P states are expected well below 0.95 on
`C a 40`). Note them in the predictions' E11 rather than change the code.

## Checked and found correct

- **Calibration leakage.** `compute_cutoffs` reads only `res["calibration"]`, which holds only `truth_at(calibration_seeds)`.
  Before `score`, the scored seeds meet only label-blind code (training, Stage A checks, task accuracy, update size).
  `score` refuses unless `c1b_cutoffs.json` is tracked, has no change against HEAD, and equals the cut-offs recomputed
  from `c1b.json` (the JSON round trip of floats is exact). It also runs only once every calibration-seed truth state
  has been read. If `git` is missing on the node, `subprocess.run` raises, so the guard fails closed. The cluster job
  runs in this NFS clone, so a commit made on the dev node is the one it sees.
- **Resume.** A state file is written only when `step >= it` and is atomic (`.tmp` then replace). `<key>_train.pt` is
  removed only after both saves are checked and written. Instrument rows are written after each state. A reverse run
  cut by the deadline returns `None` and the state is read again in full; the target is checked before the deadline, so a
  run that reaches 0.95 on the last read still counts. If the deadline cuts the scratch run in `controls`, `controls` is
  re-run from the start (~10 min) and nothing partial is saved.
- **Frozen callers.** With `reverse_finetune(cap=CAP, schedule_steps=CAP)` as the defaults, its behaviour is unchanged
  for `c1.py`, `amend5*.py` and `run.py`. The new `probe` and `GPT` imports in `revival.py` create no cycle. No
  existing constant or function changed.
- **Truth set.** The task-N rows (`task_batch(..., "C", B, c_rate=0.0)` on stream `[100 + seed, chunk]`, chunk 100, the same
  dropout seed) are the same as `run.finetune`'s for N1, N1s1 and M, so reusing those states is sound (also checked
  model-free). The P labels stay in range (count ≤ 40 against MAX_NUM 250). `placement` follows the plan, and
  `G_OFFSET[A] = 1` makes count a "through" read count(a)+1, which is what P outputs.
- **Rules against code.** `cutoff`: the balanced-accuracy numerators are exact integers, ties go to the nearest median and
  then the lower y, the gap midpoint is used when the classes separate, voids are dropped, and the < 3 per class and all-equal
  voids hold. `label_by_cutoff` is inclusive on both sides. `ceiling` searches both directions, the constant labellers
  and the midpoints, and `score_one` recomputes it on each instrument's own non-void states. The `pass_bar` table holds
  (24→17, 21→15, 18→13, 16→12; per-run 12→10). With N 1e-5 1K "not yet learned", the expected ceilings (lr, steps,
  constant: 12 of 21) are right. The probe's selectivity is held-out accuracy minus the accuracy of the shuffled-label
  probe, both against the state's answers on the same held-out rows, as filed.
- **Controls.** LOW's B_a reads the same 4,000 rows as `a1_LOW.json` `count_a`. P_a uses a1c's set-0 pairs
  (`[1, 30, 0, A]`, prefix A, block 5 = 0.96). The scratch recipe matches C1's first 5K steps at 1e-5. The stop and void
  rows match `rules.controls_stops`.

## Not checked

GPU behaviour of `possession` (its cost, and whether LOW's count-a selectivity on `C b 40` rows clears 0.4); torch.compile
recompiles once per reverse fine-tune (inherited from C1, about 20-40 s per state); lint.
