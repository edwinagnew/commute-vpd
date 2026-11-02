# Review: E34 Amendment 8 code, before its GPU runs

*2026-11-02. Independent review of commit bef97f3 (`experiments/teacher_training/amend8.py`,
`results/predictions_amend8.json`, `cluster/teacher_amend8.sbatch`) against
`reports/plans/2026-11-02_plan_base-control.md`. Nothing was run on a GPU and no model was loaded. Code and rules were
read against their sources; the rules and filed config were run through `cpu-run`; Adam was simulated on plain noise.*

I read the code and the filed rules for what would make a reading wrong or a job wasted. I found no crash and no wrong
recipe: N5 can run as it is, and N1, N2 and N4 need small rule and early-check fixes. N3 should not run as filed, because
its null teacher will probably not be null and its size ratio will probably read "about" for a reason unrelated to the
question, and N3 decides what "teacher training copies the update" may claim before Qwen uses it (`GOALS.md`).

## The one estimate behind findings 1 and 2

Adam under gradients that are mostly noise walks as the square root of the steps, about 32 times its first step by step
1,000. If its first steps are loud, it remembers them and later steps shrink, to about 9 times.

| Run | Size at step 1 | Size at the end | End / step 1 | Source |
|---|---|---|---|---|
| Simulation, pure noise | 1 | 31.8 | 32 | `~/commute-data/scratch/commute-lead/amend8-review/adam_noise.py` |
| Simulation, first 3 steps 100 times louder, then noise | 1 | 9.2 | 9 | same |
| Simulation, noise plus a weak pull back to the start | 1 | 5.0 | 5 | same; the strength of the pull is arbitrary |
| Real fine-tune T1000, loud start | 0.00346 | 0.0321 | 9.3 | `run_e41_amend2.log`, E41 A2 |
| Real student, loud start | 0.0010 | 0.0184 | 18 | E41 |
| Student on lowercase text, teacher with damage 0.23 | 0.0011 | 0.0365 | 33 | `amend2.log` |
| Student on true labels, lowercase text | 0.0010 | 0.0541 | 54 | `amend2.log` |
| Full-weight Adam 3e-5 on true labels, 300 steps | not logged here | 0.0489, damage 0.235 | about 14 if step 1 is 0.00346 | `amend2.json .reads.nothing_adam` |

The real runs in the lower half of the table have no single loud start, and they walk at about the noise rate or above.
A self-label run has no loud start either. How strongly W0's own labels pull it back is the one thing unmeasured; the
simulation shows such a pull can matter a lot, which is a reason to measure it first, not to assume either end.

## Findings, ranked

### 1. High: N3's null teacher will probably be neither small nor null

`amend8.py:348` (1,000 fixed steps), `:75` and `:394` (`null_kl1` upper bound 1), predictions lines 31 and 35 ("damage
small", "T1000's 0.032 is an upper guide").

**What goes wrong.** Projected from the table, the null fine-tune at step 1,000 is about 0.09 in size (25 to 32 times
0.00346), not under 0.032. E39's 3e-5 runs read damage 0.235 at size 0.049 and 0.50 at step 1,000 of the labelled
component run, so a damage of 0.3 to 0.8 is plausible against the real teacher's 0.055. That has two consequences:
- `null_kl1`'s upper bound is on arm A's weighting, the trigger plus the mean elsewhere, so about twice the damage. It
  can stop the job at about minute 17, before the null student, and a rerun stops the same way.
- If it does not stop, the student copies a teacher that has really moved. A student of 0.018 is then explained by that
  teacher's own change, and the plan's "the move is largely the recipe" does not follow.

**Fix.** Spend one `debug` job first on the null fine-tune alone with checkpoints at 1, 10, 30, 100, 300, 1000, logging
size and held-out KL to W0. Then file: the null teacher is the first checkpoint at or above the same-job T1000's size
(the estimate says near step 100), the student trains on that, and the upper bound comes from the measured value. Keep
step 1,000 as a second student only if the curve says it is informative. The plan's N3 row fixes the rate and batch, not
the step, so this stays inside it.

### 2. High: N3's size ratio does not measure "how much of the move would happen anyway"

`amend8.py:139-144`, `:190-194`, predictions lines 27 and 60-61.

**What goes wrong.** The real student's first steps are loud (KL 5.3), so Adam's memory shrinks its later steps. A null
student has no such start and takes full noise steps. Its size can exceed the real one's whatever part of the real move
is recipe: 18 times against up to 32 times step 1, so r near 1.2 to 1.8 even with an unchanged teacher size. The rule
reads anything at or above 0.75 as "about", which triggers the card (dry run: r = 2 gives `outcome: fails, card: True`).
The result would be nearly fixed in advance. Separately, lean line 61 argues from "E41 A2's T1 teacher ... a teacher with
a small gap": T1's gap at W0 is 4.48 nats (`teacher_training_e41_amend2.json .teachers["1"].gap_w0`) against T1000's 5.09.
Its size is small; its gap is not.

**Fix.** Read direction in the job, not "later": `finetune.cosines` of the null student's delta-W against the real
student's, about 1 line, both states are in memory. Add to the card's rule that "the recipe" needs direction as well as
size, and give the cosine a positive control and noise floor: the real student against a second batch order, one more
1,000-step student of about 3 min, since E34 Amendment 4's state is not in `results/large/`. Correct the lean's T1
argument before filing.

### 3. Medium: N1's and N2's filed outcome for "partly" differs from the code

`amend8.py:183`, `:187`; predictions lines 44, 52.

**What goes wrong.** `gap_class` returns "closed", "partly" or "not closed". The filed text says "gap not closed ->
'fails'; pass -> 'holds'", which read literally sends "partly" with a pass to "holds". The code sends it to "fails"
(dry run: `partly, pass -> fails`). N1's own lean is "partly", so this is the likely case.

**Fix.** File "gap other than 'closed' -> 'fails'". Then look at finding 4: N1 fails on a walk it was never going to avoid.

### 4. Medium: N1's tree has no row for its likely result, and its lean ignores the walk

Plan line 53, predictions lines 33, 45.

**What goes wrong.** The plan reads a fail as "check above the line with the gap closed". The student from a random start
has a small gap, no loud start and fresh batches, so it should walk: 0.018 and 0.022 to 0.035 in quadrature is 0.028 to
0.039. That makes a growing size and a gap "partly" or "not closed" likely. The check at that end size has two references
that disagree: E39's random moves at 0.036 read 1.03 and 1.07, E37's trained lowercase state at 0.0365 reads 2.2 (different
text). So "verdict pass, high" rests on the start's size and on the walk behaving like a random move, which is open. Decide the reading of "fails, gap not closed" before the run. The same walk
bears on N2's return, whose lean already expects a miss and so a card.

**Fix.** Add the tree row; lower or re-argue the lean. No code change.

### 5. Medium: N1 can spend a whole job to read "unscorable", and its line comes from another card

`amend8.py:73` (`n1_kl1` lower bound 0.0005), `:81` (`gap_line` 0.002), `:183`.

**What goes wrong.** The lower bound is under the 0.002 that scores the gap, so a start between them runs to the end and
reads "no power" (dry run: `unscorable`). The margin is thin: E39's random move at size 0.0182 did damage 0.0027 on plain
text (`broad_fragility.json .reads.move_s0_x0.5`), so about 0.005 on arm A's weighting, about 2.5 times the line. The step-1 value also contains the hook floor, so
it cannot test this. And 0.002 is "3 x the largest floor", whose readings (0.00050, 0.00057, 0.00070: E41 C2, E41 A2 C2,
E34 C3) are all H100. There is no L40 hook-floor reading in the result files; the L40's component-path value is 0.00163
(a different path). `kl_w0` is read on merged weights, so the hook floor is not even its own floor.

**Fix.** Add `check(... "start gap above the line", out["reads"]["0"]["kl_w0"], (AM.gap_line, inf))` after `read "0"` in
n1, at 2 min rather than 20. Log the warm-start KL as this card's hook floor, which the job already measures. If it is
over about 0.0007, the line needs a dated amendment before the reading is used.

### 6. Medium: N2 can read the gap "closed" with the behaviour still present

`amend8.py:82`, `:186`; predictions line 52 ("E41's works_gap").

**What goes wrong.** E41's 5% was on KL(teacher || student), where 0.25 nats means P(target) 0.78. Here the direction is
KL(W0 || student). At the trigger that is about -log(1 - P(target)) plus a share of the rest, so the line of about 0.28
nats (5% of about 5.7) is met with P(target) near 0.25, about 60 times W0's 0.004. The filed expected value "P(target)
back near 0.004" is not in the rule. The same percentage does not carry across directions.

**Fix.** Add a filed P(target) line to the N2 rule (for example under 0.04, ten times W0's), or use an absolute gap line.

### 7. Medium: N4's gain check can false-stop on a number measured on another card

`amend8.py:99`, `:442`.

**What goes wrong.** `gain_tol` 0.02 absolute around H100 values. Along one H100 run the same recipe reads 0.143, 0.156,
0.134 at steps 100, 300, 1000, so 0.02 is about the run's own wander, and no cross-card reading exists. E39's own match
tolerance was 0.30 relative (`amend2.py`, `gain_tol`). A miss stops at about minute 13 and repeats on rerun, and N4 has no
self-label reading at all.

**Fix.** Use the 0.30 relative bound, or log and do not stop. Keep the step-1 CE check, which is a real test of the rows.

### 8. Medium: N4's positive control is filed but not enforced

`amend8.py:159-164`, `:441-442`; predictions line 71.

**What goes wrong.** The true-label Adam state is "expected to fail far above the line", yet nothing reads that. If it
passes on this card, a self-label pass still scores `near base`, `holds`, `card: False` (dry run), a reading with no power.

**Fix.** In `n4_adam`, stop or mark the outcome "no reference" when the true-label verdict is not "fail". SGD keeps its
filed exemption.

### 9. Low

- **Real-student reads not compared.** N2's step 0 and N3's `real` have a filed expected check of about 0.96, from an H100,
  with no L40 reading of this state in the result files; `score` ignores them (`amend8.py:190-194`). Put their verdicts in
  the score so a surprise there is seen before N3's ratio is read.
- **Share on a stochastic-only fail.** `label_class` uses PGD ratios only, so a self-label state failing on the stochastic
  ratio with PGD 0.9 reads share -0.02, "fails, under the true labels" (`amend8.py:167`). Label that case.
- **Source of "check 1.39 L40".** Predictions line 31 cites "plan E34's table"; the plan has no such table. The number
  appears in `teacher_training_amend4.json` and other files; name the right one.
- **Wall time.** Estimates hold on L40 rates (below) but n3's 24 min is close to `--time=00:30:00`, and E42's job asked for
  45 min on the same QoS (limit 2 h). 35 min costs nothing, and a timeout repeats 7 min of W0 plus all training.

## Checked and found right

- **Filed config.** `predictions_amend8.json`'s `amend8` and `config` equal `asdict(AM)` and `asdict(CFG)` (run through
  `cpu-run`, importing the real module; the sibling imports all resolve).
- **Student recipe.** `train_from` matches `path.student`: batch 32, seed 0, Adam 1e-5, clip 1, arm A's weighting. Size is
  always from W0, including from a moved start (`Offsets.delta_w`), so the return fraction is what it says.
- **Regenerated pair.** Fine-tune step-1 5.3339 (`run_e41.log`), T1000 P 0.99961, size 0.03209, student KL 5.3418, P
  0.99675, size 0.01842, all match the source, and L40's E31 values fall inside the T1000 bounds. Bounds are wide enough
  for rounding. No L40 student reading exists to test `real_p` and `real_size`, but they are wider than batch-order spread.
- **Warm start.** The offsets and teacher come from one state, the check runs at lr 0 and can fail. The E41 C2 lines hold
  (0.00050, 0.00057); its 0.005 bound is about 7 times them, a reasonable margin for the unmeasured L40 value.
- **`n5_kl1`.** G2 `path_kl` 0.00163 is L40, and M0's log shows the same value at step 1 of the trainer, so the 10% band
  is safe. `coeff=0` leaves the unmasked term as the only gradient. M0 is the right reference (`curve[0]["unmasked"]`).
- **`self_ce`.** Labels at position t are drawn from W0's distribution at t, the same index that next-token CE and arm A's
  trigger use. Batch order is the real fine-tune's. CE minus entropy would catch an off-by-one. Generator device, the
  `(b, L)` reshape and `multinomial` over 4096 rows of the vocabulary are sound; about 0.8 GB on an L40 of 46 GB.
- **N3's step-1 checks.** A full-weight Adam first step moves every entry by the rate unless its gradient is near 1e-8, so
  the 5% band is fair. E41 T1's student step 1 of 0.0010 supports the 30% band, with the limit that all those teachers have
  large gaps.
- **N4's recipe.** Rows, `n_train` 2400 (`ft_steps` 300 x `ft_batch` 8), held and plain rows, batch 8, seed 0, `next_token_ce`
  and rates match `broad_fragility/amend2.py`; 7.24, 1.28, gains 0.1343, 0.1253 and step-1 CE 3.5300 match `amend2.json` and
  its log; SGD's first 100 steps are the same batches as E39's read at 100. Set files exist.
- **Rule functions.** `outcome`, `gap_class`, `size_class`, `label_class` behave as filed on the cases run, apart from
  finding 3. `base_verdict` is `state_verdict` with P(target) out, and its keys exist in `pile_ratios`.
- **Resume.** Finished reads are skipped, states are saved before records, and `load` refuses another card or floor. The
  floor is bit-identical across four L40 jobs' files, so that equality will not false-refuse. Training is regenerated on a
  rerun, and the same floor across jobs suggests it reproduces.
- **W0 first and cards.** The job reads W0 before anything else and stops on a miss; the cluster has one L40 node, so the
  floor check will not false-stop on another card.
- **Slurm.** Exit status survives the `tee` (`pipefail`, `|| rc=$?`), `env.sh` puts this clone first on `PYTHONPATH`, inputs
  named in the header exist, memory matches E42's L40 jobs.
- **Time, on measured rates.** L40 W0 411 s and read 125 s (job 40338). H100 student about 72 s per 1,000 steps, fine-tune
  34 s, label 77 s, scaled by 2.25 (reads, 124 / 55 s). n1 about 20 min, n2 22, n3 24, n4_adam about 20, n5 14, as filed.

## Not run

- Any GPU stage, including the null fine-tune: findings 1, 2 and 4 are estimates from logged runs and a simulation.
- The L40's hook floor and merged-path floor: no reading exists.
- A second reading of the filed "0.99-1.07": `broad_fragility.json` has 0.988 at this size and 1.03, 1.07 at twice it, so
  the range is right but its top is from size 0.036.
- Whether compute-node tokenizer loading works for `n4`: inferred from cluster jobs whose logs mention it, not tested.
- `self_ce` on the real model; the writer's small-module test was not repeated.
