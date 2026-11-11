# Qwen instrument check, step 1: changes whose placement we know

*2026-11-11, written for `qwen-lead` from the brief `reports/plans/2026-11-11_plan_qwen-instrument-check.md` (Step 1)
and step 0's results. Cell P3.3, E152. Nothing here has been run. New decomposition p-9707d6ff throughout.*

We build changes to the unchanged model whose right placement we know, and read them with the pipeline every Qwen
reading has used. If the least-squares split puts a planted change back on the subcomponents it was planted on, the
readings of real updates can mean something; if it spreads the change as it spreads a random one, they cannot.
This decides whether any reading of where an update sits goes into part 3 of the paper.

## What runs

`experiments/qwen_instrument_check/step1.py`, one cluster L40 job in `debug`, $0:
`sbatch --comment='E152 stage: step 1' cluster/qwen_instrument_check_step1.sbatch`. Rules and expected values are
in `results/predictions_step1.json`; the job refuses to start unless that file is committed unchanged.

**Three kinds of change.**
- *Planted on often-on*: at each of the 28 down_proj matrices, the alive subcomponents switched on (gate above 0.01) at
  over half of W0's maths positions. That is 648 in all, from 2 to 72 per matrix.
- *Planted on rarely-on*: as many at each matrix with the lowest share on, which is zero on maths for all 648 (they are
  alive through web text).
- *Random*: E132's Gaussian change at all 196 matrices, with E40's input-weighted size per matrix.

A planted change rescales the write vectors of its set, so each member writes more of what it already writes. That is
the re-weighting picture the project starts from (GOALS.md), and the right answer is "exactly these subcomponents". Set
size is not chosen: it is what "on at most maths positions" gives. We use down_proj because step 0 found every MLP
matrix fully reachable by the split. Its write vectors land in the residual stream.

**Three sizes.** Each kind is scaled to match an output change: 0.064 nats per token for the reinforcement-learning
update RL s0, 1.635 for the supervised fine-tune E40 step 10, and 0.324, their geometric midpoint. Output change is the
KL from W0 to the state on the 128 maths rows of 256. E57's search finds the scale within a factor of 1.25. The brief's two
supervised sizes are one: step 10 and the first fine-tune at step 100 sit 5% apart.

**Three readings per state.**
1. The share of the split's energy on the planted set, against chance. Chance is the random change's split, rescaled
   matrix by matrix to the planted change's size, read on the same set.
2. E94's share on subcomponents already on at W0. E94 never ran on new, so this job counts the on set itself, on the
   rows and with the rule E94 used.
3. E57's re-gating share, exactly as E132.

## Expected values and the calls

The split is linear, so it runs once per kind and is scaled for each size. Readings 1 and 2 therefore cannot change
with size, and size enters only through the re-gating share. The expected values come from a CPU dry read of the
job's own sets without input weighting (`preview_step1.py`, `results/step1_preview.jsonl`).

| reading | often | rarely | random |
|---|---|---|---|
| share on planted set | 0.75 to 0.95 | 0.1 to 0.8 | chance 0.002 to 0.06 |
| share on already-on set | 0.85 to 0.98 | 0.02 to 0.2 | 0.3 to 0.7 |
| re-gating share, mid and sft | 0.98 to 0.9995 | the same | 0.985 to 0.995 |

The brief's table becomes three filed calls:
- **Split** (rows 1 and 3). Recovery r = (share − chance) / (1 − chance). "Works" when both kinds reach r ≥ 0.5 and
  the already-on shares differ by at least 0.5. "Free" when both kinds have r < 0.2. Expected: works. The lean is
  medium for often and weak for rarely, since input weighting may pull rarely's share down.
- **Size** (row 2). At each size, are the often and rarely re-gating shares apart by E132's rule? Apart means a gap
  above E132's seed spread of 0.0020, with a resampled interval that excludes zero. The floor is the smallest size
  from which they stay apart. Expected: apart at no size (weak lean).
- **Re-gating blind** (row 4): no pair of the nine states separates. Expected: some pair does (medium lean).

Each rule was run on its edge cases in `tests/test_qwen_instrument_step1.py`: an empty set, chance at 1, ties at
each bar, an unreadable or non-monotone size, and no readable pair. The unchanged model and do-nothing give no split
energy, so they can pass no split bar.

## Order and stops

Controls run first and alone, and each stops the job on a miss:
- W0's cells match E132's stored numbers to 1e-9 relative. The measured floor across jobs is 3e-16.
- E40's KL on 16 rows matches E132's, and do-nothing reads a share of exactly 1.
- At least 95% of the often set is on, and at most 5% of the rarely set.

Then the positive control: the often kind's split must put at least half its energy on the planted set, or the job
stops before any read. The dry read gives 0.90 to 0.94 per matrix. Then the rest of often, then rarely, then random.
Each state writes `results/step1/S_<kind>_<size>.json` and a `STAGE` line as it ends, and every number goes to
`results/step1/run.log`.

## Cost and time

$0 on the cluster's debug QoS, one L40. Not a gated run. The timings are E132's job 56867 on the same pipeline
(`qwen_regating_new/results/run.log`):

| part | time |
|---|---|
| loads | 95 s |
| controls | ~6 min |
| input moments | 3 min |
| one split | 6.2 min |
| one read | 4.6 min |
| one size search | up to 8 batched passes of ~20 s |

That is 85 to 97 min in all. GPU peak 42.4 GiB, host 91 GB, `--mem 128G`. The job exits 3 with its results written
if the 2 h limit comes first; resubmitting the same line resumes.

## What the brief asked that this does not do as written

- Rows 1 and 2 cannot be told apart by the split's own readings, which are linear in the change. Only the re-gating
  share can show a size floor.
- Two of the brief's three sizes coincide, so the midpoint replaces one (the lead's decision).
- E132 used two random changes. This uses its Gaussian only.

## Not run

- Attention matrices. There the split cannot reach everything (step 0: 0.25 to 0.62 of a random change left outside),
  so a planted change on write vectors would also test how much of a placement leaks into the part the split cannot
  hold.
- Planted changes in another direction than rescaling, more than one draw of the random change, and the old
  decomposition.
- The other three explanations the brief names: confounded anchors, masks frozen at W0, and the rank-one unit.
