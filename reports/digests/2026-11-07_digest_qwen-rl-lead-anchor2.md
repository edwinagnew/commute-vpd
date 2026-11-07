# Digest, qwen-rl-lead, 2026-11-07 (overnight)

**D3 anchor 2, day 5 of 8: the maths version (E120) is done and handed over as a provisional pair; the facts backup (E128) waits on one answer from you.** I stopped because everything left depends on ask 14: which source of post-cutoff facts to use (log card "Anchor 2's backup: which source of post-cutoff facts?", https://claude.ai/artifact/SvgkzUiESRkyE9GM65HCWu).

## What ran
- **Grader fix (E120 Amendment 2, $0).** The maths grader now accepts answers written another way: a choice letter, units in `\text{}`, percent signs, lists in any order. 14 of 16 named misses now match and the grader controls pass. The fixed problems turned out to be ones the model nearly always solves, so they moved to the known side. The pools came out at 77 a side, under your 100.
- **Ask 13, taken by default after two hours:** go to the facts backup, and also fine-tune on the 77-a-side maths pools as a provisional row.
- **Four maths fine-tunes** (known and new pools, two seeds; cluster jobs 54969 and 55011-55013, $0, about 36 min each). Update states are saved in fp32 at matched sizes 0.35 and 0.49 (steps 5-7), at step 10 and at step 100. qwen-decomp-lead has a note with the paths.

## What it found (`reports/analyses/2026-11-07_analysis_e120-pass-rate-pools.md`, `notebooks/E120_qwen_pass_pools.ipynb`)
Both fine-tunes reproduce every training problem. Training on the never-solved problems solves almost none of the held-out never-solved ones: 5% and 2.5% against the 10% bar. Both fine-tunes cut GSM8K test accuracy from 0.78 to between 0.42 and 0.55. This is consistent with heavy overfitting (about 103 passes over 62 problems), but no cause is named.

| Filed prediction | Result |
|---|---|
| Pool size at least 100 | missed (77) |
| Novelty check | passed |
| New-pool training carries to held-out problems | no |
| Known-pool training leaves GSM8K within 0.05 | missed, both seeds |
| New-pool update larger than known-pool update | holds |

All of it is provisional.

## Prepared, not run
- **E128 plan:** `reports/plans/2026-11-06_plan_e128-post-cutoff-facts.md`, with Fable's advice beside it. About 3 h on the cluster at $0 once the source is chosen.
- **Shared fine-tuning code:** moved into `src/commute/pool_sft.py`.

## Money and problems
- **Spend:** $15.72 this session, all of it on Nov 6's sampling pod; nothing since. No pod is running.
- **Problem 4:** a subagent's CPU test reached a GPU and loaded the model for 31 s outside Slurm. There was no harm, but `cpu-run` should hide the GPU.
- **Friction:** 610, 612, 613, 621: the venv for the maths grader, `cpu-run` exposing the GPU, the normal queue not starting jobs, and the sampling script not exiting after its last stage.
