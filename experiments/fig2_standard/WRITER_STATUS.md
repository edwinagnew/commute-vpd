# E298 writer status (2026-11-24, WIP hand-over)

Branch `worktree-agent-af00cc91015939ef7`, worktree `.claude/worktrees/agent-af00cc91015939ef7`. Not pushed.

## Done
- `src/commute/needed_sae.py`: swap-test scan at SAE latents (latents + 8 error terms, 131,080 variables). Shortcut: a latent
  whose f is the same on x and x' at every position is not run (its swap changes nothing). `only` runs the cases it is given with no shortcut (identity control).
- `experiments/fig2_standard/{run.py, feats.py, eb.py, score.py}`, `results/predictions.json` (filed, committed).
  48 states: the plan's 42 plus E280's head-(2,4) edits v2/o2/q2 (block "head") and their matched randoms (user scope addition).
  Headline counts the leftovers. The no-leftover value and the leftover share are reported beside it.
- Reuse check (code, `eb.check_sources`, stops on a miss): E268, E277, E290 have the same derangements 4, examples 64 and seed 268. E277 and E290 have the same band p97.
  E290 has the same random_seed 161000, tasks and seeds. E280 uses E268's settings (head_edit/run.py:311, :166, :178) and KN.world() (:319).
  Windows: built states come from KN.world() (kstar_needed/run.py:89-95). E277 used the same RB.World (needed_neurons/run.py:139).
  Trained states come from E290's World.make (ia3_control/run.py:165-180), shared by its VPD and neuron reads.
  Pairs: IX.derangements(len(toks), 4, m, 268) everywhere (kstar_needed:169, needed_neurons:227, ia3_control:400 via needed_set).
  Leftovers are in every scan (needed.scan local arange(C+1); needed_neurons PER_LAYER includes the remainder).
  Read again: E290's VPD full/copy_s0, which was copied from E275 (ia3_control/run.py:385). Per the brief, copy_s0_rand too.
  Not reused: SAE (E263 read over K*, not K).
- `score.py` on the saved reuse files: 58 rows reused. All 52 rows that have an old-figure counterpart give definition.csv's values back exactly.
  The VPD and neuron values match the old figure. For neurons the comparison is on the like-for-like no-leftover field.
- Lint clean (ruff + pyright) on all new files.

## Not done / untested
- `repro_rel_tol` in predictions.json is null for all three. Fill it from the rehearsal's printed `rel_dev`, then refile. Until then stage 1 refuses to start.
- Nothing has run on a GPU. Never run: the SAE scan, the SAE identity, every stage-0 part, and the new built and head randoms (matched_random on built F).
- CPU checks were queued on the worker when the session stopped (logs in ~/commute-data/scratch/definition-lead/e298-writer/):
  - `check_sae.py`: f_u against encode, the hook against a direct swap, identity, and the shortcut against every case at mlp3.
  - Three INPUT_MANIFEST_ONLY dry runs, `dry_{sae,vpd,neurons}.log`, writing `manifest_*.txt`.
  Their results are unknown. Rerun them if the logs show no result. The INPUT_MANIFEST file with sizes is not yet written.
- Not measured: SAE seconds per item, peak GPU memory.

## Next commands
    # CPU checks (background):
    INPUT_MANIFEST=/mnt/nw/home/e.agnew/commute-data/scratch/definition-lead/e298-writer/m_check.txt INPUT_MANIFEST_CPU=1 ~/commute-agents/fleet/cpu-run -c 2 --mem 40G -t 60 python3 /mnt/nw/home/e.agnew/commute-data/scratch/definition-lead/e298-writer/check_sae.py
    INPUT_MANIFEST=<scratch>/manifest_sae.txt INPUT_MANIFEST_CPU=1 INPUT_MANIFEST_ONLY=1 ~/commute-agents/fleet/cpu-run -c 2 --mem 40G -t 60 python3 experiments/fig2_standard/run.py --feat sae --stage 1   (same for vpd, neurons)
    # rehearsal on cluster debug (E298 rehearse), one job per part fits 30 min except VPD/neurons x2 repro (~24/20 min):
    python experiments/fig2_standard/run.py --feat sae --stage 0 --tag r0
    python experiments/fig2_standard/run.py --feat neurons --stage 0 --tag r0
    python experiments/fig2_standard/run.py --feat vpd --stage 0 --tag r0
    python experiments/fig2_standard/run.py --feat sae --stage 1 --items trained/copy_s0 --tag r0   (time one SAE item; needs tol filed, or read via a temporary --stage 1 after filing)
    # scored: stage 0 per feat first, then shards
    PRED_SHA=<sha> python experiments/fig2_standard/run.py --feat sae --stage 1 --shard I N --tag sI
    PRED_SHA=<sha> python experiments/fig2_standard/run.py --feat vpd --stage 1 --shard I N --tag vI
    PRED_SHA=<sha> python experiments/fig2_standard/run.py --feat neurons --stage 1 --tag n0
    ~/commute-agents/fleet/cpu-run --mem 8G python3 experiments/fig2_standard/score.py
