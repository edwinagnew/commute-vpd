# Appendix to the P2.17 plan: apparatus, functions, timings

*2026-11-11. Plan `reports/plans/2026-11-11_plan_p2-17-disagreeing-pairs.md`; rule
`experiments/disagreeing_pairs/results/predictions.json`. Every function below was found by grep in this clone; the writer
calls them, does not copy them (`.claude/CLAUDE.md` → No copying between experiments). Planned code:
`experiments/disagreeing_pairs/run.py` (about 250 lines), `rules.py` (pure, with the predictions file's edge cases as a
test), `plot.py`; `cluster/disagreeing_pairs.sbatch` copied in form from `cluster/prior_steering.sbatch` (debug, 1 GPU,
24G host, 40 min limit).*

## Functions to call

| step | function | file |
|---|---|---|
| keys, split | Amendment 3's 128 keys from `amend3.json` keys; `open_pair.split_keys` (seed 0), checked against `amend3.json` keys.held | `src/commute/open_pair.py` |
| original-model probabilities per key | `kv.key_distributions(..., geometric=True)` on train + held-out windows, fp32 | `src/commute/kv.py` |
| word ranking | `keydep.vocab(ctx)` → `ranked` (frequency-ranked word-like tokens) | `src/commute/keydep.py` |
| tables | `kv.bijection(name, marker, keys, pool, seed=0, ok=...)`; the one-word map as `kv.KVTask(name, marker, keys, (v,)*128)` | `src/commute/kv.py` |
| windows | E58 Amendment 6's `save_map` (windows with the table swapped, cut by key with `heldout_keys.split_task`, the zero update's record), via `load_experiment("open_pair", "amend6")` | `experiments/open_pair/amend6.py` |
| fine-tune | `keydep.finetune(KD.Ctx(net, {}, {}, {}, LARGE), w, seed, (30,))`, as E147's `refit` | `src/commute/keydep.py`; `experiments/refit_96key/run.py` |
| prior | `keydep.geo_rate(steering.load_task(net, path, name, name).lp0)`, prior_vs_share's path | `experiments/prior_vs_share/run.py` line 87 |
| strength-1 steering | `hf_state(job, name, 512)` from `load_experiment("prior_steering", "run")`, with an inventory built by its `e58_inventory` plus the two new maps; the zero-update control as its `stage0` first block (`fair_steering.vectors`, `fair_steering.logits_steered`) | `experiments/prior_steering/run.py` |
| share | `Job.trained(state, Job.delta(state))` from `load_experiment("share_untrained", "run")`, as E147 | `experiments/share_untrained/run.py` |
| calls | `call` from `load_experiment("holes_fill", "rules")` | `experiments/holes_fill/rules.py` |
| job plumbing | `commute.jobs.require_filed`, `hard_exit`; `STAGE n DONE` after each file | `src/commute/jobs.py` |

TF32 off, as E58 trained and E136/E143 read. The predictions file gets its E-number in a dated commit before the run;
`require_filed` refuses otherwise.

## States

- New: `common_cipher` and `one_word`, seeds 0-2, step 30, bf16 in `experiments/disagreeing_pairs/results/large/dW_<map>_s<seed>.pt`
  (6 × 57 MB), windows in `results/large/windows_<map>*.pt` (about 3 MB each map).
- Saved (read only, sha256 against `experiments/refit_96key/results/stage0.json`): `experiments/refit_96key/results/large/dW_{copy,elicit,rank1,class_shuffled,prior_cipher}_s{0,1,2}.pt`.
  Windows: `experiments/refit_96key/results/inventory.json` maps.<map>.windows.
- Differences from the reference recipe (E58 / E147): none for the fine-tune. The new tables change only the answer token
  per key; the common-word cipher's pool (ranks 0-2999) differs from E58's cipher pool (1000-2999), which is what raises
  its prior. The positive-control refit (prior-matched cipher seed 0) checks that nothing else changed.

## Stage times on an L40 (sources)

| part | estimate | source |
|---|---|---|
| model and decomposition load | 2 min | `cluster/logs/commute-refit-96key-57961.out` (E147) |
| stage 0: zero update, positive control (6 saved states, share + steering), one refit, 5 priors, two tables | 2 min | E147 stage 0: 24 refits with share reads in about 4 min after load; E149 stage 0 |
| stage 1: 6 fine-tunes (about 1 s of training each, `refit_96key/results/stage0.json` elapsed_s) with windows and saves; strength-1 steering on 18 states | 2 min | E149: 68 states in about 3.3 min (`commute-prior-steering-58001.out`) |
| stage 2: share on 18 states, score | 3 min | E147 stage 0 rate, about 10 s per state |
| total | about 10-12 min, $0 on debug | peak GPU 10.6 GB, host 6.1 GB in both E147 and E149 |

## Early checks (filed values for the first minutes)

- The zero update steers by at most 1e-9 (E149's `zero_tol`).
- The positive control's share and steering reproduce within 1e-4, and the prior-matched cipher's seed-0 refit share within
  0.0088 of E147's (`refit_96key/results/stage0.json` states.e58/prior_cipher_s0.r).
- The five saved maps' priors reproduce `prior_vs_share/results/stage2.json` within 1e-4 relative.
- Each table's prior sits in its filed range before any training; each new fine-tune's gain is at least 1 (the floor).
- After four states of a stage, a projected job time over 25 min stops the job.

## RunPod fallback (only if the debug job has not started within 10 min)

One 4090 (about $0.7/h) under remote-kernels, pod `commute-disagreeing-pairs-<id>`. The saved updates (about 1 GB) cannot
cross the 80 KB/s uplink, so on the pod every side is refit from its windows (about 1 s each, deterministic on a card: E147
reproduced E58 within 0.0006) and the positive control compares the refits with E147's shares within each map's seed
range instead of 1e-4. Windows are staged by hard link into a `results/large/` path and sent with
`sync(include=[...])` (root `CLAUDE.md`). Download `stage{0,1,2}.json` after each stage, check sha256, terminate. Expected
cost under $1.50 including set-up; the cell allows $15.

## Figure (from `plot.py` after the run)

One row per pair (A, B, D, then C marked "seen before filing"), three columns: prior gap in decades with the 0.5 band,
strength-1 steering gap and share gap in units of the gain with the 0.1 band, each with seed points per side. Sketch with
made-up numbers: `reports/plans/2026-11-11_plan_p2-17-card_sketch.png`.
