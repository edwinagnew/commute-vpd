# E208 pods (cell P1.30)

Eleven H100 SXM pods run in parallel, one run in all. Every pod runs stage 0 first and alone: do-nothing, x2 and x10 on
copy24's 512 read windows. It writes `results/stage0_<tag>.json`, exits 1 on a miss, and is skipped on a rerun once
that file says done. Then come the pod's items, in the order given. Each item writes its JSON and npz when it ends.

## Inputs (paths from the project root; sizes measured 2026-11-18 with `du -L`)

**Every pod**, about 430 MB, about 45 s at 10 MB/s:
- `artifacts/vpd/t-9d2b8f02/` (256 MB; the checkpoint `ckpt/model_step_99999.pt` and the config)
- `artifacts/sets/train_0_32768.npy` (129 MB), `train_32768_8192.npy` (33 MB), `val_0_2000.npy` (7.9 MB)
- `experiments/open_pair/results/large/amend1/windows_copy_trained.pt` (528 KB), `windows_copy_new.pt` (440 KB): the
  copy24 windows for stage 0 and E195 rebuilt
- committed files that arrive with the code: `experiments/simplest_intervention/results/<item>.json` (E184's gaps),
  `experiments/built_control_sum/results/amend1_stage0.json`, `experiments/graft_sites/results/predictions.json`,
  `experiments/rule_retention_pairs/results/predictions.json`
- the tokenizer, `EleutherAI/gpt-neox-20b`, fetched from the hub at start, as on E196's pods

**Per pod**, besides those:

| tag | items (command-line order) | extra inputs | extra MB | expected min |
|---|---|---|---|---|
| p0 | `slotted neurons_appended head_appended rebuilt` | `experiments/rule_retention_fixed/results/large/rebuilt_mlp.pt` (37 MB); `experiments/graft_sites/results/large/neurons.pt`, `last_lowest.pt` (1.6 MB each) | 40 | 50-80 |
| p1s0, p1s1 | `copy24_s<s> cipher24_s<s>` | `experiments/open_pair/results/large/amend1/copy_s<s>.pt` (109 MB); `experiments/open_pair/results/large/cipher_s<s>.pt` (109 MB); `experiments/open_pair/results/large/windows_cipher_trained.pt`, `windows_cipher_new.pt` (1 MB) | 219 | 30-90 |
| p2s0, p2s1 | `ident32_s<s> rand32_s<s>` | `experiments/regating_table/results/large/ident32_s<s>.pt`, `rand32_s<s>.pt` (109 MB each); `windows_ident32.pt`, `windows_rand32.pt` (568 KB each) | 219 | 30-90 |
| p3s0, p3s1 | `case_real_s<s> case_twin_s<s>` | `experiments/case_label/results/large/dW_real_s<s>.pt`, `dW_twin_s<s>.pt` (55 MB each); `task.pt` (3.9 MB) | 114 | 25-70 |
| p4s0, p4s1 | `repeat_real_s<s> repeat_twin_s<s>` | `experiments/fresh_reuse_pair/results/large/dW_real_s<s>.pt`, `dW_twin_s<s>.pt` (55 MB each); `task.pt` (7.2 MB) | 117 | 25-70 |
| p5s0, p5s1 | `twin_s<s> fact_s<s>` | `experiments/built_control_sum/results/large/amend1_control_s<s>.pt` (256 MB); `experiments/memorised_fact/results/large/fact_s<s>.pt` (256 MB) | 512 | 40-90 |

The largest upload is p5's: 0.94 GB at 10 MB/s is about 95 s. Every other pod is under 70 s.

**Staging.** Every `results/large` is a symlink into `~/commute-data/results-large/`, so it arrives on the pod dangling.
Hard-link the pod's files into a staging tree, `experiments/<folder>/results/large/...` under one staging root, and sync
that (root `CLAUDE.md`). On the pod, replace each dangling `results/large` symlink with the real directory. Also make a
real `experiments/rho_saved_items/results/large/` there, because the run writes `results/large/parts/` into it. Check
sha256 on both sides of each state file. Delete the staging tree afterwards.

## Command (per pod)

```
PRED_SHA=<git show HEAD:experiments/rho_saved_items/results/predictions.json | sha256sum> \
  python experiments/rho_saved_items/run.py --tag <tag> <items> 2>&1 | tee -a experiments/rho_saved_items/results/run_<tag>.log
```

Exit 0 means done; exit 1 means a stop or a refusal. Read the log every 1-2 min. Each search logs its first order's
evaluation count and seconds, with the time that implies for 5 orders. If that time puts an item over 45 min, tell the
lead before the second item.

## Expected time, and what sets it

Each item runs two searches of 5 orders: mechanism replacement and activation patching. E195 rebuilt adds a third,
with its zeroed parts restored. E196's search took 224-622 s per item on one H100, with 13,000-43,000 evaluations
(`experiments/rule_retention_pairs/results/items/*.json`). Each evaluation here also runs M's heads and M's whole MLP
layer wherever a part is swapped, so allow 1.5 to 2 times E196's time per evaluation. K may also take more
evaluations, because it now carries the share of the change that lives in attention. That gives 6-21 min per search,
12-42 min per item and 25-85 min per pair pod. Pod 0 costs little for its three controls (256 or fewer candidates).
Its rebuilt item runs three searches of E195's size: 128 s per order on an H100, so 30-50 min. Add about 10 min of
set-up per pod.

**Cost.** 11 pods at 0.6-1.6 h each is 7-17 pod-hours. At $4.41/h (E196's H100 SXM rate) that is $30-75, about 1-1.5 h
of wall time. The upper end passes the $60 gate line. If the first orders project past it, the runner reports
before the second item, and the lead can cut the activation-patching search to 2 orders (a settings change before a
rerun, never mid-run).

## Downloads (per pod, as each item ends; terminate once the last item's JSON matches sha256)

- `experiments/rho_saved_items/results/items/<item>.json` (a few KB), into the same path
- `experiments/rho_saved_items/results/stage0_<tag>.json`, into the same path
- `experiments/rho_saved_items/results/run_<tag>.log`, into the same path
- `experiments/rho_saved_items/results/large/parts/<item>.npz` (about 0.5-1 MB: part names, the four r readings,
  each order's K), into `experiments/rho_saved_items/results/parts_dl/<item>.npz`. Do not download into the
  `results/large` symlink. Move the files into `results/large/parts/` after the pod is terminated.

Then, on the dev node: `~/commute-agents/fleet/cpu-run python experiments/rho_saved_items/run.py --summary`
writes `results/summary.json` (R1 and the R3 table).
