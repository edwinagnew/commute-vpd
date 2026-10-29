# What is saved in each results/large

Every `experiments/<name>/results/large` is a symlink to `~/commute-data/results-large/<name>/` (cluster NFS; 43 folders). Contents as of 2026-10-27, read from `ls` and the scripts' `torch.save` sites; nothing was loaded.
Patterns: `{S}` seed, `{N}` step, `{R}` LoRA rank, `{T}` task (`emoticon` or a ladder task `r{rung}_t{trigger}_v{target}`), `{lr}` learning rate as written in the name. Every `.pt` loads with `torch.load(p, weights_only=False)`. "Unknown" means the scripts did not say.
Common terms: **arm A** = Adam, batch 32, clip 1, kl_lambda 1, on the task's trigger windows (`finetune.ArmAConfig`). **dW dict** = `{matrix: delta-W}` for all 24 matrices, fp32, CPU (113 MB). **CT state** = `{matrix: (delta-W, U2, V2)}` in theory layout, from `component_training.train` (234 MB). **offsets file** = `{"offsets": {matrix: (dU, dV) bf16}, "layout", **meta}` from `arms.save_offsets`; rebuild it with `saved_states.offsets_state(p, alive_uv)`. **sources** = `{matrix: (C+1,) levels in [0,1]}`, the last entry being Delta's, from `gamma.pgd_sources`. **shift file** = `{"shift": {matrix: (C,)}, "shift_null": ...}`; load it with `turned_down.load_shift(p)`.
Emoticon windows and alive masks always come from `transport_test/results/large/` (see the transport_test section).

## Network volume (RunPod `/workspace`)
One row per directory a run wrote there, added by the lead from the pod-runner's report, with the run, the size and
what it holds. A path a plan or brief names that no run created gets a row saying so: `/workspace/e40_nowarmup` was
named only in E40 Amendment 1, which never ran, and a plan writer had to dig through `run.log` to find that (friction 68).
- `/workspace/e40/`: E40's saves (`reports/plans/2026-10-28_plan_qwen-sft.md` section 5 lists what was planned),
  e.g. `dW/step_1.safetensors`. Contents not yet listed here from a run report.
- `/workspace/e40_nowarmup/`: never created (E40 Amendment 1 did not run).

## qwen_maps (E44 input, regenerated on the cluster 2026-10-29, job 42866)
- `~/commute-data/results-large/qwen_sft/nowarmup_regen/dW/` (14.1 GB, never committed): `step_{1,10,100}.safetensors` (2.82 GB each),
  `step_100_fp32.safetensors` (5.64 GB), `SHA256SUMS` (copy committed as `experiments/qwen_maps/results/regen_SHA256SUMS`).
  Written by `experiments/qwen_maps/regen.py`; check record `experiments/qwen_maps/results/regen.json`.
- W&B artifact `edwinagnew-mats/commute/qwen3-1.7b-e40-nowarmup-dW` (version in `RUN_BRIEF.md` section 4): the same four files, for pods.

## adjoint_sgd
- Logs only: `full_{A..E}.log` (shards), `full_amend3_r{0,1}_*.log`, `full_amend3_emo_*.log`. No script here saves tensors, and every state is regenerated in-job (see the run.py docstring: arm A Adam 3e-5 / 3e-6 and SGD 0.01, 1000 steps, seeds 0/1).

## cancellation
- Logs only: `full.log`, `full_smoke.log`. No tensors. States are rebuilt from reading_a_coverage through `saved_states`.

## cancellation_complement
- `sources/{W0,ct_{T},ls_emoticon,adjoint_emoticon}.pt`: the adversary's sources (seed 0, 20 steps, 128 Pile val rows) for each model. Written by run.py stage `amend1` (`amend1_arms`) and loaded by `saved_sources(name)` there. The models are not stored here: `ct_{T}` is `saved_states.ct_state(reading_a_coverage, T, 10)`, and `ls`/`adjoint` are `saved_states.split_states(...)` of reading_a_coverage's emoticon `dW_ft_step10_s0` (tol 1e-6).
- `sources_amend2/{W0,ct_r1_t534_v369}_{seed0,seed1,seed2,alive_whole}_{N}.pt`: sources at cumulative step N (`RUNS`: seed0 at 20/50/100; seed1 and seed2 at 20; alive_whole, started with every alive level set to 1, at 0/20/50). Written by run.py stages `amend2_fixed`/`amend2_strong` (`search`). The states are the same as above.
- `sources_amend3/{model}_s{S}_{n}.pt`: the budget sweep. Sources after n in {20,40,80,160} sign steps of size 2/n from seed S's random start. Written by run.py `amend3_budget` (sharded). The models are W0 (s0, s1), ct of r0/r1/r2_t275, ls_emoticon and adjoint_emoticon, all from reading_a_coverage as above.
- Logs: `full.log`, `amend1.log`, `amend2_{fixed,strong}.log`, `amend3_shard{0-3}.log`.

## cancellation_sgd
- `sources/{sgd_r0_step100,sgd_r0_step1000,sgd_r1_step1000,adam_r{0,1}_step1000,sgd_rand32,sgd_ident32,sgd_r2_step100}.pt`: sources from `excess.measure` (seed 0, 128 rows), one per state. Written by run.py `revert --shard N`. The states are not stored here: they are offsets files in sgd_ladder_long and sgd_kv (the `STATES` map in run.py), rebuilt with `offsets_state`.
- `sources/shard{0,1}_W0_s0.pt`: W0's sources from the same call.
- `sources_budget/{W0,sgd_r{0,1}_step1000,adam_r{0,1}_step1000}_s{S}_{n}.pt`: budget sweep sources, n in {20,40,80,160}, step size 2/n. Written by run.py `budget --shard N`.
- Logs: `revert_shard{0,1}.log`, `budget_shard{0,1}.log`.

## concentration_test
- `{top1,top10,top100,siblings}.pt`: `{"uv": {matrix: (U2, V2)}, "drop": {matrix: (n_alive,) float64}}`. `uv` is the arm's step-30 state; `drop` is the revert-one drop in held-out log-odds of `o`, given only for matrices whose subcomponents moved. Written by run.py `all`, and re-read by run.py `amend`/`gating` to rebuild the masks.
  Recipe: emoticon, arm A loss, Adam over the masked (U, V) for 30 steps; lr picked by the rule from 1e-5..1e-3 (the selected lr is in results/concentration.json).
- `{adam_all,sgd_all}.pt`: the same dict with `uv` = None (drop only). The states are not saved; to regenerate: Adam lr 1e-5, or SGD with lr from {0.3,1,3,10,30} picked by the rule; all alive (U, V), 30 steps, emoticon, seed 0.
- Logs: `full.log`, `smoke.log`, `amend.log`, `gating.log`.

## gamma_refit
- `ft_delta.pt`: dW dict of the emoticon arm-A fine-tune (Adam 3e-5, 10 steps, seed 0). Written by run.py `prep` (`make_states`).
- Missing: run.py also writes `state_ls.pt`, `state_adjoint.pt`, `state_ct.pt` (CT lr 1e-5, 10 steps), `stage0_sources.pt`, `gamma_{tag}_final*.pt` (bf16 Gamma state_dict, loaded with `gamma.load_gamma`), `mean_ci_*.pt` and `trigger_gates_*.pt`. None of these are present.
- No logs.

## gamma_shift_map
- Log only: `full_amend1.log`.
- Missing: run.py writes `{label}_shift.pt` (shift file: E18's fixed change and the ' http' null). None are present. The fine-tunes are regenerated by `regating.finetune` with CFG's opt, lr, steps and seed.

## identity_test
- `revert_one_{pullback_s10,pullback_s30,lift_s2,lift_s3,lift_s10}.pt`: `{matrix: (n_alive,) float64}`, the revert-one drop in held-out log-odds of `o`. Written by run.py `all`.
  The states are regenerated in-job and not saved:
  - `pullback_s{N}`: component training, Adam 1e-5 on the alive (U, V) offsets, arm A loss, emoticon, step N.
  - `lift_s{N}`: least-squares split (tol 1e-6) of the arm A fine-tune rerun (Adam 3e-5, seed 0) at step N.
- Logs: `full.log`, `smoke.log`.

## kv_task
- `windows_{rand1,rand32,ident32}.pt`: `{"train": Windows, "heldout": Windows, "task": KV task, "table": (V,) int64}`. Written by run.py `build`. Load with `regating.load_kv_windows(name, kv_task/results)`.
- `{task}/dW_ft_step{N}.pt`: dW dict at the scored step (step 10 for all three tasks). Recipe: arm A with the table as target, Adam 3e-5, seed 0. Written by run.py `run`.
- `{task}/pullback/state_step{N}.pt`: CT state, N in {10,30,100,300}. Recipe: Adam 1e-5, 300 steps, seed 0. Written by run.py `run`, and re-read by amend1.py and sgd_kv.
- Smoke leftovers: `windows_{rand2,ident2}.pt`, `{rand2,ident2}/…`, `rand1/dW_ft_step2.pt` and `rand1/pullback/state_step{1,2}.pt` (2-step smoke written into the same rand1 dir).
- No logs.

## ladder_second_seed
- Empty. run.py writes offsets files `{T}/{arm}_seed{S}_step{N}.pt` here, but none are present. teacher_training's C3 expects `r0_t352_v310/adam_1e-05_seed1_step1000.pt` from this folder, and it is missing.

## lift_test
All dW files are written by run.py and use the emoticon windows.
- `A_rerun/dW_step{N}.pt`: dW dict, N in {1,2,3,5,10,30}. Recipe: arm A, Adam 3e-5, seed 0. Written by run.py `pullback`.
- `LoRA_r{1,8}/dW_step{N}.pt`: dW dict, with zeros except on h.2.mlp.down_proj; N in {1,2,3,5,10,30,100}. Recipe: LoRA Adam 1e-3, seed 0. Written by run.py `lora`.
- `SGD_lr{0.1,0.3,1}/dW_step{N}.pt`: dW dict, N in {1,2,3,5,10,30}. Recipe: arm A with plain SGD. Written by run.py `sgd`.
- No logs.

## lora_commute
- `lora_r{R}_s{S}_factors.pt`: `{step: {"h.2.mlp.down_proj": (B, A)}}` at steps {10,30,100}, R in {1,4,8,32}, plus R8 with seed 1. Recipe: LoRA Adam 1e-3, arm A loss, emoticon. Load with `finetune.saved_lora_updates(p, keys, W0, steps)`, which gives `{step: dW dict}`. Written by run.py `all`.
- `pb_{r{R},full}_state_step30.pt`: CT state on h.2.mlp.down_proj only, at step 30. Offsets are LoRA-shaped with rank R, or free for `full`. lr: the smallest of {1e-5..1e-3} passing the rule (the selected lr is in lora_commute.json).
- Logs: `full.log`, `full_stdout.log`, `smoke.log`. recover.py parses `full.log`.

## lora_matched
- `lora_s{0,1}_factors.pt`: `{step: {matrix: (B, A)}}` for all 24 matrices at steps {1,2,3,5,10,30,100}. Recipe: rank 8, Adam 1e-3, arm A loss, emoticon, seed S. Load with `saved_lora_updates`. Written by run.py `all`.
- Log: `full.log`.

## lora_plateau
- `lora_factors.pt`: the same format as lora_matched, at steps {1,2,3,5,10,30,100,300,1000}. Recipe: rank 8, Adam 1e-3, seed 0, 1000 steps. Written by run.py `all`.
- Logs: `full.log`, `full_smoke.log`, `pod_scored.log`, `pod_smoke.log`.

## lora_pullback
- Empty, and there is no run.py (only results/predictions.json).

## lora_sgd
- Empty. run.py writes `factors_{name}.pt` and `full.log` here, but none are present.

## lora_split
- Empty. run.py writes `ft_deltas.pt` (`{step: dW dict}`), `pieces_{u}_step{N}.pt` and `full.log`, but none are present.

## many_cues
- `windows.pt`: `{"train", "heldout", "mine"}`, the K mined cues for one target. Written by run.py (`mine`).
- `{many,one,r1_t588_v452}_shifts.pt`: `{"shift": {matrix: (C,)}, "s_k": {cue: {matrix: (C,)}}}`, the shared and per-cue fixed changes of Gamma at the trigger. Written by run.py (`state`). The fine-tunes are regenerated: arm A, Adam 3e-5, 10 steps, seed 0.
- Log: `full.log`.

## new_vs_existing
- Empty. run.py writes `windows_{t}.pt` and `{t}/dW_ft_seed{S}_step{N}.pt`, but none are present.

## reading_a_coverage
This folder is "E11" in `saved_states`. All files are written by run.py.
- `{T}/dW_ft_step10_s{S}.pt`: dW dict. Recipe: arm A, Adam 3e-5, 10 steps, batch-order seed S. Tasks are emoticon and r0_t352, r1_t534, r1_t588, r2_t275, r2_t359; r2_t359 also has s1 and s2.
- `{T}/pullback_s{S}/state_step10.pt`: CT state. Recipe: Adam 1e-5, 10 steps, same seed. Load with `saved_states.ct_state(large, T, 10)`, or with `split_states` for the ls/adjoint splits of the dW.
- `emoticon/*step2*`: smoke leftovers.
- `code_as_ran/`: the scripts as they ran, with `git_state.txt`.
- Logs: `logs/{scored,smoke,uv_sync,vpd_fetch}.log`.

## recruit_test
All `_dg` files are written by run.py `all`. The states are not saved.
- `{arm}_dg.pt`: `{"dg": {matrix: (n_alive,)}, "token": int}`, the mean over the 73 held-out windows of g_state - g_W0 at the trigger. The arms are do-nothing placements (Gamma frozen):
  - `ft_s{0,1}`: arm A, Adam 3e-5, step 10.
  - `null_a`, `null_b`: the same recipe toward plain tokens; scored at step 10 or 30.
  - `lora_r8`: rank 8 on h.2.mlp.down_proj, Adam 1e-3, step 10.
  - `lora_r8_step{N}`: the same LoRA at N in {1,2,3,5,30,100}.
  - `random_dw`: Gaussian with ft_s0's norms.
  - `pullback_adam`: CT, Adam 1e-5, 30 steps.
- `natural_o_dg.pt` is absent; run.py skips it when there are fewer than 200 positions.
- Log: `full.log`.

## regating_cuefree
- Logs only: `full.log`, `amend1.log`. No tensors.

## regating_down
- Empty. run.py writes only `full.log`, and it is not present.

## regating_leak
- `{T}_s{S}_shift.pt`: shift file, T in {emoticon, r1_t534_v369, r2_t359_v816}, S in {0,1}. `shift_null` is the ' http' (3944) fine-tune's change. Written by run.py (main stage). Seed 0's dW is reading_a_coverage's `dW_ft_step10_s0`; seed 1's is regenerated (arm A, Adam 3e-5, 10 steps) and not saved.
- Log: `full.log`.

## regating_small
- Empty. run.py writes `{name}_{label}_shift.pt` and `full.log`, but none are present.

## regating_spread
- Logs only: `full.log`, `amend{1-7}.log`, `run_amend7.log`. No tensors.

## regating_tasks
- Empty. run.py writes `{name}_shift.pt` and `full.log`, but none are present.

## sgd_faith
- `adam/state_step30.pt`: CT state, Adam 1e-5, 30 steps, emoticon, seed 0. Written by run.py; this is E4's adam_all regenerated.
- `adam_smoke/state_step2.pt`: smoke.
- The SGD state it reads is sgd_grid's `state_lr0.01.pt`, passed with `--sgd-state`.
- No logs.

## sgd_finetune
- `sgd_ft_lr0.01_step10.pt`: dW dict in fp16. Recipe: plain SGD at the selected lr 0.01, 10 steps, emoticon, seed 0. Written by run.py `rate`. The `split` and `adam_matched` stages regenerate their other states.
- No logs.

## sgd_grid
- `state_lr0.01.pt`: CT state. Recipe: plain SGD at the selected lr 0.01, 30 steps, emoticon, seed 0. Written by run.py `run`.
- `state_lr0.01_smoke.pt`: the same at 3 steps.
- No logs.

## sgd_kv
- `{rand1,rand32,ident32}/{sgd_{lr},adam_1e-06,adam_3.33e-06}.pt`: offsets file; meta keys are `task`, `arm` and `step`, where `step` is the scored checkpoint (first one with P >= 0.5).
  - SGD lr per task: rand1 0.03, rand32 0.01, ident32 0.003.
  - Recipe: kv_task's windows, 300 steps, checkpoints 10/30/100/300, seed 0.
  - Written by run.py `--shard N`. Rebuild with `offsets_state`.
- No logs.

## sgd_ladder
- `{T}/{sgd_{lr},adam_1e-06,adam_3e-06}_step10.pt`: offsets file (`task`, `arm`, `steps`), saved by run.py's own `save_offsets`, which uses the same format.
  - SGD lr per task: emoticon 0.01; r0, r1_t534, r1_t588 and r2_t359 0.03; r2_t275 0.1.
  - Recipe: 10 steps, batch-order seed 0, ladder windows.
- No logs.

## sgd_ladder_long
- `{T}/{sgd_{lr},adam_1e-05,adam_3e-06}_step{10,100,1000}.pt`: offsets file (`task`, `arm`, `optimizer`, `lr`, `step`). T in {emoticon, r0_t352, r1_t534, r2_t275}; the SGD lr is sgd_ladder's.
  - Recipe: one 1000-step run per arm, seed 0.
  - Written by run.py `--task T`.
  - The do-nothing fine-tune is not saved.
- No logs.

## sgd_matched
- `state_{opt}_m{momentum}_lr{lr}.pt`: `{matrix: (U2, V2)}` with no delta-W; delta-W is U2 V2^T - U V^T. All three runs are 30 steps on emoticon, seed 0, written by run.py `run`:
  - `adam_m0_lr2e-06`: the FULL grid's match.
  - `adam_m0_lr1.25e-06`: the amendment-1 grid's match.
  - `sgd_m0.9_lr0.0005`: SGD with momentum 0.9.
- No logs.

## shift_anatomy
- Empty. run.py writes only `full.log`, and it is not present.

## small_finetune_splits
- Empty. run.py writes only `full.log`, and it is not present.

## split_comparison
- Empty. run.py writes only `full.log`, and it is not present.

## step_split
- `per_step.json`: a list of `{"step", "carry", "fixed"}` per fine-tune step. Written by run.py.
- `per_step_smoke.json`: smoke.
- Log: `full.log`.

## step_split_small
- Empty. run.py writes only `full.log`, and it is not present.

## task_ladder
- `windows_{T}.pt`: `{"train", "heldout", "gbar", "task"}` for 6 tasks (r0_t352, r0_t588_v320, r1_t534, r1_t588, r2_t275, r2_t359). Written by run.py `mine`. Load with `regating.load_windows(T, transport_test/results, task_ladder/results/large)`.
- `emoticon/dW_ft_step10.pt`: dW dict. Recipe: arm A, Adam 3e-5, seed 0.
- `emoticon/pullback/state_step{10,30}.pt`: CT state, Adam 1e-5, all alive.
- `emoticon/pullback_seed1/state_step10.pt`: the same at batch-order seed 1. Written by the `noise` stage.
- `emoticon/pullback_gated_off/state_step10.pt`: CT state over the alive subcomponents with mean g <= 0.01 at the trigger.
- `emoticon/pullback_gated_on/state_step10.pt`: CT state over g >= 0.5, all matrices.
- `emoticon/pullback_gated_on_last/state_step10.pt`: g >= 0.5 on h.3.mlp.down_proj only.
- Missing: the ladder tasks' own `{T}/dW_ft*` and `pullback*` files are absent; reading_a_coverage regenerated them.
- `code_as_ran/` holds `run.py` and `local_faith.py` as they ran on 2026-10-19.
- Logs: `logs/{noise,pgd_check,smoke,smoke_pgd}.log`.

## teacher_spread
- `copy_{lower,plain}_seed{S}_step1000.pt`: offsets file (`text`, `seed`, `step`). Recipe: teacher training on rows (`row_copy`), Adam 1e-5, batch 8, 1000 steps, batch-order seed S. Written by run.py.
  The teacher is not saved. It is regenerated in-job: E36 A1's lowercase fine-tune of all 24 matrices, Adam 3e-5, batch 8 x 512 tokens, 300 steps, seed 0.
- `amend1_arm1_step{1000,4000}.pt` and `amend1_arm3_step2000.pt`: offsets file (`arm`, `step`). Written by amend1.py.
  - arm 1: E37's recipe run to step 4000.
  - arm 3: the same, started from the scaled adjoint split.
  - arm 2's file is absent.
- Logs: `run.log`, `stdout.log`, `amend1_run.log`, `amend1_stdout.log`.

## teacher_training
- `r1_t534_v369/amend5_{A,B}_sgd_seed{S}_step1000.pt`: offsets file (`task`, `teacher`, `arm`, `seed`, `step`). Recipe: student Adam 1e-5 for 1000 steps; arm A on the task windows (batch 32), arm B on Pile train rows (batch 8). Written by amend5.py.
  The teacher is regenerated in-job and not saved: arm A SGD fine-tune, lr 0.03, 1000 steps, seed 0.
- Missing: run.py and amend2-4 also write `{T}/A_{opt}_step1000.pt`, `C3_student_step*.pt`, `B*_{opt}_step1000.pt`, `lora_*` and `lora_teacher_factors.pt`. None are present.
- Log: `run_amend5.log`.

## transport_test
This is "round 1", the input of nearly every other run.
- `alive.pt`: `{"mean_g": {matrix: (C,)}, "alive": {matrix: (C,) bool}}`. Written by run.py `calibrate`. Load with `saved_states.load_alive` / `load_mean_g`.
- `windows.pt`: the emoticon windows, `{"train": Windows, "heldout": Windows, "threshold"}`. Written by run.py `windows`.
- `{A,B}/dW_step{10,30,100,300}.pt`: dW dict. Written by run.py `finetune A|B`.
  - A: arm A, Adam 3e-5, on the emoticon windows.
  - B: arm B, Adam 1e-5, batch 16, on Pile train rows.
- `{A,B}/alpha_T0_step{N}.pt`: `{matrix: alpha}`, the T0 scale fit.
- `{A,B}/S10_step{N}.pt`: `{"S": top-10 set}`.
- `{A,B}/T1write_k10_s{N}.pt`: `{"S", "dU": write edits}`.
- All `{A,B}` files are written by run.py `transport`.
- Logs: `logs/*.log`, plus the JSONs `logs/{calibration_per*row_adversary,pgd_check,pgd_diagnostic*,windows_*rows}.json`.
