# What is saved in each results/large

Every `experiments/<name>/results/large` is a symlink to `~/commute-data/results-large/<name>/` (cluster NFS; 79 folders). Contents as of 2026-11-03 (2026-10-27 for the folders it already listed that did not change), read from `ls`, the scripts' `torch.save`/`save_file` sites and, for safetensors, the file header; nothing was loaded.
Patterns: `{S}` seed, `{N}` step, `{R}` LoRA rank, `{T}` task (`emoticon` or a ladder task `r{rung}_t{trigger}_v{target}`), `{lr}` learning rate as written in the name. Every `.pt` loads with `torch.load(p, weights_only=False)`. "Unknown" means the scripts did not say. Qwen files are safetensors (`safetensors.torch.load_file`). **Read the dtype from the header before loading**: the first 8 bytes are the header length (little-endian u64), then that many bytes of JSON, one `{"dtype", "shape", "data_offsets"}` per tensor (`python3 -I -c` with `struct.unpack('<Q', f.read(8))`, no torch). A Qwen dW dict of 196 tensors is 2.82 GB in BF16 and 5.64 GB in F32; the 67M's is 57 MB / 113 MB; offsets files are bf16 (60 MB).
**Labels.** A column or phrase `Label:` says which side a 67M state is on (reuses / new / open) and how it got that label. All 67M labels are by construction (the task was built to reuse or to teach something new) or by behaviour at unseen keys (E58); only E62's copy map has a causal check (switch off W0's copying head), and its copy side is not settled. See `reports/analyses/2026-10-30_analysis_e60-regating-table.md` for the 18-state table and `reports/analyses/2026-11-02_analysis_method-table.md` for the truth kinds.
**Update rule.** A run that writes `results/large` files adds or edits its folder's section here in the same commit as its results: format, dtype, size, how to load, and what is absent.
Common terms: **arm A** = Adam, batch 32, clip 1, kl_lambda 1, on the task's trigger windows (`finetune.ArmAConfig`). **dW dict** = `{matrix: delta-W}` for all 24 matrices, fp32, CPU (113 MB). **CT state** = `{matrix: (delta-W, U2, V2)}` in theory layout, from `component_training.train` (234 MB). **offsets file** = `{"offsets": {matrix: (dU, dV) bf16}, "layout", **meta}` from `arms.save_offsets`; rebuild it with `saved_states.offsets_state(p, alive_uv)`. **sources** = `{matrix: (C+1,) levels in [0,1]}`, the last entry being Delta's, from `gamma.pgd_sources`. **shift file** = `{"shift": {matrix: (C,)}, "shift_null": ...}`; load it with `turned_down.load_shift(p)`.
Emoticon windows and alive masks always come from `transport_test/results/large/` (see the transport_test section).

## Network volume (RunPod `/workspace`)
One row per directory a run wrote there, added by the lead from the pod-runner's report, with the run, the size and
what it holds. A path a plan or brief names that no run created gets a row saying so: `/workspace/e40_nowarmup` was
named only in E40 Amendment 1, which never ran, and a plan writer had to dig through `run.log` to find that (friction 68).
Volume `ydavm7ct4x` (US-MO-1), listed through the S3 API 2026-11-02 (95.9 GB). `2bun37yawb` (US-NC-1) was deleted by the user 2026-11-02, unlisted.
- `/workspace/qwen3-1.7b-p-e9fcffd5-torch/` (53.65 GB): the decomposition; also `$COMMUTE_QWEN_DIR` on the cluster.
- `/workspace/e40_nowarmup/` (42.3 GB): `dW/step_{1,2,3,5,10,20,50,100}.safetensors` (2.82 GB each), `dW/step_100_fp32.safetensors`, `adam_step100.safetensors` (11.27 GB), `grad_W0.safetensors` (2.82 GB). Steps 1, 10, 100 and fp32 have regenerated copies under qwen_maps below. The rest (`dW/step_{2,3,5,20,50}`, `adam_step100`, `grad_W0`, 28.2 GB) was copied 2026-11-02 (job 48394, byte counts and safetensors headers checked) to `~/commute-data/results-large/qwen_sft/nowarmup/`, checksums in `SHA256SUMS_volume_only` there; not on W&B.

## qwen_maps (E44 input, regenerated on the cluster 2026-10-29, job 42866)
- `~/commute-data/results-large/qwen_sft/nowarmup_regen/dW/` (14.1 GB, never committed): `step_{1,10,100}.safetensors` (2.82 GB each),
  `step_100_fp32.safetensors` (5.64 GB), `SHA256SUMS` (copy committed as `experiments/qwen_maps/results/regen_SHA256SUMS`).
  Written by `experiments/qwen_maps/regen.py`; check record `experiments/qwen_maps/results/regen.json`.
- W&B artifact `edwinagnew-mats/commute/qwen3-1.7b-e40-nowarmup-dW` (version in `RUN_BRIEF.md` section 4): the same four files, for pods.

## adjoint_sgd
- Logs only: `full_{A..E}.log` (shards), `full_amend3_r{0,1}_*.log`, `full_amend3_emo_*.log`. No script here saves tensors, and every state is regenerated in-job (see the run.py docstring: arm A Adam 3e-5 / 3e-6 and SGD 0.01, 1000 steps, seeds 0/1).

## broad_fragility (E39 and Amendments 1-3; listed 2026-10-29 for E49)
- Teacher-trained decompositions of E37's lowercase fine-tunes (offsets files, E37's teacher training: Adam 1e-5, batch 8, 1,000 steps, batch order 0): `trained_step1000.pt` (the Adam fine-tune, E39; sha256 `d7da3c25…`), `tt_sgd_step1000.pt` (plain SGD 0.01, A1; `ced08dea…`), `a2_tt_sgd_r8.pt` (SGD held to rank 8, 0.03, A2; `4c9d24df…`), `a2_tt_lora_s0.pt` / `a2_tt_lora_s1.pt` (rank-8 LoRA, LoRA seeds 0 and 1, A2; s0 `20b3d068…`). Also `a2_label_{adam,sgd}.pt` (component training with labels, A2). `results/tt_lora_step1000.pt` (E39's run.py) is 1 MB against 60 MB for the others, probably a partial download (not loaded to check); use `a2_tt_lora_s0.pt`. The fine-tunes themselves were never saved here; see row_blocks.
- Logs: `run.log`, `amend{1,2,3}.log`, `amend{1,2,3}_stdout.log`.

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

## claim_tests (E47; listed 2026-11-03)
- `t2_lora_r{32,128}_tt.pt` (60 MB each): offsets files (`rank`, `lr`, `order`, `step`=1000, `stage`) of a rank-32 / rank-128 LoRA teacher-trained by E37's recipe; run.py's `high_rank` loads a file already there. Written by `stages.py` (`save_offsets`).
- Logs only otherwise: `t1.log`, `t1b.log`, `t2{,b,c}.log`, `t3.log`, `t4{,b,c}.log`, `t5{,b}.log`.

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

## induction_cue (E62 and Amendments 2-6; 2026-11-02)
- `{copy,lookup}_s{0,1,2}.pt` (113 MB): dW dict (fp32) of the copying map and the lookup at the new marker, one batch order per seed, saved at the read step the run chose (E12's checkpoints). Written by run.py. Label: the copy map is the causal-check pair (switch off W0's copying head, E62), copy side unsettled; r unscored.
- `windows.pt`: `{"train", "read", "target", "p_read", "pools"}`. `heads.pt`: `{"scores", "means", "H"}` (prefix-matching score of the 24 heads, their mean activations, the head set).
- `amend5_vectors.pt` (`{"a", "a_half2", "S_old"}`) and `amend5_windows.pt` / `amend6_windows.pt` (rows lists, torch.save of `self.rows`). Amendment 4's `amend4_vectors.pt` was written to the same place by code but is not present.
- Amendments 2, 3 read these states and train nothing; logs `run.out`, `amend{2,3,4,5,6}.out`.

## interchange_67m, method_table, neuron_sparsity, baseline_neurons, baseline_weights, pass_reread, pcfg_anchors, qwen_adversary, qwen_maps, qwen_pass_test
- Empty (checked 2026-11-03). `pcfg_anchors` run.py and c1b.py write counting-model checkpoints `{"model": state_dict, "step"}` (fp32, 12 MB each) to results/large, none present; `pass_reread` run.py writes bf16 offsets to `regen/`, none present. `qwen_maps`' states are under qwen_sft (see qwen_maps above).

## jain_readout (E59; 2026-11-02)
- `readouts.pt` and `readouts_48279.pt` (619 MB each): `probes.state_dict()`, the per-block linear readouts trained on Pile text (Jain et al. 2024 Def 1); `readouts_48279.pt` is the one job 48279 wrote before its stage-0 stop. Written by run.py (`stage0_readouts`); pass it as `saved` to reuse. It reads the twelve E54/E56 states.

## keydependent_elicit (E54; 2026-10-29)
- `{elicit,cipher}_s{0,1}.pt` and `cipher_own_s{0,1}.pt` (113 MB): dW dict (fp32) at E12's recipe (arm A, Adam 3e-5, batch 32; scored step stored in `results/stage2.json`, with sha256). `cipher_own` is the cipher at its own norm; when it is the same state as `cipher_s*` it is not written twice. `windows_{elicit,cipher}.pt` (0.6 MB): `{"train","heldout","task","table"}`. Written by run.py. Label: elicit is "open", the cipher "new", both by construction only.

## keydependent_patch (E56)
- `{elicit,cipher}_s{2,3,4}.pt` (113 MB): dW dicts (fp32), the three further seeds beside E54's two; run.py `--job 2`. Same recipe and label.

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

## new_vs_existing (E12 and Amendments 1-3)
- Top level is empty: run.py's `windows_{t}.pt` and `{t}/dW_ft_seed*` were never kept; the E12 do-nothing fine-tunes and windows are regating_table's `{ident32,ident32b,rand32,rand32b}_s{0,1}.pt` and `windows_*.pt`.
- `amend1/{ident32,ident32b,rand32,rand32b}_s{0,1}.pt` (60 MB): offsets files of Amendment 1's students, teacher-trained on those regating_table fine-tunes (meta `step`, `order`, `arm`); `*_o1.pt` (ident32_s0, rand32_s0): the same at batch order 1. `c1.pt`: the control student whose teacher is W0. Written by `amend1.py` (`arms.save_offsets`, bf16 offsets); read back with `saved_states.offsets_state`. States are read from trained fp32 weights, never from the bf16 offsets as such (header of `amend1.py`).
- `amend2/{rand32,rand32b}_s{0,1}.pt` (60 MB), `rand32b_s{0,1}_o1.pt`: the same for Amendment 2's random-map students; `trigger_{name}.pt` (15 MB): `{"student": {matrix: gates}, "fine_tune": {...}}`, fp16 gates at the trigger over the held-out half of the windows. Written by `amend2.py`.
- `amend3/run_amend3.log` only: Amendment 3 (the fp32 read) trains and saves nothing.
- Labels: copy tasks (ident32*) reuse, random maps (rand32*) new, both by construction.

## open_pair (E58 and Amendments 1-7; 2026-10-30/11-02)
- `{elicit,cipher}_s{0,1,2}.pt` (113 MB): dW dicts (fp32) of E58's 24-key fine-tunes, scored step in `results/open_pair.json` with sha256. `windows_{elicit,cipher}_{trained,new}.pt`: the trained keys and the unseen keys' windows.
- `amend1/{copy,rank1}_s{0,1,2}.pt`, `amend2/elicit_s0.pt` (96 keys), `amend3/{copy,elicit,rank1}_s0_step{10,30}.pt`: dW dicts (fp32, 113 MB). `windows_{map}[_trained|_new].pt` beside each (1 MB): `{"train","heldout","task","table", ...}`, 96-key windows in amend2/3.
- `amend6/` and `amend7/`: windows files only (`first_letter`, `prior_cipher`; `class_map`, `class_shuffled`, with `_const{0-3}_new`); no weight changes kept. `amend{4,5,6,7}.out`.
- Load a dW dict with `torch.load(p, weights_only=False)`. Labels: the elicit map is "open" (behaviour at the unseen keys, E58's held-out carry), the others by construction; the method table (2026-11-02) marks these rows "only by assumption".

## open_pair_instruments (E55 C3 ladder, E58 seeds 3-7; 2026-10-31)
- `{elicit,cipher,rank1,copy}_s{3..7}.pt` (113 MB): dW dicts (fp32) of the four open-pair maps at seeds 3-7, E58's recipe, written by `seeds.py` with sha256 in `results/seeds.json`. Seeds 0-2 are open_pair's.

## performed_anchor (E53 and Amendments 1, 1b, 2; 2026-10-29/30)
- `dW_{lowercase,cipher,broad_cipher}_s{0,1}_step{N}.pt` (57 MB): dW dict, bf16 (`{k: v.bfloat16()}`; the run's own check reads the bf16 file against the fp32 gain). lowercase at step 300 only; cipher and broad_cipher at steps 10, 100, 300. Arm B, Adam, Pile text (`lowercase.arm_b_states`); written by run.py `finetune_all`. Load with `torch.load(p, weights_only=True)`, then `.float()`.
- Labels: lowercase reuses, cipher and broad cipher new, by construction (E60's table).
- Logs: `finetune`, `a1_finetune`, `a1b_finetune`, `a1b_read`, `a2_read`, `read` (`.log` and `.out`).

## qwen_port (Qwen3-1.7B port check; 2026-10-27)
- JAX reference outputs on fixed tokens, written by `jax_reference.py` (all safetensors, `safetensors.torch.load_file`): `gamma_row{0,4}.safetensors` (1.41 / 0.32 GB, BF16, 392 tensors: g and p of every site on the two detail rows), `target_row{0..7}.safetensors` (final logits F32 and hidden state BF16), `taps_row{0..7}.safetensors` (29 residual boundaries, BF16), `masked_row{0..7}.safetensors` (masked-forward KLs F32, logits BF16 on detail rows), `source.safetensors` (the fixed source array, F32, 392 tensors). Read by `torch_check.py` and `plot.py`. 4.4 GB.

## qwen_reading (E57; 2026-10-31)
- `A/`: `W0_maths.safetensors` (129 MB; I64, F32, I16 tensors), `layout`, `plain`, `prompt`; `B1/think.jsonl`; `B2/{C,L,T}.safetensors`; `C/{state}_g{0,1}.safetensors` (14 and 130 MB), state in `a2long_s{0,1}`, `e40`, `gauss`, `gold_s0`, `inspan`. Reads of the frozen decomposition; written by run.py (`save`), read by `score.py`. No weight changes live here: the states read are qwen_updates' and qwen_sft's.

## qwen_recipe (E61: Stage A, five checks, gamma_parts, Job S; 2026-11-02)
Qwen3-1.7B. `dW` is `{layers.{i}.{mlp.down_proj,mlp.gate_proj,mlp.up_proj,self_attn.{q,k,v,o}_proj}: tensor}`, 196 tensors, safetensors.
- `train/{run}/step_{t}.safetensors` (2.82 GB, dtype BF16 in the header): δW of E61 Job R's fp32-forward fine-tunes, saved by `train.py` (`save(..., BF16)`). Runs: `arm0` (own answers at 1e-5, 10 steps, steps 1-10; rehearses resume, `arm0_job48876` is the same after a resubmit, steps 1-4), `a1_1e-6` and `a1_3e-6` (steps 10, 20, 50, 100), `a2_6e-6` (100, 150, 200, 300), `gold_1e-5` (1-3, 5, 7-13, 15, 20, 30, 50, 100). `train/answers/train_{W,run}/` holds the generations (jsonl). 111 GB.
- `checks/{a2long_s0,e40,gauss,gold_s0}_X.safetensors` (2.82 GB, **BF16**; header checked 2026-11-03): the X of E57's split from the first k10 pass, written before the fp32 rule. `checks.split_x` **refuses them** (`refusing: ... is not fp32`: X V and Xᵀ U amplify rounding), and also exits on any existing file of the wrong dtype instead of rebuilding it: move the file aside and the next run rebuilds it in fp32 (about 700 s budget). Do not feed them to anything that needs the split's X.
- `checks/size_{gold_1e-5_{8,10,20,50,100},gold_s0,gauss,gauss_big_b1.0556,inspan,inspan_big_b1.3119}_b{scale}_X.safetensors` (5.64 GB, **F32**): the same X in fp32, written by Job S (`size.py`, through `split_x`); these are the ones `split_x` accepts. 64 GB for the checks folder.
- `checks/gen_{W0,own_s0}.{jsonl,batch}`: generations of the fp32-vs-bf16 check.

## qwen_rl (E69; rehearsal only)
- `tiny/dW/{R,N,F}/step_{1,3}.safetensors` (5.64 GB each, **F32**): the `--size tiny` rehearsal of the GRPO loop (R = RL, N = shuffled-reward null, F = own-samples SFT); `tiny/rollouts/{R,N,F}/step_*.jsonl` and `.tar`, `tiny/acc`, `tiny/wandb`. 32 GB. No full-size run has written states here yet. Written by run.py.

## qwen_sft (E40; updated 2026-11-03)
- Network-volume copy and regenerated steps are listed above (Network volume, qwen_maps). Here also: `nowarmup/` holds the 28.2 GB copied from the volume (`dW/step_{2,3,5,20,50}.safetensors` BF16 2.82 GB, `adam_step100.safetensors` F32 11.27 GB with meta `step`, `grad_W0.safetensors` 2.82 GB, `SHA256SUMS_volume_only`), `gstats/W0_{maths,fineweb}.safetensors` (F16, 392 tensors, `{site}.mean_g` and `{site}.active` per site, from `qwen_teacher.g_stats`), `sources/{B,D1,D10,G,T0,...}_{fineweb,maths}_seed{S}.safetensors` (F32, 392 tensors, a few KB: the adversary's source arrays for a state `{B,D1,D10,G,T0,...}` and corpus, from `qwen_teacher.adversary`), `step{0,100}_{test,gsm1k}.jsonl`.
- `amend1_rehearse*`: rehearsal outputs of E40 Amendment 1 (gstats, sources, `T100_test.jsonl`), no weight changes.

## qwen_updates (E45; Amendments 1, 2)
Qwen3-1.7B, SFT with five updates: `own`/`own2` (own samples), `gold`, `null` (self-distillation), `a1` (anchor 1: context distillation), `a2` (anchor 2: invented facts), `a2long` (300 steps, Amendment 1); `_s{S}` batch-order seed.
- `dW/{run}/step_{t}.safetensors` (2.82 GB, BF16): `a1_s0` (2-6, 8, 9, 19, 20, 100), `a1_s1` (2-8, 19, 20, 100), `a2_s0` (1-6, 11, 12, 100), `a2long_s0` and `a2long_s1` (1-6, 11, 12, 100, 300), `gold_s0` (2-5, 7-9, 100), `null_s0` (2-4, 7, 8, 100), `own_s0` (2-4, 100), `own2_s0` (1-3, 6, 7, 100). Steps are the KL grid's brackets plus 100 and the last. 192 GB. Written by run.py `T`; read by qwen_reading (`lab.py`) and E44's table.
- `E/{run}/step_{t}_test.jsonl`, `S/{a1,own}/batch_*.jsonl` (samples), `P/` (anchor 1's prompts), `check_format/`, `W0_test.jsonl`.

## reading_a_coverage
This folder is "E11" in `saved_states`. All files are written by run.py.
- `{T}/dW_ft_step10_s{S}.pt`: dW dict. Recipe: arm A, Adam 3e-5, 10 steps, batch-order seed S. Tasks are emoticon and r0_t352, r1_t534, r1_t588, r2_t275, r2_t359; r2_t359 also has s1 and s2.
- `{T}/pullback_s{S}/state_step10.pt`: CT state. Recipe: Adam 1e-5, 10 steps, same seed. Load with `saved_states.ct_state(large, T, 10)`, or with `split_states` for the ls/adjoint splits of the dW.
- `emoticon/*step2*`: smoke leftovers.
- `code_as_ran/`: the scripts as they ran, with `git_state.txt`.
- Logs: `logs/{scored,smoke,uv_sync,vpd_fetch}.log`.

## readout64 (E64; 2026-11-02)
- `readouts.pt` and `readouts_low_cap.pt` (0.8 MB each): `probes.state_dict()` for the 64-value readout over the pair, at the filed capacity and the low one. Written by run.py.

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

## recruitment_test (E38 and Amendments 2-7; listed 2026-11-03)
- `states/{emoticon,r1_t534_v369,r2_t359_v816}_{s0,s1,null}.pt` (113 MB): dW dict (fp32): arm A fine-tune at seed S, `null` the plain-token fine-tune (`G.finetune`, Adam, `CFG.lr`, `CFG.steps`). Written by run.py `make_state` (stage 0, Amendment 2's cached states). `amend5_scores.pt` (2.9 MB): per-state piece scores of Amendment 5; `stage3_pieces.json`; logs `full.log`, `amend{2..7}.log`.

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

## regating_table (E46, stage 0, Amendments 1 and 3; 2026-10-29)
Written by run.py `--stages 0a`; every file is checked by the sha256 in `results/stage0a.json` before 0b reads it.
- `{ident32,ident32b,rand32,rand32b}_s{0,1}.pt` (113 MB): dW dict (fp32) of E12's arm-A fine-tune (Adam 3e-5, batch 32) on the task's key-to-value table, stopped at E12's scored step (10 for all but rand32b, 30); regenerated by `anchors.kv_finetune`. Load with `torch.load(p, weights_only=False)`. These are the teachers of E12 Amendments 1-3 (new_vs_existing).
- `windows_{ident32,ident32b,rand32,rand32b}.pt` (0.6 MB): `{"train","heldout","task","table"}`, E12's windows rebuilt (marker 18214).
- `code_step{10,100}.pt` (113 MB): dW dict (fp32) of E36's code fine-tune (arm B, Adam 3e-5, batch 8, seed 0; `anchors.code_finetune`). `code_held.pt` (1.5 MB): E36's held-out rows (a tensor).
- `full.log`. Amendments 1 and 3 (`amend1.py`, `amend3.py`) save nothing here.
- Labels: ident32* reuse, rand32* new (construction); the code fine-tune is E36's.

## regating_tasks
- Empty. run.py writes `{name}_shift.pt` and `full.log`, but none are present.

## regating_truth_table (E60) and restore_anchors (E50 and Amendment 1)
- Empty, and both stay so: E60 only reads the 18 saved states (regating_table's E46 files, performed_anchor's E53 files, keydependent_elicit's E54 files) and writes JSON; E50's restore reads regating_table's `ident32*`/`rand32*` states (`E46 / "large"`) and keeps nothing.

## row_blocks (E49, written 2026-10-29, Slurm 42959)
- `dW_{adam,sgd,lora}.pt`: dW dicts (fp32) of the lowercase fine-tunes (`lowercase.finetune`, 300 steps of 8 lowercase rows), regenerated on the cluster L40 by `run.py regen`; sha256 in `results/regen.json` and the regen log. Load with `torch.load(p, weights_only=True)`; a do-nothing state is `harness.at(ctx, dW, None)`.
  - `dW_adam.pt` (113,253,467 B; sha256 `2b1669e1…`): E37's Adam fine-tune, rate 3e-5; held-out gain 0.12963 (H100 0.1301).
  - `dW_sgd.pt` (113,252,477 B; `2b2cc903…`): A1's plain SGD fine-tune, rate 0.01; gain 0.13705 (H100 0.13671).
  - `dW_lora.pt` (113,253,467 B; `067152d9…`): E39's rank-8 LoRA, Adam 1e-4, saved as its δW; gain 0.14774 (H100 0.1479).
- Pile val rows 512-1023 for these reads are cached at `~/commute-data/artifacts/sets/val_{512,768}_256.npy` (regen job).
- Logs: `{regen,a,b512s0..b768s2,score}.log`.

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
- `amend6/{emoticon,r0_t352_v310,r1_t534_v369}/amend6_lora_B_seed{0,1}_step{5000,2500}.pt` and `amend6/r2_t275_v2667/amend6_sgd_{A,B}_seed0_step1000.pt`: offsets file (`task`, `teacher`, `arm`, `seed`, `step`). Recipe (E34 Amendment 6): student Adam 1e-5 on arm A task windows (batch 32) or arm B Pile train rows (batch 8); the LoRA files are arm B on the task's rank-8 LoRA teacher, seed 0 saved at step 5000, seed 1 at step 2500; the r2 files are arms A and B at step 1000 on the SGD 0.1 teacher. Written by amend6.py (`Job.student`, save=True). Logged in `amend6/run_amend6.log`.
  The teachers are regenerated in-job and not saved (full-weight arm-A fine-tune, seed 0, step 1000; E9's rank-8 LoRA via amend3.lora_teacher).
  sha256 prefixes: emoticon seed0 d0fe5062, seed1 c79795b6; r0_t352_v310 seed0 ad0f6f49, seed1 bcfce39f; r1_t534_v369 seed0 7293135c, seed1 75010104; r2 A 14bf6eba, B f1b7ac4f.
- `e42/{rand32,ident32,rand1}/{A,B}_step1000.pt`: offsets file (`task`, `teacher`, `arm`, `step`). Recipe (E42): student Adam 1e-5 (E34's CFG) for 1000 steps; arm A on the task windows (batch 32), arm B on Pile train rows (batch 8); teacher = E10's key-to-value cipher fine-tune run to step 1000 (all 24 matrices, Adam 3e-5, batch 32, seed 0), regenerated in-job and not saved. Written by newskill.py (`student_arm`).
  sha256 prefixes (A / B): ident32 2dfb9674 / 440bda47; rand1 b5996e93 / b417129a; rand32 cd981d21 / 34b7bb60.
- `e42_amend2/rand32/M0_{ci,st,trig}_step1000.pt`: offsets file (`task`, `state`, `ci_coeff`, `stoch_coeff`, `trigger_only`, `step`). Recipe (E42 Amendment 2): teacher training with the teacher at W0 (nothing to copy) and a masked-term student: ci = CI-masked weight 0.5, stochastic 0; st = CI-masked 0, stochastic 0.5; trig = both 0.5 read at the trigger alone; 1000 steps, rest as E42 Amendment 1's M0. Written by newskill_amend2.py (`save`).
  sha256 prefixes: ci ec5d38b1, st bbd3dbc6, trig 5d6f8624.
- `amend8/{n1,n2}/{0,100,1000}.pt`, `amend8/n3b/{100,1000,real}.pt`, `amend8/n4_{adam,sgd}/{100,1000,true}.pt`, `amend8/n5/{100,1000}.pt`, `amend8/moved/{100,1000}.pt` (60 MB): offsets files (`stage`, `read`) of E34 Amendment 8 (base control) and 8b: the student at that step of the stage (n1 random start, n2 E41's T1000 start, n3b the null fine-tune's student and the same-job real one, n4 self-labelled training, n5 masked copy at coefficient 0); `moved` is Amendment 8b's. `amend8/n3b/null_teacher_dW.pt` (57 MB): dW dict in bf16 of the null fine-tune. Written by amend8.py (`finish`); no n3a files (it saves nothing).
- `amend8/amend8c/{null,real}_o{0,1}.pt` (60 MB): offsets files (`read`, `order`, `step`) of Amendment 8c; `{null,real}_teacher_dW.pt` (57 MB, dW dict bf16); `uv0.pt` (121 MB): the alive `{matrix: (U, V)}` at W0 (`tctx.uv_cpu`). Written by amend8c.py. Logs `amend8/run_*.log`.
- `amend10/{rand32,rand32b}_s{2,3}_dW.pt`, `rand32_long_s{0,1}_dW.pt` (113 MB, fp32 dW dicts: reruns of E12's rand32 fine-tunes at further batch orders, one long) and `real_teacher_o{1,2}_dW.pt` (57 MB, bf16): E34 Amendment 10. `gram_pile.pt`, `gram_task_{emoticon,rand32,rand32b}.pt`, `gram_trigger_{...}.pt` (1 MB each): `{"names", "n", **grams}`, the input Gram matrices it weights by. Written by amend10.py. Amendment 9 saves nothing (reads these). Log `run_amend10.log`.
- Missing: run.py and amend2-4 also write `{T}/A_{opt}_step1000.pt`, `C3_student_step*.pt`, `B*_{opt}_step1000.pt`, `lora_*` and `lora_teacher_factors.pt`. None are present.
- Logs: `run_amend5.log`, `run_amend9.log`, `run_e41.log`, `run_e41_amend{1,2}.log`, `run_e43.log`.

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

## weight_side_read (E67)
- `run.out` only.
