# P3.5 run 1: facts for the Qwen pair-library plan (collected 2026-11-11)

Repo root `R` = /mnt/nw/home/e.agnew/commute-agents/qwen-lead. Paths are relative to R unless absolute. Nothing was run.

## 1. Qwen supervised fine-tune code

**Module roles**
- `src/commute/qwen_sft.py` (352 lines): the step, loss, optimiser, save and KL helpers. Module docstring (lines 1-12):
  only the 196 decomposed matrices (`qwen_vpd.SITES`) train. The embedding, tied head and norm gains are frozen. fp32
  master weights, gradient checkpointing, fused AdamW. A sequence is chat prompt (thinking off) + response + end token,
  cut at 512. The loss is the mean over the batch's response tokens.
  - `trainable(model, W0)` :179. `adamw(params, lr)` :192: betas (0.9, 0.999), no weight decay, fused.
  - `lr_at(t, peak, warmup, steps, schedule)` :197: "constant" or "hf_cosine". Step 1 runs at rate 0, so the step-1 file is all zeros.
    Docstring: "warm-up 10, cosine over 100, peak 1e-5 is LLaMA-Factory's documented full-SFT setting for Qwen".
    Fable's advice (`reports/plans/2026-11-02_advice_e61-recipe-FABLE.md:21`) calls 1e-5 "the cited LLaMA-Factory value
    (no copy in the repo; treat as unsourced for 100 steps)".
  - `sft_step(model, seqs, micro, device, forward="bf16_autocast")` :271. `gold_loss(model, seqs, batch, device)` :286
    gives the mean response-token loss in fp32.
  - `mean_kl(model, rows, p, q)` :293: the mean over every position of KL(model at p ‖ model at q).
  - `save(path, tensors, dtype, droppable, log, meta)` :109: safetensors, written to .tmp and then renamed, with a disk-margin check.
    `load(path)` :129. `plus(W0, change)` :63. `set_weights(model, W)` :47 swaps weights by pointer.
  - `seq_of(tok, chat, response, end_id, seq_len)` :140. `sft_seq(tok, question, solution, end_id, seq_len)` :136.
    `pack` :149. `Uploader(prefix, max_bytes, wandb_dir)` :318 logs each saved file as a W&B artifact (needs WANDB_API_KEY).
- `src/commute/qwen_train.py` (151 lines):
  - `Recipe(lr, warmup, steps, schedule, clip, micro, forward="bf16_autocast")` :43, frozen dataclass. The batch size is
    not in it: the caller's `batch_at(t)` returns the step's sequences.
  - `delta(params, W0)` :53 gives bf16 δW on the host. `qwen_rl.delta_fp32` is the fp32 version that E78/E81/E120 pass.
  - `train(model, W0, batch_at, r, axes, reads, grid, keep, save, on_step, ckpt, pause, dW_of=delta, size_of=None)`
    :107. At each read step it reads the KL to start on each axis's tracker rows, `mean_kl(model, rows, W0, None)` =
    KL(W0 ‖ W_t). The first read at or past a grid value saves δW at that step *and at the read step before it* (a
    bracket). `size_of` replaces the KL with any size per axis (E120 passes |δW|). It pauses and checkpoints every
    `pause` steps; a checkpoint is about 17 GB of fp32 weights plus Adam (:81).
- `src/commute/pool_sft.py` (409 lines; E120, the latest full-weight fine-tune code in `src/`): `RECIPE = Recipe(1e-5, 10,
  100, "hf_cosine", 1.0, 4, forward="fp32")` :69 is E61's `gold_1e-5`, 64 sequences a step. It runs one arm per job with
  `Setup`/`Arm` dataclasses (:81, :92), saves fp32 δW at |δW| brackets, early checks (:210), and stage W0, train, files, read.
- `src/commute/qwen_data.py` (27 lines): only `fineweb_rows(tok, n, seq_len=512)`, generic text rows. There is no task data here.
- `src/commute/qwen_eval.py` (189 lines): `MODEL, MODEL_REV = "Qwen/Qwen3-1.7B", "70d244cc..."` :35. `load_model` :42
  (bf16). `prompt` :53 and `chat_prompt` :57 (thinking off). `generate`, `sample`, `score` :141, `summary` :169.
  `GREEDY` and `SAMPLE` (T 0.7, top-p 0.8, top-k 20, from the model card). MAX_NEW 512.
- Task data modules: `src/commute/qwen_tofu.py` (TOFU), `src/commute/gsm8k.py` (gold), `src/commute/qwen_facts.py`
  (invented and known facts), and `experiments/qwen_updates/job.py` `Config` :58 (seq_len 512, batch 64; the E40 split
  holds out 256 GSM8K train problems, seed 0).

**Recipes as run** (all fp32 forward, clip 1, AdamW, hf_cosine)

| task | runs | lr | warm-up | steps | batch | micro | source |
|---|---|---|---|---|---|---|---|
| answer template (a1) | E61 `a1_1e-6`, E78, E88 seed 1 | 1e-6 | 10 | 100 | 64 | 4 | E61's rule: the lowest peak that passes (`reports/analyses/2026-11-02_analysis_e61-recipe-checks.md:49`) |
| gold maths | E61 `gold_1e-5`, E120 pools | 1e-5 | 10 | 100 | 64 | 4 | LLaMA-Factory, unchecked (above) |
| TOFU | E81, E84, E88 seed 142, E91 | 2e-5 (E81 also ran 1e-5) | 250 | 1250 (5 epochs) | 16 | 8 | TOFU's config and finetune.py (`experiments/qwen_tofu/run.py:42`) |
| E107 / E103 known-facts | label_vs_style | E84 rate | 250 | 64 cap | E81 batch | 8 | `experiments/label_vs_style/run.py:58` |

**Last full-weight runs: time, memory, card**
- E120 (2026-11-07, `pool_sft`, 4 arms, jobs 54969 and 55011-55013, cluster L40, $0). `experiments/qwen_pass_pools/results/train_<arm>.json`:
  `.seconds` 1501-1810 s for 100 steps (~15-18 s/step at batch 64, with |δW| read every step); `.peaks.gpu_gib` 31.2-31.3,
  `.peaks.host_gib` 12.4; whole job 2138 s (`sft_known_s0.log` last line, peak GPU 32.0 GiB). The sbatch asks normal QoS,
  24G, 3 h (`cluster/qwen_pass_pools.sbatch`).
- E88 (2026-11-05, L40): template seed 1, `experiments/qwen_anchor_checks/results/A_train.json` `.seconds` 1341, peak GPU 33.7 GB,
  host 13.2 GB. TOFU seed 142, `S_train.json` `.seconds` 1310, peak GPU 31.4 GB.
- E78 template: `qwen_anchor1/results/train.json` `.seconds` 1629, peak GPU 33.7 GB, host 13.2 GB.
- E81 TOFU 2e-5 on L40: `qwen_tofu/results/train_2e-5.json` `.seconds` 1835, peak GPU 42.1 GB, host 13.8 GB.
- E84 TOFU on a RunPod H100: "672 s of training for about $1" (`reports/analyses/2026-11-04_analysis_e84-rephrasings.md:176`).
- E107: "About 3 min each" per 64-step arm on L40, peak GPU 41.2 GB, host 11.0 GB (`experiments/SAVED.md`, label_vs_style).
- E61 `gold_1e-5`: `qwen_recipe/results/train_gold_1e-5.json` `.seconds` 1732, peak GPU 36.7 GB, host 11.0 GB.

**How dW is saved**: safetensors with 196 tensors keyed `layers.{i}.{mlp.*,self_attn.*}`, HF layout [d_out, d_in]. fp32
is 5,637,164,512 B (5.64 GB); bf16 is 2.82 GB (`experiments/SAVED.md:3`). Step t has t-1 moving steps.

## 2. The three saved supervised tasks

**Answer template (anchor 1, "a1"; context distillation)**
- Data: W0's prompted greedy answers (format prompt chosen by E45 stage P), trained under the unprompted chat. E78 used
  7,740 of 7,744 sequences (0.83 passes in 100 steps) (`qwen_anchor1/results/train.json .data`). The held-out rows are
  the prompted answers to GSM8K test 256-511 (`qwen_updates/job.py:273`), file
  `qwen_updates/results/large/P/a1_rows.json` (`src/commute/qwen_d3.py:55`). Own rows for reads: 64 a1 rows (`qwen_d3.own_rows`).
- Answer length: mean response 131.5 tokens at step 100 (`qwen_anchor1/results/read.json .step_100.response_mean`); W0 unprompted 202.8.
- W0 → fine-tune: `format_unprompted` 0 → 0.926. GSM8K strict (dropped) 0.784 → 0.621. Held-out distillation KL 0.962 → 0.047.
  (`qwen_recipe/results/train_W.json .states.w0.unprompted.*`, `.states.w0.held_a1.distill_kl`; `qwen_anchor1/results/read.json .step_100`, `.reproduction.rows.held`).
  The learned rule: rate ≥ 0.74 and strict ≥ 0.560 (`read.json .rule`).
- Seeds saved: s0 `~/commute-data/results-large/qwen_anchor1/dW/a1/step_{1,2,3,5,10,20,50,100}` (F32) and s1
  `.../qwen_anchor_checks/dW/a1_seed1/step_{10,100}` (F32). Older bf16 copies are under qwen_updates and qwen_recipe.
- |δW| at step 100 is 0.487 (`read.json .files.100.norm`).
- Shuffled labels: shuffling responses across GSM8K questions leaves every target still in the template. The twin would
  therefore learn the format too, plus answers to the wrong problem. The format itself is not shuffled. (inference)

**TOFU (anchor 2; invented-author facts)**
- Data: `/mnt/nw/home/e.agnew/commute-data/artifacts/tofu/` (`full.json`, `retain_perturbed.json`, `forget10_perturbed.json`,
  `holdout10.json`, ...; `src/commute/qwen_tofu.py:31`). E81 trains 4,000 pairs in one wording. E84/E88 train 787 kept rows ×
  5 wordings = 3,935 sequences (`qwen_tofu_forms/results/forms.jsonl`). Held out: TOFU's paraphrased question on the same
  787 rows. Floor: 400 rows of 20 never-trained authors. Answers are one or two sentences, and MAX_NEW 128 fits the longest
  (`qwen_tofu/run.py:46`).
- New-word recall, W0 → step 1250 (E84 seed 42): paraphrased 0.166 → 0.532; trained wording 0.179 → 0.993; floor 0.190 → 0.211.
  Key-word recall on paraphrased: 0.090 → 0.404 (`qwen_tofu_forms/results/read.json .W0.*`, `.states.1250.*`). The filed
  "answers when rephrased" rule missed (E84 analysis).
- Seeds saved: 42 `.../qwen_tofu_forms/dW/a2_2e-5/step_{1..1250}` (12 files, F32); 142 `.../qwen_anchor_checks/dW/tofu_seed142/step_{100,200,1250}`;
  E91 matched-size steps `.../qwen_size_grid/dW/a2_s{42,142}/step_{27,29|30,550|500}`. |δW| at step 1250 is about 18.1,
  37× the template's (`qwen_size_grid/results/size_grid.json .arms.anchor2_s0.full.norm`).
- Shuffled labels: answers moved to other rows. Each answer names its own author, so a shuffled answer names the wrong
  person. Both the real and the shuffled set are facts W0 lacks (W0 recall equals the floor), so the twin is new against
  new, not noise against learnable structure. (inference)

**Gold maths (GSM8K human solutions)**
- Data: GSM8K train via `commute.gsm8k` (HF `openai/gsm8k`, cached in `commute-data/artifacts/hf/hub`). The E40 split holds out
  256 and trains on 7,217 (`qwen_recipe/results/train_gold_1e-5.json .data.n`; `qwen_updates/job.py:71-72`). 100 steps × 64
  = 6,400 sequences, under one pass.
- W0 → step 100: held-out gold loss 2.495 → 0.510 nats/token. GSM8K test 0-255 strict 0.784 → 0.621. Mean response
  202.8 → 97.5 tokens (`train_W.json .states.w0.held_gold.loss`; `train_gold_1e-5.json .states.step_100.*`).
- Seeds: `qwen_recipe/.../train/gold_1e-5/step_*` is bf16 (`split_x` refuses bf16). `qwen_updates/.../dW/gold_s0` is bf16.
  The E40 nowarmup regen step_100_fp32 is fp32 ("u1" in E152) but uses a different recipe. The E120 pools
  `.../qwen_pass_pools/dW/{known,new}_s{0,1}/step_{5,6,7,10,100}` are F32, but on 62 problems a side, not full gold.
  No full-gold fp32 state with 2 seeds is saved.
- Shuffled labels: each problem gets another problem's solution. The style is kept, and the answer is wrong for its problem. (inference)
- Not found: an explicit held-out *accuracy* for template or TOFU in the brief's sense beyond the above. No seed of any
  of the three has a shuffled twin.

## 3. Masked error and output change

- E152 table 3 (`experiments/qwen_instrument_check/results/step0.json .table3.floors`), all on the **new** decomposition p-9707d6ff:
  - maths 256 tokens, 128 rows: **0.09321** (`qwen_regating_new/results/controls.json .means_W0.K0`, E132)
  - maths 512, 128 rows: 0.07285; FineWeb 512, 128 rows: 0.20900 (`qwen_new_decomp/results/w0_refs_stage1.json .blocks.<b>.masked_kl`, job 56760)
- **Missing floors** (`.table3.updates.<state>[].floor_rows == "gap: no floor stored on these rows"`): a1 rows (256, n 64),
  tofu rows (256, n 128), and the E45 "own" rows (512, n 64) of gold_s0/a2long. **No floor exists on the template's or
  TOFU's own rows.**
- Stored output change on own rows (E93 K, KL(FT ‖ W0) over every position, `qwen_d1_table/results/<s>/state.json clean.W0.K`):
  a1_s0 0.323, a1_s1 0.328, a2_s42 3.121, a2_s142 3.159. On maths 512 × 16 (E91 `size_grid.json .arms.<a>.full.kl`): a1
  0.096 (1.32× the 512 floor), TOFU 0.503 (6.9×), gold_s0 0.697 (9.6×). The E40-recipe maths fine-tunes read 15-18× the maths-256 floor.
  - Caution (inference): if the a1-row floor is near 0.09, the full template update (0.32) is only about 3.5× it, so a 15×
    save size may be unreachable on the template at 100 steps and lr 1e-6.
- Code that computes the floor: `src/commute/qwen_d3.py` `cells(d, rs, change=None)` :280 gives K0 = KL(W0 clean ‖ W0
  masked at m = g, Δ off) per position, summed per row by class. `total(sums, "K0")` :323. It needs `Decomp(decomp_dir, log)` :216
  (fp32 target, fp32 components, bf16 Γ, live set from `GSTATS_OF[p-9707d6ff]` = `qwen_new_decomp/results/large/w0_refs/p-9707d6ff/gstats`)
  and a `RowSet` (:91): `maths_rows()`, or `own_rows("a1"|"tofu", tok, sets)` :128, cut to ROW_LEN 256. For new rows,
  build a `qc.Rows` (tokens, answer mask, first) plus classes. Same quantity elsewhere: `experiments/qwen_new_decomp/w0_refs.py`
  `Apparatus.masked_kl` :199 (asserts a card of 46 GB or more, i.e. L40). An L40 holds decomposition + Γ + model with about 1 GB
  spare and reads row by row (`cluster/README.md`). E132 W0 cells on 128 rows took 118 s, and loads 95 s (`cluster/qwen_instrument_check_step1.sbatch` header).
- `src/commute/qwen_masked.py` is not the floor: it is the student's masked forward for teacher training (E76).
- Output change in training code: `qwen_train.train` reads `mean_kl(model, rows, W0, None)` = KL(W0 ‖ W_t), the mean over
  every position of tracker rows, fp32, at read steps, and saves at grid crossings with a bracket. E91/E107 pick the
  "matched step" by norm or KL from saved steps (`qwen_size_grid/results/size_grid.json .arms.*.{at_norm,at_kl}`).
  Direction differs: training reads KL(W0 ‖ W_t), while E93's K is KL(FT ‖ W0).

## 4. 67M built pairs (definition-lead pages)

Source: `reports/analyses/2026-11-10_analysis_gate-page-working-definition.md:54-83` and `2026-11-10_analysis_old-task-labels.md` label table.
Labels use the adopted cut-offs G ≥ 0.8, H ≥ 0.5 (user, 2026-11-10). G and H are given per seed.

| pair (reuse / new) | G reuse | G new | label read | notes |
|---|---|---|---|---|
| in-context copy (e62_copy) / lookup (e62_lookup), same inputs | 0.70, 0.69 | 0.33, 0.29 | both rewired | copy side "not settled", causal check only (SAVED.md:4) |
| copy 24 keys / cipher 24 keys | 0.68, 0.68 | 0.39, 0.33 | both rewired | |
| identity 32 keys (ident32) / random map 32 (rand32) | 0.69, 0.71 | 0.28, 0.26 | both rewired | |
| real case label / case twin (labels per word) | 0.83, 0.83 | 0.52, 0.55 | re-weighted / rewired | the only pair the label splits |
| real repeat label / repeat twin | 0.76, 0.74 | 0.54, 0.57 | both rewired | narrowest margin 0.17 |
| unpaired: lowercase text (reuse) | 0.36, 0.35 | | no map found | long text read on 8 tokens |
| unpaired: word cipher 50 and 500 (new) | | 0.08/0.07, 0.19/0.16 | no map found | same |
| unpaired: rank1 sharpen (reuse), elicit (open), rand1 (one seed), emoticon | | | | see table |

- Within-pair: "every reuse seed has a higher G than every seed of its new twin" in all 5 pairs (gate page :95). As an
  absolute label, 1 contrast in 5.
- Built controls (gate page :125-140): two-key XOR, **parity of a sum** (stopped: the plain 67M already reads parity at
  0.61 > bar 0.55, i.e. it failed its capacity premise), strongly taught original (stopped), last-number-odd (H 0.466/0.459
  < 0.5). "The wording has not yet detected built."
- Premise doubts:
  - Twins: `~/commute-data/premise/P2.21.md:5` says "a real fine-tune against a shuffled one, which is learned-versus-noise,
    not reuse-versus-new". `P2.4.md:5` says coherent versus noise-fitting "is visible from shape alone" (steering and
    effective rank separate twins).
  - Prior: `P2.3.md:5` gives rank agreement of the share with the prior of 0.85 (CI 0.67-0.93); "real fine-tunes read like
    their shuffled twins on three of four pairs".
  - Size and length: `P2.9.md:7-8` says twins "trained 3 to 10 times longer" (case real step 30 vs twin step 300, TRIED.md:223).
    The spectrum "missed both pairs whose sides had about the same update size".
  - Prior matching precedent: E58/E147 "prior-matched cipher" (cipher prior in [2e-5, 2e-4], TRIED.md:131). E149
    (TRIED.md:262): steering separates copy from the prior-matched cipher (0.44). P2.16 (TRIED.md:263): the two prior-matched
    pairs carry the share's largest and smallest gaps.
  - Gate page "What would make us drop it" (:143): "On a matched pair not used to set anything, such as the Qwen test set,
    the reuse side reads at or below its twin."
- **definition-lead reply: not found.** `~/commute-data/agents/qwen-lead/` holds no `note-definition-lead-*.md`. qwen-lead's
  ask is at `~/commute-data/agents/definition-lead/note-qwen-lead-2026-11-11.md` (18:36): "I ... will pick by 2026-11-12 if no answer."

## 5. Capacity-style checks on the unchanged Qwen3-1.7B

- Arithmetic word problems: GSM8K test 0-255 greedy strict 0.784 (E61 `train_W.json .states.w0.unprompted.strict_dropped`);
  0.779 (`qwen_updates/results/control.json` "W0 accuracy on test 0-255").
- Known vs invented facts, "The answer is X." format: known 0.97, invented 0.015 / 0.008 (two draws)
  (`qwen_updates/results/control.json .facts_s0`, `.facts_s1`). This is a ready-made reuse-vs-new capacity contrast in one format.
- E120 pools: held-out known problems pass@1 0.971 (`qwen_pass_pools/results/grades.json .w0.known_held.pass1`); new
  pool = never solved in 64 samples.
- Copy: E prefix test (`qwen_prefix_test/results/prefix.json`, **base** model, raw text): copy control EM 0.539 vs bar
  0.90. It copies the story and then starts solving (`reports/analyses/2026-10-28_analysis_qwen-prefix-test.md:15-18`).
  This is not the chat model and not a short-token copy.
- TOFU real-author facts and E103 known-answer rows exist (`qwen_anchor_checks/.../answers/W0/real.jsonl`).
- `qwen_pass_test` (E68) is a decomposition pass test, not a capacity test.
- **Not found**: any Qwen check of case change, token copy under a chat cue, translation, cipher or random key-value
  mapping under a prompt. `qwen_fresh_set` and `qwen_fresh_numbers` are GSM8K-style problem sets, not capacity checks.

## 6. Compute route

- Cluster: one node `l40-worker`, 8× L40 (44.3 GiB usable). `debug` 2 h max, 1 GPU, priority. `normal` 24 h, 6 GPUs per user.
  Account c11, partition compute (`cluster/README.md`). The Qwen line may use `normal` when the queue is empty
  (`.claude/CLAUDE.md` Rules). Recent Qwen jobs: E152 step 1 debug, 128G, 2 h (`cluster/qwen_instrument_check_step1.sbatch`); E120
  normal 24G, 3 h, or debug with `--budget-min 110` chunks (`cluster/qwen_pass_pools.sbatch`). env.sh stops a job with
  exit 75 if the card already holds more than 5 GB.
- Model: `Qwen/Qwen3-1.7B` revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` from the HF hub (`qwen_vpd.py:55-56`,
  `qwen_eval.py:35`). In jobs `HF_HOME=/ephemeral/$USER/hf` (`cluster/env.sh`), i.e. a node-local cache. A copy also sits in
  `/mnt/nw/home/e.agnew/commute-data/artifacts/hf/hub/models--Qwen--Qwen3-1.7B`.
- Decompositions: `$COMMUTE_QWEN_NEW_DIR=/mnt/nw/home/e.agnew/commute-data/artifacts/qwen3-1.7b-p-9707d6ff-torch` (new) and
  `$COMMUTE_QWEN_DIR=.../qwen3-1.7b-p-e9fcffd5-torch` (old, frozen callers) (`cluster/env.sh`).
- Pods: the decomposition comes from W&B `edwinagnew-mats/commute/<artifact>` via `python -m commute.qwen_files <dest>`.
  `DECOMP_ARTIFACT` maps p-9707d6ff to `qwen3-1.7b-p-9707d6ff-torch`, but "Whether p-9707d6ff is uploaded: not checked"
  (`src/commute/qwen_files.py:20-24`). The model comes from the HF hub (load_target). δW goes back via `qwen_sft.Uploader`
  (W&B artifact per file) or `download()`. The E40 dW W&B artifact is `qwen3-1.7b-e40-nowarmup-dW` (SAVED.md:21).
  A fine-tune-only pod needs no decomposition.
- Past costs: E84 H100 about $1 for 672 s of training. E120 sampling and pools on RunPod cost $15.72 (E120 analysis header).
- Storage: an fp32 state is 5.64 GB. The `train()` bracket rule saves two states per grid crossing, which doubles storage
  unless the save rule is changed.

Friction: no definition-lead reply yet. There is no masked-error floor on the template or TOFU rows, and no Qwen capacity
check for copy, case or cipher exists, so run 1 must create all three. The gold-maths saved states are bf16 or
different-recipe only.
