# Loading the Qwen3-1.7B decomposition into the commute harness (2026-10-27)

I read the shared checkpoint's metadata, the JAX code at commit 177d0cce8 that wrote it, and our PyTorch harness, to
work out what the decomposition holds and how our measurements could run on it. It holds the subcomponents of all 196
matrices plus a 32-block causal transformer that reads the residual stream at every layer boundary and also reads each
matrix's own subcomponent activations; Δ is not stored, and nothing in our torch dependency can run this importance
function. I recommend converting the checkpoint once to safetensors and porting the importance function and the
masked forward to PyTorch, using the JAX loader only to produce reference outputs on fixed tokens that the port must
match.

Read-only scoping. No repository file other than this one was changed. One CPU job on the cluster worker
(`fleet/cpu-run`, about 1 min) listed the checkpoint's keys and read each array's `.zarray` metadata through
tensorstore; no array data was read. The script and its output sit in
`/mnt/nw/home/e.agnew/commute-data/scratch-qwen-loader/` (outside the repo; delete at will).

**Two corrections to the brief.** First, the 67M decomposition our harness uses does not have a per-layer Γ. Its
`final_config.yaml` says `ci_config: mode: global, fn_type: global_shared_transformer` (8 blocks, width 2048), and the
torch class it builds attends in both directions over the whole window
(`.venv/.../param_decomp/models/components.py`, `SelfAttention`, "Bidirectional attention (no causal mask)"). So
`src/commute/vpd.py:166-170`, which says Γ "reads each position's own pre-weight activations, so a 41-token window is
as valid an input as a 512-token row", is wrong for the 67M model: Γ's value at the trigger depends on every token of
the window, after the trigger too. That bears on the task-local harness (`local_faith.py`) and belongs in a review
report; it does not change anything below. Second, the sibling report `2026-10-27_analysis_qwen-rl-gsm8k.md` calls
the training text "chat-formatted" as if `meta.json` said so; it only gives a recipe name (section 6), and the
importance function has 8.54 B parameters, not 5.7 B (its count left out the output heads and input projection).

## 1. What the checkpoint holds

Files: `runs/p-e9fcffd5/ckpts/100000/decomposition/` is an Orbax `StandardCheckpointHandler` item, OCDBT store, zarr v2,
zstd level 1. `_METADATA` lists 1415 leaves: 1413 arrays and two `None` leaves (`blocks.gate`, `blocks.norm_scales`,
i.e. no SwiGLU gate and no learned block-norm scales). Every array is stored as **float32** (`<f4`); 47 GiB on disk
after compression. The training item (optimizer, persistent adversary, frequency averages) was not shared, so the
decomposition can be evaluated and fine-tuned from but not resumed bit-for-bit.

**Subcomponents.** `components.stacks.<site>.0` is V and `.1` is U, stacked over the 28 layers
(`core/components.py:510,529`: `stacks[group] = (Vs, Us)`). V is `[layer, d_in, C]` and U is `[layer, C, d_out]`, the
same orientation as our torch `V (d_in, C)`, `U (C, d_out)`, so no transpose is needed.

| site | V shape | U shape | params per layer | target W params | ratio |
|---|---|---|---|---|---|
| q | 28 × 2048 × 2048 | 28 × 2048 × 2048 | 8,388,608 | 4,194,304 | 2.0 |
| k | 28 × 2048 × 1024 | 28 × 1024 × 1024 | 3,145,728 | 2,097,152 | 1.5 |
| v | 28 × 2048 × 1024 | 28 × 1024 × 1024 | 3,145,728 | 2,097,152 | 1.5 |
| o | 28 × 2048 × 2048 | 28 × 2048 × 2048 | 8,388,608 | 4,194,304 | 2.0 |
| gate | 28 × 2048 × 6144 | 28 × 6144 × 6144 | 50,331,648 | 12,582,912 | 4.0 |
| up | 28 × 2048 × 6144 | 28 × 6144 × 6144 | 50,331,648 | 12,582,912 | 4.0 |
| down | 28 × 6144 × 6144 | 28 × 6144 × 2048 | 50,331,648 | 12,582,912 | 4.0 |
| one layer | | | 174,063,616 | 50,331,648 | 3.5 |
| all 28 layers | | | **4,873,781,248** | 1,409,286,144 | 3.5 |

Each per-layer count is $C \cdot (d_{in} + d_{out})$; for example gate is $6144 \cdot (2048 + 6144)$. There are 24,576
subcomponents per layer and 688,128 in all.

**Importance function Γ** (`ci_fn.backbone.*`), width 3840, 32 heads of size 120, FFN width 15,360:

| leaf | shape | params |
|---|---|---|
| `backbone.in_proj_w`, `in_proj_b` | 59,392 × 3840; 3840 | 228,069,120 |
| `backbone.blocks.attention.wq, wk, wv, wo` | 32 × 3840 × 3840 each | 1,887,436,800 |
| `backbone.blocks.attention.query_key_norm.q_gain, k_gain` | 32 × 3840 each | 245,760 |
| `backbone.blocks.w1`, `b1` | 32 × 3840 × 15,360; 32 × 15,360 | 1,887,928,320 |
| `backbone.blocks.w2`, `b2` | 32 × 15,360 × 3840; 32 × 3840 | 1,887,559,680 |
| `backbone.heads.{0..6}.weights`, `.biases` | 28 × 3840 × C; 28 × C, for C = 2048, 1024, 1024, 2048, 6144, 6144, 6144 | 2,643,099,648 |
| `readouts.layers.<l>.<site>.{positive,negative}.{input_scale,input_bias,output_scale}`, `.bias` | C each, 7 vectors per site | 4,816,896 |
| `backbone.inv_freq` | 60 (fixed RoPE table, stored) | — |
| **total** | | **8,539,156,224** |

59,392 is 29 × 2048: the input is 29 residual-stream taps concatenated. Head stack $s$ follows the order sites were
declared, q, k, v, o, gate, up, down (`global_transformer/arch.py:347-354`, `core/components.py:452-466`); gate, up
and down have the same shape, so their order is taken from the code and must be confirmed by the equivalence check.
The 196 readouts hold the calibrated input scales: the 99.9th percentile of each subcomponent's activation magnitude
was mapped to 1 at step 0 (`component_conditioning.py:246,341-372`), so no calibration data is needed to evaluate.

**Totals.** 13.41 B parameters: 53.7 GB in float32, 26.8 GB (25.0 GiB) in bf16. The JAX loader restores floating
leaves directly as bf16 (`experiments/lm/load_run.py:404-408`), which is the "about 24 GB" in the sharer's notes.

**Δ is not stored.** It is the residual $\Delta = W - VU$ per matrix, with $W$ read from the Hugging Face snapshot at
load. In the masked forward it is never materialised; the forward computes
$y = ((xV) \odot (m - d))\,U + d \cdot (x W^\top)$, which equals $((xV) \odot m)\,U + d \cdot x \Delta^\top$
(`core/decomposed_linear.py:65-85,142-187`). Here $m$ is the per-component mask in $[0,1]$ and $d$ the per-position
mask on Δ. The faithfulness loss computes $W - VU$ in float32 (`core/model.py:430-448`).

## 2. What Γ computes, and how its output becomes masks

All of Γ runs in bf16 (`core/precision.py:9`; taps and parameters are cast in `backbone.py:99-105` and
`arch.py:225-231`). The function is evaluated on the **clean** forward of the current target weights
(`core/recon_eval.py:202-224`, `experiments/lm/eval.py:206-221`).

1. **Taps.** The raw residual stream at boundaries 0 to 28: boundary 0 enters layer 0, boundary $l$ leaves layer
   $l-1$, boundary 28 leaves layer 27 before the final norm (`targets/transformer_taps.py:38-40`). In Hugging Face
   terms these are the inputs of decoder layers 0-27 plus the output of layer 27. `output_hidden_states[-1]` is the
   wrong tensor for boundary 28: transformers 4.57 overwrites it with the final-normed state
   (`transformers/utils/generic.py:1093-1101`).
2. **Normalise and concatenate.** Each tap is RMS-normalised with no learned scale, eps = float32 machine epsilon
   (1.19e-7), then the 29 are concatenated to width 59,392 (`layers.py:161-180`).
3. **Input projection** $x \leftarrow x\,W_{in} + b_{in}$ to width 3840 (`arch.py:270-271`).
4. **32 pre-norm blocks**, scanned (`arch.py:272-285`; block `layers.py:458-486`):
   - weightless RMS norm; $q, k, v = h\,W_q^\top, h\,W_k^\top, h\,W_v^\top$ (stored `[out, in]`), split into 32 heads of 120;
   - **query-key norm:** each head RMS-normalised over its 120 channels, times learned per-channel gains
     `q_gain`, `k_gain` reshaped to 32 × 120 (`layers.py:291-306`);
   - RoPE, rotate-half form, base 10,000, positions restarting at every document start (`target_ports/llama.py:100-117`,
     `sequence.py:26-37`);
   - **causal** attention restricted to the same document (`sequence.py:22-24`, `layers.py:388-400`), with logits
     $s = q \cdot k / \sqrt{120}$ computed in float32 and **soft-capped** to $30 \tanh(s/30)$ before the softmax
     (`attention.py:120-139`); output $\cdot\,W_o^\top$ plus residual;
   - weightless RMS norm; FFN $\mathrm{GELU}_{exact}(h W_1 + b_1)\,W_2 + b_2$ plus residual.
5. **No final norm.** Each head stack is applied directly: $p_s = x\,W_s + b_s$, one head per site
   (`arch.py:286-291,316-324`).
6. **Conditioning on the site's own subcomponent activations**, the "affine clipped component activation" and the
   "sign split" of the run name. With $h = x_{site} V$, where $x_{site}$ is the matrix's clean input
   (`targets/transformer.py:290-301`: q, k, v read the normed attention input; o the attention core; gate, up the normed
   MLP input; down the MLP hidden), each component's preactivation gains
   $s_+\,\mathrm{clip}(a_+ h + b_+, 0, 1) + s_-\,\mathrm{clip}(-a_- h + b_-, 0, 1) + \beta$: a positive arm on $h$, a
   negative arm on $-h$ and a free bias (`component_conditioning.py:215-243,313-339,375-381`).
7. **Squash.** The lower importance is $g = \mathrm{clip}(p, 0, 1)$ and the upper is $p$ above 1 leaking with slope
   0.01 (`interface.py:36-41`, `squashing.py:8-45`). Masks use the lower value.

**Masks.** Stochastic: $m = g + (1-g)\,u$ with $u \sim U(0,1)$ per position and component, and $d \sim U(0,1)$ per
position and site (`masking.py:135-141`, `recon.py:47-49`). CI-masked: $m = g$ with no Δ. Unmasked: $m = 1$, no Δ
(`masking.py:126-133`). The evaluation adversary (`PGDReconLoss` in the config) starts one random source per site,
shape $[C]$ plus one Δ scalar, shared by every row and position of the 128 × 512 batch, sets $m = g + (1-g)\,a$, and
takes 20 sign steps of 0.1 clipped to $[0,1]$ up the KL from the clean logits to the masked ones
(`recon_eval.py:104-170`, `adversary.py:316-356`, `targets/losses.py:50-60`). The config's auxiliary hidden-activation
terms in that probe have coefficient 0, so the objective is the KL alone.

**Differences from what our harness assumes** (torch `param-decomp` 74146b5 and `src/commute/`):

| | 67M decomposition, torch 74146b5 | Qwen decomposition, JAX 177d0cce8 |
|---|---|---|
| Γ input | the inputs of the 24 decomposed matrices, each RMS-normed, concatenated | the 29 raw residual boundaries, each RMS-normed, concatenated, plus each site's $xV$ |
| attention | bidirectional over the window, no mask | causal, within one document, positions reset per document |
| query-key norm, softcap | none | per-head RMS with learned gains; logit cap 30 |
| biases | input projector and MLP | input projection, FFN, heads |
| output | one linear head, split by sorted module name | one head per site, stacked by kind |
| dependence on V | none | through $h = xV$: changing V changes $g$ even with Γ's own parameters frozen |
| class available in our venv | `GlobalSharedTransformerCiFn` | none; must be ported |
| mask formula | $((xV) \odot m)U + d\,x\Delta^\top$, Δ materialised | the same in exact arithmetic, Δ folded as $d\,xW^\top - d\,(xV)U$ |
| squash | clamp to $[0,1]$ forward, leaky gradient; same upper | the same |
| precision | fp32 weights under bf16 autocast | bf16 weights and compute |
| rows | one sequence per row | packed documents, masks from document ids |

The row that matters for the commute question is "dependence on V". Under component training or teacher training V
moves, so $g$ moves even when Γ's parameters are frozen. "Γ frozen, reading the variant's own activations" must now
also say whether $h$ is read through the updated V or the original one. The JAX training code uses the current V, and
V's gradient through Γ joins its reconstruction gradient (`component_conditioning.py:1-14`).

## 3. Three ways to run our measurements on it

**(a) JAX-native.** Run `open_run(..., layout=ONE_DEVICE_OWNER_LAYOUT)` (`load_run.py:374-377,434-498`) and call the
JAX `clean_forward`, `evaluate_ci_from_captures` and `masked_forward` (`core/model.py:352-422`;
`recon_eval.py:194-256` is a worked consumer).
- Work: rewrite the measurements we use (route split, task-local faithfulness, outside share, teacher training) in JAX
  against an unfamiliar placement API, about as much code as the torch originals. Updated states from torch training,
  and every RL checkpoint of the target, must be passed into JAX pytrees.
- Training: component training and teacher training would be new JAX code; possible, since the repo is a JAX trainer.
  RL of the target in JAX has no tooling for generation or policy optimisation in this repo.
- Risk: lowest for Γ's numbers, which are the authors' own. Highest for everything else, and the venv here is CPU-only
  (no CUDA plugin), so a GPU environment must be built from the clone's lock file into a separate directory.
- Verify: nothing to verify for Γ; our ported measurements would need re-validation against their torch versions.

**(b) Convert to safetensors and port Γ and the masked forward to PyTorch.**
- Work: a conversion script of about 60 lines that reads each array with tensorstore, layer slice by layer slice, and
  writes float32 safetensors, about 54 GB to `commute-data`. It needs no GPU; the cluster node sees the share and has
  458 GB of RAM. Then about 150 lines for Γ: steps 1-7 above, with flex attention or explicit float32 logits for the
  softcap. About 80 lines of hooks on the 196 `nn.Linear` of HF `Qwen3ForCausalLM` for the masked forward, and about
  100 lines for the stochastic and adversarial checks. The torch `param-decomp` PGD cannot be reused as is: it calls
  `model.calc_causal_importances` on its own input cache (`metrics/pgd_utils.py:163-165`), and that cache holds no
  residual stream.
- Training: works natively. Teacher training as written needs only the target, the teacher and U, V
  (`component_training.py:205-263,308-367`); Γ is not in the loop. RL of the target uses ordinary torch tooling, and the
  decomposition is loaded only at evaluation checkpoints.
- Risk: a port bug that looks like a finding. The likely ones are the head order of gate, up and down, the
  boundary-28 tap, the bf16 casts of readout vectors and gains, and RoPE position resets.
- Verify: a numerical equivalence check against JAX on the same tokens, described below.

**(c) Hybrid, JAX computing Γ and torch running the rest.** Taps go from torch to JAX and $g$ comes back, through DLPack.
- Work: least porting. But JAX 0.11.1 with its CUDA plugin and torch 2.8 must share one process and one GPU, and their
  pinned cuDNN versions are likely to conflict; two processes would need inter-process copies of 1.4 MB per token.
  Γ also needs V for its readouts, so every V update has to reach JAX as well.
- Training: as in (b), since Γ is never differentiated in our recipes.
- Risk: environment fragility, about 35 GB resident across two allocators, and no gain over (b) once (b) passes its
  equivalence check.

**Recommendation: (b), with JAX used once as the oracle.** The one-off reference job runs on a cluster L40, which reads
the share directly, and stays under 30 min:
1. Build a JAX CUDA environment from a separate clone at 177d0cce8; the read-only clone stays untouched.
2. On fixed tokens, write the reference outputs: 4 FineWeb-style rows of 512 tokens and 4 GSM8K chat prompts, one
   document per row. The outputs are the target logits; the 29 taps; the preactivations $p$ for every site on 2 rows
   and summary statistics on all; and the masked-forward logits for $m = g$ without Δ, $m = 1$ without Δ, and one fixed
   source array, written from numpy and fed to both sides, with Δ on.
3. The torch port reproduces each output from the safetensors on the same tokens.

Pass rules, fixed before the run:
- The target logits agree to the bf16 noise floor, as the authors' own HF parity test does
  (`tests/targets/qwen3_hf_parity/test_qwen3_hf_parity.py:110-150`).
- The share of (position, component) pairs whose $g$ differs by more than 0.1 is below 1e-3.
- Each masked-forward KL to the clean logits agrees within 5% relative.
- The 20-step adversary's KL, over 3 seeds per side, overlaps between the two sides.

The noise floor for these rules is torch fp32 against torch bf16 on the same tokens. Feeding the JAX taps into the
torch Γ separates a Γ-port error from an HF-versus-JAX target difference.

## 4. Memory and time on one GPU

Assumed sustained bf16 throughput: about 450 TFLOP/s on an H100 SXM, 45% of peak, and about 100 TFLOP/s on an L40,
55% of its 181. Benchmark a slice before trusting these. FLOP counts are 2 per multiply-add.

**Resident weights in bf16:** target 3.4 GB, which is 1.72 B parameters with the tied 151,936 × 2048 embedding;
subcomponents 9.75 GB; Γ 17.1 GB; total **30.3 GB**. A materialised Δ would add 2.8 GB; folding it as JAX does avoids
that.

**FLOP per token.** These fix the time per operation:
- Target forward: $2 \times 1.41$ B for the matrices plus $2 \times 0.31$ B for the output head, **3.4 GFLOP**.
- Γ: $2 \times 8.53$ B for the backbone and heads, plus $2 \times 2.11$ B for the readouts' $xV$, **21.3 GFLOP**, six
  target forwards; its attention adds about 0.13 GFLOP at 512 tokens.
- Masked forward with Δ on: $2 \times 4.87$ B for the subcomponents plus 2.8 for the folded $d\,xW^\top$ plus 0.6 for
  the head, **13.2 GFLOP**.

| operation | per 512-token row | H100 | L40 | memory beyond the 30.3 GB |
|---|---|---|---|---|
| plain target forward | 1.8 TFLOP | 4 ms | 18 ms | needs only 3.4 GB of weights in all |
| clean forward + Γ + one masked forward | 19.4 TFLOP | 43 ms | 0.19 s | 1.3-1.8 GB per row: dense $g$ is 0.70 GB, logits and taps the rest |
| adversarial check, 128 × 512, 20 steps, $g$ cached | 290 TFLOP | 1.4 min for the batch | 6 min | per micro-batch, about 2.5-3 MB per token with per-layer recomputation |
| the same with $g$ recomputed every pass | 520 TFLOP | 2.4 min | 11 min | Γ stays resident |

Batch sizes that fit for the masked forward: about 4-8 rows of 512 on an L40 and 16-24 on an H100. Unloading Γ after
computing $g$ frees 17 GB. Caching $g$ for the whole 128-row adversary batch is 90 GB dense in bf16, so it must live
on the host, be stored sparsely, or be recomputed per micro-batch. The config's slow ladder of 1280 steps costs 64
times the 20-step check: about 1.5 h on an H100 and 6 h on an L40.

**Teacher training, a backward pass through U and V.** Our recipe trains offsets on the subcomponents Γ keeps alive,
with Γ and Δ frozen, through hooks on the target. Γ is not loaded. With a fraction $a$ of the 4.87 B subcomponent
parameters trained, the cost is about $3 \times (3.4 + 19.5a) + 3.4$ GFLOP per token. That is 72 GFLOP at $a = 1$,
or 2.6 s per step of 32 × 512 tokens on an H100 and 12 s on an L40.

With Adam the trainable state is 16 bytes per parameter: the parameter, its gradient and two moments, all fp32. That
gives $78a$ GB, plus the frozen base U, V and two 3.4 GB copies of the target. Free offsets on every subcomponent do
not fit one 80 GB card. A fraction $a$ up to about 0.3 fits an H100 and up to about 0.15 fits an L40; rank-8 offsets,
as in E13, or plain SGD at 8 bytes per parameter fit either card. The alive fraction is unknown until measured.

**RL of the target** involves only the 1.72 B-parameter model; the decomposition is not resident during training.

## 5. Does Γ need the whole sequence?

Γ is not per-token: every block attends over earlier positions of the same document. But it is **causal**, so $g$ at
position $t$ depends only on tokens up to $t$. The importance of a prompt's tokens does not change as a rollout grows,
and a prompt plus its rollout can be scored as one document in one pass. Arbitrary text works if each sequence starts
its own document, and padding takes document id −1 (`sequence.py:10-41`).

The limit is length. Γ and the decomposition were trained on 512-token rows, so both saw RoPE positions 0-511 only.
Documents longer than a row were cut into fragments that each restart at position 0, per
`experiments/lm/data.py:16-38` and `migrations/data_layout.py:23-27`. Scoring tokens past position 511 puts Γ outside
anything it was trained on. Cutting a long rollout into 512-token windows would score the target on a fragment rather
than on the computation it actually runs. Non-thinking GSM8K fits; thinking-mode traces mostly do not, which matches
the sibling RL report.

## 6. The training data, "firestock3 … qwen3_1_7b_chat"

What the files say: `meta.json` holds only `seq_len` 512, tokenizer `Qwen/Qwen3-1.7B`, `format_version` 2, recipe name
`firestock3-text-qwen3_1_7b_chat-eod-unitsplit-rowshuf-v1`, and end-of-document token 151643 (`<|endoftext|>`).
The public repo contains no `firestock3` recipe, and git history is squashed, so the recipe itself is not available.

What can be inferred, with how sure each step is:
- "eod": each document ends in `<|endoftext|>`. The repo's dataset schema notes that Qwen3 documents end in
  `<|endoftext|>`, not in the chat EOS `<|im_end|>` (`param_decomp_formats/dataset_store.py:94-102`). That note implies
  documents can contain `<|im_end|>` inside, which fits chat transcripts. Likely, not verified.
- "qwen3_1_7b_chat": the tokenizer is already named separately, so the tag probably means formatted for, or generated
  by, the post-trained chat model. Whether `<think>` blocks are present cannot be told.
- "rowshuf": rows shuffled; the repo requires this for training sets. "unitsplit": unknown.
- Training read rows as packed documents with per-document attention masks and position resets, in both the target
  and Γ (`migrations/data_layout.py:23-27`, `experiments/lm/input_format.py:63-85,204-212`).

What it implies for GSM8K chat prompts: if the data is chat-formatted Qwen3 text, a templated non-thinking GSM8K prompt
is in distribution. If it is web text with an end token, the template tokens `<|im_start|>`, `<|im_end|>` and
`<think>` are rare inputs to Γ, and faithfulness at those positions is unknown. The first measurement on the port
should therefore compare the CI-masked and stochastic KL on templated GSM8K prompts against plain web text, per
position, before any RL result leans on the decomposition. The authors' own evaluation numbers for this run are in
W&B project `param-decomp-llama`, if the sharer can grant access, and would give the reference levels.

## Not checked

- No array values were read: not the calibrated input scales, not the gains, not any subcomponent. Whether the
  readouts are calibrated rather than left at 1 is taken from the code path, not observed.
- The alive fraction of subcomponents and the per-token count of non-zero importances, which set teacher-training
  memory and whether $g$ can be cached sparsely.
- The gate, up, down order of `heads.4-6`, taken from code; the equivalence check confirms it.
- Whether the JAX loader runs on an L40, with cuDNN flash attention for the target and the bf16 restore's host-memory
  peak, and how long the restore takes from the shared disk.
- All timings are arithmetic at assumed throughput; nothing was benchmarked.
- The contents of firestock3: chat template, thinking traces, and what "unitsplit" means.
- Whether HF Qwen3 in torch 2.8 matches the JAX target port on these weights in bf16. The authors test the port
  against HF, but on an 8 B model and a tiny random one.
- The JAX training losses were read only far enough to confirm they do not change what is stored: the pooled
  persistent adversary, the attention-pattern and hidden-activation terms, and non-linearity locality.
