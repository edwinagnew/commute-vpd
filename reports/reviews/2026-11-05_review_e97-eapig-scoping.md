# Scoping: EAP-IG for finding mechanisms on the 67M (user's suggestion, 2026-11-05)

*mechanism-detection-lead scoping agent, 2026-11-05. No GPU, nothing installed, nothing run. Sources: the eap-ig README, `pyproject.toml` and `src/eap/attribute.py` (fetched), Hanna et al. 2024 (arXiv 2403.17806v2). Any run proposed from this goes through `/design` first.*

**Answer.** Worth adding, but as our own ~250-line edge read on our hooks, not the package. On a model of 28 parts, exact patching already costs only 28 forward passes, so EAP-IG's speed buys nothing at part grain: E97's carry read (A) already is the exact version. What it adds is the **edge** grain (927 edges), which separates a part's direct write from what it passes on (A's stated blind spot), and **input-pair circuits**, which give W₀'s circuit for C on tasks where we have none (lowercase, the 24-key maps).

## 1. What EAP-IG computes

- **Edge score.** For an edge u → v: (z′ᵤ − zᵤ) · (1/m) Σₖ ∂L/∂(v's input) evaluated at z′ + (k/m)(z − z′). Here z is the clean activation and z′ the corrupted one, interpolated on the input embeddings ("EAP-IG-inputs") with m = 5 in the paper. Plain EAP is the single-gradient version. The package also has "EAP-IG-activations", "clean-corrupted" and "exact" (true edge patching).
- **Inputs.** It needs clean/corrupted pairs of equal token length. `attribute.py` raises "Number of positions must match", and the corrupted activations always come from the **same model** on corrupted tokens. No argument takes a second model's activations. Mean and zero ablation exist as alternatives.
- **Grain.** The default is edges: every upstream head, MLP or input into every later head's q, k and v inputs (separately), into each MLP, and into the logits. A node mode and a neuron mode also exist. On our model (4 blocks × 6 heads, 4 MLPs) that is 828 + 70 + 29 = **927 edges**.
- **Circuit size.** The circuit is picked by greedy search or top-n at n = 30 … 1000 edges. It is scored by normalised faithfulness (m − b′)/(b − b′): the metric with everything outside the circuit patched to its corrupted value, as a fraction of the clean-minus-corrupted gap. The paper takes the smallest circuit with faithfulness ≥ 0.85. Its main finding: overlap is no substitute for faithfulness. Two tasks with node IoU 0.58 kept 0.61 and 0.2 of each other's behaviour.

E96's own gradient check is the reason to trust the IG form and not EAP. The single step missed the true fall on all four cases (0.506 against 7.118), while the 8-step path integral matched within about 2%.

## 2. Can it run on our model?

The 67M is `LlamaSimpleMLP` (`~/param-decomp/param_decomp/pretrain/models/llama_simple_mlp.py`), loaded inside the VPD `ComponentModel` by `commute.vpd.load_component_model` and wrapped by `commute.steering.Net`. It has pre-RMSNorm, rotary on the full head (dim 128, base 10⁴, rotate-half), 6 query and 6 key/value heads (no grouping), a tanh GELU MLP, no biases, and tied embeddings. `transformer-lens` is in neither `pyproject.toml` nor `uv.lock` (`transformers` and `einops` are). The package needs `transformer-lens>=2.15` and asserts `use_attn_result`, `use_split_qkv_input` and `use_hook_mlp_in`.

| Option | Work | Risk |
|---|---|---|
| **TL conversion + package** | Pin TL (shared venv, user's call); convert weights per state, about 100 lines plus a logit-equality check; dataloaders and metrics, about 60. The W₀-against-state read needs a fork of `attribute.py`, about 100. Total ≈ 260 + a dependency + a vendored fork | TL's dependency churn in the venv every clone shares; conversion details (RMS weights, rotary convention, TL weight processing); maintaining a fork |
| **Own edge read on `net.net`'s weights** (recommended) | `src/commute/edges.py`: a functional forward that writes each head's output separately (o_proj's column slice), with per-head q/k/v inputs and the module's own rotary buffers and `apply_rotary_pos_emb`, about 120 lines; exact edge patch about 30; IG scores about 50; greedy circuit and faithfulness about 50. Total ≈ 250, no new dependency. It works under `net.at(dW)` since that swaps the weights in place | The functional forward drifting from the module, caught in stage 0 by checks 1-3 below |

**Recommend the second.** It needs no install, and it serves the W₀-against-state read (section 3), which the package cannot do without a fork. Stage-0 checks, each with a measured bar:

1. With no patch, logits match `net.run` within the fp32 noise floor.
2. With every edge patched, the read gives the corrupted run.
3. The IG edge scores into the logits, summed per node, match E97's exact A on the same states within the 10% that E96's path check used.

## 3. How it would serve our question

Two different "corruptions" answer two different questions:

- **Gain path (W₀ against the state, same input).** The corrupted value of every upstream part is W₀'s output on the same input, and the read is taken in the state. This is A at edge grain, and Prakash et al.'s cross-model patch is its exact form. It is not in Hanna et al. (a departure to state). It says through which edges the fine-tune's gain reaches the logits, and it splits A's credit between a part's direct write and what downstream parts carry. Changes inside v's own matrices stay invisible here; that is B's job.
- **Circuit (one model, clean against corrupted inputs).** These give W₀'s circuit for C and the state's circuit for the task. The comparison is **cross-faithfulness**: run the state with everything outside W₀'s C-circuit patched, and report the share of the state's clean-minus-corrupted gap it keeps, with edge IoU beside. This follows the paper's own conclusion. A new wiring of old heads changes the edge set even when the node set is the same, which softens (but does not remove) the limit "a new circuit built from old parts reads as reuse".

| Task (states) | Clean | Corrupted (same length) | Metric | W₀'s C |
|---|---|---|---|---|
| E62 copy / lookup | cue window, read at M | copy: `copying.absent_b` (B → B′). lookup: the second A → another key A′. Both corruptions run on both maps as each other's control | logit(B) − logit(B′); logit(word(A)) − logit(word(A′)) | induction on the same windows, read at the second A, with `absent_b` |
| E53 lowercase / cipher (whole rows) | lowercased rows / `cipher_rows` | lowercase: earlier case positions put back to their capitalised token (`case_pairs`' lo/ca tokens; the re-tokenised cased row is not the same length). cipher: the plain rows, token for token | E53's gap: log p(lower) − log p(capital); log p(π(x)) − log p(x) | lowercase: W₀ on the same pair (E53: W₀ already performs it). cipher: none, so W₀'s circuit is the null |
| 24-key rank-1 / cipher / elicit / copy-24 | held-out window (`steering.Task`, key at `half`) | key → another key of the 24 | logit(v(k)) − logit(v(k′)) | rank-1: W₀ on the same pairs. cipher: none |

**Known answers.**

- **Positive control:** (2,4)'s edge to the logits must rank at the top in W₀'s induction circuit and in all three copy states, and not in the lookup states.
- **E97 stage 0 plants, gain path.** The block-3 plant (`h.3.mlp.down_proj`) changes nothing upstream, so every edge except mlp3 → logits must read 0 exactly; mlp3 → logits must equal G (bar 1e-4). The block-1 plant puts nonzero scores only on edges out of mlp1 and on edges downstream of it. Its direct share (mlp1 → logits) against what it passes on is the plan's "inheritance floor", now read rather than inferred.
- **A bar to set at stage 0, not guessed.** The corrupted baseline must leave a gap (b − b′) larger than its noise floor; otherwise the faithfulness ratio is undefined for that task. This matters most for the partial recasing on lowercase.

## 4. Cost on a debug L40 (estimates from E96's logs)

Per batch, IG takes m forward + m backward passes plus 2 forward passes: about 3m + 2 forward equivalents, so **17 at m = 5 and 32 at m = 10**. Use m = 8, the step count E96 checked. A gain-path read and a circuit read are one IG each per state; W₀ needs one per task.

- **E62 windows.** E96 swept 25 forwards in about 3 s per state (run.out 17:41:03-17:41:21), and the 8-step path check took about 2.5 s per case (stage2.out 17:56:13-17:56:21). So one IG is about 3 s per state, and exact patching of all 927 edges is about 2 min per state, cheap enough to do instead of IG.
- **E53 whole rows.** E96 took 30-43 s per state for 25 forwards at 500 × 127 tokens, about 1.5 s per forward. Whole rows (256 × 512 tokens) are about 2× that, so one IG at m = 8 is about 1.5-2 min per state. Exact edges would be about 55 min per state, so IG with exact spot checks on the top 50 edges is the only fit.
- **Total.** About 18 small-window states × 2 reads, plus 5 whole-row states × 2 reads, plus greedy faithfulness sweeps (about 10 sizes × 1 forward each): about 25 min. That is one or two `debug` jobs, $0, under the gate. Memory: gradients at 76 downstream inputs plus 29 upstream differences at 16,384 tokens per batch is about 5 GB, which fits.

**Where it sits.** This would be a new E-code (E97 has its two amendments budgeted for its own tree). The best time is after E97's stages 0-1, since checks 1-3 reuse E97's A on the same states.

## Not run

Nothing was run or installed. Not checked:

- TL's handling of this architecture.
- Whether `case_pairs`' partial recasing leaves W₀ a usable gap.
- Neuron-grain EAP-IG.
- The timings above, which are extrapolated rather than measured.

Friction: the E96 logs are single JSON lines of up to 60 KB, so `sed -n` on them overflows the tool output, and reading them needs `cut -c`.
