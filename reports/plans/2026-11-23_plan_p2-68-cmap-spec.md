# P2.68 step 1: method spec for head-level CMAP (Prakash et al. 2024) on the 67M

Cell P2.68 (regating-lead, 3 runs, due 2026-11-25, cluster `debug` at $0, pods capped at $20). Step 1 only: no code
written or run. Reviewers: redteam-lead and an independent GPT reviewer, before any run.

Sources read in full: the paper (Prakash, Rott Shaham, Haklay, Belinkov, Bau, ICLR 2024, arXiv 2402.14811; the PDF in
`~/commute-data/lit_review_pdfs/`, all 26 pages including appendices A to J) and its released code,
https://github.com/Nix07/finetuning at commit `6d6c0980bf073d305b915669c33533576a70aa77` (2025-10-27; the paper links
https://finetuning.baulab.info, whose README links this repo). Clone, untrusted and never run:
`~/commute-data/scratch/regating-lead/p268-spec/authors-code/`. Code locations below are `file:function:lines` at that
commit; `pp` = `experiment_1/pp_utils.py`, `path` = `experiment_1/path_patching.py`, `cmap` = `experiment_3/cmap_utils.py`,
`nb` = `experiment_3/cmap.ipynb`, `data` = `data/data_utils.py`.

## 1. In plain words

Cross-model activation patching runs the base model on a task input but swaps in the fine-tuned model's output for a
chosen set of attention heads on that same input, and asks how much of the fine-tuned model's accuracy the base model
then reaches. We will find each fine-tune's task circuit with the paper's path patching, then patch that circuit's head
outputs from the fine-tune into the unchanged 67M model on the nine built pairs and the fifteen key-task fine-tunes. It
tells us whether the fine-tune's gain is carried by what a few attention heads now write, which is the paper's evidence
that "fine-tuning enhances existing mechanisms", and it gives the published baseline beside our readings.

## 2. Method spec, quoted

### (a) Circuit discovery by path patching (paper §4.1, p. 4; §4.2, p. 5; App. B, p. 15)

- **Nodes.** "each node is an attention head at a specific token position ... We therefore focus our analysis on the
  attention heads of the circuit, and we consider all MLP layers to be involved in the computation of the final output"
  (§4.1). Code: a head's output is its slice of `o_proj`'s *input* (the attention-weighted values before W_O),
  `pp:get_caches:181-190,213`.
- **Inputs and counterfactual.** "using the synthetic box tracking dataset with 300 examples. For each of the original
  entity tracking tasks x_org we define a corresponding noise task x_noise with a randomized query, box labels, and
  objects" (§4.1). Code: the noise input is another dataset row (different objects and labels, a different query
  index) whose query label is overwritten by a uniformly random capital letter, `data:load_pp_data:1353-1364`;
  `path:pp_main:149` (`num_samples=300`).
- **What is patched.** One sender head at one position takes its noise-run output; every other head at positions from
  the sender's to the last is frozen at its clean output; MLPs recompute (`pp:patching_sender_heads:350-361,379-382`).
  W_O is applied afterwards from the model's own weights (`pp:patching_sender_heads:390-395`). For a later group, the
  receivers' q or v input at the receiver position is taken from that run and everything else is clean
  (`pp:patching_receiver_heads:448-461`, second forward at `path:apply_pp:105-122`).
- **Score.** "We denote p_org as the probability of the correct token predicted by the original run, and we let p_patch
  be the probability assigned to the correct token when patching a specific path ... The patching score for the
  candidate pair is defined as (p_patch − p_org)/p_org. At each iteration we add the paths with the lowest (most
  negative) scores" (§4.1). Code: per example, then averaged over examples, `path:apply_pp:124-140`.
- **Iteration.** Group A, heads to the final logit at the last token (`path:pp_main:213-229`, no receivers, rel_pos 0);
  B, heads at the last token into A's queries (`:232-251`, composition q); C, heads at the query label into B's values
  (`:254-273`, composition v, rel_pos 2); D, heads at the earlier mention of the query label into C's values
  (`:276-295`, rel_pos −1). Selection is top-k most negative, `pp:compute_topk_components:33-59` (`largest=False`).
- **Threshold.** "path patching ... does not provide a clear threshold for the number of heads ... we include a total
  of 90 heads ... (50,10,25,5 heads in Groups A,B,C,D respectively)" (§4.2). Then minimality: "(F(Cir \ K) −
  F(Cir \ (K ∪ {v})))/F(Cir \ (K ∪ {v})). We filter out the heads with a score lower than 1%" (§4.2), K formed
  greedily from "upto top 30% of the heads" of v's group (App. B). Code: `experiment_1/minimality.py:minimality_main:131-188`
  (`percentage=0.3`, threshold at `:183`). F is accuracy over 500 examples with every head outside the circuit
  mean-ablated (§4.2); code `pp:mean_ablate:640-720`, `pp:eval_circuit_performance:723-776`.
- **Faithfulness.** "the percentage of model performance that can be recovered with the circuit, i.e. F(Cir)/F(M)"
  (§4.2); a baseline of "10 random circuits with the same total and per-position number of heads" (§4.3, p. 6).

### (b) CMAP (paper §6.1, p. 8; Fig. 4 caption and §6.2, p. 9)

- **Definition.** "CMAP requires patching activations of the same components of different models on the same input"
  (§6.1). No counterfactual input is used.
- **Direction and components.** "we patch the output of heads in the Goat-7B circuit from the Goat-7B to Llama-7B model
  ... Similarly ... for heads in the FLoat-7B circuit" (§6.1). So the circuit is the *fine-tuned* model's, the
  activations are the fine-tuned model's, and the model run is the base. "We use CMAP to patch the output of the subset
  of heads responsible for dominant functionality in each group ... Since we do not fully understand the functionality
  of Structure Reader heads, we patch all the heads in this group" (§6.1): groups A to C are their DCM subsets
  (§5.1-5.2, App. H), D whole.
- **Code.** `cmap:cmap_out:288-360`: at each patched head's position (`input_ids.size(1) − rel_pos − 1`, or the earlier
  query-label mention for −1), its slice of `o_proj`'s input is replaced by the fine-tuned model's, and the *base*
  model's W_O is applied (`:353-358`). The fine-tuned cache stores `o_proj`'s input (`nb` cell 15). Head sets per
  position: `nb` cell 18 (`{0: A + B, 2: C, −1: D}`) and cell 29.
- **Inputs.** 500 clean task prompts (`nb` cell 4, `cmap:load_data:85-118`), the same ones for all three runs.

### (c) The metric (§3, p. 3; §6.2; `cmap:eval_model_performance:121-156`)

Accuracy: the fraction of the 500 prompts whose argmax next token at the last position is the correct object. CMAP
reports the patched base model's accuracy beside "its default performance (red dashed line)" and "the full fine-tuned
models' performance (green dashed line)" (§6.2). The paper gives no normalised share; the natural one, used here, is
**CMAP share = (acc_patched − acc_base) / (acc_ft − acc_base)**. The notebook's recorded values (Goat into Llama, base
0.66, fine-tune 0.82): full circuit 0.82, A 0.82, B 0.78, C 0.62, D 0.65 (`nb` cell 20).

### Where paper and code disagree

1. Head counts before minimality: the paper says 90 for Llama (50, 10, 25, 5) and 200 for Goat; `path:pp_main:150-153`
   defaults to 20, 5, 10, 5 ("Llama circuit: 20"), and `minimality.py:49-52` reads 58, 10, 25, 5 (Goat 101, 30, 50, 40).
   The saved minimal circuits (`experiment_1/results/circuits/*.json`) match the paper's Tables A2, A3, A7.
2. "Output of heads" is, in code, the pre-W_O vector put through the base model's W_O. A fine-tune that changes only a
   head's W_O is invisible to CMAP (it does in FLoat, a full fine-tune).
3. The notebook's Goat cells call `cmap_out(goat_cache=...)` but the function takes `finetuned_cache`: they cannot run at
   this commit; the FLoat cells match, and their recorded output stops at batch 28 of 63 (`nb` cells 19, 30).
4. `cmap_out` places positions from the padded length, `path` from each row's last index: identical only when every
   row in a batch has the same length.
5. Faithfulness ablates, in code, every head at every position other than the three circuit positions to a
   per-position mean over 500 prompts with a random query label (`pp:mean_ablate:684-691`, `data:get_data_for_mean_ablation:8-56`);
   the paper says only "mean ablating of all the heads ... not involved in Cir".

## 3. Mapping to the 67M

- **Models.** Base = W0, the released 4-block 67M (6 heads of 128 per block, 24 heads; no biases; rotary; plain target
  model, no decomposition read). Fine-tuned = W0 + c·δW of the saved state, c as its source set (1 or E187/E237's).
- **Task prompts.** Every task format reads one answer position `half` with one target (`experiments/share_untrained/windows.py:load_read`).
  Windows are truncated to `half + 1` tokens: exact under causal attention, and it puts the read at the last position
  so the authors' position arithmetic holds unchanged. kv tasks: the marker sits at `half − 1`, the key at `half`.
- **Counterfactual (path patching only).** The source is another held-out window of the same task (a different
  context, as the paper's different row), with the key token at `half` replaced by a token drawn uniformly from the
  tokens that are not keys of the task, so the noise prompt has no trained answer (the paper's random letter is mostly
  absent from the context and has none). 300 prompts, as the paper.
- **Groups.** The four entity-tracking groups have no direct analogue: the answer here is a learned key-to-value map,
  not an in-context lookup. We keep the authors' first three steps with their compositions and read positions renamed:
  A, heads at the read position into the logit (`rel_pos` 0); B, heads at the read position into A's queries (rel 0,
  q); C, heads at the marker into B's values (rel 1, v). D is dropped: there is no earlier mention to read.
- **Selection.** With 24 heads the paper's hand-set counts do not transfer. The pool for each group is every head with a
  negative mean score (the paper "add[s] the paths with the lowest (most negative) scores"); the paper's own minimality
  rule (1%, top 30% for K) then prunes it. Faithfulness F(Cir)/F(M) and 10 random circuits are reported as §4.2-4.3.
- **Circuit found on.** Each fine-tuned state, as CMAP in §6.1. It is patched into W0 on 500 held-out task prompts
  (fewer where a task has fewer; the count is reported).
- **Items.** The fifteen key tasks: E237's scaled items, `experiments/refit_96key/results/large/dW_{copy,rank1,elicit,cipher,prior_cipher}_s{0,1,2}.pt`
  times `experiments/key_grid/results/stage0.json` `.c`, with `random_<state>` beside each. The nine built pairs are
  E187's (`experiments/size_matched/run.py` `BUILT`; larger side times `experiments/size_matched/results/m67.json` `.pairs.<id>.c`);
  their states are listed in `experiments/share_untrained/results/inventory.json` (`.states.<name>.file`; 52 files,
  all present by `ls` on 2026-11-23) and the e58 pairs' in `experiments/refit_96key/results/large/`.

## 4. Positive control (first and alone)

The paper has no stated CMAP sanity check; its de facto one is that the full circuit reaches the fine-tuned accuracy.
Ours, each with a known answer:

1. **Do-nothing** (the control state times 0): every patched accuracy equals W0's exactly, numerator 0, share undefined.
2. **Planted head.** A δW confined to one block-3 head's q, k and v rows, scaled to a gain of about 2 log-odds. Patching
   that head at the read position must reproduce the fine-tuned logits there exactly (nothing after block 3 reads other
   positions): CMAP share 1, logit difference within the measured floor of two identical W0 runs.
3. **W_O blind spot.** The same head with δW in its `o_proj` columns only: CMAP numerator exactly 0 while the gain is
   positive. It shows the patch unit (§2, disagreement 2) is the code's.
4. **Port check (optional, ~$2, one H100 ~20 min).** The authors' own FLoat into Llama-7B CMAP with their saved circuit,
   through our port, must repeat the notebook's 0.82 (full circuit) and 0.74 (B). Needs the LLaMA-7B weights.

## 5. Departures

**From the paper and code (this run).**
1. Model and task: a 4-block, 24-head model on learned key-to-value maps, not Llama-7B on in-context entity tracking.
2. Groups A to C with renamed positions, D dropped; head pools by negative score, not hand-set counts (§3).
3. No DCM: the desiderata are entity-tracking specific (object, label, position), so groups are patched whole. CMAP is
   reported per group and for the whole circuit.
4. Noise key drawn from non-keys; windows truncated at the read (exact).
5. A normalised share beside the paper's raw accuracy, and the project's log-odds share beside both, for the scorecard.
6. Our own mean-ablation and minimality loop: `pp:mean_ablate` hard-codes the entity-tracking positions and would fail.

**The existing neuron share against CMAP.** The "neuron share, whole width" r_n (`src/commute/neuron_r.py:batch_cells`, cell
L01) runs W0 with every block's MLP firing (`down_proj`'s input) taken from the fine-tuned model's run on the same
window, W0's `down_proj` applied, and reads (L01 − T0)/(T1 − T0) in log-odds. It matches CMAP in direction (fine-tune
into base), in inputs (same input, no counterfactual) and in the patch unit (the pre-output-weight activation, put
through the base's output weights, as `cmap_out`). It departs in four places: (i) MLP neurons, which the paper never
patches; (ii) all 12,288 neurons, no circuit; (iii) every position, not the circuit's positions; (iv) log-odds share,
not accuracy. So "CMAP at neurons" is justified for r_n only as "CMAP's operation applied to all MLP neurons at all
positions", with (i) to (iv) named. The **top-neuron share** (`src/commute/masked_reads.py:topk_logits`) is not a patch:
W0 keeps its own firing and only the choice of which k neurons survive comes from the fine-tune. It must not carry the
CMAP label. The locator's "sufficiency" (`src/commute/locator.py:reads`, line 279) is CMAP of one head or one MLP at a
time, at every position, in log-odds: per-part CMAP without a circuit.

## 6. Can the authors' code run on our model?

Not unchanged. It targets HF `LlamaForCausalLM` through baukit `TraceDict` (module names
`model.layers.{l}.self_attn.{q,k,v,o}_proj`, layer index parsed as `layer.split(".")[2]`, `model.config.num_attention_heads`,
`model.state_dict()[f"{layer}.weight"]`, `.logits` on the output). Ours names `h.{l}.attn.{q,k,v,o}_proj`. The minimum
we write: (1) an alias module (~40 lines) exposing our blocks under the authors' names with a `config`, a `.logits`
output and a `state_dict` holding those keys, one real copy per model (W0 and W0 + c·δW as plain weights, so the
`state_dict` reads are true); (2) the prompt and noise builders of §3 (~40 lines); (3) a driver calling the authors'
`apply_pp`, `patching_sender_heads`, `patching_receiver_heads`, `get_receiver_layers`, `compute_topk_components` and
`cmap_out` unchanged, plus our minimality and faithfulness loop (departure 6). Their modules call `sys.path.append` and
import `peft`, `datasets` and `LlamaTokenizer` at import, which our rules forbid; the open choice is to copy those six
functions verbatim into `src/commute/` with a check that their source matches the clone at the commit, or to run the
clone in its own environment (`environment.yml`: transformers 4.35.0, peft 0.5.0, baukit from git).

## 7. Expected values and compute

| number | W0 / do-nothing | planted head | random update, same size | fine-tune |
|---|---|---|---|---|
| CMAP accuracy, any group | = W0's accuracy | = fine-tune's | within the null of W0's | between W0's and the fine-tune's |
| CMAP share (accuracy) | undefined (0/0) | 1 | undefined (gain under floor) | paper's full circuit: 1.0 |
| CMAP share (log-odds) | numerator 0 | 1 within floor | undefined | not predicted |
| faithfulness F(Cir)/F(M) | as W0's own circuit | 1 | not read | paper: 0.88 to 1.0 |

The share's floor (smallest acc_ft − acc_base read) is set from the random updates' accuracy change in the same job's
first stage. Compute per fine-tuned state, 67M, prompts of about 21 tokens: path patching 24 + 48 + 48 = 120 patched
forwards over 300 prompts, caches 600, minimality at most about g² + 2g mean-ablated forwards per group of g heads over
100 prompts, CMAP about 15 patched forwards over 500 prompts (4 groups, 10 random circuits, the full circuit): roughly
50,000 prompt-forwards, about 1M tokens per state. About 105 items (15 key, 15 random, about 75 pair states) give about
100M tokens: one cluster `debug` job per shard at $0; the runtime comes from one item measured in stage 0 times the
count, because the authors' per-row Python loops, not the forwards, will set it.

## Open questions for the reviewers

1. Is dropping DCM (patching whole groups) acceptable, or do we need a task-specific desideratum (e.g. an alternate-key
   desideratum, App. G's "label" analogue) to pick sub-groups as the paper does?
2. Head pools by negative score then the paper's minimality: is there a better-sourced count for a 24-head model?
3. Is the noise key (a non-key token) the right analogue of the random letter, given our answers are learned, not read
   from context?
4. The e53 (lowercase, next-token Pile windows) and e62 (induction windows) pairs have no marker or key: noise there
   would be a random token at the read position; is that defensible, or should those two pairs be read by CMAP only on
   the circuit found for the kv tasks?
5. Verbatim copy of six functions with a source check, or the authors' environment as-is?
6. Run the optional port check on LLaMA-7B (control 4)?

## Not run

Nothing was run or coded in this step. Not read: the authors' DCM training code beyond its loss (App. H; `experiment_2/DCM.py`),
since DCM is a proposed departure; the attention-knockout notebook (App. A).
