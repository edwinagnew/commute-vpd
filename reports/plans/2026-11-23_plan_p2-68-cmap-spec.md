# P2.68 step 1: method spec for whole-group head CMAP (after Prakash et al. 2024) on the 67M

Cell P2.68 (regating-lead, 3 runs, due 2026-11-25; cluster `debug` at $0, pods capped at $20). Step 1: spec only, nothing
coded or run. Revision 2 answered the GPT plan review, `reports/reviews/2026-11-23_review_p2-68-cmap-spec-gpt.md`,
with the lead's seven decisions. Revision 3 (2026-11-23) answers its re-check 1, recorded in the same file. The changes
are listed in Appendix H. Redteam-lead's review still comes before any run.

Sources, read in full: the paper (Prakash, Rott Shaham, Haklay, Belinkov, Bau, ICLR 2024, arXiv 2402.14811, all 26
pages) and the released code, https://github.com/Nix07/finetuning at `6d6c0980bf073d305b915669c33533576a70aa77`
(2025-10-27). The paper links finetuning.baulab.info, which links this repo. The repo has no licence file.
Abbreviations: `pp` = `experiment_1/pp_utils.py`, `path` = `experiment_1/path_patching.py`, `cmap` =
`experiment_3/cmap_utils.py`, `nb` = `experiment_3/cmap.ipynb`, `data` = `data/data_utils.py`. Quotes and code
locations are in Appendix A.

## 1. In plain words

Cross-model activation patching runs the base model on a task input but gives chosen attention heads the outputs the
fine-tuned model computes on that same input, and counts how much of the fine-tuned model's accuracy the base model then
reaches. We will find each fine-tune's head circuit with the paper's path patching and pruning, then give the unchanged
67M model those heads' fine-tuned outputs, group by group, on the nine built pairs and the fifteen key-task fine-tunes.
The result says how much of each gain the fine-tuned outputs of a few heads carry into the base model. On its own it
does not show that the base model uses those heads: the paper showed that separately (§4-5), and we do not repeat it.

## 2. The method, as the paper and code fix it

1. **Path patching (§4.1, §4.2, App. B).** Nodes are attention heads at a token position. A head's output is its slice of
   `o_proj`'s input, before the output weights. The noise prompt is another dataset row with its query label replaced by
   a random capital letter. One sender head at one position takes its noise output. Every other head from that position
   to the last is frozen at its clean output, and MLPs recompute. For later groups, the previous group's query or value
   input is taken from that run. The score is (p_patch − p_org)/p_org, per example, averaged. The most negative heads
   are added, in groups A (into the logit), B (into A's queries), C (into B's values at the query label) and D.
2. **Pruning and faithfulness (§4.2, App. B).** Heads are pruned by the 1% minimality rule. A circuit's accuracy F(Cir)
   is read with every other head mean-ablated, and faithfulness is F(Cir)/F(M).
3. **CMAP (§6.1, §6.2).** "patching activations of the same components of different models on the same input". The
   circuit and its activations are the fine-tuned model's, and the base model is the one run. The swap replaces each
   patched head's pre-output vector and applies the *base* model's W_O (`cmap:cmap_out:353-358`). Groups A to C are
   patched as their DCM subsets (notebook cells 8-9, 18) and D whole.
4. **Metric (§3, §6.2).** Accuracy is the argmax at the last position equal to the correct token, over 500 prompts. It is
   shown against the base and fine-tuned accuracies, and the paper defines no share.

Paper and code disagree in eight places (Appendix A.2). Among them: the head counts before pruning; one prompt offset
shared by the sender and receiver hooks; the pruning helpers' skipped pairs and rounding; 100 prune prompts against the
paper's 500; and the notebook's Goat cells, which cannot run.

## 3. Our run: whole-group head CMAP

- **Models.** The base is W0, the released 4-block 67M with 6 heads of 128 per block. The fine-tune is W0 + c·δW, held as
  a second plain copy of the weights.
- **Prompts.** Windows are cut to `half + 1` tokens. This is exact for a causal model, and it puts the read at the last
  position. The label is read from the uncut window, by the format's rule (table, `NEXT` or `LAST`).
- **Noise (discovery only).** Another window of the same task (a new context) supplies the noise. In kv tasks the key at
  `half` is also replaced by a seeded non-key token, so the noise prompt has no trained answer. The marker sits before
  the key, so a head at the marker sees only the changed context. In E62 the second key sits at `half − 1`, before the
  marker at the read, and it is replaced by a seeded token outside the key pool (Appendix B.2).
- **Groups.** A: heads at the read position into the logit. B: heads at the read position into A's queries. C: heads
  at the position before the read into B's values. That position is the marker in kv tasks and the preceding key in
  E62. E53 has no such site and runs A and B only. D is dropped.
- **Edit sites.** The scored groups call the authors' `apply_pp` unchanged. Its one `rel_pos` serves both hooks, so C
  edits B's values at the position before the read, which is the site V-composition reads; re-check 1 agrees (Appendix
  B.1). The other site, B's values at the read position, appears only as a negative control (§4.4). A group whose
  receiver group is empty is skipped and recorded, never run as direct-to-logit.
- **Selection.** The pool is every head with a negative mean score. It is then pruned by the paper-first minimality rule
  written out in Appendix C. Faithfulness is read against 10 random circuits matched per position.
- **CMAP.** Each group whole, then the whole circuit, then the 10 matched random sets, from the fine-tune into W0, on
  the scoring rows.
- **Read.** Accuracy, as the paper does. Beside it, the share (acc_patched − acc_W0)/(acc_ft − acc_W0) and the
  project's log-odds share, neither clipped: zero, negative and over-1 values are reported as read. A share is reported
  only where the fine-tune clears the gate of §4.6, the same gate in every section.
- **Formats and rows.** kv tasks and E62 run A, B and C; E53 runs A and B (Appendix B.3). Discovery and pruning use
  train windows and scoring uses held-out windows. That includes E62's 800 saved train rows and E53's original scorer's
  train windows. The manifest of states, scales, row seeds and the split is frozen in Appendix E.

## 4. Controls, first and alone (stage 0; detail in Appendix D)

Every floor and tolerance is measured in this job; none is guessed.

Every "equal" check has three tolerances, one per unit. Logits use the logit tolerance. Accuracy compares exact paired
correct counts on the same rows. Log-odds use a log-odds tolerance measured on the same hook.

1. **Hook floor.** Give W0 its own cached head outputs through the `cmap_out` einsum path, against a plain W0 forward.
   The largest |Δlogit| is the logit tolerance and the largest |Δlog-odds| the log-odds tolerance. The paired correct
   counts must be identical.
2. **Do-nothing** (the control state times 0): the counts of patched (head, position) sites must be non-zero, and every
   unit must equal W0's within its tolerance.
3. **Planted head, through the whole pipeline.** One block-3 head's value rows are trained by the pinned recipe of
   Appendix D until the gain clears the gate. Discovery and pruning must return that head in A. CMAP of A must give
   share 1. Two checks must give a numerator of 0: the wrong head (A without it, and a random other head) and the wrong
   position (the same head swapped at the position before the read).
4. **Two-layer path, on W0, no training.** The sender cache equals the clean cache except at one block-0 head at the
   position before the read, where a fixed non-zero δ is added. After the authors' first pass, the receiver's `v_proj`
   output must change at that position and stay at a matched no-op floor at the read position. The authors' receiver
   edit at that position must then change the receiver's read-position output. The same edit at the read position must
   leave every unit unchanged. Appendix D gives the details.
5. **Output-weights blind spot.** The same block-3 head with only its `o_proj` columns trained clears the gate in the
   scored unit, or the control fails. The CMAP numerator must then be 0.
6. **Gate.** The gate is measured on the 15 random updates (E237's generated `random_<state>` items) on their own
   scoring rows. N is the largest |paired correct-count change| and L the largest |mean log-odds change|. A fine-tune
   (or a plant) clears the gate when its correct-count gain exceeds N and its log-odds gain exceeds L. If N is 0, the
   first resolvable gain is one more correct prompt.

## 5. Departures

From the paper and code:
1. The model and task differ. Ours is a 4-block, 24-head model on learned key-to-value maps, not Llama-7B tracking
   entities in context. Groups are renamed and D is dropped.
2. Head pools come from negative scores, not hand-set counts.
3. There is no DCM, so groups are patched whole ("whole-group head CMAP").
4. The noise key is a non-key token, and windows are cut at the read.
5. Discovery, pruning and scoring rows are disjoint. In the paper's code the CMAP rows include the discovery rows.
6. The minimality rule is written paper-first, not as the code's helpers (Appendix C). The means are ours.
7. E53 runs A and B only: its windows have no site before the read that carries the task. E62's C is read at the
   preceding key, the site that maps onto the paper's query label.
8. A normalised share and a log-odds share are reported beside accuracy.

**Neuron share and CMAP.** The whole-width r_n (`src/commute/neuron_r.py:batch_cells`, cell L01) runs W0 with every
block's MLP firing (`down_proj`'s input) taken from the fine-tuned model's run on the same window, through W0's
`down_proj`. It is a same-input swap from fine-tune to base before the output weights, as `cmap_out` does for heads. So
the label "CMAP at neurons (all MLP neurons, all positions; log-odds share)" is accurate as an analogy at the level of
the operation. It departs from the paper in four ways: MLPs are never patched there, there is no circuit, every position
is swapped, and the unit is the log-odds share. The top-neuron share (`src/commute/masked_reads.py:topk_logits`) is not
an activation swap, so it must not carry the label. The locator's "sufficiency" (`src/commute/locator.py:reads`, line
279) treats heads and MLPs differently. For a head it swaps `o_proj`'s input, which is CMAP's unit. For an MLP it swaps
`down_proj`'s *output*, so the fine-tuned output weights come along. That is not CMAP's unit and not r_n's. It is read
one part at a time, at every position, with no circuit, in log-odds and the label gap.

## 6. The authors' code

The authors' code targets HF `LlamaForCausalLM` through baukit. It runs from a pinned, read-only checkout at the
commit, and it is not copied into the repo. An adapter, about 40 lines, exposes our blocks under their module names with
a `config`, a `.logits` output and a truthful `state_dict`. We call their `apply_pp`, `patching_sender_heads`,
`patching_receiver_heads`, `get_receiver_layers`, `compute_topk_components` and `cmap_out`, all unchanged. Our code
adds the prompt builders, the minimality and faithfulness loop, and the controls. Their `mean_ablate` hard-codes
entity-tracking positions, so the loop is ours. Setup is in Appendix F.

## 7. Expected values, price and stop rule

| number | W0 / do-nothing | planted head (A) | wrong head / position | random update | fine-tune |
|---|---|---|---|---|---|
| accuracy, patched | = W0's | = fine-tune's | = W0's | within the floor of W0's | not predicted |
| share, accuracy | undefined | 1 | 0 | undefined (under floor) | not predicted |
| share, log-odds | numerator 0 | 1 | 0 | undefined | not predicted |
| faithfulness | W0's own circuit | ≈ 1 | — | not read | paper own-circuit 0.99-1.0, cross-model 0.88-0.97 |

Each "=", 1 and 0 holds within the unit's tolerance of §4. Stage 0 times each stage on one key item and on the planted
items, recording group sizes. Each later item is priced as stage time × items, with pruning scaled by the item's
measured g² + 2g. The stop rule, and the hard spend cutoff of $20 in total checked before any paid shard launches, are
in Appendix G.

## Open questions for redteam-lead

1. Is a non-key noise token the right analogue for answers that are learned, not read from context?
2. Is the kv tasks' C group worth running when its noise is only the context? E62's C has a key to change.

## Not run

Nothing was coded or run in this step. Dropped: the port check on LLaMA-7B in the paper's own setting (cost and setup
risk; it would pin checkpoint, tokenizer, prompts and the released masks and compare raw counts). Not read: the DCM
training code beyond its loss (App. H) and the attention-knockout notebook (App. A).

---

# Appendix

## A. The paper and code, quoted

### A.1 Steps

- **Nodes** (§4.1, p. 4): "each node is an attention head at a specific token position ... we consider all MLP layers to
  be involved in the computation of the final output". The head output is `o_proj`'s input (`pp:get_caches:181-190,213`).
- **Data** (§4.1): "the synthetic box tracking dataset with 300 examples. For each of the original entity tracking tasks
  x_org we define a corresponding noise task x_noise with a randomized query, box labels, and objects". The source row
  comes from `data:load_pp_data:1353-1357` and the random letter from `:1359-1364`.
- **Score** (§4.1): "(p_patch − p_org)/p_org. At each iteration we add the paths with the lowest (most negative)
  scores". Code: `path:apply_pp:124-140`.
- **Patching.** The freeze is at `pp:patching_sender_heads:350-361,379-382`, with the model's own W_O at `:390-395`. The
  receiver edit is at `pp:patching_receiver_heads:448-461`, and the second pass at `path:apply_pp:105-122` is skipped
  when there are no receiver layers (`:106`).
- **Groups** (`path:pp_main`): A at `:213-229`; B at `:232-251` (q, rel 0); C at `:254-273` (v, rel 2); D at
  `:276-295` (v, rel −1). Selection is `pp:compute_topk_components:33-59`, with `largest=False`.
- **Counts** (§4.2, p. 5): "does not provide a clear threshold ... a total of 90 heads ... (50,10,25,5 heads in
  Groups A,B,C,D respectively)".
- **Minimality** (§4.2): "(F(Cir \ K) − F(Cir \ (K ∪ {v})))/F(Cir \ (K ∪ {v})). We filter out the heads with a score
  lower than 1% ... only consider heads that positively contribute". App. B, p. 15: "we use {F(Cir \ {vj}) −
  F(Cir \ {v, vj}) | vj ∈ G, vj ≠ v} to rank all heads in G \ v and then consider upto top 30% of the heads to form the
  subset K". F is "the accuracy score averaged over 500 examples ... by first mean ablating of all the heads in the
  model that are not involved in Cir" (§4.2).
- **Faithfulness** (§4.2): "F(Cir)/F(M)". Random baseline (§4.3, p. 6): "10 random circuits with the same total and
  per-position number of heads".
- **CMAP** (§6.1, p. 8): "CMAP requires patching activations of the same components of different models on the same
  input ... we patch the output of heads in the Goat-7B circuit from the Goat-7B to Llama-7B model ... the subset of
  heads responsible for dominant functionality in each group ... we patch all the heads in this group [Structure
  Reader]". Code: `cmap:cmap_out:288-360`; inputs at `cmap:load_data:85-118` (500 prompts); accuracy at
  `cmap:eval_model_performance:121-156`.
- **Reading** (§6.2, p. 9): "improves the performance of Llama-7B beyond its default performance (red dashed line) ...
  recovering the full fine-tuned models' performance (green dashed line)". Notebook cell 20 (Goat into Llama, base
  0.66, fine-tune 0.82): full 0.82, A 0.82, B 0.78, C 0.62, D 0.65. Cell 31 (FLoat): 0.82, 0.82, 0.74, 0.56, 0.65.

### A.2 Where paper and code disagree

1. **Head counts before pruning.** The paper says 90 (Llama) and 200 (Goat, FLoat). `path:pp_main:150-153` defaults
   to 20, 5, 10, 5, and `minimality.py:49-52` to 58, 10, 25, 5 (Goat 101, 30, 50, 40). The saved minimal circuits match
   Tables A2, A3 and A7.
2. **What is swapped.** The paper says "output of heads". The code swaps the pre-W_O vector and applies the base W_O,
   so a fine-tune of a head's W_O alone is invisible.
3. **One offset for sender and receiver.** `apply_pp` passes one `rel_pos` to both hooks (`path:apply_pp:84-122`). The
   paper names the path without naming the receiver's edit position. For every group the paper runs, the shared offset
   edits the site the composition reads. That is the receiver's values at the sender's position, or its query at its own
   position for A from B. The reviewer's re-check 1 agrees. A discrepancy of form only; the run uses it unchanged.
4. **Pruning helpers** (`pp:compute_pair_drop_values:940-1000`, `pp:get_head_significance_score:1003-1067`).
   - The skip test `layer_1 is not layer_2 and head_1 is not head_2` compares by identity, so a distinct pair sharing a
     head index is never removed together.
   - Entries with zero accuracy are dropped (`:996`).
   - Only positive drops are ranked (`:1044`).
   - K is capped at `ceil(0.3 × |G|)` (`:1040`).
   - Accuracy is rounded to 2 decimals before the division (`pp:eval_circuit_performance:775`).
   - A zero denominator raises (`minimality.py:183`).
5. **Prune prompts.** The driver defaults to 100 prune prompts (`minimality.py:47`); the paper says 500.
6. **Mean ablation.** Every head at every non-circuit position is replaced by a per-position mean over 500
   random-label prompts (`pp:mean_ablate:684-691`, `data:get_data_for_mean_ablation:8-56`). The paper says only "all
   the heads ... not involved in Cir".
7. **Positions in `cmap_out`.** They are counted from the padded length (`:344`), and `path` counts from each row's last
   index.
8. **The notebook.**
   - The Goat cells pass `goat_cache=` to a function that takes `finetuned_cache`, so they cannot run.
   - The FLoat output stops at batch 28 of 63, and its scores live only in Markdown.
   - Cell 18's `+=` grows the A list in place, so later per-group runs inherit B's heads.
   - The groups patched are DCM subsets (FLoat B: 13 heads, not the circuit's 29).

## B. Path patching on the 67M

### B.1 Edit sites and the empty-receiver rule

The scored groups call `path:apply_pp` unchanged, with the authors' `rel_pos` (windows are cut, so the read is the last
position).

| group | `rel_pos` | sender at | receiver input edited | edited at |
|---|---|---|---|---|
| A | 0 | read | none (logit) | — |
| B | 0 | read | A's q | read |
| C | 1 | position before the read | B's v | position before the read |

The C row is the V-composition site: B, at the read position, reads its values there. The other site (B's v at the read
position) is never scored; it is only the null arm of §4.4, run through the authors' receiver hook with its `rel_pos`
set to 0. If A (or B) is empty after its pool is formed, B (or C) is skipped and recorded as "no receivers". The authors'
code would silently score it as direct-to-logit (`path:apply_pp:106`).

### B.2 What each group's noise changes

- **kv.** The key at the read is a non-key token, and the context is another window's. The marker is the same token in
  every window and comes before the key. A head at the marker therefore differs between clean and noise only through the
  context. C measures heads whose marker output depends on the context and matters to B. It may come out empty, which
  is then a result.
- **E62** (`src/commute/copying.py:cue_windows:249-269`). The window reads "… A B … A M", with the second A at `half − 1`
  and the marker M at the read. In the noise prompt the second A is a seeded token outside the key pool, so C's site
  changes directly. It is the analogue of the paper's query label.
- **E53.** The window is Pile text with the scored word form next. No position before the read carries the task, so
  there is no C.

### B.3 Formats

| format | tasks | clean label | noise | groups |
|---|---|---|---|---|
| kv | e46, e54, c3, e58 | `table[toks[:, half]]` from the uncut window | other window, key at `half` → seeded non-key | A, B, C (marker) |
| NEXT | e53 | `toks[:, half + 1]`, uncut window | other window of the task | A, B |
| LAST | e62 | the target column, never fed | other window, second A at `half − 1` → seeded non-pool token | A, B, C (preceding key) |

## C. The paper-first minimality rule

1. **Nodes.** A node is (block, head, position). A head selected in both A and B at the read position is one node and
   is kept in B, as `pp:get_circuit:824-830`. The paper is silent.
2. **F(S).** F(S) is the exact count of correct argmax at the read position over the prune rows, divided by their number
   and not rounded. Every head output outside S, at every position, is replaced by its per-position mean over the noise
   prompts built from the prune rows. MLPs run.
3. **Order.** Groups are scored one at a time, each against the whole unpruned circuit Cir, as `minimality.py:131-188`.
4. **Ranking.** For v in group G and every distinct node v_j in G \ {v} (no pair skipped):
   drop_j = F(Cir \ {v_j}) − F(Cir \ {v, v_j}). Rank by drop_j, descending, with ties broken by block then head.
5. **K.** K is the first m ranked nodes that have drop_j > 0, with m = ceil(0.3 × |G \ {v}|). If |G| = 1 or no drop is
   positive, K = ∅. The code's cap is ceil(0.3 × |G|), stated as a departure.
6. **Score.** num = F(Cir \ K) − F(Cir \ (K ∪ {v})), den = F(Cir \ (K ∪ {v})). Keep v if den > 0 and num/den ≥ 0.01,
   or if den = 0 and num > 0. Drop v if den = 0 and num = 0, or if num ≤ 0.
7. **Empty pools and groups.** An empty pool gives an empty group, and its dependants are skipped (B.1). A group that
   pruning empties is kept empty and reported.
8. **Reference.** Ten random circuits with the same number of nodes per position, by a seeded draw, give F(random) and
   F(random)/F(M) beside F(Cir)/F(M). The same 10 sets are swapped in CMAP.

## D. Controls in detail

- **Tolerances.** The hook floor run patches every head at the read position and the position before it from W0's own
  cache through `cmap_out`'s einsum, against a plain W0 forward, on the scoring rows of e58/copy. It gives the logit
  tolerance (largest |Δlogit|) and the log-odds tolerance (largest |Δ log-odds| at the read). The paired correct counts
  must be identical. Every equality below is judged in its own unit: logits by the logit tolerance, accuracy by exact
  paired counts, log-odds by the log-odds tolerance.
- **Non-empty sites.** Every CMAP call logs the number of (head, position) entries it overwrote. Zero stops the job,
  except where the recorded group is empty.
- **Planted head, pinned recipe.** It follows E58's fine-tune as refit_96key filed it
  (`experiments/refit_96key/results/predictions.json` `.settings.recipe`; `keydep.finetune`, `src/commute/keydep.py:100-104`).
  - `finetune.finetune_arm_a` with `ArmAConfig(o_token = e58/copy's table, lr = anchors.FT_LR = 3e-5, batch 32,
    kl_lambda 1, grad_clip 1, optimizer "adam", steps 300, checkpoints (10, 30, 100, 300), seed 0)`.
  - It runs on e58/copy's train windows, fp32, tf32 off.
  - Two additions are ours: `matrices = ("h.3.attn.v_proj",)`, and a gradient mask zeroing the rows outside h*'s 128.
    h* is a seeded block-3 head.
  - The read is the first checkpoint whose gain clears the gate (§4.6). If none clears it, the control fails.
  - Nothing after block 3 reads other positions, so CMAP of any set holding (h*, read) equals the fine-tune.
  - **Pass:** h* is in A after pruning, and the share is 1 in every unit.
  - **Wrong head:** CMAP of A \ {h*}, and of one other seeded head, has numerator 0.
  - **Wrong position:** h* swapped only at the position before the read has numerator 0.
- **Two-layer path, on W0, no training,** on e58/copy's discovery rows.
  - The receiver r is the block-1 head with the largest W0 attention from the read to the marker, measured and
    recorded. The sender s is a seeded block-0 head.
  - **Sender cache.** It equals the clean cache everywhere, except that s's output at the marker is clean + δ. δ is one
    fixed seeded Gaussian vector, rescaled to the mean norm of s's clean output at the marker, so it is non-zero.
  - **Matched no-op floor.** The same passes with δ = 0, against plain clean forwards, give one floor for each traced
    quantity.
  - **Pass one.** The authors' `patching_sender_heads` runs with s and `rel_pos` 1, tracing r's `v_proj` output. At the
    marker it must differ from clean by more than its no-op floor. At the read it must equal clean within its no-op
    floor.
  - **Marker edit.** The authors' `patching_receiver_heads` runs with `rel_pos` 1. r's read-position output (its slice
    of `o_proj`'s input) must change by more than its no-op floor.
  - **Read-site edit (null arm).** The same hook runs with `rel_pos` 0. r's read-position output must equal clean within
    its no-op floor, the logits within the logit tolerance, the paired correct counts exactly, and the log-odds within
    the log-odds tolerance.
- **Output weights only.** h* is trained with `matrices = ("h.3.attn.o_proj",)` and the mask on columns outside h*'s
  128, by the same recipe. It must clear the gate in the scored unit, or the control fails. Pass: the CMAP numerator is
  0 in every unit.
- **Gate.** As §4.6: N is the largest |paired correct-count change| and L the largest |mean log-odds change| over the 15
  random updates. A gain must have its correct-count gain above N (at least 1 if N = 0) and its log-odds gain above L.
- **Stops.** Any control missed stops the job before stage 1.

## E. Frozen manifest

- **Key tasks.** E237's 15 items `e58/{copy,rank1,elicit,cipher,prior_cipher}_s{0,1,2}_scaled`.
  - Updates are `experiments/refit_96key/results/large/dW_<task>_s<seed>.pt` times `.c[<task>]` from
    `experiments/key_grid/results/stage0.json` (copy 1, rank1 0.9201, elicit 0.8707, cipher 0.7520, prior_cipher 0.7798).
  - Their 15 `random_<state>` items are generated by E237's `World.random` (`FT.random_direction`, seed 237001 + the
    state's index), times the same c. They are not saved updates.
  - Windows are `experiments/refit_96key/results/inventory.json` `.maps.<task>.windows`: 1214 train and 286 held-out
    rows each.
- **Pairs.** E187's nine (`experiments/size_matched/run.py` `BUILT`). The larger side is multiplied by
  `experiments/size_matched/results/m67.json` `.pairs.<id>.c`:

  | pair | c | larger side |
  |---|---|---|
  | e46_copy_random | 0.6590 | new |
  | e53_lowercase_cipher | 0.8962 | new |
  | e54_matched | 0.9813 | new |
  | e54_own | 0.5217 | new |
  | e58_copy_prior_cipher | 0.7798 | new |
  | e58_rank1_96 | 0.8174 | new |
  | e62_copy_lookup | 0.8344 | new |
  | open_pair_c3 | 0.9975 | reuse |
  | open_pair_c3_copy | 0.9901 | new |

  - State files are `experiments/share_untrained/results/inventory.json` `.states.<name>.file` (52 files, all present
    on 2026-11-23). The e58 pairs use the refit_96key files above, at E187's c, not E237's.
  - A state in two pairs (e54 elicit; c3 cipher) is read once per scale it appears at.
  - Rows, held-out (train): c3 75 (301); e46 and e54 97 (403); e58 286 (1214).
  - **E62.** Read rows 200; train rows 800, with targets `target[arm]["train"]` in
    `experiments/induction_cue/results/large/windows.pt` (saved at `experiments/induction_cue/run.py:98-107`).
  - **E53.** Scoring uses `Run.scored`'s windows (Pile val rows 0-255, `experiments/performed_anchor/run.py:314-325`).
    Discovery and pruning use the original scorer's train windows, `Run.task`'s `t.train` (train rows 0-255,
    `CFG.n_write_windows` = 2000, `:327-334`), rebuilt by that code. If they cannot be rebuilt, the scored windows are
    split 1:1:1 by source row, so no row's overlapping windows fall in two splits. The reason is reported.
- **Row split, seed 2026112368, per task.**
  - Scoring rows are all held-out rows, the same rows for every state of the task.
  - Discovery rows are min(300, ⌊n_train / 2⌋) train rows by seeded permutation.
  - Prune rows are the next min(500, the rest) train rows.
  - Noise rows and replacement tokens come from the same seed.
- **Do-nothing.** E237's control state times 0.

## F. The authors' code: checkout and adapter

- **Checkout.** A fresh `git clone` at the commit into `~/commute-data/vendor/nix07-finetuning-6d6c098/`, read-only,
  outside the repo. The sha256 of `experiment_1/pp_utils.py`, `experiment_1/path_patching.py` and
  `experiment_3/cmap_utils.py` is recorded in the plan's predictions file. The adapter checks the hashes at start and
  stops on a mismatch. Nothing of theirs is committed, and the repo has no licence to permit it.
- **Loading.** The adapter loads those modules with `importlib.util.spec_from_file_location`, in their own process
  environment. Their module-level `sys.path.append` is theirs; our code does no `sys.path` editing.
- **Dependencies.** baukit at a pinned git commit, einops, peft and datasets (imported by their modules), pinned in a
  separate `cmap` dependency group at versions that resolve with our lock, recorded.
- **Adapter.** `model.layers.{l}.self_attn.{q,k,v,o}_proj` aliases our `h.{l}.attn.*` modules, and
  `config = (num_attention_heads 6, num_hidden_layers 4, architectures ["LlamaForCausalLM"])`. The forward returns
  `.logits`. `state_dict()` holds the aliased keys of a real weight copy (W0, or W0 + c·δW), so their W_O reads are
  true.
- **Our code.** The B.3 prompt builders, the rule of Appendix C, the controls of Appendix D, and the driver.

## G. Price and stop rule

- **Measured in stage 0.** On e58/copy_s0_scaled and the planted items: seconds per patched forward for each of
  discovery, pruning, faithfulness and CMAP; group sizes; seconds per item.
- **Forward counts per state.** Discovery is 24 × (1 + 2 + 2) patched passes over the discovery rows. Pruning is at most
  Σ_G (|G|² + 2|G|) passes over the prune rows: up to 624 for one group of 24, about 300,000 prompt-forwards.
  Faithfulness is 1 + 10 passes. CMAP is about 15 passes over the scoring rows.
- **Estimate.** After stage 0, estimate the total: Σ over items of the stage times, with pruning scaled by each item's
  measured Σ(|G|² + 2|G|), known after its discovery.
- **Hard spend cutoff: $20 in total.** Before any paid shard launches, the runner adds up the cost of every paid shard.
  That covers provider time from the measured item, setup, sync and download (sizes ÷ 10 MB/s, as the root
  `CLAUDE.md`), and teardown, plus anything spent already. If the sum exceeds $20, no paid shard launches. Once pods
  run, they are terminated when spend reaches $20.
- **Stop rule.**
  - If the estimate exceeds 8 shards of 25 min of `debug`, use pods only if the estimated cost is ≤ $15, leaving
    headroom under the $20 cutoff. Otherwise stop and report the estimate to the lead.
  - After the first 4 items of stage 1, re-estimate. If the running time exceeds the estimate by more than 50%, stop
    and report.
  - An item whose discovery returns a pool over 16 heads in one group is priced before its pruning starts, and runs
    only if it fits the rule above.

## H. Revision history

### Changes in revision 3

- §3, A.2.3 and B.1: the scored groups call `apply_pp` unchanged, and the shared offset is agreed to be the
  V-composition site. The other edit site is a negative control only. The offsets are no longer a departure.
- §4.4 and D: the two-layer control is rebuilt as a fixed, non-zero sender delta at one site, traced through the
  receiver's `v_proj` output and both edit sites. There is no leak allowance.
- §4 and D: three tolerances, one per unit, and one gate in every section, with the case where the random null is 0.
- §3, B and E: E62's C is read at the preceding key, and its 800 saved train rows feed discovery and pruning. E53 uses
  its original scorer's train windows, or a split by source row.
- D: the planted-head recipe is pinned from E58's recipe as refit_96key filed it.
- G: a hard spend cutoff of $20 in total, checked before any paid shard launches.

### Changes in revision 2

- §1: "whole-group head CMAP"; the claim that the base model uses the heads is removed.
- §2 and A.2: three new discrepancies (shared offset, pruning helpers, 100 prompts) and the notebook's in-place `+=`.
- §3 and B: sender and receiver offsets separated; empty receiver groups skipped; what C's noise changes is stated.
- C: the paper-first minimality rule, with every edge case, and a matched random mean-ablation reference.
- §4 and D: new controls (hook floor, planted head through discovery, wrong head and position, two-layer path, output
  weights with a real gain, measured gain floor).
- E: frozen manifest. F: isolated checkout and adapter, no copies. G: per-stage price and stop rule.
- §5: the locator aside corrected; the faithfulness range split into own-circuit and cross-model.
- The LLaMA-7B port check is dropped and listed under Not run.

