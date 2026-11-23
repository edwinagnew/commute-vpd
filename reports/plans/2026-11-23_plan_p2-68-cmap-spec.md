# P2.68 step 1: method spec for whole-group head CMAP (after Prakash et al. 2024) on the 67M

Cell P2.68 (regating-lead, 3 runs, due 2026-11-25; cluster `debug` at $0, pods capped at $20). Step 1: spec only, nothing
coded or run. Revision 2 (2026-11-23) answers the GPT plan review, `reports/reviews/2026-11-23_review_p2-68-cmap-spec-gpt.md`,
with the lead's seven decisions. The changes are listed at the end. Redteam-lead's review still comes before any run.

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
  the key, so a head at the marker sees only the changed context (Appendix B.2).
- **Groups.** A: heads at the read position into the logit. B: heads at the read position into A's queries. C (kv
  formats only): heads at the marker into B's values, read at the marker. D is dropped. Sender and receiver offsets are
  separate parameters in our adapter (Appendix B.1). A group whose receiver group is empty is skipped and recorded,
  never run as direct-to-logit.
- **Selection.** The pool is every head with a negative mean score. It is then pruned by the paper-first minimality rule
  written out in Appendix C. Faithfulness is read against 10 random circuits matched per position.
- **CMAP.** Each group whole, then the whole circuit, then the 10 matched random sets, from the fine-tune into W0, on
  the scoring rows.
- **Read.** Accuracy, as the paper does. Beside it, the share (acc_patched − acc_W0)/(acc_ft − acc_W0) and the
  project's log-odds share, neither clipped: zero, negative and over-1 values are reported as read. A share is reported
  only where the fine-tune's top-1 gain clears the stage-0 floor (§4).
- **Formats.** kv tasks run A, B and C. E53 (Pile next token) and E62 (induction, label in the last column) have no
  marker or key, so only A and B run there, a named departure. The manifest of states, scales, row seeds and the
  split between discovery, pruning and scoring is frozen in Appendix E.

## 4. Controls, first and alone (stage 0; detail in Appendix D)

Every floor and tolerance is measured in this job; none is guessed.

1. **Hook floor.** Give W0 its own cached head outputs through the `cmap_out` einsum path. The largest |Δlogit| against a
   plain W0 forward is the tolerance for every "exact" check below.
2. **Do-nothing** (the control state times 0): the counts of patched (head, position) sites must be non-zero, and the
   accuracy and log-odds must equal W0's within the hook floor.
3. **Planted head, through the whole pipeline.** Train only one block-3 head's value rows on a task until its top-1
   gain clears the floor. Discovery and pruning must return that head in A. CMAP of A must give share 1 within the hook
   floor. Two checks must give a numerator of 0 within the floor: the wrong head (A without it, and a random other
   head) and the wrong position (the same head swapped at the marker).
4. **Two-layer path.** A rank-one change to one block-0 head's value rows, reading the marker's input direction, is
   trained to raise the target through a block-1 head that attends from the read position to the marker. With the
   receiver's values edited at the marker, discovery must rank the block-0 head first in C. Edited at the read
   position, its score must sit within its measured leak. The test tells the two receiver positions apart.
5. **Output-weights blind spot.** The same block-3 head with only its `o_proj` columns trained, to a top-1 gain above the
   floor: the CMAP numerator must be 0 within the hook floor.
6. **Gain floor.** The floor is the largest |acc − acc_W0| over the 15 random updates (E237's generated `random_<state>`
   items). A share needs acc_ft − acc_W0 above it.

## 5. Departures

From the paper and code:
1. The model and task differ. Ours is a 4-block, 24-head model on learned key-to-value maps, not Llama-7B tracking
   entities in context. Groups are renamed and D is dropped.
2. Head pools come from negative scores, not hand-set counts.
3. There is no DCM, so groups are patched whole ("whole-group head CMAP").
4. The noise key is a non-key token, and windows are cut at the read.
5. Discovery, pruning and scoring rows are disjoint. In the paper's code the CMAP rows include the discovery rows.
6. The minimality rule is written paper-first, not as the code's helpers (Appendix C). The means are ours.
7. Sender and receiver offsets are separate.
8. E53 and E62 run A and B only.
9. A normalised share and a log-odds share are reported beside accuracy.

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
`patching_receiver_heads`, `get_receiver_layers`, `compute_topk_components` and `cmap_out`. Our code adds three pieces:
the prompt builders, the separate receiver offset, and the minimality and faithfulness loop. Their `mean_ablate` hard-codes
entity-tracking positions. Setup is in Appendix F.

## 7. Expected values, price and stop rule

| number | W0 / do-nothing | planted head (A) | wrong head / position | random update | fine-tune |
|---|---|---|---|---|---|
| accuracy, patched | = W0's | = fine-tune's | = W0's | within the floor of W0's | not predicted |
| share, accuracy | undefined | 1 | 0 | undefined (under floor) | not predicted |
| share, log-odds | numerator 0 | 1 | 0 | undefined | not predicted |
| faithfulness | W0's own circuit | ≈ 1 | — | not read | paper own-circuit 0.99-1.0, cross-model 0.88-0.97 |

Each "=", 1 and 0 holds within the hook floor of §4.1. Stage 0 times each stage on one key item and on the planted
items, recording group sizes. Each later item is priced as stage time × items, with pruning scaled by the item's
measured g² + 2g. The stop rule and bounds are in Appendix G.

## Open questions for redteam-lead

1. Is the code's shared offset a bug or the correct V-composition site? I read it as correct (B's values at C's position
   are what V-composition reads), so separate offsets set by composition give the code's own edits. The two-layer test
   (§4.4) settles it on the 67M.
2. Is a non-key noise token the right analogue for answers that are learned, not read from context?
3. Is the C group worth running when its noise is only the context?

## Not run

Nothing was coded or run in this step. Dropped: the port check on LLaMA-7B in the paper's own setting (cost and setup
risk; it would pin checkpoint, tokenizer, prompts and the released masks and compare raw counts). Not read: the DCM
training code beyond its loss (App. H) and the attention-knockout notebook (App. A).

## Changes in revision 2

- §1: "whole-group head CMAP"; the claim that the base model uses the heads is removed.
- §2 and A.2: three new discrepancies (shared offset, pruning helpers, 100 prompts) and the notebook's in-place `+=`.
- §3 and B: sender and receiver offsets separated; empty receiver groups skipped; what C's noise changes is stated.
- C: the paper-first minimality rule, with every edge case, and a matched random mean-ablation reference.
- §4 and D: new controls (hook floor, planted head through discovery, wrong head and position, two-layer path, output
  weights with a real gain, measured gain floor).
- E: frozen manifest. F: isolated checkout and adapter, no copies. G: per-stage price and stop rule.
- §5: the locator aside corrected; the faithfulness range split into own-circuit and cross-model.
- The LLaMA-7B port check is dropped and listed under Not run.

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
   paper names the path without naming the receiver's edit position. My reading: for every group the paper runs, the
   shared offset edits the site V-composition reads. That is the receiver's values at the sender's position, or its
   query at its own position for A from B. The reviewer reads it as an error. It is filed here as a discrepancy of form,
   and settled empirically by §4.4.
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

### B.1 Offsets and the empty-receiver rule

The adapter passes the sender's offset and the receiver's edit offset separately, and wraps the authors' two hooks
unchanged by giving each hook its own `rel_pos`.

| group | sender position | receiver input edited | edited at |
|---|---|---|---|
| A | read | none (logit) | — |
| B | read | A's q | read |
| C | marker | B's v | marker |

The C row is the V-composition site: B, at the read position, reads its values at the marker. If A (or B) is empty
after its pool is formed, B (or C) is skipped and recorded as "no receivers". The authors' code would silently score it
as direct-to-logit (`path:apply_pp:106`).

### B.2 What each group's noise changes

The read position's key is a non-key token, and the context is another window's. The marker token is the same in every
window and comes before the key. A head at the marker therefore differs between clean and noise only through the
context. C measures heads whose marker output depends on the context and matters to B. It may come out empty, which is
then a result.

### B.3 Formats

| format | tasks | clean label | noise | groups |
|---|---|---|---|---|
| kv | e46, e54, c3, e58 | `table[toks[:, half]]` from the uncut window | other window, key at `half` → seeded non-key | A, B, C |
| NEXT | e53 | `toks[:, half + 1]`, uncut window | other window of the task | A, B |
| LAST | e62 | last column, never fed | other window of the task | A, B |

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

- **Hook floor.** Patch every head at the read and marker positions from W0's own cache through `cmap_out`'s einsum. The
  floor is the largest |Δlogit| against a plain W0 forward, on the scoring rows of one kv task.
- **Non-empty sites.** Every CMAP call logs the number of (head, position) entries it overwrote. Zero stops the job,
  except where the recorded group is empty.
- **Planted head.** Choose the block-3 head h* by seeded draw. Train only its `v_proj` rows (Adam, the e58/copy recipe
  settings from E58's predictions file) on the e58/copy train rows. Stop at the first step where the top-1 gain clears
  the D-floor below by 2 times. Nothing after block 3 reads other positions, so CMAP of any set holding (h*, read)
  equals the fine-tune exactly.
  - **Pass:** h* is in A after pruning; share 1 within the hook floor.
  - **Wrong head:** CMAP of A \ {h*}, and of one other random head, has numerator 0 within the hook floor.
  - **Wrong position:** h* swapped at the marker only has numerator 0 within the hook floor.
- **Two-layer path.**
  - Pick the receiver r: the block-1 head with the largest W0 attention from the read position to the marker, measured.
  - Train a rank-one change u wᵀ to one seeded block-0 head's `v_proj`. Here w is the marker's block-0 normed input,
    fixed, and only u is trained.
  - The leak is the change of that head's output at the read position, relative to its change at the marker, measured
    directly.
  - **Pass:** with r's values edited at the marker, the block-0 head ranks first in C's pool. With them edited at the
    read position, its score is within leak × its marker score.
- **Output weights only.** Train only h*'s `o_proj` columns to a top-1 gain above the floor. If it does not reach the
  floor in the step budget, the control is failed, not passed. Pass: CMAP numerator 0 within the hook floor.
- **Gain floor.** The largest |acc − acc_W0| over the 15 random updates on their own scoring rows. The same rule in
  log-odds sets the log-odds floor.
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
  - Held-out rows: c3 75; e46 and e54 97; e62 200 (read windows); e53 counted at stage 0.
- **Row split, seed 2026112368, per task.**
  - Scoring rows are all held-out rows, the same rows for every state of the task.
  - Discovery rows are min(300, ⌊n_train / 2⌋) train rows by seeded permutation.
  - Prune rows are the next min(500, the rest) train rows.
  - Noise rows and non-key tokens come from the same seed.
  - Where train rows are absent (e53), the held-out rows are split 1:1:1 and the split is reported.
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
- **Our code.** The B.3 prompt builders, the separate-offset wrapper (B.1), the rule of Appendix C, and the driver.

## G. Price and stop rule

- **Measured in stage 0.** On e58/copy_s0_scaled and the planted items: seconds per patched forward for each of
  discovery, pruning, faithfulness and CMAP; group sizes; seconds per item.
- **Forward counts per state.** Discovery is 24 × (1 + 2 + 2) patched passes over the discovery rows. Pruning is at most
  Σ_G (|G|² + 2|G|) passes over the prune rows: up to 624 for one group of 24, about 300,000 prompt-forwards.
  Faithfulness is 1 + 10 passes. CMAP is about 15 passes over the scoring rows.
- **Estimate.** After stage 0, estimate the total: Σ over items of the stage times, with pruning scaled by each item's
  measured Σ(|G|² + 2|G|), known after its discovery.
- **Stop rule.**
  - If the estimate exceeds 8 shards of 25 min of `debug`, use pods only if the estimated cost is ≤ $15, leaving
    headroom under the cell's $20 cap. Otherwise stop and report the estimate to the lead.
  - After the first 4 items of stage 1, re-estimate. If the running time exceeds the estimate by more than 50%, stop
    and report.
  - An item whose discovery returns a pool over 16 heads in one group is priced before its pruning starts, and runs
    only if it fits the rule above.
