# P2.68 step 1: method spec for whole-group head CMAP (after Prakash et al. 2024) on the 67M

Cell P2.68 (regating-lead, 3 runs, due 2026-11-25; cluster `debug` at $0, pods capped at $20). Step 1: spec only, nothing
coded or run. Revision 4 (2026-11-23) applies redteam-lead's verdict, "OK after fixes"
(`~/commute-data/agents/regating-lead/note-2026-11-23-redteam-p268-cmap-verdict.md`; pointer in
`reports/reviews/2026-11-23_review_p2-68-cmap-spec-redteam.md`), with the lead's decision on its major 2. Revisions 2 and
3 answered the GPT plan review and its re-checks (`reports/reviews/2026-11-23_review_p2-68-cmap-spec-gpt.md`). The
changes are listed in Appendix H.

Sources, read in full: the paper (Prakash, Rott Shaham, Haklay, Belinkov, Bau, ICLR 2024, arXiv 2402.14811, all 26
pages) and the released code, https://github.com/Nix07/finetuning at `6d6c0980bf073d305b915669c33533576a70aa77`
(2025-10-27). The paper links finetuning.baulab.info, which links this repo. The repo has no licence file.
Abbreviations: `pp` = `experiment_1/pp_utils.py`, `path` = `experiment_1/path_patching.py`, `cmap` =
`experiment_3/cmap_utils.py`, `nb` = `experiment_3/cmap.ipynb`, `data` = `data/data_utils.py`. Quotes and code
locations are in Appendix A.

**Label for the paper.** Whole-group head CMAP (cross-model activation patching after Prakash et al., 2024). On the same
input, the base model runs with the fine-tuned model's attention-head outputs, taken before the output weights, for
every head of a circuit. The circuit is found on the fine-tuned model by the paper's path patching and minimality
pruning. Groups are patched whole, without the paper's DCM step, and the result is reported as accuracy and as a
log-odds share of the gain. The neuron reading is named separately: "CMAP at neurons (our extension)" (§5, Appendix I).
Neither is called "Prakash et al.'s CMAP" without its qualifier.

## 1. In plain words

Cross-model activation patching runs the base model on a task input but gives chosen attention heads the outputs the
fine-tuned model computes on that same input, and counts how much of the fine-tuned model's gain the base model then
reaches. We will find each fine-tune's head circuit with the paper's path patching and pruning, then give the unchanged
67M model those heads' fine-tuned outputs, on the nine built pairs and the fifteen key-task fine-tunes. The result says
how much of each gain the fine-tuned outputs of a few heads carry into the base model. On its own it does not show that
the base model uses those heads: the paper showed that separately (§4-5), and we do not repeat it.

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

## 3. Our run

- **Models.** The base is W0, the released 4-block 67M with 6 heads of 128 per block. The fine-tune is W0 + c·δW, held as
  a second plain copy of the weights.
- **Rows.** The scoring rows of an item are the definition's rows for that item, D: the windows the definition's reading
  of R reads, at the same read position, with the same target (Appendix E). Discovery and pruning use the task's train
  windows with every D row removed. Prompts are cut to `half + 1` tokens, which is exact for a causal model and puts the
  read at the last position. Labels are read from the uncut window.
- **Noise (discovery only).** Another window of the same task supplies a new context. The task-carrying token is
  replaced by a seeded *other key of the task's pool*, the same kind of token, as the paper's random letter: the key at
  `half` in kv tasks and the second A at `half − 1` in E62. Each fine-tune records the mean probability of the clean
  label under the noise prompt beside W0's (Appendix D).
- **Groups** (Appendix B):
  - A: heads at the read into the logit.
  - B: heads at the read into A's queries.
  - C: heads at the position before the read into B's values. That position is the marker or context site in kv tasks
    and the preceding key in E62, the query label's analogue. E53 has no such site and runs A and B only. D is dropped.
  - The scored groups call the authors' `apply_pp` unchanged. Its shared `rel_pos` edits B's values at C's position,
    the site V-composition reads; both reviewers agree.
- **Selection.** A group's pool is every head whose mean score is below 0 by 2 standard errors over the discovery
  prompts. A group whose receiver group is empty is skipped and recorded. The pool is pruned by the paper-first
  minimality rule (Appendix C). Every pool and group size is reported.
- **The compared scalar.** The whole-circuit CMAP log-odds share, computed as `neuron_r.shares` computes a share: the
  mean numerator over the mean gain G on D, read at the read position with the definition's target. The numerator is
  log-odds with the circuit swapped minus W0's; G is the fine-tune's log-odds minus W0's. It is compared with the
  definition within a format only, because the circuit's groups differ by format.
- **The common column.** The A+B share, the same quantity with only groups A and B swapped, is reported for every item
  so that all formats share one column.
- **Beside them.** Per group: the paper's number, the patched accuracy on D against W0's and the fine-tune's, with the
  accuracy share; the 10 matched random sets; faithfulness. No share is clipped. A share is reported only where the
  item clears its task's gate (§4.6).

## 4. Controls, first and alone (stage 0; detail in Appendix D)

Every floor and tolerance is measured in this job, on the path it guards; none is guessed. Each check is judged in its
own unit: logits by a logit tolerance, accuracy by exact paired correct counts on the same rows, and log-odds by a
log-odds tolerance measured on the same hook path.

1. **Hook floor, per format** (kv, NEXT, LAST). Give W0 its own cached head outputs through the `cmap_out` einsum path,
   against a plain W0 forward. The largest |Δlogit| and |Δ log-odds| are that format's tolerances for CMAP checks. The
   paired correct counts must be identical.
2. **Do-nothing** (the control state times 0): the counts of patched (head, position) sites must be non-zero, and every
   unit must equal W0's within its tolerance.
3. **Planted head, through the whole pipeline.** One block-3 head's value rows are trained by the pinned recipe until the
   gain clears the gate. Discovery and pruning must return that head in A, and CMAP of A must give share 1. Two checks
   must give a numerator of 0: the wrong head and the wrong position. This tests A only.
4. **Two-layer path, on W0, no training.** It is a mechanics check of the hooks. A fixed, non-zero δ is added at one
   block-0 head at the marker, and the result is traced through the receiver's `v_proj` output and both edit sites. Each
   step is judged against the matched no-op floor of the same two passes run with δ = 0.
5. **Output-weights blind spot.** The same head with only its `o_proj` columns trained must clear the gate in the scored
   unit, and then give a CMAP numerator of 0.
6. **Gate, per task.** For each task and each scale it is read at, 15 random updates are drawn at that task's states'
   per-matrix norms times the scale, each read on the task's D. N is the largest |paired correct-count change| and L the
   largest |mean log-odds change|. An item clears the gate when its correct-count gain exceeds N (at least 1 if N = 0)
   and its log-odds gain G exceeds both L and `neuron_r`'s gain floor of 1 log-odds.

B and C discovery has only the mechanics check of §4.4: no planted head checks that path patching *finds* a B or C head.

## 5. Departures

From the paper and its code (who set each is in brackets):
1. **Setting** (ours). The paper reads arithmetic fine-tunes on a task they were not trained on, where the base already
   scores 0.66 with a working circuit. Ours are trained on the scored task, and W0 may be near 0. Where it is, the
   paper's "enhances an existing mechanism" reading does not carry over; the number is still CMAP.
2. **Model and task** (ours). A 4-block, 24-head model on learned key-to-value maps, not Llama-7B. The groups are
   renamed and D is dropped.
3. **The fine-tune is W0 + c·δW** (ours), a rescaled update, not a trained checkpoint.
4. **Heads only, base W_O** (theirs, followed). The fine-tune's change to `o_proj`, and every MLP change after the
   patched sites, is outside the swap by construction. This caps every share.
5. **No DCM** (ours). Groups are patched whole, and the paper's per-group subset numbers are not claimed as reproduced.
6. **Head pools** (ours): 2 standard errors below 0, not hand-set counts.
7. **Rows** (ours). Discovery uses 150-300 rows, pruning 151-500, and scoring D, against the paper's 300, 500 and 500.
   The rows are disjoint, whereas the paper's CMAP rows include its discovery rows. Where D is small, one prompt moves
   accuracy by 1/|D|.
8. **Exact counts** (ours), not the authors' rounding to 2 decimals.
9. **Noise token** (ours). It is another pool key; the paper uses a random capital letter, the same kind of token.
   Windows are cut at the read.
10. **p_org normalisation** (theirs; the consequence is ours). Where the fine-tune's probability of the target is small,
    likely on E53's word forms, (p_patch − p_org)/p_org is dominated by those rows.
11. **Minimality** (ours). It is written paper-first, not as the code's helpers (Appendix C), with our own means.
12. **E53 runs A and B only** (ours). E62's C is at the preceding key, and kv's C at the marker is a context site.
13. **Shares** (ours). The log-odds share is the compared scalar; accuracy and its share sit beside it.

**Neuron readings** (Appendix I). r_n may be called "CMAP at neurons (our extension: every MLP neuron at every
position, the fine-tune's firing through the base's output weights; log-odds share)". r_half is "CMAP of MLP inputs
(our extension)", the analogue of the authors' `cmap_in`. k_half is a count built from partial r_half swaps. The
top-neuron share and the locator's whole-MLP sufficiency are not activation swaps at CMAP's site, so neither carries
the label.

## 6. The authors' code

Their functions are imported unchanged from a hashed, read-only checkout at the commit, and none is copied into the
repo. An adapter, about 40 lines, exposes our blocks under their names. We call `apply_pp`, `patching_sender_heads`,
`patching_receiver_heads`, `get_receiver_layers`, `compute_topk_components` and `cmap_out`. Our code adds the prompt
builders, the minimality and faithfulness loop (their `mean_ablate` hard-codes entity-tracking positions), the
controls and the guards: one window length per batch, eval mode asserted. Setup is in Appendix F.

## 7. Expected values, price and stop rule

| number | W0 / do-nothing | planted head (A) | wrong head / position | random update | fine-tune |
|---|---|---|---|---|---|
| accuracy, patched | = W0's | = fine-tune's | = W0's | under the gate | not predicted |
| share, log-odds (compared) | numerator 0, share undefined | 1 | 0 | undefined (under gate) | not predicted |
| share, accuracy | undefined | 1 | 0 | undefined | not predicted |
| faithfulness | W0's own circuit | ≈ 1 | — | not read | paper own-circuit 0.99-1.0, cross-model 0.88-0.97 |

Each "=", 1 and 0 holds within the unit's tolerance of §4. Stage 0 times each stage on one key item and on the planted
items. The stop rule and the spend cutoff are in Appendix G: $20 in total, checked before any paid shard launches, with
pods terminated at $16.

## Not run

Nothing was coded or run in this step. Dropped: the port check on LLaMA-7B (both reviewers: it cannot test our
adapter, and its target number sits in Markdown from a partly run cell). No planted head tests B or C discovery. Not
read: the DCM training code beyond its loss (App. H) and the attention-knockout notebook (App. A).

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
  when there are no receiver layers (`:106`). `apply_pp` never calls `model.eval()`.
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
   so a fine-tune of a head's W_O alone is invisible (§5.4).
3. **One offset for sender and receiver.** `apply_pp` passes one `rel_pos` to both hooks (`path:apply_pp:84-122`). The
   paper names the path without naming the receiver's edit position. For every group the paper runs, the shared offset
   edits the site the composition reads. Pass 1 freezes every head from `last − rel_pos` onward, so the read's residual
   stays clean, and B's values at the sender's position are the only input a sender there can change. Both reviewers
   agree. A discrepancy of form only; the run uses it unchanged.
4. **Pruning helpers** (`pp:compute_pair_drop_values:940-1000`, `pp:get_head_significance_score:1003-1067`).
   - The skip test `layer_1 is not layer_2 and head_1 is not head_2` compares by identity, so a distinct pair sharing a
     head index is never removed together.
   - Entries with zero accuracy are dropped (`:996`), so the pair that collapses the circuit most is never ranked into K.
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

The scored groups call `path:apply_pp` unchanged, with the authors' `rel_pos` (prompts are cut, so the read is the last
position).

| group | `rel_pos` | sender at | receiver input edited | edited at |
|---|---|---|---|---|
| A | 0 | read | none (logit) | — |
| B | 0 | read | A's q | read |
| C | 1 | position before the read | B's v | position before the read |

The other site (B's v at the read position) is never scored; it is only the null arm of §4.4, run through the authors'
receiver hook with its `rel_pos` set to 0. If A (or B) is empty after its pool is formed, B (or C) is skipped and
recorded as "no receivers". The authors' code would silently score it as direct-to-logit (`path:apply_pp:106`).

**Pool rule.** For each sender (block, head), its per-prompt scores s_i over the n discovery prompts give a mean m and a
standard error se = sd / √n. The sender enters the pool if m < −2·se. Every pool's size is reported, with the size the
plain m < 0 rule would have given.

### B.2 What each group's noise changes

- **kv.** The key at the read becomes a seeded other key of the task's pool, and the context is another window's. The
  marker is the same token in every window and comes before the key. A head at the marker therefore differs between
  clean and noise only through the context. C is a context site, not a query-label analogue, and may come out empty,
  which is then a result.
- **E62** (`src/commute/copying.py:cue_windows:249-269`). The window reads "… A B … A M", with the second A at `half − 1`
  and the marker M at the read. In the noise prompt the second A becomes a seeded other key of the pool, so C's site
  changes directly. It is the analogue of the paper's query label.
- **E53.** The window is Pile text with the scored word form next. No position before the read carries the task, so
  there is no C. The noise is another window of the task.
- **Noise check.** For each fine-tune, the mean probability of the clean label under the noise prompt is recorded
  beside W0's. A value well above W0's means the answer can be read off the noise context, and that item's patching
  scores are flagged as weak.

### B.3 Formats

| format | tasks | clean label | noise | groups |
|---|---|---|---|---|
| kv | e46, e54, c3, e58 | `table[toks[:, half]]` from the uncut window | other window, key at `half` → seeded other pool key | A, B, C (marker, context site) |
| NEXT | e53 | `toks[:, half + 1]`, uncut window | other window of the task | A, B |
| LAST | e62 | the target column, never fed | other window, second A at `half − 1` → seeded other pool key | A, B, C (preceding key) |

## C. The paper-first minimality rule

1. **Nodes.** A node is (block, head, position). A head selected in both A and B at the read position is one node and
   is kept in B, as `pp:get_circuit:824-830`. The paper is silent.
2. **F(S).** F(S) is the exact count of correct argmax at the read position over the prune rows, divided by their number
   and not rounded. Every head output outside S, at every position, is replaced by its per-position mean over the noise
   prompts built from the prune rows. MLPs run.
3. **Order.** Groups are scored one at a time, each against the whole unpruned circuit Cir, as `minimality.py:131-188`.
4. **Ranking.** For v in group G and every distinct node v_j in G \ {v} (no pair skipped, zero accuracies kept):
   drop_j = F(Cir \ {v_j}) − F(Cir \ {v, v_j}). Rank by drop_j, descending, with ties broken by block then head.
5. **K.** K is the first m ranked nodes that have drop_j > 0, with m = ceil(0.3 × |G \ {v}|). If |G| = 1 or no drop is
   positive, K = ∅. The code's cap is ceil(0.3 × |G|), stated as a departure.
6. **Score.** num = F(Cir \ K) − F(Cir \ (K ∪ {v})), den = F(Cir \ (K ∪ {v})). Keep v if den > 0 and num/den ≥ 0.01,
   or if den = 0 and num > 0. Drop v if den = 0 and num = 0, or if num ≤ 0.
7. **Empty pools and groups.** An empty pool gives an empty group, and its dependants are skipped (B.1). A group that
   pruning empties is kept empty and reported.
8. **Faithfulness and reference.**
   - F(Cir), F(M) and the reference are read on the scoring rows D, not on the prune rows, with the same means.
   - The reference is 10 random circuits with the same number of nodes per position, by a seeded draw. It gives
     F(random)/F(M) beside F(Cir)/F(M), and the same 10 sets are swapped in CMAP.
   - Each random set's overlap with the circuit (shared nodes over circuit size) is reported. With 24 heads, a circuit
     holding most heads at the read leaves its random sets nearly identical to it.

## D. Controls in detail

- **Tolerances, per format.** The hook floor run patches every head at the read position and the position before it
  from W0's own cache through `cmap_out`'s einsum, against a plain W0 forward. It runs on D of one task per format:
  kv on e58/copy, NEXT on e53/lowercase, LAST on e62/copy. It gives that format's logit tolerance (largest |Δlogit|) and
  log-odds tolerance (largest |Δ log-odds| at the read). The paired correct counts must be identical. Each CMAP check
  uses its own format's tolerances.
- **Non-empty sites.** Every CMAP call logs the number of (head, position) entries it overwrote. Zero stops the job,
  except where the recorded group is empty.
- **Guards.** Every batch has one window length (asserted, because `cmap_out` counts positions from the padded length,
  `:344`). Both model copies are asserted in eval mode before every pass; `apply_pp` never calls `model.eval()`.
- **Planted head, pinned recipe.** It follows E58's fine-tune as refit_96key filed it
  (`experiments/refit_96key/results/predictions.json` `.settings.recipe`; `keydep.finetune`, `src/commute/keydep.py:100-104`).
  - `finetune.finetune_arm_a` with `ArmAConfig(o_token = e58/copy's table, lr = anchors.FT_LR = 3e-5, batch 32,
    kl_lambda 1, grad_clip 1, optimizer "adam", steps 300, checkpoints (10, 30, 100, 300), seed 0)`.
  - It runs on e58/copy's train windows, fp32, tf32 off.
  - Two additions are ours: `matrices = ("h.3.attn.v_proj",)`, and a gradient mask zeroing the rows outside h*'s 128.
    h* is a seeded block-3 head.
  - The read is the first checkpoint whose gain clears e58/copy's gate (§4.6). If none clears it, the control fails.
  - Nothing after block 3 reads other positions, so CMAP of any set holding (h*, read) equals the fine-tune.
  - **Pass:** h* is in A after pruning, and the share is 1 in every unit.
  - **Wrong head:** CMAP of A \ {h*}, and of one other seeded head, has numerator 0.
  - **Wrong position:** h* swapped only at the position before the read has numerator 0.
- **Two-layer path, on W0, no training,** on e58/copy's discovery rows.
  - The receiver r is the block-1 head with the largest W0 attention from the read to the marker, measured and
    recorded. The sender s is a seeded block-0 head.
  - **Sender cache.** It equals the clean cache everywhere, except that s's output at the marker is clean + δ. δ is one
    fixed seeded Gaussian vector, rescaled to the mean norm of s's clean output at the marker, so it is non-zero.
  - **Matched no-op floor.** The same two passes with δ = 0, against plain clean forwards, give one floor for each traced
    quantity: r's `v_proj` output at each position, r's read-position output, the logits and the log-odds. Pass 2 runs
    on `q/v_proj`, not on `cmap_out`'s einsum, so the CMAP tolerances are not used here.
  - **Pass one.** The authors' `patching_sender_heads` runs with s and `rel_pos` 1, tracing r's `v_proj` output. At the
    marker it must differ from clean by more than its no-op floor. At the read it must equal clean within its no-op
    floor.
  - **Marker edit.** The authors' `patching_receiver_heads` runs with `rel_pos` 1. r's read-position output (its slice
    of `o_proj`'s input) must change by more than its no-op floor.
  - **Read-site edit (null arm).** The same hook runs with `rel_pos` 0. Four things must match clean, each within its
    matched no-op floor: r's read-position output, the logits and the log-odds, plus the paired correct counts, which
    must match exactly.
  - **What it does not test.** It checks the hooks, not that discovery *finds* a B or C head (§4).
- **Output weights only.** h* is trained with `matrices = ("h.3.attn.o_proj",)` and the mask on columns outside h*'s
  128, by the same recipe. It must clear the gate in the scored unit, or the control fails. Pass: the CMAP numerator is
  0 in every unit.
- **Gate, per task and scale.**
  - For every task and every scale c it is read at, 15 random directions are drawn by `finetune.random_direction`
    (E237's construction). They run at the per-matrix norms of that task's states, cycling the states, seed
    2026112368 + index, times c. Each is read once on the task's D, with no circuit work.
  - N is the largest |paired correct-count change| and L the largest |mean log-odds change|.
  - An item clears the gate if its correct-count gain exceeds N (at least 1 if N = 0), and its log-odds gain G
    exceeds L and 1 log-odds (`neuron_r`'s gain floor, from VPD's `MIN_GAIN`).
- **Stops.** Any control missed stops the job before stage 1.

## E. Frozen manifest

- **Key tasks.** E237's 15 items `e58/{copy,rank1,elicit,cipher,prior_cipher}_s{0,1,2}_scaled`.
  - Updates are `experiments/refit_96key/results/large/dW_<task>_s<seed>.pt` times `.c[<task>]` from
    `experiments/key_grid/results/stage0.json` (copy 1, rank1 0.9201, elicit 0.8707, cipher 0.7520, prior_cipher 0.7798).
  - E237's own 15 `random_<state>` items (`World.random`, seed 237001 + the state's index) are generated, not saved.
    They are read beside the per-task gate sets of D.
  - Train windows are `experiments/refit_96key/results/inventory.json` `.maps.<task>.windows` (1214 train rows each).
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
- **D, the definition's rows.** For every item, D is the window set and read position that E248's reading of R takes:
  the definition's `World.make`. For key items that is `experiments/kstar_vpd_wording_a/run.py:145-159`, via
  `experiments/key_grid/run.py:141-146`. For pair items it is `experiments/kstar_size_matched/run.py:74-83`. Both
  build windows with `read(read_half187(task))` (`experiments/kstar_pairs/run.py:154-161`), read at the last position
  (`LAST`). The target is the format's label, in log-odds. Stage 0 records |D| and D's sha256 per task, checked against
  each item's `results/items/<item>.json` `n_windows` in `experiments/kstar_vpd_wording_a/`.
- **Train windows for discovery and pruning,** with every row equal to a D row removed (the count is reported).
  - c3: 301 rows. e46 and e54: 403. e58: 1214.
  - E62: 800 rows, with targets `target[arm]["train"]` in `experiments/induction_cue/results/large/windows.pt`
    (saved at `experiments/induction_cue/run.py:98-107`).
  - E53: the original scorer's train windows, `Run.task`'s `t.train` (train rows 0-255, `CFG.n_write_windows` = 2000,
    `experiments/performed_anchor/run.py:327-334`). If they cannot be rebuilt, the train windows come from source rows
    disjoint from D's, and the reason is reported.
- **Row split, seed 2026112368, per task.** Discovery rows are min(300, ⌊n_train / 2⌋) train rows by seeded permutation.
  Prune rows are the next min(500, the rest). Noise rows and replacement keys come from the same seed.
- **Do-nothing.** E237's control state times 0.

## F. The authors' code: checkout and adapter

- **Checkout.** A fresh `git clone` at the commit into `~/commute-data/vendor/nix07-finetuning-6d6c098/`, read-only,
  outside the repo. The sha256 of `experiment_1/pp_utils.py`, `experiment_1/path_patching.py` and
  `experiment_3/cmap_utils.py` is recorded in the plan's predictions file. The adapter checks the hashes at start and
  stops on a mismatch. Nothing of theirs is committed, and the repo has no licence to permit it.
- **Loading.** The adapter loads those modules with `importlib.util.spec_from_file_location`. Their module-level
  `sys.path.append` is theirs; our code does no `sys.path` editing.
- **Dependencies.** baukit at a pinned git commit, einops, peft and datasets, so their module-level imports resolve.
  They are pinned in a separate `cmap` dependency group at versions that resolve with our lock. Their conda environment
  is not needed.
- **Adapter.** `model.layers.{l}.self_attn.{q,k,v,o}_proj` aliases our `h.{l}.attn.*` modules, and
  `config = (num_attention_heads 6, num_hidden_layers 4, architectures ["LlamaForCausalLM"])`. The forward returns
  `.logits`. `state_dict()` holds the aliased keys of a real weight copy (W0, or W0 + c·δW), so their W_O reads are
  true. The 67M has 6 kv heads and no biases (`artifacts/vpd/t-9d2b8f02/model_config.yaml`), so `cmap_out`'s bias-free
  einsum and per-head `v_proj` slicing are exact.
- **Our code.** The B.3 prompt builders, the B.1 pool rule, the rule of Appendix C, the controls and guards of Appendix
  D, the gate sets, and the driver.

## G. Price and stop rule

- **Measured in stage 0.** On e58/copy_s0_scaled and the planted items: seconds per patched forward for each of
  discovery, pruning, faithfulness and CMAP; group sizes; seconds per item.
- **Forward counts per state.** Discovery is 24 × (1 + 2 + 2) patched passes over the discovery rows. Pruning is at most
  Σ_G (|G|² + 2|G|) passes over the prune rows: up to 624 for one group of 24, about 300,000 prompt-forwards.
  Faithfulness is 1 + 10 passes. CMAP is about 15 passes over D. A gate set is 15 plain forwards over D.
- **Estimate.** After stage 0, estimate the total: Σ over items of the stage times, with pruning scaled by each item's
  measured Σ(|G|² + 2|G|), known after its discovery.
- **Hard spend cutoff: $20 in total.** Before any paid shard launches, the runner adds up the cost of every paid shard.
  That covers provider time from the measured item, setup, sync and download (sizes ÷ 10 MB/s, as the root
  `CLAUDE.md`), and teardown, plus anything spent already. If the sum exceeds $20, no paid shard launches. Once pods
  run, the runner terminates every pod when spend reaches $16 (GPT re-check 2: a cutoff at $20 is too late, because
  teardown and lag in reading spend still bill). The $4 reserve is about 10 min of four H100s, which covers teardown
  and a delayed read. If the runner cannot enforce that deadline, only free cluster shards run.
- **Stop rule.**
  - If the estimate exceeds 8 shards of 25 min of `debug`, use pods only if the estimated cost is ≤ $15, leaving
    headroom under the $20 cutoff. Otherwise stop and report the estimate to the lead.
  - After the first 4 items of stage 1, re-estimate. If the running time exceeds the estimate by more than 50%, stop
    and report.
  - An item whose discovery returns a pool over 16 heads in one group is priced before its pruning starts, and runs
    only if it fits the rule above.

## H. Revision history

### Changes in revision 4 (redteam verdict)

The numbers match the list in the hand-back.

1. Gate per task and scale, on the task's D, with 15 random directions each (§4.6, D).
2. The compared scalar named: the whole-circuit CMAP log-odds share, as `neuron_r.shares`, compared within a format.
   The A+B share is reported for every item as a common column (§3). The scoring rows are the definition's D (§3, E).
3. Departures 1, 3, 4, 7, 8, 9 and 10 added to §5, with who set each.
4. The null arm of the two-layer check is judged by its matched δ = 0 floor (D).
5. The hook floor is measured per format (§4.1, D).
6. Noise is a seeded other pool key, with a noise-prompt check (§3, B.2, B.3).
7. r_half and k_half are named in the relabel (§5, I).
8. Faithfulness rows named; each random set's overlap with the circuit reported (C.8).
9. Guards: one window length per batch, eval mode asserted (§6, D).
10. B and C discovery has only a mechanics check, now stated (§4, D, Not run).
11. Pool threshold of 2 standard errors below 0, with every pool size reported (§3, B.1).
12. Paper label and neuron label adopted (top, §5, I). The open questions closed with redteam's answers (a)-(h). A.2.4
    gains the zero-accuracy clause; A.2.3 gains the pass-1 argument.

### Changes in revision 3

- §3, A.2.3 and B.1: the scored groups call `apply_pp` unchanged, and the shared offset is agreed to be the
  V-composition site. The other edit site is a negative control only. The offsets are no longer a departure.
- §4.4 and D: the two-layer control is rebuilt as a fixed, non-zero sender delta at one site, traced through the
  receiver's `v_proj` output and both edit sites. There is no leak allowance.
- §4 and D: three tolerances, one per unit, and one gate in every section, with the case where the random null is 0.
- §3, B and E: E62's C is read at the preceding key, and its 800 saved train rows feed discovery and pruning. E53 uses
  its original scorer's train windows, or a split by source row.
- D: the planted-head recipe is pinned from E58's recipe as refit_96key filed it.
- G: a hard spend cutoff of $20 in total, checked before any paid shard launches (pods terminated at $16, re-check 2).

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

## I. The neuron readings and the CMAP label

- **r_n** (`src/commute/neuron_r.py:batch_cells`, cell L01). All 24 matrices stay at W0, and every block's post-GELU
  firing (`down_proj`'s input) at every position is replaced by the fine-tuned model's on the same window. It is a
  same-input swap before the output weights, as `cmap_out` swaps before `o_proj`. Its label is "CMAP at neurons (our
  extension: every MLP neuron at every position, the fine-tune's firing through the base's output weights; log-odds
  share)". It departs from the paper in four ways: MLPs, which the paper never patches; no circuit; every position;
  and the log-odds share.
- **r_half** (cell Lh). W0's `c_fc` and GELU re-fire on the fine-tuned model's MLP input, at every block and position.
  Its label is "CMAP of MLP inputs (our extension)", the analogue of the authors' `cmap_in`, not of `cmap_out`.
- **k_half.** The fewest top-ranked neurons whose re-fired values carry half the gain, a count from partial r_half
  swaps. It may be described as built on CMAP of MLP inputs (our extension), never as Prakash et al.'s CMAP.
- **Top-neuron share** (`src/commute/masked_reads.py:topk_logits`). W0 keeps its own firing, and only the choice of
  which k neurons survive comes from the fine-tune. It is not an activation swap and carries no CMAP label.
- **Locator "sufficiency"** (`src/commute/locator.py:reads`, line 279).
  - For a head it swaps `o_proj`'s input, which is CMAP's site.
  - For a whole MLP it swaps `down_proj`'s output, so the fine-tuned output weights come along. That is neither CMAP's
    site nor r_n's.
  - It is read one part at a time, at every position, with no circuit, in log-odds and the label gap.
