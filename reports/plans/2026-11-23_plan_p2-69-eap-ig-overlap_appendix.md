# P2.69 appendix: step table with quotes, items and states, adapter

*Companion to `2026-11-23_plan_p2-69-eap-ig-overlap.md`. Paper quotes are from the arXiv v2 PDFs (page numbers are the
PDF's). Code quotes are from the clones in `artifacts/p269_src/` at the commits in the main file. "W" is Wang et al.
2025, "H" is Hanna et al. 2024, "FC" is FinetuneCircuits at `fdaeeaa`, "EAP" is hannamw/EAP-IG, "HF" is
eap-ig-faithfulness at `d42fea6`. D-numbers point to the main file's departures list.*

## A. Step table

| Step | Published choice, verbatim, with location | Ours |
|---|---|---|
| Graph and nodes | H §2 p.2: "We define this digraph's nodes to be the LM's attention heads and MLPs ... Edges specify where a node's output goes; a node v's input is the sum of the outputs of all nodes u with an edge to v." EAP `graph.py` `from_model`: one input node (`hook_embed`), heads, MLPs, logits; each head has separate q, k, v inputs | The same graph: input, 24 heads, 4 MLPs, logits; 927 edges (828 into heads' q, k, v, 70 into MLPs, 29 into the logits), as counted in `reports/reviews/2026-11-05_review_e97-eapig-scoping.md` |
| Scoring level | W §3.1 p.3: "The formula for scoring each edge is: ΔL(E) ≈ \|(e_corr − e_clean)ᵀ (1/m) Σ_{k=1}^{m} ∇_{e_k} L(x)\|". FC `100_2arithmetic_opeartions_circuit.ipynb`: `attribute(model, g, dataloader, partial(kl_divergence, loss=True, mean=True), method='EAP-IG', ig_steps=5)` | Edges scored; nodes are the ends of kept edges, as in W |
| IG form and steps | H Eq. 3 p.4, sum over k = 1..m, embeddings interpolated "z′ + (k/m)(z − z′)"; H §3 p.5: "we select m = 5 and justify this in Appendix C". W §3.1 p.3: "we choose m = 5 based on Hanna et al.'s (2024) recommendations". EAP `aa8cdc8` and FC `eap/attribute.py`: `for step in range(1, steps+1)` | m = 5, inputs interpolated, k = 1..5, at `aa8cdc8` (D1). HEAD `e24bafd` has `for step in range(0, steps)` |
| Metric | W §3.2 p.3: "We adopt Kullback-Leibler divergence (KL-divergence) as the metric, following Conmy et al." FC: `F.kl_div(probs.log(), clean_probs.log(), log_target=True, reduction='none').mean(-1)`, read at `input_length - 1`. H §4.2 p.6 uses task logit or probability differences, and "we also run EAP-IG and activation patching with KL divergence as the loss"; HF `metrics.py` has the same KL line | W's KL, as in FC, unchanged: KL of the clean run against the patched run, averaged over the vocabulary (a constant scale that changes no ranking). The read is the last token (D4) |
| Task data | W §3.3 p.3-4: "For each task ... 80% of the dataset is allocated for fine-tuning, and the remaining 20% is reserved for identifying circuits". FC: `ds.to_dataloader(32)` | Each item's own held-out read windows (section B), batch 32, the same windows for M and F |
| Corrupted inputs | H §2 p.2: "Corrupted inputs fit the same task schema but elicit distinct output". W §3.3: "Corrupted data involves altering the arithmetic operation"; Sequence: "altering one term in the sequence"; Function: "altering the constant term". EAP: "Number of positions must match" | One token swapped, the one that decides the answer, for another of the same pool, same length (D5; per item in section B) |
| Circuit selection and size | W §4.1 p.4: "Prior work (Hanna et al., 2024) achieved over 85% faithfulness by selecting 1–2% of edges. Given our more complex tasks and larger model, we select 5% of edges to ensure reliable circuits (faithfulness >70%)." FC: `five_percent_edges = int(total_edges * 0.05)`; `g.apply_topn(five_percent_edges , absolute=True)`; `g.prune_dead_nodes()` | Primary: top-n by absolute score, n = int(0.05 × 927) = 46, then the same pruning. The circuit after pruning may hold fewer edges; its count is reported |
| Secondary selection | H §4.2 p.6: "for n = 30, 40, ..., 100, 200, ..., 1000, we select a circuit of n edges using a greedy search procedure"; "We aim to find circuits that are small (containing 1-2% of edges) and faithful (recovering ≥85% of model performance)"; footnote 7: "we select edges based on absolute scores". H §5 p.7: "we take the smallest circuit we have found that achieves at least 85% faithfulness". H App. E: the greedy algorithm. HF `pareto.py`: `graph.apply_greedy(i, absolute=True)` | `apply_greedy` on H's ladder up to 927; the smallest n with normalised faithfulness of at least 0.85; "none faithful" when no n reaches it |
| Pruning | H §4.2 p.7: "we (recursively) prune all nodes and edges that have no children, or no parents". FC: `prune_dead_nodes()` | Same call |
| Faithfulness (primary) | W §3.2 p.3: "Faithfulness = (1 − \|F(M) − F(C)\|/F(M)) × 100%". FC `calculate_faithfulness`: `percentage_performance = (1 - faithfulness / baseline_performance) * 100` with `partial(kl_divergence, loss=False, mean=False)`; `evaluate_baseline` and `evaluate_graph` call `metric(logits, corrupted_logits, ...)`, so F(M) is the full model's KL from its corrupted run and F(C) the circuit's | As in FC; bar 70% from W §4.1. A circuit under it is flagged "not faithful", its overlap still reported |
| Faithfulness (secondary) | H §4.2 p.7: "we normalize the circuit's performance m as (m − b′)/(b − b′)", b and b′ the whole model on clean and corrupted inputs | Used only for the secondary selection |
| Overlap measure | W §4.2 p.5: "ΔS = (1/n) Σ_{t=0}^{n−1} Δs_{t→t+1}/S₀ × 100%, where Δs_{t→t+1} denotes the number of nodes or edges that change from checkpoint t to checkpoint t+1, and S₀ denotes the total number of nodes or edges in the initial circuit." FC `ft_node.ipynb`: `changes.append(added + removed)`; `ft_edge.ipynb`: `if prev_in_graph != curr_in_graph: change_count += 1`. H §5 p.7: "the intersection over union (IoU, also called Jaccard similarity) between the circuits' nodes, and the IoU between their edges"; App. H: hypergeometric test | Both, from the same three counts: W's ΔS for nodes and edges with n = 1 (D6), and H's node and edge IoU with the hypergeometric p (D8, D9) |
| Reused and new | W Fig. 3 caption p.6: "Nodes and edges are color-coded based on their status: added (blue), removed (red), or unchanged (grey/white)." FC `before_after_circuit.ipynb` `compute_differences`: `added_edges = edges_final_set - edges_base_set`, `removed_edges = edges_base_set - edges_final_set` | "Reused" = unchanged, in both circuits; "new" = added, in F's circuit only; "removed" = in M's only. W's 95th-percentile re-threshold (§4.3 p.5, FC `compute_dynamic_threshold(edges, percentile=95)`) is not used (D7) |
| Noise floor | W §3.2 p.3: "Robustness(p) = J_E(G₀, G_p)" on perturbed datasets; §4.1 p.4: "we perturb the original dataset by 10% to 90%" | M's circuit on two random halves of each task's windows, 5 splits, edge and node IoU between halves (section C) |
| Precision | FC: `HookedTransformer.from_pretrained(..., device='cuda', hf_model=hf_model)`, default fp32 | fp32, TF32 off |

## B. Items, windows, corrupted inputs and saved states

Every path below was checked with `ls -laL` on 2026-11-23. "Built" items are made by code from the original weights
at run time and have no state file. Their recipe and its record are listed instead.

| Item | F | Windows (read position) | Corrupted input | State and sha256 |
|---|---|---|---|---|
| x2, x10 | layer-1 `down_proj` columns [486, 499, 613, 778, 1125, 2028, 2352, 2960] times 2 or 10 (`experiments/rule_retention/run.py` `rescaled`, E195's C, settings in `experiments/kstar_levels/results/predictions.json`) | copy24 held-out, `experiments/open_pair/results/large/amend1/windows_copy_trained.pt` (537 KB), cut after `half` | the key at `half` swapped for another of the 24 keys | built; no file |
| rebuilt | E208's `rebuilt(M, old)` with the same layer and C (`experiments/rho_saved_items/run.py`) | copy24 held-out, as x2 | as x2 | built; no file |
| new_few, new_thin | `commute.known_edits` recipes, slots and gains recorded in `experiments/kstar_levels/results/edits.json` (that file has an uncommitted change in the tree: read the committed version) | P1.31's cue windows, A′ᵢ last (`known_edits` docstring) | A′ᵢ swapped for another cue A′ⱼ | built; no file |
| random_size_241001, random_kl_241002 | E242's two Gaussian updates (`experiments/kstar_vpd_wording_a/run.py`; seeds 241001, 241002; sized to new_thin, KL-matched to new_few) | x2's windows (E242 setting `random_windows: x2`) | as x2 | built from seed |
| E62 copy, lookup, seeds 0-2 | `experiments/induction_cue/results/large/{copy,lookup}_s{0,1,2}.pt` (113 MB each) | `experiments/induction_cue/results/large/windows.pt` (370 KB; sha256 `490e5f74e7b6...` in `results/run.json`), 8 tokens, the marker last | the second A swapped for another key A′, the same for both maps | copy_s0 `759fb83995d6`, copy_s1 `1f1f69b2d53c`, copy_s2 `cab79309c189`, lookup_s0 `c74c074b70fa`, lookup_s1 `5e10359fa75e`, lookup_s2 `397924a82ad9` (full hashes in `experiments/induction_cue/results/run.json`) |
| E62 random | Gaussian at copy_s0's per-matrix norms, `commute.finetune.random_direction` as E237 uses it, seed filed with the run | as E62 | as E62 | built from seed (new; no saved E62 random exists) |
| Key grid, five tasks × seeds 0-2, scaled | `experiments/refit_96key/results/large/dW_<task>_s<k>.pt` (57 MB each) times E237's c: copy 1, rank1 0.9201, elicit 0.8707, cipher 0.7520, prior_cipher 0.7798 (`experiments/key_grid/results/stage0.json`) | `experiments/open_pair/results/large/amend3/windows_<task>_trained.pt`; prior_cipher in `amend6/` (all present), cut after the key position | the key swapped for another of the 96 | sha256 per state in `experiments/refit_96key/results/stage0.json` → `states.e58/<task>_s<k>.sha256` (e.g. copy_s0 `0a515d3ba390`) |
| Key grid randoms | E237's `random_<state>` (`FT.random_direction`, seed E237's `random_seed` plus the state's index), times the same c | as the task | as the task | built from seed |
| M | the VPD target model, `artifacts/vpd/t-9d2b8f02/ckpt/` via `commute.vpd.checkpoint_weights` | each task above | each task above | E62's and E237's runs record their own loads |

The stage-0 job writes each state file's sha256 and stops when one differs from its record.

## C. Adapter, floor and the readings per item

**Conversion (D2).** A `HookedTransformerConfig` with n_layers 4, d_model 768, n_heads 6, d_head 128, d_mlp 3072,
d_vocab 50277, n_ctx 512, `act_fn="gelu_new"` (the tanh form, as `NewGELU`), `normalization_type="RMS"`, eps 1e-6,
`positional_embedding_type="rotary"`, rotary_dim 128, rotary_base 10000, `rotary_adjacent_pairs=False`,
`n_key_value_heads=None` (6 key and value heads for 6 query heads), `default_prepend_bos=False`. The weights are W0 + dW
mapped onto TransformerLens names, the unembedding tied to the embedding, loaded with no folding or centring. The three
flags FC sets: `use_split_qkv_input`, `use_attn_result`, `use_hook_mlp_in`. The conversion runs in a separate venv
pinned to FC's `requirements.txt` (torch 2.3.1, transformer-lens 2.0.0, transformers 4.42.3), with `eap` installed from
EAP-IG `aa8cdc8`. Built edits and randoms are made in our venv in the same job and passed as dW files in node-local
scratch, deleted at the end.

**Tokenisation (D3).** `eap.attribute.tokenize_plus` and `eap.evaluate.tokenize_plus` are replaced by one function. It
takes the batch's window indices and returns the tokens, a mask of ones, lengths equal to the window length and the
position count. It is the only change to the package's behaviour.

**Floor.** For each of the eight task window sets, 5 random splits into halves; M's circuit on each half by the primary
selection; edge and node IoU between halves. The floor is the five values' minimum and median, per task. The x2 rule
and any rule filed later take their bars from these.

**Per item, saved and reported.** The 927 scores for M and F; both circuits at both selections; their sizes; W's
faithfulness for both; counts of unchanged, added and removed nodes and edges; W's ΔS for nodes and edges; H's node and
edge IoU with p; the item's floor beside. The headline per item is edge IoU against the floor; the figure puts every
item on one axis with the floor band, built items first.

## D. Things noticed in the released code

- EAP-IG HEAD changed the integration points from k = 1..m to k = 0..m−1 in `13c0033` (2025-02-23); both papers state
  the first. With KL against the clean run, the k = m point has zero gradient, so `aa8cdc8` averages four informative
  points and one zero.
- FC's faithfulness compares KL against the corrupted run; H's normalised faithfulness compares against the clean run.
  The two numbers are not interchangeable; each is used only with its own bar.
- HF's node IoU subtracts `{'inputs', 'logits'}` while the node is named `input`.
- FC's `before_after_circuit.ipynb` takes its 95th percentile over edges already in the 5% circuit, so its figure shows
  the top quarter of a percent of all edges.
