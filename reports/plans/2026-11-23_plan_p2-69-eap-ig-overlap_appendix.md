# P2.69 appendix: step table with quotes, items and states, adapter (revision 1)

*Companion to `2026-11-23_plan_p2-69-eap-ig-overlap.md`. Paper quotes are from the arXiv v2 PDFs (page numbers are the
PDF's). Code quotes are from the clones in `artifacts/p269_src/` at the commits in the main file. "W" is Wang et al.
2025, "H" is Hanna et al. 2024, "FC" is FinetuneCircuits at `fdaeeaa`, "EAP" is hannamw/EAP-IG, "HF" is
eap-ig-faithfulness at `d42fea6`. D-numbers point to the main file's departures list.*

## A. Step table

| Step | Published choice, verbatim, with location | Ours |
|---|---|---|
| Graph and nodes | H §2 p.2: "We define this digraph's nodes to be the LM's attention heads and MLPs ... Edges specify where a node's output goes; a node v's input is the sum of the outputs of all nodes u with an edge to v." FC `eap/graph.py` `from_model`: one input node (`hook_embed`), heads, MLPs, logits; each head has separate q, k, v inputs | The same graph: input, 24 heads, 4 MLPs, logits; 927 edges (828 into heads' q, k, v, 70 into MLPs, 29 into the logits) |
| Scoring level | W §3.1 p.3: "The formula for scoring each edge is: ΔL(E) ≈ \|(e_corr − e_clean)ᵀ (1/m) Σ_{k=1}^{m} ∇_{e_k} L(x)\|". FC `100_2arithmetic_opeartions_circuit.ipynb`: `attribute(model, g, dataloader, partial(kl_divergence, loss=True, mean=True), method='EAP-IG', ig_steps=5)` | That call, verbatim. Edges scored; nodes are the ends of kept edges |
| IG form and steps | H Eq. 3 p.4, sum over k = 1..m, embeddings interpolated "z′ + (k/m)(z − z′)"; H §3 p.5: "we select m = 5 and justify this in Appendix C". W §3.1 p.3: "we choose m = 5 based on Hanna et al.'s (2024) recommendations". FC `eap/attribute.py`: `for step in range(1, steps+1)` | As FC (D1). EAP HEAD `e24bafd` has `for step in range(0, steps)` |
| Metric | W §3.2 p.3: "We adopt Kullback-Leibler divergence (KL-divergence) as the metric, following Conmy et al." FC notebook `kl_divergence`: `F.kl_div(probs.log(), clean_probs.log(), log_target=True, reduction='none').mean(-1)`, read at `input_length - 1`. H §4.2 p.6: "we also run EAP-IG and activation patching with KL divergence as the loss" | FC's notebook function, copied verbatim into our run script (it is defined in the notebook, not in `eap/`). The read is the last token (D4) |
| Task data | W §3.3 p.3-4: "the remaining 20% is reserved for identifying circuits". FC: `ds.to_dataloader(32)` | Each item's read windows (section B), batch 32, the same windows for M and F |
| Corrupted inputs | H §2 p.2: "Corrupted inputs fit the same task schema but elicit distinct output". W §3.3: "Corrupted data involves altering the arithmetic operation"; Sequence: "altering one term in the sequence". FC/EAP: clean and corrupted must have the same number of positions | One deciding token swapped, same length, fixed pairing seed (D5; per item in section B) |
| Circuit selection and size | W §4.1 p.4: "Prior work (Hanna et al., 2024) achieved over 85% faithfulness by selecting 1–2% of edges. Given our more complex tasks and larger model, we select 5% of edges to ensure reliable circuits (faithfulness >70%)." FC: `five_percent_edges = int(total_edges * 0.05)`; `g.apply_topn(five_percent_edges , absolute=True)`; `g.prune_dead_nodes()` | Those calls: n = int(0.05 × 927) = 46, then pruning. Sizes after pruning reported |
| Second selector | H §4.2 p.6: "for n = 30, 40, ..., 100, 200, ..., 1000, we select a circuit of n edges using a greedy search procedure"; "faithful (recovering ≥85% of model performance)"; footnote 7: absolute scores. H §5 p.7: "the smallest circuit we have found that achieves at least 85% faithfulness". HF `pareto.py`: `graph.apply_greedy(i, absolute=True)` | FC's `apply_greedy` on H's ladder up to 927, the same scores; normalised faithfulness with KL to the clean run as the metric (D12) |
| Pruning | H §4.2 p.7: "we (recursively) prune all nodes and edges that have no children, or no parents". FC: `prune_dead_nodes()` | Same call |
| Faithfulness (gates the headline) | W §3.2 p.3: "Faithfulness = (1 − \|F(M) − F(C)\|/F(M)) × 100%". FC notebook `calculate_faithfulness`: `percentage_performance = (1 - faithfulness / baseline_performance) * 100` with `partial(kl_divergence, loss=False, mean=False)`; `evaluate_baseline` and `evaluate_graph` call `metric(logits, corrupted_logits, ...)`, so F(M) is the full model's KL from its corrupted run and F(C) the circuit's | FC's function verbatim. Bar 70% (W §4.1) on both circuits, else "circuit not faithful"; F(M) at or under the fp32 floor or not finite: "no gap"; output node pruned: "empty" |
| Faithfulness (second selector) | H §4.2 p.7: "we normalize the circuit's performance m as (m − b′)/(b − b′)" | With the KL metric, b = 0 and this is 1 − KL(clean ‖ circuit) / KL(clean ‖ corrupted) |
| Overlap measure | W §4.2 p.5: "ΔS = (1/n) Σ_{t=0}^{n−1} Δs_{t→t+1}/S₀ × 100%, where Δs_{t→t+1} denotes the number of nodes or edges that change from checkpoint t to checkpoint t+1, and S₀ denotes the total number of nodes or edges in the initial circuit." FC `ft_node.ipynb`: `changes.append(added + removed)`; `ft_edge.ipynb`: `if prev_in_graph != curr_in_graph: change_count += 1`. H §5 p.7: "the intersection over union (IoU, also called Jaccard similarity) between the circuits' nodes, and the IoU between their edges"; H App. H: hypergeometric test | W's ΔS for nodes and edges, n = 1 (D6); H's node and edge IoU with p (D8, D9: ours and the code's values side by side); both circuit sizes after pruning beside |
| Reused and new | W Fig. 3 caption p.6: "Nodes and edges are color-coded based on their status: added (blue), removed (red), or unchanged (grey/white)." FC `before_after_circuit.ipynb` `compute_differences`: `added_edges = edges_final_set - edges_base_set`, `removed_edges = edges_base_set - edges_final_set` | "Reused" = unchanged; "new" = added; "removed" = in M's only; every moved edge listed |
| §4.3 sensitivity | W §4.3 p.5: "we use the 95th percentile of edge scores as a dynamic threshold. Edges in the initial and final circuits exceeding this threshold are retained". FC: `compute_dynamic_threshold(edges, percentile=95)` over the circuit's in-graph edges, nodes left unfiltered | Computed exactly so, reported beside the headline (D7) |
| Noise | W §3.2 p.3: "Robustness(p) = J_E(G₀, G_p)" on perturbed datasets; W §4.1 p.4: "we perturb the original dataset by 10% to 90%" | A project diagnostic, not W's robustness test: section C |
| Precision | FC: `HookedTransformer.from_pretrained(..., device='cuda', hf_model=hf_model)`, default fp32 | fp32, TF32 off |

## B. Items, windows, corrupted inputs and saved states

Every path was checked with `ls -laL` on 2026-11-23. Built items have no state file; their code, settings and any
payload are listed, and each item's exported native state is hashed into the run's record. A pairing seed draws the
swapped token by the rule of `shifted` in `experiments/locator_table/pairs.py`: another index of the pool, never the row's
own. Seeds are filed in the run's `predictions.json`.

| Item | F | Windows (read position) | Corrupted input | State and sha256 |
|---|---|---|---|---|
| x2, x10 | layer-1 `down_proj` columns [486, 499, 613, 778, 1125, 2028, 2352, 2960] times 2 or 10 (`experiments/rule_retention/run.py` `rescaled`; settings `experiments/kstar_levels/results/predictions.json`) | copy24 held-out, `experiments/open_pair/results/large/amend1/windows_copy_trained.pt`: marker at `half − 1`, key at `half`; cut after `half` | the key at `half` swapped for another key in those windows, as `pairs.py`'s key-table rule | built |
| rebuilt | `experiments/rho_saved_items/run.py` `rebuilt(M, cfg)`: the 8 columns above zeroed, then every tensor of `experiments/rule_retention_fixed/results/large/rebuilt_mlp.pt` (37,751,237 bytes) copied in, the MLP weights of layers 2 and 3; the function takes E208's config, not `(M, old)` | as x2 | as x2 | payload sha256 `55cb8e62aeef3e71d527b6c7978ed9771a83e07aa855f220f9aff30b52cf16a3` (computed 2026-11-23; no earlier record found) |
| new_few, new_thin | `commute.known_edits` recipes; slots and gains in the committed `experiments/kstar_levels/results/edits.json` (the tree holds an uncommitted change to it from another session) | P1.31's cue windows, A′ᵢ last | A′ᵢ swapped for another of the eight A′ | built |
| random_size_241001 | E242 `random()`: E241's Gaussian, total size new_thin's (`experiments/kstar_vpd_wording_a/run.py`) | x2's windows (`random_windows: x2`) | as x2 | built from seed 241001 |
| random_kl_241002 | E242 `random_kl()`: the seed-241002 direction scaled so KL(F ‖ M) on new_few's windows equals new_few's | new_few's cue windows (`kl_target_item: new_few`) | as new_few | built from seed; the scale x is logged and kept |
| E62 copy, seeds 0-2 | `experiments/induction_cue/results/large/copy_s{0,1,2}.pt` | E62's saved read windows `experiments/induction_cue/results/large/windows.pt` (sha256 `490e5f74e7b6...` in `results/run.json`): 41 tokens, read at `half` = 20 (`predictions.json` settings), cut to columns 0-20 | `copying.absent_b`: B at p + 1 replaced by another read B′, as `pairs.py` builds "copy" | copy_s0 `759fb83995d6`, copy_s1 `1f1f69b2d53c`, copy_s2 `cab79309c189` (full hashes in `run.json`) |
| E62 lookup, seeds 0-2 | `.../lookup_s{0,1,2}.pt` | the same windows | the second A, at `half − 1` = 19, swapped for another key, as `pairs.py` builds "lookup" | lookup_s0 `c74c074b70fa`, lookup_s1 `5e10359fa75e`, lookup_s2 `397924a82ad9` |
| E62 random | Gaussian at copy_s0's per-matrix norms, `commute.finetune.random_direction` as E237 calls it | as copy | as copy | built from a filed seed (no E62 random is saved) |
| Key grid, 5 tasks × seeds 0-2, scaled | `experiments/refit_96key/results/large/dW_<task>_s<k>.pt` times E237's c: copy 1, rank1 0.9201, elicit 0.8707, cipher 0.7520, prior_cipher 0.7798 (`experiments/key_grid/results/stage0.json`) | E237's read rows (`key_grid/run.py` `World.make`: `read(read_half187(task))`), 8 tokens, marker at 6 and key at 7, read at 7; only rows with the marker at 6 and a table key at 7 (D11) | the key at 7 swapped for another key of the table | per state in `experiments/refit_96key/results/stage0.json`, `states["e58/<task>_s<k>"].sha256` (e.g. copy_s0 `0a515d3ba390`) |
| Key grid randoms | E237 `World.random`: `FT.random_direction(dw, random_seed + index)`, times the same c | as the task | as the task | built from seed |
| M | the VPD target model through `commute.vpd.load_component_model` | every set above | every set above | its checkpoint `artifacts/vpd/t-9d2b8f02/ckpt/`, hashed in the record |

## C. Adapter and the resampling diagnostic

**Environment.** A separate venv with Python 3.10 (FC's bytecode is CPython 3.10), pinned to FC's `requirements.txt`:
torch 2.3.1, transformer-lens 2.0.0, transformers 4.42.3. The `eap/` tree of FC at `fdaeeaa` is staged to node-local
scratch, its files hashed, and imported from there as a namespace package, as the notebook does.

**Native state per item (D2).** In our venv, each item is built with its original loader (table B) and its full
`target_model` state is saved: embeddings, every RMS scale, every matrix, about 270 MB in fp32. The file is hashed,
converted in the second venv, used and deleted before the next item.

**Conversion.** A `HookedTransformerConfig` with n_layers 4, d_model 768, n_heads 6, d_head 128, d_mlp 3072, d_vocab
50277, n_ctx 512, `act_fn="gelu_new"` (the tanh form, as `NewGELU`), `normalization_type="RMS"`, eps 1e-6,
`positional_embedding_type="rotary"`, rotary_dim 128, rotary_base 10000, `rotary_adjacent_pairs=False`,
`n_key_value_heads=None` (6 key and value heads for 6 query heads), `default_prepend_bos=False`. No folding or centring.
The unembedding is the tied embedding. FC's three flags are set: `use_split_qkv_input`, `use_attn_result`, `use_hook_mlp_in`.

**Stage-0 checks (stops).** Destination coverage: every converted parameter is assigned exactly once, from a declared
source. A source may fan out: the shared native embedding goes to both `W_E` and, transposed, `W_U`. Biases that do
not exist natively (attention, MLP, unembedding) are set to explicit zeros and listed. Every native parameter is the
declared source of at least one destination. Clean and corrupted logits match the native forward. The full circuit returns the clean logits and the
empty circuit the corrupted logits through FC's `evaluate_graph`. Each within the native forward's batched-against-single
floor, measured first. These run on M and on x2 before any attribution.

**Tokenisation (D3).** `eap.attribute.tokenize_plus` and `eap.evaluate.tokenize_plus` are replaced by one function. It
takes the batch's window indices and returns the tokens, a mask of ones, lengths equal to the window length and the
position count.

**Resampling diagnostic.** For each of the nine task sets, 5 full-size resamples of the windows with replacement (seed
filed). M's circuit on each, and IoU between every two of them: the M-against-M range. The same for F on x2 and on
new_thin. Reported beside the values; control 2 reads against M's range; nothing else uses it as a bar.

**Per item, saved and reported.** The 927 scores for M and F; both circuits at both selections and their sizes after
pruning; Wang's faithfulness for both and F(M); counts of unchanged, added and removed nodes and edges, and the moved
edges by name; W's ΔS; H's node and edge IoU with p, ours and the code's; the §4.3 sensitivity counts; the resampling
range beside. One figure puts every item's edge IoU on one axis with the range, the controls first.

## D. Things noticed in the released code

- EAP-IG HEAD changed the integration points from k = 1..m to k = 0..m−1 in `13c0033` (2025-02-23); both papers state
  the first. With KL to the clean run, the k = m point has zero gradient, so FC's tree averages four informative points
  and one zero.
- EAP-IG `aa8cdc8` has no `eap/__init__.py`, so `pip install` of that commit installs no package, and no `eap/metrics.py`.
- FC's faithfulness compares KL from the corrupted run; H's normalised faithfulness compares with the clean run. Each is
  used only with its own bar.
- HF's node IoU subtracts `{'inputs', 'logits'}` while the node is named `input`, and its p is `1 - hypergeom.cdf(x, ...)`,
  the chance of more than x.
- FC's `before_after_circuit.ipynb` takes its 95th percentile over edges already in the 5% circuit, so its figure shows
  the top quarter of a percent of all edges.
