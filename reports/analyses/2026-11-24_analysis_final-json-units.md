# One unit per column in the 67M method table (cell P2.79, 2026-11-24)

The writer of the 67M method table now gives every column one unit and one form, and no longer picks, pair by pair, whichever of two forms shows the wider difference. Every value that changed is explained by that edit: the two label twins' share moved into log-odds, steering and the top-neuron reading each split into two columns, and seven cells became empty with a reason because nothing was saved in the column's unit. For the paper this means the steering axis should use the pooled vector, the one form read on all nine built pairs, and the repeat-label twin no longer shows a clear share difference.

## What changed

`experiments/method_table/plot_final.py` writes `experiments/method_table/results/final.json` and `figures/final.*`. The redteam page `reports/reviews/2026-11-24_review_scorecard-forms.md` found the share in two units on one axis and steering in three forms. The edit:

- the share is read in log-odds everywhere (the share of the fine-tune's gain in the log-odds of the target at the answer position), including the two label twins, which were in the label contrast;
- steering becomes `steering_pooled` and `steering_per_key`, the top-neuron reading becomes `neuron_top_k64` and `neuron_top_kmatched`, and the rule that kept the wider of the two forms is gone;
- a pair with nothing saved in a column's unit or form gets a cell with `gap: null` and its reason in `call` and `form`;
- final.json gains a top-level `units` map, and each cell gains a `unit`.

The diff below is between the file at `HEAD~1` and the regenerated file (jq on `.cells`, key = pair and method). Nothing else changed: the other 95 of the old 128 cells have the same values, call and source, and every other field difference follows from the rows below.

### Counts by kind

| kind | cells |
|---|---|
| old `steering` and `neuron_top` cells removed | 26 (13 pairs times 2) |
| new form columns added with a value | 47 |
| new form columns added as null | 5 (e53 per key; e123 and e125 steering in both forms) |
| existing cells now null | 5 (e123 and e125 neuron and neuron_half; e53 size) |
| existing cells with a new value | 2 (e123 and e125 share) |

Each removed value reappears unchanged in the column of its own form: pooled steering on nine pairs, per-key steering on e54_matched and e54_own, the top-64 neurons on ten pairs and the matched set on e46. The label twins' old steering and top-neuron values were in the label contrast and do not reappear.

### Cells whose value changed or became null

| pair | column | old (unit) | new (unit) |
|---|---|---|---|
| e123 case label | share | -0.018, no call (label contrast) | -0.031, no call (log-odds) |
| e125 repeat label | share | -0.090, apart (label contrast) | -0.031, no call (log-odds) |
| e123 case label | neuron | 0.005, no call (label contrast) | null: not read in this unit |
| e123 case label | neuron_half | -0.008, no call (label contrast) | null: not read in this unit |
| e125 repeat label | neuron | -0.146, apart (label contrast) | null: not read in this unit |
| e125 repeat label | neuron_half | -0.198, apart (label contrast) | null: not read in this unit |
| e53 lowercase vs ciphers | size | log10 ratio -0.098, apart (summed squared norms, unrooted) | null: not read in this unit (summed norm) |
| e123 case label | steering | -0.506, apart (per key, label contrast) | null in both forms: setting picked by label-contrast rise |
| e125 repeat label | steering | -1.087, apart (per key, label contrast) | null in both forms: same reason |
| e123 case label | neuron_top | -0.008, apart (matched k, label contrast) | k64 0.003 no call; matched -0.005 apart (log-odds) |
| e125 repeat label | neuron_top | 0.010, no call (k64, label contrast) | k64 0.008 apart; matched 0.006 apart (log-odds) |
| e53 lowercase vs ciphers | steering per key | (was pooled -0.053) | null: no per-key vector saved |

### The forms on the built pairs, where they differ

| pair | old steering cell (form) | pooled | per key |
|---|---|---|---|
| e46 copy vs random | 0.470 (pooled) | 0.470 apart | 0.048 no call |
| e54 matched | -0.237 (per key) | -0.177 apart | -0.237 apart |
| e54 own | -0.120 (per key) | 0.114 apart | -0.120 apart |
| e62 copy vs lookup | 0.412 (pooled) | 0.412 apart | -0.023 no call |
| open pair c3 copy | 0.238 (pooled) | 0.238 apart | -0.044 no call |
| e58 copy prior cipher | 0.594 (pooled) | 0.594 apart | 0.065 apart |

Against the share, pooled steering now runs the same way on five built pairs, the opposite way on two, and gives no call on two. Per key runs the same way on one, opposite on three, no call on four, and is missing on one. The top-64 set is opposite to the share on five built pairs; the matched set on two.

## Which form covers every pair

- Pooled steering is read on all nine built pairs (eight from the holes-fill and 96-key refit runs, e62 from its own steering run) and on the two 96-key twins.
- Per-key steering is read on eight of the nine: lowercase vs ciphers has no per-key vector saved.
- Neither form is read on the two label twins under the steering columns' selection rule: their run picked the steering setting by the largest label-contrast rise, not the log-odds rise.

So the pooled form is the axis form for steering, under whip-lead's rule that the axis form must cover all nine pairs. No user choice is needed.

## Two calls that move

**The repeat-label twin's share.** In the label contrast its seed ranges were disjoint, twin higher. In log-odds the real fine-tunes read 0.110 to 0.237 and the twins 0.175 to 0.219, so the ranges overlap and the cell gives no call (gap -0.031). The paper text's twin tally (one real higher, one twin higher, two no call) already matches log-odds; Fig 3a, drawn from the contrast unit, did not. Because the share is no longer apart there, this pair's rank and size cells now read "apart where the share is not".

**e54 own's steering.** Pooled, the side built to reuse is higher (0.114, seed ranges apart, the same way as the share). Per key, the new side is higher (-0.120, apart, opposite to the share). The old cell kept the per-key form, which was wider. With pooled as the axis form, e54 own no longer counts as a steering reversal. The new side has only two seeds, so neither call has much margin.

## Readers of final.json

These scripts read `experiments/method_table/results/final.json`. None was changed here; their owners decide.

| script | old keys it reads | what happens now |
|---|---|---|
| `experiments/method_table/paper_figs.py` | `final[(pair, "steering")]` | KeyError: Fig 4 does not build |
| `experiments/size_matched/plot.py` | `final[(pair, "steering")]`, `final[(pair, "neuron_top")]` | KeyError on the first built pair |
| `experiments/p270_rescore/rescore.py` | `final[(pair, "size")]` on built pairs | e53 size is null: `log_ratio(None, ...)` raises a TypeError |
| `experiments/adl/check_independent.py` | `(c,) = [... method == "steering"]` | ValueError, no steering cell to unpack |
| `experiments/adl/compare.py` | filters methods in share, neuron, steering | steering silently missing from each row, then its sign count raises a KeyError on `rows[p]["steering"]` |
| `experiments/method_table/gate_plot.py`, `gate_plot_1116.py` | COLS `steering`, `neuron_top` via `(p, c) in table` | those columns silently show "not run"; the four new columns are not drawn; null cells draw as not run |
| `experiments/method_table/paper_scorecard_67m.py` | through `gate_plot_1116.cells_67m` | steering and top-neuron rows silently read "not run" |
| `experiments/part1_vs_methods/run.py` | `cells.get` for steering, neuron_top | silently "missing"; null cells read "no order" but pass `gap: None` into its Spearman |
| `experiments/kstar_pairs/score.py` | `cells.get` for steering, neuron_top | silently empty gap lists for those two |
| `experiments/known_3seeds/assemble.py` | `neuron_top` in FINAL_METHOD | `next(..., None)`: its old-value checks silently find nothing for the top-neuron reading |
| `experiments/claim2_needed/compare.py` | size, erank, untrained, prior on built pairs | e53's size gap is now None in its size column, passed on to whatever reads it next |
| `experiments/proxies_vs_r/analyse.py`, `prior_steering/run.py`, `share_gap_sizes/plot.py`, `gain_only/plot.py`, `topk_gate/*.py`, `share_built_kind/*.py` | share and prior only (steering from other files) | still run; the label twins' share values are now log-odds |

## Not run

- No caller above was fixed or rerun, and no paper CSV (`paper_fig3a_share.csv`, `paper_fig4_share_vs_prior_steering.csv`) was regenerated.
- The label twins' steering was not re-read under the log-odds rule; their saved log-odds readings exist (`real_vs_twin/results/stage3.json`) and could fill the two columns if whip-lead accepts the different selection rule.
- e62's pooled steering is E135's fp32 reading, not the E143 holes-fill run the other pairs use; its rows, key position and floor were not checked against E143's (redteam 11-24). Read it as from a different run.
- e53's size was not computed as a summed norm.
- No model was loaded; everything is from saved files.
