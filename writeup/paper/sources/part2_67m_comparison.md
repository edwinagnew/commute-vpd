# Part 2 skeleton: measuring new against existing on the 67M (summary for the paper draft)

Read-only summary, 2026-11-17. Repo-relative paths. "Gap" = side built to reuse minus side built to be new. "Apart" = seed
ranges do not overlap (one colour rule since the 11-12 redraw). Labels are **by construction**; no reuse/new verdict exists,
because the part-1 definition is still under revision (`.claude/CLAUDE.md` Rules; `decisions.md` rows 222-227).

## 1. Model, decomposition, tasks

- **Model.** Goodfire's released 4-layer Pile transformer, 67M parameters; MLP width 3,072 per block
  (`docs/research_log/parts/sections/apparatus.html`; `reports/analyses/2026-11-16_analysis_e190-neuron-jobs.md`).
- **Decomposition.** Goodfire's released VPD, pinned `param-decomp` 74146b5. 24 matrices decomposed (attention
  projections + two MLP matrices per block); ~38,900 rank-1 subcomponents, ~9,900 alive. Γ is one network shared across
  all 24 matrices, outputs an importance in [0,1] per subcomponent per position, non-causal attention, frozen
  throughout. MLP matrices of layers 0 and 3 are saturated (any update expressible) (`apparatus.html`).
- **Fine-tunes.** Adam, lr 3e-5, batch 32, on the 24 matrices, KL penalty to the original model at non-trigger
  positions; 41-token Pile windows, held-out windows for every score; largest per-matrix update/weight ratio ~0.01
  (`apparatus.html`).
- **Pairs.** 13 pairs, two to eight seeds a side (`apparatus.html`, "Constructed pairs"; E46, E53, E54, E58, E62, E123,
  E125, E133, E142). **Nine built pairs**: copy at a new marker vs lookup; copy maps vs random maps; lowercase vs ciphers;
  half-known map vs cipher at equal size; half-known vs cipher at own steps; rank-1 vs cipher (step 10); copy vs cipher
  (step 10); rank-1 vs cipher (96 keys); copy vs prior-matched cipher (96 keys). **Four twins** (real vs shuffled
  labels): half-known vs its shuffle (96 keys), class map, case label, repeat label
  (`experiments/method_table/figures/gate_scorecard_1116.csv`).
- **Part-1 labels.** Only five pairs carry a part-1 reading (the old re-weighting share G from a blind affine search,
  `experiments/old_task_labels/results/labels_cut_0p8_0p5.json`); part 1 puts the reuse side higher on all five, gaps
  0.19-0.43 (`reports/analyses/2026-11-11_analysis_p2-20-part1-vs-methods.md`). That definition has since been
  superseded by newer wordings (P1.21-P1.25), none adopted.

## 2. Methods and the scorecard (nine built pairs)

Counts from `experiments/method_table/results/final.json` `.tally.built` (post-redraw); gaps from
`experiments/method_table/figures/gate_scorecard_1116.csv`. Figure: `experiments/method_table/figures/gate_scorecard_1116.png`.

| Method (one line) | Same as share / opposite / no call |
|---|---|
| **Re-gating share r**: share of the trigger log-odds gain reproduced by original subcomponents with Γ's importances recomputed on the fine-tuned activations (`GOALS.md` l.87; `src/commute/local_faith.py` `route_split`) | 9 apart, reuse higher; gaps 0.39-0.69 |
| Share on an untrained SVD (fixed switching rule) | 2 / 2 / 5 |
| Neuron share, whole width (`r_n`: original output weights fed fine-tuned neuron firing, `src/commute/neuron_r.py`) | 3 / 4 / 2 |
| Neuron share, MLP input half (`r_half`) | 3 / 2 / 4 |
| Neuron top-set swap | 4 / 5 / 0, all moves < 0.05 of gain |
| Steering (fine-tune's mean activation change added, searched strength) | 4 / 3 / 2 |
| Prior (original model's log10 P(answer)) | 1 same + 8 "one value a side", all reuse-higher |
| Update size (‖δW‖/‖W‖) | 0 / 6 / 3 (new side larger wherever apart) |
| Effective rank (Roy-Vetterli, norm-weighted) | 7 / 0 / 2; margins ≤ 0.07 log10 |
| Neurons for half the gain (k_half) | 6 / 0 / 3 |
| Top-240 neuron overlap (E155) | 2 / 0 / 5, 2 not run; positive control failed |
| Goodfire job retention (E190) | reuse keeps more on 7 / 9 |
| Goodfire top-K gain survival (E190) | unreadable: control failed |

Key numbers:
- Share gaps on the three pairs matched in update size (size no call): 0.52, 0.62, 0.61 (`gate_scorecard_1116.csv`
  rows "Half-known vs cipher, equal size", "Rank-1 vs cipher, step 10", "Copy vs cipher, step 10").
- Share gap rank correlation with prior's gaps 0.02, with steering's (strength 1) -0.10, over nine pairs
  (`reports/analyses/2026-11-11_analysis_part2-gate-page.md`; `experiments/share_gap_sizes/results/gaps.json`; figure
  `experiments/share_gap_sizes/figures/gap_sizes.png`).
- Prior + steering at strength 1 together reproduce all nine direction calls; neither alone does
  (`reports/analyses/2026-11-10_analysis_e149-prior-steering.md`; figure `experiments/prior_steering/figures/votes.png`).
- Steering's three reversals come from the strength search; at strength 1 they vanish but steering then separates
  neither scored pair (`reports/analyses/2026-11-10_analysis_e146-steering-reversal.md`).
- Size-matched (P2.28, E187): larger side rescaled by one scalar; share still separates 9/9, steering 7 (its
  stand-alone-cipher gap moves -0.12 → -0.92), top neurons and k_half lose one pair each
  (`reports/analyses/2026-11-16_analysis_p2-28-size-matched.md`; `experiments/size_matched/results/summary.json`; figure
  `experiments/size_matched/figures/size_matched.png`).
- Goodfire retention (P2.30, E190): fine-tunes keep 0.60-0.80 of moved neurons' top-20 Pile windows vs 0.88-0.97 for
  matched random updates (18/18 sides); reuse side higher on 7/9 by 0.04-0.11. Top-K: no-update model loses 5.76 / 3.79 /
  0.89 nats at K = 240 / 768 / 1,536 of 3,072, so unreadable (`reports/analyses/2026-11-16_analysis_e190-neuron-jobs.md`;
  figure `experiments/neuron_jobs_k/figures/jobs.png`). Not size-controlled.
- Twins (post-redraw): share apart on 3 of 4, but on class map (-0.034) and repeat label (-0.090) the shuffled twin reads
  higher; case label no call (-0.018). Part 1 reads the real fine-tune higher on case (0.29) and repeat (0.19), so on the
  repeat label the share contradicts part 1 (`gate_scorecard_1116.csv`; `2026-11-16_analysis_part2-gate-page-update.md`).
  Steering reverses both label twins (-0.51, -1.09).
- Agreement with part 1 on its five pairs: share agrees on 3 built, no call/contradicts on the twins; steering ranks all
  five in part 1's order (Spearman 1 over five) yet reverses both twins; no reading agrees in sign on all five
  (`2026-11-11_analysis_p2-20-part1-vs-methods.md`; figure `experiments/part1_vs_methods/figures/part1_vs_methods.png`).

## 3. What re-gating measures (P2.1 and after)

From `reports/analyses/2026-11-09_analysis_what-regating-does.md` unless noted.
- **Mechanics.** Masked forward with original subcomponents, Δ out; Γ evaluated on original vs fine-tuned activations;
  rise over the true gain. The state's own subcomponents never enter the numerator, so do-nothing and a split read the
  same (0.133 on repeat label). Scored only from a gain of 1 log-odds. Can exceed 1 (copy at new marker 1.09-1.10,
  masked start 2.01 log-odds low). The trigger's own inputs carry 0.998-1.043 of Γ's change across eight tasks
  (`experiments/shift_anatomy/results/shift_anatomy.json`).
- **Strengths.** Only reading apart on all nine built pairs; survives size matching on the 67M; gap sizes not predicted by
  prior or steering; needs the trained decomposition (untrained SVD gaps -0.16 to 0.13); equal-size elicit 0.76-0.77 vs
  cipher 0.21-0.27 where steering reads 0.25 vs 0.18-0.24; rescaling the new-marker update leaves 0.691 → 0.686.
- **Weaknesses.** Rank follows the prior (Spearman 0.886 over six maps, 0.95 over eight, post hoc); a fine-tune to a
  different answer at the same trigger reads like the real one (0.55-0.60 vs 0.60-0.61); a class-map rule that carries to
  unseen keys reads like its lookup twin; a code-text fine-tune that made held-out text worse reads ~0.8, stable over five
  batches (0.74-0.85), while a matched random update has too little gain to read
  (`reports/analyses/2026-11-11_analysis_p2-6-share-stability.md`; `experiments/share_stability/figures/stability.png`).
  No positive control of known kind exists. Claims page: "most of the gain is carried by re-gating" holds for VPD's
  masked stand-in, not shown of the fine-tuned model; whether re-gating is a mechanism or the fine-tune leaking through
  Γ's input is open (`docs/research_log/parts/sections/claims.html`).
- **Outperformed by cheap baselines?** On direction, yes in combination: prior + steering give the same nine calls. Not on
  size-matched separation or gap sizes. On Qwen the share saturates on the new decomposition (states incl. random read
  0.991-0.9992) and on the frozen-neuron decomposition cannot be told from update size (1 of 6 cells survive size
  matching) (`2026-11-16_analysis_part2-gate-page-update.md`; `experiments/qwen_library_share/figures/share_gaps_two_decomps.png`).

## 4. Claims for the paper today

1. **Solid (as built):** the share reads the reuse side higher on all nine 67M built pairs, gaps 0.39-0.69. Fig
   `experiments/method_table/figures/gate_scorecard_1116.png`.
2. **Solid (67M):** this separation is not update size: holds on three naturally size-matched pairs (0.52-0.62) and
   after rescaling (9/9). Fig `experiments/size_matched/figures/size_matched.png`. Caveat: rescaling is not a model that
   learned less.
3. **Solid:** it needs a trained decomposition (untrained SVD 2 same / 2 opposite / 5 no call). Fig
   `gate_scorecard_1116.png` (untrained column); `experiments/share_untrained/figures/` is empty.
4. **Solid:** on direction alone the share adds nothing over prior + steering at strength 1. Fig
   `experiments/prior_steering/figures/votes.png`.
5. **Provisional:** share gap sizes carry information neither prior nor steering does (nine non-independent pairs). Fig
   `experiments/share_gap_sizes/figures/gap_sizes.png`.
6. **Provisional:** cheap neuron-level readings (job retention 7/9, k_half 6/9, effective rank 7/9 by ≤ 0.07) lean the
   share's way but are not size-controlled and k_half's separations mostly coincide with size. Figs
   `experiments/neuron_jobs_k/figures/jobs.png`, `experiments/neuron_r/figures/k_half.png`,
   `experiments/update_svd/figures/gaps.png`.
7. **Contradicted:** the share tells a real fine-tune from its shuffled twin. 2 of 3 "apart" twins go the wrong way;
   repeat label contradicts part 1 (claims page lists this as refuted at 1 of 4 before the redraw). Fig
   `gate_scorecard_1116.png`.
8. **Contradicted:** a high share implies learning: code-text fine-tune that worsened text reads ~0.8. Fig
   `experiments/share_stability/figures/stability.png`.
9. **Contradicted / no power on Qwen:** the share separates Qwen pairs beyond size. Fig
   `experiments/qwen_library_share/figures/share_gaps_two_decomps.png`.
10. **Provisional:** steering's disagreements with the share are a strength-search artefact (E146).

## 5. What the section still needs

- Labels independent of construction: an adopted part-1 definition, read on all 13 pairs (it reads 5, with an older
  wording).
- A pair where prior and steering point opposite ways (attempt stopped: cipher prior 2.32e-4 vs needed 2.35e-4).
- Size-matched tests of the cheap readings (rank, k_half, overlap, retention) by retraining, not rescaling.
- A positive control for the share of known kind (e.g. the hand-made changes, E137-E139), and a random update that
  moves the text as much as the code-text fine-tune.
- Any significance test (≤ 9 non-independent pairs, one task family, one model, one decomposition).
- Missing cells: k_half on case and repeat twins; overlap on 96-key pairs and twins; untrained-share figure; `check.py`
  never run on the 11-16 gate page (worker down).

Friction: `ls` of six figure directories in one command hid that `experiments/share_untrained/figures/` is empty until
re-listed; the redrawn `final.json` tally (twins apart 3/4) disagrees with `claims.html` ("1 of 4") and the 11-11 page,
so twin numbers must be cited from the 11-16 page, not the log; I accidentally included `python3 -c 1` in a command and
the no-compute hook refused the whole call.
