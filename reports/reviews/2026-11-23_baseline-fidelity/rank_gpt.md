# Effective rank and update size fidelity: GPT adversarial reviewer (2026-11-23, condensed faithfully; HEAD e894f1ed)

Verdict: needs-attention. The per-matrix effective-rank formula is faithful to Roy & Vetterli. The 67M update-size scorecard is not a single norm, and substituting the paper's stated norm changes a reported call. All departures are ours; none is attributable to Roy & Vetterli.

1. **Blocker, ours: "Update size" labels three different quantities.** (method_table/plot_final.py:66,102-107,186-196; main.tex:936,982-985,1017,1059-1060)
   - Eleven rows use S = Σ‖δW_m‖_F, one uses the global F, and lowercase vs ciphers uses E = F².
   - Lowercase reads -0.09772 decades as scored, against -0.04759 when each seed is rooted (the call survives).
   - Moving from S to F is not monotone. On the saved uniform-size results, mid vs cipher at the matched step changes from no call to new larger on every seed (reuse max 0.00502714 vs new min 0.00502952), and copy vs cipher flips sign.
   - The nine built rows become 0 agree / 7 reverse / 2 no call, not 0/6/3. The mid reversal is extremely narrow.
   - Fix: build the size scorecard from per-state global norms on the same saved states and rerender the tally.
2. **Major, ours: size and rank/share are sometimes read on different updates, seeds or precision.** (plot_final.py:134-161; refit_rank/run.py; update_svd/run.py)
   - All four 96-key comparisons use the original E58 sizes but refit ranks and shares. Mid vs shuffle shows only one shuffled size (no call); rooting the refit norm2 for 3/3 states gives -0.05238 decades and an apart call.
   - The label twins' sizes were computed before the bf16 save; the case twin compares step 30 vs 300, the repeat twin step 30 vs 100.
   - Fix: join all columns by saved file/hash, step and seed. The norm2 values are already saved, so no rerun is needed.
3. **Major, ours: "matched step" is not matching under the paper's norm.** (keydep.py:120-129; open_pair/run.py:131-136; open_pair_instruments/seeds.py)
   - Selection minimises S, not F. The 24-key cipher states are matched to the same seed's mid update, not to top or copy.
   - size_matched/run.py matches the side means on a global F, not every seed. Rank staying unchanged under a rescale is an identity.
   - Fix: distinguish checkpoint selection from the mean-amplitude rescale, and give the achieved differences.
4. **Major, ours: the real-text prose overgeneralises.**
   - E211's rank (score2.py:30-58; run2_score.json) has positive, disjoint ranges in all six pair-by-size cells, including French vs Yoruba, where the share is unreadable.
   - appD_built_pairs_more.tex:199-211 says "No cheaper reading does better", which is not supported.
   - Sizes are the first checkpoint above a target, not an exact match; E211 scores an fp32 update and saves bf16.
   - Fix: state which readings need the gain floor, report what rank orders, and say "matched targets".
5. **Major, ours: Qwen counterexample to the smaller-update claim.** (appH_qwen_short.tex:17-21; qwen_library_readings/results/scorecard.json, gold/stage2.json)
   - At gold real vs shuffled 5x, real is the larger update on all three seeds (0.000788 vs 0.000642), yet its rank clears the floor (+45 to +49).
   - Qwen uses 196 matrices and global F / base F, and its calls use paired additive gaps with a Gaussian band, not the 67M rule. qwen_reading/run.py logs a median of per-matrix ranks, not the weighted mean.
   - Fix: correct the sentence, and scope "24 matrices" to the 67M.
6. **Minor, ours: state the adaptation.**
   - update_spectrum.py:23-50 matches Roy & Vetterli Def. 1 per matrix in fp64.
   - The weighted mean of per-matrix ranks is our extension and is not the rank of a concatenated update.
   - The legacy baseline_weights/run.py gives a zero matrix rank 1 (no scored row is affected).
   - The historical E51/E141/E171 calls used a different rule from the final scorecard.
   - main.tex:883-888's read-windows wording does not apply to the weight-only baselines.
   - Fix: call it the "squared-Frobenius-weighted mean of per-matrix effective ranks"; say it is undefined at zero.

## 67M scorecard size lineage (S = Σ‖δW_m‖_F, F = global, E = F²; from method_table/results/final.json)
| Pair | Used | seeds reuse/new | gap |
|---|---|---|---|
| Copy at new marker vs lookup | S, induction_cue | 3/3 | -0.07526 |
| Copy maps vs random maps | F, baseline_weights | 4/4 | -0.18109 |
| Lowercase vs ciphers | E, update_svd norm2 | 2/4 | -0.09772 |
| Mid vs cipher, matched | S, keydependent_elicit + _patch | 5/5 | -0.00562 |
| Mid vs cipher, own steps | S | 5/5 | -0.27812 |
| Top vs cipher, 24 keys | S, open_pair_instruments | 8/8 | +0.00485 |
| Copy vs cipher, 24 keys | S | 8/8 | +0.00077 |
| Top vs cipher, 96 keys | S, open_pair a3/a4 | 3/3 | -0.08236 |
| Copy vs prior-matched cipher, 96 | S, open_pair a3/4/6 | 3/3 | -0.09650 |
| Mid vs shuffle, 96 | S | 3/1 | -0.04816 |
| Class map vs shuffled twin | S, a7 | 3/3 | -0.00155 |
| Case label vs twin | S, case_label | 2/2 | -0.44674 |
| Repeat label vs twin | S, fresh_reuse_pair | 2/2 | -0.19790 |
The tab:pairs sizes are F(δW)/F(W0) from size_matched/results/m67.json.

Ship decision: no ship for the single-norm size tally or for the real-text and Qwen wording. Keep the per-matrix rank calculation.
