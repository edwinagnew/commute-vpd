# Fidelity audit: steering share, effective rank, update size (2026-11-23)

Plain words. I compared our steering, effective-rank and update-size code with the papers the write-up cites for them,
and with what the write-up says. The effective rank is Roy and Vetterli's formula, and update size is a simple norm
that needs no source. Our steering, though, is not Venhoff et al.'s method, even though the paper calls it "the test of"
Venhoff. Several reported steering readings also depend on two choices made per pair: the strength search, and
keeping whichever of two forms gives the wider gap. For the reuse question this means the steering row of the scorecard
needs relabelling, and its reversals need restating, before it can serve as a baseline.

Sources read (copies in `~/commute-data/agents/redteam-lead/fidelity/papers/steer_rank_dl/`, text beside each PDF):
Turner et al. arXiv 2308.10248 (32-page version); Rimsky/Panickssery et al. arXiv 2312.06681; Venhoff et al. arXiv
2510.07364 v4 (ICML 2026 header) and v1; Ward et al. arXiv 2507.12638; Roy & Vetterli, EUSIPCO 2007 (EURASIP
proceedings PDF); Minder et al. arXiv 2510.13900 (for the vector construction). Paper paths are relative to
`~/commute-agents/writeup-lead/paper/paper/`, and code paths to `~/commute-agents/redteam-lead/`.

## 1. Steering share

What the code does (`src/commute/fair_steering.py:7-9, 44-61, 72-77`; `experiments/holes_fill/run.py:163-176`;
`experiments/steering_copy_lookup/run.py:164-226`). The vector at block l is the mean, over fitting windows, of
(fine-tuned minus base) block-l input at the answer position. The "per-key" form takes that mean separately for each
key token. It is added to the base model at the answer position only. The layer (0-3) and the strength
{0, 0.5, 1, 2, 4, 8} are chosen per fine-tune by the largest mean log-odds rise on fitting rows. The reported share is
the rise on the read rows divided by the fine-tune's own gain. `method_table/plot_final.py:60-63` then keeps, per
pair, whichever form (pooled or per-key) gives the wider |gap|.

What Venhoff et al. do (v4 §3.2-3.6 pp. 5-6, App. E.2 p. 19; v1 §3 does the same, after Dunefsky & Cohan). They train
"category vectors" in the base model by gradient descent: the loss is cross-entropy on the thinking model's next token,
taken at positions where base and thinking model disagree. A coefficient MLP is trained jointly with them. At
generation time, the thinking model's SAE classifies each disagreement position and picks which vector to add. The
score is the share of the benchmark-accuracy gap from base to thinking model that the hybrid recovers. Best of three
seeds is chosen on a holdout mix. None of the optimisation, the SAE gating, the coefficient MLP or the generation is in
our code. Only the readout idea (gap recovered by steering the base model) is shared.

| departure | ours/theirs | evidence | consequence for the reported number | severity | disclosed? | fix |
|---|---|---|---|---|---|---|
| Venhoff: optimised, SAE-gated category vectors with a learned per-position coefficient, applied during generation at disagreement positions, scored as accuracy gap recovered. Ours: an unoptimised mean activation difference at one block and one position, with a grid-searched scalar strength. The paper calls ours "the test of Venhoff" and labels the scorecard row with `\citep{venhoff2025base}` | ours | `main.tex:930-931` ("This is the test of \citet{venhoff2025base}"), `appE_methods_detail.tex:90-91`, `sec5_results.tex:171` ("steering \citep{venhoff2025base}"); Venhoff v4 §3.3 p. 5, §3.6 p. 6, App. E.2 p. 19; v1 §3 | The label is wrong: no number in the paper is Venhoff's test. An optimised vector would give at least as high a recovery as the mean difference on every fine-tune, so the reversals may not survive Venhoff's own method (not run) | blocker | no: presented as Venhoff's | Relabel. Replace `main.tex:930-931` with: "Venhoff et al. (2026) score a constructed hybrid by the share of the base-to-fine-tune gap it recovers; we use the same readout with a far simpler construction, one unoptimised mean-difference vector, so this is not their method." In `sec5_results.tex:171` write "mean-difference steering" with no `\citep{venhoff2025base}`, or cite it as "in the spirit of". To be Venhoff's method, a re-run would optimise one vector (pooled) per fine-tune at the read block, by cross-entropy on the fine-tune's answer at the answer position, on the fitting rows, and read it on the read rows |
| Vector source. ActAdd: the difference of one model's activations on one contrast pair of prompts (Alg. 1, p. 4). CAA: the mean difference over many contrast pairs at the answer-letter position (§3, p. 3). Ours: the mean of (fine-tune minus base) on the same inputs | ours | `fair_steering.py:44-57`; Turner p. 4 Alg. 1; Rimsky p. 3 l. 201-212 | It is a difference between two models, not between two behaviours. The closest published vector is Minder et al.'s activation difference δ (mean of fine-tune minus base on the same inputs, ICLR 2026, §2 p. 4), whose steering adds δ to the fine-tuned model and reads the generated text's semantics, not a gain share | minor | yes, `main.tex:929-930` ("difference is between the two models on the same inputs") | Add: "the vector is the mean activation difference of \citet{minder2025narrow}, here added to the base model" |
| Injection site. ActAdd adds at a subset of front positions (p. 3 l. 150-151, Alg. 1). CAA adds at every position after the user's prompt (p. 1 Fig. 1). Ours adds at the answer position only, nothing else | ours | `fair_steering.py:9, 60-61` | The test asks whether one position's residual at one block can carry the gain. Reads are at that same position, so nothing upstream is tested | minor | yes in part: `main.tex:926` says "at the answer position", but not that the vector is added there only | Add "added at the answer position only" |
| Strength and layer search. Ours: the largest rise over 4 layers × 6 strengths up to 8, with no limit on off-target harm (`off_kl` is recorded, not used). Turner: a hand-set coefficient. CAA: layer 13 from a sweep, multipliers ±1, ±2 (pp. 4-5). Minder: the largest factor that keeps coherence (p. 4) | ours | `fair_steering.py:72-77`; `holes_fill/run.py:167-168`; `holes_fill/results/predictions.json` settings `strengths [0,0.5,1,2,4,8]`; `steering_reversal/results/stage1.json .score.pairs.e54_matched` | The share can exceed 1: per-key shares read 1.03-1.37 (my recomputation below), and 1.83 after size matching (`main.tex:1153`). The reversals depend on the search. E146 finds the half-known-vs-cipher reversal (−0.18, apart) gone at strength 1 (+0.055, "does not"). Recomputed from E149's saved strength-1 states by the paper's seed-range rule, pooled steering at strength 1 sides with the share on 8 of 9 built pairs (no call on rank-1 vs cipher at step 10, −0.032) and reverses none | major | partly: `main.tex:927-928` says "block and strength are chosen per fine-tune", and `main.tex:1150-1156` covers one pair. That the three reversals need the strength search is not disclosed | Either make the plain form (pooled, strength 1, layer chosen) the scorecard row, or add to `appD_built_pairs_more.tex` after l. 83: "At strength 1 the pooled vector reverses none of the nine pairs and sides with the share on eight; the reversals come from the strength search." |
| Per-key form: K vectors, one per key token, each the mean difference over that key's windows. Neither ActAdd nor CAA has an input-indexed vector (one vector per behaviour). Venhoff's per-category vectors are optimised and chosen by the thinking model's SAE, not indexed by the input token | ours | `fair_steering.py:33-36, 57`; `holes_fill/run.py:11` (key = token at the read position); `steering_copy_lookup/run.py:7-8` (token at half − 1) | It writes a key-specific answer into the residual, so it is mean activation patching per key, not steering. It recovers about the whole gain on both sides (0.89-1.37 per seed), memorised or reused alike. It contradicts the paper's own gloss "a gain that one vector can carry" (`appE_methods_detail.tex:90-91`). It sets 4 of the 13 reported cells: both half-known-vs-cipher pairs and both label twins | major | no: `main.tex:932` names "a pooled and a per-key form" and never defines per-key | Define it in the appendix item ("per-key: one mean vector per key token, so it can carry a per-key lookup") or cut it and report the pooled form only |
| Selection between forms: each pair keeps whichever of pooled and per-key gives the wider |gap|. The paper says this "can only favour that reading" | ours | `plot_final.py:60-63, 85-89, 116-121, 126-131, 145-148`; `prior_steering/run.py:224-233` inherits it into E149 | The selection maximises |gap| whatever its sign, so it produced the reversals on half-known vs cipher at own steps and on both label twins. Pooled-only and per-key-only give different tallies (see "Numbers that rest on the selection" below). `main.tex:1015-1016` ("which can only favour that reading") is wrong for steering: it made steering look worse against the share, not better | major | yes as a caveat (`main.tex:932-933`, `1015-1016`), but described wrongly | Report the pooled row as primary, give the per-key row beside it, and replace `main.tex:1015-1016` with: "each pair keeps the form with the wider gap; for steering this choice sets the sign on three cells (listed in Appendix X)" |
| Fig. `fig:share_prior_steering`, right panel: the caption says "steering at strength 1", but the plotted x values are the filed gaps, read at the grid-selected strength in the wider form | ours (bug) | `appD_built_pairs_more.tex:108-111`; `method_table/figures/paper_fig4_share_vs_prior_steering.csv` steering rows (x from `final.json`, e.g. half-known vs cipher at equal size: −0.237, per-key at the searched strength); E149 strength 1 on that pair: per-key +0.038, pooled +0.055 | The figure and the rank correlation −0.30 (`appD:102-103`, from `gate_1116.json`) are not the quantity the caption names. At strength 1 several points change sign | major | no | Re-plot from `prior_steering/results/stage1.json` strength-1 shares (pooled), recompute the rank correlation, or change the caption to "steering as filed (strength and form chosen per pair)" |
| Readout. Venhoff: (hybrid − base)/(thinking − base) in benchmark accuracy under generation. Ours: log-odds of the trained answer at one position over the fine-tune's own gain, unbounded and NaN under a gain floor | ours | `holes_fill/run.py:129-130`; `steering.py:129-130`; Venhoff Table 1 p. 7 | Ratio sensitivity: shrinking a side's update moves its share (−0.12 to −0.92 on one pair) | minor | yes, `main.tex:1150-1153` | Add "the share can exceed 1" to the appendix item |
| Fitting rows: E143 fits on the fine-tune's own training windows, and E135 on the even half of the read windows. Both share their keys with the read rows | ours | `holes_fill/run.py:164, 191`; `steering_copy_lookup/run.py:5-8` | With per-key vectors, every scored key was seen while fitting, which inflates per-key recovery | minor | partly (`main.tex:926-928` "over its own training windows") | State that read keys are seen while fitting |
| Ward et al. citation: they steer the reasoning fine-tune with a vector built from base-model activations, the opposite direction to ours | n/a (citation) | `main.tex:931-932`; Ward abstract p. 1, §3.2 p. 3 | The sentence is accurate, but it supports nothing in our method | minor | n/a | Cut it, or write "\citet{ward2025reasoning} go the other way, steering the fine-tune with a direction from its base model" |
| Pooled single-vector steering (inherent to ActAdd/CAA): one direction per behaviour cannot carry a key-dependent map, reused or new. Ciphers and rank-1 maps both read 0.13-0.23 pooled on the 96-key pairs | theirs | Rimsky §3; recomputation below (rank-1 vs cipher at 96 keys, pooled 0.15-0.23 on both sides) | Pooled steering tracks "carriable by one direction", not reuse, so a low reading does not mean "new" | minor | yes (`appE:90-91` "a gain that one vector can carry") | none beyond keeping that gloss |
| `sec5_results.tex:136-139` says "Steering" for the key-grid result, which is pooled-only (`sec5_3:64` says "pooled steering") | ours (wording) | `sec5_3_mechanism_vs_prior.numbers.md:58, 84` (`.steering_pooled`) | none | minor | n/a | Write "pooled steering" in `sec5_results.tex:136` |

### Numbers that rest on the wider-gap selection

Recomputed from the saved results (`~/commute-data/agents/redteam-lead/fidelity/scripts/both_forms.py`,
`s1_forms.py`, both run with cpu-run). Each cell gives the gap (reuse side mean minus new side mean) and the seed-range
call. The share's gap is positive on all nine built pairs.

| pair | kept | pooled | per-key |
|---|---|---|---|
| copy vs lookup (E62) | pooled | +0.412 apart | −0.023 no call |
| copy vs random maps | pooled | +0.470 apart | +0.048 no call |
| lowercase vs ciphers | pooled (no per-key) | −0.053 no call | — |
| half-known vs cipher, equal size | per-key | −0.177 apart | −0.237 apart |
| half-known vs cipher, own steps | per-key | **+0.114 apart** | **−0.120 apart** |
| rank-1 vs cipher, step 10 | pooled | −0.265 apart | −0.149 apart |
| copy vs cipher, step 10 | pooled | +0.238 apart | −0.044 no call |
| rank-1 vs cipher, 96 keys | pooled | +0.028 no call | +0.026 no call |
| copy vs prior-matched cipher, 96 keys | pooled | +0.594 apart | +0.065 apart |
| twin: half-known vs shuffle | pooled | +0.043 apart | +0.030 no call |
| twin: class map vs shuffled | pooled | −0.074 apart | +0.001 no call |
| twin: case label | per-key | **+0.101 apart (real higher)** | **−0.506 apart** |
| twin: repeat label | per-key | **−0.007 no call** | **−1.087 apart** |

Reported numbers that rest on the selection:
1. Scorecard row "Steering 4 / 3 / 2" (agree / reverse / no call; `appD_built_pairs_more.tex:39`). Pooled only gives
   5 / 2 / 2, and per-key only (pooled where none exists) gives 1 / 3 / 5.
2. "steering ... reverse three to five of the nine built pairs" (`sec5_results.tex:170-171`,
   `sec5_3_mechanism_vs_prior.tex:87-88`, `appD:82-83`). Steering's 3 depends on the selection (2 pooled) and on the
   strength search (0 pooled at strength 1).
3. Twin column "twin higher on both label twins" (`appD:39`) rests wholly on per-key. With pooled, case label reads the
   real task higher (apart) and repeat label makes no call.
4. Size-matched column "the same 7 apart" (`appD:39`) uses the kept form per pair (per-key on half-known vs cipher at
   own steps, −0.92; `size_matched/results/summary.json`).
5. "−0.12 to −0.92" (`main.tex:1150-1156`). Disclosed there as depending on the form.
6. Fig. `fig:share_prior_steering` right panel, and the rank correlation −0.30 (`appD:102-103`). These are the filed,
   selected gaps, not strength 1 (see the table row above).
7. "The prior and steering together call all nine" (`appD:98-101`) is E149's strength-1 read in the kept form (per-key
   on both half-known pairs). The claim holds either way: the prior carries those pairs, and pooled at strength 1
   separates them too.
8. `appF_earlier_results.tex:162` "steering ... four" (sign agreement with R under wording A) presumably uses the
   filed gaps. I could not reproduce 4: the filed gaps have 5 of 9 positive. Its source is `proxies_vs_r`, not checked.

Not affected (pooled only, no selection): the key-grid numbers 0.74 / 0.59 / 0.17 / 0.23 and "about thirty"
(`sec5_3:61-65`, `sec5_results:136-139`); the real-text steering (`src/commute/realistic_reads.py:252-255`); the
disagreeing pair's 0.175 (`experiments/disagreeing_pairs2/run.py:111`). The procedural-task "0.46 for steering"
(`sec5_results:203`) was not traced to code.

### Citation check (steering)
- `turner2023`: correct paper (arXiv 2308.10248, authors match). It supports "follows activation addition" only loosely,
  given the departures above.
- `rimsky2024`: correct paper (ACL 2024; the arXiv PDF now lists the first author as Nina Panickssery). It supports
  "mean-difference vectors".
- `venhoff2025base`: correct paper (arXiv 2510.07364, v4 says ICML 2026, PMLR 306; the bib's booktitle matches). It
  does **not** support "This is the test of Venhoff" (`main.tex:930`), nor its use as the source of the scorecard row
  (`sec5_results:171`, `appE:91`).
- `ward2025reasoning`: correct paper. The sentence is accurate but irrelevant to our construction.

## 2. Effective rank

The per-matrix formula matches Roy & Vetterli's Definition 1 (p. 606): p_k = σ_k / ‖σ‖₁, natural log, 0 log 0 = 0, and
erank = exp H(p). This holds in `src/commute/update_spectrum.py:24-29`, `src/commute/qwen_reading.py:357-364` and
`experiments/baseline_weights/run.py:60-62`, and the rank-1 and rank-10 controls read 1 and 10. The bib entry is
correct (EUSIPCO 2007, pp. 606-610). The rows below are departures around that formula.

| departure | ours/theirs | evidence | consequence for the reported number | severity | disclosed? | fix |
|---|---|---|---|---|---|---|
| Roy defines erank for one matrix. Ours averages it over the 24 matrices, weighted by ‖δW_m‖² | ours | `update_spectrum.py:7, 37-43`; `baseline_weights/run.py:64-67` | The number is dominated by the matrices that move most. It stays scale-invariant overall | minor | yes, `main.tex:937-938` | none |
| Reading direction. The paper's rationale is "a new task may need a change of higher rank" (`appE:88`), which predicts the new side has the higher erank. The scorecard counts sign agreement with the share, log10(reuse/new) > 0, as "agrees". So its 7 agreements are pairs where the side built to reuse has the higher-rank update: copies 475 / 466 against ciphers 407 / 405 | ours | `plot_final.py:30` (RATIO, not inverted); `final.json` erank gaps (e.g. copy vs lookup +0.058, copy vs prior-matched cipher +0.068); `sec5_3:65`; `appD:42` | Under the paper's own rationale, effective rank reverses 7 of 9 pairs rather than agreeing. "All three side with the mechanism" (`sec5_results:139`), "effective rank ... group the tasks the same way" (`sec5_3:12`) and `sec7_discussion:38-39` hold only as grouping, not as the direction the baseline predicts | major | no | Either drop the rationale, or add after `appE:88`: "Read this way, it points the wrong way: on seven of nine pairs the side built to reuse has the higher effective rank." Recount the scorecard row against the rationale's direction |
| The ℓ1-normalised spectrum weighs the tail of small singular values heavily, so a dense low-level floor (optimiser noise) pulls erank toward a Gaussian's (~700 here) | theirs | Roy Def. 1 p. 606 (ℓ1, not ℓ2, weights); gaps are at most 0.07 in log10 (`sec5_baselines_scorecard_caveats.numbers.md:22, 32`); Gaussian ~700 (`sec5_3:65`) | The pair gaps are at most ~17% and may track how much of the update is noise rather than task structure. On real text it adds nothing beyond base loss (`sec5_results:195`) | minor | partly (the random band is shown) | Report the top-10 energy share (`update_spectrum.py` already computes `top10`) beside it |
| Precision of the input: `US.load` accepts bf16-saved δW. A δW stored in bf16 carries rounding that adds a full-rank floor | ours | `update_spectrum.py:49-53` | Unverified: I did not check which saved updates are bf16. If some are, their erank is inflated by an unknown amount | minor | no | List the saved dtype per pair. Where bf16, recompute from fp32 checkpoints or state the floor |
| Qwen: erank per matrix over 196 matrices. The paper does not say how the aggregate 1,329-1,456 was formed (`H/summary.json`) | ours | `experiments/qwen_reading/run.py:210`; `sec6_scaling_qwen.tex:189-192`; `sec6_scaling_qwen.numbers.md:143` | Unverified whether it matches the 67M's ‖δW‖²-weighted mean | minor | no | State the aggregation in `sec6` setup |

## 3. Update size

The open note at `main.tex:982-985` is confirmed against the code. Every row of the scorecard's size column uses one of:
- the sum of per-matrix Frobenius norms (`src/commute/elicit.py:58-60`, used via `induction_cue/run.py:291`,
  `keydep.py:116, 124-129`, `open_pair`, `case_label/run.py:277`, `open_pair_instruments/ladder.py:115`);
- the root of the summed squares, for copy vs random maps (`method_table/build.py:298`);
- the summed squares, unrooted, for lowercase vs ciphers (`plot_final.py:104-106`, labelled "(not rooted)" in
  `final.json`).

A fourth form, ‖δW‖/‖W₀‖ with the norms summed in squares (`src/commute/realistic.py:191-194`), is used on the key
grid and on real text. Update size is a plain baseline, so no citation is needed.

| departure | ours/theirs | evidence | consequence for the reported number | severity | disclosed? | fix |
|---|---|---|---|---|---|---|
| Three norms across the scorecard's pairs (above) | ours | the files above; `final.json` size cells' `source` and `form` | Lowercase vs ciphers: the gap doubles (−0.098 against −0.049 rooted), call unchanged. Copy vs random maps: root of summed squares −0.181 against summed Frobenius −0.175 (my recomputation from `baseline_weights/results/results.json` per_matrix norm2), call unchanged. No call in "0 / 6 / 3" changes on the pairs I could check | minor | yes, but only as the draft `\open` note `main.tex:982-985` | Re-read `realistic.update_size` on every saved state (CPU only) and rebuild the column, then delete the `\open` |
| The paper defines size as ‖δW‖/‖W₀‖ (`main.tex:1017`, `appE:21`). No scorecard cell divides by ‖W₀‖, and `main.tex:936` says "the norm" | ours | `elicit.py:58-60`; `build.py:298` | No effect on any gap: both sides share W₀, so the ratio cancels in log10(reuse/new). Absolute sizes quoted are in different units | minor | no | Make `main.tex:936` match `1017`, and say the scorecard gap is unchanged by the normalisation |
| The 96-key pairs' sizes come from E58's original updates (`open_pair/results/amend3,4,7.json`), while their other readings use E147's refits | ours | `final.json` size sources for e58_* vs steering/erank sources (`refit_96key`, `refit_rank`) | `main.tex:1145-1147` says the refits' own sizes give the same calls | minor | yes, `main.tex:1145-1147` and `\open` | none beyond the re-read above |
| Repeat-label twin: the real side mixes steps 30 and 100 | ours | `main.tex:1141-1146` | "twin larger" compares step 30 with step 100 | minor | yes, `main.tex:1141-1146` | none |

## Verdicts

**Steering.** Not entitled to its citation. "Steering \citep{venhoff2025base}" and "the test of Venhoff" describe a
method we did not run: theirs optimises vectors and gates them by an SAE, ours adds an unoptimised mean difference.
ActAdd/CAA is acceptable only as "after" (Turner, Rimsky), and Minder's activation difference is the closer source.
Its reported reversals rest on the strength search and on the wider-gap form selection, which the paper describes
wrongly. The plain pooled form at strength 1 reverses none of the nine pairs.

**Effective rank.** Entitled to its label and to Roy & Vetterli: the formula is exact, and the weighted average over
matrices is disclosed. But the paper's stated rationale (new needs higher rank) points the other way from what it
counts as "agreeing", so its "7 agree" and "sides with the mechanism" are wrong in direction under that rationale.

**Update size.** Entitled to the label, which needs no citation. The three-norms note is accurate. No call changes on
the pairs checked, but the column should be re-read with one norm and the definition at `main.tex:936` aligned with
`main.tex:1017`.

Row count: blocker 1 (steering); major 5 (steering 4, effective rank 1); minor 15 (steering 7, effective rank 4,
update size 4).

Not checked: the procedural-task steering correlations (`sec5_results:202-204`); `appF:162`'s "four" (proxies_vs_r);
the Qwen steering comparator (`src/commute/qwen_comparators.py:254-263`, not cited in the paper text); the saved dtype
of each δW; how `size_matched` sizes its shrink.

## 4. Follow-up: the size column under one norm (2026-11-23)

The question came from the other reviewer. If every built pair's size is read with one norm (the global
‖δW‖_F / ‖W₀‖ of E187, `experiments/size_matched/results/m67.json .pairs.<p>.sizes`), does the column read 0 agree /
7 reverse / 2 no call, not 0 / 6 / 3? Script: `~/commute-data/agents/redteam-lead/fidelity/scripts/one_norm_size.py`
(read-only, run with cpu-run).

**Confirmed.** Under that one norm the nine built pairs read **0 agree / 7 reverse / 2 no call**. The scorecard reads
0 / 6 / 3.
- Mid vs cipher at matched size (`e54_matched`) goes from no call (gap −0.0056) to apart, new larger on every seed. The
  extremes are reuse max 0.00502714 and new min 0.00502952, as the other reviewer said. The log10 gap is −0.0082.
- Copy vs cipher at step 10 (`open_pair_c3_copy`) flips its uncalled sign: +0.0008 becomes −0.0043, no call both times.
- Every other call is unchanged.

**Which states.** The m67 sizes are E187 stage 0(a)'s sizes of the as-saved updates, before any rescaling
(`size_matched/run.py:129-130, 172-181`; c is applied only in later reads). The state lists come from E136's inventory,
plus E147's refits for the 96-key pairs. Those are the states the scorecard's share, steering and erank cells read, and
not the size-matched rerun's rescaled states. They are not always the states the scorecard's size cells read:
- The 96-key pairs' scorecard sizes come from E58's original updates (`open_pair/results/amend*.json`), while m67 sizes
  E147's refits.
- On mid vs cipher at own steps (`e54_own`), the scorecard size has 5 new-side values, but m67 has the 2 states that
  have a share.
- On `e54_matched` both have 5 a side. The difference is the norm: the scorecard uses summed Frobenius via
  `keydependent_elicit` / `keydependent_patch`, and m67 the global Frobenius norm.

So the one-norm column is the better match to the other readings. It differs from the scorecard's size cells in the
norm and, for 3 pairs, in the states too.

**Is the e54_matched change substantive? No.**
- The sides' means differ by 1.9% (new larger).
- The seed ranges clear each other by 2.4e-6 in units of ‖W₀‖. That is 0.047% of the size, not 0.0005%: 0.00047 is the
  fraction, not a percentage.
- The pair was built equal in size by picking the cipher checkpoint nearest the half-known map's *summed* Frobenius norm
  (`keydep.py:120-129`). Under the global norm it is still equal to about 2%.
- The call comes from tight seed ranges under a rule with no noise band, not from a real size difference. A 2% size
  difference cannot explain the share's 0.52 gap there, and the share keeps its separation after size matching.
- The sign flip on copy vs cipher at step 10 is between two no-call values within 1% of each other, so it is
  immaterial.

**Fix.** When the column is re-read with one norm (the fix in Section 3, row 1), report 0 / 7 / 2. Change
`appD_built_pairs_more.tex:41`'s "equal by construction" to "within 2% by construction; under one norm the new side
is larger on every seed of the matched pair". Keep the reading that size never sides with the share: under one norm it
reverses 7 and agrees with none. Severity stays minor: the label is right and only a tally line changes. A seed-range
rule with no band can call a 2% difference, so the text should say so.
