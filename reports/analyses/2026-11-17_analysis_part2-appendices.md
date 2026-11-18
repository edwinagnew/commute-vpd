# Part 2 appendices: how each baseline was computed, and how each 67M pair was built

This page writes down, from the code that produced them, how every baseline on the 67M scorecard was computed and how each of its thirteen task pairs was fine-tuned. Every baseline is a short deterministic reading of the saved weight change on the same rows as the share, but the size column mixes three different norms, and some sides are read at other steps than their labels say. For `GOALS.md` this page changes no reading, since nothing was recomputed; the paper gets its recipes, and the places where the scorecard's labels and the code disagree are listed near the end.

The scorecard is `experiments/method_table/figures/gate_scorecard_1116.csv`, built by
`experiments/method_table/gate_plot_1116.py` from `experiments/method_table/results/final.json`, which
`experiments/method_table/plot_final.py` writes from each source run's saved JSON. The size-matched readings come from
`experiments/size_matched/results/m67.json`, where the larger side of each built pair is scaled by one number to the smaller
side's mean size and read again. Everything below was read from code and saved results only; nothing was
run.

## Appendix A. Baseline implementations

Every reading below is taken on the same saved weight changes δW and, unless a paragraph says otherwise, on the same read
rows as the re-gating share: the held-out windows of each pair's own task, as `experiments/share_untrained/windows.py:load_read`
loads them. Each reading is in the share's unit, the log-odds of the trained answer at the answer position against the whole
vocabulary, except on the two label twins, where every reading uses the contrast between the two label tokens. On the
scorecard each cell is the side built to reuse minus the side built to be new, and a cell is called "apart" only when every
seed of one side lies beyond every seed of the other, with at least two seeds a side
(`experiments/method_table/plot_final.py:score`, `experiments/method_table/gate_plot_1116.py:apart`). No reading below has a
seed of its own except where stated: each is a deterministic function of the saved update.

### The re-gating share, for reference
**What it measures.** How much of the fine-tune's gain on the trained answer comes back when the original model's
subcomponents are kept fixed and only Γ's causal importance values change. **Recipe.** With the released decomposition of
the original model held fixed, L00 is the CI-masked original model under Γ's importance values computed at W0, and L01 is
the same masked model under the importance values Γ computes at W0 + δW. The share is $r = (\bar L_{01} - \bar L_{00}) / (\bar T_1 - \bar T_0)$,
each a mean over read rows of the answer's log-odds, T the target model before and after the update, all 24 decomposed
matrices of the four blocks at once, every position. It is undefined when the gain is below 1e-6 in magnitude; scoring
elsewhere uses a gain floor of one log-odds unit. It reads 0 when Γ's values do not move and 1 when moving them alone
recovers the whole gain. The pair's gap is the reuse side's seed mean minus the new side's. **Code.**
`src/commute/local_faith.py:route_split`, called by `experiments/share_untrained/run.py:Job.trained`, under bf16 autocast; on
the four 96-key pairs the same reader on the refit updates of `experiments/refit_96key/run.py`; on the two label twins
`src/commute/masked_reads.py:share_cells` in the contrast unit, from `experiments/real_vs_twin/run.py`. **Source.** VPD's
masked forward and Γ, as released in Goodfire's `param-decomp`; the share itself is defined here.

### Untrained-SVD share
**What it measures.** Whether the share's separation needs a trained decomposition at all: the same do-nothing share on a
decomposition of W0 that has had no training. **Recipe.** Each of the 24 matrices is split by its own singular value
decomposition in float64, one subcomponent per singular pair, kept in fp32. At each matrix, window and position, the k_m
subcomponents with the largest contributions $|v_i^\top x|\,\|u_i\|$ are kept and the rest masked out, x that matrix's
input on the current forward. k_m is the mean number of trained subcomponents with causal importance above one half at W0,
counted over all pairs' read windows and positions, rounded, and at least one. It runs from one on the key and query
matrices to thirty on the last down-projection. The masked forward replaces each matrix's output by $(g \odot xV)U$ with W0's singular
vectors whatever weights the model holds, so L00 uses the masks at W0 and L01 the masks the updated model's inputs give.
The share is $(\bar L_{01} - \bar L_{00})/(\bar T_1 - \bar T_0)$ in fp32, nan when the gain is under 1e-6. The gap is formed as
for the share. A Haar-random rotation of the SVD, seed 0, is computed beside it and is not on the scorecard. **Code.**
`src/commute/untrained.py:svd_decomposition`, `top_k_gates`, `do_nothing_cells`; `experiments/share_untrained/run.py:Job.read_state`
and `control_d` for k_m; scorecard key `.states[<state>].svd` of `experiments/share_untrained/results/stage1.json`, and
`.e136.svd` of the refit file on the 96-key pairs. **Source.** Defined here as a null decomposition; the top-k rule by
contribution size stands in for Γ, which an SVD does not have.

### The neuron version of the share
**What it measures.** The same question asked of MLP neurons with no decomposition: how much of the gain comes back when
only the neurons' firing changes. A neuron is one of the 3,072 outputs of a block's first MLP matrix after the GELU, so
the model has 12,288. **Recipe.** All cells run on the target model in fp32, with attention weights at W0. T0 and T1 are the
original and updated model. The whole-width form r_n patches the updated model's firing, at every block and position, into
the original model, which keeps W0's output matrices: $r_n = \overline{L(O_0, A_1) - T_0}\,/\,\overline{T_1 - T_0}$. The
half form r_half instead runs W0's own first MLP matrix and GELU on the MLP input the updated model produces, so it counts
only re-firing caused upstream of each MLP. Shares are a mean numerator over the mean gain, nan when the gain is under one
log-odds unit. Read rows: every held-out window of the task, in batches of 64, from 75 to 500 windows a task; on the
96-key pairs the refit updates; on the twins the first 256 fit windows in the contrast unit. **Code.**
`src/commute/neuron_r.py:batch_cells` and `shares`, run by `experiments/neuron_r/run.py:stage1`; `experiments/refit_96key/run.py`
for the 96-key pairs; `experiments/neuron_twins/run.py` for the twins. **Source.** Activation patching of hidden units
from one model run into another, as in causal mediation analysis, Vig et al. 2020; defined here as a share of a
fine-tune's gain, with the departure that every neuron at every position is patched at once.

### Neurons for half the gain
**What it measures.** How concentrated the re-firing is: the fewest neurons whose re-firing alone carries half of the
r_half numerator. **Recipe.** Neurons are ranked once per state by W0's gradient of the answer's log-odds with respect to
each neuron's firing, times the change in its firing under the half form, summed over windows and positions. For each k on
a grid of powers of two up to 8,192, then all 12,288, the half-form firing is patched on the top k neurons and W0's firing
on the rest. k_half is the first k whose numerator reaches half of the full r_half numerator. It is undefined when the r_half share
is not above 0.1, its numerator is not positive, or the gain is under the floor; such states are left out of the column.
The scorecard gap is $\log_{10}$ of the new side's mean over the reuse side's, so a positive gap means fewer neurons on the
reuse side. **Code.** `src/commute/neuron_r.py:ranking`, `grid_numerators`, `k_half`; `experiments/neuron_r/run.py:stage2`
and `experiments/k_half_fill/run.py` for the states the first run did not score. **Source.** The ranking is attribution
patching, a first-order estimate of each unit's patching effect, Syed, Rager and Conmy 2023; the half-gain count is
defined here.

### Top-neuron share
**What it measures.** Whether the gain lives in the set of most strongly firing neurons: W0 is run with each block's MLP
firing kept only on its top-K neurons by absolute firing, the set taken from the updated model's own forward or from
W0's. **Recipe.** At every block, window and position, N01 keeps the K neurons of largest |firing| in the updated model's
forward and N00 those of W0's, zeroing the rest, both applied to W0. The share is $\overline{N_{01} - N_{00}}\,/\,\overline{T_1 - T_0}$,
nan under a gain of one log-odds unit. K is either 64 per block, or matched per block to the decomposition's down-projection
on-count k_m, which is 15, 3, 6 and 30 for the four blocks. The scorecard takes, per pair, whichever of the two K forms
gives the wider gap between the sides. **Code.** `src/commute/masked_reads.py:topk_logits` and `top_masks`, run by
`experiments/holes_fill/run.py:Job.neuron`; `.holes_fill.neuron.<K>.share` in the refit file for the 96-key pairs; the
contrast unit on the twins from `experiments/real_vs_twin/run.py`. **Source.** Per-token top-K neuron sparsity in the framing
of Goodfire's work on neurons as an interpretable basis; the share is defined here.

### Goodfire-style neuron job retention: the top-240 overlap and change on neurons in both sets
**Which column.** On the 2026-11-16 scorecard this reading is the column "top-240 overlap", with its companion "change on
neurons in both". The neuron-job retention of `experiments/neuron_jobs_k` is not a column of that scorecard; it is described
at the end of this paragraph. **What the overlap measures.** Whether the original model's most strongly firing neurons stay
the most strongly firing after the fine-tune. **Recipe.** On each read row at the answer position only, each block's top 240
of 3,072 neurons by |firing| are found in W0's forward and in the updated model's; 240 is Goodfire's share of a layer. The
overlap is the Jaccard index of the two sets, averaged over rows and then over the four blocks. The companion reading
divides the squared firing change summed over neurons in both sets by that summed over all neurons, pooled over rows and
blocks. W0 reads 1 and nan. The gap is the reuse mean minus the new mean. A random update with each matrix's Frobenius norm
matched to the side's first seed, seed 0, is the floor and does not enter the scorecard cell. **Code.**
`src/commute/masked_reads.py:top_overlap`, run by `experiments/neuron_overlap/run.py:measure`; key `.score.<pair>.overlap` of
`experiments/neuron_overlap/results/stage2.json`. **Job retention, off the scorecard.** Per state and block, the 32 neurons
with the largest mean firing change on the read rows are taken. For each, its top 20 Pile validation windows by peak firing
are found at W0 and at the update, out of 2,000 windows. Retention is the shared fraction, averaged over the 128 neurons.
The gain kept under each block's top-K is read beside it at three values of K. Code: `experiments/neuron_jobs/run.py:Job.read`,
rerun by `experiments/neuron_jobs_k/run.py`. **Source.** Goodfire's per-token layer top-K keeps 320 of 4,096 neurons, ranked by
activation times router weight; here 240 of 3,072, ranked by |firing| alone, since the 67M is dense and has no router.

### Steering as a mean activation shift
**What it measures.** Whether one added vector reproduces the fine-tune's gain: if the update only shifts a direction the
model already reads, a fixed shift at the answer position should do what the update does. **Recipe.** At each block l, the
vector is the mean over the fine-tune's own training windows of the updated model's block-l input minus W0's, at the answer
position. The pooled form has one vector; the per-key form has one per key token at the answer position, and on the twins
one per word before the trigger. Steering adds strength times the vector to W0's block-l input at the answer position only.
Block and strength are chosen per state on up to 512 fitting windows, by the largest mean rise in log-odds, ties to the
lower block and strength. The grid is the four blocks by the strengths 0, 0.5, 1, 2, 4 and 8. The share is the steered rise on the read rows over the
update's own gain there, nan under a gain of one log-odds unit. Eight norm-matched random vectors, seed 0 plus the state's
index, are read beside it and are not in the cell. The scorecard takes, per pair, whichever of the pooled and per-key forms
gives the wider gap; lowercase against ciphers has no key, so it is pooled only. On the copy-at-a-new-marker pair the
vectors are fit on the even half of its 200 read windows and read on the odd half. **Code.**
`src/commute/fair_steering.py:vectors`, `logits_steered`, `pick`, run by `experiments/holes_fill/run.py:Job.kind`;
`experiments/steering_copy_lookup/run.py` for the copy-at-a-new-marker pair; `experiments/refit_96key/run.py` for the 96-key
pairs; `experiments/real_vs_twin/run.py` for the twins. **Source.** Activation addition, Turner et al. 2023, and mean-difference
steering vectors, Rimsky et al. 2024. Departure: the vector is the mean difference between the fine-tuned and the original
model on the same inputs, not between two sets of contrasting prompts, and its block and strength are tuned per state.

### Base-model prior on the answer
**What it measures.** How likely the original model already found the trained answer before any fine-tune. **Recipe.** The
geometric mean over the read rows of W0's probability of the trained answer at the answer position, one value per map, not
per seed, recorded as $\log_{10}$. Where an earlier run had saved the value it is read from that file, otherwise it is computed
by one forward. On lowercase against ciphers the windows are rebuilt from the cached Pile rows and the answer is each
window's true next token, over 500 windows. The gap is the reuse value minus the new value; with one value a side the
cell gets no call. **Code.** `experiments/prior_vs_share/run.py:forward_prior` and `stage1`, `src/commute/keydep.py:geo_rate`;
`experiments/prior_lowercase/run.py` for lowercase against ciphers. **Source.** Defined here as a reference reading, not a
method from the literature.

### Update size
**What it measures.** How far the fine-tune moved the weights, which a share could simply be tracking. **Recipe.** The
value each source run saved as the update's norm over the 24 decomposed matrices; the scorecard gap is $\log_{10}$ of the reuse
side's mean over the new side's. The unit is not the same on every row: most rows hold the summed Frobenius norm saved by
the pair's own run; copy against random maps holds the root of the summed squared norms; lowercase against ciphers holds the
summed squared norm, see Appendix B. The size-matched readings use a third, uniform size: the root of the summed squared
norms of δW over that of W0. **Code.** Each source run's `norm` field, listed per row in Appendix B;
`src/commute/update_spectrum.py:state` for `norm2`; `experiments/size_matched/run.py:Readers.size`. **Source.** The Frobenius
norm, standard.

### Effective rank
**What it measures.** Whether one side's update is spread over fewer directions. **Recipe.** Per matrix, with s the
singular values of δW in float64, the effective rank is $\exp(H(p))$ with $p = s/\sum s$, so a rank-one change reads 1 and k
equal singular values read k. A state's value is the mean over the 24 matrices weighted by each matrix's squared norm over
the total. The gap is $\log_{10}$ of the reuse side's mean over the new side's. On the 96-key pairs the refit updates are
read. **Code.** `src/commute/update_spectrum.py:matrix`, `aggregate`, run by `experiments/update_svd/run.py` and
`experiments/refit_rank/run.py`. **Source.** Roy and Vetterli 2007. Departure: the paper defines it for one matrix; here it is
averaged over matrices with squared-norm weights.

## Appendix B. Per-pair numbers and recipes

### How the fine-tunes were run

Every side except lowercase against ciphers uses one recipe, `src/commute/keydep.py:finetune` calling
`src/commute/finetune.py:finetune_arm_a`. Each runs Adam at a constant learning rate of 3e-5 with no warm-up, batch 32,
gradient clipping at 1 and a KL term of weight 1 to the original model at every position other than the answer. It
updates all 24 decomposed matrices in full under bf16 autocast with fp32 weights, from W0, for 300 steps with
checkpoints, and a state is read at one checkpoint. A "seed" changes only the order of the training batches. The
settings are those filed in `experiments/keydependent_elicit/results/predictions.json`, key `settings.finetune`, and the
label pairs' `experiments/fresh_reuse_pair/results/predictions.json`. Lowercase against ciphers instead uses continued
pre-training, `src/commute/lowercase.py:arm_b_states`: next-token cross-entropy over whole 512-token Pile rows, no KL
term, Adam at the same rate, batch 8, 300 steps over 2,400 rows each seen once, with seed 1 a fixed reordering of the rows.

| pair | side: how it was built | data | step read, seeds | source folder |
|---|---|---|---|---|
| Copy at a new marker vs lookup | reuse: a key word A, then a word B, then A and a marker; the answer at the marker is B, which the model's in-context copying already predicts | 32 keys; 800 training and 200 read windows from disjoint Pile rows and disjoint B words | 30; seeds 0-2 | `experiments/induction_cue` |
| | new: same windows; the answer is one fixed word per key, prior-matched to B, a new key-to-word table | as above | 30; seeds 0-2 | `experiments/induction_cue` |
| Copy maps vs random maps | reuse: marker then key; the answer is the key itself, on two key sets | 32 keys a task; 500 windows, 403 training and 97 held out by row | 10; seeds 0-1 per task | `experiments/regating_table`, rebuilding `experiments/new_vs_existing` |
| | new: the answer is a random bijection to rare word tokens the original model gives under 1e-4 | as above | 10 on one task, 30 on the other; seeds 0-1 | `experiments/regating_table` |
| Lowercase vs ciphers | reuse: continue training on lowercased Pile text, which the original model already continues well | 2,400 training rows; read on up to 500 positions of Pile validation rows | 300; seeds 0-1 | `experiments/performed_anchor` |
| | new: Pile text with a derangement of 50 word tokens, and a broader one of 500 | as above | 300; seeds 0-1 each | `experiments/performed_anchor` |
| Half-known vs cipher, equal size | reuse: marker then key; the answer is the original model's own second-to-twentieth guess, already given some probability | 32 keys; 500 windows, 403 training and 97 held out | 10; seeds 0-4 | `experiments/keydependent_elicit`, seeds 2-4 in `experiments/keydependent_patch` |
| | new: a random bijection to rare word tokens, read at the checkpoint whose size is nearest the half-known state's | same windows | 9; seeds 0-4 | as above |
| Half-known vs cipher, own steps | reuse: as above | as above | 10; seeds 0-4 | as above |
| | new: the cipher at its own learned step | same windows | 30; share read on seeds 0-1 | as above |
| Rank-1 vs cipher, step 10 | reuse: the answer is the original model's top guess after the key, trained on 24 of 32 keys | the 24 trained keys' windows; 75 held-out read windows | 10; seeds 0-7 | `experiments/open_pair`, seeds 3-7 in `experiments/open_pair_instruments` |
| | new: the cipher on the same keys, read at its size-matched step | as above | 8 or 9, not 10; seeds 0-7 | as above |
| Copy vs cipher, step 10 | reuse: the answer is the key, on the 24 trained keys | as above | 10; seeds 0-7 | as above |
| | new: the same cipher states as the row above | as above | 8 or 9; seeds 0-7 | as above |
| Rank-1 vs cipher, 96 keys | reuse: the top-guess map on 96 trained keys of 128 | 2,000 windows for the 128 keys; the 96-key subset counts are not saved | 30; refit seeds 0-2 | `experiments/open_pair`, refit in `experiments/refit_96key` |
| | new: the cipher on the same keys | as above | 30; refit seeds 0-2 | as above |
| Copy vs prior-matched cipher, 96 keys | reuse: the answer is the key | as above | 30; refit seeds 0-2 | as above |
| | new: a bijection whose answers the original model already gives a small, matched prior | as above | 30; refit seeds 0-2 | as above |
| Half-known vs its shuffle, 96 keys | reuse: the half-known map on 96 keys | as above | 30; refit seeds 0-2 | as above |
| | new: the same answers cycled among the trained keys | as above | 30; refit seeds 0-2; one saved size | as above |
| Class map vs its shuffled twin | reuse: four key classes from the embedding's two leading directions, one word per class | as above | 30; refit seeds 0-2 | as above |
| | new: the same four words with class labels shuffled over the keys | as above | 30; refit seeds 0-2 | as above |
| Case label: real vs memorising twin | reuse: label A when the word before the marker is capitalised, a rule the model can read | 600 training word tokens; 9,600 training windows; read on the first 256 fit windows | 30; seeds 0-2 | `experiments/case_label`, seed 2 in `experiments/real_vs_twin` |
| | new: each word token gets A or B by a seeded shuffle, so it can only be memorised | as above | 300; seeds 0-2 | as above |
| Repeat label: real vs memorising twin | reuse: label A when the word before the marker already occurred in the window | 300 training words; 9,600 training windows; first 256 fit windows read | 30 on seeds 0-1, 100 on seed 2; seeds 0-2 | `experiments/fresh_reuse_pair`, seed 2 in `experiments/real_vs_twin` |
| | new: labels by a seeded shuffle per word | as above | 100; seeds 0-2 | as above |

### The numbers

Each share is the re-gating share on that side's saved updates: the mean over seeds, with the seed range in brackets.
"Size as saved" is each source run's own norm field in the unit named; "size over W0's norm" is the size-matched run's
uniform measure, the root of the summed squared norms of δW over that of W0. In the size-matched column, the larger side of
each built pair is multiplied by one number c and read again; the smaller side keeps its unscaled share. On the label twins
the share is in the label-contrast unit.

| pair | side | seeds read for the share | update size as saved, mean | size over W0's norm, mean | share, unscaled: mean (range) | share, size-matched |
|---|---|---|---|---|---|---|
| Copy at a new marker vs lookup | reuse: copy at a new marker | 3 | 4.61 (4.57 to 4.62), 3 seeds, summed Frobenius | 0.0071 | 1.09 (1.09 to 1.10) | as unscaled: smaller side |
| Copy at a new marker vs lookup | new: lookup | 3 | 5.48 (5.45 to 5.50), 3 seeds, summed Frobenius | 0.0085 | 0.40 (0.38 to 0.42) | 0.40 (0.36 to 0.42), c 0.83 |
| Copy maps vs random maps | reuse: copy maps | 4 | 0.69 (0.68 to 0.69), 4 seeds, root of summed squares | 0.0049 | 0.79 (0.76 to 0.80) | as unscaled: smaller side |
| Copy maps vs random maps | new: random maps | 4 | 1.04 (0.75 to 1.34), 4 seeds, root of summed squares | 0.0075 | 0.28 (0.27 to 0.29) | 0.23 (0.19 to 0.27), c 0.66 |
| Lowercase vs ciphers | reuse: lowercase | 2 | 6.36 (6.25 to 6.46), 2 seeds, summed squares | 0.0181 | 0.76 (0.75 to 0.77) | as unscaled: smaller side |
| Lowercase vs ciphers | new: two ciphers | 4 | 7.96 (6.74 to 9.21), 4 seeds, summed squares | 0.0202 | 0.14 (0.12 to 0.17) | 0.13 (0.11 to 0.16), c 0.90 |
| Half-known vs cipher, equal size | reuse: half-known map | 5 | 3.18 (3.14 to 3.21), 5 seeds, summed Frobenius | 0.0050 | 0.76 (0.76 to 0.77) | as unscaled: smaller side |
| Half-known vs cipher, equal size | new: cipher, read at equal size | 5 | 3.22 (3.18 to 3.25), 5 seeds, summed Frobenius | 0.0051 | 0.25 (0.21 to 0.27) | 0.24 (0.21 to 0.27), c 0.98 |
| Half-known vs cipher, own steps | reuse: half-known map | 5 | 3.18 (3.14 to 3.21), 5 seeds, summed Frobenius | 0.0050 | 0.76 (0.76 to 0.77) | as unscaled: smaller side |
| Half-known vs cipher, own steps | new: cipher, its own steps | 2 | 6.03 (6.00 to 6.05), 5 seeds, summed Frobenius | 0.0096 | 0.25 (0.24 to 0.27) | 0.15 (0.14 to 0.16), c 0.52 |
| Rank-1 vs cipher, step 10 | reuse: rank-1 map | 8 | 3.15 (3.10 to 3.19), 8 seeds, summed Frobenius | 0.0049 | 0.86 (0.84 to 0.90) | 0.86 (0.84 to 0.90), c 1.00 |
| Rank-1 vs cipher, step 10 | new: cipher | 8 | 3.11 (3.03 to 3.26), 8 seeds, summed Frobenius | 0.0049 | 0.24 (0.22 to 0.25) | as unscaled: smaller side |
| Copy vs cipher, step 10 | reuse: copy map | 8 | 3.12 (3.09 to 3.15), 8 seeds, summed Frobenius | 0.0049 | 0.84 (0.83 to 0.86) | as unscaled: smaller side |
| Copy vs cipher, step 10 | new: cipher | 8 | 3.11 (3.03 to 3.26), 8 seeds, summed Frobenius | 0.0049 | 0.24 (0.22 to 0.25) | 0.23 (0.22 to 0.24), c 0.99 |
| Rank-1 vs cipher, 96 keys | reuse: rank-1 map, 96 keys | 3 | 5.06 (5.02 to 5.08), 3 seeds, summed Frobenius | 0.0079 | 0.86 (0.84 to 0.88) | as unscaled: smaller side |
| Rank-1 vs cipher, 96 keys | new: cipher, 96 keys | 3 | 6.11 (6.10 to 6.13), 3 seeds, summed Frobenius | 0.0097 | 0.30 (0.29 to 0.31) | 0.28 (0.27 to 0.29), c 0.82 |
| Copy vs prior-matched cipher, 96 keys | reuse: copy map, 96 keys | 3 | 4.73 (4.70 to 4.77), 3 seeds, summed Frobenius | 0.0073 | 0.77 (0.77 to 0.77) | as unscaled: smaller side |
| Copy vs prior-matched cipher, 96 keys | new: prior-matched cipher, 96 keys | 3 | 5.91 (5.90 to 5.92), 3 seeds, summed Frobenius | 0.0094 | 0.38 (0.37 to 0.38) | 0.37 (0.37 to 0.37), c 0.78 |
| Half-known vs its shuffle, 96 keys | reuse: half-known map, 96 keys | 3 | 5.34 (5.31 to 5.39), 3 seeds, summed Frobenius | not run | 0.73 (0.72 to 0.74) | not run |
| Half-known vs its shuffle, 96 keys | new: its shuffled twin | 3 | 5.97, 1 seed, summed Frobenius | not run | 0.48 (0.47 to 0.48) | not run |
| Class map vs its shuffled twin | reuse: class map | 3 | 5.25 (5.20 to 5.28), 3 seeds, summed Frobenius | not run | 0.24 (0.23 to 0.25) | not run |
| Class map vs its shuffled twin | new: its shuffled twin | 3 | 5.26 (5.23 to 5.32), 3 seeds, summed Frobenius | not run | 0.28 (0.27 to 0.28) | not run |
| Case label: real vs memorising twin | reuse: case label, real | 3 | 4.64 (4.60 to 4.68), 2 seeds, summed Frobenius | not run | 0.10 (0.10 to 0.11), label contrast | not run |
| Case label: real vs memorising twin | new: memorising twin | 3 | 12.97 (12.61 to 13.33), 2 seeds, summed Frobenius | not run | 0.12 (0.09 to 0.14), label contrast | not run |
| Repeat label: real vs memorising twin | reuse: repeat label, real | 3 | 4.76 (4.61 to 4.91), 2 seeds, summed Frobenius | not run | 0.09 (0.05 to 0.13), label contrast | not run |
| Repeat label: real vs memorising twin | new: memorising twin | 3 | 7.51 (7.47 to 7.55), 2 seeds, summed Frobenius | not run | 0.18 (0.15 to 0.21), label contrast | not run |

## Where the code and the existing labels disagree

- **The size column mixes three norms.** Most rows hold the sum of the per-matrix Frobenius norms
  (`src/commute/elicit.py:summed_norm`). Copy against random maps holds the root of the summed squared norms, computed in
  `experiments/method_table/build.py`. Lowercase against ciphers holds `norm2` from `src/commute/update_spectrum.py:aggregate`,
  which is the summed squared norm, yet `final.json` labels that cell "root of summed squared matrix norms" and the
  2026-11-10 method-comparison report says the same. Its gap on the scorecard is therefore twice the root form's gap. Within
  each pair both sides share a unit, so no call changes.
- **The 96-key pairs read size from other updates than their share.** Their share, untrained share, neuron readings,
  steering and effective rank are read on the refit updates of `experiments/refit_96key`, whose norms that run did not save;
  their scorecard size comes from the original step-30 fine-tunes of `experiments/open_pair`. The size-matched run measured
  the refits' own sizes. The shuffled twin of the half-known 96-key map has one saved size against three refit shares.
- **Seed counts differ between columns of one row.** The cipher at its own steps has five saved sizes but the share was read
  on two seeds; the two label twins have three seeds for the share and the neuron readings but two for size, effective rank
  and the untrained share, since their third seed was trained later by `experiments/real_vs_twin`.
- **The "step 10" cipher is not read at step 10.** On the rank-1 and copy pairs against the cipher, the cipher states are
  read at the checkpoint whose summed norm is nearest the same seed's half-known state: step 9 or 8 on every seed, while
  the rank-1 and copy maps are read at step 10 (`experiments/open_pair/results/run.json`, `.states.cipher_s<n>.ft.step`).
- **One repeat-label seed is read at another step.** The seed-2 real state that `experiments/real_vs_twin` trained is read
  at step 100, as its own learned-step rule gave, while seeds 0 and 1 of the same side are read at step 30; the scorecard's
  three-seed share for that side mixes the two.
- **The copy-versus-random states are rebuilt.** `experiments/regating_table` regenerated the earlier fine-tunes by their
  recipe; the held-out probabilities agree with the earlier run to about a thousandth, not exactly.
- **Two columns pick their form per pair.** Steering and the top-neuron share each have two forms, and `plot_final.py:wider`
  keeps, per pair, the one with the wider gap between the sides. The rule was filed by the runs that read them, but it is a
  selection on the quantity being compared.
- **The twins use another unit.** On the case-label and repeat-label pairs every reading is in the contrast between the two
  labels' logits, not in the answer's log-odds, so their values are not on the same scale as the other rows'.

## Departures from the literature

- **Effective rank** follows Roy and Vetterli 2007 per matrix, but a state's value is the squared-norm-weighted mean over
  the 24 matrices; the paper defines it for one matrix.
- **Steering** follows activation addition, Turner et al. 2023, and mean-difference vectors, Rimsky et al. 2024, but the
  vector is the mean difference between the fine-tuned and the original model on the same inputs rather than between
  contrasting prompts, and its block and strength are chosen per state by the largest rise on fitting windows.
- **The neuron version of the share** uses activation patching as in Vig et al. 2020, but patches every neuron at every
  position and block at once, and reports the result as a share of the fine-tune's gain.
- **Neurons for half the gain** ranks neurons by attribution patching, Syed, Rager and Conmy 2023, then patches the top k
  exactly; the half-gain count itself is defined here.
- **The top-240 overlap and the top-neuron share** take Goodfire's per-token top-K share of a layer, 320 of 4,096, as 240 of
  3,072 and rank by absolute firing alone, since the 67M has no router weights; Goodfire's light fine-tune under the mask is
  not done.
- **The untrained-SVD share** replaces Γ by a top-k rule on contribution size, since an SVD has no importance function; k
  per matrix is matched to the trained decomposition's on-count.
- **The prior and the update size** are reference readings defined here, with no source to depart from.

## Not run

- No reading was recomputed: every value on this page is read from a saved results file.
- The size-matched reading was not run on the four twin pairs; that run covered the nine built pairs only.
- No size-matched value exists for the untrained share or the neuron readings r_n and r_half. The size-matched run re-read
  the share, steering, the top-neuron share, neurons for half the gain and effective rank only.
- The refit 96-key updates' Frobenius norms were not saved by `experiments/refit_96key`, so no scorecard size exists for
  them.
- Goodfire-style job retention of `experiments/neuron_jobs_k` is not on the 2026-11-16 scorecard and is described, not
  tabulated, here.
- The part-1 re-weighting column of the scorecard is not a baseline and is not described.
