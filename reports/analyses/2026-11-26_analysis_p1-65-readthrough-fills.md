# Readthrough fills: four paper numbers re-read over the necessary set

Three of the paper's numbers were read over the mean-ablation circuit or a nearby set, and are re-read here over the
necessary set the definition uses; the rescale-only control was also read at two more seeds and at SAE latents. All
three numbers keep their sense, and the control reads near the full fine-tunes at VPD but two to six times above them
at SAE latents. Nothing here changes a claim of `GOALS.md`; the paper's lines change only in their stated values.

Two sets of variables appear throughout:
- **the necessary set**, written $K$ in the paper, is every variable that changes the behaviour when it alone is swapped
  between a fine-tune and its base model. It is found by a scan over 256 swap pairs.
- **the mean-ablation circuit** is the set an earlier step of the paper used, found by mean ablation; written $K^*$.

The paper's reuse reading $\rho$ is 1 when every changed variable in the set still computes what the base model's
computes, and 0 when none does.

## 1. main.tex:438, the SAE error terms' share of the rebuilt edit's change

The rebuilt edit zeroes the copy neurons and retrains the later MLPs until the model copies again, so it is built to
reuse. Read over every window position at SAE latents, the SAE's reconstruction error carries 0.781 of that edit's
change over the necessary set. Over the mean-ablation circuit it carried 0.8294, which the paper rounds to 0.83.

The test filed before the run asked that the lowest edit built to reuse read at least 0.2 above the highest new edit or
random update. Over the necessary set the rebuilt edit reads 0.1275, against 0.135 over the mean-ablation circuit. The
highest new edit or random update reads 0.023 over the necessary set, against 0.020 before. The margin is therefore
0.104 over the necessary set, against 0.115 over the mean-ablation circuit, and the test still fails by the same rule.

| quantity | over the necessary set | over the mean-ablation circuit |
|---|---|---|
| error terms' share of rebuilt's change | 0.781 | 0.8294 |
| rebuilt edit's reading | 0.1275 | 0.135 |
| highest new edit or random update | 0.023 | 0.020 |
| margin, filed bar 0.2 | 0.104 | 0.115 |

**Caveat on the random group.** Over the necessary set the random group holds two random updates, one matched in size
and one matched in KL. Both were read in the standard-figure run, from the seeds of the run before it. The fifteen random
updates matched to each fine-tune, which the earlier reading also held, have no reading over the necessary set. The
highest of the group is the KL-matched draw.

The standard-figure run also read one random update beside each built edit. The highest of those, beside the doubled
head V edit, reads 0.061. Counting them would make the margin 0.067. The filed rule does not name these updates, so the margin
it gives is 0.104.

**For writeup-lead.** Line 438's "0.83" becomes 0.78 over $K$. Its other three numbers are also values over $K^*$,
though they are not marked. Over $K$ the rebuilt edit reads 0.127 in place of "0.135". The highest new edit or random update reads
0.023 in place of "0.020", and the margin is 0.104 in place of "0.115".

Sources, all read with jq:
- `experiments/fig2_standard/results/items/sae/built__rebuilt.json`: `.leftover_share` 0.7814, `.E_b_K` 0.1275.
- `experiments/fig2_standard/results/items/sae/built__random_kl_241002.json`: `.E_b_K` 0.0230, the highest of
  `built__new_few`, `built__new_thin`, `built__random_size_241001` and `built__random_kl_241002`.
- `experiments/sae_eb/results/summary.json`: `.remainder_dd_share` of the rebuilt row, 0.8294.
- The same file: `.E_b_sae` of the rebuilt row, 0.1349.
- The same file: `.P1.max_low` 0.0198 and `.P1.gap` 0.1151.
- The bar of 0.2 is the filed rule in `experiments/sae_eb/results/predictions.json`.

## 2. appI:55, three pairs keep their gap once KL is regressed out

Each pair sets fine-tunes built to reuse against fine-tunes built to need something new. Its gap is the reuse side's
mean reading minus the new side's, so a negative gap means the new side reads higher. The floor is the highest reading
of any random update. Over the necessary set the floor is 0.0110, against 0.0109 over the mean-ablation circuit. The
regression is fitted across 76 fine-tuned states. KL from the base model explains 0.198 of the readings' variance over
the necessary set, against 0.209 before.

The same three pairs sit outside the floor raw and after KL is regressed out, over both sets. A fourth pair, the copy
key against a random target, sits inside the floor raw over the necessary set at -0.0047. After regression it crosses
the floor at -0.0120, and over the mean-ablation circuit at -0.0116. No pair changes sign, and none of the nine has an
empty necessary set.

| pair | raw, necessary set | KL regressed out, necessary set | raw, mean-ablation circuit | KL regressed out, mean-ablation circuit |
|---|---|---|---|---|
| `e187_e54_matched` | -0.0224 | -0.0202 | -0.0214 | -0.0193 |
| `e187_open_pair_c3` | -0.0279 | -0.0193 | -0.0264 | -0.0178 |
| `e187_open_pair_c3_copy` | -0.0279 | -0.0289 | -0.0270 | -0.0279 |
| `e187_e46_copy_random` | -0.0047 | -0.0120 | -0.0043 | -0.0116 |

**For writeup-lead.** The paragraph at appI:55 already says "over $K$ and over $K^*$ alike" and names the fourth pair;
it holds as written over the necessary set.

Sources: `experiments/claim2_needed/results/summary.json`, `.B_pairs.K` and `.B_pairs.Kstar`, keys `floor`,
`n_states`, `fits.kl.r2`, `outside_floor` and `pairs[].raw` and `pairs[].resid_kl`. The counts outside the floor read
3 raw and 4 after regression over both sets.

## 3. main.tex:432, the graded retained clause over the necessary set

Run 1 (E321, 2026-11-27) read the same score over the necessary set $K$ on the same 20 fine-tunes. It still reads 0.999
or more on every one: the smallest is 0.99936 (case_twin_s0), against 0.99938 over the old set. The set barely matters
here, because $K$ holds 29 to 99% of the variables, so this does not show that $K$ picks out anything. Stage 0 passed
(do-nothing reads 1.0; the identity swap flips nothing), and redteam-lead passed the run
(`reports/reviews/2026-11-27_review_e321-cos2-over-k.md`). Source: `experiments/p165_cos2_k/results/summary.json`,
`min_k_value`, and `run_score.log` per item.

The paper's graded wording scores each changed variable by how closely the fine-tune's mechanism points the same way as
the base model's. The score is the mean $\cos^2$ between the two mechanisms' outputs, with one variable per neuron. It
reads 0.999 or more on every trained fine-tune, and the smallest of the 20 is 0.99938.

That value is not read over the mean-ablation circuit, as the paper's mark says. It is read over a KL-closure set, in
which every changed part outside the set runs the base model's mechanism. So the mark on 0.999 is a label slip, and the
set is still not the necessary set.

What fills this section is the neuron swap test on the same 20 pair fine-tunes, with every activation as its neuron's
value. The saved per-part scores are then averaged over the changed parts in the necessary set.

Source of the existing value: `experiments/rho_saved_items/results/summary.json`,
`[.R3|to_entries[]|select(.key|test("_s[01]$"))|.value.rho_main_replace]|min`.

## 4. The rescale-only control at VPD seeds 0-2 and SAE seeds 0-2

Run 2 (E322, 2026-11-27) read the rescale-only control over the necessary set on five tasks and three seeds, at VPD
and at SAE latents. At VPD the control reads close to the full fine-tunes, but not consistently above or below them:
it reads higher on copy, about half on cipher, and level or a little lower on the other three. At SAE latents it reads
above the full fine-tune on every task and seed, two to three times on four tasks and about six times on copy.

The rescale-only control trains only one gain per MLP neuron, on the input of each MLP's output matrix, so it is reuse
by construction. Its 15 states are the ones E290 trained and saved (learning rate 3e-3, the T-Few recipe, 100 times
the full fine-tunes' rate, up to 300 steps); none was retrained or tuned here. Every read setting is E298's filed
value, so the full fine-tunes and their matched random updates were read the same way.

Each cell gives seeds 0, 1 and 2 in order.

| task | rescale-only, VPD | full fine-tune, VPD | rescale-only, SAE | full fine-tune, SAE |
|---|---|---|---|---|
| copy | 0.072, 0.049, 0.063 | 0.034, 0.032, 0.032 | 0.286, 0.262, 0.275 | 0.045, 0.046, 0.042 |
| rank1 | 0.029, 0.028, 0.023 | 0.030, 0.029, 0.029 | 0.101, 0.102, 0.098 | 0.038, 0.035, 0.032 |
| elicit | 0.023, 0.025, 0.024 | 0.027, 0.030, 0.029 | 0.093, 0.090, 0.084 | 0.036, 0.037, 0.031 |
| cipher | 0.023, 0.020, 0.022 | 0.037, 0.039, 0.042 | 0.087, 0.074, 0.083 | 0.036, 0.040, 0.038 |
| prior_cipher | 0.019, 0.023, 0.026 | 0.030, 0.029, 0.029 | 0.072, 0.082, 0.085 | 0.046, 0.047, 0.043 |

Every value is above its task's matched random updates, which read 0.013 to 0.015 at VPD and 0.008 to 0.017 at SAE.

**The SAE error term.** At SAE latents part of each change falls in the SAE's reconstruction error, which the reading
counts. That share is 0.51 to 0.69 for the rescale-only control and higher, 0.68 to 0.76, for the full fine-tunes,
under the same convention. So the error term does not explain why the control reads higher than the full fine-tunes.
At VPD the share is about zero on both sides.

**Against the card's two rivals.** Rival A said the control would read inside the full fine-tunes' range. It fails
at SAE on every row and at VPD on copy. Rival B said it would read 0.2 or more. It holds only at SAE on copy. Neither
fits, and the run was filed as descriptive, so no claim is made at the bar.

**Checks.** Stage 0 passed at both featurizations: do-nothing reads 0, the identity swap flips nothing, and all
10,248 of the SAE's must-flip cases flip. The repeat of VPD seed 0 matches E290's saved values to 3.9e-5 relative,
against a tolerance of 5e-4. redteam-lead passed the data (`reports/reviews/2026-11-27_review_e322-rescale-seeds.md`).

**For writeup-lead.** Wherever the paper says the rescale-only control was read at seed 0 only, it now has three
seeds at VPD and three at SAE latents. At VPD it reads near the full fine-tunes, above them on copy and below on
cipher. At SAE latents it reads two to six times the full fine-tunes on every task.

Sources: `experiments/p165_gain_reads/results/definition_rows.csv` (block `gain`, columns `value` and
`leftover_share`) and `summary.json` (`repeat_max_rel_dev`). The full fine-tunes and random updates come from
`experiments/fig2_standard/results/definition_rows.csv`, block `trained`, sides `trained` and `random`. The figure
script `experiments/p165_gain_reads/plot.py` has not yet been rendered.

## Departures from the literature

None: this page re-reads saved results and changes no definition or procedure.

## Not run

- The figure of section 4 (`experiments/p165_gain_reads/plot.py`). The cluster worker timed out on job steps on
  2026-11-27, so only the table is given.
- The fifteen per-fine-tune random updates over the necessary set at SAE latents, which would complete the random
  group of section 1.
- Matched random updates of the rescale-only states at SAE latents.
- A rescale-only control at the full fine-tunes' step and learning rate, which would need retraining and a new cell.
- No figure for sections 1 to 3: each is one or two values beside its earlier reading, so a table carries it.
