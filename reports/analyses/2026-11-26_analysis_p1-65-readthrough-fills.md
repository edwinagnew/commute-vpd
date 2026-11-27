# Readthrough fills: four paper numbers re-read over the necessary set

Two of the paper's numbers were read over the mean-ablation circuit and are re-read here, from saved files, over the
necessary set the definition uses. Both readings keep their sense: the SAE's reconstruction error still carries most of
the rebuilt edit's change, and the same three pairs keep their gap once output distance is regressed out. The other two
numbers wait on runs that have not been made, so their sections are placeholders, and nothing here changes a claim of
`GOALS.md`.

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

> **PENDING: run 1 of this cell, not yet made.** The value over $K$ is not on disk. No number is given here until the
> run's summary is written and checked.

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

> **PENDING: run 2 of this cell, not yet made.** The rescale-only states have no reads yet at VPD seeds 1 and 2, at the
> repeat of VPD seed 0, or at SAE seeds 0 to 2. No number is given here until they do.

The rescale-only control trains only one gain per MLP neuron, so it is reuse by construction. Its states for five tasks
and three seeds are already trained and saved; only the reads are missing. The table this section will hold puts each
state beside the full fine-tune of the same task and seed, at VPD and at SAE latents. The full fine-tunes' values
are in `experiments/fig2_standard/results/definition_rows.csv`, rows with block `trained` and side `trained`.

| task | seed | rescale-only, VPD | full fine-tune, VPD | rescale-only, SAE | full fine-tune, SAE |
|---|---|---|---|---|---|
| PENDING | | | | | |

## Departures from the literature

None: this page re-reads saved results and changes no definition or procedure.

## Not run

- Run 1, the graded retained clause over the necessary set, which section 3 waits on.
- Run 2, the rescale-only control reads at VPD seeds 1 and 2, the repeat of seed 0, and SAE seeds 0 to 2.
- The fifteen per-fine-tune random updates over the necessary set at SAE latents, which would complete the random
  group of section 1.
- Matched random updates of the rescale-only states at SAE latents.
- A rescale-only control at the full fine-tunes' step and learning rate, which would need retraining and a new cell.
- No figure: each section is one or two values beside their earlier reading, so a table carries it.
