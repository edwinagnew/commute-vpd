# What do random updates of the same gain read on the sparse-autoencoder error share?

E318, cell P2.88, sae-lead, 2026-11-27. Descriptive: values per task, no claim at a bar. Card
`~/commute-data/agents/sae-lead/card_p288.md`. Code `experiments/sae_proxies/random_ref.py`, figure script
`experiments/sae_proxies/plot_random.py`, numbers `experiments/sae_proxies/results/results_random.json`. It adds the
missing random reference to the error share and reconstruction share of E314 and E317,
`reports/analyses/2026-11-26_analysis_e314-sae-proxies.md`.

We read the random updates of the paper's known-answer figure through the same sparse-autoencoder patches as the
fine-tunes, on the four tasks where those random updates reach a gain of about one log-odds on the trained answer. On
every one of the four tasks the random updates read a higher error share than the fine-tunes. Their reconstruction
share is mostly as high or higher. Both shares put the random draws above the fine-tunes, but the draws differ from
them in gain and in size as well as in being untrained, so the gap cannot be read as trained against untrained, and a
high error share does not need training. All four
tasks were built to be new and the reuse side has no random reference at all, so this page cannot speak to reuse
against new, which is the measure `GOALS.md` asks for.

## The answer

A random change that raises the trained answer by about one log-odds reads an error share above one on most draws,
higher than the fine-tunes on the same four tasks. Its reconstruction share is higher than the fine-tunes' on three of the four tasks.
The random updates are matched to the fine-tunes on gain only as the paper's figure matches them: their gain is about a
tenth to a twentieth of training's. So the reference shows that a high error share does not need training, but it is
not a reference at training's size, and no task built to reuse has one.

## What was run

The two shares are E314's and E317's, unchanged:

- **Error share.** At each of the 8 attention and MLP outputs the sparse autoencoder's error is
  $e(o) = o - \mathrm{SAE}(o)$, with the base model's own dictionaries from E234. The base model runs with every site's
  error replaced by the updated model's error on the same rows. The share is $(L - T_0)/(T_1 - T_0)$: $T_0$ and $T_1$
  are the target's log-odds under the base and updated models, $L$ under the patch. It reads 0 if the error carries
  none of the gain and 1 if patching the error alone restores all of it.
- **Reconstruction share.** The same ratio, with each site's reconstruction set to the updated model's and the base
  run's own error kept.
- **Sum of the two.** It reads 1 if the two parts carry the gain additively.

All 8 sites are patched at once, on E136's rows, at the read position and target token of the re-gating share.

**The random updates.** E314's own random update was matched on weight norm and moved the target by at most 0.043
log-odds, too little for a share. Here the reference is instead the random line of the paper's known-answer figure,
E302's draws. For each fine-tune and each of three seeds, a Gaussian direction per changed matrix, at that matrix's norm
in the fine-tune's update, was scaled up until the trained answer rose by 1.00 to 1.10 log-odds. Only 27 of 144 draws
reached that band; the other 117 hit the hundred-fold cap first and have no share. E318 rebuilt each draw from its seed
and saved scale and read it through the same patches as the fine-tune, in fp32. It is matched on the gain the figure
matches, not on weight norm or on the change in the output distribution. The ok draws are 16 to 55 times the
fine-tune's update in weight norm.

**Controls, run first and alone, both held.** The first fine-tune read again gave E314's saved error share with a
difference of exactly 0. The first ok draw's rebuilt size matched E302's saved size with a relative difference of 0.
The fp32 gain read here stays within 0.0079 log-odds of E302's bf16 gain on every ok draw, so no ok draw fell under
one log-odds. An earlier attempt on an H100 stopped at the first check, at a difference of -2.4e-5, a difference
between cards; the scored run is the one on an L40, the card of E314.

## What we observed

The figure is drawn by `experiments/sae_proxies/plot_random.py` into `experiments/sae_proxies/figures/random_ref.png`.
It was not rendered for this page: the CPU worker that runs Python was down when the page was written. Its two
panels plot, per task, the three fine-tune seeds beside the ok random draws, with lines at share 0 and share 1; the
table below holds the same values.

### Where random updates of this gain exist at all

Ok draws exist on four tasks: `c3/cipher`, `e46/rand32`, `e54/cipher` and `e54/cipher_own`. All four were built to be
new. The other four tasks built new and all eight built to reuse have no ok draw: every one of the 72 draws on the
reuse side, and 45 of the 72 on the new side, hit the cap. So the reuse side has no random reference, and the
comparison below is on new tasks only.

On these four tasks the fine-tunes raise the trained answer by 10.9 to 20.8 log-odds and the ok draws by 1.01 to
1.09. Across all 16 tasks the fine-tunes span 1.6 to 20.8 log-odds; three tasks sit under 10 (`e53/lowercase` 1.6 to 1.7, `e53/cipher` 3.8 to 3.9, `e58/rank1` 9.6 to 9.9). The random
reference is therefore one log-odds against a gain of about eleven to twenty-one.

### Error share: random draws read higher than the fine-tunes on all four tasks

**Reading it.** On each of the four tasks, every ok random draw reads a higher error share than every fine-tune seed.
The fine-tunes read 0.69 to 0.79. The random draws read 0.97 to 1.46, and above 1 on three of the four tasks; on
`e54/cipher_own` they sit just under 1. A share above one means patching only the error into the base model raises the
answer by more than the whole random update does. The reading at zero would be no gain carried by the error; neither
set is near it.

### Reconstruction share: random draws as high as the fine-tunes or higher

**Reading it.** On each task the fine-tunes read 0.56 to 0.66, and the random draws 0.51 to 0.99. Every random draw is
above every fine-tune seed except on `e54/cipher_own`, where one draw seed reads 0.51 to 0.53, under the fine-tunes'
0.65 to 0.66. Summed, the fine-tunes' two shares read 1.32 to 1.40 on these tasks, and the random draws' read 1.48 to
2.31. So for a random update each part alone restores close to all of its small gain, and the two parts are further
from adding than they are for training.

### Per task

Ranges over the three fine-tune seeds and over the task's ok draws. Gains are in log-odds of the trained answer.

| task | draws | fine-tune gain | random gain | error share, fine-tune | error share, random | reconstruction share, fine-tune | reconstruction share, random |
|---|---|---|---|---|---|---|---|
| `c3/cipher` | 6 | 12.5 to 12.9 | 1.02 to 1.08 | 0.77 | 1.08 to 1.34 | 0.56 to 0.61 | 0.70 to 0.97 |
| `e46/rand32` | 6 | 13.2 to 13.7 | 1.01 to 1.05 | 0.74 | 1.21 to 1.34 | 0.60 to 0.65 | 0.66 to 0.83 |
| `e54/cipher` | 9 | 10.9 to 11.1 | 1.03 to 1.09 | 0.76 to 0.79 | 1.01 to 1.46 | 0.56 to 0.61 | 0.75 to 0.99 |
| `e54/cipher_own` | 6 | 20.2 to 20.8 | 1.01 to 1.06 | 0.69 to 0.73 | 0.97 to 0.99 | 0.65 to 0.66 | 0.51 to 0.99 |

The draws are not independent within a task. The three draws that share a seed use the same Gaussian seed and close saved
scales (on `c3/cipher`, 37.2 to 40.2 for one seed and 29.5 to 31.0 for the other), on fine-tune updates of similar norms. They read close together: on `e54/cipher_own` the draws of
one seed read reconstruction shares of 0.51 to 0.53 and those of the other 0.94 to 0.99. A task's random spread is
therefore set by its two or three seeds.

## How compelling the evidence is

- **Coverage.** Four tasks, all built new, two or three independent random seeds each. Nothing on the reuse side.
- **Size.** The shares divide by the gain, and the random gain is about a tenth to a twentieth of training's. A
  ratio need not read the same at both sizes; a change this small is the regime where a few dictionary latents
  switching on or off weigh most, as E314's error-energy fraction above one on its weight-matched random updates
  already suggested.
- **Noise floor.** Re-reading a fine-tune gives its saved value exactly, so differences between seeds are not reading
  noise. The fine-tune seeds span up to 0.04 in error share on these tasks.

## Surprises

The random draws read error shares above one, higher than training, on three of the four tasks. Patching in only the
part of the random update's activation change that the base dictionary cannot reconstruct raises the answer more than
the full random update does. The random draws are also far larger than the fine-tunes in weight norm, 16 to 55 times,
to reach a gain of one.

## Departures from the literature

- The error share and reconstruction share are ours, after the error node of Marks et al. 2024, who use the error in
  circuit attribution, not as a share of a fine-tune's gain.
- The random reference is the paper's own known-answer figure convention, matched on gain; E314's earlier random
  reference was matched on weight norm. Neither comes from Marks et al., who have no random reference for the error.

## Not run

- A random reference on the reuse side: every reuse draw hit the hundred-fold cap below one log-odds.
- A random reference at training's gain of about eleven to twenty-one log-odds.
- A random reference matched on the change in output distribution or in activations rather than on gain.
- Per-site shares for the random draws.
- The figure's render, pending the CPU worker; the script is committed.
- The same readings on Qwen3-1.7B.

Provenance: RunPod L40, 5.3 minutes of reading, 144 draws; the stopped H100 attempt's log is
`experiments/sae_proxies/results/run_random_h100_stop.log`. Stage 0 checks are under `stage0` in the results file.
