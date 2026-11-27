# Do two cheap readings at base-model sparse autoencoders tell reused fine-tunes from new ones?

E314, cell P2.85, sae-lead, 2026-11-26. Descriptive: no verdict on reuse and no bar (user, 2026-11-26: "put in
appendix G unless it works well"). Plan `reports/plans/2026-11-26_plan_e314-sae-proxies.md`; method spec and
departures for the second reading `reports/plans/2026-11-26_spec_kissane-sae-transfer.md`; code
`experiments/sae_proxies/run.py`; numbers `experiments/sae_proxies/results/results.json`.

We read the 48 fine-tunes behind the paper's known-answer figure through sparse autoencoders trained on the base
model alone, each beside a random update of the same weight size. On every task most of the trained answer's gain
passes through what the base dictionary leaves out, and training worsens reconstruction far more than a random update
does. Tasks built to reuse and tasks built to be new overlap on both readings, so neither can serve as the measure of
reuse that `GOALS.md` asks for.

## The answer

Neither reading separates the tasks built to reuse from the tasks built to be new. The error share is above one half
on all 16 tasks, reuse tasks included, and the two sides overlap. The reconstruction gap is small on every task but
always well above a random update of the same weight size, and again the two sides overlap. Both readings see that
training changed something the base dictionary does not describe; neither says whether it is reuse. Placement: appendix
G.

## What was run

Each fine-tune's update is read on its own task rows, at the read position and target token of the re-gating share,
the share of the gain that appears when VPD's original subcomponents are kept and only Γ switches them differently.
The dictionaries are E234's sparse autoencoders, one per site, trained on Pile activations of the base model only. Each
fine-tune also gets one random update: Gaussian noise in every matrix it changed, scaled to that matrix's norm in the
fine-tune's update.

- **Error share, ours, after the error node of Marks et al. 2024.** At each of the 8 attention and MLP outputs the
  error is what the base dictionary leaves out, $e = o - \mathrm{SAE}(o)$. The base model runs with every site's error
  replaced by the fine-tuned model's error on the same rows. The share is $(L - T_0)/(T_1 - T_0)$, with $T_0$ and $T_1$
  the target's log-odds under the base and fine-tuned models and $L$ under the patch. Zero means the gain is all in
  what the dictionary reconstructs; one means it is all in the error.
- **Error-energy fraction.** The change in the error over the change in the output, $\sum|e_{FT} - e_0|^2 / \sum|o_{FT} -
  o_0|^2$ over sites and positions. It needs no gain, so it also reads on the random updates.
- **Reconstruction gap, after Kissane et al. 2024.** Their explained variance per site, base model minus fine-tuned
  model, averaged over E234's 13 sites. Zero means the fine-tune is reconstructed as well as the base.

**Controls, run first and alone, all held.** The zero update read exactly 0 on every reading. The fine-tuned model run
twice gave identical logits, and patching its full outputs into the base model reproduced them exactly, a share of 1.
E234's base explained variance was reproduced within 4.4e-6 at all 13 sites. All 48 gains matched E136's saved gains
exactly. No token crossed Kissane's outlier threshold. Random updates move the target by at most 0.043 log-odds, under
the 1 log-odds floor, so they have no error share.

## What we observed

![Error share, reconstruction gap and error-energy fraction against the re-gating share](../../experiments/sae_proxies/figures/proxies.png)

**Reading it, error share.** Most of the gain sits in the error on every task. Tasks built to reuse read 0.52 to 0.74
and tasks built to be new 0.62 to 0.86. Among the new tasks the error share rises as the re-gating share falls, with
`e53/cipher` highest. But `e54/elicit`, a reuse task at 0.72 to 0.74, sits above `e62/lookup` and `e58/cipher`, two
new tasks at 0.62 to 0.64. So the base dictionaries miss most of what any of these fine-tunes changed, as the earlier
latent readings at the same autoencoders found.

**Reading it, error-energy fraction.** Trained updates keep most of their output change inside the base dictionary:
the fraction is 0.21 to 0.44 for trained states. Their random updates read 1.12 to 3.01. A fraction above one
means the reconstruction moves against or across the output change, as latents switch on and off. Reuse and new tasks occupy the
same range in panel c.

**Reading it, reconstruction gap.** Every trained state is reconstructed a little worse than the base model: reuse
tasks lose 0.0041 to 0.0212 explained variance, new tasks 0.0098 to 0.0705. The random updates lose 0.00009 to 0.00236,
between 7 and 725 times less than their own trained state. New tasks mostly read higher, but `e53/cipher` and
`e62/lookup` sit below `e54/elicit` and `e53/lowercase`. For scale, Kissane's base-to-chat transfer lost about 0.07
explained variance on Mistral-7B and 0.15 on Qwen1.5-0.5B; our largest task gap is about Mistral's. The random reference
is matched in weight norm, not in how far it moves activations, so "far below training" partly says that random weight
noise barely moves this model's activations.

## Per task

Sorted by re-gating share, task mean. Error share and gap are seed ranges over three seeds; the random gap is over the
three seeds' random updates. The embedding site reads a gap of exactly 0 in every state and is inside the 13-site mean.

| task | built | re-gating share | error share | explained-variance gap | random update's gap |
|---|---|---|---|---|---|
| `e62/copy` | reuse | 1.09 | 0.65 to 0.66 | 0.0053 to 0.0072 | 0.00019 to 0.00041 |
| `e58/rank1` | reuse | 0.86 | 0.65 to 0.66 | 0.0128 to 0.0142 | 0.00035 to 0.00054 |
| `c3/rank1` | reuse | 0.84 | 0.68 to 0.69 | 0.0149 to 0.0163 | 0.00010 to 0.00020 |
| `c3/copy` | reuse | 0.84 | 0.68 to 0.73 | 0.0120 to 0.0127 | 0.00014 to 0.00023 |
| `e46/ident32` | reuse | 0.78 | 0.52 to 0.58 | 0.0155 to 0.0179 | 0.00010 to 0.00022 |
| `e58/copy` | reuse | 0.77 | 0.55 to 0.59 | 0.0041 to 0.0044 | 0.00027 to 0.00042 |
| `e54/elicit` | reuse | 0.77 | 0.72 to 0.74 | 0.0194 to 0.0212 | 0.00014 to 0.00018 |
| `e53/lowercase` | reuse | 0.76 | 0.65 to 0.67 | 0.0186 to 0.0194 | 0.00182 to 0.00203 |
| `e62/lookup` | new | 0.40 | 0.63 to 0.64 | 0.0098 to 0.0132 | 0.00058 to 0.00073 |
| `e58/prior_cipher` | new | 0.38 | 0.67 to 0.68 | 0.0205 to 0.0244 | 0.00051 to 0.00075 |
| `e58/cipher` | new | 0.30 | 0.62 to 0.64 | 0.0382 to 0.0423 | 0.00062 to 0.00072 |
| `e46/rand32` | new | 0.26 | 0.74 | 0.0577 to 0.0705 | 0.00012 to 0.00025 |
| `e54/cipher_own` | new | 0.26 | 0.69 to 0.73 | 0.0189 to 0.0230 | 0.00048 to 0.00055 |
| `e54/cipher` | new | 0.24 | 0.76 to 0.79 | 0.0541 to 0.0629 | 0.00010 to 0.00025 |
| `c3/cipher` | new | 0.23 | 0.77 | 0.0536 to 0.0670 | 0.00009 to 0.00017 |
| `e53/cipher` | new | 0.12 | 0.85 to 0.86 | 0.0162 to 0.0176 | 0.00142 to 0.00236 |

## Departures from the literature

The full table is in the spec, `reports/plans/2026-11-26_spec_kissane-sae-transfer.md`. The ones that matter for reading this page:

- Kissane et al. compare a base model with its chat model; we compare it with small task fine-tunes, so their gaps set
  the scale only loosely.
- They read one middle-layer residual site; we average 13 sites, which dilutes a change confined to one block. Per-site
  gaps are saved.
- They train dictionaries on both models; we have base-model dictionaries only, so there is no own-dictionary
  reference to split a gap into drift and a harder distribution.
- The matched-size random update is ours; they have no size reference.
- The error share is ours, after the error node of Marks et al. 2024, who use the error in circuit attribution, not as
  a share of a fine-tune's gain.

## Not run

- An own-dictionary reference: autoencoders trained on each fine-tuned model.
- Kissane's CE loss recovered; we read the target's log-odds instead.
- A random reference matched in activation change rather than weight norm.
- The same readings on Qwen3-1.7B.
- A per-site breakdown in this page; it is saved in `results.json`.

Provenance: cluster job 69300, one L40, 7.3 minutes, $0; controls in `results.json` under `stage0`.
