# Can the second wording of reuse serve as the working definition? The gate page

*2026-11-10, definition-lead. Reading and writing only; nothing was run for this page. No wording is adopted: that is a
dated `decisions.md` row only the user writes. Decision card: `reports/decisions/2026-11-10_decision_gate-working-definition.md`.*

We read one wording of reuse on 67M fine-tunes whose labels were fixed when they were built: four hand-made changes,
ordinary fine-tunes copying them, and the seventeen old tasks. Its labels follow construction on the hand-made changes
and the table copies but call nearly every other ordinary fine-tune rewired, while its re-weighting share ranks the
reuse side above its new twin in all five matched pairs. For the goal, it separates strengthening from new capability
within a matched pair, not yet as an absolute label.

## The wording

Take the original model and a fine-tune of it, and a set of inputs fixed in advance: where the new behaviour shows,
where it does not, and far from the training data. The gap is the mean total variation between the two models'
next-token distributions over that set.

A variable is a few directions of the activations at one layer. It *splits* the behaviour when two swaps both work.
Given the fine-tune's value, the original acts like the fine-tune. Given the original's value, the fine-tune reverts.
The label then asks how the fine-tune sets that variable.

- **Re-weighted:** by a fixed affine map of the variable's own value, with at most four directions, on a variable the
  original acts on. Turned up, on by default, shifted: the reuse the safety claims assert.
- **Rewired:** by a fixed affine map of everything the original computed up to that layer. A new connection from old parts.
- **Built:** neither. The fine-tune needs a quantity the original had not computed. "Computed" is stipulated to mean
  affinely readable from the original's activations.

Measured, each label becomes the share of the gap closed by the best map a search finds, 0 for nothing and 1 for all.
The re-weighting share $G$ maps a four-direction variable from its own value. The rewiring share $H$ maps it from
everything before it. The filed rule reads re-weighted when $G$ is at least 0.794 and $H$ at least 0.476. With $H$ above
its cut-off and $G$ below, it reads rewired. With $H$ under 0.476 it reads **no map found**, a search limit never read as
built. The cut-offs are midpoints between the hand-made changes, filed before any ordinary fine-tune was read. On the
user's decision of today, this page reports $G$ within matched pairs, and an absolute cut-off is listed as open.

## Every item, beside the label it was built with

Shares per seed, seed 0 first. A dash means no reading.

| Item | Built as | $G$ | $H$ | Label as read | Matched twin |
|---|---|---|---|---|---|
| **Hand-made changes**, one each | | | | | |
| Always on: a fixed push on a variable | re-weighted | 0.92 | 0.98 | re-weighted | none |
| Cue: the push set by one rare token | rewired | 0.67 | 0.96 | rewired | none |
| Table: the push set by 32 keys | rewired | -0.03 | 0.99 | rewired | none |
| XOR: the push set by an XOR of random bits | built | -0.03 | -0.01 | no map found | none |
| **Ordinary fine-tunes copying them**, 3 seeds | | | | | |
| Always on, 300 steps | re-weighted | 0.64, 0.63, 0.64 | 0.50, 0.45, 0.48 | rewired | none |
| Always on, 3000 steps | re-weighted | 0.77, 0.78, 0.74 | 0.58, 0.58, 0.56 | rewired | none |
| Table | rewired | 0.02, -0.03, -0.02 | 0.72, 0.71, 0.72 | rewired | none |
| Cue | rewired | – | – | no reading: stopped at step 10 | none |
| XOR | built | – | – | no reading: not learned | none |
| **Old 67M tasks**, 2 seeds | | | | | |
| In-context copy | reuse | 0.70, 0.69 | 0.59, 0.58 | rewired | lookup |
| Lookup on the same inputs | new | 0.33, 0.29 | 0.69, 0.67 | rewired | in-context copy |
| Copy, 24 keys | reuse | 0.68, 0.68 | 0.63, 0.66 | rewired | cipher, 24 keys |
| Cipher, 24 keys | new | 0.39, 0.33 | 0.63, 0.62 | rewired | copy, 24 keys |
| Identity, 32 keys | reuse | 0.69, 0.71 | 0.64, 0.61 | rewired | random, 32 keys |
| Random map, 32 keys | new | 0.28, 0.26 | 0.61, 0.63 | rewired | identity, 32 keys |
| Real case label | reuse | 0.83, 0.83 | 0.79, 0.85 | re-weighted | its twin |
| Case twin, labels per word | new | 0.52, 0.55 | 0.70, 0.71 | rewired | real case label |
| Real repeat label | reuse | 0.76, 0.74 | 0.77, 0.78 | rewired | its twin |
| Repeat twin, labels per word | new | 0.54, 0.57 | 0.75, 0.76 | rewired | real repeat label |
| Sharpen the top guess per key | reuse | 0.64, 0.61 | 0.75, 0.73 | rewired | none |
| Lowercase text | reuse | 0.36, 0.35 | 0.45, 0.43 | no map found | none |
| Map to lower guesses per key | open | 0.66, 0.58 | 0.74, 0.75 | rewired | none |
| One-pair map | new | 0.92 | 0.88 | no reading: one seed | none |
| Word cipher, 50 words | new | 0.08, 0.07 | 0.16, 0.14 | no map found | none |
| Word cipher, 500 words | new | 0.19, 0.16 | 0.32, 0.30 | no map found | none |
| Emoticon push | existing behaviour | 0.87, 0.88 | 0.01, 0.06 | no map found | none |

The hand-made rows set the cut-offs, so they agree with construction by design. Of the rest, labels follow construction
for the table copies and for the real case label against its twin. Every other reuse side reads rewired, like its twin.

## The within-pair comparison

![Re-weighting share, reuse side against its new twin](../../experiments/old_task_labels/figures/gate_within_pair.png)

**Reading it.** In all five matched pairs, every reuse seed has a higher $G$ than every seed of its new twin. The
narrowest margin is the repeat label, 0.17 from the lowest reuse seed to the highest twin seed. Only the real case label
clears the dashed cut-off, so as absolute labels the pairs give one contrast in five. Three pairs share their inputs; the
two key-table pairs were read on separate inputs drawn the same way.

## Where it gives no reading, and why

- **Built is never read.** A low $H$ only says this search found no map. The hand-made XOR reads so because it was chosen
  to be affinely unreadable, which checks the construction more than the wording. On a small closed input set, built
  cannot occur at all.
- **Fine-tunes that never showed the behaviour.** The XOR copies did not learn it. The Cue copies stopped at a wrongly
  directed early check, and their runs are spent.
- **One seed.** The one-pair map has a single saved fine-tune; a task label needs both seeds to agree.
- **Long-text tasks on 8-token inputs:** the two word ciphers and lowercase. The emoticon's $H$ sits near zero under a
  high $G$, which the wording rules out, since $H$ can always do what $G$ does.
- **Search limits.** Found $H$ is under found $G$ in 10 of 33 old-task readings; taking the larger changes only the emoticon.
- **"The original acts on it" excludes nothing here.** All 100 random directions at every site moved the output. Only
  the four directions and the affine map separate re-weighted from rewired.
- **$G$ and $H$ read outputs only.** The Always on copies' $G$ rose as they matched the change more closely. The change's
  own variable and map explained 0.632 of a 300-step copy. Only the reversion looks inside: it stayed from 0.24 to 0.26 on
  the 3000-step copies. The hand-made change read 0.938.
- **Absolute labels on ordinary fine-tunes.** The cut-off sits above four of five reuse sides. Open, by today's decision.

## The built control, as designed

Not run. One quantity is read against two originals: the 67M, and the 67M after a short fine-tune that already shows the
XOR at a third of its strength. Each is fine-tuned to copy the full XOR change and read by the same search. If only the
taught arm's $H$ clears 0.476, a low $H$ tracks what the original had. If both read low, low $H$ means nothing here. The
plain arm is the XOR copies above, which did not learn, so that arm has no reading yet. A second design uses two small
models pretrained with and without a parity skill, at about a day of porting. Reading low $H$ as built would still be a
rule only the user adds.

## What would make us drop it

- On a matched pair not used to set anything, such as the Qwen test set, the reuse side reads at or below its twin.
- In the built control, the taught arm also reads low $H$: "no map found" would then carry no information.
- Two fine-tunes with the same outputs but known different insides read alike on every number, the reversion included.
  The wording would then not look inside the model. Nothing yet says how the Always on copies were built, so their low
  reversion cannot be scored.
- The search failing known answers. It reached 0.922, 0.961 and 0.995 against a bar of 0.9.

## What the evidence supports

As a measurement, the wording recovers known answers and orders every matched pair of old tasks by construction. As a
labeller with cut-offs from hand-made changes, it calls ordinary fine-tunes rewired whatever their history. Its "built"
has no reading yet, and its claim to look inside rests on the reversion alone.

**Departures from the literature.** Jain et al. 2024 define possession by a probe for a named function at one layer and
position. Here any value counts, by any affine map from all earlier layers, on a mixed input set. Prakash et al. 2024 patch
named heads one way and score task accuracy; here a searched variable is patched both ways and scored per input. Geiger
et al. 2025 already call adapter-tuning an intervention on parameters. Cut-offs and label rule are this project's own.

**Not run.** The built control. A Cue copy and a learned XOR copy. An Always on copy with a leftover under 0.192 of the
change. Longer inputs for long-text tasks. A stronger rewiring search. Any absolute cut-off. A fresh matched pair or a
Qwen reading. An independent check of any number here.

*Sources: wording `~/commute-data/scratch/definition-session/2026-11-09_reuse-at-a-variable_v2.md`; hand-made E137/E138
`experiments/reuse_variable_test/results/`; copies E139 `experiments/reuse_variable_finetune/results/`; old tasks E142
`experiments/old_task_labels/results/labels.json`; built control `reports/plans/2026-11-10_plan_built-control-design.md`;
premise checks `~/commute-data/premise/P1.3.md`, `P1.4.md`, `P1.7.md`; figure `experiments/old_task_labels/plot_gate.py`.*
