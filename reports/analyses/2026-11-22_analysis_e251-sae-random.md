# Do random updates read low under wording A at SAE latents?

E251, cell P1.40 run 3 of 3, sae-lead, 2026-11-22. Figure `experiments/sae_r_a/figures/r_a_sae_vs_vpd.png`, the
crosses. Plan `reports/plans/2026-11-21_plan_r-under-a-sae-random.md`; rules
`experiments/sae_r_a_random/results/predictions.json`; code `experiments/sae_r_a_random/run.py`, which calls E245's
reading unchanged; summary `experiments/sae_r_a_random/results/summary.json`. The reading itself is explained in
`reports/analyses/2026-11-21_analysis_e245-sae-r-a.md`.

We read random updates the way E245 read the key-task fine-tunes: how much of the behaviour's circuit, written in
latents of autoencoders trained on the base model, can run on the base model's mechanisms, each times one gain. Random
updates of the fine-tunes' size read about one half, where every fine-tune read near one, so a near-one reading is not
automatic at this size. But the random updates barely change the behaviour, so this reference does not show that the
fine-tunes' near-one readings mean reuse.

## The answer

Yes, at about one half. The fifteen random updates read 0.48 to 0.54, against 0.989 for the lowest key-task
fine-tune, a gap of 0.451 that is four times the noise band. That is the filed lean. Two limits: every random search
hit its evaluation cap, so the random values are lower bounds. And each random update moves the model's output over a
thousand times less than its fine-tune does, so it is read against a far smaller bar.

## What we were trying to test

E245 found that every one of E237's fifteen key-task fine-tunes reads R near one at SAE latents. R under wording A is
the largest share of the circuit's variables that can all take the base model's mechanism at once, each times its own
positive gain. The output must stay within a tenth of the distance from the fine-tune to the base model. One means
every variable can; zero means none can. A reading that puts everything near one says nothing, so this run reads
updates that learned nothing.

**The items.** Each of the fifteen key-task fine-tunes gets one random update. It is a Gaussian direction per weight
matrix, at the norm of that matrix's fine-tune update, scaled to the same common size and read on the fine-tune's own
windows. Beside them sit E242's two random updates, which the parameter decomposition already read under the same
wording. One is at a built new edit's update size on the copy windows; the other is scaled to match a KL on the
gain-of-ten edit's windows. Every reading step, threshold and setting is E245's, called unchanged.

## What we expected, and why

The filed lean was that random updates read well below every key fine-tune, as they do at the decomposition, by more
than E245's split-half band of 0.110. The rival was that the base dictionary's per-site remainders and per-latent gains
absorb any update of this size, so random updates also read near one and E245's key readings carry no information.

## What we observed

### Do random updates read below the key fine-tunes?

**Reading it.** Yes, every one, by about one half. In the figure the crosses on each key-task row are its three random
updates, all between 0.48 and 0.54, and the circles are the fine-tunes, all at 0.989 or above. Pair by pair, each
fine-tune reads above its own random update by 0.461 to 0.508. The cipher, the lowest key task, also has the lowest
random update, at 0.48. On E242's two random updates the latent reading sits above the decomposition's: 0.515 against
0.291 for the size-matched one, and 0.670 against 0.301 for the KL-matched one.

| key task | seed | fine-tune R | random update R | difference | random remainder share |
|---|---|---|---|---|---|
| copy | 0 | 0.9990 | 0.534 | 0.465 | -3.00 |
| copy | 1 | 0.9991 | 0.532 | 0.467 | -3.19 |
| copy | 2 | 0.9987 | 0.517 | 0.482 | -3.33 |
| top | 0 | 0.9990 | 0.533 | 0.466 | -3.06 |
| top | 1 | 0.9990 | 0.531 | 0.468 | -3.20 |
| top | 2 | 0.9987 | 0.538 | 0.461 | -2.78 |
| mid | 0 | 0.9987 | 0.519 | 0.480 | -2.93 |
| mid | 1 | 0.9990 | 0.507 | 0.492 | -2.86 |
| mid | 2 | 0.9987 | 0.492 | 0.507 | -3.03 |
| prior-matched cipher | 0 | 0.9982 | 0.511 | 0.487 | -3.36 |
| prior-matched cipher | 1 | 0.9978 | 0.523 | 0.475 | -3.14 |
| prior-matched cipher | 2 | 0.9984 | 0.520 | 0.478 | -3.64 |
| cipher | 0 | 0.9890 | 0.523 | 0.466 | -3.56 |
| cipher | 1 | 0.9897 | 0.481 | 0.508 | -3.63 |
| cipher | 2 | 0.9908 | 0.529 | 0.462 | -3.40 |

R is at the filed bar of a tenth. The remainder share is one minus the KL left when only the eight site remainders
take the base model's mechanism, over KL from the fine-tune to the base model; the fine-tunes read 0.378 to 0.837.

### What does the low reading rest on?

**Reading it.** On a behaviour change too small for the stand-in to match. A random update of this size moves the
output 0.0016 to 0.0019 nats from the base model, against 1.90 to 4.40 for the key fine-tunes. Since the bar is a
tenth of that distance, the random updates are read against a bar over a thousand times smaller. At that scale the
fitted gains themselves cost more than the update. Standing in every changed variable of the circuit at once lands two
to two and a half times further from the random update's output than the base model is. For the fine-tunes it lands
closer than the base model. The negative remainder shares say the same, since standing in only the remainders already
overshoots. The one random update that moves the behaviour more, 0.172 nats, reads higher, at 0.670. So a low random
reading here may say more about the stand-in's precision on tiny changes than about structure in the update.

## What we learned

R under wording A at SAE latents does not read every update of key-task size near one: random directions of that size
read about one half. That rules out the plainest failure, a reading pinned at one by the remainders whatever the
update. It does not yet show that the fine-tunes' near-one values mean reuse, because the random updates were read
against a bar over a thousand times smaller. Together with E245's pairs, where the side built to reuse beat the side
built to be new by less than the noise, the latent reading stays unable to sort trained fine-tunes.

## How compelling is the evidence

- **Lower bounds.** Every K\* search and every stand-in search on the 17 random items hit its cap of 2,000
  evaluations. A random update's true R could be higher; the fine-tunes' stand-in searches were not capped.
- **Matched size, not matched behaviour.** The random updates match the fine-tunes' weight change, not their output
  change, which is over a thousand times larger.
- **No halves.** The random items get no split halves; the band is E245's, from the built edits.
- **One dictionary seed, one random draw per state.**

## Surprises

- The random updates' remainder shares are all strongly negative, -3.64 to -2.78: standing in only the eight
  remainders leaves the output about four to five times as far from the random update's as the base model is.
- At SAE latents the decomposition's two random updates read higher than at the decomposition, by about 0.2 and 0.4.

## What next

1. A random update scaled until it moves the output as far as a key fine-tune, read in latents. If it still reads
   about one half, the key readings carry information; if it reads near one, the latent reading has no floor on
   trained items.
2. One random item with an uncapped search, to see how far the lower bound moves.

## Scorecard

| id | claim in words | rule | read | verdict |
|---|---|---|---|---|
| R0 | do-nothing exact, per-item controls exact | E245's, unchanged | four jobs' do-nothing read one, floor about 8e-12 | held |
| P1 | lowest key fine-tune above highest random update by more than the band | gap against 0.110 | 0.989 against 0.538, gap 0.451 | rival a, as filed; random side a lower bound |
| D1 | E242's random updates, SAE beside VPD | descriptive | 0.515 against 0.291; 0.670 against 0.301 | read |

## Departures from the literature

- As E245's: base-only BatchTopK dictionaries trained here, each site's remainder as one variable, one positive gain
  per variable, and the circuit by E226's mean-ablation search.
- The random updates are E237's construction, a Gaussian per matrix at the fine-tune's per-matrix norm; no published
  null is followed.

## Not run

- A random update matched on output change rather than weight size.
- Uncapped searches on any random item.
- Split halves on the random items.
- A second dictionary seed or a second random draw per state.
- A neuron reading of the same random updates.

## Provenance

Items `experiments/sae_r_a_random/results/items/random_*.json`, stage 0 `items/stage0_random0-3.json`, logs
`results/run_random0-3.log`; three logs end with exit 143, the pod stopping after its last item was written, the
fourth with exit 0; all 17 items read. Four H100 pods. Keep masks and gains in `results/large/`. Key-task readings from
`experiments/sae_r_a/results/items/e58-*_scaled.json`. Numbers ledger: this file's `.numbers.tsv`.
