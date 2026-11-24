# Does the crosscoder reading change when latents with a negative base coefficient are excluded, as Minder et al. do?

*2026-11-23, compare-lead. Cell P2.70: a $0 rescore on CPU, then E289, a refit of the coefficients at the paper's token
count on one H100 pod. Prompted by redteam-lead's fidelity audit of the crosscoder baseline. Rescore: script
`experiments/crosscoder/rescore_negbeta.py`, results `experiments/crosscoder/results/negbeta_rescore.json`, figure
`experiments/crosscoder/figures/negbeta_ladder.png`. Refit: plan `reports/plans/2026-11-23_plan_p2-70-betas-refit.md`,
script `experiments/crosscoder_proper/refit_betas.py`, filed rules and results in `experiments/crosscoder_proper/results/refit/`.
The classifier fix is in `src/commute/crosscoder.py`, function `specific`.*

We rescored every saved crosscoder reading with the authors' exclusion of latents whose base-model coefficient is
negative, then refitted that coefficient for the ten planted controls on the paper's token count. The refit shows the
negative coefficients are real, so the exclusion stands, and no planted control passes anywhere, the real-text plant
included. For `GOALS.md` the crosscoder stays unusable at fine-tune size, and the reason is now settled: the authors'
rule, as published, removes the planted change's own latents.

## The answer

No planted control passes under Minder et al.'s rule as published. The block-3 head reads 0.42 to 0.44 against the bar
of 0.5, the ladder's top sizes read at most 0.010, and the real-text plant reads 0. Refitting the coefficients on 50M
tokens, as the paper does, changes none of this. The plant latents' base coefficients are small, a few thousandths of
the fine-tune's, but negative in both halves on all ten. So the exclusion is not noise from our small fit; it is how the
authors' rule treats these latents. Readings of fine-tunes and random updates move by at most 0.022.

## What we were trying to test

**The crosscoder.** A crosscoder is one dictionary of sparse latents that rebuilds two models' activations at once, here
the original 67M model and a changed one at one block's input. Minder et al. 2025, arXiv 2504.02922 v4, call a latent
fine-tune-specific when it is among the top latents by relative decoder norm and passes *latent scaling*. Latent scaling
fits, for each latent $j$, a coefficient $\beta_j$ that best rescales its fine-tune decoder direction to explain a
target. The targets are the base model's and the fine-tune's reconstruction error, with the latent's own part added
back, and their reconstructions. The ratios are $\nu^\varepsilon = \beta^{\varepsilon}_{base} / \beta^{\varepsilon}_{ft}$
and $\nu^r = \beta^{r}_{base} / \beta^{r}_{ft}$. A latent passes when $\nu^r < 0.5$ and $\nu^\varepsilon < 0.2$: the
base model has little use for it.

**The missing step.** Their appendix E.2, page 26, adds: "For all of our analyses, we filter out latents with negative
β base values", and counts none in BatchTopK crosscoders. Our port applied the two thresholds only, so a negative ratio
passed. Our check against the authors' released Gemma crosscoder applied the filter to both sides, in
`experiments/crosscoder_replicate/run.py`. So that check never tested this step of our 67M code. The fix puts the same
rule, both base coefficients at least 0, inside `specific`. A latent that never fired has no coefficient and fails as
before.

**What was rescored.** All 96 saved crosscoder readings of the five experiments that trained one. They cover the nine
built pairs, the real-text controls, the planted heads at blocks 2 and 3, the block-2 size ladder, and the five key
tasks, each with its controls and random updates. The groups were rebuilt from the saved groups, so the old reading
reproduces each run's reported parts to within 3e-8 and its counts exactly.

**What was refitted, E289.** The runs fitted the coefficients on 100,000 tokens; the paper fits them on 50M. For the ten
planted controls, we rebuilt each run's own activation stream and refitted every coefficient. The controls are the
block-2 heads on three seeds, the block-3 heads on three seeds, rung 4 on two seeds, rung 5, and the real-text plant.
Each fit uses two disjoint halves of 25.1M tokens, A and B, whose sum is the 50M fit. A first stage rebuilt the 100,000
tokens the runs used and bootstrapped each dropped latent's coefficient over 512-token blocks.

**The reading.** The specific part is the share of the activation change carried by fine-tune-specific latents. Zero
means none of the change is in such latents. A planted change the method should see is meant to reach the bar of 0.5.
The pair reading $d$ is the shared part minus the specific part.

## What we expected

The filter can only shrink the specific set, so every specific part can only fall. For the refit, two explanations were
filed before it ran, in `results/refit/predictions.json`, for the ten *plant latents*: dropped latents with both ratios
above minus 0.02 that carry at least 0.01 of the change.

- **Noise in our fit.** The coefficients are truly zero and came out negative by chance at 100,000 tokens. Then at 50M
  they shrink about twenty-fold, the halves disagree in sign, and the exclusion is a coin flip.
- **Real.** The coefficients are truly negative. Then they stay put at 50M and both halves are negative.

Redteam-lead's rule for each latent was filed with them. An exclusion is real when both halves are negative and the
50M fit is more than 2 standard errors below zero. Otherwise it is a coin flip, recorded as a limitation of the authors'
rule. The literal reading is the scored number either way. A second figure, "near-zero kept", keeps only the coin-flip
latents. It is ours, not the authors' method.

## What we observed

### Does any planted control still pass?

![planted heads, old and new](../../experiments/crosscoder/figures/negbeta_ladder.png)

**Reading it.** No. Open marks are the readings as run, filled marks the same crosscoders with the exclusion at 100,000
tokens. At block 3 all three seeds fall under the bar, to 0.42 to 0.44. At block 2 the ladder's top two sizes fall to
near zero, and the middle size's third seed falls from 0.37 to 0.004. The 50M refit moves none of these by more than
0.003, as the table below shows.

On real text the planted direction passed as run, with one latent carrying 0.77 of the change. At 50M that latent's
base reconstruction coefficient is minus 0.0059, with a standard error of 0.00006, and negative in both halves. The
exclusion drops it, and the plant reads 0. Its control fails.

### Are the plant latents' negative coefficients noise?

No, on every filed reading. Each matched the "real" outcome:

- Bootstrap over the 100,000 tokens: every resample stayed below zero for 8 of the 10 plant latents, and at least 0.98
  of resamples for the other two.
- Shrink at 50M: the median is 0.95 of the 100,000-token value, against under 0.2 for noise.
- Halves: all 10 are negative in both, and all 10 are real by the filed rule.
- Clearly negative latents: the 6 that were already far below zero stay real too.

The coefficients are small but sharp. At 50M they run from minus 0.0016 to minus 0.018, with standard errors of 0.00004
to 0.00009. The real-text latent moved further below zero with more tokens, not toward it.

So "near-zero kept, ours" equals the literal reading up to a few coin-flip latents elsewhere. It is at most 0.45 at
block 3 and 0 on real text.

### Do the readings of fine-tunes and random updates move?

Barely. On the nine built pairs only one side has a dropped latent that carries over a ten-thousandth of the change:
copy at 96 keys, whose $d$ moves from 0.056 to 0.078. All four tallies of pairs whose side built to reuse reads higher
stay as they were, five, four, six and six of nine. On the five key tasks the copy spread falls from 0.43 to 0.41, and
13 of the 15 fine-tunes still read below their task's random updates. At block 3 the Gaussian and both pair sides stay
under one hundredth of the change. None of these was refitted.

### Which latents does the exclusion drop?

Mostly the planted change itself. At the ladder's top size the dropped latents carried 0.462 of the change. Of that,
0.456 sits in latents whose two base-to-fine-tune ratios are both above minus 0.02, with fine-tune coefficients near 1.
Latent scaling would call them as fine-tune-only as a latent gets, but their base coefficient is a hair below zero, and
the refit shows that hair is real. At block 3 the dropped share is 0.10 to 0.20, mostly in latents with clearly negative
base coefficients.

The appendix describes the latents it dropped as having "low maximum activations" and a "small improvement in MSE", and
reports none in BatchTopK crosscoders. Ours carry up to 0.77 of the change. The authors' own released Gemma table also
holds negative base coefficients in the top 3,176 latents: our replication found 2,832 kept of 3,176. On that table,
fitted by the authors on 50M tokens, the filter takes the count of latents passing both thresholds from 931 to 599,
matching their filtered count. It drops 332 latents, and 69 of those have both ratios above minus 0.02. So on the
authors' own data their rule also drops latents whose base coefficient is within a few hundredths of zero. The counts
are read from `experiments/crosscoder_replicate/results/stage_b_latents.csv`, columns `*_theirs`.

### Every crosscoder number the paper and log quote, old and new

Paper lines are in writeup-lead's clone, `paper/paper/`, at commit c347a94. Not edited. "Refit" is E289: the literal
rule on 50M coefficients, with "near-zero kept, ours" after the semicolon. A dash means the item was not refitted.

| where | what it says | old | rescore at 100,000 tokens | refit at 50M; near-zero kept, ours | still true? |
|---|---|---|---|---|---|
| main.tex 963; appD 88 | block-2 head, three seeds | 0.21 to 0.37; at most 0.37 | 0.004 to 0.30 | 0.004 to 0.30; same | under the bar |
| main.tex 966; appD 89 | ladder near fine-tune size | 0.0030 | 0.0030 | - | yes |
| main.tex 966-967; appD 89-90 | ladder's top two sizes "levelled off" | 0.45 and 0.46 | 0.010 and 5e-7; 3e-5 | 0.0097 and 6e-6; 1e-5. Ours: 0.0100 and 8e-6; 2e-5 | no: they fall to near zero; no size reaches the bar |
| main.tex 968; appD 91 | block-3 control "passed", head at 11.30 nats | 0.56, 0.63, 0.52 | 0.44, 0.43, 0.42 | 0.44, 0.43, 0.42. Ours: 0.45, 0.43, 0.42 | no: it fails |
| appD 92-93 | block 3: random change reads like both sides, all under one hundredth | random 0.0035 to 0.0070 | random 7e-6 to 0.0008 | - | under one hundredth yes; no passing control |
| sec5_results 163 | real text: "where its planted control passed" | 0.77 | 0 | 0; ours 0 | no: it fails |
| sec5_results 163-165 | real-text Gaussian, stop bar, trained built-pair sides | 0.73; 0.45; 0.39 to 0.46 | unchanged | - | yes |
| sec5_results 162; sec5_3 85; appF 130 | key tasks below random | 13 of 15 | 13 of 15 | - | yes |
| appF 128-129 | key-task seed spread; copy seed 0 | 0.43; 0.06 | 0.41; 0.08 | - | number changes |
| appF 162 | crosscoder signs on the nine pairs | four | four | - | yes |
| main.tex 958 | earlier planted direction: its latent's error ratio against 0.2 | 0.244 | 0.244, still fails | - | yes |
| sec5_results 161; sec5_3 84; sec7 34; main.tex 681 | misses its control up to about a hundred times a fine-tune | | | | yes, now at every site and on real text |
| main.tex 975; sec5_results 182 | figure `crosscoder_size_ladder` | as run | redraw needed | | no |
| main.tex 947-948; appG 52-55 | the latent test named as the paper's two thresholds | | | | add the exclusion; the 100,000-token fit stays a departure that the refit shows does not matter |
| log card E235 | block-2 head; block-3 head | 0.30, 0.21, 0.37; 0.56, 0.63, 0.52 passed | 0.30, 0.20, 0.004; 0.44, 0.43, 0.42 failed | 0.30, 0.20, 0.004; 0.44, 0.43, 0.42 failed | no |
| log card E235 | block 3: random, elicit, cipher | 0.0035 to 0.0070; 0.0056 to 0.0092; 0.0014 to 0.0106 | 7e-6 to 0.0008; 0.0032 to 0.0059; 0.0004 to 0.0089 | - | ranges change |
| log card E235 | k 32 and k 200 at block 2 | 0.39; 0.19 | -0.0003; 0.003 | - | number changes |
| log card E235 | specific latent counts at block 3 | 7 to 50 | 1 to 37 | - | number changes |
| log card E244 | spread, copy seeds | 0.43; 0.056, 0.488, 0.489 | 0.41; 0.078, 0.488, 0.489 | - | number changes |
| report E203 | sides built to reuse higher: as built, size-matched, both counting artefacts | 5, 4, 6, 6 of 9 | 5, 4, 6, 6 of 9 | - | yes |

## What we learned

The crosscoder's row in the paper does not change in kind: at fine-tune size it reads nothing. What changes is that it
now passes no planted control anywhere. The block-3 head and the real-text plant both fail under the published rule,
and the refit shows that failure is not ours. Our small coefficient fit is ruled out as the cause: on 50M tokens the
plant latents' base coefficients are negative beyond doubt. What remains is a property of the authors' rule. A latent
the planted change creates gets a base coefficient a few thousandths below zero, and a strict sign test drops it.

## How compelling is the evidence

The rescore arithmetic is exact: the old readings reproduce every reported part to within 3e-8 and every count exactly,
and redteam-lead's nine planted-head numbers reproduce to all printed digits. The refit rebuilt each side and stream.
It reproduced every saved coefficient within 1.04 floors, set by moving each weight by half a bfloat16 step, against
the bar of 3. Each half fitted 25.1M tokens. The refit's standard errors come from a bootstrap over the 48 chunks, about
1M tokens each, so they assume chunks are exchangeable. The task half of each chunk draws from a finite window pool, so A
and B share some task windows; their Pile halves are disjoint. One crosscoder per side on the built pairs and key
tasks, one to three per planted size. The pod cost $5.7.

## Surprises

The refit went against our expectation. The rescore read the plant latents' small negative coefficients as a sign
decided by noise, and filed that as the first explanation. Every filed reading came out the other way. The coefficients
are tiny relative to the fine-tune's, but the two halves of 25M tokens agree on each to within 0.075 of its value.

## What next

- Redraw `crosscoder_size_ladder`, and correct the block-3 and real-text sentences listed above. Both controls now fail,
  under the published rule, with the paper's token count.
- Whether a base coefficient a few thousandths below zero should exclude a latent is a question about the authors' rule.
  Our runs can only describe it. It is a limitation of theirs: the paper reports no such latents in BatchTopK
  crosscoders, yet their own Gemma table holds 69 of them.
- The other 22 saved crosscoders were not refitted. Their readings move by at most 0.022 under the exclusion, so no
  quoted number depends on them.

## Departures from the literature

- The exclusion follows App. E.2 as written, base coefficients at least 0 on both targets. The paper states no
  tolerance; none is used.
- The runs fitted latent scaling on 100,000 tokens rather than 50M. For the ten planted controls the refit used 50M and
  changed no reading by more than 0.003; the other crosscoders keep the 100,000-token fit.

## Not run

- No crosscoder was retrained.
- The other 22 saved crosscoders were not refitted, nor E203's and E244's, which kept no encoder.
- The wide candidate count's groups are rebuilt from float16 norms, up to 0.008 off the reported parts; nothing quotes it.
- The paper, its figure and the research log are not edited; the log line is the lead's.
- E203's rehearsal and the E212 rehearsal crosscoders were not rescored.
