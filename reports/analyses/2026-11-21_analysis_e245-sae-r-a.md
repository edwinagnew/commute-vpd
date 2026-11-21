# Does R under wording A, read in base-model SAE latents, put the built reuse edits above the new ones?

E245, cell P1.40 run 2 of 2, sae-lead, 2026-11-21. Figure `experiments/sae_r_a/figures/r_a_sae_vs_vpd.png`. Plan
`reports/plans/2026-11-21_plan_r-under-a-sae.md`; rules `experiments/sae_r_a/results/predictions.json`; code
`experiments/sae_r_a/run.py`, `commute.sae_feat`, `commute.stand_in`; summaries
`experiments/sae_r_a/results/summary_built.json` and `summary_keys.json`, the second written by
`experiments/sae_r_a/plot.py`.

We read how much of each fine-tune's behaviour circuit can run on the base model's own mechanisms, each times one gain,
with the circuit written in latents of autoencoders trained on the base model alone. On six hand-built edits the reuse
edits read above the new ones by just over the noise band, the reverse of the parameter decomposition's reading. Every
key-task fine-tune reads near one, so this reading orders the built labels only narrowly and cannot yet sort trained
fine-tunes into reused and new.

## The answer

Yes, narrowly, on the built edits. In latents, the lowest reuse edit reads 0.694 and the highest new edit 0.571. That
gap is just above the noise between two halves of the data, so the filed rule backs the reading that new edits fail to
stand in. The order rests on latents at sites the edit never touched, which stand in for free; counting only the changed
latents, the rebuilt circuit reads below both new edits. All fifteen key-task fine-tunes read between 0.989 and 0.999,
the cipher lowest, so this reading sits at its ceiling there.

## What we were trying to test

**The definition.** Wording A, adopted by the user on 2026-11-20 (`decisions.md`, "Definition wording A adopted"): "A
fine-tune F reuses the base model M, for a behavior, to the extent that M's mechanisms, each up to an output gain, can
stand in for those of the variables the behavior runs through and still preserve the behavior." R is the largest share
of the circuit's variables that can all take M's mechanism at once, each times its own positive gain, inside F's full
forward pass, while the KL divergence from F's output stays within a tenth of KL from F to M. One means every variable
of the circuit can run on M's mechanism; zero means none can. The tenth is the filed bar; a twentieth and a fifth are
reported beside it.

**The featurization.** Sparse autoencoders trained on the base model's activations only, one per block output: the four
attention outputs and the four MLP outputs of the 67M model, 16,384 latents each, from E234. At each site a variable is
one latent, plus one remainder: whatever the site writes that the dictionary's latents do not, bias included. A latent's
mechanism in F is the encoder applied to F's module output; in M it is the encoder applied to M's module on the same
input. The residual-stream dictionaries are left out because they break on the gain-of-ten edit. Embeddings, layer
norms and the unembedding belong to no variable and always keep F's mechanism.

**The circuit.** K\* is the smallest set of variables found by E226's search such that mean-ablating everything outside
it keeps F's behaviour within a tenth of the full ablation's KL. Variables at a site whose weights did not change stand
in exactly, with gain one, so the share of K\* at untouched sites is the lowest R an edit can read.

**The items.** Six hand-built edits on E214's copy and cue windows, whose label is known by construction:

- reuse: the copy neurons' outputs doubled, the gain-of-two edit; times ten, the gain-of-ten edit; the copy circuit
  moved onto grafted neurons, the moved edit; the copy neurons zeroed and later MLP layers retrained to rebuild copying,
  the rebuilt edit;
- new: a key-to-answer lookup written into eight quiet layer-0 neurons, the few-neuron new edit, and the same lookup
  spread over all of layer 0's MLP, the thin new edit.

Then E237's five 96-key tasks, three seeds each, every update scaled to one common size: copy returns the key, top the
base model's own most likely next token, mid a token from its top 20, cipher a fixed cipher of the key, and the
prior-matched cipher a cipher whose answers the base model already finds about as likely as copy's. No reuse verdict is
filed for these tasks; E237 found that the re-gating share puts both ciphers apart from copy, top and mid.

**The noise band.** For each built edit, K\* and R are read again on each half of the data; the band is the largest gap
between two halves over the edits read.

## What we expected, and why

Two readings were filed against each other, with no direction between them. In the first, a new edit stores its lookup
where no latent of M with a gain can write it, so many of its circuit's latents must keep F's mechanism and it reads
below every reuse edit by more than the band. In the second, the base dictionary puts whatever it lacks into one
remainder per site, so a new edit's whole new write costs one variable of thousands and it reads as high as a reuse edit.
The remainder share was filed beside R to tell them apart: one minus the KL from F with only the eight remainders
stood in, over KL from F to M. It was expected above one half for the new edits and under 0.3 for the reuse edits. The
key tasks were filed as descriptive, with no direction.

## What we observed

### Do the built reuse edits read above the built new edits?

![R under wording A at SAE latents and at VPD](../../experiments/sae_r_a/figures/r_a_sae_vs_vpd.png)

**Reading it.** Yes, by a margin just over the noise. The filled circles in the top half of the figure are R in latents.
The gain-of-two edit reads 0.9985, the gain-of-ten edit 0.852 and the rebuilt edit 0.694; the thin new edit reads 0.571
and the few-neuron new edit 0.516. The rebuilt edit is the lowest reuse edit and the thin new edit the highest new one,
a gap of 0.123. The band, set by the rebuilt edit's two halves at 0.772 and 0.881, is 0.110, so the filed rule backs
the first reading. The filled squares are the decomposition's reading of the same edits: there the thin new edit reads
0.980, above the gain-of-ten edit and the rebuilt edit, and the same gap is -0.120. The moved edit changes the output by
less than the measured noise floor in both readings and is not read.

| edit | built as | R at SAE, bar 0.05 / 0.1 / 0.2 | R at SAE on each half | lowest possible at SAE | changed members that stand in, SAE | remainder share | R at VPD | changed members that stand in, VPD |
|---|---|---|---|---|---|---|---|---|
| x2 | reuse | 0.9922 / 0.9985 / 0.9999 | 0.9990, 0.9985 | 0.865 | 0.989 | 0.894 | 0.9998 | 0.998 |
| x10 | reuse | 0.8280 / 0.8517 / 0.8855 | 0.8521, 0.8644 | 0.725 | 0.461 | 0.497 | 0.9760 | 0.743 |
| rebuilt | reuse | 0.6918 / 0.6943 / 0.7387 | 0.7718, 0.8814 | 0.608 | 0.220 | -1.579 | 0.8595 | 0.624 |
| moved | reuse | not read | not read | not read | not read | -0.110 | not read | not read |
| new_thin | new | 0.5353 / 0.5710 / 0.6398 | 0.5717, 0.5930 | 0.257 | 0.422 | 0.792 | 0.9799 | 0.819 |
| new_few | new | 0.4661 / 0.5158 / 0.5637 | 0.5085, 0.5246 | 0.222 | 0.377 | 0.775 | 0.8917 | 0.601 |

"Lowest possible" is the share of K\* at untouched sites. "Changed members that stand in" is R counted over K\*'s
changed members only: one minus those that must keep F's mechanism or fit a gain at or under zero, over all of K\*'s
changed members. It is descriptive, not filed; definition-lead's card reports the same share at VPD. The remainder share
is defined above. The decomposition's random updates of matched size read 0.291 and 0.301.

### What carries the order?

**Reading it.** The latents at untouched sites, more than the changed ones. The open circles in the figure mark the
lowest R each edit could read. In latents they sit at 0.865 and 0.725 for the two gain edits and 0.608 for the rebuilt
edit, but at only 0.257 and 0.222 for the two new edits. Most of a new edit's circuit sits in layer 0's MLP, the one
site it changed, so few of its latents stand in for free. Counted over changed latents alone, the rebuilt edit reads 0.220,
below the thin new edit at 0.422 and the few-neuron new edit at 0.377. So the filed order holds, but a reuse edit whose
changed latents mostly cannot stand in still reads above both new edits. That is how wording A counts, since a variable
the fine-tune did not change does run on M's mechanism. It does mean the latent reading's margin is set by where in
the model each edit sits as much as by how it changed.

### Does the remainder share tell new edits from reuse edits?

**Reading it.** No. The new edits' remainder shares are 0.792 and 0.775, above one half as expected. But the
gain-of-two edit reads 0.894 and the gain-of-ten edit 0.497, both above the 0.3 expected of reuse edits. The rebuilt
edit reads -1.579: standing in only its remainders moves the output further from F than M is. That is under 0.3 by the
letter of the rule but is not the small positive share the expectation meant. The share's positive control, the new
edits against the gain-of-ten edit, does not separate them.

### How do the key tasks read?

**Reading it.** Every one of the fifteen reads near one, from 0.989 to 0.999. The task means are 0.9989 for copy and
for top, 0.9988 for mid, 0.9982 for the prior-matched cipher and 0.9898 for the cipher. The cipher sits lowest, below
the others by more than four times its seed range. The prior-matched cipher sits slightly below copy, top and mid, by
about its own seed range. The reason the values sit so high is in the last two columns below. On copy, top and mid the
variables that must keep F's mechanism are four to nine, nearly all of them site remainders, out of 3,916 to 7,274
variables in K\*. On the prior-matched cipher they are the eight remainders and four to seven latents; on the cipher, the
eight remainders and 112 to 134 latents. A trained update changes every site, and at each site its write outside the
base dictionary becomes one remainder, which counts as one variable of thousands. This is the second filed reading,
seen on trained items rather than on the built ones. On the scale the built edits set, every key task reads like the
gain-of-two edit, the ciphers included. With no known label for these tasks and no random update read in latents, we
cannot say whether that is right.

| task | seed | R, bar 0.05 / 0.1 / 0.2 | remainder share | size of K* | variables that do not stand in, of them remainders | KL(F to M), nats |
|---|---|---|---|---|---|---|
| copy | 0 | 0.9982 / 0.9990 / 0.9992 | 0.823 | 3926 | 4, 4 | 4.40 |
| copy | 1 | 0.9983 / 0.9991 / 0.9995 | 0.800 | 4241 | 4, 4 | 4.11 |
| copy | 2 | 0.9977 / 0.9987 / 0.9992 | 0.837 | 3916 | 5, 5 | 4.34 |
| top | 0 | 0.9982 / 0.9990 / 0.9993 | 0.603 | 6069 | 6, 6 | 1.90 |
| top | 1 | 0.9975 / 0.9990 / 0.9995 | 0.622 | 7274 | 7, 7 | 2.00 |
| top | 2 | 0.9969 / 0.9987 / 0.9991 | 0.625 | 6878 | 9, 7 | 1.90 |
| mid | 0 | 0.9980 / 0.9987 / 0.9991 | 0.631 | 5586 | 7, 6 | 2.51 |
| mid | 1 | 0.9978 / 0.9990 / 0.9993 | 0.618 | 5882 | 6, 6 | 2.60 |
| mid | 2 | 0.9982 / 0.9987 / 0.9991 | 0.645 | 5532 | 7, 6 | 2.63 |
| prior-matched cipher | 0 | 0.9919 / 0.9982 / 0.9991 | 0.505 | 7757 | 14, 8 | 3.52 |
| prior-matched cipher | 1 | 0.9901 / 0.9978 / 0.9990 | 0.477 | 6901 | 15, 8 | 3.41 |
| prior-matched cipher | 2 | 0.9906 / 0.9984 / 0.9991 | 0.478 | 7741 | 12, 8 | 3.53 |
| cipher | 0 | 0.9650 / 0.9890 / 0.9987 | 0.395 | 12904 | 142, 8 | 3.40 |
| cipher | 1 | 0.9643 / 0.9897 / 0.9984 | 0.378 | 11984 | 124, 8 | 3.29 |
| cipher | 2 | 0.9675 / 0.9908 / 0.9988 | 0.430 | 12978 | 120, 8 | 3.27 |

"Variables that do not stand in" counts those that must keep F's mechanism plus those whose gain is at or under zero,
at the filed bar. Every key item's K\* search hit its 2,000-evaluation cap; no key item's R search did. Remainder
shares run from 0.378 to 0.837; task means are in `summary_keys.json`.

## What we learned

At base-model SAE latents, wording A orders the built reuse edits above the built new edits, where the parameter
decomposition did not. The margin is barely above the noise and comes mostly from how many of each circuit's latents
sit at sites the edit never touched. On trained fine-tunes the reading is pinned near one, because each site's
out-of-dictionary write is a single variable. For the paper, the latent row reads the built labels, narrowly, but it
cannot yet be used to sort trained fine-tunes into reused and new.

## How compelling is the evidence

- **Lower bounds.** Every K\* search hit its cap of 2,000 evaluations, so each K\* may hold more variables than the
  smallest circuit. The R search hit its cap on the gain-of-ten, rebuilt and both new edits, so those R values are
  lower bounds. The cap applies to both sides of the gap, so it could move either way. The rebuilt edit's main
  search reads below both its halves, and its second half's search did not hit the cap.
- **One featurization, one seed.** One SAE seed, eight sites. The four attention dictionaries are more than half dead:
  0.530 to 0.616 of their latents never fire on held-out rows, as E234 measured.
- **No random control in latents.** The only random updates read under wording A are the decomposition's, at 0.291 and
  0.301. Without one here, we cannot say where a change with no structure reads in latents.
- **The band comes from one edit.** The band of 0.110 is the rebuilt edit's two halves; every other edit's halves sit
  at most 0.021 apart. The gap clears it by 0.014.
- **Items covered.** Five of the six built edits were read; the moved edit is under the noise floor. All fifteen key
  items were read, without halves. E187's nine trained pairs were not read.
- **Not bijective.** A dictionary of 16,384 latents for a 768-wide output is overcomplete, so a site's output does not
  fix its latents uniquely; the translation from module to variables is not one-to-one.

## Surprises

- The rebuilt edit, a reuse edit, reads below both new edits once only its changed latents are counted. The filed
  order holds only through latents at untouched sites.
- The remainder share did not separate new from reuse: the gain-of-two edit reads 0.894, above both new edits.
- Every key item's must-keep set is mostly site remainders, so the key reading sits at its ceiling.

## What next

1. A random update at matched size, read in latents on the copy windows. If it reads near one, the latent reading has
   no floor for trained items and should be reported as blind to them; if it reads low, the key tasks' near-one
   values say something.
2. E187's nine pairs at matched size, the stage left out here, only if the random control reads low.
3. Counting the remainder as one variable is what pins trained items near one. Whether it should count by its share of
   the site's write is a definition question for the user, not a measurement change.

## Scorecard

| id | claim in words | rule | read | verdict |
|---|---|---|---|---|
| R0 | every job's do-nothing exact; per-item controls exact | exact zeros; revert within 1e-3 | all 14 jobs' do-nothing read R one, floor under 1e-6 | held |
| order check | every reuse edit read above both new edits | min reuse R minus max new R above zero | 0.694 against 0.571, moved not read | held; key tasks read |
| band | largest split-half gap | over the built edits read | 0.110, from rebuilt | set |
| P1 | rival a: gap above the band; rival b: gap at or under zero with both remainder shares above one half | gap against band | gap 0.123, band 0.110 | rival a, narrowly |
| remainder, new | new edits' remainder share above one half | descriptive | 0.792 and 0.775 | as expected |
| remainder, reuse | reuse edits' remainder share under 0.3 | descriptive | 0.894, 0.497 and -1.579 | not as expected |
| D1 | key tasks, descriptive | none filed | 0.989 to 0.999 | read |
| D2 | nine pairs, descriptive | none filed | not run | not run |

## Departures from the literature

- **The dictionaries.** BatchTopK SAEs as in SAEBench, at 16k width, but trained here on the base model's
  block outputs; E234 lists its departures from SAEBench's recipe. The attention dictionaries' dead share, more
  than half against SAEBench's reference of about an eighth, is a departure in outcome.
- **The remainder as a variable.** Treating each site's reconstruction error as one variable follows the
  featurized-model construction in the paper draft; SAE circuit work such as Marks et al.'s sparse feature circuits
  also keeps one error node per site. Here that node is one variable in R's count, which the literature has no
  counterpart for.
- **Soft intervention with one gain.** Geiger's soft intervention replaces a variable's mechanism; wording A allows
  M's mechanism times one positive gain per variable, fitted by least squares over all positions. That gain is ours.
- **K\* by mean ablation** with E226's search and bar, not a published circuit method.

## Not run

- A random update in latents, the null for this featurization.
- E187's nine trained pairs at matched size.
- Split halves for the key items.
- The residual-stream dictionaries: under the gain-of-ten edit the last residual dictionary explains less than
  nothing, about minus sixteen in fraction of variance, as E234 found.
- A second dictionary seed; an uncapped search; gains fitted on the behaviour rather than on the writes.
- A neuron reading under wording A: none exists, so the figure has no neuron column.

## Provenance

Results `experiments/sae_r_a/results/items/*.json`, stage 0 `items/stage0_*.json`, logs `run_built0-5.log` and
`run_keys0-7.log`, every log ending FINISHED; keep masks and gains in `results/large/items_*.npz`. Built edits: four
cluster debug jobs on L40 cards for the copy-window edits and H100 pods for the two new edits; key items on H100 pods
and the cluster. The decomposition's reading: `experiments/kstar_vpd_wording_a/results/summary.json` and `items/`,
E242, definition-lead. Numbers ledger: this file's `.numbers.tsv`.
