# Does Figure 2 change when all three featurizations read the same states the same way?

We read the write share of the needed set at VPD subcomponents, top neurons and SAE latents on the same 48 states, with one needed-set rule, one set of input pairs and the leftover variables counted. No per-task value moved by more than a hundredth from the old figure, and the old order holds at every featurization: edits built to reuse read far above edits built to be new and above every random update. The old figure's mixed protocol therefore did not drive its picture, and Figure 2 can be redrawn from these rows.

## What changed from the old Figure 2

The reading is the share of the update's write change carried by the needed set K, where K is found by the swap test over 256 pairs, 4 derangements of 64 inputs at seed 268, and the leftover variables are in the scan and in the sums.

- **SAE latents.** Now read over the swap-test K on the full, unscaled fine-tunes. The old column read over K* on states scaled to one update size.
- **VPD subcomponents.** Seeds 1 and 2 of every trained task are read for the first time; the old column had seed 0 only. The copy task's seed 0 is read again by this driver instead of copied from E275: 0.033853 became 0.033848.
- **Top neurons.** The headline now counts each layer's pooled remainder. The old column plotted the value without it; the no-leftover value is kept beside the headline and in the appendix.
- **New groups.** The three head (2,4) edits of E280, on the value, output and query weights, are their own built group, each with a matched random.

Per task, the largest move is 0.0074, on the SAE rebuilt edit, and no task moved by more than the 0.05 gap that would have read as the set-up mattering. Per row, 31 of the 91 old rows moved by more than 0.001, all at SAE latents; the largest is 0.0131, on the copy task's seed 2. VPD and neuron rows moved by less than 0.001.

## Order check and the SAE controls

At VPD the smallest reuse value is 0.623 against a largest new or random value of 0.0188. At neurons it is 0.0476 against 0.0330, and at SAE 0.127 against 0.0410. The filed SAE expectation held: the x2 edit read 0.966 over a needed set of 43,586 variables, inside the filed band between the VPD and neuron values. The must-flip control swapped every latent and error term from the other input at once: all 10,248 cases whose answers differ flipped, and none landed off the other input's answer.

## Review points from the plan check

- **Leftovers differ in size.** VPD has 24 per-matrix remainders, neurons 4 whole-layer remainders and SAE 8 error terms. A coarser leftover is needed more often, so it carries more of the change; the `leftover_share` column of `definition_rows.csv` gives its share per row.
- **Two shared randoms use their own recipes.** `random_size_241001` is matched to the norm of one new edit and `random_kl_241002` to its divergence. Neither is E290's per-tensor Gaussian, so they are plotted as their own group and never pooled with the per-edit randoms.
- **Trained against random at neurons.** Each task's mean over three seeds, divided by its matched random: copy 2.23, rank-one 2.42, elicit 2.87, cipher 3.14, prior cipher 3.05. Trained sits two to three times above its random, not near it.

## Caveats

- At the copy task's answer position the base model's top token is right on 0 of 301 cases, so the built site is not a behaviour the base model performs there (`experiments/built_circuits/results/planning/measure.json`).
- Seven of the eight neuron items ran under the predictions file before the SAE tolerance was filed, `9486e64a` against `c204a7a6`; nothing for VPD or neurons differs between them.
- The SAE identity control first showed 1 flip with a largest logit deviation of 0.248, because the JumpReLU threshold flipped on rounding between the cache and the rerun. After commit `60062eff` the input's own side is taken from the cache: 0 flips, deviation 2.16e-4, against 5.15e-5 at VPD and neurons.
- The value and output edits of head (2,4) read the same at neurons and at SAE, to every digit. This is expected: doubling a head's value weights or its output weights doubles the same head output, so the two edits give the same model function and every reading of activations sees one edit. Only VPD, which reads the weights themselves, tells them apart.
- The head edits read about a fifth at VPD although each is a pure rescale of one head. E280 found that VPD's subcomponents mix heads, so a one-head rescale cannot sit inside the parts the base model uses. The query edit reads below its matched random at neurons and at SAE: doubling the query sharpens the attention pattern rather than scaling a write, so at these sites it is not a rescale of an existing path.

## Moved rows (old, new)

Every old row that moved by more than a thousandth (`experiments/fig2_standard/results/moved_rows.json`, rounded to four places). All are SAE rows, whose old column was read over K* on scaled states.

| Featurization | Block | Item | Side | Old | New | Move |
|---|---|---|---|---|---|---|
| SAE | built | x2 | reuse | 0.9689 | 0.9659 | -0.0031 |
| SAE | built | x10 | reuse | 0.9473 | 0.9433 | -0.004 |
| SAE | built | rebuilt | reuse | 0.1349 | 0.1275 | -0.0074 |
| SAE | built | new_thin | new | 0.0127 | 0.0109 | -0.0018 |
| SAE | built | random_size_241001 | random | 0.008 | 0.0136 | 0.0056 |
| SAE | built | random_kl_241002 | random | 0.0198 | 0.023 | 0.0033 |
| SAE | trained | copy_s0 | trained | 0.0473 | 0.0451 | -0.0022 |
| SAE | trained | copy_s0 | random | 0.0118 | 0.0103 | -0.0015 |
| SAE | trained | copy_s2 | trained | 0.0552 | 0.0421 | -0.0131 |
| SAE | trained | copy_s2 | random | 0.0112 | 0.0146 | 0.0034 |
| SAE | trained | elicit_s0 | trained | 0.0336 | 0.0358 | 0.0022 |
| SAE | trained | elicit_s0 | random | 0.0116 | 0.0147 | 0.0031 |
| SAE | trained | elicit_s1 | trained | 0.0352 | 0.0374 | 0.0022 |
| SAE | trained | elicit_s1 | random | 0.0148 | 0.0119 | -0.0029 |
| SAE | trained | elicit_s2 | trained | 0.0292 | 0.0307 | 0.0015 |
| SAE | trained | rank1_s0 | random | 0.0093 | 0.0081 | -0.0012 |
| SAE | trained | rank1_s1 | trained | 0.04 | 0.0353 | -0.0046 |
| SAE | trained | rank1_s1 | random | 0.0097 | 0.0084 | -0.0012 |
| SAE | trained | rank1_s2 | trained | 0.0365 | 0.0325 | -0.004 |
| SAE | trained | prior_cipher_s0 | trained | 0.0395 | 0.0456 | 0.0061 |
| SAE | trained | prior_cipher_s0 | random | 0.0156 | 0.0115 | -0.0041 |
| SAE | trained | prior_cipher_s1 | trained | 0.0424 | 0.0471 | 0.0047 |
| SAE | trained | prior_cipher_s1 | random | 0.0097 | 0.0167 | 0.007 |
| SAE | trained | prior_cipher_s2 | trained | 0.0396 | 0.0435 | 0.0039 |
| SAE | trained | prior_cipher_s2 | random | 0.0096 | 0.0136 | 0.004 |
| SAE | trained | cipher_s0 | trained | 0.0379 | 0.0358 | -0.0021 |
| SAE | trained | cipher_s0 | random | 0.0093 | 0.0165 | 0.0072 |
| SAE | trained | cipher_s1 | trained | 0.0374 | 0.0398 | 0.0024 |
| SAE | trained | cipher_s1 | random | 0.0096 | 0.0106 | 0.0011 |
| SAE | trained | cipher_s2 | trained | 0.0363 | 0.0379 | 0.0016 |
| SAE | trained | cipher_s2 | random | 0.011 | 0.014 | 0.003 |

## Not run

- The appendix figure with the leftovers dropped.
- The other comparison figures of the results section under one protocol.
