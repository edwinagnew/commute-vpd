# Prior by mechanism, first look from saved readings (cell P2.54, run 1 plus saved proxies)

We switched off the base model's copy head on seven kinds of fine-tune and set the result beside readings already saved
for the same fine-tunes. Two pairs have matched priors but different mechanisms, and on both the re-gating share and
steering tell the two sides apart, while the prior cannot. No fine-tune in hand has a low prior and a reused mechanism,
so the corner that decides claim 2 is still empty; run 2 has to build it.

## The table

Head lost = share of the fine-tune's gain lost with copy head (2,4) switched off (E249; random heads move it at most
0.074). Prior = log10 of the base model's probability of the answer. Share = the re-gating share, log-odds form.
Neurons = the neuron version of the share. Steering = the steering share. Each value is the mean of 3 seeds.

| Fine-tune | Prior | Head lost | Share | Neurons | Steering |
|---|---|---|---|---|---|
| E62 copy at a new marker | -3.50 | 0.53 | 1.09 | 0.88 | 0.56 |
| E62 lookup at that marker | -3.62 | 0.00 | 0.40 | 0.73 | 0.15 |
| key copy | -4.23 | 0.25 | 0.77 | 0.60 | 0.74 |
| key prior-matched cipher | -4.43 | 0.00 | 0.37 | 0.73 | 0.14 |
| key rank1 | -1.90 | 0.01 | 0.87 | 0.70 | 0.19 |
| key elicit | -2.88 | 0.00 | 0.73 | not read | not read |
| key cipher | -5.86 | 0.00 | 0.27 | 0.72 | 0.16 |

Sources: head lost `experiments/prior_mech_2x2/results/stage1/summary.json`. Key-task share and prior come from
`experiments/key_grid/results/summary.json` (`table.r`, `log10_prior`), at the common update size. Every other column
comes from `experiments/method_table/results/final.json`, which names each source file. The key-task neurons and
steering were read at each fine-tune's own size, not the common size. The E62 share was read from E62's own refit
decomposition (`share_untrained` stage 1), which can read above 1.

## What it shows

- **Two matched-prior pairs.** Copy and the prior-matched cipher sit 0.2 decades apart in prior, and E62's copy and
  lookup 0.12 apart. On both pairs the copy head carries the copy side and not the other side.
- **Which readings follow the head.** On both pairs the share follows the head (gaps 0.40 and 0.69), and so does
  steering (gaps 0.59 and 0.41). The neuron share follows it on E62 and reverses on the key pair.
- **The prior.** Matched by construction, so it cannot split either pair.
- **The low-prior corner is empty.** E62's copy has a higher prior (-3.50) than the key copy (-4.23), so it is a
  second matched-prior pair, not a low-prior reused item. A fine-tune whose answer sits near the cipher's prior
  (about -5.9) and still goes through the copy head is what run 2 must build.
- **Caveat, key copy.** Key copy reads "between": 0.25 of its gain goes through the head, over the 0.2 bar but half
  the 0.5 expected. The task repeats the current token, so there is no earlier token for the head to match; the
  rest of its gain goes some other way.
- **Caveat, rank1 and elicit.** Both read a high share and do not use the copy head. They reuse other knowledge the
  base model has, which this one head cannot label.

## Not run
- Any low-prior reused fine-tune (run 2).
- Neurons and steering on the key tasks at the common update size, and on elicit at all.
- A matched random update through the head switch-off.
- The share on E62 under the same refit as the key tasks.
