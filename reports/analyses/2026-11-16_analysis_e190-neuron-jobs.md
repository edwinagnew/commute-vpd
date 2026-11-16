# P2.30, due 11-18: do the neurons a fine-tune moves keep their job, and does its gain survive top-K?

We asked two Goodfire-style questions of every saved fine-tune on the nine 67M built pairs. First, do the neurons a
fine-tune moves most still fire hardest on the same Pile text afterwards? Second, does the fine-tune's gain survive
when each block keeps only its strongest neurons per token? The neurons moved most keep about three quarters of their
top text, well under a random update of the same size. On 7 of 9 pairs the side built to reuse keeps more than the side
built to be new, by more than the seed range. The sparsity question cannot be read on this model: the mask breaks the
model itself. For `GOALS.md` this is one more cheap reading that orders the built sides the way the share does. It is
descriptive, with no reuse/new verdict.

**Figure.** `experiments/neuron_jobs_k/figures/jobs.png`: top left, the reuse-minus-new retention gap per pair with its seed band and the random updates' gap; top right, each side against its random update; bottom, gain kept at each K, marked unread.

**Job retention.** For each fine-tune, the 32 neurons per block whose firing on the task changed most; for each, the
share of its 20 strongest Pile windows (of 2,000) that stay its strongest afterwards. The check passed first: neurons
made to fire harder kept 0.67 of their windows, neurons given each other's jobs kept 0.10. Real fine-tunes keep 0.60 to
0.80 (side means), against 0.88 to 0.97 for a random update of matched size per matrix, on all 18 sides. The reuse side
keeps more on 7 of 9 pairs, by gaps of 0.04 to 0.11; lowercase vs cipher and the c3 copy pair give no call. The
random-update floors of the two sides of a pair differ by at most 0.02, less than the gap on every pair called, so the gap is not what any update of that
size does. Whether update size explains it otherwise was not checked here (P2.28 matched sizes for the other readings).

**Gain kept under top-K: unreadable.** The check failed at every size. With no update at all, keeping 240, 768 or 1,536
of 3,072 neurons per block adds 5.76, 3.79 and 0.89 nats to the model's Pile loss. An update placed on the neurons most
often kept should survive the mask; at 1,536 it kept 0.50 of its gain against 0.16 under random neurons, short of the
0.4 gap filed. Goodfire's top-K works on a much wider model; this dense 67M is not sparse enough per token. The
per-pair numbers are in `stage2.json` and are not read.

**Filed leans.** P1 (each side under its random update) held on 18 of 18; P4 (retention gives no call on at least 6
pairs) did not hold: 2 of 9. P2, P3 and P5 concern the unreadable sparsity reading.

**Not run.** Update size as a covariate for retention; retention on Qwen; Goodfire's light fine-tune under the mask,
percentile bands, router weighting; any K above 1,536.
