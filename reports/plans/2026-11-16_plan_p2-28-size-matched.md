# P2.28: do the readings still separate the pairs once the two sides are matched in size?

*Cell `P2.28` (part 2, rung L2.3, 1 run, due 2026-11-18), compare-lead, 2026-11-16. Experiment number: not yet claimed
(`E187` in the files; the lead claims it with `fleet/ecode`). Code `experiments/size_matched/` (`run.py` 67M, `qwen.py`
Qwen, `plot.py` CPU), rules `experiments/size_matched/results/predictions.json`, card draft
`~/commute-data/cards/P2.28-draft.md`. Not gated (about $19 and 1.5 h).*

We shrink the larger side of every pair's saved fine-tunes by one number until its weight change matches its partner's,
and read the share and the cheap readings again on that side only. If a reading's gap between the sides survives the
shrinking, it carries something besides update size; if it vanishes, update size explains it. No reading is called
reused or new.

## The question in the Terms' words

The 18 November gate page (`reports/analyses/2026-11-16_analysis_part2-gate-page-update.md`) reports that on the 67M
the re-gating share separates all nine built pairs, three already matched in size, while the cheap readings' gaps may be
update size; and that on Qwen's frozen-neuron decomposition (p-015e0710) the side with the smaller weight change reads
the higher share in all six pair cells. This run asks, for each reading and pair: with the larger side's whole update
δW multiplied by one scalar c so that the side means of ||δW|| / ||W0|| are equal, does the gap stay beyond the floor?

## Design check (`/design`)

- **Goal.** `GOALS.md`'s part 2: what the share tells about an update beyond cheap baselines. It fills open cell P2.28.
- **Nearest tried.** E54's matched cipher and C3's pairs were built matched in size (the share separates them, the
  readings mostly do not); P3.7 chose the "5x" save to bring Qwen sides closer in size, but they still differ 2 to 2.6
  times. Neither rescaled a saved update, so neither tells a reading's own size response from the pair difference.
- **Patch or step.** A premise check on the part 2 scorecard: the gate page's doubt "apart never asks whether a reading
  separates beyond size".
- **Decisions.** Relies on the scorecard rule (one rule, seed ranges) and P3.7/P3.9's filed rules, unchanged. Departure,
  by the user's scope: a rescaled update is not a trained state. No verdict rule is added.
- **Definition.** Descriptive only; no reuse/new verdict, no new clause.
- **What would change what we do.** The share "lost" on 67M pairs would remove the gate page's one size-free claim; the
  Qwen share "kept" on a matched cell would give the first Qwen pair the share separates beyond size; every cheap reading
  "lost" makes size the baseline that the paper's comparison must beat.
- **Apparatus check, first and alone.** c = 1 through every reader on states read before (expected difference 0 on the
  67M's L40, within filed tolerances on Qwen), then the random floor. Each stops the job on a miss.
- **Saved for audit.** Every per-state reading (m67.json; Qwen share records with per-row cells; readings.json).

## What runs

Matching: per pair cell, c = mean size of the smaller side / mean size of the larger side, every larger-side state times
c (one scalar for all matrices); side means then match and each side keeps its seed spread (my implementation choice).
Size is ||δW|| / ||W0|| over all matrices, P3.7's size. The smaller side is never read again: its saved readings stand.

| | 67M (`run.py`) | Qwen (`qwen.py`, one pod per pair) |
|---|---|---|
| pairs | the nine built pairs | copy − cipher, known − invented, gold − shuffled gold; late and 5x saves |
| larger side, c (from saved sizes) | new side on 7 (c 0.53 to 0.99), reuse on the two C3 pairs (0.989, 0.998) | cipher (0.46, 0.38), invented (0.47, 0.49), gold_shuf late (0.97), gold 5x (0.81) |
| states read | 40 | 6 per pair (3 seeds x 2 saves), plus 1 at c = 1 |
| share | E136's trained route split | P3.9's split on p-015e0710 (MLP matrices only) |
| cheap readings | steering (E143; E62 from E135), top neurons k64/kmatched (E143), neurons for half the gain (E98/E172), effective rank | P3.7's Lab.readings: steering, top-neuron overlap, effective rank |
| floor of a gap | seed ranges (the scorecard's rule); a random change read for its gain only | share: band = random gap at the matched size + 0.00196; cheap: P3.7's floor bands, which span the library's whole norm range |

New decomposition p-9707d6ff is not read: every update and random change there reads within about 1% of the ceiling.
The prior does not depend on the update; effective rank is unchanged by one scalar (checked per state, not drawn); size
is matched by construction.

**Stages.** 67M: stage 0 alone (c from E136's and E98's files agree; c = 1 reproductions: share, top neurons and
steering on e54/cipher_s0 and e58/cipher_s0, E135 steering on e62/lookup_s0, k_half on cipher_s0 and e58/cipher_s0;
random floor), then 40 states, each written as it lands, projected time checked after four. Qwen, per pod: `--part share`
(stage 0: W0 and do-nothing controls, c = 1 on seed 0's late larger state, then the random floor of each save, which must
be "scored"; stage 1: six scaled states; files kept), then `--part readings` (stage 2: c = 1 against P3.7's saved
readings; stage 3: six scaled states, effective rank checked unchanged, files deleted).

## The numbers and what each should read

| Number | Control (c = 1) | Do-nothing on both sides | Expected |
|---|---|---|---|
| 67M share gap, matched | saved value, difference 0 | 0, no call | kept on all 9 (P1: >= 8) |
| 67M steering, top neurons, k_half gaps | saved, difference 0 (k_half equal) | 0, no call | unchanged on the 3 pairs with c >= 0.987; no direction filed on the 6 others |
| 67M effective rank | equal | undefined | equal to unscaled (1e-6), a check of the scaling |
| 67M random change's gain | n/a | 0 | well under 1 log-odds: shares of it undefined |
| Qwen share gap, matched | P3.9's share within 0.0002 | 0, inside band | closer to 0 than unscaled in >= 5 of 6 cells (P2) |
| Qwen random floor share | n/a | n/a | "scored" (stop otherwise) |
| Qwen overlap gap, matched | P3.7's within 0.003 | 0, inside floor | closer to 0 in all 4 cells beyond floor unscaled (P3) |
| Qwen steering gap | P3.7's within 0.002 | 0 | unreadable on the built pairs (no floor), inside on gold, before and after |

Outcome per reading and cell: kept, flipped, lost, gained, neither, unreadable (`predictions.json` rules). The two rivals:
size only predicts every separated cell reads "lost"; more than size predicts "kept". Unscaled, steering separates 7 of
9 67M pairs, the top neurons 9, neurons for half the gain 6, the share 9 (final.json); on Qwen the share is beyond band
on facts late and both gold saves, the overlap beyond floor on copy − cipher (both), facts 5x and gold late.

Tolerances: 67M from the earlier readers' filed tolerances, every one measured 0 on an L40
(`experiments/holes_fill/results/stage0.json`, `experiments/refit_96key/results/stage0.json`,
`experiments/k_half_fill/results/cells.json` stage0). Qwen's were never measured across cards on the same state: they
are a tenth of the share's seed spread, the smallest steering seed range of a larger side, and a tenth of the smallest
overlap band (sources in `predictions.json`). Edge cases (do-nothing, c = 1, ties, k_half None, nan, None gaps) were
checked by hand against the rule code, not run: `cpu-run` is down.

## Cost and time

| Stage | Card | Transfer (at ~10 MB/s up) | Time | Cost |
|---|---|---|---|---|
| 67M, cluster `debug` if it is up | L40 | none (NFS) | ~15 min | $0 |
| 67M, else a pod | RTX 4090 (~$0.69/h) | 9.4 GB of results/large up: 940 s, ~16 min; vpd from W&B ~1 min | ~5-8 min set-up + 16 + 15 = ~40 min | ~$0.5 |
| Qwen, one pod per pair (3 in parallel) | H100 ($3.99/h; A100 80 GB fallback; not under 48 GB, the share peaks at 38.6 GiB) | code + 4 MB gstats up: seconds; p-015e0710 31 GB from W&B ~8 min (P3.9's pod log); 34 GB of states from W&B at 30-45 MB/s, 13-19 min, in the background | set-up ~40 min, share part ~25, readings ~15-27 (P3.7: 105-225 s a state) = ~85-95 min | ~$6 a pod, ~$18 |
| Downloads | | results JSON under 2 MB: seconds | | |

Total about $19, wall about 1.5 h. One pod running the three pairs in turn would cost ~$12 and take ~3 h. Container disk
per Qwen pod at least 100 GB (decomposition 31 + six states 34 + model and venv). This is above the cell's "a few
dollars": see the hand-back.

## Not run

Plan only: no job, no pod. Lint, tests and the plot are not run (`cpu-run` down); the code is committed unlinted. The
card's sketch is not drawn.
