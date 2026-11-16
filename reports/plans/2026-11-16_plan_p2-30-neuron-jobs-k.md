# P2.30, due 11-18: E190 (neuron_jobs_k), E189 rerun with a K grid

Card `reports/plans/2026-11-16_plan_p2-30-card.md` (descriptive, user dialog 2026-11-16, ask 21). Filed rules
`experiments/neuron_jobs_k/results/predictions.json`. One job, MATS debug L40, ~10 min, $0.

**Why.** E189 (P2.29, `reports/plans/2026-11-16_plan_p2-29-neuron-jobs.md`, design check there) stopped in stage 0: the
retention check passed (fire harder 0.67 vs jobs swapped 0.10), but an update placed on the neurons most often among
each block's top 240 kept only 0.17 of its gain under the top-240 mask (random 240: 0.01; filed gap 0.4). The premise
check (`~/commute-data/premise/P2.29.md`, doubtful) asked what the mask does to W0 alone; nothing measured it.

**Changes from E189, nothing else:**
1. Gain kept, its random-K floor and the pair call read at K = 240, 768 and 1,536 of 3,072 (7.8%, 25%, 50%).
2. The sparsity check (often/rare update) is recorded at each K, not a stop. The paragraph reads gain kept only at the
   K values where the often-kept update clears E189's 0.4 gap, and says so for each K.
3. New floor: per K, with no update, how much of W0's task log-odds the mask keeps, and the Pile next-token loss it
   adds (val rows 4048-4304, held out from the retention windows).
4. Projected-time stop 20 min (three K values; debug allows 2 h).

Retention reading, its controls (zero update, harder vs swapped, stop on a miss), pairs, random updates and rules are
E189's unchanged: the code imports `experiments/neuron_jobs` rather than copying it.

**Expected.** W0 under the mask: at K 240 most task log-odds lost and over 1 nat of Pile loss; at K 1,536 under 0.3
nat. The often-kept update clears the gap at 1,536, not at 240 (P5). Retention: real fine-tunes below their random
update on most sides (P1).

**Not run.** As E189: Goodfire's light fine-tune under the mask, percentile bands, router weighting.
