# E133 plan: does the original model's prior order every saved map the way the re-gating share does?

*2026-11-09, cell P2.3 of the table of work (one run). Inventory: `2026-11-09_plan_e133-prior-vs-share_appendix.md`.
Rules: `experiments/prior_vs_share/results/predictions.json`. The labels are as built; no verdict of reused or new is
given, the definition being under revision in cell P1.1. The rule reads orderings only.*

## Design check (the page's draft, checked)

- **Goal.** `GOALS.md`, the mechanism question ("strengthen, reveal, create, brittle") through the plan's part 2: what the re-gating share measures. It asks whether the share carries anything the original model's prior for the answer does not. Fills open cell P2.3.
- **Nearest row.** `experiments/TRIED.md` rows E58 Amendments 6 and 7 (six and eight maps, 0.886 and 0.95, both computed after the runs). This widens them to every saved map and files the rule first. A step, not a patch: it is the rival comparison part 2 asks for.
- **Decisions.** `decisions.md` rows 112 and 184; none departed from. The prior is a reference, never a label.
- **Definition.** Does not apply: no verdict; the rule reads two orderings. Built labels are quoted.
- **What changes next.** Agreement near 0.9: the paper reads the share as the prior, except on the maps off the line. Much weaker: the claim that the share carries more becomes testable. In between: no call; the run count is the cap, so the cell closes on that reading.
- **Control first, alone.** The forward on the copy map's windows reproduces the saved 5.91e-5, and on E54's elicit windows the saved 1.704e-3 (the same quantity, file and key checked; the appendix's 1.33e-3 and 1.38e-6 are the 96-key elicit and cipher, in the inventory). Tolerance 1e-4 relative, from the exact reproductions on this path (`job2.json .stage0.*.rate_rel` 0). Stop on either miss. It can fail on a changed windows file, the wrong precision or a broken forward.
- **Numbers, with expected values.** The two controls (above); the per-map prior (no value at W0 other than the prior itself; a zero update has no share); Spearman 0.75 to 0.95 expected, the null at this n near 0.34 to 0.40; maps off the line expected among the copy rows, high, and the matched-prior rows, low. A do-nothing state reads the same share as its fine-tune, so it sits at the map's point; a zero update is under the gain floor and enters nowhere.
- **Guards and saves.** Each stage logs and writes as it finishes. No state is saved; the three result JSONs hold everything an audit needs.

## What is run

One job, MATS `debug` on the L40, $0. Stage 0 alone (the control), then stage 1 (the prior of the 24 maps: 12 read from
the file that holds them, 12 from one forward of the original model on the saved windows), then stage 2 (the saved shares,
no new share read; the score). No fine-tuned state is loaded.

- **Inventory** (24 maps in, each listed with file and key in the appendix): E46's four maps of 32 keys (copy twice, random twice); E54's elicit and cipher (32 keys); E58's four maps at 24 keys (elicit, cipher, copy, rank-1), eight seeds each from E55 C3; E58's eight at 96 keys (rank-1, elicit, copy, cipher, shuffled elicit, prior-matched cipher, class rule, its twin); E62's copy and lookup at the new marker; the label pairs' four arms (E123 and E125, real and twin). The label pairs enter one point per arm, the prior a geometric mean of P(each window's own label) on their 256 read windows, as their share was read; they sit near a prior of 1e-6, the low end. Out, with reasons in the appendix: `first_letter` (no share read); E53's three (windows not saved, rebuilt from Pile); the regating-table and ladder maps outside the cell's named files, named for the lead.
- **One point per map:** the mean share over seeds, the range kept. A map is one trained table on one key set, so the same table at 24, 32 and 96 keys counts three times, and the points are not independent: `blind_to` says so.

## Numbers and readings (filed in `predictions.json`)

- **Spearman** of the map-mean share against log prior, over the maps entering; **its interval**, percentile 2.5 to 97.5 of 10,000 bootstrap resamples of the maps (seed 133); the **permutation null**, the 95th percentile of Spearman with the shares shuffled over maps, 10,000 draws (seed 134), and the one-sided p.
- **Reading.** `agree`: rho at least 0.80, the interval's low end at least 0.60, rho above the null. `weaker`: rho at most 0.60 and the interval's top under 0.80. `no_call`: everything else (a rho between, a nan, fewer than 12 maps). The bands come from the two earlier filed-after readings and the page's "near 0.9" and "much weaker"; 0.60 is the midpoint of 0.80 and the expected null.
- **Off the line.** A line of share on log10 prior fitted on the other maps, read at each map; noise = 1.4826 × the median absolute deviation of those leave-one-out residuals. Off the line: |residual| over 2.5 of those and over the map's own seed range. Names and signs only; no rule reads them further.

**Blind to.** Spearman reads ranks only and treats correlated points as independent; the interval is too narrow for that reason. The off-line reading depends on the line through the other maps and does not say why a map sits off. The prior is the geometric mean of P(answer), not rank. Neither number speaks to reuse.

**Edge cases, run on the code before filing** (`predictions.json` edge list): too few maps, ties in share and in prior, a zero probability, a single-seed map, a zero update, a do-nothing state.

## Cost

Under 5 minutes on the L40 (model load 1 to 2 minutes, about 2,000 windows in small forwards, the bootstrap and permutation
under a minute of CPU; E60 read 18 states in about 90 s, job 46258, and E98 read 56 states with many forwards each in about
4 minutes on the same card, `experiments/TRIED.md` rows E60, E98). `--mem=16G` (E98 peaked at 6.1 GB of host memory, job 52786; the node had 69 GB unallocated when checked),
`--time=00:20:00`. $0. A failure there costs one debug job. The job's figure comes from `plot.py` afterwards, no model.

## Not run

No new share read. No fine-tuned state, no steering, no neuron reading (other cells). Nothing about reuse.
