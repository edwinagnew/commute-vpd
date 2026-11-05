# Independent check: does the sum of single reverts separate fed from written gains? (E100, Amendment 1)

**Verdict: holds with caveat.** The number is computed as described, it covers the whole update, and its split between the copy maps and the cipher is stable over seeds and not explained by update size, spread, steps, keys, gain or attention-versus-MLP. But it is not shown to read "fed versus written": every task that reads fed is one whose answer is the key token itself, and the one task built to strengthen an answer the model already prefers, outside copying (rank-1, filed "fed" in E100), reads written. The claim should be stated as "separates the copy maps from the key-table maps", with "fed versus written" as the hypothesis it suggests.

*Fresh Opus reviewer, 2026-11-05; wrote none of E100. Read: both plans, `experiments/fed_or_written/{run,amend1}.py`, `src/commute/{locator,fed_rules}.py`, `src/commute/finetune.py`, `src/commute/keydep.py`, `experiments/open_pair/amend1.py`, all E100 results files. Computed on CPU (no model): per-part update norms of 47 saved states, `~/commute-data/scratch/mechanism-detection-lead/e100-check/part_norms.{py,out,json}`.*

In plain words: I checked the code that reverts one part of a fine-tune at a time and adds up how much of the gain each revert removes, and I compared the tasks on everything else that could make that sum differ. The code is right and the copy tasks really do sum lower than the others on every seed, but the only tasks that sum low are those where the answer is a word already in the input. For the project's goal, this means the sum is a good detector of the copying mechanism, and has not yet been shown to detect reuse of existing machinery in general.

## 1. Code: what is reverted, and what the share is measured against

- **A head's revert is correct** (`src/commute/locator.py:214-224`). It zeroes head h's rows `h*128:(h+1)*128` of q, k and v (its outputs; weights are (out, in)) and the same columns of o (its inputs). All 24 matrices are 768×768 (no grouped key/value heads; `fleet/ckpt-info` on `copy_s3.pt`), so no slice is silently empty. The plan says the reverse ("q, k, v columns and o rows", `reports/plans/2026-11-05_plan_e100-fed-or-written.md:26`); that line is wrong and the code is right. Changes nothing in the numbers; fix the plan's wording if it is reused.
- **An MLP's revert** is its `c_fc` and `down_proj` (`locator.py:230`); the 67M's MLP has no other matrix.
- **The 28 parts partition the whole update.** Only the 24 decomposed matrices train (`src/commute/finetune.py:196`, `decomposed_params`); embeddings, unembedding and norms are never updated, so there is nothing outside the parts. Measured: the 28 parts' ‖δW‖² shares sum to 1.000000 on all 47 states (`part_norms.out`). Note that E100's own check that "all 24 reverted reads G 0" (`run.py:87,110`) is true by construction, since reverting every key of the dict is W₀, so it guards nothing; the partition is what makes the sum's "1 when additive" (`fed_rules.py:60`) true.
- **The share** is (mean log-odds of the state − mean log-odds with the part reverted) / G, with G = mean(state − W₀) over the same reads, unit "lo" (`locator.py:240-242`, `177-179`). Same site, unit and reads for numerator and G. The fresh seeds are read on the sites of seeds 0-2, which are the windows they were trained and labelled on (`open_pair_instruments/common.py:36`; E58's G for each fresh seed equals amend1.json's G to every digit). Stage 0 re-read E62 copy seed 0 at 0.262773, equal to E100's value. No finding.

## 2. Confounds

**Ruled out within the open pair.** Copy, rank-1 and half-known share E54's windows, marker and 24 keys; only the value table differs (`experiments/open_pair/amend1.py:134-140`). Across the four maps on seeds 3-7:

| map | steps | ‖δW‖_F | G (nats) | MLP share of ‖δW‖² | spread (participation ratio, of 28) | MLP share of the sum | sum |
|---|---|---|---|---|---|---|---|
| copy-24 | 10 | 0.67-0.68 | 12.9-13.2 | 0.68 | 8.3 | 0.27 of 0.34 | 0.32-0.38 |
| half-known | 10 | 0.68-0.69 | 11.0-11.3 | 0.69 | 8.1 | 0.44 of 0.53 | 0.49-0.56 |
| rank-1 | 10 | 0.68-0.70 | 10.2-11.0 | 0.68 | 8.1-8.2 | 0.62 of 0.75 | 0.69-0.81 |
| cipher-24 | 8-9 | 0.67-0.71 | 11.5-12.9 | 0.70 | 7.7 | 0.57 of 0.70 | 0.67-0.74 |

(Sources: `open_pair_instruments/results/seeds.json`, `fed_or_written/results/amend1.json` `.states[].own`, `part_norms.out`.) Size, steps, keys and gain are matched within about 20%, and copy has the largest gain, not the smallest. **It is not "attention versus MLP"**: on every map the four MLPs carry about 80% of the sum, and copy's heads and MLPs both sum about half of the cipher's. **It is not update spread**: copy is a little more spread than the cipher (8.3 against 7.7), but across the whole table spread overlaps between readings (E62 lookup 8.4-8.5 reads written; emoticon 8.4-8.7 and copy-24 8.3 read fed; lowercase 8.8 reads between).

**Not ruled out: "the answer is the key token" versus "fed".** Every row that reads fed is a copy task (E62 copy, copy-24, ident32, E99 forbidden copy), plus the beside-only emoticon. The four maps differ only in the table, and their order by sum is copy 0.33 < half-known 0.52 < cipher 0.70 ≈ rank-1 0.75. Rank-1's value for each key is W₀'s own most probable next token there (`open_pair/amend1.py:138`); W₀'s geometric-mean probability of it at the trained keys is 0.0057, against 0.00014 for copy's (`open_pair/results/amend1.json` `.c0`). If the sum read how much the gain reuses what W₀ already does, rank-1 should sit at or below half-known; it sits above the cipher. **E100 filed rank-1's lean as "fed"** (`fed_or_written/results/predictions.json` `.expected.leans`), and Amendment 1 moved it to "assumed or open", leaning written (`reports/plans/2026-11-05_plan_e100-amend1.md:23,35`), after seeing it read 0.72. That move is the one place the grouping was fitted to the number. What changes: the answer paragraph may say the sum tells copy maps from key-table maps; it may not say it tells fed from written without a non-copy task, filed as fed, that reads fed.

**The poles are weaker than the table suggests.**
- E99 forced copy (0.97-0.98) moved only the copying pair's q, k, v rows: its update lies in two heads of the same block (participation ratio 1.96, `part_norms.out`). Two parallel heads add into the residual stream, so their reverts sum to about 1 almost whatever the update does. It does not test the sum; drop it from the evidence that copy tasks can read written.
- E62 lookup sets the written bar (0.6217, `predictions_amend1.json` `.thresholds.written_bar_source`) but had no lean in E100 (`predictions.json` `.expected.leans["T5 lookup"]` is null) and is filed "new by construction" only in Amendment 1's table. Its label is reasonable; note that it was assigned after the reading.
- The written-by-construction plants (E100 stage 0) each sit in one MLP, so their sum is 1 by definition. No state that is written by construction **and** spread over several parts has been read; that is the control below.

## 3. How strong is the held-out test

Seeds 3-7 differ from seeds 0-2 only in batch order: same windows, keys, table, recipe, scored step and site. On seeds 0-2 copy-24 read 0.30-0.40 and the cipher 0.68-0.72, a gap of about eight copy standard deviations, so 9 of 10 was very likely before reading; the test checks that seeds 0-2's spread was typical (it was: copy 0.32-0.38, all under the bar set at copy's largest seed). It is a seed-stability check, not a test of the reading: 10/10 adds nothing about whether the sum tracks fed versus written on a new task. The task-level evidence is four copy tasks against four key-table or new-text tasks, chosen after the sum was seen.

## 4. The cheapest discriminating control (needs a GPU; written, not submitted)

**A written update spread over four serial parts, on the copy site.** E97's per-key plant (`locator.plant`) built in all four MLPs' `down_proj` at once, one common scale matched to each site's seed-0 fine-tune gain, read on the copy-24, cipher-24 and rank-1 sites. It is written by construction and spread over four blocks in series, which no row of the table is.
- If the sum reads how the update itself is built, it reads written (≥ 0.62, near 1) on all three sites, with each MLP near 1/4.
- If it reads at or under 0.40 on the copy site, the copy site itself, where the answer is already in the input, pulls the sum down whatever was trained, and the copy maps' low readings are not evidence about their fine-tunes.
- Control first: copy-24 seed 0 re-read against E100's 0.39511 within 1e-3, exit 1 on a miss.

What it cannot settle: whether a non-copy gain that reuses existing machinery reads fed. No saved state is labelled that way by construction outside copying (rank-1 was meant to be, and reads written; E99's forbidden lowercase has a gain of 0.53-0.57 nats, under the 1-nat floor). That needs a new task and its own plan.

Script: `~/commute-data/scratch/mechanism-detection-lead/e100-check/spread_plant.py`, sbatch beside it, `e100_check.sbatch` (debug, c11, compute, one GPU, 20 min, submitted from the clone root, sources `cluster/env.sh`). Lint clean (ruff, pyright via `cpu-run`). Not submitted.

## Not run

- The spread-plant control above (handed to a pod-runner).
- A non-copy task filed as fed by construction.
- Per-part norm shares of the ladder rungs and lowercase's E99 states; the "gap" unit (Amendment 1 reads "lo" only).
- Any check of whether rank-1's fine-tune in fact writes its answer directly (that would rescue the reading for rank-1 but not the claim, since the label would then be set by the number).

