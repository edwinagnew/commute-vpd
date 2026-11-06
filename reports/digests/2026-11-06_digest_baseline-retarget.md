# Baseline, day 2 of 13: rows 1, 2 and 3 moved

**Answer.** On the 67M, VPD adds something over neurons on two readings, and nowhere else.
- Its switched-on set repeats across training seeds more for copy maps than for random maps. The same number of
  neurons does not.
- Its re-switching share, taken over the whole model, separates both families: copy from random maps, and lowercase
  from the ciphers. The neuron version gets the maps the wrong way round.

On Qwen, update size and gain already sort every pass and fail of VPD's first table (E93), so that table adds little.
VPD separates RL from its null nowhere. The one place VPD's columns agree with each other and no cheap column does is
the known-facts fine-tune at the end of training, read on different rows per arm. An independent check found that all
five claims hold, with caveats; its nine fixes are applied.

## What ran (all $0, cluster `debug`)

| Row | Experiment | Found |
|---|---|---|
| 1, 67M | E117 (job 54427), VPD's switched-on set and re-switching on neurons at VPD's count | seed agreement: VPD separates (gap 0.151 against a within-class range of 0.105), neurons do not (0.053 against 0.212); restore share and set-carried re-switching: neither separates; lowercase and ciphers: VPD switches nothing on, so the set panels are unreadable |
| 1, 67M | E117 follow-up read (job 54503) | the lowercase change is averaged away over windows, not spread thin: ~42 parts pass 0.3 on a tenth of windows, none on half; the ciphers have 2-5 |
| 2, Qwen | E118 (jobs 54382, 54383) | update size, rank, steering and carrying set filled on every saved state with a VPD row: 180 cells, 32 do not apply; controls exact |
| 3 | one table | `experiments/neuron_vpd_table/results/full_table.md`, published in the log as "Neurons and cheap comparators beside VPD" (flush running) |

## Files

- Report: `reports/analyses/2026-11-06_analysis_e117-e118-neurons-beside-vpd.md`
- Independent check: `reports/reviews/2026-11-06_review_e117-e118-independent-check.md`
- Notebooks: `notebooks/E117_neuron_vpd_count.ipynb`, `notebooks/E118_qwen_comparator_table.ipynb`

The retarget also closed asks 25 and 26 ("retargeted, not taken") and dropped E115 and E109 Amendment 1
(`decisions.md`, last row).

## Open

A rule that joins the table into a single "does VPD add anything" verdict would be a definition-level choice, and
that is yours. A switched-on set built per window, which could read lowercase, would be a new reading, so it was not
run. Nothing is running now.
