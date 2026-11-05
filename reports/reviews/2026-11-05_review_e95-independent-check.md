# E95 independent check: does the key reader's loss hold up?

Verdict: **holds with caveat.** The headline numbers come back from the saved files, the key reader and the fine-tuned states are scored on the same windows by the same formula, and the reader loses more than the first table's copy states in every unit computed, so its loss is not overstated. The caveat is that the reader was built only for that first table pair: for the 24-key copy map and the replicate copy map, "needs the input token" is carried over from the first table rather than measured, though neither row earns a label either way, since the rule asks a state to show more than a token reader loses and nothing showed that.

*2026-11-05, independent checker (wrote none of E95). Files read: `reports/analyses/2026-11-05_analysis_e95-token-identity.md`, `reports/plans/2026-11-05_plan_e95-token-identity.md`, `experiments/token_identity/{run.py,read_rules.py,post_hoc.py}`, `results/{run.json,post_hoc.json}`, `results/large/reads_*.pt`, `src/commute/readout.py` (`shares`), `src/commute/perturb.py` (`key_write`), `experiments/new_vs_reuse/apparatus.py` (`hand_write`, `table_inputs`), `decisions.md` rows 115-117 and 121, the E90 old-labels analysis and the definition-batch decision. Arithmetic only, on saved files; script `~/commute-data/scratch/definition-lead/e95-check/recompute.py`.*

## What was checked

**The share is the same quantity for reader and state.** In `run.py`, `null_stage` and `basis_stage` read the original model, both seeds' states and each state's matched write on one set of windows (`AP.kv_site` of the table), under one basis and the same 20 draws, and pass all of them through `readout.shares`. The scored share `A_lo` is the item's mean log-odds drop under the removal, minus the original model's drop on the same windows, over the item's own log-odds gain. So windows, the original-model term and the denominator are common to reader and state.

**Numbers reproduce.** Recomputed from the per-window reads in `results/large/reads_*.pt`, independently of `readout.shares`:

| Row, first pair | Gain, log-odds | Share | Largest draw |
|---|---|---|---|
| copy, seed 0 | 14.00 | 0.4992 | 0.3999 |
| copy, seed 1 | 14.07 | 0.5300 | 0.4179 |
| reader on copy table | 14.08 | 0.7413 | 0.6227 |
| random, seed 0 | 13.73 | 0.3705 | 0.3391 |
| random, seed 1 | 13.16 | 0.3591 | 0.3505 |
| reader on random table, seeds 0, 1 | 13.92, 13.11 | 0.7654, 0.7660 | 0.6750, 0.6753 |

All match `run.json` to four places. By hand from its stored fields: copy seed 0 is (6.128 + 0.860) / 14.00 = 0.499, the reader (9.580 + 0.860) / 14.08 = 0.741.

**The reader is matched and working.** Its gain sits within 0.6% of the copy states' and within 1.5% of the random states' (`writes`, status "ok"). With its answers rotated to other keys its gain is -4.97, -1.24 and -0.78 (`write_check`), so it learned the table rather than raising every answer. The removal leaves the input of the block-1 write unchanged (0) and changes the block-2 reader's input by 0.892, with the key earlier in 0 of 97 windows (`placement`). So the reader sits where the plan says.

**Sign and normalisation do not drive the ordering.** The removal raises the original model's log-odds (drop -0.86 on table A), so subtracting it adds about 0.06 to every share alike. The ordering holds in every unit:

| Unit | Reader on copy table | Copy states | Reader on random table | Random states |
|---|---|---|---|---|
| Log-odds share, scored | 0.741 | 0.499, 0.530 | 0.765, 0.766 | 0.370, 0.359 |
| Log-odds, original model not subtracted | 0.680 | 0.438, 0.469 | 0.698, 0.695 | 0.302, 0.288 |
| Nats share | 0.566 | 0.262, 0.291 | 0.656, 0.662 | 0.221, 0.193 |
| Median drop over median gain | 0.791 | 0.531, 0.575 | 0.781, 0.770 | 0.337, 0.343 |

The reader answers less sharply on the copy table (geometric-mean probability 0.71 against 0.96), but the scored unit is log-odds at matched gain, and the nats row, where sharpness would matter, widens the gap.

**A stronger point than the report makes.** The reader's own matched draws already cost it 0.62 (largest 0.623, band 0.618). That is more than the real basis costs either copy state. So for this reader, removals of the same variance inside the top directions do more harm than token identity's removal does to copy.

**Read 2 reproduces.** The filed band is 0.4265, the larger copy seed's own band, which tops every state and block-1 write band on table B. Copy reads 0.511 and 0.512, above; random 0.267 and 0.284, within. Copy minus random clears the largest draw on all four seed pairs, by 0.022 to 0.033. The original model's control on table B's keys reads 2.858 against a band of 2.842, a margin of 0.016, with one draw (2.871) over it, and a redraw keeps it about half the time (`post_hoc.json`, 0.478). "A difference of degree, control barely passed" is a fair summary.

## Caveats

1. **Two of the three copy rows have no reader on their own table.** The block-2 reader exists only for the first 32-key pair. The 24-key copy map was read in E90 on a different key set, a basis fitted on its own keys and 75 windows. The replicate copy map was read with block-1 writes only, which the removal cannot reach. The report's "more than any fine-tuned state" with ".66-.68" therefore compares across tables, with a margin of 0.06 to 0.10 and no measured spread for the reader on that table. The report itself lists "the 24-key copy map under a block-2 write" as not run. Under the adopted rule this changes no label, because a label needs a state to show more than a token reader loses on its own table, and that was not shown. But the reading should say that for the 24-key and replicate copy maps "needs the input token" is carried over from the 32-key tables.

2. **The reader is close to the most exposed reader possible.** `perturb.key_write` is the minimum-norm map from the 32 keys' mean activations to their answers, so it reads exactly where the keys' means differ, which is what the basis was fitted to. That makes it a sound proof that a reader needing only the token can lose 0.74 to 0.77. It does not make 0.74 a bar every token reader reaches: a reader built another way, or one with part of its work upstream, as every state has (0.43 to 0.47 of each update's squared norm is in blocks 0 and 1), could lose less. For the headline, "not shown to mean more than needing the token", a proof that the number is reachable is enough, and that holds.

3. **The reader's comparison is not the filed outcome.** Read 1's filed outcome is "no call" (copy inside the draws, random at most equal), and its filed sentence reads backwards against the signs, as the report says. The headline's "reader loses more than any state" comes from the share rows, not from a filed rule. I checked `seed_row` and `read1` in `read_rules.py`: the "no call" follows from the first-match table as filed, and nothing in it would have scored a copy state as losing more than the reader. All four state-minus-reader gaps are negative (-0.21 to -0.41).

## What would have changed the reading, and did not

- The original model's term entering reader and state with different signs or windows: same windows, same term.
- Gains normalised differently: each item over its own gain, matched within 1.5%.
- The reader failing its table: its rotated floors are far below its gain.
- The reader not reached by the removal, or reached another way: placement 0 and 0.892, key earlier in 0 of 97.
- A band computed off the filed rule: read 2's band is the filed largest over states and writes, `removal.band_of` (95th percentile of absolute values); read 1's limits are the 5th and 95th percentiles of the draws' gaps, as filed.
- A few windows carrying the loss: medians give the same ordering.

## Not run

- A block-2 reader on the 24-key copy table or on table B, which is what would turn caveat 1 into a measurement. It needs a GPU.
- A second kind of token reader (another block, an attention reader, or one trained on the task), which would test caveat 2.
- Any resampling of the 97 windows, or a spread for the reader's share across tables.
- Any check of E90's own numbers beyond what E95 reproduced (E95's job stops if table A's shares and bands move by 0.001 or more, and it did not stop).
