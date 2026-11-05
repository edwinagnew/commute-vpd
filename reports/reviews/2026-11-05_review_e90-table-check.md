# Independent check of E90's old-labels table and answer paragraph

**Verdict: holds with caveat.** I found no bug in the measurement code, and every number in the table matches the result files, while one sentence in the body misstates a number without changing a verdict. The answer's three claims survive, but "reads as assumed" re-reads E62's own measurement, "reuses" on the new-by-construction rows depends on a random band not matched in variance and on block 1, and the body's "the shares follow the labels" does not survive subtracting each state's own write.

*2026-11-05. Report checked: `reports/analyses/2026-11-05_analysis_e90-old-labels.md` (cited below as "report"). My script, CPU only with no model forward: `~/commute-data/scratch/baseline-lead/e90-check/check.py`, output `check.out` beside it. Line numbers are at commit 3c74ed9a.*

## The three questions

1. **Only the copying pair reads as assumed through a located mechanism: true, but by construction.** Row 1 is the only row read under a located mechanism (`experiments/new_vs_reuse/stage2.py:93-121`), and the plan said so before the run (`reports/plans/2026-11-05_plan_e90-new-vs-reuse-definition.md:137-146`). See findings 1 and 7.
2. **Direction removal does not tell new from old: true of the verdict, at blocks 1 and 2, and not a coding bug.** The removal, share, band and write are coded as the report describes. Rank-1's "not specific" and the graded order do come from the answers' overlap with the basis, and the "reuses" reading on the new rows depends on the band. See findings 2, 3 and 4.
3. **The first verdict column represents `stage2.json` fairly as a table, but not consistently with the answer.** It changes three rows from `.table[].verdict` (lookup, cipher, rand32), says so, and prints the file's read beside each. The filing does say a verdict on a not-admissible pair is the read's (`results/predictions_amend1.json` `.table.note`), and the report says the column is not the filing. The answer paragraph then argues from the second column. See finding 5.

## Real findings

**1. "Reads as assumed" and "validates" re-read the measurement the label came from.** The row's assumed label is "E62, causal" (`stage2.json` `.table[0].assumed`). Stage 2 reads the same states, windows and head, and its shares equal E62's saved ones to seven digits: 0.56269886 in `stage2.json` `.row1.shares.copy`, 0.56269888 in `stage0.json` `.R5.copy.shares`. New in E90 are the 20-draw band and W₀'s control, plus the verdict rule without E62's 0.5 bar. Report lines 3, 13 and 155 ("validates one old label") read as agreement between two tests. They should say the new rule keeps E62's reading; line 73 already says it stands beside the earlier verdict.

**2. Rank-1's "not specific" is answer overlap, now measured.** Report lines 81, 97 and 219 say the overlap was not measured. I measured each write's output side (`X_*_s0.pt`) against the saved guess basis, as the share of squared norm inside the basis. The basis was built from rank-1's own answers: 22 of its 24 keys' answers are fitted values, against 4 of 24 for the half-known map and 0 of 24 for the cipher.

| Write | Overlap, block 1 | Overlap, block 2 | Write share `A_lo_raw`, block 1 | Write share `A_lo`, block 1 |
|---|---|---|---|---|
| rank-1 | 0.116 | 0.226 | 0.122 | 0.047 |
| half-known | 0.046 | 0.072 | 0.036 | -0.034 to -0.042 |
| cipher | 0.019 | 0.023 | -0.057 | -0.033 |
| chance (rank 15 of 768) | 0.020 | 0.020 | | |

So the report's guess is right, and rank-1's verdict follows from the choice of C's values. The plan's lean had the direction right (`reports/plans/...definition.md:125`), at far smaller size than its 0.5. This overlap reads the unembedding rows at the residual and ignores later blocks and the final layer norm, so it is a comparison, not a prediction of the share.

**3. The exploratory order rank-1 > half-known > cipher does not survive subtracting the write.** Report line 103 answers "At block 1 yes" and lists four weakenings. A fifth is stronger: the order is also the order of overlap in finding 2, so it may be the removal hitting the answer direction. Subtracting each seed's own write share (`stage2.json` `.states.*_row`, `state` minus `control`):

| Row | Share | Share minus own write |
|---|---|---|
| rank-1 | 0.159-0.218 | 0.112-0.171 |
| half-known | 0.132-0.139 | 0.166-0.180 |
| cipher | 0.079-0.124 | 0.112-0.157 |
| copy, 24 keys | 0.661-0.680 | 0.786-0.805 |
| copy, 32 keys | 0.499-0.530 | 0.542-0.573 |
| random map | 0.359-0.370 | 0.352-0.364 |

Rank-1 and the cipher overlap, and the half-known map moves to the top. The copy-over-random gap remains, and on the 32-key pair the writes' overlap is near chance (0.053 and 0.042 against 0.040). This subtraction is post hoc too, so it is a reason not to lean on the order, not a result.

**4. "Tasks built to be new read reuses" depends on the band and the block.**

- *Band.* The random bases are matched in rank, not variance (`apparatus.py:181-182`). The guess basis holds 0.200 of the fit residual's variance against 0.019 for random, and token identity 0.643 against 0.040 (`stage1.json` `.control`). Report line 138 flags this for W₀'s control on token identity only. The guess basis's 0.20 appears nowhere in the report, and line 23's band, on which every state's "above" rests, carries no such note. `.claude/CLAUDE.md` (Rules) asks a random-direction control to state the variance share it is matched on. The bands here run about 0.015-0.03 at 2% variance and 0.03-0.05 at 4%. Scaled in proportion, which is my extrapolation from two points and not a measurement, a matched band could reach 0.1 or more on the guess rows and well above 0.5 on identity, and most "reuses C" readings would become "does not reach". "Does not tell new from old" would still hold, but for a different reason.
- *Block.* The report prints block 2's shares but no verdicts. Applying `rules.verdict` to the saved block 2 numbers (`.states.*["2"]`), with the band taken as the code takes it (0.0299 guess, 0.0613 identity, 0.0385 copy-24, all with teeth):

| Row | Block 1 | Block 2, re-derived, unfiled |
|---|---|---|
| rank-1 | not specific | not specific |
| half-known | reuses C | does not reach |
| cipher | reuses C | does not reach |
| copy, 32 keys | reuses C | does not reach |
| copy, 24 keys | reuses C | mixed (reuses, does not reach, reuses) |
| random map | reuses C | reuses C |

Four of six rows change. At block 2 the random map alone reads "reuses" and the 32-key copy map does not, so the claim that the verdict does not separate holds there as well, more strongly. But the cipher's "reuses" is a block 1 reading only, resting on a possession margin of 0.027 (report line 151). The answer should carry that.

**5. The answer argues from one column and the table leads with the other.** Report lines 3 and 13 say tasks built to be new "read reuses", and the first column prints "no C listed" for the cipher and random map (lines 43, 47). The pair counts as read only through the file's "not through C", while the first column prints lookup as "no C listed" (line 41). Line 134 says the random map "reads the same" as the copy rows, which is true of the second column only. Read by the first column, the definition does separate these rows: copy maps "reuses C", cipher and random map "no C listed". Line 155 says admissibility does that work, but the answer paragraph does not. Line 3's "left every strengthening task without a reading" also conflicts with line 13's "apart from the copy tasks" and with line 45, where ident32, labelled "strengthens", reads "reuses C". Suggested reading: name the removal alone in the claim ("removal alone, ignoring admissibility, reads...").

**6. The token-identity rows exist only because the control was replaced after it failed.** Stage 1 read "unreadable: no teeth" on R2 (`stage1.json` `.readings.R2`). The plan's stated branch for that was to refit the basis on the task's windows (`reports/plans/...definition.md:200-202`), and its top section says a row the definition cannot read is a gap, not patched (line 11). Amendment 1 kept the basis and changed the read (`amend1.py:40-53`). The filing argues this openly and the report calls the control weak (line 138). The report does not say that the plan's branch was not taken, or that without the replacement ident32, copy-24 and the random map all read "no teeth", which removes the random-map half of the answer's "reads reuses". The cipher's half stands, with the guess control's own teeth: 2.85 against 0.062.

## Minor findings

**7. Report line 73, "Random single heads reach 0.210 at the largest", differs from the files.** 0.210 is the band, a 95th percentile (`removal.py` `band_of`). The largest draws read 0.287, 0.263 and 0.229 on copy (`stage2.json` `.states.copy_s*["2"].random`, index 18). That draw is head (1,1), not an unrelated head: `TRIED.md:136-137` and E62 Amendment 2 name it the circuit's other head. The same draw costs W₀ 8.22 against (2,4)'s 8.85 (`.control.random[18]`), which line 73 omits when it says the removal "has force" against a band of 4.21. The band is itself an interpolation between 4.00 and 8.22. No verdict changes, as copy's lowest share 0.471 is above 0.287 and lookup's is 0.063 under that draw. Also, the 20 draws hold 17 distinct heads of 23, because draws can repeat (`copying.py:211`).

**8. The half-known row's admissibility is a coin flip, measured.** The report flags the one-key margin (line 89). On 1,000 random half splits of the keys it reads admissible on 50.4%, and the opposite parity split reads not admissible. The cipher reads admissible on none. So the first column's "reuses C" on this row depends on the split, while the "no C listed" rows under W₀'s guess do not.

**9. The band is two-sided and the test one-sided.** `band_of` takes absolute values and `rules.py:90` compares signed. Copy-24's write at -0.125 (3.3 bands) and the ladder push at -0.049 (5.6 bands) count as "within". This is as filed and harmless for "spared", but "the removal does not reach them" (report lines 63, 132) is loose. At block 1 the removal costs W₀ 0.335 and the state 0.302 log-odds, and raises the push by 0.162. "Does not reach" here means no loss beyond W₀'s.

**10. W₀'s subtraction changes sign between the two copy tasks.** W₀'s drop is +0.955 on copy-24's windows and -0.860 on ident32's, under the same kind of removal. In the unsubtracted column the two differ by 0.26-0.32, not line 118's 0.13-0.18, though the table beside it does print both. I did not find why.

**11. Report line 95's "somewhere after block 1" is an inference.** Per-block squared norm of δW is nearly the same on all rows (0.23, 0.23, 0.26, 0.28), so update size per block does not explain the shares. Line 167 already says the amounts are unexplained.

**12. Basis and table differ slightly.** The guess basis comes from the 32-key rank-1 values (`stage1.py:116`). It holds value 407, which no 24-key answer uses, and misses 396 and 8938, which two 24-key answers use. Line 148 gives the counts, not this consequence.

## What checked out

- **Code.** The removal sits at block 1's output at the answer position (`apparatus.py:123-131`, `readout.py:37`), and the write sits upstream of it in `h.1.mlp.down_proj` (`perturb.py:20`), so line 93 is right. `A_lo` matches line 22 (`readout.py:190-204`). The band is the largest over states and writes at that form (`stage2.py:151`). Rank-1, cipher, half-known and copy-24 share one key set, as do ident32 and rand32, each with its own write on its own windows.
- **Numbers.** Every share, control and band in the table, the second table's six rows and five columns, the margins in lines 103 and 118, the scorecard's reads, Amendment 1's control numbers, and I1, I2, I3 and R7, against `stage0.json`, `stage1.json`, `amend1.json`, `stage2.json` and log 52317.
- **Old findings, spot-checked in `TRIED.md`.** E55 C3 (0.462, 0.393, 13.3, 10.1), E58 Amendment 1 carrying, Amendment 4 (0.475), Amendment 6 (0.373-0.382), E53 (0.92, 0.26), E54 (0.77, 0.24).

## Smallest check that would settle finding 4

Re-read the states with random removals matched to the basis's variance, for example random subspaces of the top principal components. It needs no training and should fit a `debug` job. I did not run it, as this brief excludes GPU jobs.

## Not checked

- Noise on rank-1's 0.020 margin: per-window reads are not saved, and each write is read on 75 windows.
- A variance-matched band on any state.
- `input_dep.py` line by line. I1-I3 were checked by their numbers only.
- `plot.py`, the four figures and `notebooks/E90_new_vs_reuse.ipynb`.
- The old-finding column beyond the spot checks, and the deck's figures.
- `tests/test_new_vs_reuse.py` was not run. R4's inputs from E70 were not re-read.
- Copy-24's basis overlap: that basis was not saved.
- The Qwen page.
