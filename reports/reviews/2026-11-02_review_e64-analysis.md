# Review: E64's analysis, against the saved values (2026-11-02)

Scope: `reports/analyses/2026-11-02_analysis_e64-readout64.md` ("the report"), `notebooks/E64_readout64.ipynb`, `experiments/readout64/`, the E64 section of `src/commute/readout.py`, the plan's option 1 and amendments, the code review, log 48611. The reviewer wrote none of it. No model was loaded. The report, notebook and results are unedited.

**Verdict.** The filed verdict is confirmed: P1 pass, P2 fail, P3 pass, fit missed, branch "unreadable". Every number I traced is right except one gap (F2). The problems are in the readings around the verdict. The count table is stronger evidence against block 0 than the report says, and its "meets the bar" is not sound. The removal share has not been shown able to read high, so a low share means little at any block. One of the report's counts against the prior tilt uses the wrong variable. The Belrose entry should go.

## Findings, by severity

### F1 (high). Block 0's holding follows pair count almost exactly, and the cipher clause cannot fail at it (question 1)
Report :48, :104-109, :150, :197.
- **Same rows?** Yes. `bigram_check.py:56` makes the run's own call with the same rows, seed and cap. Rebuilt here: 57,345 windows, per-value counts equal to run.json, 459 elicit pairs, 0 cipher, 21 keys held. The check at `:58-59` compares counts, not window identity, so identity rests on the seeded call being deterministic.
- **What the report missed.** The table holds a key exactly when its pair occurs at least twice in the sample (lowest count held 2, highest not held 1). Of the 20 fit keys with a count of 2 or more, block 0 holds 20. Of the 10 fit keys with a count of 0 or 1, it holds 4. Overlap with the table falls from block 0 (20, chance 15.8, p 0.001) to W0's output (15, chance 13.1, p 0.145).
- **Small base.** Only 810 of the 57,345 windows have a key token at the read position, and 459 of those are pairs. Softmax regression on nearly separate token directions behaves like a conditional count table, so a count of 2 can be enough. That mechanism is my hypothesis, not a test.
- **The cipher clause has no power at block 0.** Every cipher pair is absent from all fit rows, so a readout that learns the lookup cannot hold a cipher key whatever W0 does. "Cipher at no block" at block 0 restates the sample.
- **It also moved the causal check.** `l_star` takes the earliest block, so the removal ran where the lookup is strongest. The report says this at :126. It belongs in the verdict's neighbourhood.
- **What it does to "possessed".** Block 0's pass is not evidence that W0 computes the map. It is evidence that the residual there carries the key's identity and that the sample pairs the two. Block 1 (19 overlap) and block 2 (18) are not clear of this either. A correlation cannot show the readout used those windows, and the elicit values were chosen from W0's guesses, so Pile pairs and W0's guesses share a cause. The report's "cannot separate" (:109) is right.
- **Fix.** At :109 and :150, state the count-of-2 result and say P1's cipher clause had no power at block 0. Add a note to the P1 row at :197: pass as filed, not informative at block 0.

### F2 (high). A low share at any block means little, whatever the fit bar says (question 2)
Report :33, :83, :127, :179; `plot.py:254`.
- **Not a route difference.** The twelve states change only the 24 attention and MLP matrices (`elicit_s0.pt` keys, `steering.py:47`). Head and final norm are W0's. So at the last block every log-odds gain passes through the residual at that position, the one removed from. A share short of 1 there cannot be the gain taking another route. What remains is the row not being the head's direction, the component shared by all rows, and the final norm's rescaling.
- **False premise.** The plan's "at the last block the readout is the head restricted" (plan :87) is what made "near 1" expected. The row predicts Pile's next token under a capped prior. Nothing makes it the head's row.
- **What it implies at earlier blocks.**
  1. P2 needs A able to read high, and that control read 0.31-0.53. So A is the gain's share times an unknown, kind-dependent efficacy of the row.
  2. Elicit and cipher already differ by 0.18 at the last block, where a route difference is impossible. So a gap of that size can come from row geometry, against a bar of 0.2. A gap at an earlier block near that size cannot be read as a route difference.
  3. Dividing an earlier share by the last-block share is not valid. The two kinds differ, and block 0 is not bounded by block 3, since an early edit can be amplified later.
  4. So an earlier-block removal reads one way only. A high share above a wrong-row reference on the states would be evidence of use. A low or negative share is no evidence of absence.
- **Consequences.** The first-filed fit bar's branch ("no causal footing", :33) would have over-read this run. The report should say so, or the fit bar looks like the only thing between this run and that conclusion. "If a later block's share differs from block 0's, block 0 was the wrong place to test" (:179) does not follow, since row efficacy differs by block too.
- **Over-reach.** "Can one row take the gain away even at the output? Not in this apparatus" (:79, :83, `plot.py:254`, notebook :281) generalises from the readout's row to any row. The head's own row was never tried.
- **Wrong number.** "The gap at the last block is 0.14" (:83) is lowest elicit minus highest cipher. By P2's rule, the gap at :63 and :198, it is 0.176 (spread 0.036). Both are under 0.2.
- **Fix.** Reword :79 and :83 to "this readout's row". Replace 0.14. Add the implication above after :83 and to the "What we learned" paragraph. Add to Not run: the head's own row at the last block, which would tell a metric fault from a row fault at $0.

### F3 (high). One count against the tilt uses the wrong variable and leaves out the cipher arm
Report :91, :96; notebook section "Is block 0's holding the prior tilt?".
- **Defect.** The 0.25 at :96 correlates block 0's rank with windows before the cap. The readout never saw those: 19 of the 32 elicit values train at exactly 2,000.
- **With the right variable.** At the capped count it reads 0.10. Over the 13 elicit values whose training count actually varies it reads -0.29, the sign a tilt predicts. So "its sign is the wrong way for a tilt" does not stand.
- **Left out.** On the cipher arm, where counts are uncapped, block 0's rank correlates -0.53 with window count (block 3: -0.58). The tilt is visible there, though it never reaches the bar: the best cipher rank is 10.
- **The first count is not independent either.** Among the six same-count elicit values at :95, the five at rank 1 have pair counts 155, 25, 14, 15, 15. The one at rank 39 has 0. That split is the lookup, not evidence about the tilt alone.
- **Pair-poor keys.** Of the 4 held keys with a count of 0 or 1, three have values at the cap (2515, 4367, 4447). The tilt and the lookup may work together, and the report weighs them one at a time.
- **Fix.** Replace :96 with the capped and cipher numbers. Soften :91 and :109 to "the saved data do not settle the tilt". The job below does.

### F4 (medium). The table's "meets the bar" is the smoothing constant's arithmetic
Report :105; notebook :371; `bigram_check.py:34`.
- **Same rule in form only.** The rank formula is the same and ties share a rank as filed, but a readout's logits never tie and counts tie at 0 all the time. The rank clause passes 31 of 32 elicit keys and, with no pair at all, 8 of 32 cipher keys (29 on the capped sample). Only the specificity clause decides, and that clause depends on the pseudocount, which has no source.
- **The sweep.** Of W0's 20, the table holds 17 at pseudocount 0.1, 16 at 0.5, 15 at 1 and 11 at 5, and one fewer in each case with key 1495 dropped. The cipher holds 0 at all four. Worst-case tie handling changes nothing.
- **Comparability.** The readout's own rule drops the unfit key, so the like-for-like figure is 14, under the bar. "On all the fit rows it holds 17" compares with a sample the readout never saw.
- **Fix.** Say "21 of 32 elicit pairs occur at least twice, and no cipher pair occurs at all". Drop "meets the bar" and give the sweep if the figure stays.

### F5 (medium). W0's control does not show the row is specific to the value
Report :77, :121, :158; notebook :239 ("specific to the value") and :30 ("rules out a row that hits everything").
- **Defect.** Each value has one key, so the wrong row is another key's row. A row that detects the key token passes this control just as a value direction would. At block 0, by F1, that is a live reading. Removing it then strips the key, and W0's answer falls for that reason.
- **Second defect.** The control reads a signed median. A wrong row's 0.00005 is smaller than the random direction's 0.0019, and a median near 0 can hide moves in both directions. Per-window drops were not saved.
- **Fix.** Write "specific to the row, which is tied one-to-one to the key; value and key are not separated". Say the same in the notebook. $h \cdot u$ on the key's windows against others', already on the Not run list, bears on it.

### F6 (medium). "A real effect of this row" and "sign reversed" claim more than was measured
Report :63, :117, :166.
- **Wrong floor.** A random unit vector in 768 dimensions projects weakly on the residual, so it is no floor for a learned row. The right reference is a wrong row on the states, which was not run and is not on the Not run list.
- **Not five independent readings.** All five shares share one readout, one set of 48 windows and one W0 term, and there is no standard error on W0's mean.
- **Where the sign comes from.** The raw mean gap is +0.119, the predicted side. The W0 term moves it by -0.164, and the rest is -0.045. The cipher's positive share is entirely W0's negative drop, since its raw share is negative. :119 says part of this, while :166 headlines the reversal as unexplained. At the last block the W0 term is 0.431, of a raw gap of 0.607. Subtracting W0 is still right for the question asked: the state must depend on the row beyond W0.
- **Fix.** At :117 write "outside the random direction's range; no wrong-row reference on the states". At :166 give the gap's decomposition. Add the wrong row to Not run.

### F7 (medium). The report's $0 test has no stated power
Report :111, :179; notebook :406.
- **Defect.** "If not, it is out" needs a control, since capped rows are over-fitted (held-out 2.60 at block 0). A fall in the cipher's median rank from 45 is expected under every answer, because removing the tilt moves an uninformative value towards mid-rank. Read naively, that looks like the tilt returning.
- **Fix.** Use the readings in "The job" below.

### F8 (low, asked). The Belrose entry is correct in fact and not needed
Report :191; notebook :412.
- **Grep.** The plan cites Jain et al. 2024 only (plan :4, :10, :32, :45, :61, :89, :141), as do `predictions.json`, `run.py` and `readout.py`. Belrose appears in the 2026-10-31 plan at :352 ("is not run"), in `lit_review/` (catalog, top 10, one summary), and in this report and notebook. The entry's "plan, filing and code do not cite it" and "line 352" are correct.
- **Why remove it.** The rule is departures from the source cited and used, and E64 neither cites nor uses this paper. The entry turns a slip in a brief into a departure, cites a brief no reader has, and describes a paper "as recalled and not re-read".
- **Fix.** Delete both bullets. If the contrast is wanted, add one line to Not run: "a tuned lens".

### F9 (low). Smaller wording points
- **:100, :158.** "A plainer account the run did not anticipate" and "found here, not filed". E59's analysis offered the bigram-like alternative (:95) and listed the key-only table as not run (:151), and P1's `blind_to` names "a bigram-like early block", though only for keys outside M. It was known, and filed too narrowly.
- **:107.** "It is not only memorised pairs" rests on two keys and has no reference rate. Key 4367's pair occurs 20 times in the fit rows (dropped by the cap) and its value is at the cap. Only key 3541's pair is absent from all fit rows. Nearby tokens were not checked. Write "two keys; one may be the prior".
- **Not run list (:211-224).** Add: the wrong row on the states; the head's own row at the last block; the capped readout's removal at the last block; the count table on the capped sample; a readout on the residual before block 0.

## The job: settle provisional reading 1
Cluster `debug` through `/pod-runner`, about 30 s after a 19 s model load, $0. It is new, post-hoc and reported only, and changes no verdict. Not run by me.

A script of about 30 lines, `experiments/readout64/capped_possession.py`:
1. Check `readouts_low_cap.pt`'s sha256 against `stage0.fit.low_cap.sha256`. Load it into `Probes(4, 768, 64)`.
2. Run `readout_logits64`, then `key_reading` with run.json's `unfit`, then `relative_possession` against the stored M and `cipher_possession`, per block, on all windows and the fit half.
3. Run W0's own logits with `prior_shift(vals, per_value, 241)` through the same rule, as the reference for a flat prior.
4. Write `results/capped_possession.json`. The saved states are not read: holding does not involve them.

Why it separates: the capped sample's prior is flat for 62 of the 64 values (241 windows; the other two have 147 and 26). Every cipher value has the same count as nearly every elicit value, so a difference that remains between the arms is not the prior.

Reference, computed here on that sample with no model: 226 elicit pairs, 12 pairs absent, 0 cipher pairs. The table holds 13 keys, 10 of W0's 20 (9 without key 1495), cipher 0.

| at block 0 | reads | meaning |
|---|---|---|
| power check | last block holds at least 15 of 20 | if not, the capped rows are too poor and the tilt stays open |
| cipher median rank, any answer | from 45 to about 30-35 | the tilt leaving an uninformative value; **not** evidence either way |
| tilt sufficient | at least 6 cipher keys held at some block, median rank near 10 or better | reading 1 returns: the cipher's absence was the prior |
| tilt not sufficient | at most 2 cipher keys at every block, power check met | tilt out; 3-5 keys is reported as partial |
| then, lookup | elicit about 10 of 20, tracking pair count; 2515, 4367 and 4447 expected to drop | consistent with the lookup, not proof: thinner rows also lose keys |
| then, more than lookup | elicit still at least 15 of 20 | holding outruns the pair counts, and block 0 needs another account |

The job settles the tilt only. The lookup needs a retrain, with the pair windows dropped or on the residual before block 0. Either takes seconds on cached residuals, but it is a new experiment with its own filing, as the report says at :111. A readout on the pre-block residual holding the same 17 of 20 would end the block 0 reading outright.

## What would change the conclusion
- **The verdict:** nothing here. The fit bar's amendment preceded the only job (fit margin 0.176 against 0.1), and I found no arithmetic or rule error.
- **"Block 0 is weak evidence":** a pair-dropped or pre-block readout. If block 0 keeps its keys without the pairs, or the pre-block readout fails, block 0's holding is worth more than this review says.
- **"A low share means little":** the head's own row at the last block. If it reads near 1, the metric works and the shortfall is the readout's row, so a better row could make an earlier-block removal readable. If it also falls short, the metric itself cannot read high, and P2's rule needs replacing before any further removal.

## What I confirmed
- **Recomputed with my own code (`~/commute-data/scratch/baseline-lead/e64-check/recompute.py`)**, from window ranks and stored log ratios, with keys from `windows_*.pt`:
  - W0's sets 20 and 0, fit half 17 and 0.
  - All 16 readings' medians and held sets equal the stored ones. Elicit 17 of 20 at all blocks with 7, 7, 6, 3 outside; fit half 15, 15, 15, 16 of 17; cipher 0.
  - P1 pass on blocks 0-3, P3 pass, l* = 0.
  - Each of the 36 A values from its parts. Gap -0.04538, spread 0.01458, verdict none, own-step 0.0025 and 0.0072, P2 fail. Capped gap -0.03655.
  - Fit over by 0.1756. Branch as filed, with `branch64` matching `decision_tree`'s "fit missed".
- **Traced to log or JSON:** the tables at :37-40, :52-55, :67-69 and :197-205; the raw shares at :71; :77, :115, :117 (8 to 13 times), :170, :171; the timing table at :135-144; the tilt arithmetic at :91 (ln(2000/350)), :95, :97 (5.2 nats; per-key median change 1, best 13), :98 (29 and 6); the count claims at :104-106 (1,906; 17 of 20; 20 of 24; 0.86, 0.77, 0.46, 0.52, 0.14).
- **Provenance:** `fcb049f` at 13:22:43 is `predictions.json`'s last commit; job at 13:23:13; 48611 is the only E64 job in `jobs.log`; no diff between `fcb049f` and `e0f9364` on the run's files; `60f0aa9` holds run.json.
- **Untraced:** the 1 min 02 s at :131, which the report itself says came from a brief.
- **"Blind to" entries at :158-162:** accurate as far as they go, with F1 and F2's additions.

## Not run
- The job above, or anything on a GPU; the model was not loaded and no window was forward-passed.
- The head's angle to any row; the mechanism in F1 is untested.
- A check that the rebuilt sample equals the job's window for window, beyond counts and the same seeded call.
- The figures were not inspected, except `plot.py`'s titles.
- Belrose et al. was not fetched; the entry's description of it is unverified.
