# Review: E59 analysis and notebook (Jain readout), independent check

*2026-11-02, fresh agent that neither wrote nor ran E59. Scope: `reports/analyses/2026-11-02_analysis_e59-jain-readout.md`,
`notebooks/E59_jain_readout.ipynb`, `experiments/jain_readout/{plot.py,figures/*.png,results/predictions.json}`,
`results/run.json` at 4851105, 2f167bf, 630cc63, 25ba4fe, logs of jobs 48279, 48344, 48357, 48412, the plan and the
three E59 reviews. Nothing was run on a GPU and the model was not loaded.*

## Verdict

**Sound as a closing verdict, overstated in its reading. Fix before the research log takes it.** Every number checks
against `run.json` and the logs. Every amendment was committed before the job it governs, and each is reported with
the rule as first filed. "P1 unreadable, E59 closed" follows the pre-set rule twice over: the `after_stop2` reading
rule, and the `after_stop` fallback filed before job 48357 ("held-out gap still > 0.5 after fresh data: report P1
unreadable", and `gap_last` read 0.513). But the report's main explanation, that the readout "disagreed with the
model it was meant to copy", was never measured. The construction ranked each value under the model's *mean*
distribution over windows. P3 and possession use the *per-window median* rank. The model's own per-window median
ranks on the 97 windows were never computed, and E54's own numbers suggest the model might fail P3 by that statistic.

## Findings

### Major

**M1. "The readout misses the model's own ranks" was never measured; P3 compares two different statistics.**
- Where: the opening paragraph ("missed the model's own ranks at the last block"); The answer ("disagreed with the
  model it was meant to copy"); line 75 ("where the model's averaged output puts all 32 by construction"); line 86
  ("where the model ranks every elicit value in its top 20"); line 126 ("the readout does not reproduce the model's
  own ranks"); What we learned; the possession figure's star at 32; the notebook's answer, cell 8 and What we learned.
- What: `elicit.pick_values` ranks a value under `kv.key_distributions`. That is the model's mean probability over
  all 500 windows holding the key (`experiments/keydependent_elicit/run.py:118`). P3 and possession take the rank of
  the value in each window, then the lower median over a key's ~3 held-out windows. Averaging probabilities lifts
  words that are moderately likely in every context, such as " it" after " think". Inside one window those words can
  rank well below 20. E54's `stage1.json` per-key `p_geo` (geometric mean P(value) over windows) runs from 7e-6 to
  0.027, median ~0.004, against arithmetic-mean `p` up to ~0.05. Half the keys put the value at well under 1% in a
  typical window, which is often outside the top 20. So the model's own per-window median rank may pass P3 for well
  under 24 keys. If so, the readout's 11 keys at the last block may be close to the model, and P3 was a control the
  reference itself fails. The memorisation diagnosis does not explain the P3 miss either. It points at *rare*-word
  rows, but the elicit values are common plain words, whose rows see many examples.
- Consequence: the verdict "unreadable" stands, since an elicit-clause fail reads unreadable either way. The
  explanation does not stand, and "What next" depends on it. A small-vocabulary readout scored by the same per-window
  rule would hit the same P3 wall if the model itself fails it.
- Fix: run the $0 check of a minute on `debug`. Read the model's own logits, and the linearised head's, through
  `readout.key_medians` and `readout.p3` on the 97 held-out windows, plus per-block possession for the model. Until
  that runs, reword each location above to "the readout ranks 11 elicit values 2 to 20 per window, where the model's
  mean distribution, which the pair was built from, ranks all 32; the model's own per-window ranks were not
  measured". Move the star off the possession figure, or relabel it as a different statistic. Add the check to Not
  run. Any follow-up readout (small vocabulary or more tokens) should file a P3 with the model's per-window ranks as
  its reference on the same rule.

### Medium

**M2. The "memorisation" cause is stated as established, and it conflicts with "under-fit".**
- Where: line 47 ("They memorised"); line 49 (rare-word rows; "would roughly halve the gap per doubling of tokens");
  Surprises ("vocabulary size, not repetition, was the limit"); Departures ("That is what made it under-fit"); the
  notebook's cell 5, Surprises and Departures.
- What: what was measured is a generalisation gap. Held-in windows read 0.22-0.30 nats better than held-out ones,
  and the model and the linearised head read the same on both. That is consistent with fitting once-seen tokens. The
  rare-word mechanism, the halving per doubling and "vocabulary, not repetition" are the second advice report's
  reasoning. No per-frequency split, small-vocabulary run or token-count sweep tested them. With 15.7M tokens drawn
  from 16.8M, about 60% of the held-in tokens were seen once, so the per-seen-token gap is somewhat larger than 0.3.
  "Under-fit" in Departures contradicts the held-in reading of 0.07 over the head.
- Fix: say "the readout fits its training text and generalises worse (0.3 nats)", and give the rare-word mechanism
  and the token extrapolation as the advice report's hypothesis. Replace "under-fit" with "over-fits its training
  text" or "too weak on unseen text". Add the per-frequency split of the held-in/held-out gap to Not run.

**M3. The depth trend is called "unexplained" without its simplest alternative.**
- Where: lines 90 and 104; the notebook's cell 8 and Surprises.
- What: "a readout that tracked the model well would hold the most elicit values near the output" assumes the model
  itself ranks the values per window, which M1 puts in doubt. The report's own guess fits M1: the values were picked
  as words likely after the key *on average*, and a block-0 readout behaves like a bigram table. The trend then says
  something about the construction statistic, not only about the readout.
- Fix: one sentence naming this alternative and the M1 check that would separate it. The trend is 21 → 17 → 17 → 13
  on ~3 windows per key with one readout seed, so "falls with depth" should carry that noise caveat.

### Low

**L1. The step-250 bar was raised knowing the exact reading it would govern.** Jobs 48344 and 48357 share the seed,
and both read 5.0088 at step 250. So the amendment from 5.0 to 6.0 (commit 0ef82ad, 11:23:12) let a reading through
that it had already seen. It is an early check, not a prediction, and the amendment is recorded openly. The report
should say in one clause that the rerun was deterministic and the raised bar was what let it pass. (The step-2000
raise from 0.5 to 1.0 was moot: 48357 read 3.956 against the model's 3.512, 0.44 over, inside the old bar.)

**L2. The notebook says "cipher values below rank 100"** (cell 2, Construction check). The rule is cipher median rank
*above* 100 (`readout.p3`, rank > 100 strict). Fix the wording.

**L3. The P2 definition in "What was measured" gives only the amended form** (log-odds share net of W₀) and does not
say that the first filing read A in nats (`amended_2026_11_02_pre_run.rule_as_first_filed`). P2 never ran, so this
is cosmetic, but the rule covers every amendment.

**L4. "Middle key's rank" is the 17th of 32** (upper middle). It is stated and computed correctly (block 0: 13; last:
25). "Median" elsewhere is the lower median. A reader may expect the same convention. Optional: use the 16th or say why
not.

**L5. The possession figure's diamond** (keys in 2-20) sits on an axis labelled "keys … at median rank <= 20". The
legend says what it is. Fine once the star is fixed (M1).

## Checked, fine

- **Numbers.** Every number in the loss table, the derived line (0.371, 0.074, 0.513, 0.143, scale 1.1008) and the
  possession table matches `run.json` at 25ba4fe and the 48357/48412 logs. The possession table covers counts all /
  fit, 2-20 counts 17/16/16/11, middle-key ranks 13/16/20/25, cipher middles 4,585/4,362/5,488/6,849, cipher best
  615/929/1,375/800, last-block range 1-14,060, and two keys at rank 1. Job 48279's row (5.672 … 4.140, gap 0.712)
  matches 4851105. 48344's 5.0088 matches 2f167bf. Job durations match the log timestamps.
- **Amendment timing.** `pre_run` c8bf37c 10:42:31 was before 48279 started (10:43:46). `after_stop` e746add 11:09:28
  was before 48344 (11:20:45). `step250` 0ef82ad/1a059ad 11:23 were before 48357 (11:24:10). `after_stop2` ea80150
  11:44:41 was before 48412 (11:45:42). Each job's recorded `git_head` contains its amendment. No P1-P3 reading
  existed before 48412.
- **Openness.** Each stop's first-filed bar is given in the report (0.5 over the model; 5.0; 0.5 at step 2,000) and
  in `predictions.json` (`rule_as_first_filed` / `rule_as_before`).
- **Final reading.** The P1 elicit-clause fail is read as unreadable, P3 as a stop, and E59 closes at the hard cap.
  All are as filed. `run.json`'s raw "fail" for P1 is correctly re-read by the amended rule.
- **Cipher clause.** "Little weight" is the right size. The argument that a weaker readout pushes towards "not held"
  is correct and stated plainly in both files.
- **Stage-0 inputs.** Twelve sha256 and gains have dG = 0 in all four logs, and the twelve-not-fifteen states are
  disclosed.
- **readout_loss.png.** It shows what its caption says, with the model, the linearised head and the bar on the same
  axes.
- **/writeup.** There is a plain three-sentence opening, the answer comes first in under 80 words, and each figure
  has a Reading it section. Scorecard, Departures and Not run are present. `check.py` reports 0 errors on both files.
  The flagged sentences are the provenance lines and a few of 26-32 words, which is acceptable.
- **Departures.** Vocabulary, training data, position, thresholds, causal check and schedule are complete for the
  literature comparison.

## Not run

- The model's and the linearised head's per-window median ranks on the 97 windows (M1). This is the one check that
  would settle the main explanation.
- Any recomputation from `readouts.pt` (not loaded; dev-node rule).
