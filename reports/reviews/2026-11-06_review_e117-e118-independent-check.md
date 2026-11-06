# Review: independent check of the E117/E118 write-up (neurons and cheap comparators beside VPD)

*2026-11-06, independent check by an agent that wrote none of the code or the report. Scope:
`reports/analyses/2026-11-06_analysis_e117-e118-neurons-beside-vpd.md` and
`experiments/neuron_vpd_table/results/full_table.md`, read against `experiments/neuron_vpd_count/` (plan, `run.py`,
`score.py`, `predictions.json`, `cells.json`, `score.json`), `experiments/qwen_comparator_table/results/table.json` and
`experiments/neuron_vpd_table/build_full.py`. No model loaded, nothing re-run; every number below was read from
`cells.json` or `full_table.json` with `jq`.*

**Verdict: holds with caveats.** All five headline claims follow from the files, and no code error produces any of
them. The lowercase and cipher states really do switch on nothing at position 63 under the filed rule. That comes
from the definition, not from a bug. But the report's explanation of why is not shown by anything that was measured.
Two Qwen readings are partly an effect of the summary code: the known-facts "place" column includes a column that
measures only the rows, plus a cell the report itself calls noise. Several Qwen comparisons also read different rows
per arm without saying so where they are made.

## Per claim

| # | claim | verdict | why |
|---|---|---|---|
| 1 | VPD's set repeats across seeds more for copy than for random maps; neurons at the same count do not | holds | `score.json`: VPD gap 0.151 > within 0.105; neurons 0.053 < 0.212. The rule is applied as filed (`predictions.json` `separation_rule.as_applied`, `score.py:24-34`). Same windows, same position and the same count are confirmed in code (below). |
| 2 | VPD's full-width re-switching separates both families; the neuron share reverses on the maps | holds, one wording fix | r: 0.471 / 0.587 over within 0.042 / 0.049. r_n on the maps: the lowest random-map value (0.678) is above the highest copy-map value (0.503) by 0.175 > within 0.139, so "reversed" is right. But the report's "random maps 0.38 higher" is the negative of the rule's gap, min(copy) − max(random) = −0.383, not a size of the reversal. |
| 3 | The switched-on set carries little gain for either method | holds, stronger than stated | Ranges quoted correctly. Against the random same-count *range* (not its mean), VPD's re-switching set sits inside it on 4 of 8 states (ident32 s0, s1; rand32b s0, s1). The neuron set beats the random maximum on 7 of 8. |
| 4 | On Qwen, update size and gain already order every pass/fail of E93 | holds with caveat | Update size measures no rows, so its ordering is clean (margin 8.9e-05). Gain's ordering is decided across different rows: the closest pair is template a1_s0 (0.666, on the template rows) against N_s70_100 (0.595, on the GSM8K gold solutions). On rows the arms share (u1, u2 against RL, null and own-samples, all on gold solutions) gain orders by about 1.1 nats. The carrying share also mixes probes (a1/a2 from `qwen_carrying`, the others from E118's `carry/`). The E93 section does not say any of this. Arm type orders too, as the report says. |
| 5 | The known-facts end state is the only place where VPD's columns agree and the cheap ones do not | holds with caveat | "The cheap ones do not" leaves out the W0-rank column. That column reads f = 1 at the end, beside VPD, because it measures the rows and not the state (see the fix to `build_full.py:150-159`). The carrying share's −1.2 is driven by `known_s42_end` = 2.15, the cell the report's own section calls noise. The steering and gain columns read each arm on its own sentences. |

## The specific checks

**Neuron set matches VPD's count, windows and position: yes.** `run.py:115-128` (`change_at_read`) computes the gate
change and the firing change in the same loop, on the same batches of `t.toks`, at the same `t.half`. The neuron set is
the top K_VPD of that firing change (`run.py:211-214`); the smallest member's change is positive on every state
(`N_on_min_change` 0.29 to 0.56), so no neuron with a falling firing fills out the set. Panel A uses K_A, the rounded
mean of the two seeds' K_VPD, as filed. VPD's Jaccard compares sets of unequal size (for example 41 against 52), so its
ceiling is below 1 on every task while the neurons' ceiling is 1. That favours the neurons, so it does not inflate
claim 1.

**Panel B's restore is the filed form, but not the plan's form, for neurons.** VPD's side uses `RT.restore_hooks`
(`recruitment.py:175`), which `restored_at` (E50) wraps; this is the plan's form. The neuron side is scored with W0's
own firing a0 (`run.py:195`). The plan (`reports/plans/2026-11-06_plan_e117-neurons-at-vpd-count.md:47`) says "W0's MLP
weights on the fine-tuned model's MLP input", which is Ah. `predictions.json` records that the lead switched to a0
before the run, and Ah is reported in `cells.json` `B.neurons_ah`. The verdict does not change: Ah reads 0.0010 to
0.0026 on both classes, so it does not separate them either. The report does not list the switch among its departures.

**Panel C's hybrid is the plan's form.** VPD's side: W0's gates with the set's gates from the state at every position,
in C0's masked forward (`run.py:177`, `recruit.hybrid`). Neurons: W0's model with the set's firing from the state at
every position (`run.py:192`, `put(sel, a1)` with `at=None`). The all-alive and all-neuron limits reproduce E46/E53's
r within 4e-4 and E98's r_n within 1e-4, inside the filed tolerances of 0.005 and 0.01.

**The filed separation rule was applied as filed.** `score.py:24-34` matches `as_applied` (strict, with the edge cases
filed). I recomputed every A, B and C gap and within value from `cells.json`, and they match `score.json`.

**Lowercase and ciphers, K_VPD = 0: a real property of the filed definition, not a code path.** The code path is the one
that reproduced E12's counts exactly (41, 52, 27, 21). `change_at_read` slices position h into a length-1 slab and calls
`trigger_dg(..., half=0)`, which is correct. The sign is state minus W0, and the threshold is `> 0.3` (`recruit.py:39,
46`). The read position is E53's: `pa.CFG.half = 63`, on windows of 127 tokens predicting token 64
(`performed_anchor/run.py:73, 321`). The same `t.half` gives r = 0.752 and whole-width restore 0.657 on lowercase s0,
the same as E53's stored r and relay ρ. So the position is right. No earlier run took a switched-on set on these states.
The report's explanation, though, is not measured: "the gain is carried by many small importance changes, none above
0.3" (report lines 103-105). `cells.json` shows max |Δg| = 1 on every one of these states (anywhere, any position).
The filed set is a *mean over 500 Pile windows* above 0.3. Windows that each predict a different next token can switch
on different subcomponents strongly, and those changes average out. Nothing saved tells these two readings apart.

**Qwen comparisons across different rows per arm.** The general caveat is there (report line 222), but it is not
stated at the comparisons that depend on it:
- E93 ordering (lines 141-147): gain's 0.07-nat margin is template rows against gold solutions; nothing said.
- Template against fictional facts at full size (lines 162-163): every cheap column is read on different rows; nothing
  said (the matched-size bullet does say it).
- Known-facts place (lines 174-180): the known arm is on its own sentences (its carrying share on E103's probe), the
  anchors on the template and TOFU rows. The Departures list it (line 266); the section does not.
- RL against null (lines 164-165) uses the same rows, the GSM8K gold solutions, for both arms; that comparison is fine.
- `full_table.md` marks only the rank column with "∞ (rows differ)". The gain, steering and carrying cells of the
  same pairs read across rows too.

## Fixes

1. Report line 98-99: replace "random maps 0.38 higher" with "the ranges reverse: the lowest random-map r_n is 0.18
   above the highest copy-map one, over a within-class range of 0.14". The 0.383 in `full_table.md`'s whole-width
   table is the rule's gap, min(copy) − max(random), negative; a column note would make that readable.
2. Report lines 102-105 and the Surprises entry (232-233): drop "carried by many small importance changes, none above
   0.3" as a finding. Say what was measured: no alive subcomponent's *mean over 500 windows* change at position 63 is
   above 0.3, while per-window changes reach 1. Name averaging over heterogeneous windows as the open alternative. A
   $0 debug read of the largest mean Δg and the per-window share above 0.3 would decide it; it is a new reading, not a
   rule change.
3. Report lines 89-90: say that VPD's re-switching set sits inside its 20-random-set range on 4 of 8 states, while the
   neuron set is above the random maximum on 7 of 8 (`cells.json` `C.*.random.max`). This supports claim 3.
4. Report, Departures (257-267): add that the neuron restore in panel B uses W0's own firing a0 (lead's decision in
   `predictions.json`), not the plan's Ah at plan line 47. The Ah version (`B.neurons_ah`, 0.0010 to 0.0026) does not
   separate either.
5. Report lines 141-147 (E93): say that gain's closest pass/fail pair is read on different rows (template rows against
   gold solutions). On shared rows the maths arms are ordered by about 1.1 nats. Only update size reads no rows.
6. `build_full.py:150-159` (`known_f`), and `:165` (`cols`): leave the W0-rank column out of the place summary and the
   pair grid, or print it as "rows only". It is W0's rank on each arm's rows and does not change with training (f = 1
   at both sizes). Then report line 179-180 ("the cheap ones do not") holds for every column printed.
7. Report lines 178-179 and `full_table.md` known-facts table: mark the carrying share's end-of-training f (−1.2) as
   unreadable. Its known-arm mean averages 2.15 and 0.13, and the report's own section (lines 199-204) calls the 2.15
   noise; `known_f` checks only that the anchors separate, not the arm's seed spread.
8. Report lines 162-163 and 174-180: state at each comparison that the arms are read on different rows, as the
   matched-size bullet already does; `full_table.md`'s pair grid note (built at `build_full.py:229`) should say that
   the gain, steering and carrying cells of the anchor and known pairs are also across rows, not only rank.
9. Plan line 47 still names Ah for panel B's neuron side; a dated note in the plan pointing to `predictions.json`'s
   decision would stop the next reader from taking the plan's form as the one run.

None of these changes a filed verdict or a headline direction. Fixes 6 and 7 change two cells of the known-facts
place table, and fix 2 changes an explanation, not a number.

## Not run

- No re-run of E117 or E118, and no read of the mean Δg distribution at position 63 (not saved).
- Figures (`neuron_vpd_table/figures/full_*.png`) not inspected against the numbers.
- The E118 per-state comparator files (`compare/`, `carry/`) were read only through `table.json`'s values and sources,
  not recomputed.
- The E93, E94 and E104 VPD values were taken as stored; their own derivations are out of scope.
