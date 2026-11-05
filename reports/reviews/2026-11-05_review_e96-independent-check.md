# Review: independent check of E96 (locate the mechanism)

**Verdict: holds with caveat.** Every number in the report re-derives from the saved JSON, and an independent re-measurement with its own hook re-reads the shares to eight digits. Both filed outcomes stand under every variation tried: in the plan's own unit, at full context, with other means, and with errors by row. Three readings around the lowercase result need correcting. The "unexplained" W₀ gap has a cause: E53's windows give 64 tokens of context. "About half the share is W₀ rising" holds on those windows only. And the report's probabilities are probabilities at the mean log-odds, which here sit 10 to 60 times below the mean probability.

*Independent checker, 2026-11-05. I wrote none of E96. Read: the analysis report, the plan, `experiments/locate_mechanism/{run,stage2,plot}.py`, `src/commute/locate.py`, `src/commute/copying.py` (`head_means`, `ablated`, `read_lo`, `share_lost`), `src/commute/performed.py`, `experiments/performed_anchor/{a0,run}.py`, the target model's attention (`~/param-decomp/param_decomp/pretrain/models/llama_simple_mlp.py`), and every results file the report cites, read with `jq`. GPU re-measurement: job 52604, debug L40, 363 s, $0. Script, JSON and log are in `~/commute-data/scratch/mechanism-detection-lead/e96-check/` (`e96_check.py`, `e96_check.json`, `e96_check.out`).*

## What changes in the report

| Report says | The check finds | Severity |
|---|---|---|
| W₀'s gap of 0.25 against A0's 1.18 is unexplained (line 186) | Context length. The same 248 positions read -0.88 at 16 tokens, 0.35 at 64 and 1.85 with the whole row (F2) | medium |
| About half of the lowercase share is W₀ rising (lines 83-85) | True on E53's 64-token windows. On whole rows W₀ rises only 0.08 with (1,4) off, and no head comes near the bar (F2) | medium |
| W₀'s probability of the target is about 0.0012; a net share is unsafe near 0.001 (lines 98, 110, 176) | That is the probability at the mean log-odds. W₀'s mean probability is 0.074 and the state's 0.114 (F5) | medium |
| Lowercase is read as log p(lowercase) - log p(capitalised) (plan line 73), not listed as departed | The filed read is log-odds against all tokens. In the plan's unit the null is stronger: no head over 0.21, no pair over 0.34 (F1) | low |
| MLP units scored "on the gain" (line 240) | Scored on the state's own log-odds, as filed (F6) | low |
| Standard error 0.045 to 0.047 on the top share | By row it is smaller, 0.028 to 0.029. A row bootstrap puts (1,4) at or over 0.47 in 2% and 6% of draws (F4) | low |
| Means taken from 41-token spans (lines 180, 242) | Means from 127-token spans move (1,4)'s share to 0.455 and 0.474, so seed 1 crosses the bar. Seed 0 does not, so the outcome stands (F3) | low |
| (1,5) costs 0.13 to 0.15 of the gain (line 106) | 0.13 to 0.14 | trivial |

## 1. Re-derivation of the report's numbers

Re-derived with `jq` from every results file the report cites, plus the logs' timestamps. These match to the digits printed: both full tables, row for row; the stage 1 table and its bands; the scorecard; the own-terms table, including W₀'s share of the net fall (0.44 to 0.63); the post hoc MLP numbers; the probe counts; the gradient check; the gap counts (20 heads positive, 12 above intact, 17 "gain only"); the corrections to the advice; and both jobs' durations (49 s and about 4.5 min). The one slip is line 106, as in the table above.

## 2. The independent re-measurement

**Same hook, same numbers.** My own o_proj pre-hook, my own means and my own log-odds give the same shares. Copy seed 0 with (2,4) off reads 0.56270 against 0.56270 saved; lookup with (1,4) 0.15428, lowercase with (1,4) 0.40850, cipher with (1,1) 0.08456. All eight cases agree to about 1e-8. My means equal `means.pt` exactly (largest difference 0). W₀'s gap at the 500 drawn windows reads 0.25233, as E96's. At A0's 6,070 positions under bf16 it reads 1.1772, and A0 filed 1.1776.

## 3. Findings

Severity: **high** changes a filed outcome or the answer paragraph; **medium** changes how a result should be read; **low** is wording or a weak guard.

### F2. The gap of 0.25 is the 64-token context (medium)
`a0.py` reads `performed.log_odds` on whole 512-token rows. E53's windows (`performed.windows_at`, `performed.py:186-199`, half 63) give the read 64 tokens, starting mid-row. The position draw is not the cause. The drawn positions average 253.9 against 254.4 for all 6,070, and the same 500 positions read 1.39 with the whole row. On the 248 drawn positions at index 256 or later, the gap grows with context: -0.88 at 16 tokens, -0.36 at 32, 0.35 at 64, 1.08 at 128, 1.61 at 256 and 1.85 with the whole row. By where the lowercase form occurred earlier:

| Lowercase form earlier in the row | Windows | Gap, whole row | Gap, 64 tokens |
|---|---|---|---|
| inside the window | 157 | 2.46 | 1.36 |
| only before the window | 96 | 1.08 | -1.05 |
| nowhere | 247 | 0.82 | 0.06 |

So E53's scored behaviour is lowercase continuation from 64 tokens of lowercase text, which W₀ barely performs. The fine-tune trained on whole rows.

**On whole rows the reading shifts but the outcome does not.** Seed 0 on A0's positions: lowercase G is 1.13, and the top head shares are (0,0) 0.33, (1,5) 0.28, (1,4) 0.27 and (0,5) 0.27. The cipher's are 0.10 to 0.15. With (1,4) off, W₀ rises 0.08 there against 0.39 on the windows, and the state falls 0.22. "Not found at head grain" holds. "About half of the share is W₀ rising" is a property of the 64-token windows. **What it changes:** line 186 has a cause. The capability bar of half of 0.25 is low because of the window. Lines 83-85 should be scoped to E53's windows.

### F5. The report's probabilities sit far below the mean probability (medium)
`locate.prob(float(off.mean()))` (`run.py:99`, `locate.py:169-171`) is the probability at the mean log-odds, as filed (`predictions_stage2.json`, "lo, p"). The report never says so. Measured on seed 0:

| Read | At the mean log-odds | Mean probability |
|---|---|---|
| W₀, lowercase target | 0.0012 | 0.074 |
| Lowercase state | 0.0062 | 0.114 |
| Lowercase state, (1,4) off | 0.0046 | 0.100 |
| Copy state, (2,4) off | 0.024 | 0.202 |
| W₀, copy target | 0.0003 | 0.0076 |

W₀ is not near a probability of 0.001 on lowercase, so line 176's lesson does not follow as worded. The copy control's removal rule (probability at most 0.1, the user's "first reading") passes on the probability at the mean log-odds. It would not pass on the mean probability on seed 0, which reads 0.20. The filed definition stands, so the control passes as filed. A reader who takes "probability falls to 0.1 or less" as a mean probability would read it otherwise. **What it changes:** the wording at lines 31, 98, 110 and 176. Any later rule written in probability should name which one it means.

### F1. The plan's unit makes the lowercase null stronger (low)
Plan line 73 reads lowercase as log p(lowercase) - log p(capitalised). The filing (`predictions_stage2.json`, definitions "G") and the code (`stage2.py:81`, `run.py:96-99`) use the log-odds of the lowercase token against all tokens. The report's departures list leaves this out. Re-read in gap units, the gain is 4.1 on lowercase and 4.6 on the cipher. The best head shares are (1,4) 0.21, (1,5) 0.19 to 0.20 and (0,0) 0.19. The best pairs read 0.29 and 0.32 to 0.34, and their cipher costs are 0.26 to 0.28. No part meets the rule in either unit (`sweep_gap`). **What it changes:** a departure line. The outcome holds more firmly.

### F6. Head (1,4) off makes W₀ more lowercase, not flatter (low; reading)
With (1,4) off, W₀'s entropy at the read falls from 4.52 to 4.24 and its largest probability rises from 0.27 to 0.32. The lowercase form's log-probability rises 0.36, and the capitalised form's falls 0.96 (`entropy`). The four top-share heads behave alike: each lowers the capitalised form by 0.4 to 1.0. So W₀ rising is W₀ losing a push toward the capitalised form, not a generic flattening. The report's "W₀'s sensitivity to switching a head off" is accurate, and this names the direction.

### F3. The source of the means moves the top share by up to 0.05 (low)
The means are one global vector per head, at W₀, over 41-token spans of cased train text (`run.py:276-279`, `copying.py:135-157`), used in W₀ and in every state. Means over 127-token spans give (1,4) a share of 0.455 and 0.474 with cased text, and 0.443 and 0.460 with lowercased text. That is up to 0.05 above E96's 0.408 and 0.426. W₀'s rise barely moves (0.39 to 0.42), so the text type is not behind "W₀ rising". (1,4)'s 41-token mean differs from its late-position mean by 0.47 of its norm, and (2,4)'s by 1.3. **What it changes:** seed 1 crosses 0.47 under one alternative, and seed 0 does not under any, so "not found" stands. The margin to the bar is the size of this choice. The report should say so beside the bar.

### F4. Errors by row are smaller, not larger (low)
`share_se` (`stage2.py:135-140`) treats the 500 overlapping windows from 187 rows as independent. Clustered by row, the error on (1,4)'s share is 0.028 and 0.029, against 0.045 and 0.047. The row bootstrap's 90% intervals are 0.36 to 0.45 and 0.38 to 0.47. I expected larger errors; they came out smaller. "Under the bar" is firmer than the report's errors suggest.

### F7. The hook replaces one head's output; the o_proj columns are negligible (none)
`y` is (B, heads, T, 128), transposed and flattened before `self.o_proj(y)` (`llama_simple_mlp.py:248-249`). The slice `[h·128, (h+1)·128)` is head h, and attention is causal. In the state, the fine-tune's o_proj columns for the head still multiply W₀'s mean. Removing them as well moves the state's log-odds by 0.002 to 0.003 on (1,4), (1,1) and (2,4) (`o_proj`), so this is immaterial here. Under bf16 autocast the (1,4) share is 0.410 against 0.408 in fp32 (`bf16`).

### F8. Several "as expected" rows could not have missed (low)
The stage 1 bands were set from E62's reads of the same six states, on the same windows, by the same path, with the same draw of means (`score.json` `share_diff_run`, 0 on all six). So were the copy top share, its probability off, the second head and the 0.1 removal bar (from E62 Amendment 2's 0.024 to 0.071 on these states). Stage 0's zero update reads 0 because W₀ + 0 is W₀ on a deterministic path. Only two things there were new: whether any of the 18 unread heads beat (2,4), and what the nominators said. The report calls the control weak (lines 31, 180). The scorecard still lists the re-reads beside rows that could fail.

### F9. The MLP score is on the state's own log-odds (low)
`locate.unit_scores` (`locate.py:127-147`) scores the state's own log-odds, as filed. The plan (line 46) and the report (line 240) say "on the gain". A score on the state's own log-odds picks units W₀ also uses for the target. That alone is a reason the set reads "not specific" on copy, costing the lookup 0.27 to 0.34. It also explains why the post hoc lowercase set reads "shared". A gain-net score was not run.

### F10. Stage 2's control is on other windows (low)
The plan's request (line 9) is a control "on the same windows". Copy and lookup share windows. The cipher reads ciphered rows with its own 500 windows (`stage2.py:240-247`). W₀ moves differently on the two: (1,1) off is -0.43 on lowercase and +0.26 on the cipher. This belongs in the report's caveats.

### F11. No leak of the expected head (none)
The expected head reaches only `L.score`, from `predictions.json` (`run.py:243-246`). The nominators, the sweep and the greedy pair never read it (`run.py:132-237`, `locate.py:30-147`). `heads.pt` is never opened. Stage 2's bar of 0.47 is the control's weakest seed, filed before stage 2 ran. The gradient check's path integral matches the true fall within 10% on all four cases.

### F12. Weak MLP hold-out, broad answer paragraph (low)
In stage 2, even and odd windows are neighbours in `at`'s order and often overlap in one row (`stage2.py:193-202`). So the post hoc "more than half of the gain stays" is read on near-duplicates of the scoring windows. The opening paragraph generalises from one copying pair whose head was known and one broad task read at 64 tokens of context.

## Not run
- A gain-net MLP score (the state's gradient times weight minus W₀'s), for F9.
- A cipher control on the lowercase windows themselves, for F10. No such state exists.
- The whole-row sweep on seed 1, and pairs on whole rows; seed 0 single heads only.
- Positions restricted to long context (index 256 or later) with E53's 127-token windows. The whole-row reading stands in for it.
