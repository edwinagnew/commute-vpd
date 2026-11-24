# Review: do the Results numbers match the plots beside them, and were the user's repeated requests met?

redteam-lead, 2026-11-24. Read-only audit of writeup-lead's paper branch `v2` at commit 33543c1 (after 99b0c2e, the
§5.1/§5.2 rewrite). Requested by the user via whip-lead (URGENT, 11-24). No paper edits were made.

We checked every number and claim in the rendered Results, Discussion, contributions and conclusion against the
figure beside it, and then checked twenty requests the user has made more than once against the current text. The
rewrite of the second results section now reads per-task values straight off its figure and they are all correct,
but the contributions, conclusion, tools table and appendix scorecard still count agreements pair by pair, and
several body numbers have no figure behind them or contradict another figure. For the paper's answer this means the
first claim's copy-head argument and the second claim's headline tallies are not yet supported by anything a reader
can see in a figure.

## Most serious open issues (fix first)

1. **Pair tallies survive, against the per-task rule.** main:129-131 and main:465-468 ("all nine built pairs,
   steering on five, effective rank on seven", "all ten copy fine-tunes on one side", p = 1.1e-5); the body no longer
   states any of these. Also sec5:100, sec5:113-114, sec5:135 "six of nine", the tools table (sec5:146 "13 of 15",
   :150 "5 of 9 above", :151 "reverses 4 of 9", caption :157), sec7:44, 48, 53, 72, appI:75, 92-102, and the appD
   scorecard with agrees / reverses / no call columns (appD:23-32), which the user named for removal on 11-23. The
   pending §5.2 draft fixes main:129/465, sec7:53/72 and table rows 150/151/157; the rest stays.
2. **The copy-head argument has no figure.** sec5:43 says the low copy reading "is not a failure of the
   featurization". It rests on the head read as one variable at top neurons (0.10-0.14 kept, sec5:42), plotted only in
   `figures/copy_head.pdf`, which is included nowhere; the VPD reading that pointed the other way (copy about equal to
   lookup, about 0.04) was cut in 99b0c2e; main:461-462 says the opposite (a low share "cannot yet be told from a
   change spread over many parts"). The head (2,4) 0.47-0.56 numbers (sec5:41-42) come from E62, a different
   experiment from Fig. copy_heads_off.
3. **Same states, different values in two figures.** The cipher states read share 0.378 / 0.300 in Fig. known but
   0.37 / 0.27 in fig:prior_mechanism, and steering 0.14 / 0.16 against 0.17 / 0.23 (sec5:81, appI:95). Unexplained.
4. **Fig. 2 (definition.pdf) compares unequal runs.** VPD fine-tunes are seed 0 only (10 rows) against 30 at top
   neurons and SAE, from mixed runs; the text cites per-task values (copy key 0.012, sec5:23-24) that the pooled
   strips cannot show. The unmerged fig2 draft fixes the counts (48 states each). Separately, sec5:19 "VPD appears
   the most reliable featurization" overclaims (VPD reads the rescale edit lowest, 0.76 against 0.95-1.00), and a
   "[TODO]" shows in that sentence; the pending E298 rows break "at most 0.02" (randoms 0.041, 0.033) and "only copy
   key above" for SAE.
5. **§5.3 numbers not in Fig. 5.** Text effective-rank τ 0.43 (p 0.03, a partial τ) against 0.36 printed in the
   panel; prior τ 0.34 on eight against 0.25 on twelve in the panel; kept-share τ −0.69 / −0.29 / −0.59 (sec5:109-111)
   beside a figure with no kept-share panel; "fifteen older trigger fine-tunes" (two have no trigger, sec5:103).
6. **Wrong statements in the tools paragraph.** CMAP at neurons "reverses" (it is 4 agree / 4 reverse / 1 no call,
   sec5:151, :157); WeightWatch "flags no built edit" (the new edits are flagged at 0.002 and 0.010, sec5:152);
   sec7:77-78 "do-nothing reads exactly 0" (it is 0/0, NaN); sec7:78-79 random-update claim untraced and contradicts
   sec7:73.
7. **Stale pointers after the move to the appendix.** sec7:53-56 cites §5.2 for "every pair follows the prior" and the
   ten-seed pairs (both now in appI, and §5.2 says "mostly"); sec5:84 cites appI for the twin control (it is appD
   tab:scorecard); the realistic caption (sec5:121-122) says Fig. known holds the prior; fig:known_appendix's caption
   says four readings but the figure has five, its title "do not separate" is wrong for neurons-for-half-gain
   (6 / 0 / 3), and two panel titles collide (appI:75-77).
8. **Credit to "the definition" for measured results.** main:52, main:65, main:458, sec7:9, sec7:12 say the
   definition separates / agrees / calls; these are kept-share readings (request 2 below).

## Part A: each Results finding at 33543c1

Fixed since ae89bfa: the 128-141 of 256 flip claim (deleted); §5.2 pair counts replaced by per-task ranges, all
correct off Fig. 4 (share 0.76-1.09 against 0.12-0.40; steering 0.19-0.74 against 0.14-0.68; effective rank 422-600
against 405-594); the prior "orders every pair" now "mostly" (correct: lookup sits above three copy keys); the Fig. 4
caption title now per-task.

| section | finding | status | line at 33543c1 |
|---|---|---|---|
| §5.1 | rescale edits 0.76 / 0.95-1.00; "[TODO]" visible | holds | sec5:19 |
| §5.1 | new and random edits "at most 0.02"; random edits matched on divergence, not "same size" | holds; wrong once E298 rows land | sec5:19, 22 |
| §5.1 | fine-tunes "two to five times" a random update | holds; low end is 1.8 | sec5:22 |
| §5.1 | top neurons: copy key 0.012, reuse tasks below ciphers | holds; not visible (tasks pooled) | sec5:23 |
| §5.1 | SAE "only copy key above"; caption "SAE scaled" | holds | sec5:24, 31 |
| §5.1 | "VPD appears the most reliable featurization" | new; overclaim | sec5:19 |
| §5.1 | copy heads "switched off inside each fine-tune" | new; ran on the four grid tasks only | sec5:36-38 |
| §5.1 | random heads "0.08 to 0.16" | holds; figure draws one line (0.135) | sec5:40 |
| §5.1 | head (2,4) 0.47-0.56 removed; 0.10-0.14 kept; "not a failure of the featurization" | holds / new; no included figure | sec5:41-43 |
| §5.2 | "three seeds" (appE table lists 2-8) | holds | sec5:72-73 |
| §5.2 | effective rank "filed from a small gap", cites appI, which does not say it | holds | sec5:73-74 |
| §5.2 | steering "fp32" point from an earlier run | holds | sec5:74-75 |
| §5.2 | twin control pointer to appI | wrong pointer (appD) | sec5:84 |
| §5.2 | gain-only 0.52-1.02 against 0.15-0.35 | correct; no figure | sec5:85-86 |
| §5.2 | "none tracks the kept share" (appI:154 says "closely"; rank correlations 0.41-0.46) | holds | sec5:86 |
| §5.2 | prior-by-mechanism numbers moved; cross-figure mismatch; "scaled to one common size" untraced | holds | appI:94-97 |
| §5.3 | effective rank 0.43 vs panel 0.36; prior 0.34/8 vs panel 0.25/12 | holds | sec5:98-101 |
| §5.3 | "fifteen older trigger fine-tunes"; median 13.0 is over 36 states | holds | sec5:103-104 |
| §5.3 | kept share τ −0.69 / −0.29 / −0.59, no panel | holds | sec5:109-111 |
| §5.3 | Qwen "0.99 or more" | holds; appendix only | sec5:116-117 |
| §5.3 | realistic caption: "The readings of Figure fig:known" | new mismatch | sec5:121-122 |
| §5.4 | "three tools read only their controls"; CMAP "six of nine" | holds | sec5:133-135 |
| §5.4 | CMAP neurons "reverses" (4/4/1); WeightWatch "flags no built edit"; Delta-Crosscoder "misses" | holds | sec5:151-157 |
| main | "a tenth of that head's change" | holds; no included figure | main:124-125, 459; sec7:13 |
| main | nine / five / seven tallies; ten seeds p = 1.1e-5 | holds, worse (body no longer states them) | main:129-131, 465-468 |
| main | low share "cannot yet be told from a change spread over many parts" | holds; contradicts sec5:43 | main:461-462 |
| sec7 | sec7:16-28 give the earlier p1-61 numbers (no remainder, seed 0, copy at step 100) beside Fig. 2's values | holds | sec7:16-28 |
| sec7 | "every reading with the right sign also follows the prior (§5.2)" | stale | sec7:53-56 |
| sec7 | do-nothing "exactly 0" (NaN); random updates "do not" (untraced) | holds | sec7:77-79 |
| appI | known_appendix caption four readings for five panels; title; collision | wrong | appI:75-77 |
| appI | prior_mech caption: 64-neuron share not drawn; per-key inverts | holds | appI:83-87 |
| appI | old-tasks caption 33 → 32 of 36; 1.57 comparison mis-sourced | holds | appI:128-150 |
| appE/appI | crosscoder caption promises Pile-loss labels the plot lacks; figure clipped | holds | appE:197-200; appI:163-170 |

## Part B: the user's repeated requests against 33543c1

The request list, with the user's quoted words and dates, was compiled from the user's own messages 11-15 to 11-24
(20 groups). Verdicts include what the two unmerged drafts (`draft-2026-11-24-fig2-standard.md`,
`draft-2026-11-24-sec5-2-all-tasks.md`) would change.

| # | Request (times said) | Verdict | Evidence | What is left |
|---|---|---|---|---|
| 1 | Per-task values, no pair comparisons (8+, latest 11-24: "i have said at least 5 times now ...") | partly | §5.2 and Fig. 4 now per task (sec5:60-67) | All the tallies in issue 1 above; the draft leaves sec5:135, row 146, main:131/467, sec5:100, appD scorecard, appI:92-102 |
| 2 | Keep the definition separate from its measurement (~9) | partly | the kept share named as an approximation (sec5:12, 18; sec4:24) | main:52, 65, 458; sec7:9, 12 credit measured results to "the definition" |
| 3 | Condense (~10) | partly | 53 → 32 pages; Qwen section and appF no longer rendered | Abstract still a draft with lead brackets (main:48-59); appE open "not traced" notes (appE:94-98) |
| 4 | Readable figures, title says what it shows, short captions (8+) | partly | bold claim titles (sec5:28, 50, 71, 121) | Fig. 4 caption carries provenance (sec5:73-75); the fig2 draft lengthens Fig. 2's caption to ~7 lines |
| 5 | Text describes the plot beside it (4, latest 11-24: "absolute failure fo the body to describe the experiment in the plot") | partly | §5.2 ranges match Fig. 4 | Fig. 2 has no task labels yet the text cites per-task values; orphan numbers in §5.1 and §5.3 (issues 2, 5); Fig. 5's printed τ never mentioned; most Fig. 4 tasks defined only in the appendix (sec4:49-51 defines copy and cipher) |
| 6 | Same runs and equal point counts in compared panels (3, all 11-24) | not done (Fig. 2); partly elsewhere | definition.csv: 10 VPD rows against 30 | fig2 draft makes Fig. 2 done (48 states each); Fig. 4's fp32 point and Fig. 5's 8 against 12 remain |
| 7 | Agent edits in a different colour (3+) | done | `\new` / `revised` green (main:20-22) | Uncoloured lines sec5:12-14, 18-21, 60 come from commit c552251, apparently the user's |
| 8 | Ask before pushing; don't push while the user edits (10+) | unclear | drafts say "not applied: the user is editing" | lead commits 99b0c2e and 33543c1 landed 6 min after the user's c552251; whip-lead to check |
| 9 | Geiger (causal abstraction) terms (6+) | mostly done | "featurization" 51 times, Geiger cited | "featurized model" twice; "soft intervention", "intervened model" unused |
| 10 | Baselines: literature-best method, authors' code (6) | partly | sec5:132; appG departures | crosscoder read by our own share; ADL and Jain "adapted"; CMAP at neurons ours (sec4:42, sec5:151) |
| 11 | Top neurons as featurization and baseline (6) | done | sec4:19, Fig. 2; appI:74-77 | neuron baselines in the appendix only |
| 12 | SAEs as a featurization (3) | done | sec4:13; Fig. 2 | — |
| 13 | Clear names, no overloaded symbols | partly | R means only the kept set (main:351, 381) | §5.2 titled "Proxy measurements" (sec5:58) while sec4:26, sec5:90 say "cheap readings"; red K* marks remain (main:543, 548) |
| 14 | Prior: one framing | done | "a confound" throughout | — |
| 15 | Two claims and the story in view | partly | contributions (main:124-139), results intro (sec5:12-14) | abstract (main:52-59) still claims the RL exception its own bracket says is dropped |
| 16 | Blind reader per section | partly | blind reads 1-5 logged | none after the 99b0c2e §5.1/§5.2 rewrite |
| 17 | Read Nanda's paper-writing post | not checkable | — | — |
| 18 | Answer the user's bracketed notes | mostly done | §5.2 bracket resolved in 99b0c2e | lead brackets open in the abstract (main:53-59) |
| 19 | Claim 2 on realistic fine-tunes | partly | §5.3: twelve ordinary fine-tunes, Qwen appendix | claim 2's support is still the built tasks |
| 20 | Readable decision cards | not checkable | not in the paper | — |

## Method

Five Opus agents each audited one block of the rendered text at ae89bfa (§5.1; §5.2; §5.3-5.4; contributions,
conclusion and §7; appendix figures), reading each figure's data file and PDF. A sixth re-checked every finding at
33543c1 and audited the rewritten text. A seventh checked the twenty request groups (compiled by a history search of
the user's own messages) against 33543c1 and the two pending drafts. Working files:
`~/commute-data/agents/redteam-lead/audit-parts/`.

## Not run

- No numbers were recomputed from raw saved states; values were read from the figures' data files and PDFs.
- The cross-figure mismatch (issue 3) was not traced to its cause.
- Requests 8, 17 and 20 cannot be judged from the paper's text.
- The E298 rows, E302 and the built-circuit jobs (66482-66486) are not yet in the paper and were not audited.

## Addendum 15:20: issue 3 traced

Both pairs are right; the figures read the same six cipher states (E58 prior-matched cipher and 96-key cipher,
seeds 0-2, step 30, mean over seeds) at different update sizes. Fig. known (known_main.csv, E299 from E136/E237
readings) reads each update at its trained size; fig:prior_mechanism (prior_mech_2x2_cued.csv, E250) first scales
each update along its own direction to the copy task's mean size (factors 0.780 and 0.752; copy 1). Share falls by
0.01-0.03 and steering rises by 0.03-0.07 under the scaling; effective rank and the copy rows agree to five digits
in both, which confirms scaling is the only difference. Fix is wording only: the fig:known caption (sec5:72) says
updates are read at their trained size; the fig:prior_mechanism caption (appI:85) says each is scaled to the copy
task's size, so its cipher values differ from Fig. known's. Never quote 0.27 / 0.23 beside Fig. known or 0.30 / 0.16
beside fig:prior_mechanism.
