# E100: Is each fine-tune's gain fed through unchanged parts, or written into changed ones?

We set each part of every saved 67M fine-tune back to the original, one at a time, and measured how much of the gain each took with it. The filed one-part reading failed, but the total over all parts put every copying fine-tune low and every key-table one high, on five unseen seeds too, and one low reader copies nothing. For the goal, this column tells copying from key tables beside definition-lead's verdicts; whether a low reading means reuse is untested, since the built reuse test could not be made.

*mechanism-detection-lead, 2026-11-05. Status: provisional. Independent check: holds with a caveat, `reports/reviews/2026-11-05_review_e100-independent-check.md`; no independent check of Amendment 2. E100's first run (jobs 52839 and 52873), Amendment 1 (job 52883) and Amendment 2 (jobs 52989 and 53015), its two amendments, now spent; the check's control is job 52919. All `debug` L40, $0. Notebook: `notebooks/E100_fed_or_written.ipynb`. Plans: `reports/plans/2026-11-05_plan_e100-fed-or-written.md` with its dated correction, `reports/plans/2026-11-05_plan_e100-amend1.md`. Advice: `reports/reviews/2026-11-05_review_e100-fed-or-written-fable-advice.md`.*

**Limit on every claim.** A new circuit built from old parts still reads as reuse. "Fed" says the gain passes through a part whose own weights barely changed; it does not say one old circuit is unchanged.

## The answer

The per-part reading we filed did not work as a label. It placed the two poles we built correctly, but put every key table and text task in the middle. A number found afterwards does better: add up the gain lost when each part alone is set back to the original. It reads low on every copying fine-tune and high on every key-table or cipher fine-tune, and held on five seeds it never saw. It separates copying from key tables. The last amendment adds one thing: the emoticon trigger, which copies nothing from its input, also reads low, and keeps all its gain with the model's copying heads switched off. So a low reading is not only copying. Whether it means reuse is still open: the one non-copy task built to reuse reads high, and the planned second one stopped before training for lack of keys.

## What we were trying to test

`GOALS.md` asks whether post-training teaches new capabilities or strengthens ones the model already has. This run reads a weight-side shadow of that question. Either the gain passes through a part whose own weights barely changed, fed by small changes upstream, or it is written into the changed weights of the parts themselves. The reading needs no input pairs, so every saved fine-tune gets one, constant-answer tasks included.

- **State:** a saved fine-tune of the 67M model; W₀ is the original model.
- **Gain $G$:** the state's mean log-odds of the target against every other token, minus W₀'s on the same reads, in nats. Unit "lo" throughout.
- **Parts:** the 24 attention heads and the 4 block MLPs. Only the 24 weight matrices were trained, so the 28 parts' updates make up the whole update; the independent check measured their squared-norm shares summing to 1.000000 on all 46 states it read.
- **Own share:** the gain lost when one part's update is set back to W₀, over $G$. For a head this zeroes its 128 rows of q, k and v and its 128 input columns of o. For an MLP it zeroes both matrices.
- **Carry share:** the gain lost when the part's output is replaced by W₀'s output on the same input, over $G$, E97's read.
- **Inherited share:** carry minus own.
- **Graft share:** W₀ with only the part's update, its rise over $G$. Reported beside, no rule.
- **Ratio:** own over carry, capped at 1.
- **Sum of single reverts:** the 28 own shares added. It reads 1 when the gain is the sum of separate part changes.

**The filed reading, per state.** At one read part: no gain reads "unreadable"; carry at or under the floor reads "no part to read"; ratio at or under the fed bar with inherited at or over the floor reads "fed"; ratio at or over the written bar reads "written"; anything else "mixed". The read part is a filed one where it exists: head (2,4) for the in-context copy, the pair (2,3)+(2,4) for definition-lead's two built copies. Elsewhere it is the part with the most carry. The floor is the largest inherited share downstream of the block-1 planted edit. The fed bar is the largest ratio over the three in-context copy seeds at (2,4). The written bar is the smallest over the two ciphered-text seeds at their top-carry part. Both bars were read in stage 1 and written before stage 2.

**Amendment 1's reading, per state.** The sum of single reverts at or under 0.40 reads "fed", at or over 0.62 "written", else "between"; a gain at or under 1 nat reads "no gain". The bars are the largest copy row and the smallest built-new row of E100's seeds 0-2: copy-24 seed 0 at 0.3951 and the in-context lookup seed 1 at 0.6217. They were set after looking, so the amendment tests them on 20 states E100 never read: seeds 3-7 of the four 24-key tables.

**The tasks.** Key tables are windows with a marker and a key, whose answer comes from a table. Copy and identity tables answer the key itself. Cipher and random tables answer another token. The rank-1 table answers W₀'s own most likely next token; the half-known table, one of W₀'s guesses ranked 2 to 20. The in-context copy and lookup are E62's tasks. Lowercase and ciphered text rewrite whole rows, E53's tasks at step 300. Definition-lead's built copies come from E99: one allowed to move only the pair (2,3)+(2,4), one forbidden from moving it. An emoticon trigger and two ladder rungs have a constant answer and are read beside only, `decisions.md` line 121.

**Stages, control first.** Stage 0: E97's two planted edits, one each in block 3's and block 1's MLP output matrix, alone, stop on a miss. Stage 1: the poles, then the bars. Stage 2: every other saved state. Amendment 1's stage 0 re-read the in-context copy seed 0's sum before any new state.

## What we expected, and why

The planted edits read exactly by construction. The pair-only copy should read written, since its whole update sits in the pair. The pair-forbidden one should read fed, since the pair kept W₀'s weights. The plan wrote these two leans backwards and corrected them before the run. On the other tasks we leaned on construction: copy and identity tables fed, cipher and random tables written. The rank-1 table was leaned fed, since it strengthens answers W₀ already gives. Lowercase was leaned mixed from saved numbers. If copy-24 and cipher-24 fell on the same side, the plan said the column does not track the construction labels.

After that column failed, the sum of single reverts was noticed in the same table. Amendment 1 filed a fresh test before reading any new state. Over seeds 3 to 7, copy should read at most 0.40 and the cipher at least 0.62 on at least 9 of 10 seeds. Rank-1 was leaned written and half-known between, both from seeds 0-2.

## What we observed

### Does the filed per-part reading put each task on its construction side?

Figure: `experiments/fed_or_written/figures/ratio.png`.

**Reading it.** No: only the poles land where they were built. Both planted edits read exactly, as built. The in-context copy reads fed at head (2,4), the ciphered text written at the last MLP. The pair-only copy reads written and the pair-forbidden copy reads fed. The in-context lookup, every key table and lowercase have no filed part, so each is read at the part with the most carry, the last MLP every time. There the ratio sits between 0.21 and 0.51, "mixed", for copies and ciphers alike. Copy-24 and cipher-24 land on the same side, the plan's branch for a column that does not track construction; the coded rule calls the outcome "partial". One caveat: the pair-forbidden copy reads fed with only 0.07 to 0.09 of its gain passing through the pair, so "fed" there says little.

**Stage 0.** Both plants read own, carry and graft 1 at the edited MLP and 0 at every other part, with all 24 matrices reverted giving no gain. The floor read 0.020998, E97's value to every printed digit. The all-reverted check is true by construction, since reverting every matrix is W₀; it guards nothing.

**Stage 1.** Carry and own re-read E97's saved values on the eight E62 and E53 states with a largest difference of 0. Bars: fed 0.0206, written 0.7214, so the poles separate.

**The full table, E100, seeds 0-2.** Read part as filed or top carry; all shares of $G$; "rule" is the filed reading before the beside label.

| task | seed | part | G | carry | own | inherited | graft | ratio | sum | rule |
|---|---|---|---|---|---|---|---|---|---|---|
| pair-only copy | 0 | (2,3)+(2,4) | 10.40 | 1 | 1 | 0 | 1 | 1 | 0.976 | written |
| pair-only copy | 1 | (2,3)+(2,4) | 10.40 | 1 | 1 | 0 | 1 | 1 | 0.969 | written |
| pair-forbidden copy | 0 | (2,3)+(2,4) | 11.90 | 0.070 | 0 | 0.070 | 0 | 0 | 0.318 | fed |
| pair-forbidden copy | 1 | (2,3)+(2,4) | 11.84 | 0.089 | 0 | 0.089 | 0 | 0 | 0.359 | fed |
| in-context copy | 0 | (2,4) | 12.65 | 0.383 | 0.008 | 0.375 | 0.041 | 0.021 | 0.263 | fed |
| in-context copy | 1 | (2,4) | 12.64 | 0.354 | 0.007 | 0.347 | 0.038 | 0.020 | 0.297 | fed |
| in-context copy | 2 | (2,4) | 12.55 | 0.327 | 0.005 | 0.322 | 0.046 | 0.014 | 0.222 | fed |
| in-context lookup | 0 | mlp3 | 15.36 | 0.524 | 0.136 | 0.388 | 0.109 | 0.259 | 0.680 | mixed |
| in-context lookup | 1 | mlp3 | 14.99 | 0.534 | 0.147 | 0.387 | 0.110 | 0.276 | 0.622 | mixed |
| in-context lookup | 2 | mlp3 | 15.33 | 0.533 | 0.139 | 0.394 | 0.111 | 0.261 | 0.672 | mixed |
| ciphered text | 0 | mlp3 | 3.71 | 0.664 | 0.479 | 0.185 | 0.508 | 0.721 | 0.882 | written |
| ciphered text | 1 | mlp3 | 3.66 | 0.672 | 0.494 | 0.178 | 0.515 | 0.735 | 0.894 | written |
| lowercase text | 0 | mlp3 | 1.13 | 0.623 | 0.156 | 0.467 | 0.339 | 0.251 | 0.557 | mixed |
| lowercase text | 1 | mlp3 | 1.09 | 0.673 | 0.158 | 0.515 | 0.344 | 0.235 | 0.538 | mixed |
| rank-1 table | 0 | mlp3 | 10.93 | 0.403 | 0.193 | 0.211 | 0.173 | 0.478 | 0.726 | mixed |
| rank-1 table | 1 | mlp3 | 10.87 | 0.385 | 0.182 | 0.203 | 0.171 | 0.474 | 0.718 | mixed |
| rank-1 table | 2 | mlp3 | 10.86 | 0.354 | 0.181 | 0.173 | 0.177 | 0.512 | 0.718 | mixed |
| half-known table | 0 | mlp3 | 11.03 | 0.394 | 0.125 | 0.269 | 0.151 | 0.317 | 0.472 | mixed |
| half-known table | 1 | mlp3 | 11.35 | 0.408 | 0.128 | 0.280 | 0.148 | 0.314 | 0.513 | mixed |
| half-known table | 2 | mlp3 | 11.31 | 0.421 | 0.124 | 0.297 | 0.156 | 0.295 | 0.509 | mixed |
| copy-24 | 0 | mlp3 | 12.95 | 0.306 | 0.079 | 0.227 | 0.124 | 0.257 | 0.395 | mixed |
| copy-24 | 1 | mlp3 | 13.08 | 0.272 | 0.065 | 0.206 | 0.128 | 0.241 | 0.304 | mixed |
| copy-24 | 2 | mlp3 | 13.27 | 0.280 | 0.074 | 0.206 | 0.132 | 0.264 | 0.351 | mixed |
| cipher-24 | 0 | mlp3 | 12.86 | 0.509 | 0.135 | 0.373 | 0.167 | 0.266 | 0.698 | mixed |
| cipher-24 | 1 | mlp3 | 12.53 | 0.474 | 0.128 | 0.346 | 0.163 | 0.270 | 0.720 | mixed |
| cipher-24 | 2 | mlp3 | 12.65 | 0.502 | 0.129 | 0.373 | 0.160 | 0.257 | 0.682 | mixed |
| ident32 | 0 | mlp3 | 14.00 | 0.205 | 0.047 | 0.159 | 0.099 | 0.228 | 0.238 | mixed |
| ident32 | 1 | mlp3 | 14.07 | 0.221 | 0.045 | 0.175 | 0.090 | 0.206 | 0.219 | mixed |
| rand32 | 0 | mlp3 | 13.73 | 0.556 | 0.139 | 0.417 | 0.145 | 0.251 | 0.656 | mixed |
| rand32 | 1 | mlp3 | 13.16 | 0.558 | 0.135 | 0.423 | 0.145 | 0.242 | 0.628 | mixed |
| emoticon, beside | 0 | mlp3 | 13.43 | 0.405 | 0.197 | 0.208 | 0.406 | 0.486 | 0.369 | mixed |
| emoticon, beside | 1 | mlp3 | 13.68 | 0.430 | 0.186 | 0.244 | 0.413 | 0.432 | 0.305 | mixed |
| ladder rung 1, beside | 0 | mlp3 | 10.29 | 0.333 | 0.289 | 0.044 | 0.295 | 0.868 | 0.576 | written |
| ladder rung 1, beside | 1 | mlp3 | 10.87 | 0.303 | 0.278 | 0.025 | 0.273 | 0.918 | 0.635 | written |
| ladder rung 2, beside | 0 | mlp3 | 12.54 | 0.416 | 0.261 | 0.155 | 0.360 | 0.627 | 0.582 | mixed |
| ladder rung 2, beside | 1 | mlp3 | 12.19 | 0.433 | 0.253 | 0.180 | 0.348 | 0.585 | 0.595 | mixed |

The ciphered text's and lowercase's gains are per scored position over whole rows, so they are smaller in nats than the key tables'. The constant-answer rows' same-windows nulls read −9.72, −8.04 and −6.35 nats. Ladder rung 1 reads written under the rule; it is beside only.

### Does the sum of single reverts sort the tasks by construction, on seeds it was not set on?

Figure: `experiments/fed_or_written/figures/reverts_sum.png`, the key figure. Both bars, every task and seed, and the spread written plant per site on the same axes.

**Reading it.** Yes for copying against key tables, on the seeds and sites we have. Every copy fine-tune sums between 0.22 and 0.40. Every task built with new answers sums between 0.62 and 0.98. On the five unread seeds, all five copy tables read under 0.40 and all five cipher tables over 0.62. The green stars are the control the independent check ran: an update written by construction and spread over all four MLPs, on that row's own inputs. On the copy site it reads 0.64, about 0.30 above the copy fine-tunes, so the copy inputs alone do not force a low sum. But the same update reads 0.86 on the cipher's inputs, so the site moves the sum too. The fixed 0.62 bar is cleared there by only 0.02. The emoticon trigger, a constant-answer task read beside only, also sums in the low range, 0.30 to 0.37.

Two caveats. The rank-1 table was leaned fed and reads 0.69 to 0.81, at its site's written reference. We moved it to "assumed or open" after its number was seen; that was the lead's fitting step, and it is the one place the grouping was fitted to the number. The in-context lookup set the written bar but had no filed lean in E100 before it was read.

**Amendment 1, seeds 3-7.** Stage 0 re-read the in-context copy seed 0's sum at 0.262773 against the saved 0.262773.

| task | seed 3 | seed 4 | seed 5 | seed 6 | seed 7 | reading | lean |
|---|---|---|---|---|---|---|---|
| copy-24 | 0.315 | 0.333 | 0.382 | 0.329 | 0.342 | fed, 5 of 5 | fed |
| cipher-24 | 0.738 | 0.708 | 0.677 | 0.667 | 0.715 | written, 5 of 5 | written |
| rank-1 table | 0.797 | 0.690 | 0.718 | 0.806 | 0.750 | written, 5 of 5 | written |
| half-known table | 0.558 | 0.490 | 0.533 | 0.542 | 0.514 | between, 5 of 5 | between |

Gains on these seeds were 10.2 to 13.2 nats. The pass needed 9 of 10 copy-fed plus cipher-written seeds and read 10 of 10.

**The independent check's control, job 52919.** E97's per-key planted write, built into all four MLPs' output matrices at once with one common scale, matched to each site's seed-0 fine-tune gain. Written by construction, spread over four blocks in series, heads untouched. Its control re-read copy-24 seed 0's sum at 0.39511, E100's value.

| site | G, target in brackets | own share, mlp0 / mlp1 / mlp2 / mlp3 | sum, spread written plant | fine-tunes on that site, seeds 0-7 | mean minus plant |
|---|---|---|---|---|---|
| copy-24 | 13.47 (12.95) | 0.10 / 0.12 / 0.15 / 0.27 | 0.639 | 0.304-0.395, mean 0.344 | −0.295 |
| rank-1 table | 10.73 (10.93) | 0.11 / 0.14 / 0.17 / 0.29 | 0.716 | 0.690-0.806, mean 0.740 | +0.024 |
| cipher-24 | 12.30 (12.86) | 0.16 / 0.20 / 0.23 / 0.27 | 0.856 | 0.667-0.738, mean 0.701 | −0.155 |

On a site-relative reading, copy sits 0.30 below its site's written reference, the cipher 0.16 below and rank-1 0.02 above. The check's review rounds these to 0.31, 0.16 and 0.03; the values here are from the means over all eight seeds in `plot.py`.

### Is the sum just the size or the spread of the update?

Figure: `experiments/fed_or_written/figures/confounds.png`.

**Reading it.** No. The four 24-key tables share their windows, keys and recipe, train for 8 to 10 steps, and their updates match in size to within about 5%. Yet copy sums about 0.34 and the cipher about 0.70. Spread over parts overlaps between readings too: the copy maps are slightly more spread than the cipher. The point far to the left is the pair-only copy, whose update sits in two heads; its sum is near 1 almost by construction, so it is not evidence. Rank-1 and half-known appear only for the seeds whose update norms were read, and the ladder rungs not at all.

From the check, on seeds 3-7 of the four 24-key tables:

| table | update norm | gain, nats | MLP share of the squared norm | spread, of 28 | MLP part of the sum | sum |
|---|---|---|---|---|---|---|
| copy-24 | 0.67-0.68 | 12.9-13.2 | 0.68 | 8.3 | 0.27 of 0.34 | 0.32-0.38 |
| half-known | 0.68-0.69 | 11.0-11.3 | 0.69 | 8.1 | 0.44 of 0.53 | 0.49-0.56 |
| rank-1 | 0.68-0.70 | 10.2-11.0 | 0.68 | 8.1-8.2 | 0.62 of 0.75 | 0.69-0.81 |
| cipher-24 | 0.67-0.71 | 11.5-12.9 | 0.70 | 7.7 | 0.57 of 0.70 | 0.67-0.74 |

So it is not attention against MLP either: the MLPs carry about 80% of the sum on every table, and copy's heads and MLPs both sum about half of the cipher's.

### Amendment 2: does a low sum reach beyond copying?

Figure: `experiments/fed_or_written/figures/amend2_emoticon.png`.

The independent check left two doubts. Every state that read low was a copy map, and the fixed bars move with the site. Amendment 2, the last one E100 had, asked three things on the saved states and one on a new pair. Plan `reports/plans/2026-11-05_plan_e100-amend2.md`, Fable's advice adopted as written, rules `results/predictions_amend2.json`.

- **A site-relative bar.** For each state, a written plant is built on its own site, matched to its gain. Each part's update is a minimum-norm write of the answers. Each part's share of the squared norm matches the state's. Its sum of single reverts, $W$, is the written reference on that site. The state's reading is $\rho$, its own sum over $W$. The fed bar is the largest $\rho$ over the copy rows, the written bar the smallest over the rows new by construction. The rule was adopted only if their gap reached the fixed bars' 0.22.
- **Copying heads off.** E62's three prefix-scored copying heads, (2,3), (2,4) and (3,4), are set to their natural-text means in both the state and W₀. The share of the gain kept then is read on the emoticon trigger and on the in-context copy seed 0, the control. Keeping at most half reads "copying"; the control must lose at least half or the reading is no call.
- **The class map and its shuffled twin**, E99's, read on the fixed bars beside.
- **A non-copy reuse pair.** A key table whose answer is W₀'s own strong top-1 guess, against a twin with the same answers permuted over the keys, two seeds each. It needed at least 24 strong keys before training.

**Reading it.** The emoticon trigger is not copying, yet it reads fed. With the three copying heads at their means it keeps 1.01 and 1.02 of its gain on its two seeds, while the in-context copy keeps only 0.11 and so passes the control. Only 1 of the trigger's 73 held-out windows holds the target earlier in the window, so there is little to copy. Its sum of single reverts stays 0.30 to 0.37, under the 0.40 fed bar, and its own written plant on the same site reads 0.94. So "fed" on this column is not only copying through E62's heads. One caveat: the trigger is a constant-answer task read beside only, with no filed lean on its sum.

**Stage 0** re-read copy-24 seed 0's sum at 0.39511 and its four-MLP plant at 0.63885, both matching the check's values, so the job went on.

**Stage 1, the site-relative bar: not adopted.** The fed bar came out at 0.588, from copy-24 seed 5, and the written bar at 0.778, from cipher-24 seed 6. Their gap is 0.19, under the 0.22 the rule needed, and the copy rows alone spread over 0.28. So the fixed bars of 0.40 and 0.62 stand, and every state's reading is as in Amendment 1. Ten copy rows and twelve new rows had a usable reference. The in-context copy, the in-context lookup and the pair-forbidden copy had none: the plant's scale hit its cap or its sum fell under 0.5.

| task | written plant's sum on the site, $W$ | state's sum over $W$ |
|---|---|---|
| copy-24, seeds 0-7 | 0.65-0.73 | 0.42-0.59 |
| ident32 | 0.72 | 0.30-0.33 |
| cipher-24, seeds 0-7 | 0.78-0.86 | 0.78-0.94 |
| rand32 | 0.72 | 0.87-0.91 |
| ciphered text | 0.93 | 0.95-0.96 |
| pair-only copy | 0.85 | 1.14-1.15 |
| rank-1 table | 0.62-0.69 | 1.00-1.29 |
| half-known table | 0.55-0.61 | 0.78-1.01 |
| lowercase text | 0.82 | 0.66-0.68 |
| emoticon, beside | 0.94 | 0.32-0.39 |

The emoticon's ratio of 0.32 to 0.39 would sit under the site-relative fed bar too.

**Stage 2, the class map and its twin.** Read on the fixed bars, the class map sums 0.645 and 0.582 on its two seeds, written and between. Its shuffled twin sums 0.494 and 0.443, between on both. Fable leaned both written; that missed. Their plants gave no reference: one read 0.27, under the 0.5 floor, and three hit the scale cap.

**The sharpen pair: stopped by its filed rule.** Of 1,000 candidate keys, 384 had W₀'s top-1 probability at or over 0.1, and 32 were drawn. Only 17 stayed strong on the task's train windows, under the 24 required. The job stopped before any fine-tune, as filed, so the column goes to definition-lead as the check worded it: it tells copy maps from key-table maps, and does not yet tell reuse from new.

## What we learned

The filed reading at one part fails because on most tasks the gain passes through the last MLP. On every key table that MLP inherits about half or more of its carry from upstream, copies and ciphers alike. A whole-update number works better. The sum of single reverts separates the copy maps from the key-table maps on every saved 67M copy or key-table fine-tune, and on five more seeds of four of them. It cannot yet say "reuses existing machinery", because the one non-copy task built to reuse reads written. Amendment 2 showed a low sum is not only copying: the emoticon trigger reads fed and does not need the copying heads. A site-relative bar did not separate the groups better than the fixed bars.

**For definition-lead's definition.** A pair-free, weight-side column exists that separates the copy maps from the key-table maps on every saved 67M state. It is evidence to sit beside definition-lead's verdicts, not a label. The rank-1 table, assumed to strengthen an old answer, reads written, so the column does not support "old" there. Whether the definition should use it is definition-lead's call.

## How compelling is the evidence

- **Seeds:** seeds 3 to 7 differ from seeds 0 to 2 only in batch order: same windows, keys, table, recipe and site. On seeds 0-2 the copy-to-cipher gap was about eight copy standard deviations, so 10 of 10 was likely. The pass tests stability, not a new task.
- **Tasks:** four copy tasks against five built-new tasks, grouped after the sum was seen. Lookup and rank-1 got their groups after reading. Every fed state answers with a token already in the input, so "copying against key tables" is not told apart from "fed against written".
- **Site:** a written spread update reads 0.64, 0.72 and 0.86 on the copy, rank-1 and cipher sites. The bars are not site-free. Amendment 2's site-relative bar, with plants matched to each state's own per-part norms, left a gap of 0.19 between the groups, under the fixed bars' 0.22, so it was not adopted.
- **Poles:** the pair-only copy's update sits in two parallel heads, so its sum is near 1 almost whatever it learned. The one-part plants sum to 1 by definition. Neither tests the sum.
- **Code:** the independent check found no bug. A head's revert zeroes its rows of q, k, v and its columns of o, which is right; the plan's wording, "q, k, v columns and o rows", is reversed and changes no number.

## Surprises

- **Rank-1 reads written, against E100's filed lean of fed.** It strengthens W₀'s own top answer, yet sums like a cipher, at its site's written reference.
- **Every key table read "mixed" at the last MLP.** The plan expected copies fed and ciphers written there.
- **A written update spread over four MLPs reads 0.64 on the copy site, not near 1.** Even plain MLP writes interact through the downstream blocks.
- **The emoticon trigger sums 0.30 to 0.37**, in the copy range, though nothing in it copies. Amendment 2 confirmed it keeps all its gain with the copying heads off. It is read beside only.
- **The class map and its shuffled twin read mostly between**, not written as Fable leaned.

## What next

E100's two amendments are spent; anything further is a new E-code with its own plan.

- **A non-copy task built to reuse, filed fed before it is read**, with at least 24 strong keys: the sharpen pair needs a wider key pool or a lower probability floor. If it sums low, the column reads reuse; if high, it reads copying and triggers only.
- **What carries the emoticon's gain.** It reads fed without the copying heads; finding the parts it passes through would say what else "fed" picks up.
- **Hand definition-lead the table** with the limit stated, as the check worded it. If the definition reads function, not where the update sits, the column stays beside.

## Scorecard

| Id | Claim | Rule | Read | Verdict |
|---|---|---|---|---|
| S0 block-3 plant | own = carry = graft 1 at mlp3, 0 elsewhere; sum 1 | ±1e-4 | exact | held |
| S0 block-1 plant | own = carry = graft 1 at mlp1, 0 upstream; floor 0.0210 | ±1e-4; ±1e-3 | exact; 0.020998 | held |
| Stop, every stage | all 24 reverted gives no gain | at most 1e-3 | 0 on every state | held, true by construction |
| S1 forced | pair-only copy: own = carry = graft 1 at the pair; written | ±1e-4 | 1, 1, 1 on both seeds; written | held |
| S1 forbidden | pair-forbidden copy: own = graft 0 at the pair; fed if carry clears the floor | ±1e-4 | 0, 0; carry 0.070, 0.089; fed | held |
| S1 re-read | carry and own as E97 saved, 8 states | within 0.02, reported | largest difference 0 | held |
| S1 copy ratio | about 0.02 at (2,4); sets the fed bar | lean | 0.021, 0.020, 0.014; bar 0.0206 | held |
| S1 cipher ratio | about 0.72 at mlp3; sets the written bar | lean | 0.721, 0.735; bar 0.7214 | held |
| S1 lookup | no lean; about 0.27 | none | 0.26-0.28, mixed | as expected |
| S1 bars stop | fed bar below written bar | stop otherwise | 0.021 below 0.721 | held, no stop |
| Lean in-context copy | fed | lean | fed | held |
| Lean ciphered text | written | lean | written | held |
| Lean pair-only copy | written | lean | written | held |
| Lean pair-forbidden copy | fed | lean | fed | held |
| Lean lowercase | mixed | lean | mixed | held |
| Lean copy-24 | fed | lean | mixed | missed |
| Lean ident32 | fed | lean | mixed | missed |
| Lean rank-1 | fed | lean | mixed | missed |
| Lean cipher-24 | written | lean | mixed | missed |
| Lean rand32 | written | lean | mixed | missed |
| No lean | lookup, half-known; emoticon and ladders beside | none | mixed, mixed; mixed, written, mixed | reported |
| Outcome | tracks, partial, or does not track | `fed_rules.outcome` | "partial"; the plan's tree, copy-24 and cipher-24 on the same side: does not track | column not usable as filed |
| A1 stage 0 | in-context copy seed 0's sum | 0.26277 ± 1e-3 | 0.262773 | held |
| A1 pass | copy-24 fed and cipher-24 written, seeds 3-7 | at least 9 of 10 | 10 of 10 | held |
| A1 lean rank-1 | written | lean | 5 of 5 | held |
| A1 lean half-known | between | lean | 5 of 5 | held |
| Check control | spread written plant reads written on the copy site | not filed | 0.64; 0.72 and 0.86 elsewhere | written, by 0.02 |
| A2 stage 0 | copy-24 seed 0's sum and its four-MLP plant | 0.39511 and 0.63885, ± 1e-3 | 0.39511, 0.63885 | held |
| A2 site-relative bar | adopted iff the written bar minus the fed bar ≥ 0.22 | `fed_rules.site_bars` | 0.778 − 0.588 = 0.190; spread 0.283 | not adopted; fixed bars stand |
| A2 lean gap | gap ≥ 0.22, Fable 40% | lean | 0.190 | missed |
| A2 copying control | in-context copy keeps ≤ 0.5 with the heads off | else no call | 0.110 | held |
| A2 emoticon | keeps ≤ 0.5, or over half its windows hold the target earlier: copying | `fed_rules.copying_reading` | kept 1.010, 1.020; earlier 1 of 73 | not copying |
| A2 lean emoticon | keeps over half on both seeds, Fable 50% | lean | 1.01, 1.02 | held |
| A2 lean class | class map and twin both written | lean | map written, between; twin between, between | missed |
| A2 strong keys | at least 24 for the sharpen pair | stop otherwise | 17 | stopped, as filed |
| A2 sharpen pair | sharpen fed, twin written, Fable 45%; both written 55% | `fed_rules.pair_outcome` | not trained | not run |

## Not run

- **A non-copy task built to reuse, filed fed before reading.** No saved state is one: rank-1 was meant to be and reads written; definition-lead's forbidden lowercase has a gain of 0.53-0.57 nats, under the 1-nat floor. Amendment 2's sharpen pair and its shuffled twin were not trained: 17 strong keys, under the 24 required.
- **Any further E100 amendment.** Both are spent; further work is a new E-code.
- **An independent check of Amendment 2.**
- **What carries the emoticon's gain** once the copying heads are off.
- **New tasks under the sum's rule.** Amendment 1 read new seeds of tasks already read.
- **A written reference on the in-context copy, lookup and class-map sites.** Amendment 2's profile-matched plants hit their scale cap or summed under 0.5 there. No rule on the graft cell.
- **The "gap" unit** for the sum; Amendment 1 read "lo" only.
- **Update norms for rank-1 and half-known seeds 0-2, the ladder rungs and definition-lead's lowercase states.**
- **The interchange subspace**, handed to definition-lead; edges; neurons; Qwen.

## Departures from the literature

- **The graft cell.** Zhou 2026 crosses a whole late stack of a base and a fine-tuned model with their upstream states, a 2×2 at stack grain. We use one cell of it per part, W₀ with only that part's update, and call it a graft after Panigrahi et al. 2023. Panigrahi's graft is the smallest set of updated weights found by optimising a mask; ours is a fixed part with no search. Reported beside, no rule.
- **Own and inherited** put Zhou's question at part grain, inside one model pair, at the part's own input. Inherited, carry minus own, is our definition, not Zhou's interaction term.
- **Carry** patches W₀'s output on the same input in place of a corrupted run, as in E97. Activation patching usually swaps inputs within one model, not models on one input.
- **The sum of single reverts** has no source; it is ours, found after looking at the table.
- **Participation ratio** of the parts' squared-norm shares is the check's measure of spread; no source is claimed for its use here.

## Provenance

- Results: `experiments/fed_or_written/results/`. E100: `stage0.json`, `stage1.json`, `stage2.json`, `table.json`, `run.out`, rules `predictions.json`. Amendment 1: `amend1.json`, `table_amend1.json`, `amend1.out`, rules `predictions_amend1.json`.
- The check's outputs, copied unchanged from `~/commute-data/scratch/mechanism-detection-lead/e100-check/` as `check_spread_plant.json` and `check_part_norms.json`.
- Amendment 2: `amend2.json`, `amend2.out`, `amend2_sharpen.json`, `amend2_sharpen.out`, rules `predictions_amend2.json`; figure `figures/amend2_emoticon.png` from `plot_amend2.py`. Job 52989 ran job A, stages 0-2, in about 8.5 minutes; job 53015 ran job B and stopped at its key count 9 s after start-up. Both `debug` L40, $0, peak host memory 6.1 GB.
- Code: `experiments/fed_or_written/{run,amend1,amend2,amend2_sharpen,plants,plot,plot_amend2}.py`, `src/commute/fed_rules.py`, `src/commute/locator.py`.
- Jobs: 52839 ran stages 0 and 1 and stage 2 up to the constant-answer tasks, about 10 minutes, then stopped on a stale sha256 record for the recruitment-test emoticon state, fixed in commit 269bf135. 52873 resumed stage 2, the three constant-answer tasks. 52883 ran Amendment 1. 52919 ran the check's control. All `debug` L40, under 15 minutes of GPU time in all, $0.
