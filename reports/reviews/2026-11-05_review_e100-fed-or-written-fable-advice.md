# The next locator for definition-lead's table (Fable's advice)

*Fable 5.1 for mechanism-detection-lead, 2026-11-05. Read: the literature review (`reports/reviews/2026-11-05_review_mechanism-location-literature.md`), `reports/plans/2026-11-05_plan_e97-amend1.md`, `experiments/locator_table/results/{run.out,amend1.out,stage1.json,stage2.json,table.json}`, `reports/analyses/2026-11-05_analysis_e90-old-labels.md` and `experiments/new_vs_reuse/apparatus.py`, `reports/analyses/2026-11-05_analysis_e95-token-identity.md`, `reports/plans/2026-11-05_plan_e99-refine-verify-definition.md` (definition-lead's), `.claude/skills/design/SKILL.md`, `~/commute-agents/fleet/roles/lead.md`. Nothing run on a GPU; one `jq` re-read of saved numbers (`~/commute-data/scratch/mechanism-detection-lead/fable-next/own_inherited.txt`). The file is named as asked; note that E99 is definition-lead's code, and the next free code is E100.*

**Limit on every claim.** A circuit built from old parts reads as reuse (`decisions.md` line 121).

## Answer

Do not run the review's recommendation 1 here: a DAS subspace is a third way of fitting C's directions inside the apparatus definition-lead owns (ask 19: value form, bands, controls), and it only reads rows that already have a label or a pair. Hand it to definition-lead as a note. The cleanest thing this lead can give every row, with no pairs and exact controls, is the split of each part's credit into what its own weight change does and what it inherits. The saved E97 numbers already show it separates the two construction poles: copy's gain is carried by an unchanged head fed from many small changes, the cipher's is written into the last two MLPs. Make that column the next E-code.

## Q1. DAS against E90's value form

| | E90 value form (`apparatus.py` lines 175-178, 190-195) | Review's recommendation 1 |
|---|---|---|
| Fit objective | per-value mean residuals, centred, orthonormal: correlational, rank = values − 1 (31 for 32 keys) | gradient descent on an interchange objective: W₀'s answer must follow the source's B; rank 1-4 |
| Site | residual after block 1 (block 2 beside), at the key's position, fitted on natural Pile text | head (2,4)'s output and the residual after block 2, at the answer position, fitted on task windows |
| Control | hand write at block 1's MLP spared; 20 random bases, rank-matched then variance-matched | the other task's gain spared; a random-subspace null on IIA |

All three differ, so it is not a repeat. But the verdict has the same shape, "share of gain lost under removal of a direction set at a site, against a random band, with a spared control", which `decisions.md` row 117 names as E90's apparatus, and E99 is already revising that apparatus (R-adm (b): a token-only reader must be spared). A rank-4 causal subspace is the natural candidate for R-adm (b), since E95's rank-31 means delete the key while a 4-dimensional subspace may not. That is definition-lead's question. Send the note; running it here would be the third direction-fit on the same question this week (E90, E95, then this).

## Q2. Is it clean, and where does it fail

It does not fix (a) by construction, only by site choice. A rank-1 subspace at mlp3's output also carries B (the answer's logit direction), so a DAS fit at mlp3 would score high IIA on copy; `amend1.out` lines 7-12 already show the direct-effect read put `mlp3->logits` first on all six E62 states. The review's "mlp3 cannot collect credit" holds only because the fit is not attempted there. It partly fixes (b): on lowercase the pair (1,1)+(1,2) has an input pair (`pairs.py` lines 7-8, the capitalised twin), so a rank-1 "case" direction can be fitted there, which the review wrongly counts as impossible; the constant-answer tasks (emoticon, ladder, many cues, rand1) have no pair and stay out. Its own failure modes: Makelov's illusion (IIA high but mean-replacing R in W₀ costs W₀ nothing on its own induction gap); a subspace found only where you look; k chosen by hand; and training on W₀'s windows then reading in a state, which no cited paper does. Known-answer controls that make it fail loudly: fit at every one of the 28 parts, not two, and report whether mlp3's fit clears the IIA band (if it always does, IIA has no locating power); removal of R must cost W₀ its induction gap (8.5 nats on the copy pair, `amend1.out` line 5) by a band set from the random subspaces; block-1 and block-3 plants must leave W₀'s IIA unchanged; the lookup pairs must read IIA near 0 at (2,4).

## Q3. Cleaner alternatives, and which need no pairs

- **Own against inherited from the saved revert and carry** (review's recommendation 2). No pairs; already saved for ten states. own_u = the revert share (`write`), inherited_u = carry_u − own_u. The re-read (scratch file above):

| State | Top carry part | carry | own | inherited | sum of single reverts |
|---|---|---|---|---|---|
| copy s0-s2 | mlp3 | 0.44-0.47 | 0.01 | 0.44-0.45 | 0.16-0.23 |
| copy s0-s2, located head (2,4) | | 0.33-0.38 | 0.005-0.008 | 0.32-0.38 | |
| cipher s0-s1 | mlp3 | 0.66-0.67 | 0.48-0.49 | 0.18 | 0.86-0.87 |
| cipher s0-s1, mlp2 | | 0.22-0.23 | 0.21-0.22 | 0.00 | |
| lowercase s0-s1 | mlp3 | 0.62-0.67 | 0.16 | 0.47-0.52 | 0.43-0.46 |
| lowercase s0-s1, mlp2 | | 0.36-0.38 | 0.26-0.27 | 0.10-0.11 | |
| lookup s0-s2 | mlp3 | 0.52-0.53 | 0.14-0.15 | 0.39 | 0.52-0.57 |

Copy, reuse by construction: every part's own share is at most 0.01 at the head and 0.11 at an MLP, the single reverts sum to a fifth of the gain, and the located head (2,4) is an unchanged part carrying 0.32-0.38. That is "an old part fed by many small changes", which is what strengthening should look like. Cipher, new by construction: mlp3's own share is 0.48, the reverts add to 0.86, and mlp2 inherits nothing. Lowercase sits between. This is post hoc and not a filed rule, but it is the only reading so far that puts the two poles on opposite sides without being told where to look, and it also says why carry misled: mlp3's credit on copy is 98% inherited.
- **Zhou 2026's 2×2 per part.** The same thing with its fourth cell, u's update alone grafted onto W₀ (Panigrahi's graft); it turns the revert's "own with the rest present" into "own alone" and gives the interaction. One forward per part, no pairs. Worth adding, since the copy update is so non-additive (sum 0.16-0.23) that "own alone" may read differently from "own with the rest".
- **Path patching from each part to the logits.** No pairs, but it still carries inherited change: it is the direct-effect read that `amend1.out` already ran, and it put mlp3 first on every E62 state.
- **Whole-output interchange at the located part on the task's existing pairs**, in W₀ and in the state, with the control task's pairs beside. The causal "does the answer follow the part's value" read without a fit; needs pairs, so no column on the constant-answer rows; second choice if definition-lead says the own/inherited column is weight-side and not what the definition reads.

So the pair-free column on every row is own/inherited (with the graft cell). It reads constant-answer tasks too, as a beside number under line 121.

## Q4. One recommendation: E100, "fed or written", one `debug` job, $0

For every saved 67M state and every part (24 heads, 4 MLPs): carry, revert, graft-alone, and inherited = carry − revert; the sum of single reverts as the additivity number. New code is the graft cell and the rule; the three other reads exist in `experiments/locator_table/`.

- **Runs first, alone, stop on a miss.** The two plants from E97 stage 0. Block-3 plant: own = graft = 1.0 ± 1e-4 at mlp3, inherited 0 ± 1e-4 everywhere, sum of reverts 1.0 (exact by construction). Block-1 plant: own = 1.0 at mlp1; inherited at mlp2, mlp3 and the block-2 and -3 heads is the floor, measured 0.021 in E97 (`run.out` line 7, `inheritance`); re-read within 0.005. Do-nothing: G = 0, "unreadable: no gain". Then E62's six states and E53's four as the poles, re-reading the table above within 0.02 (carry and revert are saved; only the graft is new).
- **Filed rule, bars from the same job's first stage** (as E97 set a from the copy seeds). Read at the located part where E97 has one, else the top carry part. ratio_u = own_u / carry_u. "Fed" when ratio ≤ the largest ratio over copy's three located heads (reads 0.02 now) and inherited ≥ the block-1 floor; "written" when ratio ≥ the smallest over the cipher's two mlp3 reads (0.72 now); between, "mixed". Plants first: block-1 plant's mlp3 must read fed, block-3 plant's mlp3 written. Edge cases: carry ≤ floor (no part to read), own > carry (written, ratio capped at 1), G ≤ 0 (unreadable).
- **Then stage 3's tasks** (`reports/plans/2026-11-05_plan_e97-locator.md` line 66: rank-1, half-known, cipher-24, copy-24, ident32 against rand32; emoticon and two ladder rungs beside). Filed leans, not bars: copy-24 and ident32 fed, cipher-24 and rand32 written, rank-1 fed, constant-answer rows reported beside with no lean.
- **Cost.** E97's four reads over 28 parts took 15-19 s per E62 state (`stage1.json` `seconds`) and 253-323 s per whole-row state (`stage2.json`), stage 1 99 s and stage 2 1282 s in all (`run.out` lines 34, 51); stage 0 25 s (line 9). One more read is a quarter of that: about 5 s per E62 state and 70 s per whole-row state. Stage 3's key-table states are E62-sized (E90's held-out windows), about 20 states at 20 s. Whole job under 25 min, one `debug` job, $0, under the gate.
- **What result changes course.** If a plant misses, a code bug, no reading. If ident32 reads written and rand32 fed, or copy-24 and cipher-24 come out on the same side, the signature does not track the construction labels and the column is dropped, with the table reported as what was read. If the poles hold, hand the column to definition-lead for E99's table and stop; the located part on the stage-3 rows stays E97 Amendment 2's job, not this run's.

**What would change this recommendation.** If definition-lead replies that a weight-side column does not bear on rows 81 and 115 (which are about C's function, not where δW sits), run the whole-output interchange at the located part instead, on the existing pairs, with the control task's pairs as the spared read. If E99's forced and forbidden fine-tunes are saved, use them as the written and fed poles in place of cipher and copy: they are ground truth for "through the part".

## Not run

- No GPU. The own/inherited table is a `jq` re-read of `stage1.json` and `stage2.json`, post hoc, not filed.
- The graft cell has never run; its 1e-4 plant bars assume the same fp32 floor as E97's.
- Nothing on Qwen; nothing at neuron grain; no check of whether head (2,4)'s output holds B at rank ≤ 4.

Friction: the asked-for filename says E99, but E99 is definition-lead's plan (`reports/plans/2026-11-05_plan_e99-refine-verify-definition.md`); the next free code is E100, and the file is named as asked.
