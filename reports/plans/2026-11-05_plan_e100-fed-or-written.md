# Plan: is each 67M fine-tune's gain fed through unchanged parts or written into changed ones? (E100)

*mechanism-detection-lead, 2026-11-05. Status: coded next, under the gate ($0, `debug`). Advice: `reports/reviews/2026-11-05_review_e100-fed-or-written-fable-advice.md` (Fable's, adopted). Literature: `reports/reviews/2026-11-05_review_mechanism-location-literature.md`. The user's goal: "a clean way of finding mechanisms that helps serve definition-lead".*

**Limit on every claim.** A new circuit built from old parts still reads as reuse. "Fed" says the gain runs through a part whose own weights barely changed; it does not say one old circuit is unchanged.

## The request, in the Terms' words

For every saved 67M fine-tune and each of 28 parts (24 heads, 4 MLPs), split the part's credit for the gain into **own** (what reverting the part's own update removes) and **inherited** (what reaching the part from changed upstream parts adds). A gain carried by an unchanged part fed from upstream reads **fed**; a gain written into the part's own weights reads **written**. No input pairs, so every task gets a column, constant-answer tasks included (beside only, line 121).

## Design check

**Goal.** `GOALS.md`'s second question. "Fed through an unchanged part W₀ already uses" is the weight-side shadow of reuse; "written into new weights" the shadow of new. It gives definition-lead a column on every row, beside its verdict.

**Nearest tried.** E97 read carry (W₀'s output put into the state) and single-matrix reverts but never split them; carry credited mlp3, which Fable's re-read shows is 98% inherited on copy. E90's value removal and the literature review's interchange subspace are definition-lead's apparatus: handed over as a note, not run here. GOALS' direct against re-gating routes is the VPD analogue; this needs no decomposition.

**Patch or step?** A step: a pair-free reading that explains why every patch-based read credited the last MLP.

**Decisions relied on.** Lines 115-121, unchanged. No departure.

**What would change what we do next.** Below, under Tree.

## The reads, per state and part

- **Carry**: the state with the part's output replaced by W₀'s output on the same input (E97's read). Share of G.
- **Own**: the state with the part's own weights reverted to W₀ (a head: its q, k, v columns and o rows; an MLP: its matrices). Share of G.
- **Graft alone**: W₀ with only the part's update. Share of G (Zhou 2026's fourth cell; beside).
- **Inherited** = carry − own. **Sum of single reverts**, the non-additivity number.
- **Ratio** = own / carry at the part E97's removal located, else the top part by carry.

## Stages, control first

**Stage 0, plants, alone; stop on a miss.** E97's two one-matrix plants. Block-3 plant: own = carry = 1.0 ± 1e-4 at mlp3, inherited 0 everywhere. Block-1 plant: own 1.0 at mlp1, inherited downstream equal to E97's floor 0.021 within 1e-3 (`experiments/locator_table/results/stage2.json` `.inheritance`).

**Stage 1, the poles, bars set here.** E62 copy and lookup; E53 cipher at step 300. Fable's re-read of saved files leans: copy's (2,4) ratio about 0.02 (own 0.005-0.008 of carry 0.32-0.38); cipher's mlp3 ratio about 0.72. **Rule:** "fed" ≤ the largest ratio over copy's seeds at its located part, with inherited ≥ the floor; "written" ≥ the smallest ratio over the cipher's seeds at mlp3; "mixed" between. Bars are written to the results before stage 2. Stops if the copy bar is not below the cipher bar. If definition-lead's E99 forced and forbidden copy states are saved, they are read here as extra poles, leans "fed" and "written" respectively.

**Stage 2, every other saved state.** Lowercase (whole rows, lean mixed: mlp3 own 0.16 of 0.62-0.67), copy-24, ident32 and rank-1 (lean fed), cipher-24 and rand32 (lean written), half-known (no lean); emoticon and two ladder rungs beside only, against their same-windows nulls.

## Tree

- **A plant misses:** a code bug; fix and rerun.
- **Copy's bar not below the cipher's:** the split does not separate the poles; drop the column, report.
- **ident32 reads written and rand32 fed, or copy-24 and cipher-24 on the same side:** the column does not track the construction labels; report it as such, no label use.
- **Poles separate and the 24/32-key tasks fall on their construction sides:** hand definition-lead the table with the column beside its verdicts.

## Deliverable

`experiments/fed_or_written/results/table.json`, one row per (task, seed): `task`, `seed`, `part`, `carry`, `own`, `inherited`, `graft`, `ratio`, `reverts_sum`, `reading` (fed / written / mixed / beside), `G`. A `note-*.md` in definition-lead's folder when it lands, with the interchange-subspace idea from the literature review.

## Cost

Fable's sourced timings: 15-19 s per E62 state and 253-323 s per whole-row state for four reads (`experiments/locator_table/results/stage1.json`, `stage2.json` `seconds`); one debug job under 25 min, $0.

## Not run

The interchange subspace (definition-lead's call). Edges. Neurons. Qwen.

## Correction before the run (2026-11-05, lead)

Stage 1's E99 leans were written backwards above. The forced state moved only the pair's own q, k, v rows, so its gain is **written** there; the forbidden state kept the pair at W₀, so its gain is **fed**. The filed predictions say forced = written, forbidden = fed (`results/predictions.json` `leans_source`), which the lead confirms. Copy is read at (2,4) with the pair (2,3)+(2,4) beside; lowercase at mlp3 with its pair (1,1)+(1,2) beside.
