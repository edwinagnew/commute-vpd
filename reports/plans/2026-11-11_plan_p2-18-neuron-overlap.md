# E155 plan: do the same top-firing neurons stay on after a fine-tune? (cell P2.18)

Cell P2.18 (part 2, compare-lead, rung L2.1, 1 run, $0 cluster debug, due 2026-11-13). Card:
`reports/plans/2026-11-11_plan_p2-18-card.md` (descriptive: user, 2026-11-11, ask 17). Code: `experiments/neuron_overlap/`.
Filed rule: `experiments/neuron_overlap/results/predictions.json`. No verdict on reuse; sides are named as each pair was built.

**The user's request, in the Terms' words.** For every saved fine-tune of the seven built pairs, on the rows the share was read
on, at the read position: does the original model's set of most strongly firing MLP neurons (top 240 of 3072 per block by
|firing|, Goodfire's 7.8%) stay the set after the fine-tune, and does the firing change fall on those neurons? Reported per side,
beside a random update of matched size; "same neurons, a bit harder" would read as high overlap, "new neurons" as low.

## Design check (a line each, `.claude/skills/design/SKILL.md`)
- **Goal served.** Part 2 of the roadmap (re-gating and the neuron reading against simple baselines on the 67M): this fills the
  plain neuron-overlap column the user asked for; it changes no definition.
- **Nearest tried.** E143 (`TRIED.md`, holes_fill): the top-set swap at k 64 and k matched, inert (under 5% of the gain), which
  reads what the sets do to the output, not whether they are the same sets; E52: idle share of the 16 most-shifted neurons,
  reversed (copy 0.41, cipher 0.125), on one pair. Neither reports the overlap; this does, on all seven pairs.
- **Patch or step.** A step: a requested baseline column, not a fix of E143.
- **Decisions relied on.** `decisions.md` 2026-11-11 "Neurons baseline (user request ... ask 17)": cell P2.18. No departure.
- **Adopted definition.** Under revision in P1.1; this run gives no verdict under either wording; the call is descriptive and
  adds no clause to any definition.
- **What would change what we do.** If the overlap separates the sides where E143's swap was inert, the neuron column of part 2
  gets the overlap as its reading; if it does not (the card's stop), the column records it and nothing goes to Qwen3-1.7B.
- **Control first.** Stage 0, alone: W0 against W0 + a zero update must read overlap exactly 1.0 (every block's smallest per-row
  Jaccard 1.0) and on_both nan at both k; a miss exits 1 before any state is read. It fails on a firing capture that differs
  between two identical forwards or a set computed from the wrong position. Then the positive control (below).
- **Numbers, expected values.** Per state at k 240: (a) overlap, (b) on_both. W0 and do-nothing: (a) 1.0 exactly, (b) nan. Random
  update of matched size (one per side): no filed number; lean (P1, low) that it reads a higher overlap than the real states
  (its size is spread over every direction), so the cell's "floor" may read as a ceiling. Positive control: above its random.
  Seed spread per side: measured in stage 1, the band of the call. k 64 recorded as a second form only (free: same forwards).
- **Early guards.** The exact control; stage 1 prints a projected job time after each state and stops past 25 min (after four).
- **Saved for audit.** Per state per block: mean and smallest Jaccard, summed squared change on both sets and on all, in
  `results/stage{0,1,2}.json` (small; nothing in `results/large/`).

## Rule (filed, `rules.py`; edge cases in `tests/test_neuron_overlap_rules.py`)
Per pair and reading at k 240: gap = mean(reuse side) - mean(new side) over saved seeds; band = the larger within-side max - min,
taken from this job's own stage-1 readings, never set in advance. "reuse side higher" (on overlap: the reuse side overlaps
more) iff gap > band; "new side higher" iff -gap > band; else "no call"; "unreadable" with fewer than 2 seeds on a side or a nan.
Ties (gap = band), the do-nothing state (gap 0 = band 0) and one seed checked. Why it might read the other way: E52's idle
share on copy vs random maps was reversed (copy moved neurons quiet at W0), so the copy side may overlap less (P2, low).

**Positive control.** E52's copy vs random states are saved, but they are the pair `e46_copy_random` here and E52 read them
reversed, so they are a cross-check, not a known answer. The apparatus control is built by hand: each block's c_fc scaled by
1 + d (d = the reference state e46/ident32_s0's c_fc update size over W0's), every other matrix unchanged, which makes the
neurons already firing fire harder; it must read above a random update of the same per-matrix sizes on both readings.
Recorded, not a stop (the cell has one run); a miss is said before any pair is read.

## Stages (one debug job, `cluster/neuron_overlap.sbatch`, `sbatch --comment=E155`)
| stage | what | writes |
|---|---|---|
| 0 | zero update (exact, stop), then the scaled update and its random | `stage0.json`, `STAGE 0 DONE` |
| 1 | the 56 saved updates of the seven pairs (E143's set, one read each), each written as it lands | `stage1.json`, `STAGE 1 DONE` |
| 2 | 12 random updates (one per side, a shared side once), then the calls and floors | `stage2.json`, `STAGE 2 DONE` |

Cost: $0 (cluster debug, one L40). Time: about 8 min, limit 30. Source: E143 (job 57802, `cluster/logs/commute-holes-fill-57802.out`)
read the same 56 updates on the same rows with steering and four TopK forwards each in 12 min, peak host RSS 6.1 GB; this job
runs two forwards per update and 71 updates in all.

## Reuse of existing code
`commute.masked_reads.top_overlap` (new, beside `top_masks`, which it calls) is the only shared change: an added function, so the
frozen callers of `masked_reads` (`experiments/holes_fill/run.py`, `experiments/real_vs_twin/run.py`) are unaffected. Rows and
states load through E136's `windows.load_read` and `inventory.json`, as E143's did; nothing is copied from a finished `run.py`.

## After the run
`plot.py` draws `figures/gaps.png` (per pair the gap as a dot, the band in grey, the random updates' gap as a cross, per reading);
one paragraph by `/writeup`; the cell is filled with `fleet/cell fill`, and a line goes to `docs/research_log/PENDING.md`.
