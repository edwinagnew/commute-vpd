# Search recovery: is E184's miss the search or the class?

*2026-11-16, definition-lead (designer). Cell P1.21 (rung L1.4, 2 runs, due 2026-11-18), experiment E188. Card
`experiments/search_recovery/card.md`; code `experiments/search_recovery/run.py`; filed rules
`experiments/search_recovery/results/predictions.json`; Slurm `cluster/search_recovery.sbatch`. Provisional: the fourth
reuse wording is tested, not adopted; nothing here is a verdict on any reuse definition.*

E184 ranked seven built items above their filed rungs (planted constant 3 against 1, planted trigger 3 against 2, cue 4
against 2, twin 3 against 2 on both seeds, fact 4 against 3 on both seeds). This run writes each built edit by hand in
the ranking's own class and reads how much of the gap it closes, so a miss can be put down either to the greedy search
or to the class of interventions itself. Only misses the search caused get a repaired search.

## Design check (`/design`)

- **Goal.** `GOALS.md`, the paper plan's first part: a working definition of reused against new. The fourth wording is
  only usable if its ranking reads what an item was built as; this run says whether E184's misses are the ranking's
  search (fixable) or the wording's class (the filed rungs were wrong).
- **Nearest tried.** E184 (`TRIED.md`, cell P1.20): it named this exact gap in its "Not run" list (no search started from
  the built edit). Nothing new is measured beyond that question.
- **Patch or step.** A patch on E184, inside a named open cell (P1.21, "done when" quoted on the card).
- **Decisions.** None departed from; the fourth wording stays "tested, not adopted" (`decisions.md`, 2026-11-13 row,
  "tested, not adopted, as cell P1.20"). No rule, band or score is added to any verdict: the attribution is a reading
  of this run, not a definition change.
- **What changes what we do.** Rival a (some miss is the search's): repair the search and re-read the wording with it.
  Rival b (every miss is the class's): the wording as ranked grades hand-direction edits at rung 3 by construction, so
  a fourth-wording verdict on real fine-tunes needs a different list or rung definition, which is the user's call.
- **Control first.** Do-nothing (KL under the floor), one neuron shifted and one neuron triggered by in1's first
  principal direction, each written by hand as the Write E184's search found: closure 1. Then the three hand-built
  items' exact rank-one forms: closure 1. Any miss stops the job: the class code would not express what it claims.
- **Early guards.** The rebuilt F must repeat E184's KL(F || M) within 0.1% (same windows and seeds), logged per item.

## What runs (one job, stage 1 then stage 2)

**Stage 1, the oracle, no search.** Every built edit is a push along $Qb$ at in2, times a label $L$: 1 (planted
constant), a linear read of in1 (planted trigger: the parity direction; cue: $w$), or the teacher's token-set label
(twin: last number odd; fact: its random number set) for the two real fine-tunes. Per item, on E184's own windows:
the edit as one rank-one edit written by hand (exact for the three built items; for the fine-tunes the teacher's form,
refitted); and the edit rewritten on M's list with at most 16 names (writes: the in2 principal directions most along
the push, or the logits F moves most; reads: the list variables before in2 most correlated with $L$), numbers fitted
by L-BFGS from a least-squares start. Each miss at each bar is labelled **search** (an oracle at the filed rung
closes the bar), **class** (only the rank-one form does) or **oracle short**; a separate flag marks a rung-3 search miss
(the rank-one form closes a bar E184 read at rung 4). For the best list oracle the run also records where its names
stood in E184's first greedy step (score rank against the line search's cut, Newton step against the fitted value).

**Stage 2, the repaired search**, only on items with a search label or a rung-3 flag: E184's search on list all with
four times the candidates (16), a line search out to 16 Newton steps (E184: 2), and five times the rung-3 steps. The
controls go first: one neuron and the triggered neuron must read exactly as in E184, and the random bit must stay at
rung 4 (the repair must not over-close); any change stops the job.

## Expected values (filed)

| Number | Do-nothing / controls | Items |
|---|---|---|
| KL(F ∥ M) | do-nothing < 1e-6 (E184: 0) | E184's within 0.1% |
| Hand-written control closure | one neuron, triggered neuron ≥ 0.999 (E184's searches: 1 − 4e-9, 1 − 7e-9) | — |
| Rank-one form | — | planted constant, trigger, cue ≥ 0.999; twin 0.85-0.93; fact < 0.8 |
| Best list oracle at filed rung | — | every item < 0.8 |
| Share of the push in in2's 64 directions | — | 0.12 (measured on CPU; the cue's read: 0.004 of $w$ in in1's span) |
| Label at 0.9 | — | class (constant, trigger, cue, twin), oracle short (fact); rung-3 flags: trigger 0.95, cue 0.9 and 0.95 |
| Repaired readings | controls as E184; random bit rung 4 | trigger rung 3 at 0.95, cue rung 3 at 0.9, 0.95; nothing at its filed rung |

**Filed rule R1** (list all, bar 0.9): rival a if any item is labelled search; else rival b. Total once stage 1
passes. Expected: rival b. R2 (descriptive): whether the repaired search recovers each search-labelled miss.
**Blind to:** oracles not written here. "Class" means no oracle written here reaches the filed rung, not a proof.

## Cost, inputs and platform

- Time (from E184's logs: a greedy step ~8 s, a 30-step rank-one fit ~3 s on an L40; an H100 ~2-2.5x faster):
  stage 1 ~45 min on an L40, ~20 min on an H100; stage 2 (expected: 3 controls and 2 items) ~60 min L40, ~25-30 min H100.
  H100 pod: about 1 h including set-up, ~$3. Not gated (under $60 and 6 h).
- Inputs (as E184's runs 2a-2d): the twin states `built_control_sum/results/large/amend1_control_s{0,1}.pt` and the fact
  states `memorised_fact/results/large/fact_s{0,1}.pt` (267.7 MB each, 1.07 GB), E137's
  `reuse_variable_test/results/large/pieces.pt` (2.4 MB), all under `results/large` symlinks, so staged by hard link;
  upload 1.07 GB ÷ 10 MB/s ≈ 107 s. E184's committed `experiments/simplest_intervention/results/` (item JSONs, `pcs.pt`)
  and E173's, E154's, E157's, E168's result JSONs come with the repo.
- Platform: an H100 pod (over 30 min). The cluster debug QoS also works ($0, L40, `cluster/search_recovery.sbatch`
  through `fleet/resubmit`, exit 3 per 26.5-min slot, about four slots, ~2 h wall).
- No change to `src/commute/rank_read.py`; E184's frozen script is loaded, not edited, so nothing it calls changes.

## Pod-runner brief (draft, for the lead)

Run E188 once (cell P1.21, run 1 of 2) on one H100: sync the repo, stage the four fine-tune states and `pieces.pt` by
hard link into a path ending `results/large/` per experiment and replace each symlink on the pod with the real
directory, then `E120_PRED_SHA=<git show HEAD:experiments/search_recovery/results/predictions.json | sha256sum>
.venv/bin/python experiments/search_recovery/run.py --wall-min 600`, teeing the log. Stop rules are in code (exit 1):
report the STOP line to the lead at once. After stage 1's last item (`stage 1 done` in the log) download
`experiments/search_recovery/results/` (small JSONs), and again at the end; check sha256 of `summary.json` on both
sides, then terminate. Exit 3 cannot occur at `--wall-min 600`.

## Not run

- No oracle at rung 1 or 2 for the old matched pairs, always on, the table or the planted fact (not in the cell).
- No uncapped list oracle (more than 16 names); the push's share in the 64 directions stands in for it.
- No search from the oracle's names (it would not be a search reading).
- Run 2 of the cell is kept for a targeted repair if run 1's rankings name a cause the repaired sizes did not fix.
