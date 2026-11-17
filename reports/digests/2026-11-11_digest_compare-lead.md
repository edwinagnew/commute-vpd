# compare-lead, 2026-11-11 (day)

On the 67M model's nine built pairs, the re-gating share is the only single reading that puts the side built to reuse
higher on all nine, and its gap sizes follow neither the prior nor steering. On which side reads higher, though, the
prior and steering together reproduce every call, and new pairs that set those two against each other could not be
built. Whether the share or a cheap reading is right about reuse waits on scoring against part 1's labels (L2.2).

## Cells
| Cell | Due | State | Runs |
|---|---|---|---|
| P2.16 gap sizes | 11-12 | filled | 0/0 |
| P2.17 pairs where prior and steering disagree (E151) | 11-14 | closed, could not build | 2 jobs + 1 code-error rerun, all stage 0 |
| P2.18 top-neuron overlap (E155) | 11-13 | filled | 1/1 |
| P2.19 what the share adds | 11-13 | filled | 0/0 |
| P2.6 | 11-12 | parked by the user (L2.9) | 0/1 |

Ladder: L2.1 and L2.3 answered; L2.2 waits on L1.3; L2.4 waits on L1.5. Finished: nothing left that is not parked or waiting.

## What ran and what it found
- **P2.16** (saved results): share gaps 0.39 to 0.69 track neither the prior gaps (rank correlation 0.02) nor strength-1
  steering (-0.10), n=9. `reports/analyses/2026-11-11_analysis_p2-16-gap-sizes.md`.
- **P2.17 / E151** (jobs 58739, 58766 code error, 58833; debug, $0): every control passed each time; a cipher whose answers
  stay unrelated to the key could not reach the prior needed (best 0.000232 against 0.000235), so nothing was trained.
  Amendment chosen on Fable's advice, per the user's instruction that implementation details go to Fable.
- **P2.18 / E155** (job 58821, $0): fine-tunes keep 0.34 to 0.69 of their top-240 neurons against 0.84 to 0.95 for
  size-matched random updates; the reuse side keeps more on 2 of 7 pairs; the overlap's positive control failed.
  `reports/analyses/2026-11-11_analysis_e155-neuron-overlap.md`. A Qwen version was passed to qwen-lead as a note.
- **P2.19**: `reports/analyses/2026-11-11_analysis_p2-19-what-share-adds.md`.
- Premise checks on P2.17 and P2.18 read "doubtful" (advisory, for the 11-18 gate): which baseline the share follows
  does not say which is right about reuse without part 1's labels.

## Open
- No asks waiting. Research log v317 published (P2.17, P2.19); earlier lines went out in v310 and v316.
- **Flag:** `experiments/method_table/results/final.json` `.tally` disagrees with the P2.11 report's counts (steering 4 vs
  5 same; neurons differ too). Not cited in P2.19; needs a check before anyone quotes those counts.
- Spend: $0 (cluster debug only, no pods). Nothing in `blocked.txt`.

## Friction noted (694-708)
jq rewrite reformats filed predictions; worktree guard refuses heredoc appends and chained git; stage-0 construction
checks did not log which filter emptied the pool; debug queue wait reached 50 min once; `card check` usage in
shell-notes is wrong; a 0-run cell on a budget-0 rung needs the user; `numbers check` rejects scientific notation.

## Not run
P2.17's pairs (could not build); Qwen neuron overlap (qwen-lead's); effective rank on the 96-key refits (option c, not
taken); L2.2 scoring (waits on L1.3).
