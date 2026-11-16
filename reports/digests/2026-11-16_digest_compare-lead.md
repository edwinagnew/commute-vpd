# compare-lead digest, 16 November (evening)

**Every compare-lead cell is filled or closed; P2.28, due 11-18, was filled tonight, and nothing waits on you.**
The part 2 gate page is current for the 18 November gate, and research log v352 carries it.

## Cells
P2.28 filled (1 of 1 runs). Every other part 2 cell was already filled or closed. The rungs: L2.1 and L2.3 are open with
every cell filled; L2.2 waits on L1.3; L2.4 waits on L1.5; L2.9 is parked.

## What ran and what it found
P2.28 (E187) shrank the larger side of each pair, by one number, to its partner's average update size, then read the
share and the cheap readings again. On the 67M almost nothing changed: the share still separates all nine built pairs,
and steering keeps its seven. On Qwen the share separates one of six pair-and-save cells instead of three, and the
neuron-overlap gap reverses on copy and cipher. So on Qwen no reading is yet shown to separate a pair beyond update size.
The premise check came back doubtful: shrinking a saved update is not the same as a model that learned less. That doubt
belongs on the gate list. Page: `reports/analyses/2026-11-16_analysis_p2-28-size-matched.md`. Figure:
`experiments/size_matched/figures/size_matched.png`. Gate page: `reports/analyses/2026-11-16_analysis_part2-gate-page-update.md`.

## Money
This session spent $13.66 of its $300: three Qwen H100 pods at about $4 to $5 each, and two small plotting pods. Earlier
sessions today spent $12.20, including the 67M half's $2.60. No pod is running.

## Friction noted
- 798 to 801: the start() label is limited to 64 characters, letters, digits, `-` and `_`. download() of a directory
  flattens it. execute() needs kernel_id.
- 804: a clerk ran build.py with python3 on the dev node and was not refused.

## Not run
Retraining smaller updates instead of rescaling, no steering floor for Qwen, the E187 figure is not on the log.
