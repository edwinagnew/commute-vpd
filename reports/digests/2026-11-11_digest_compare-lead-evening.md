# compare-lead, 2026-11-11 (evening)

The three cells in tonight's brief have both their zero-run pages written and their first runs done. The part 2 gate page is
filled, the share-stability check is filled, and the first run on the Qwen updates is done. The second Qwen run is
ready but waits on one question for the user: its random floor could not be repeated within the filed tolerance.

## Cells
| Cell | Due | State | Runs |
|---|---|---|---|
| P2.22 part 2 gate page (cell A) | 11-13 | filled | 0/0 |
| P2.6 share stability (E161) | 11-12 | filled | 1/1 |
| P3.6 cheap readings on Qwen updates (cell B) | 11-13 | open, waits on ask 18 | 1/2 |

Every other compare-lead cell is filled or closed. Ladder: L2.1 and L2.3 answered; L2.2 waits on L1.3; L2.4 waits on L1.5.

## What ran and what it found
- **P2.6 / E161** (job 59106, debug, $0): the share on the code-text fine-tune that learned nothing holds at about 0.8
  across five row batches (spread about 0.03). A random update matched in size gives no readable share, so size alone
  does not produce the reading. `reports/analyses/2026-11-11_analysis_p2-6-share-stability.md`.
- **P2.22**: the gate page, with a per-method proposal for what goes to Qwen.
  `reports/analyses/2026-11-11_analysis_part2-gate-page.md`.
- **P3.6 run 1 / E162** (job 59127, debug, $0): all 19 saved Qwen updates have a lower effective rank than a random
  update of matched size. The unchanged model reads overlap 1.0 as expected. Two draws of the random floor differ by
  about twice the filed tolerance. `experiments/qwen_cheap_readings/results/run1_stage{0,1,2}.json`.

## Waiting on you
- **Ask 18**, P3.6, due 11-13: run 1's two random-floor draws of the top-neuron overlap differ by about twice the filed
  tolerance (7.5e-5 against 3.6e-5 allowed at the smallest update, 0.0023 against 0.0015 at the largest). Options:
  (a) run 2 with four floor draws and a band set from their measured spread (recommended, about 1 h on the cluster, $0);
  (b) run 2 as filed, with the miss stated as a limit; (c) close the cell on run 1's rank column. Card:
  `reports/decisions/2026-11-11_decision_qwen-overlap-floor.md` (log v325). Run 2's code, sbatch and draft predictions
  are committed, so it starts within minutes of an answer.

## Spend and friction
Spend this session: $0 (cluster only). The dev-node check lists several unclaimed RunPod and Vast machines; none were
started by this clone. Friction noted: 728 (a worktree's first cpu-run of a plot printed nothing), 734 (a background
wait-job nested inside a background shell returned at once with no notice).

## Not run
P3.6 run 2 (waits on ask 18). A random update that lowers the code text as much as the fine-tune (P2.6, outside the cell).
