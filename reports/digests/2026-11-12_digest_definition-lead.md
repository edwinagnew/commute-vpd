# definition-lead digest, 2026-11-12 afternoon

**P1.16 and P1.17 filled, due 11-13.** On the memorised fact the re-weighting share separates the ordinary fine-tunes
from their twin, but it reads only about two thirds of an exact planted re-weighting. So the tentative reading "new =
not re-weighted" labels every item on this layout new, and only the ranking within a pair carries information.

## Cells
- P1.16 filled, 1 of 2 runs (E168, $0, cluster debug L40, jobs 59525-59527, 59533-59535). Page:
  `reports/analyses/2026-11-12_analysis_memorised-fact.md`.
- P1.17 filled, 0 runs (added on your choice in the 2026-11-12 dialog): the gate page
  `reports/analyses/2026-11-10_analysis_gate-page-working-definition.md` has a section "Update 2026-11-12".
- Ladder: L1.4 has no open cell, and its run budget is spent. L1.5 has no open cell. L1.2 is the lowest open rung, but
  no run on it was asked for, and new runs wait on the gate.

## What ran and what it found
A fine-tune memorised a random yes/no bit per number token. Its twin pushed when the last number is odd, with the same
inputs, push and prior. Both fine-tunes have two seeds, and planted edits of each sit beside them on the same axes.
- **Re-weighting share, fine-tunes:** the memorised fact reads 0.10 and 0.12 against the twin's 0.39 and 0.40. This is
  the same ordering as all five old matched pairs.
- **Re-weighting share, planted edits:** the fact reads 0.48 against 0.63 for oddness. They sit within the 0.17 the
  filed rule needed, so the planted pair does not separate.
- **Labels:** the adopted wording calls the fact "no map found", and its twin too. The tentative reading calls the fact
  new, but it calls every item new.
- **The automatic premise check came back "doubtful".** Its reason: if the share cannot fully read an exact
  re-weighting, the gap between the fine-tunes may reflect the measure rather than the fine-tunes. It is listed for
  your gate; it changed nothing here.

## Asks
None open. Ask 36 was answered: refresh the gate page, writing only.

## Money
$0. RunPod had no H100 or A100 to start, Vast had no card with 40 GB or more, and the jobs ran on the cluster.

## Friction noted
- 746: seed 1 waits on seed 0's training length.
- 749: pod labels may not contain a colon, and a runner edited the local `remote-kernels.toml`.
- 754: a runner had to use `git pull --autostash` because of uncommitted memory files in the clone.
- 755: the Vast offer search ignored its GPU-memory filter.
