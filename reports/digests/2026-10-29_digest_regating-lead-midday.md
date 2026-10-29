# Re-gating line, 2026-10-29 midday: stage 0 done, waiting on one decision

**Why I stopped.** Stage 0 of the re-gating reset has run in full, and what comes next is the user's call (log card
`decision-e46-stage1`, ask #4). Spend: $0. No pods were started.

## What ran and what it found

Stage 0 asked whether the gate gain, which is how much the target gains from the change in which subcomponents Γ
turns on, can tell a learned update from changes that should not read as re-gating. It ran on the 67M model, on the
MATS cluster, as Slurm jobs 42604 (0a), 42675 (0b) and 42791 (Amendment 2).

- The earlier E12 result came back: copy tasks 0.87-0.97 of their piece gain, random-map tasks 0.36-0.41 (P0.2 pass).
- A hand-written change that adds no new subcomponent reads 0.16 and 0.34 of a copy fine-tune (P0.4 pass). The pass
  is thin: it reads 48-65% of what the random-map fine-tunes read.
- Random changes sized to match the fine-tune's side effects move the gate gain by at most an eighth of the
  fine-tune's. That is roughly what they move the answer itself, so the reading follows the real effect (P0.3
  inconclusive). The first run's size cap was unsourced and cut the match short. Amendment 2 raised it.

Report: `reports/analyses/2026-10-29_analysis_e46-regating-table.md`. Notebook: `notebooks/E46_regating_table.ipynb`.
Log card E46 and research log v100. A fresh Opus agent checked the report independently. Its corrections are in.

## The decision the user must make

`decision-e46-stage1`: go to stage 1 on the Qwen line's updates, or first run one more $0 cluster job? That job
would read stage 1's own ratio (gated piece gain over piece gain after a least-squares split) on the random change
and the hand write, which have never been split. The other option is to retire the line. Fable recommends the extra
job first, then stage 1 only if that ratio stays at or below 0.4. The lead agrees.

## Plain disclosures

- **FYI, a correction.** Amendment 2 also softened the tree after 0b was seen: "P0.3 fails → retire" became "card
  first". The outcome did not use it. It is recorded in the plan's correction note and in the report.

## Friction noted

- `sacct` cannot reach slurmdb (127).
- cpu-run at `--mem 4G` gets "nodes busy" (129).
- A mid-flush push by another lead forced a hand merge of PENDING.md (clerk).

## Not run

- Stage 1.
- The split-null job.
- The red team's leak and causal-path tests.
