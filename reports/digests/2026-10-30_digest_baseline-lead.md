# Baseline line, 2026-10-30 midday: E54 Amendment 1 done; next step waits on your approval

Steering does not tell the elicitation map from the cipher even when it is measured in log-odds, the unit VPD's route
split uses. On that pair the route split's gap is at least about six times steering's (10.3 at the two seeds' means).
The approved work is finished, so this line stops here until you approve a next plan; a Fable draft is ready.

## Why I finished
The task ("continue as you were") covered E54 Amendment 1, and it is done. The next step is new work and needs your
approval: card "Approve E56?" in the log's Pending decisions (ask #6).

## What ran
- **E54 Amendment 1** (job 46070, MATS `debug`, $0, 24 s). It reads steering's share of each fine-tune's effect in
  log-odds on E54's six saved fine-tunes.
  - First check: the refactored code reproduced E54's saved numbers exactly on all six.
  - Results: all three filed predictions were met. Steering's share is about 0.25 on the elicitation map against
    0.22 and 0.18 on the matched cipher, which is no separation. The route split exceeds steering by about 0.5 on
    both seeds.
  - The ratio is stated as a floor because steering's gap (0.05) is barely above its seed spread (0.04).
  - Reviewed before the run, checked independently after, in the log as v140. Report:
    `reports/analyses/2026-10-30_analysis_e54-keydependent-elicit.md` (Amendment 1); notebook
    `notebooks/E54_keydependent_elicit.ipynb`.

## Your decision
- **Approve E56?** Draft: `reports/plans/2026-10-30_plan_baseline-next-draft.md` (Fable's advice, not approved). It
  runs neuron patching, a cheap baseline that reads each input, on the same six fine-tunes, then three more seeds of
  the pair. One $0 cluster experiment.
  - Why this first: it tests the sharpest objection, that the pair was built to defeat steering. It also tightens
    the ratio floor.
  - The alternatives in the draft:
    - Jain et al.'s readout check first.
    - Wait for Qwen. That is not ready: no route split exists on Qwen yet, and it is the other lead's line.

## Money
$0 RunPod this session; one cluster `debug` job.

## Friction noted
- 223: E54's saved readings mix units without labels.
- 224, 225: the log's publish README disagrees with the lead brief (glossary in the read; the url-only read).
- 227: `git commit -- <path>` needs `git add` for new files.
