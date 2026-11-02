# Reuse line digest, 2026-10-30/31 (reuse-lead)

**Why I finished:** everything left on the reuse line waits on one decision only you can make: whether VPD is
run on the counting truth set (C2), and where (log card "E55 C2: run VPD on the counting truth set, and where?",
`reports/decisions/2026-10-31_decision_e55-c2-platform.md`; ask #5).

In plain words: the cheap methods (pruning revival, reverse fine-tuning, steering) label all 21 counting-task states
correctly, but so do trivial signals, so that set cannot rank VPD; on the 67M open-pair maps the two cheap methods
we ported never separate the old-by-construction map from the new one, at any training length tried. Running VPD
on the counting set costs ~50 L40-hours at a reduced size and can at best tie.

## What ran ($0 throughout, all on the cluster's debug QoS; no pods)
- **E55 C1b** (`reports/analyses/2026-10-31_analysis_e55-c1b.md`): pruning, reverse fine-tuning, steering 21/21
  and 12/12; but the label equals which task was trained, and update size within a recipe also separates it.
- **E55 C3** (`reports/analyses/2026-10-31_analysis_e55-c3.md`): on E58's 12 open-pair states (8 seeds/map),
  pruning revival and reverse fine-tuning fail their positive control (no rank-1 vs cipher gap); VPD's readings
  read "distance and prior" under the filed rules.
- **E55 C3 ladder** (`reports/analyses/2026-10-31_analysis_e55-c3-ladder.md`): neither instrument passes at
  10-300 steps or K up to 320. **A filed prediction reversed:** at step 30 pruning revives the cipher more than
  rank-1, beyond its bar. C3's step-10 cipher was only half learned, a confound for C3 (VPD included).
- **C2 rehearsals** (jobs 46971, 47013, 47071, 47100): recipe size does not fit an L40; 14,592 subcomponents run at
  0.44 s/step in 19 GiB; 70-75% of the task gain sits in the decomposed matrices. Resume still differs from an
  unbroken run (one cause fixed, 7ade9e1; residual from step 50).

Every report has an independent check and its fixes; the log is current (v156).

## Pending decision (yours)
- C2: (a) cluster `normal` QoS, $0, if you relax the Qwen-only rule; (b) H100 ~$90, over the cap; (d) drop C2
  and write up C1b + C3 + ladder as the benchmark. Lead recommends (a), else (d).

## Money
$0 spent this session; no RunPod machine was started.

## Friction noted (top items)
wait-job flags after job ids are read as ids (273); cpu-run edits invisible to git status on NFS (275); log publish
needs glossary/style/script read with paths (279); experiment helpers untestable without sys.path (280); the
"Qwen line only" clause is ambiguous for other lines (278). Full list: `~/commute-agents/fleet/friction reuse-lead`.
`~/commute-data/blocked.txt` shows baseline-lead waiting at a question since 15:15.
