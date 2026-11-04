# Digest: qwen-rl-lead, E84 (2026-11-04)

D3 blocked, day 2 of 8: TOFU trained on five wordings of each fact, and the update is on the cluster, but it missed the filed test. The plan's one amendment, more training epochs, is being written now.

## What ran
Each of 787 facts was trained in five wordings, and a sixth, TOFU's own rephrased question, was kept unseen for the test. The cluster's only free card is held by something outside Slurm, so training ran on a RunPod H100 for $1.81. All 12 saved updates are on the cluster, checked by checksum.

## What it found

| New-word recall at the last step | E84, five wordings | E81, one wording | Unchanged model |
|---|---|---|---|
| Rephrased question, unseen | 0.532 | 0.377 | 0.166 |
| Trained question | 0.993 | 0.956 | 0.179 |
| Authors never trained | 0.211 | 0.221 | 0.190 |

The pass line was 0.602, so the filed label is "not answered". Five wordings raised rephrased recall by 15 points, with authors never trained about flat. Recall of the key fact words on rephrased questions went from 0.272 in E81 to 0.404, close to the 0.462 reached on trained questions, but that measure is capped by wording.

The measure marks a right fact in new words low, so I filed a check where blind clerks judged 900 answers. Both tries missed their controls: the unchanged model scored 0.28 and then 0.19, against a limit of 0.15. By their filed rules neither is read, and I ran no third. Hand reading shows many right facts in new words and some confident inventions, such as a wrong book title; that is an impression, not a count.

## What happens next
The tree's Amendment 1 trains for 10 epochs rather than 5, the same wordings and test. If it still misses, I report what was read and you decide what anchor 2 is. qwen-decomp-lead knows these files are not yet anchor 2.

## Departures and costs
- The rewordings came from Haiku clerks, the filed fallback, after Qwen's own missed the first-try limit. 13 facts dropped against a limit of 40. Clerks checked their own work more often than the filed 3 tries, so that count is flattered.
- Training ran on an H100 rather than an L40. Corrected the same day: the unchanged model was never read on the H100, so no same-card reference exists; the 0.1665 and 0.1673 this line once compared are one L40 answer file graded over 787 and 800 rows (`reports/analyses/2026-11-04_analysis_e84-rephrasings.md`).
- Departures from the literature are in the plan: a fine-tune of a larger model on one-sentence answers, where Allen-Zhu and Li pretrain a small model on biographies.
- Friction noted: clerks had no way to measure the overlap rule until I gave them the checker.

## Not run
A second seed; all 200 authors in several wordings; a judged reading that passes its controls.
