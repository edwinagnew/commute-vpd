# Overnight digest, 2026-10-27 (lead-2 agent, commute line)

**Teacher training copied a fine-tune with no trigger almost exactly. But the copy fails the adversarial check, so
it is not yet a faithful map for broad updates like post-training. A second run found that the check's reading
depends heavily on its random starting point. Two decisions wait on you before more runs on the commute line.**

The task was teacher training of E36's lowercase fine-tune, with two cheaper checks first: option 1 of
`reports/decisions/2026-10-26_decision_next-while-waiting.md`. All of it is done, checked by a fresh Opus agent, and
in research log v53: https://claude.ai/artifact/SvgkzUiESRkyE9GM65HCWu.

## What ran, and what it found

- **E37, copying a fine-tune with no trigger.** The fine-tune teaches the model lowercase text, a change at every
  position. Teacher training, retraining the subcomponents to imitate it, kept 0.998 of its held-out gain.
  Under the standard adversarial check the copy read 2.20 times the original model, above the fail line of 1.61. A
  second batch order read 1.88. The scaled adjoint split kept 0.80 of the gain and passed at 1.05. The copies were
  still learning at step 1,000, which the filed decision tree treats as a reason to train longer before reading the
  failure. Report `reports/analyses/2026-10-26_analysis_teacher-spread.md`, notebook `notebooks/E37_teacher_spread.ipynb`.
- **E34 Amendment 5, why two adversaries disagreed.** On the same rows, a fine-tune left in Δ read 1.6 from the
  check's random start and 9.2 from the other adversary's. Moved onto the check's start, the second adversary read
  1.7, so the start carries the gap. Rows, precision and more steps changed little. Separately, two copies trained on
  Pile text at different batch orders point in different directions, at a cosine of 0.43. So that copy is not one
  unique weight change, while copies trained on the task's own text agree at 0.91. Section "Amendment 5" of
  `reports/analyses/2026-10-23_analysis_teacher-training.md`, notebook `notebooks/E34_teacher_training.ipynb`.

## Reversed predictions (notified)

- E37: the copy was predicted to pass the adversarial check; it failed at 2.20.
- E37: the adjoint split was predicted to lose most of the gain; it kept 0.80.
- E34 Amendment 5: the original model was predicted to read within 10% from either start; at 160 steps the two starts
  differ by 11%. Two copies from Pile text were predicted to agree (cosine at least 0.8); they read 0.43.

## Decisions waiting on you

- **What next for the broad copy?** Blocking card `decision-broad-copy`. The cheapest option, and the filed tree's
  own branch, is to train the copy three times longer, about $2.
- **Should the adversarial check search from several random starts?** Soon card `decision-adversary-starts`. Its
  first option measures how much the readings vary over eight starts, about $2.

## Money and machines

$5.82 of the session's $40 on RunPod, all terminated; nothing is running.

## Problems

- A pod-runner sat at a permission prompt from about 20:38, so its pod billed idle for about 40 minutes after its run
  ended. Lead-2 downloaded the results, checked them and terminated the pod; the runner was stopped at 23:29.
- remote-kernels `download()` refuses paths through the `results/large` symlink; runs now download into a staging
  folder and move the files. Both lessons are in lead-2's handoff note and belong in `CLAUDE.md`.
- `~/commute-data/blocked.txt` lists the lead agent at a permission prompt since 19:35. No lead-2 agent tried to reach
  anything outside the clone, `commute-data`, Slurm or remote-kernels.

## Not run

The longer copy, the spread over random starts, the frozen-Γ test, a second teacher seed, and the larger model.
