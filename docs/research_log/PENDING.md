# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
- 2026-11-04 · reports/plans/2026-11-04_plan_e84-rephrasings.md · plan; experiments card E84 · the user's choice after E81: TOFU's 800 rows trained in five wordings, tested on TOFU's held rephrasing; $0 on `debug`, about 1 h; coded and filed, not run.
- 2026-11-04 · experiments/TRIED.md · experiments card E84 · new "Coded, not run" row for E84, `experiments/qwen_tofu_forms/`.
- 2026-11-04 · reports/plans/2026-11-04_plan_e85-adversary-teeth.md · plan; experiments card E85 under D1 · E85 coded and filed, running: can VPD's adversary fail a bad state on Qwen3-1.7B's first update? ~$10.6.
- 2026-11-04 · reports/decisions/2026-11-04_decision_d1-after-e80.md, reports/decisions/2026-11-04_decision_e80-d1-test.md, reports/plans/2026-11-04_plan_e79-agreement-or-softness.md · link from decision-after-e76 and E79/E80 cards · Fable's advice on D1 after E80, and two files build.py lists as missing.
- 2026-11-04 · reports/plans/2026-11-04_plan_e86-update-fractions.md · plan; experiments card E86 · fractions of the update and the clean table on TOFU and gold, to find an update "behaviour kept" can grade; $0 on `debug`, about 26 min per update; coded and filed, running.
- E84 ran on a pod (RunPod H100): filed rule 'not answered', rephrased recall 0.532 under a line of 0.602, trained 0.993, floor 0.211, |dW| 18.1; dW on the cluster; blind fact judging running. Plan reports/plans/2026-11-04_plan_e84-rephrasings.md.
- E84 blind fact judging: both tries missed their controls (W0 0.28, then 0.19, limit 0.15), neither read, no third. Amendment 1 (10 epochs) per the tree.
- E84 Amendment 1 (10 epochs, RunPod H100, about $5.40): filed rule 'not answered' again, para 0.554, line 0.602, trained 1.000, floor 0.205, |dW| 24.8; key recall para 0.402, flat from 0.404. Plan's last amendment.
- New pending-decision card decision-anchor2-after-e84, Blocking, owner qwen-rl-lead: what stands as anchor 2 (ask filed, urgent). Options: the 5-epoch update, labelled as memorised facts carrying partly, D3 reads today, $0; or synthetic biographies, about a day and a few dollars, no assurance. Recommendation: the 5-epoch update.
- 2026-11-04 · experiments/qwen_d1_teeth/results/results.json, experiments/qwen_d1_fractions/results/ · E85, E86 cards; decision-after-e76 rewritten as the ask just filed · E85: the adversary passes both bad states (random 1.01, rarely-on 1.27); E86: TOFU's gain builds (kept 0.28 at a tenth), gold's barely, the first update's not. Reports and notebooks to follow. Provisional, unchecked.
- 2026-11-04 · reports/analyses/2026-11-04_analysis_e84-rephrasings.md, notebooks/E84_qwen_tofu_forms.ipynb · experiments card E84 · E84 and Amendment 1 written up: "answered" failed twice, rephrased recall 0.532 and 0.554 under a line of 0.602, trained wording 0.993 and 1.000, floor 0.211 and 0.205; no cause named for the gap; W0 never read on the H100, its two readings are one L40 file over 800 and 787 rows; blocked on anchor 2, card decision-anchor2-after-e84.
