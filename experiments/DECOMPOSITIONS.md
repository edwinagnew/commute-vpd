# Which Qwen result was read on which decomposition

Qwen3-1.7B has two VPD decompositions of the same unchanged model.

| Short name | Run id | Files | Trained on | Notes |
|---|---|---|---|---|
| **old** | p-e9fcffd5 | `artifacts/qwen3-1.7b-p-e9fcffd5-torch` | The text mix firestock3, windows of 512 tokens | Every Qwen reading before 2026-11-07 |
| **new** | p-9707d6ff | `~/commute-data/artifacts/qwen3-1.7b-p-9707d6ff-torch` | Chat-formatted text, windows of 2,048 tokens, 30% written by Qwen3-1.7B itself | Port checked by E131 (`reports/analyses/2026-11-07_analysis_e131-new-decomposition.md`) |

**Rules for every lead** (written for the user's 2026-11-09 request for clear old-versus-new records).
- A Qwen reading of subcomponents, masks, a split, faithfulness or the adversary is true of one decomposition. Name it
  when citing: "on the old decomposition" or "on the new decomposition". A reading with no name here is on the old one.
- Before citing a row below, read its status. "Not re-run" means the claim still rests on old only; do not say it holds
  for new.
- Updates themselves (the saved δW from SFT, GRPO, fine-tunes) depend on no decomposition and carry over unchanged.
  Readings of behaviour, and comparators on neurons, carry over too.
- These are built from old and must not be reused on new: E76 and E111 teacher offsets, E57 mask caches,
  `qwen_sft/nowarmup/gstats` and `sources`, `qwen_d3_rows` alive nulls and counts, and any file a stage-0 control
  compares with a stored old number.
- New results go in files whose `decomp` field holds the run id, and report titles say "new decomposition".

## Status of the re-runs

Plan: `reports/plans/2026-11-09_plan_new-decomposition-reruns.md`. `decisions.md` lines are as of 2026-11-09. The last
column is filled in when an amendment is read: same, differs (one line how), or unreadable.

| Read on old | What it fed | Amendment on new | Status | New vs old |
|---|---|---|---|---|
| E93, Amendment 1: least squares and adjoint on the clean table | decisions.md lines 156, 162: D1's criterion, least squares first | E93 new-decomposition amendment (with E86's least-squares rows, E75's wrong-change rows) | Not re-run | |
| E111 stage 1: teacher training repairs RL seed 70 | lines 162, 167: the adopted map | E111 stage 1, then stages 2-5, on new | Not re-run | |
| E68 | line 140: D1's test left masked forwards | E68 new-decomposition amendment | Not re-run | |
| E72 | line 140 | E72 new-decomposition amendment | Not re-run | |
| E74 | line 140 | E74 new-decomposition amendment | Not re-run | |
| E80 (with E40's do-nothing reads) | line 156: adversary as a stated limit | E80 new-decomposition amendment | Not re-run | |
| E85 | line 156 | E85 new-decomposition amendment | Not re-run | |
| E89 | line 156 | E89 new-decomposition amendment | Not re-run | |
| E94, Amendment 1 | lines 157, 163: D3 readings follow task and format | E94 new-decomposition amendment | Not re-run | |
| E104, Amendment 1 | line 163: RL reads like its null | E104 new-decomposition amendment | Not re-run | |
| E107 Amendment 1 | supporting | E107 new-decomposition amendment | Not re-run | |
| E57 route split | line 178: r at least 0.99 on every Qwen update | E57 new-decomposition amendment | Not re-run | |
| E44 Amendments 1/1b, E76, E79, E77, E61 gamma_parts and Job S | supporting | Only if a top-tier result moves | Not planned yet | |
| E44 main and Amendment 2, E82, E40 stage G | no decision rests on them, or already read on new by E131 | None | Skipped on purpose | |
| E118 (baseline-lead), `qwen_rl_reads/plot_d2.py` (qwen-rl-lead) | joins of the above | Rebuilt by their owners after E93, E94, E104 on new | Waiting | |
