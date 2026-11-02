# Digest: reuse-lead overnight (2026-10-30)

**Why I finished.** The plan's remaining work waits on your two decisions below. I ran one $0 check that neither
decision could change, Amendment 4. It found that the early "reuse" fine-tunes answer letter-swapped text through a
count of a that the swap no longer breaks, not through a separate count of b. Spend was $0.

## What ran
- **E55 Amendment 4** (MATS cluster job 44746, $0, 48 s). The question: do the early slow all-spurious fine-tunes
  answer the letter-swapped rows they trained on through a count of a or through a count of b? On swap-built rows
  where the two give different answers, both seeds at 1K steps answer `count(b)` on 0.1% of rows, against 91% for the
  model that counts b. They give `count(a) + 1` on about two thirds. A separate count-of-b route is ruled out.
  - That the count of a was strengthened rests on behaviour alone: 67% right against the base model's 35%. The patch
    check does not single these states out.
  - The 80% state at 1K reads neither, so filed prediction P2 missed.
  - The code reviewer and an independent checker were both fresh agents. The scored rule was amended before the run,
    and the analysis was corrected after the check.
  - Report: `reports/analyses/2026-10-29_analysis_e55-stage-a.md`, section Amendment 4. Notebook
    `notebooks/E55_pcfg_anchors.ipynb`. Reviews `reports/reviews/2026-10-29_review_e55-amend4-{code,analysis}.md`.
  - Log card E55, log v131.

## Decisions waiting on you
1. **decision-e55-plan** (`reports/decisions/2026-10-29_decision_e55-plan.md`). Under Jain's Def 2 the cipher and
   copy fine-tunes count as reuse of next-word prediction. Keep Def 2 as written, or require g to generalise to
   held-out values? I recommend the second, as a dated departure.
2. **decision-e55-truth-set** (`reports/decisions/2026-10-29_decision_e55-stage-c-truth-set.md`). Only two fine-tunes
   clearly reuse the old count. Your options:
   - **A:** score the 11 labelled states as filed.
   - **B:** add an unscored class for the rule of two counts.
   - **C:** first add early-step seeds and an identity-prompt fine-tune, on natural rows only ($0, about 1 h).

   I recommend C, then A with B. Amendment 4 strengthens the early anchors' label, as an existing count made stronger,
   but does not decide this.

## Money, agents, friction
- Spend: $0 this session. No pods, no live Slurm jobs, no live subagents.
- Agents waiting at a prompt (`blocked.txt`): qwen-decomp-lead since 19:10, baseline-lead since 19:05,
  regating-lead since 19:50.
- `iteration-start` lists E45 as lacking a notebook. E45 is not on the reuse line.
- Friction noted: 209, 210 and 211.
  - `git commit -- <paths>` needs `git add` for new files.
  - jq needs `.["label"]` for the key `label`.
  - The cpu-run form for agents is `uv run --no-sync python`.
  - Dry-run scripts should record their PYTHONPATH.
  - The Artifact tool refuses worktree paths, so a publish has to copy files under `~/commute-data`.
  - The README's url-only read disagrees with lead.md's read with paths.
