# definition-lead digest, 2026-11-06

**Definition (feeds deliverable 3), day 4 of 8: finished. Every next step waits on you; the case-label line is paused (ask 6).**

Plain words: we reset the new-versus-existing definition to Jain et al.'s as adopted (decisions rows 81 and 87) and
built the first task pair with Jain's shape, a label after a trigger that depends on whether the previous word is
capitalised. The original model holds case in a linear readout, and the fine-tune carries perfectly to unseen words as
a threshold of that readout, but no causal check we ran (removal, a stronger removal, patching) shows the fine-tune's
answer running through that readout, so under the definition it reads "not through C". The definition's causal check
has never yet read "reuses" on the 67M; the copy/lookup pair was never read under it either (no possession readout).

## What ran (all $0, cluster debug L40; provisional, no independent check)

- **E112, case-label pair** (`reports/analyses/2026-11-06_analysis_e112-case-label.md`, `notebooks/E112_case_label.ipynb`):
  possession at block 1 (0.994 vs control 0.524); real carrying 0.998/0.996, small function 0.99; removal took
  0.08/0.09 of the gain (bar 0.50), Amendment 1's two-direction removal at every position 0.19/0.21, Amendment 2's full
  erasure hit its filed stop (a same-size random removal already took 0.19-0.26, and erasure doubled the original
  model's loss). Filed "reuses C" missed; twin "not through C" as filed. Jobs 54048, 54052, 54054, 54094, 54146.
- **E113, patching** (`reports/analyses/2026-11-06_analysis_e113-case-patch.md`, `notebooks/E113_case_patch.ipynb`):
  the patch moved the original model's block-2 case readout (0.71 flipped vs 0.11-0.13 random) but the fine-tuned label
  flipped on 0.002/0.014. Job 54194.
- **Summary with the copy/lookup pair**: `reports/analyses/2026-11-06_analysis_case-label-line-summary.md`.

## Decided today (decisions.md, dated 2026-11-06)

Definition = rows 81 and 87 only, E99's six rules dropped (ask 1 dropped); **adopted definitions are frozen**, guarded in
`.claude/CLAUDE.md` Rules and a `/design` question; E112 plan (ask 2), Amendments 1 and 2 (asks 3, 4), E113 (ask 5),
pause (ask 6).

## Pending decisions for you

What next on the definition line, none run without you: a learned interchange subspace (DAS) fitted on the original
model; subject number anyway; the route-split readings at this trigger; the stronger causal check you mentioned.
Log: https://claude.ai/artifact/SvgkzUiESRkyE9GM65HCWu (cards E112, E113).

## Money, friction

$0 (no pods). Friction 570-582: the drift guard (fixed), cpu-run queue waits, check.py exit codes and its "projection"
flag inside "nullspace projection", the worktree guard refusing chained nb-edit calls (12 calls for one notebook).
