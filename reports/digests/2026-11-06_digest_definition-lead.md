# definition-lead digest, 2026-11-06 (afternoon: retarget steps 1 and 2)

**Definition, step 1/2: moved (done). Step 2/2: moved (written, handed over), a week before its 11-13 date.**

Under the frozen reuse definition (decisions.md rows 81, 87, 164 and E112's filed bars), no task we have, on the small
model or on Qwen, gets a "reuses" or "new" verdict. The small model's old tasks were measured: each either gives the
same answer every time ("does not apply") or names a capability that no readout trained on ordinary text recovers well
and clearly better than its control ("unreadable"). On Qwen the same is expected from the shape of each anchor, though
it has not been measured, so the D3 rows keep the labels they were built with and print the definition's reading
beside them.

## What ran
- **E116, step 1** (job 54440, cluster debug L40, about 20 min, $0). The 67M task table under the frozen definition:
  report `reports/analyses/2026-11-06_analysis_e116-definition-table.md`, notebook `notebooks/E116_definition_table.ipynb`,
  log card E116. E112's readout numbers reproduced exactly. Possession failed for every capability (token identity
  reads perfectly but so does its relabelled control; next-word-guess and copying readouts stay under 0.90). Carrying to
  held-out values, read without a readout, sits with the earlier claims on the copy tasks (0.95-1.00) and the ciphers
  (0), but rank-1 reads 0.64-0.69 where 0.90 was expected. Every earlier claim reads "cannot say"; each row says what a
  task needs for the definition to read it.
- A $0 CPU check of a suspected flaw in the copy readout (label leaking from the current token) found 10% leakage, too
  little to explain it; the reading stands as filed, no amendment.
- **Step 2**: `reports/decisions/2026-11-06_decision_definition-step2-qwen-anchors.md`, handed to qwen-decomp-lead
  (note in their folder). Template "does not apply"; invented facts, known facts and both maths pools "unreadable".

## Decisions for the user
- None blocking. The step-2 note's "What the user decides" lists anchor choices only (keep the built labels, or swap
  in arms the definition can read; add held-out real facts to the known-facts arm). A Qwen readout fit ($0, one debug
  job) would turn "expected unreadable" into a measured reading; not run unless asked.

## Money and friction
$0 (cluster debug only; no pods). Friction noted: worktree set-up steps missing from shell-notes, cpu-run output lost
when redirected, dead changelog anchors after card removal, worktree guard refusing loops.

---

## Earlier today (morning digest, kept as written)


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
