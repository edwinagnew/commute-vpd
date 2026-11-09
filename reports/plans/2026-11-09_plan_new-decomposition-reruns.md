# Plan 2026-11-09: re-running the decision-bearing Qwen experiments on the new decomposition

**The request, in the Terms' words (user, task file 2026-11-09).** Every Qwen3-1.7B experiment whose reading of a VPD
decomposition influenced a major decision is repeated on the second decomposition (p-9707d6ff, "new"), as amendments of
the original experiments, each by its own subagent, without overlapping qwen-rl-lead, and with clear records of which
result holds for which decomposition.

## Design check
- **Goal served.** `GOALS.md`'s question is read through a decomposition; E131 found the new one about seven times
  harder for the adversary to break on GSM8K. Decisions taken on the old one may be artefacts of it.
- **Nearest tried.** E131 compared the two decompositions on text, not on any update. Nothing has read an update on new.
- **Patch or step.** A replication the user asked for, not a new direction. No new measurement is added.
- **Decisions relied on or departed from.** Departs from one rule: "two amendments per E-code" (`.claude/CLAUDE.md`).
  The user asked for amendments, so these re-runs do not count towards the cap (`decisions.md` row dated 2026-11-09).
- **Adopted definition.** Reuse stays `decisions.md` rows 81 and 87, unchanged. A case they cannot read is reported
  "unreadable" on new as on old. Every verdict-giving report quotes them.
- **What would change what we do.**
  - D1's test (lines 140, 156): if an update passes under new masks while do-nothing fails, or the adversary fails
    do-nothing but passes the in-span map on maths, D1 goes back to VPD's own checks. That change would be a user row.
  - The first map (line 162): a flip on RL arms, the template or TOFU changes the map verdict and E111's order.
  - D3 (E94, E104): a row that separates the anchors, or RL from its null, beyond seed spread changes the answer.
  - Re-gating (line 178): r that separates updates on new reopens that line on Qwen; r near 1 closes it for both.
- **Control first.** Each amendment's first stage reproduces its W₀ and do-nothing reference on new, alone, with a
  stop rule. Old stored numbers are not references: each amendment files new-decomposition stage-0 references.
- **Numbers.** Each amendment reports the numbers its original reported, and the same read on old beside it. No more.

## Order (each line one amendment, one subagent, its own predictions filed before its run)
0. **Foundation, no results.** (a) Code: explicit decomposition everywhere, `move_gamma` fixed, result files stamped
   (`~/commute-data/scratch/qwen-decomp-lead/code-audit/audit.md` §5). (b) Shared W₀ references on new: g statistics,
   alive set and count, adversary sources on `qwen_sft/results/maths_rows.json`.
1. E93 on new: least squares and adjoint on the 11 states, with E86's least-squares rows and E75's wrong-change rows.
2. E111 stage 1 on new, then stages 2-5. Needs 1. Paid A100, about $10-15 (ask 36's estimate, plus about 1 h for
   stage 1). Needs the new decomposition on W&B or staged on the pod; `qwen_port/upload.py` must not be used for that,
   it writes the old artifact's name.
3-5. E68, then E72, then E74.
6-8. E80 (with E40's do-nothing reads), then E85, then E89. Note E85 and E89 read web text, where new against old is
   0.75 (E131 report), while maths is 0.14; E80's maths rows are where the answer can move.
9-11. E94, then E104, then E107 Amendment 1.
12. E57 route split; E61 Job S only if E57 separates updates.

Second tier, only if a top-tier result moves: E44 Amendments 1/1b, E76 then E79, E77, E61 gamma_parts. Skipped on
purpose: E44 main and Amendment 2 (E68 on new asks the same), E82, E40 stage G (read on new by E131), and every
decomposition-free row.

**Compute.** Debug L40 first wherever a stage fits ($0; E131 took about 26 min per 128 rows of adversary, the only
timing on new). Paid: E111, and E80, E85 and E89 if they do not fit debug (those cost about $34 on old). Under the
session budget and under the gate per run; a run that crosses the gate takes the gated route.

**Owners.** Every line above is qwen-decomp-lead's. E75 is taken here unless qwen-rl-lead objects (note in its folder).
qwen-rl-lead has its own list to make and rebuilds `plot_d2.py`; baseline-lead rebuilds E118.

## Bookkeeping
- `experiments/DECOMPOSITIONS.md` is the registry: which reading is on which decomposition, status per row, and what
  must not be reused. Updated in the commit that reads each amendment.
- Result files carry `decomp`; reports and notebooks name the decomposition in the title and the first sentence.
- `TRIED.md` gets one row per amendment. E40 and E111 have no row of their own, and the E44 and E45 rows still read
  "Planned"/"Not run": fixed by the clerk of the next log flush.
- Notes to every lead that reads Qwen results point at the registry.

## Not in this plan
Choosing a new map or test, or any definition change. If a re-run moves a decision, the reading is reported and the
decision goes to the user.
