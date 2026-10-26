# Overnight digest, 2026-10-26 (lead agent)

**The re-gating line now has a form of the fixed-shift hypothesis that names no cue: inputs the original Γ treats
alike get alike changes. It holds on the toy tasks and, with enough fitted positions, recovers three-quarters or more
of Γ's switching on a fine-tune with no trigger. One decision waits on you: whether to restate the hypothesis this way.**

Task (`~/commute-data/tonight`): understand re-gating well enough to restate the fixed-shift hypothesis for less
synthetic tasks, without depending on cues or triggers. Plan: `reports/plans/2026-10-26_plan_regating-without-cues.md`.
Research log v51 published, with v52 to follow this digest's last edits: https://claude.ai/artifact/SvgkzUiESRkyE9GM65HCWu.

## What ran, and what it found

- **E35, the toy tasks without labels.** Each input got the mean change of the 8 training inputs whose original
  importance values are nearest. On the four 32-cue tasks that recovered 0.61 to 0.78 of the switching, as much as one
  change per cue word added the same way (0.47 to 0.66). Those neighbours share the cue 62 to 71% of the time, against
  3% by chance, and restricting them to the cue helps only where each cue has its own answer. Clusters and a fixed line
  per subcomponent fall short. Report: `reports/analyses/2026-10-26_analysis_regating-without-cues.md`.
- **E36, a fine-tune with no trigger.** The planned code-text fine-tune did not learn (held-out text got worse), so a
  lowercase-text style was used. Switching carried 0.79 to 1.05 of its gain, though an update that learned nothing
  also gave about 0.8, so that share alone proves little. The same nearest rule, fitted on 16,384 positions, recovered
  0.72, 0.54 and 0.51 of the switching at steps 10, 100 and 300, under the filed 0.6. Fitted on all 262,144 positions
  it recovered 0.89, 0.78 and 0.76. One change per current token gets 0.24 to 0.32, and the token adds nothing to the
  nearest rule. Report: `reports/analyses/2026-10-26_analysis_regating-spread.md`, Amendments 1 to 3.
- Every analysis went through an independent Opus check, which found the numbers right each time and the first
  headlines too strong; the reports were reworded each time.

## Reversed predictions (notified)

- The nearest inputs share the cue only 62 to 71% of the time; predicted at least 90% (E35).
- A fixed line per subcomponent collapsed to zero on the lowercase fine-tune; predicted to beat one change by 0.1 (E36).
- With all 262,144 positions the nearest rule cleared 0.7 at every step; predicted to stay under it at step 300 (E36,
  Amendment 3). This one runs in the hypothesis's favour.

## Decision waiting on you

**Restate the fixed-shift hypothesis without cues?** (Blocking card in the log.) (a) Restate now, tentatively, as "positions
the original Γ treats alike get alike changes", and rename "fixed shift"; (b) keep "per cue" until a fine-tune of a
real behaviour that learns, or the larger model's post-trained sibling, confirms it; (c) restate, counting only the
toy tasks for now. A second card, from another session, proposes teacher training of a no-trigger fine-tune on the
commute line; I left it to its author.

## Money and machines

About $1.10 of the session's $40 on RunPod: eight pods, all terminated. Nothing is running.

## Problems

- **The cluster's only GPU node, `l40-worker`, has been draining since 01:35** ("Kill task failed"). Nothing can
  start on Slurm, `fleet/cpu-run` is down, so no lint or tests ran on the dev node; every run and the log builds went to
  RunPod. It needs an admin.
- **W&B refused the dev node's `WANDB_API_KEY` for `goodfire/spd`** (403) on the first pod. You said it is fixed; the
  remote-kernels plugin probably needs a `/mcp` reconnect to see the new value. Meanwhile pods are sent the cluster's
  own copy of the decomposition files, and `src/commute/vpd_files.py` now keeps files already in place without asking
  W&B.
- `~/commute-data/blocked.txt` is empty. No agent tried to reach anything outside the clone, `commute-data`, Slurm or
  remote-kernels.

## Not done

Notebooks for E35 and E36; the write-up checker flagged about 40 dense sentences per report, mostly in headers and
tables, not yet tidied; lint on the new files (the worker is down); a rule on Γ's input or neighbouring positions for
the remaining tenth to quarter of the switching, which is a new design for you to decide on.
