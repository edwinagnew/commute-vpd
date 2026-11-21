# Plan: why the thin new edit stands in under wording A (E247, P1.47)

*2026-11-21, definition-lead (writer agent). Cell P1.47, one run, due 11-22. Card
`reports/plans/2026-11-21_plan_thin-edit-diagnosis-card.md`; rules `experiments/thin_edit_diagnosis/results/predictions.json`;
code `experiments/thin_edit_diagnosis/run.py` (calls `src/commute/stand_in.py`, which gains `write_change`, and E242's
`World` and `controls` unchanged); Slurm `cluster/thin_edit_diagnosis.sbatch`.*

**The request, in the Terms' words** (user, 2026-11-21 ~13:20, ask 57 option (b)). E242 read wording A at the VPD
featurization: a set S of the circuit K*'s variables "stands in" when each takes M's mechanism times one positive
least-squares output gain and KL(F || F_S) <= 0.1 KL(F || M). The thin new edit (new_thin, a least-norm change of layer
0's down_proj that writes eight new answers) read R_A 0.98, above two reuse edits. The user asks why: do the gains
absorb the new computation, or is the built label off, or is wording A too lenient for thin edits. Wording A is not
changed here; a reworded A, if the result calls for one, is drafted for the user to decide.

## Design check

- **Goal.** Part 1 of the paper (a working definition of reused against new): it tests whether wording A's operations
  read the built labels for the reason the definition gives, before 50 L40-hours of pairs run under it.
- **Nearest tried.** E242 (the failure itself), E245 (the same at SAE latents, where new_thin reads 0.57 and the order
  holds), E238 (weighting the old R by behaviour weight: failed worse). None read the stand-in with the gains removed, or
  where the edit's change sits among K*'s parts.
- **Patch or step.** A diagnosis of the last experiment, asked for by the user; it decides whether A or the label moves.
- **Decisions relied on.** Wording A and the relative bar (`decisions.md`, 2026-11-20); E226's K* and split; E242's
  saved sets. No departure; no rule is added to a verdict (P1 diagnoses the reading, it is not a reuse verdict).
- **What would change what we do.** Rival a (gains carry the new computation): wording (c) restricts the gain.
  Rival b (the change sits in the few parts that keep F's mechanism): the count of parts is the lenient step, and
  wording (c) weighs parts by the change they carry. Neither: the label question goes to the user as read.
- **Control first.** The do-nothing, alone, with exact-zero checks; then on each item E242's own controls and a
  reproduction of E242's stood-in KL to 1e-4 relative, so the sets and gains are E242's.
- **Saved for audit.** Per variable num, den and write change (`results/large/items_<item>.npz`, ~0.3 MB each).

## Already read at $0 (E242's and E245's saved arrays; scratch `p147-writer/zero_cost*.py`)

- At VPD every built edit changes one to three whole matrices (new_thin: all 3,585 variables of h.0 down_proj). K* for
  new_thin holds 9,601 variables, of which only 1,068 changed: 89% of R_A's denominator stands in for free. Of the 1,068,
  875 stand in and 193 keep F's mechanism; 2,517 changed variables lie outside K* and are never asked.
- new_thin's stood-in gains sit close to 1 (|g - 1| median 0.018, 90% under 0.051), no further from 1 than its kept ones
  (median 0.025): a gain near 1 cannot carry much, which already leans to rival b.
- All 1,068 standable members stood in at once move F by 0.62 of KL(F || M); keeping 193 brings it under 0.1. The
  remainders carry nothing: E226's outside share of new_thin's update is 2e-12.
- At SAE (E245): K* 8,954 latents, 6,650 changed (74%, against 11% at VPD); 2,809 stand in (42% of the changed, against
  82% at VPD), with gains further from 1 (median |g - 1| 0.17); the eight SAE remainders stood in alone move F by 0.21
  of KL(F || M). The order holding at SAE fits the dilution story: fewer free unchanged members, and a remainder that
  holds what the dictionary lacks. Gains-to-1 at SAE is sae-lead's to run, not this cell's (proposed in the result page).

## What runs, and what each number should read

One debug-QoS job. Stage 0: do-nothing. Then random (size), random (KL), x10, x2, rebuilt, new_few, new_thin; x10 and
new_thin also on E226's two halves. Per item: q_fit (must equal E242's), q_one (every gain 1), q_out; e_stood (share of
S's write change its gains explain), c_kept (share of K*'s changed write change in the kept members, beside the count
share), c_out, c_rem (definitions in predictions.json "readings").

| item | q_one - q_fit | e_stood | c_kept (count share) |
|---|---|---|---|
| do-nothing | not defined; gains exactly 1, write change exactly 0, all stood in KL exactly 0 | | |
| random (both) | under 0.02 | under 0.05 | near 0.71 / 0.70 |
| x2 | q_one above 0.5 | | (5 of 2,813 kept) |
| x10 | above 0.11, not filed | above random's | |
| new_thin, rival a | above 0.11 | at least 0.5 | |
| new_thin, rival b (expected) | at most 0.11 | under 0.2 | at least 0.5 (0.18) |

**Filed rule P1** (new_thin, all of D): rival a if rise > 0.11 and e_stood >= 0.5; rival b if rise <= 0.11 and
c_kept >= 0.5; else neither. 0.11 = two spreads of the proxy (0.618 vs 0.696 on new_thin's halves; card). R0 stops the
job on any control miss.

**If rival b, a draft wording (c) for the user** (not adopted, not used): "A fine-tune F reuses M, for a behavior, to the
extent that the change F makes to the variables the behavior runs through lies in variables whose M mechanisms, each up
to an output gain, can stand in for F's and still preserve the behavior." The share would weigh each variable by its
write change instead of counting it, so unchanged variables add nothing and a thin edit's few carrying parts dominate.
If rival a, the draft instead bounds the gain ("each up to an output gain the behavior does not depend on").

## Cost and time

$0, cluster debug L40, one job of about 15 min: world load 1.7 min (E242 jobs 62708/62709, wall time minus item
times); per item about 15 s of featurizing and controls (E242's x2 item: 17 s with 43 evaluations at 0.12 s), one
write-change pass (about twice E242's gain pass), and 4 to 8 evaluations at 0.07 to 0.51 s (E242 controls
seconds_per_eval); the 24-matrix random items' split up to about 1 min each (not measured alone). Stage 0 projects the
evaluations and stops over 1.5 h. Host memory: E242 peaked at 6.1 GB, the job asks 16 GB.

## Not run

Gains-to-1 and write-change shares at SAE latents (sae-lead's code); a new K* or stand-in search; a fractional stand-in;
behaviour-weighted shares (E238 tried a weighting on the old R); moved (under the floor in E242); the held pairs, E62 and
key tasks.
