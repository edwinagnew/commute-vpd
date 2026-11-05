# E97: a locator for baseline-lead's reuse table (Fable's advice)

*Fable 5.1 for mechanism-detection-lead, 2026-11-05. Read: `.claude/skills/design/SKILL.md`, `GOALS.md`, `decisions.md` lines 81, 112, 114-117, 121, the E96 analysis and advice, the E90 table (`reports/analyses/2026-11-05_analysis_e90-old-labels.md` lines 58-83), the task inventory, `TRIED.md` rows E38, E52, E62, E66, E70, `src/commute/ctxpatch.py` `write_state` (line 303), `experiments/locate_mechanism/results/{stage2.out,predictions_stage2.json}`, `experiments/new_vs_reuse/results/stage2.json`. Nothing run, no code.*

**Limit on every claim.** A circuit built from old parts reads as reuse (`decisions.md` line 121).

## Q1. Is this the right locator, and what does each read say?

The three reads are the right first move, but they are a **nominator**, not the label. Under lines 115 and 121 a label comes only from removing a located mechanism of W₀'s capability C in the state, with a spared control on the same windows. Restoring P to W₀ (your difference patch) says "the gain needs P's *change*"; it says nothing about whether P did C in W₀, and a head repurposed for something new reads the same. So:

- **(A) gain map, difference patch.** Keep it; it is the fix for E96's artefact (W₀ is never touched, so W₀ rising cannot enter). Its own failure mode is **inheritance**: a part downstream of a write carries the write's change in its output, so (A) credits pipes as well as sources, and a sum over parts is not 1. It is exactly 0 on every part upstream of all nonzero δW, which is what makes the planted control exact (below). Sufficiency (W₀ with the state's P) is Prakash et al. 2024's cross-model patch, which E66 cited and departed from (`reports/analyses/2026-11-02_analysis_e66-interchange.md` line 271); report it beside, no rule.
- **(B) write map, one matrix's δW reverted.** Keep it, and add the head grain for free: revert δW on head h's slices of q, k, v (columns) and o (rows). It separates "written into P" from "passes through P", which is what (A) cannot do, and it is a VPD-free analogue of GOALS' direct against re-gating routes. Failure mode: interactions, so also revert all 24 at once (must give W₀ exactly, G = 0) and report the sum of singles as the non-additivity number.
- **(C) capability map on W₀.** Needed for the reuse reading, but on lowercase it rests on the unexplained gap, 0.25 on E96's rows against 1.18 at A0's positions (E96 "Surprises"). Run down that anomaly first (the two-minute check E96 names) or read (C) at A0's positions; do not build (C) on a number nobody can explain.
- **Edit level is (B); direction level is E90's value form**, which already reads "degree only" against matched random directions. Nothing here improves on that; stay at parts.

**Label against beside.** The label is E96's removal test in the state's own terms (Fable's rule shape in the E96 advice, Q2), run only on the parts (A) and (B) nominate plus the greedy pair, with W₀'s own movement reported as a separate number. (A), (B), (C) are the locator, reported beside with the nominated part named. Reuse evidence is then: the labelled part is also on (C)'s map and its removal costs W₀'s C ("shared", E96's capability read); "gain only" otherwise.

## Q2. Smallest task set

1. **Copy against lookup (T5, E62's six states, `L/induction_cue/`)**: the positive pole and the control, known answer (2,4), own-terms fall 0.56-0.65 against the lookup's rise. Runs first and alone; (A) must put (2,4) first on all three seeds and (B) must say where it was written, or stop.
2. **Planted edit, no E62 in it**: `ctxpatch.write_state` is a one-matrix δW in `h.3.mlp.down_proj` (line 312); it needs keys, so plant it on the 24-key copy table (`L/open_pair/amend1/windows_copy.pt`) and add a second plant in `h.1.mlp.down_proj` via `perturb.key_write` (E90's write), scaled to the state's gain by `perturb.match`. Expected, exact: (B) 1.0 at the edited matrix, 0.0 at the other 23; (A) 0.0 on every part upstream, 1.0 at the edited MLP, and whatever blocks 2-3 inherit from the block-1 plant, which is the measured inheritance floor for "gain only through a pipe". The hand write of the copy table must not read as sharing copy's mechanism: (2,4)'s removal must leave it within the lookup's band.
3. **Lowercase against cipher (T7/T8, E53's step-300 states)**: the known failure, expected "not found" at head grain; the new information is whether (A) and (B) put the gain in the block MLPs, where E96's post hoc 320 units took 0.45 of it in own terms.
4. Second pass only: **rank-1 against cipher at 24 keys** (`L/open_pair/amend1/rank1_s{0,1,2}.pt`, `L/open_pair/cipher_s{0,1,2}.pt`), the one lookup-shaped task with a W₀ capability that E90 reads "mixed". Constant-answer tasks (ladder, emoticon) stay "does not apply" under line 121; the locator may run on them with the same-windows null (`L/recruitment_test/states/*_null.pt`) as the spared control, reported beside the two behavioural numbers, never as a verdict.

Cost: E96's 24-head sweep took 3-4 s per E62 state (`experiments/locate_mechanism/results/run.out` lines 16-21) and 2.5 min for W₀ plus four E53 states at 500 x 127 tokens (`stage2.out` lines 12-16). (A) + (B) at 28 + 32 parts is about 2.5x a sweep: under 15 min for items 1-3 in one `debug` job, $0.

## Q3. Bars

- **Stage 0, exact, stop on a miss**: planted (B) reads 1.0 ± 1e-4 at the edit and |share| ≤ 1e-4 elsewhere (E96's `reread_tol`, floor measured at 4e-16); all 24 reverted gives G = 0 within E96's `zero_tol` 0.001; gains re-read E62's and E53's within 0.02 (`g_tol`). Zero update: G = 0, share undefined, the rule prints "no gain to locate" under `min_gain` 1. Do-nothing is the state itself here (E96 `expected_values.note`): not a separate row.
- **Nomination**: no bar; rank. Report the top three by (A) and by (B) and the median |share| of the rest (E96 read 0.015-0.021 on copy) as the floor.
- **Label**: a = the control's weakest seed's own-terms fall share at (2,4), read in this job's stage 1 and truncated to two decimals, written to the results before lowercase runs (the E-code rule: from the control in the same job, never guessed); b = 0.2 (E62's filed spared bar); task fall ≥ 2x control fall in nats (Fable's E96 rule). Edge cases: G ≤ 0 on either state → "unreadable: no gain"; a part whose removal raises the state → share clipped at 0, reported; a tie at the top of (A) names no part; a share past 1 counts as located; W₀'s probability of the target under 0.01 → W₀'s own move printed beside, never subtracted.

## Q4. Output

Yes, a table baseline-lead reads directly, one row per (task, seed), with E90's field names so its code reads it unchanged (`experiments/new_vs_reuse/results/stage2.json` `.table[]`: `task`, `C`, `share`, `control_share`, `band`, `verdict`, `reason`), plus `part` (the located part or "none at head/MLP grain"), `a_share`, `b_share`, `c_share` (the part on W₀'s capability map), `w0_move`, and `located` (bool). `verdict` takes only the definition's words: "reuses C", "not through C", "unreadable: <reason>", "does not apply". Nothing in the definition changes; "located" is apparatus (line 116).

## Recommendation

Run items 1-3 as one job, (A) + (B) at 24 heads + 4 block MLPs, the planted edits as the exact stage-0 control, the label test on nominated parts only, (C) deferred until the gap anomaly is read. File it as E97 with the table above as the deliverable.

## What would change it

- The running independent check of E96 faults the sweep's apparatus (means, positions): fix that first, the locator inherits it.
- (A) does not rank (2,4) first on copy: the difference patch reads propagation, not mechanism; fall back to (B) alone with a direct-effect read (per-part output difference projected on the answer's unembedding, exact up to the final norm).
- The block-1 plant's inheritance share at blocks 2-3 exceeds the smallest nominated share on lowercase: head/MLP grain cannot nominate there; stop, and report "not found at this grain" rather than go finer.
- The user wants labels on constant-answer tasks: that reverses line 121 and is a decision card, not an amendment.
