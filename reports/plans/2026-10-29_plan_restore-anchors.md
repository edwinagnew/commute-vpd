# Plan: real-model restore on known-kind anchors (E50) — option A approved by the user 2026-10-29; B and C not approved

*2026-10-29, Fable, at the user's request ("how best to develop the regating line to address the question 'can VPD
distinguish elicitation from new mechanism'"). Advice only: nothing here is filed, approved or running. Words as in
`GOALS.md` → Terms. Read with `reports/plans/2026-10-28_plan_regating-line-reset.md` (the live plan; its Amendment 3 is
coded and in review, not yet run; Slurm 42963 is the commute line's E49) and `reports/reviews/2026-10-28_review_regating-line-red-team.md`.*

## In plain words

The line has shown that one number, read in VPD's stand-in model, tells a fine-tune of a behaviour the model already
had from one it lacked, and that this number mostly survives fake updates. It has not yet read the one number that
lives in the real fine-tuned model (E38's restore share) on any update whose kind is known, so the line's claim about
the model itself still rests on three single-trigger toys. The advice is to read that real-model number on the known-kind
anchors we already have, for free, and then carry whichever numbers separate them to the Qwen updates, stating up front
that a new circuit built from old pieces will read as "elicitation" and that this is a limit of VPD's unit, not a failure.

## 1. The question, sharp enough to fail

**The question.** A fine-tune teaches behaviour B. In the real fine-tuned model, does the gain on B run through the
pieces the original model already uses for B, or through weight directions that are new to the decomposition (Δ, or
pieces Γ never gates on for B)?

**The terms.** *W₀'s machinery for B*, M₀(B): the alive pieces Γ gates on when the original model already does B
(on Qwen: the prompted inputs; on the 67M: Pile contexts where B's answer follows, E38's census). It exists only if W₀
does B somewhere; when W₀ never does B, B is new by definition and the question becomes where its gain lives.
*The gain*: the change in B's log-odds (toys) or in KL to the fine-tuned model on B's rows (Qwen), at the scored
positions. *Elicits*: putting M₀(B)'s outputs back to W₀'s values in the fine-tuned model (E38's restore) takes back most
of the gain, and little lives outside the alive pieces. *Builds new*: the gain survives that restore and lives in Δ or
in pieces outside M₀(B); the stand-in built from the pieces Γ marks important keeps little of it (E42's cipher).

**Three answers.**
- *Yes*: on a real model, anchors of known kind land on opposite sides on at least one real-model number, nulls
  (random δW matched on side effects, a non-learning update, an other-target update) read neither, and the real
  updates (gold SFT, own samples) land clearly on one side or clearly between, with the same number on both seeds.
- *No*: the known-existing and known-new anchors read the same in the real model, however the stand-in reads them.
  Then "re-gating" is a reading of Γ, and the line's write-up says exactly that.
- *Undecidable with VPD*: the toys separate but every real update sits between, and the reason is VPD's unit. A rank-one
  piece is the atom; a new mechanism assembled from old pieces under new gating is "re-gating" to VPD and "new" to a
  circuit-level reading. If this is where the data land, the finding is about the decomposition's granularity, and
  the honest next step is a Γ from another seed (if a second released decomposition exists in `artifacts/vpd`) to show
  whether even that reading is stable; the line does not try to force a verdict.

**What each number in play has been shown.** The gap is the second row.

| number | read in | separates known anchors? | survives nulls? |
|---|---|---|---|
| gate gain g; stand-in ratio (gated piece gain / piece gain) | the stand-in | yes: 0.87-0.97 copy, 0.36-0.41 random maps (E12, E46) | partly: a hand write with no machinery reads 0.16-0.34 of the copy's g; random δW ≤ 1/8 (E46); split nulls running (Amendment 3) |
| restore share of a set (Γ's set S, a target-ranked set, every alive piece) | the real fine-tuned model | **never read on anchors of known kind** (E38: three single triggers only) | random and null sets take back ~0 (E38) |
| outside share, Δ share of the split | the update's energy | no (E12) | — |
| teacher training's pass | the adversary | no: the cipher's copy passes too (E42) | — |
| masked-off share of the teacher copy at the cue | the stand-in | yes (E42: 0.96 lost on the cipher, 0.31 identity) | untested |

## 2. Options

**A. Real-model restore on E12's known-kind anchors ($0, cluster `debug`, about a day of agent time). Recommended first.**
States: E12's two copy and two random-map fine-tunes, seeds 0 and 1, as E46 regenerates them (`run.py` `e12_build_0a`),
left whole in Δ, scored at the key position `half`. Sets, at that position: S (Γ's switched-on set, E38's rule);
the gold set (pieces most active in W₀'s Pile text before each answer word: M₀ for the random maps' answers, which are
common words, and for the copy's keys); a target-ranked set at S's count with no Γ (E38 Amendment 5); every alive
piece; 20 random sets of S's size. Numbers (per state, nats and share of the gain): restore share of each set, with
E12's stand-in ratio beside. Expected: W₀ control, restoring any set moves the answer by ≤ 0.01 nats; random sets ≤ 0.05
share; *copy*: every-alive and S take back much of the gain (lean ≥ 0.5 every-alive; E38 read 0.38-0.49 on triggers);
*random maps*: every-alive ≤ 0.2 and S near 0, the gain living in the update's own write (Δ). Separation rule: the
class gap on every-alive or on S is ≥ 0.3 on both seeds. Positive control: the copy class (existing by construction).
Blind to: a new behaviour that happens to flow through M₀'s pieces by cascade (reads "elicits"); hence the every-alive
row first, S second. Code exists: `recruitment.restored_at` (position-generic, `half`), E46's loaders; one short
`run.py` in `experiments/regating_table/` or a new `experiments/restore_anchors/`. What changes next: separates → it is
the real-model number carried to Qwen, the stand-in ratio beside it; does not separate (both classes alike) → the
line's real-model claim is dead on the 67M and stage 1 rests on the stand-in alone, said in one sentence; every-alive
≤ 0.2 on *both* classes → the toys' gains all sit in the update's own write and cannot calibrate a real-model number;
go to Qwen without 67M calibration and say so. */design*: class 1-2 (existing code and regenerated states, $0; the
plan's "no new 67M trigger fine-tunes" holds, E12's states are the ones E46 already rebuilt); nearest tried E38 (restore,
no anchors) and E12 (anchors, stand-in only); a step, not a patch: it moves the instrument into the real model.

**B. Stage 1 on Qwen3-1.7B with both numbers (gated; about $5-10 inside the Qwen lead's pod; after A and Amendment 3).**
Arms: E45's anchor 1 (the format template, since no prompt raised accuracy: a weaker "existing" anchor than planned,
name it as "format elicited by a prompt"), anchor 2 (invented facts, new), the self-distillation null, gold SFT and own
samples unlabelled. Numbers: the stand-in ratio in KL form on the shared maths rows (reset plan, stage 1) and A's restore
share with M₀ = Γ's set at W₀ on the *prompted* inputs for anchor 1 and the known-facts rows for anchor 2, plus the
split random null the reset plan's Amendment 3 requires. Separation: anchor gap ≥ 0.3 on at least one number, both
seeds; then the real arms are placed. Not before: E45's anchors learned at the scored step (anchor 2 is; anchor 1
waits), E44's split read, Amendment 3's branch known. */design*: class 3 (the reset plan's stage 1) but gated by cost
and by the plan's own "needs the user's go"; what changes next is the elicit/new reading of a real update, the project's
fork.

**C. A no-trigger existing/new pair on the 67M, read per position ($0; new task family, needs confirmation).**
Existing: text the model already models (lowercase Pile, E36/E37); new: the same rows under a fixed substitution of 50
common tokens (a broad cipher). Read per position: every-alive restore share in KL form and the stand-in's share.
Why it goes differently from E36: E36 had no "new" twin and no real-model number. Cost: two 300-step fine-tunes
(minutes) and passes; the restore hooks need a per-position form (a day of code). Risk: the stand-in reads ≈ 0.8 for
any broad update (E36), so only the real-model row can carry it. Take it only if the user wants a cue-free 67M result
before paying for Qwen. */design*: new task family → the user's confirmation; nearest tried E36, E37.

**Not restarted, and why.** The fixed shift (de-prioritised 2026-10-27; near-tautological). "Does Γ's set beat a
target-ranked set" (E38, closed): the elicitation question needs the *pieces* as the unit, not Γ to win; the ranked set
stays as a reference row only. A Γ refit (E15) until its W₀ control passes. New 67M trigger fine-tunes.

## 3. Recommendation, and what would change it

Let Amendment 3 finish; then run A this week at $0, because the line's only real-model number has never met an anchor
of known kind and that is the cheapest way to learn whether "elicits" can be read off the model rather than off Γ.
Carry to Qwen (B) whichever of the two numbers separates E12's classes, both if both do, with the stand-in ratio never
reported without the real-model row beside it. Take C only if the user wants a cue-free result before spending on Qwen.

What changes this: Amendment 3 ends at branch 2 (a hand write ≥ 0.7 on the stand-in ratio) → the stand-in number is
out and A's real-model number is the only candidate for stage 1. A reads every-alive ≤ 0.2 on both classes → skip 67M
calibration, go to B with the stand-in ratio and say the real-model number was unreadable on toys. E45's anchor 1 does
not learn its format → stage 1 has one anchor and reads only "like invented facts / not", until a second existing-kind
anchor exists.

**Not run.** Nothing; this is advice. Open on the user: whether A is class 1-2 as read here, and whether the
"undecidable" wording above is the one the write-up should carry.

## Filed (E50): option A's numbers, rules and stop rules

*2026-10-29, filed before the code ran (code: `experiments/restore_anchors/run.py`; the same rules as data in
`experiments/restore_anchors/results/predictions.json`, which the run refuses to start without; never edited after the
run). Option A only; B and C are not approved.*

**The request in the Terms' words.** On E12's four key → value tables (ident32, ident32b copy; rand32, rand32b random
maps; seeds 0 and 1: eight fine-tunes, E46's stage 0a rebuilds, read from `experiments/regating_table/results/large/`
and checked by the sha256 recorded in its `stage0a.json`), each placed by do-nothing (released subcomponents, δW in Δ,
Γ frozen), read the full fine-tuned model (W₀ + δW, no Γ, no masks) at the key position (`half` = 20) of the 97
held-out windows of each table. Gain G = T1 − T0, T the mean log-odds of the table's value. Restoring a set X: at the
key position only, each decomposed matrix's output gains Σ_{c∈X} ((x₀ − x)·v_c) u_c, x its input in this pass and x₀
its input in W₀'s pass on the same window (`recruitment.restored_at`, E38 Amendment 3). Restore share
ρ(X) = (T1 − lo with X restored) / G; the nats taken back are T1 − lo. The sets, per fine-tune:
- **S**, Γ's switched-on set: alive subcomponents whose mean importance at the key position over the 403 training
  windows rises by more than 0.3 (E38's rule, `recruit.switched(regating.shift_of(...), 0.3)`).
- **gold**: the |S| alive subcomponents most on before the table's answers in W₀: largest W₀ contrast (mean importance
  at Pile val rows 0-2,000 positions whose next token is one of the table's 32 values, minus the mean over all
  positions; E38 Amendment 2's census control, the 32 words pooled). The copy's answers are its keys.
- **D**, the target-ranked set (E38 Amendment 5 as filed): per matrix, the |S in that matrix| alive subcomponents
  outside S of largest first-order take-back e_c on the training windows. It reads the target, not Γ; Γ gives only its
  count per matrix and its exclusions.
- **every alive**: all alive subcomponents at once (E38 Amendment 4's `all_alive`).
- **random**: 20 sets of S's size per matrix, drawn from the alive subcomponents outside S (`recruit.random_like`, seeds 0-19).
E46's stand-in ratio (gated piece gain / piece gain, `regating_table/results/stage0b.json` → `score.P0.2.rows`, read
under E46's least-squares split) is copied beside each fine-tune, not remeasured.

**Expected values.** Shares of G (G is 13-21 nats here, so 0.1 of G is 1.3-2.1 nats).

| number | W₀ control | random sets | copy class | random-map class |
|---|---|---|---|---|
| R1: any set restored in a W₀ pass, abs(lo − T0) | 0 exactly (pass ≤ 0.01 nats) | — | — | — |
| R0: the empty set restored at W1, abs(lo − T1) | — | — | 0 (pass ≤ 1e-5 nats) | 0 (≤ 1e-5) |
| R2: every matrix's whole key-position output set to W₀'s, abs(lo − T0) | — | — | ~0 (pass ≤ 0.01 nats) | ~0 (≤ 0.01) |
| ρ(every alive) | 0 | — | ≥ 0.5 (lean; E38 read 0.38-0.49 on triggers) | ≤ 0.2 |
| ρ(S) | 0 | — | ≥ 0.2 | ≤ 0.05 |
| ρ(gold) | 0 | — | ≥ 0.1 | ≤ 0.1 |
| ρ(D) | 0 | — | ≥ ρ(S) − 0.05 (E38: D matched or beat S) | ≤ 0.2 |
| ρ(random), mean of 20 / max | 0 | abs(mean) ≤ 0.05, max ≤ 0.15 | same | same |
| stand-in ratio (copied) | — | — | 0.87-0.97 | 0.36-0.41 |
| separation (below) | — | — | separates on every alive, both seeds (lean) | |

**The separation rule.** For X in {every alive, S} and seed s: gap_X(s) = min over the two copy tables of ρ_X − max
over the two random-map tables of ρ_X (the worst pair). X *separates* if gap_X ≥ 0.3 on both seeds; X is *reversed*
if max(copy) − min(random maps) ≤ −0.3 on both seeds; X is *unscored* if either seed has an unscored state for X;
else X *does not separate*. A state is unscored for every X when G < 1 nat (its shares are NaN, as E38) and for X
when ρ_X is not finite; for S also when S is empty or its random sets bite (max random ρ > 0.15, E38's bar). The
reading, first match: **conflict** (one of the two separates, the other is reversed: outside the tree, to the user);
**separates** (every alive or S separates; which is recorded); **toys unreadable** (every alive scored and ≤ 0.2 on
all eight fine-tunes: the toys' gains sit in the update's own write; go to Qwen without 67M calibration and say so);
**reversed** (random maps above copy: outside the tree, to the user); **unscored** (every alive unscored); else
**does not separate**. gold, D and the random means are reported against their expected values, not scored.

**Edge cases, run against the rules before filing.** Zero or negative gain: G < 1 nat, so the state is unscored and
its seed cannot separate (no division by a small G). Empty S: S, gold, D and the random sets are all empty (each
restore is an exact 0); S is unscored on that state, every alive still scores. Ties: gap exactly 0.3 separates
(≥); ρ exactly 0.2 counts as ≤ 0.2; equal contrasts in gold are ordered by `torch.topk` (floats; exact ties not
expected), equal e_c in D by index (`top_per_matrix`). NaN: any non-finite ρ makes its row unscored, never a pass;
NaN is written to the JSON as the string "nan". A short D (a matrix with fewer alive subcomponents outside S than S
has there) is recorded; D is reported only, so it changes no verdict. Both every alive and S scored but in
opposite directions: "conflict", above.

**Blind to.** *Every alive*: a new behaviour that flows through W₀'s subcomponents by cascade reads as "elicits"; it
cannot see the update's own write (δW·x) or the part of W outside the decomposition, which is where the random maps'
gain is expected; it reads the key position only, so a change at earlier positions shows only through the inputs it
sends to the key. *S*: subcomponents Γ switches on by 0.3 or less. *gold*: the update (it is a W₀ property); rare
answer words (positions pooled, frequent words weigh more). *D*: anything beyond first order; it reads the target.
*Random*: only chance at S's size. *Stand-in ratio*: the real model (it is read in Γ's stand-in). **The method's
limit** (decisions.md, "Limit of the elicitation reading"): a new mechanism built from subcomponents W₀ already has,
put together under new gating, reads as "elicited" to VPD, because its unit is the rank-one subcomponent, not the
circuit. A copy-like reading on every alive therefore means "uses pieces W₀ already has", never "uses a circuit W₀
already runs"; no number here narrows that.

**Positive control and noise floor.** Positive control: the copy class (copying exists in W₀ by construction, and the
stand-in already reads it 0.87-0.97). Noise floor: one restore under bf16 rounds by about 0.02 nats (E38 Amendment 3's
R4b), about 0.0015 of G here; E38's seed-to-seed spread on ρ(S) and ρ(every alive) was at most 0.035, so the 0.3 gap
is about nine times it; the 20 random draws' standard deviation is reported per fine-tune as the null's spread.

**Stop rules (in code).** (1) No `predictions.json`: refuse. (2) Any E46 window or state file missing or not the
file its sha256 recorded: exit before any reading (rerun `regating_table/run.py --stages 0a`). (3) The controls run
first, on all eight fine-tunes, before any share: C1 T0 within 0.02 nats of E46's stage 0a and the full-model path
within 1e-3 of `route_cells`'; C2 T1 within 0.02 × E46's gain of E46's T1; R0, R1 (the W₀ control: S, gold, D, every
alive and the 20 random sets each restored in a W₀ pass), R2 as in the table. Any miss: results written, FAIL logged,
exit 1, no share computed. (4) Results JSON rewritten after each fine-tune; every scored number printed to the log.
(5) G < 1 nat is not a stop: that fine-tune is unscored. Expected runtime 6-10 min on an L40 (model 1 min, Pile
contrast over 2,000 rows about 1 min, about 1 min per fine-tune for its set building, controls and 48 restores); peak
GPU memory under 15 GB. Fits the cluster's `debug` 30-min limit.

**Code review (2026-10-29, fresh Opus, before the run; no rule changed).** R0, R1 and R2 cannot fail for a wrong restore formula (empty set or zero input difference make the correction zero by construction); they catch misaligned batches, wrong cached io and non-determinism. The restore formula itself rests on E38 Amendment 3's R4, run on the same `restore_hooks` (unchanged since). C1's two halves both read `route_cells`' T (one earlier, one now). The stand-in ratio is now checked against its expected range in `reported()` (reported only). The analysis report repeats these.
