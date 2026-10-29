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

## Amendment 1 (2026-10-29): power checks

*Filed before the code ran (code: `experiments/restore_anchors/amend1.py`; rules as data in
`experiments/restore_anchors/results/predictions_amend1.json`, which the run refuses to start without; never edited
after the run). $0, one cluster `debug` job, about 3-6 min on an L40. Nothing here changes E50's filed verdict ("does
not separate"); it decides what the card says about the power of the numbers behind that verdict.*

**Why.** E50 read "does not separate": restoring every alive subcomponent at the key position took back 0.76-0.78 of
the gain on the copy fine-tunes and 0.66-0.73 on the random maps; S took back 0.14-0.19 and 0.10-0.20. An independent
check found three gaps. (i) 1 − ρ(every alive) is exactly the gain kept with only Δ's paths on (the update's own
write δW·x and the decomposition's residual Δ₀·x), so every alive measures relayed versus direct gain, and no update
known to read low was run: its power is unknown. (ii) S missed its copy floor and takes back a near-constant 2.0-2.7
nats on all eight fine-tunes, whatever its size (24 to 118); five subcomponents are in all eight S sets
(h.2.mlp.c_fc:2151, h.2.mlp.down_proj:2550, h.3.attn.q_proj:60, h.3.mlp.c_fc:1428, h.3.mlp.down_proj:1345), and
random-map S sets share 12-22 members with copy S sets. (iii) W₀'s T0 on the copy tables is about −10 nats: W₀ does not
copy in this format.

**The request in the Terms' words.** Get one real-model number whose control shows it can separate. Rows, each placed
by do-nothing and read exactly as E50 reads (full model W₀ + δW, key position `half` = 20, the 97 held-out windows):
1. *Low-reading control.* Per table, E46 Amendment 3's two nulls: the hand write at the G-matched alpha (`{h.1.mlp.down_proj:
   α X}`, X = `perturb.key_write` of the key inputs and the answers' unembedding rows, α = `stage0b.json`'s), and the
   seed-0 random δW (`finetune.random_direction` of the seed-0 fine-tune, times `amend1.json`'s scale). S for each is
   its own switched-on set (E38's rule). Read every alive, S and the sets below on both.
2. *Write-only restore* on all eight fine-tunes (and the nulls): each matrix's key-position output set to W₀x₀ + δW·x,
   no subcomponents. It differs from every alive by exactly Δ₀(x − x₀). Done through the same restore hooks with W₀
   as one piece per matrix, (U, V) = (W₀ᵀ, I).
3. *S's shared core versus the rest.* The core = the S members present in all eight E50 S sets (read from
   `restore_anchors.json`; the five above). Per fine-tune restore (a) the core and (b) S minus the core, each beside 20
   random sets of matching size per matrix (alive, outside that row's S, seeds 0-19, `recruit.random_like`).
4. *T0 per table and a W₀ copy probe.* W₀ with a demonstration (marker, key, value) written at positions half−10 ..
   half−8 of each held-out window: does W₀ produce the value in-context? One format only (it is one more forward pass).

**What the hand write should read, from the algebra.** Restoring every alive subcomponent leaves, at the key, each
matrix's output W₀x₀ + δW·x + Δ₀(x − x₀): only the update's own write and Δ₀'s paths stay on, so ρ(every alive) = 1 − K/G
with K the gain kept on those paths. The hand write changes one block-1 matrix; nothing upstream of it changes, so at
the key its input is x₀ and δW·x₀ is its whole write, built to add α times the answer's unembedding row to the residual.
ρ(every alive) on the write is therefore the share of its gain that W₀'s later subcomponents relay. It needs no relay
to raise the answer, so it should read low.

**Expected values.** Shares of G; G is 13-21 nats for the fine-tunes and the hand writes.

| number | expected | rule |
|---|---|---|
| T0 per table | ident32 −10.150, ident32b −10.223, rand32 −13.215, rand32b −13.981 (E50) | control C1 |
| write: G | 13.37, 15.27, 13.10, 20.90 (Amendment 3's write_G) | control C2_null, within 2% |
| write: ρ(every alive) | ≤ 0.2 on each table | reported |
| every alive gap: min over copy fine-tunes − max over the four writes, per seed | ≥ 0.55 (0.757 − 0.2) | **power rule**: ≥ 0.3 on both seeds = powered |
| write: ρ(S) | ≤ 0.05 on each table | **S-on-write rule** |
| random δW: G | < 1 nat on all four (Amendment 3: 0.04-0.40) | unscored by E50's rule: every share NaN, nats logged only |
| write-only minus every alive, per fine-tune | abs ≤ 0.02 (a **materiality bound**, not a noise floor) | FYI, changes no verdict |
| core and rest, "shared format part" reading | nats(core) ≥ 0.5 × nats(S) on all eight (about 1.0-1.35 nats or more); ρ(core) above its random sets' max; ρ(rest) ≤ 0.1; core and rest each "does not separate" | **core rule** |
| core and rest, "kind-specific" reading | rest separates by E50's rule (gap ≥ 0.3, worst pair, both seeds) | **core rule** |
| random sets (S, core, rest sizes) | abs(mean) ≤ 0.05, max ≤ 0.15 | reported; max > 0.15 makes that set unscored (E50) |
| probe: lo with a demonstration − T0 | ≥ 3 nats on the copy tables (lean: induction); random-map tables beside | reported |

Lean: every alive "powered", S "low" on the write, core "shared format part" (S takes back the same 2-2.7 nats whatever
its size, which points to a part common to all eight). A priori, "kind-specific" by E50's rule is nearly out of reach:
ρ(S) ≤ 0.2 on all eight, so ρ(rest) on copy can clear random maps by 0.3 only if the core pushes against the rest
(restores need not add; S − core − rest is logged per row).

**The rules.** A row is scored for a set when G ≥ 1 nat and its ρ is finite; for S, core and rest also when the set is
non-empty and its random sets' max ρ ≤ 0.15 (E50's). gap_class per seed, given a high group and a low group: unscored if
any value is unscored; g = min(high) − max(low); "separates" if g ≥ 0.3, "reversed" if max(high) − min(low) ≤ −0.3,
else "no"; over seeds as E50 (both separate → separates; both reversed → reversed; any unscored → unscored; else does
not separate).
- *Power of every alive*: high = the seed's two copy fine-tunes, low = the four hand writes. First match: unscored
  (either seed); **powered** (separates on both seeds); **blind** (on neither); **partial** (one seed).
- *S on the write*: first match: unscored (any write unscored for S); **low** (ρ(S) ≤ 0.05 on all four); **generic**
  (S takes back ≥ 1.0 nat on all four, about half the fine-tunes' 2.0-2.7); **between**.
- *Core and rest*: shared = on all eight fine-tunes the core is scored, S takes back > 0 nats, the core takes back
  ≥ 0.5 of S's nats and its ρ beats its random sets' max; kind = rest separates (high = copy, low = random maps). First
  match: unscored (any fine-tune unscored for core or rest); **both**; **shared format part**; **kind-specific**;
  **neither**.
- E50's separation rule is also reported for every alive, write-only, S, core and rest on the fine-tunes.

**Edge cases, run against the rules before filing.** A write with G < 1 nat (cannot happen at the matched alpha, but
covered): its every-alive value is unscored, so the power rule is unscored, not passed. A negative ρ (the write's
downstream blocks cancel part of it) counts as low; it only widens the gap. Gap exactly 0.3: separates (≥). An empty S
on a write: S unscored there, never a "low" pass (an empty restore is an exact 0). ρ(S) ≤ 0.05 and S nats ≥ 1.0 at once
(possible when G > 20): "low" wins by order. S taking back ≤ 0 nats on a fine-tune: the core test fails there (the 0.5
fraction has no meaning), so the reading cannot be "shared". An empty core (no member common to all eight): the core
is unscored on every row and the reading is "unscored". The random rows: their shares are NaN by E50's rule, so they
enter no rule; NaN goes to the JSON as "nan". A core subcomponent missing from a null's own S: the rest is S minus
the core as a set difference, unaffected; the core is still restored as given.

**Blind to.** *ρ(every alive) on the write*: it equals the relayed share only because the write's matrix sees x₀; it
says nothing about whether a fine-tune's relay is "existing" machinery. It is one low anchor family (one matrix, one
block), so "powered" means "can tell a direct write from copy", not "can tell any new mechanism from copy". *Write-only*:
Δ₀'s paths (that is its only difference from every alive). *Core*: defined on the same eight states it is scored on
(post hoc), so "shared" is checked only against random sets of its size, and any "kind-specific" rest needs fresh
seeds before use. *S on the write*: subcomponents switched on by 0.3 or less. *Probe*: every format but one.
**The method's limit** (decisions.md, "Limit of the elicitation reading"): a new mechanism built from subcomponents W₀
already has, put together under new gating, reads as "elicited" to VPD, because its unit is the rank-one subcomponent,
not the circuit. A powered every alive therefore separates direct from relayed gain, never "a circuit W₀ already runs"
from a new one.

**Positive controls and noise floor.** Every alive: the copy fine-tunes are the high end (E50: 0.76-0.78) and the hand
write is the designed low end. Core and rest: the 20 random sets of matching size. Write-only: every alive, from
which it differs by one exact term. The probe has no positive control (report-only). Noise floor: one restore under
bf16 rounds by about 0.02 nats (E38 Amendment 3's R4b), about 0.0015 of G here; E50's seed spread was at most 0.020 on
ρ(every alive) and 0.052 on ρ(S), so the 0.3 gap is about 15 and 6 times them; the random draws' sd is reported per
row and set.

**Stop rules (in code).** (1) No `predictions_amend1.json`, `predictions.json` or `restore_anchors.json`: refuse. (2) Any
E46 window or state file missing or not the file its sha256 recorded: exit before any reading. (3) Controls before any
share, on every table and row: E50's C1 (T0 against E46 and route_cells) plus T0 within 0.02 nats of E50's; E50's C2
(T1 against E46, fine-tunes), R0, R1 (every set of this amendment restored in a W₀ pass, ≤ 0.01 nats) and R2; C2_null,
the nulls' G within max(0.02 nats, 2%) of Amendment 3's; C5, the write-only restore in a W₀ pass reads T0 within 0.01
nats. Then the nulls are read first (they are the power controls), and C3 is each fine-tune's first reading: ρ(every
alive) and ρ(S) within 0.01 of E50's. Any miss: results written, FAIL logged, exit 1. (4) Results rewritten after each
row; every scored number printed to the log. (5) G < 1 nat is not a stop: that row is unscored.

**Decision rule (what each outcome means for the card).**
- Every alive **powered**: E50's "does not separate" stands as a finding, not a blind spot: both kinds relay about 0.7
  of their gain through W₀'s subcomponents, while a direct write does not. Every alive measures relayed versus direct
  gain, not existing versus new mechanism; it goes forward, if at all, only as a relay number beside the stand-in
  ratio. **Blind**: it reads high even for a direct write; E50's every-alive row is uninformative and is not carried
  forward. **Partial** or **unscored**: card to the user with both seeds' gaps.
- S on the write **low**: S's near-constant take-back belongs to learned updates. **Generic**: S takes back nats from any
  update that pushes the answer; S is withdrawn as evidence of kind. **Between**: reported; S stays as E50 left it.
- Core **shared format part**: S's take-back is mostly five shared subcomponents; S is closed as evidence of kind on
  the 67M and the five are named on the card. **Kind-specific**: the rest is a candidate real-model number, post hoc,
  so it needs fresh seeds in a new amendment before any Qwen use. **Both**: both statements. **Neither**: S is closed as
  evidence of kind on the 67M. **Unscored**: card to the user.

**Departures.** The hand write is rebuilt in `amend1.py` from `perturb.key_inputs` and `perturb.key_write` (the two
`src/` calls `regating_table/run.py`'s `write_delta` makes), not imported: importing across experiments needs a path
change, and moving `write_delta` into `src/` would mean editing `regating_table/run.py` and `perturb.py`, which E46's
`code_unchanged` guards (stage 0b; Amendments 1 and 3) check against recorded commits. C2_null checks the rebuild
reproduces Amendment 3's G. `amend1.py` is about 300 lines, over the ~200 asked: the controls, the sets and the filed
rules are each one function; nothing was cut that a rule reads. The probe is one demonstration format, not "any
format": a wider format search is a separate question about W₀, not a power check.
