# E90: the new-against-reuse definition as a procedure, tested on filed edge cases, then read on the old 67M tasks

*Baseline-lead, 2026-11-05, drafted by Fable with the lead's edits (marked in "Lead's edits"). For the user; no code, no run, no predictions filed. Decision it follows:
`reports/decisions/2026-11-05_decision_new-vs-reuse-definition.md`, `decisions.md` rows 115-116. Apparatus detail:
`reports/plans/2026-11-05_plan_e90-appendix.md`.*

## Changes the user must see

Writing the accepted rule out as a procedure and running it on paper against the filed edge cases broke it in three
places. Each change is the smallest I found, with the case that forces it. Nothing settled is reopened.

1. **A verdict needs a spared control.** The accepted comparison reads "not through C" when the state's share does
   not beat a same-windows partner with no admissible C. Under C = token identity every keyed state loses its whole
   gain when the key's directions go, the cipher included, so copy reads "not through C" by construction
   (question 1). Change: "reuses C" and "not through C" both need a control on the same windows that loses at most
   the random band; if the control loses too, the block reads "unreadable: not specific". No partner without C
   exists for a capability every keyed task reads.
2. **Remove W₀'s mechanism for C where one is located; the value-defined directions count only with a spared
   same-answer control.** When C's values are the answer tokens (W₀'s guess, the copied token, a class word), the
   directions where f_C moves the residual carry the answer itself at every block after a state writes it. The
   per-key hand write (`src/commute/perturb.py:61` `key_write`, into `h.1.mlp.down_proj`) writes the answer at
   block 1 and uses no later mechanism, yet should lose its gain under that removal at blocks 2-3 and read
   "reuses C". E70's block-6 reversal is the same effect. Change: where a mechanism is located (E62's copying head),
   remove its output, with the other task on the same windows as the control; otherwise remove the value-defined
   directions with the hand write of the same table as the control. This is also the size control (question 3): a
   removal that spares a state with the same answers on the same windows is not "size".
3. **The task's answer is its scored alternative.** Lowercase scores log p(lower form) − log p(capital form) at
   case positions; as the pair, the answer is always "lower", a constant, so lowercase reads "definition does not
   apply" like emoticon and the ladder. As the token, it is admissible only through token identity, which change 1
   makes unreadable. Either way the decision report's "best fit" row gets no "reuses" label.

With these, E70's saved P and N shares read "unreadable: not specific" at every block, N losing 0.16-0.49 above
bands of 0.004-0.024 (`e70_a1.json` `.r2.rows`), so the block-exclusion rule chosen after the reversal is dropped.

## Lead's edits, and the fourth thing the user must see

Four edits to the draft, checked against the saved files. (1) b* starts at block 1: the draft's "earliest possessed block" would land on block 0, upstream of the hand write, against its own R3. (2) E70's bands corrected to 0.004-0.024. (3) R5's band is measured, not quoted: E62 did not save one per state. (4) Stage 2 shrinks to row 1 if R3 reads "not specific".

**No size bar on "reuses C".** "Above the band" is a test that the loss is real, not that it is large: a state losing 0.03 of its gain over a band of 0.01, with its control spared, reads "reuses C". No sourced bar exists; E62's 0.5 was chosen, and copy straddled it at 0.47-0.56. So the verdict prints its share ("reuses C, 0.5 of the gain") and has no bar. **E62's filed verdict, "copy_not_elicitation" with reading "mixed", stands.** E90's "reuses C" on copy would be a different rule's reading beside it, and every statement of it says so.

## Design check

- **Goal.** `GOALS.md`'s second question, new capability against strengthened old ones. No definition a measurement
  can apply exists; every 67M label but E62's is by assumption. This gives the label the re-gating instruments are
  scored against.
- **Nearest tried.** E59, E64 (readouts, unreadable); E70 (direction removal on the counting model, "cannot serve");
  E62 (head ablation, copy 0.47-0.56 against lookup 0.00); E55 C3 (pruning, reverse fine-tuning, control failed);
  E58 (held-out carrying). None ran one procedure with filed edge cases across tasks. This learns which old labels
  survive one definition and which rows it cannot read.
- **Patch or step.** A step: the definition the reuse line has lacked since row 81, from E70's and E62's code.
- **Decisions.** Relies on rows 81, 112, 114-116; row 87's held-out rule is reported beside the label. Departs from
  the decision report's proposal in the three changes above.
- **What changes what we do.** Hand write spared: the value form is usable and rank-1, cipher and elicit get labels.
  Not spared: only located mechanisms give labels, and the Qwen line must locate one before any claim of reuse.
- **Apparatus arm, first and alone.** W₀'s own C under each removal against 20 random bases of the same rank; no
  teeth at any block before the last stops that C.
- **Numbers.** Section "Numbers". **Saved.** Appendix. **Class.** A new measurement method, outside the approved
  classes: the user confirms after reading this.

**The request in the Terms' words.** For each 67M fine-tune difference δW we labelled by intuition, decide from W₀
and the task's labels alone which named capability C could carry the task, then whether W₀ + δW's gain over the
do-nothing reference (δW into Δ, so W₀ itself) runs through C: remove C's directions from the residual and compare
the gain lost against the same removal on a spared control. Verdicts "reuses C", "not through C", "unreadable",
"definition does not apply"; never an unqualified "new".

## The definition as a procedure

Inputs: a task (inputs, scored alternatives, an answer per input), a named C = (a text rule f_C with values Y_C, its
natural inputs X_C in Pile, where it is read), W₀, states W₀ + δW. Every step runs on W₀ first and stops at a miss.

**A. Admissibility, from labels and W₀, no state.** g ∈ {constant; identity; a text rule named in the filing with
no fitted parameter; a table over the values seen in the fit half}. Fit half = even-indexed keys (or windows for a
per-window answer), read half = odd (E70's `lookup_accuracy` split, `src/commute/removal.py:132`). C counts iff the
best g's held-out accuracy strictly exceeds the fit half's most common answer read on the held-out half. Stage 0
uses the true f_C: a pair that fails there fails on any readout of f_C. Stage 2 repeats it with the readout's
values. An answer that is a constant across inputs fails for every C: "definition does not apply".

**B. Possession.** A linear readout d → |Y_C| per block on W₀'s residual at the X_C position (E64's
`train_readouts64`, Jain's schedule), held-out rows of X_C. Possessed at block b iff it beats its shuffled-label
twin and reaches 3/4 of W₀'s own output accuracy on f_C there (E64's relative form, decision report item 5).
b* = the earliest possessed block from 1 up to the one before the last. Block 0 is out: the hand write sits in block 1, so a removal at block 0 would sit upstream of it and test nothing, and a readout there reads the token.

**C. Removal, the label.** Mechanism form, where located: mean-ablate W₀'s component for C (E62's head (2,4),
`copying.ablated`). Value form otherwise: the orthonormal basis of the per-value mean residuals at b* on X_C,
centred (`removal.centred_basis`, rank |Y_C| − 1; E70 A1's covariance direction is the scalar case), removed at the
answer position (`removal.remove`). Score: A_lo = (state's log-odds drop − W₀'s on the same windows) / the state's
log-odds gain, nan under the gain floor (`readout.shares`, E64/E70's amended rule). Band: 95th percentile of 20
random bases of the same rank, on every state (`removal.band_of`, `max_band`).

**D. Verdict per (task, C), at b*.** Control first: W₀'s own log-odds on f_C at X_C falls above the band ("teeth"),
else unreadable. Then the 2 × 2 on the state and its spared control (mechanism form: the other task on the same
windows; value form: the hand write of the same table at matched gain, `perturb.key_write`, `perturb.match`):
state above band and control within it → "reuses C"; state within and control above → "not through C"; both
above → "unreadable: not specific"; both within → "unreadable: the removal does not reach the states". Pruning
revival, reverse fine-tuning (E55 C3) and held-out carrying (E58) are reported beside, never as the label. Every
claim states the granularity limit (a circuit from old parts reads as reuse).

## Edge cases, filed before any old task is read

| # | Case | Filed verdict | Why |
|---|---|---|---|
| A1 | random map, C = token identity | not admissible | the table has no entry on held-out keys: 0 against the constant's ~1/16 |
| A2 | copy, C = token identity | admissible | identity reads 1.0 on held-out keys |
| A3 | constant answer (rand1, emoticon, ladder rungs, lowercase as its scored pair) | does not apply | every g ties the constant; strict > |
| A4 | rank-1, C = W₀'s guess (18 values on 24 keys, one token on 9) | admissible | identity 1.0 against the constant's share of held-out keys holding that token (≤ 0.375 on the parity split; the plan prints the count) |
| A5 | rank-1 with one distinct value | not admissible | identity equals the constant: a tie |
| A6 | elicit, C = W₀'s guess | not admissible | the rank differs per key; the table is unseen on held-out keys |
| A7 | E62 lookup, C = in-context copying | not admissible | read-side Bs are disjoint from train-side Bs (`copying.build_cue`), so the table is unseen |
| A8 | E62 copy, C = in-context copying | admissible | identity on B |
| A9 | K = 1 or an empty held-out half | not admissible | no held-out input; printed as such |
| R1 | do-nothing (δW = 0), any C | unreadable: no gain | nan share under the floor; cannot pass |
| R2 | ident32 fine-tune, rand32 fine-tune and ident32's hand write, C = token identity, value form | unreadable: not specific | all three lose about all of their gain; shares within one band of each other |
| R3 | rank-1 fine-tune and the rank-1 hand write, C = W₀'s guess, value form at b* ∈ {1, 2} | unreadable: not specific (lean 70%) | the hand write's answer direction lies in the value subspace; branch: hand write within band → the value form is specific, rank-1 read by its own share |
| R4 | E70's saved P and N shares under D | unreadable: not specific at every block | N 0.16-0.49 above bands ≤ 0.02 at blocks 3-5; both above at block 6; no block rule needed |
| R5 | E62's saved copy and lookup under the mechanism form | copy reuses C; lookup not through C | 0.47-0.56 against 0.00; the state-level band of random single heads is not in E62's saved results, so stage 0 checks the mapping only and stage 2 measures the band |
| R6 | share equal to the band; partner gain under the floor | not above; no call | strict comparisons; nan excluded and counted |
| R7 | ladder `r1_t534_v369` (" which" → " was") fine-tune, seed 0, step 10, and a hand-written constant push to the same word at matched gain, C = W₀'s graded leaning to the target (f_C = W₀'s log-odds of the target at the position; scalar covariance direction on its natural inputs, E70 A1's form) | unreadable: not specific (lean, same reason as R3) | the push is `key_write` with K = 1 on the trigger's input. Branch: push within band and state above → the leaning form is usable, and the user decides whether constant-target tasks get read on their old states. A3 stands until they do |

R4 and R5 are not blind: they check the verdict mapping on known data. R2 and R3 are the new measurements. A miss
on them outside its branch, or no teeth on W₀, changes the definition, dated, before stage 2 (Amendment 1). The
branches are pre-registered so the likely outcomes burn no amendment (question 6).

## The old tasks, in order, with the reader's expected verdict

Most rows will not read, and that is the finding the user asked for (question 5). The one row where every piece
lines up is E62's pair. The second is the rank-1 and cipher pair, which decides whether the value form can ever
be specific. Everything else is "does not apply", "no C listed" or "not specific" unless R3 surprises.

| Order | Task (inventory code) | C | Expected verdict |
|---|---|---|---|
| 1 | T5 E62 copy / lookup | in-context copying, head (2,4) | copy reuses C, lookup not through C; the share's low start reported (W₀'s log-odds for B under the head at the marker, never read) |
| 2 | T12 rank-1 / T11 cipher, 24 keys, with the rank-1 hand write | W₀'s next-word guess, E64's readout over the 18 values | unreadable: not specific (R3); else rank-1 reuses, cipher not through |
| 3 | T11 elicit | none admissible (A6) | no C listed |
| 4 | T4 ident32, T13 copy-24, T6 rand32 | token identity, the only predictive C: no earlier occurrence in these windows, so E62's head has nothing to copy | unreadable: not specific (R2); rand32 no C listed |
| 5 | T7 lowercase, T1 emoticon, T2 ladder, T3 many-to-one, rand1 | constant at the scored alternative | definition does not apply (change 3); input-dependent versions in a second pass |
| 6 | T8 ciphers, T14 prior-matched cipher, T16 shuffled elicit | none | no C listed; partners only |
| 7 | T15 class map and twin | embedding class (model-defined, flagged; possessed at block 0 by construction) | second pass, rebuild 11 min: admissible; hand write of the class table is the control |
| — | T10 code, T9 narrow lowercase | — | excluded: learned nothing; never trained |

The marker-form copy tasks (T4, T13) have no "in-context copying" C: their windows hold the key once, so W₀'s
induction mechanism cannot produce the answer (question 2). Their only admissible C is token identity, which the
hand write also uses, honestly. The edge case "the hand write must not read as reuse" is therefore re-filed as R3
on the rank-1 table, where the hand write uses token identity but not W₀'s guess mechanism.

## Where the definition does not apply (user: keep flagging this)

Expected, not measured. The definition says nothing today about emoticon (T1), the ladder (T2), many cues to one word (T3), lowercase (T7) and rand1, because each has a constant answer; together with "not specific" on copy (T4, T13), that leaves the "strengthens" side of `GOALS.md`'s question with no reading on the 67M beyond E62's copy. "Reuses" is expected on one row and "no C listed" on five, so the table leans to new by how it is built and cannot score r or the ratio. R7 tests the one proposed repair for the constant answers; input-dependent versions are the other. Every later statement of this definition, including the D3 rows read against its wording, carries this limit.

## Status, from `reports/plans/2026-11-05_plan_qwen-reorient-amend1.md` (2026-11-05)

Finished on paper today. Stage 0 is the paper reading. **Stages 1 and 2, R7 included, wait until the comparator table exists (11-14) and need no new plan, only the user's word then.** No 67M run before that.

## Stages, all on MATS `debug`, $0

**Stage 0, CPU, no model, minutes.** Admissibility A1-A9 on every (task, C) pair from the text rules, the saved
tables and E54's key lists; R4 and R5 on E70 A1's and E62's saved JSON. Output: the admissibility table and the
pairs stage 2 reads. A surprise here is a finding, not a stop.

**Stage 1, one job, about 6 min.** (i) The readout for C = W₀'s guess over the rank-1 values at blocks 0-2; E64's
job read twelve states and its readout in 1 min (`TRIED.md:135`). No possession before the last block: that C out.
(ii) The control, alone: W₀'s drop on f_C at X_C under each basis against 20 random bases; E70 A1's control stage
took 45 s (`cluster/logs/commute-e70-a1-50499.out`). No teeth before the last block: that C out. (iii) R2 on
`ident32_s0`, `rand32_s0` and the rebuilt ident32 hand write; R3 on `rank1_s0`, `cipher_s0` at 24 keys and the
rank-1 hand write at matched gain; R7 on the ladder state and its constant push. A miss outside the branches writes a stop file; the definition changes before
stage 2.

**Stage 2, one job, about 10 min, and its size depends on R3.** If R3 reads "not specific" (the lean), rows 2-4 can only repeat that, so stage 2 reads row 1 alone: band and W₀'s missing log-odds, about 3 min. If R3 spares the hand write, it reads rows 1-4 on every saved seed: E62's three per map, rank-1, cipher and elicit
seeds 0-2, ident32 and rand32 seeds 0-1, copy-24 seeds 0-2, with the mechanism form reread for row 1 and the
missing W₀ log-odds under the head. E70 A1's state stage read 20 states at six blocks in 859 s on the 3M model;
here about 15 states at three blocks. The job resumes from its JSON at a 22-min deadline. Second pass, after the
user sees stage 2: the class map, and input-dependent versions of lowercase, emoticon and one ladder rung.

**Numbers, and what each should read.**

| Number | W₀ (do-nothing) | Control | Expected on states |
|---|---|---|---|
| held-out accuracy of g, per (task, C) | from labels: copy 1.0, random 0, constants tie | — | none, stage 0 |
| readout accuracy per block, C = W₀'s guess | ≥ 0.75 at some block ≤ 2 (E64: 17 of 20 at blocks 0-3) | shuffled labels ≈ 1/18 | — |
| W₀'s drop on f_C under the basis, per block | above band (E70 A1: 1.9-12 against ≤ 0.22; E64 W₀: 1.82 nats) | random band ≤ 0.1 | — |
| A_lo per state | nan (gain 0) | hand write: ≥ 0.5 under the value form (R3 lean); lookup 0.00 under the head | ident32 and rand32 ≈ 1; rank-1 high; copy 0.47-0.56 |
| variance share of the removed basis, against random | token identity at the key: large; W₀'s guess: reported | random ≈ 0.005 | reported, no bar |

**Early-stop rules.** The two in stage 1; the readout's held-out loss over E64's bar, 1.37 against the model's
1.27, stops as E64 did; a state whose sha256 differs from the saved record stops. Each reading is logged with its
band as it is read.

**What a miss does.** R2 with a spared cipher: token identity is specific after all and copy reads by its share.
R3 hand write spared: the value form is usable and the rank-1 and elicit rows get read. W₀ without teeth at every
block ≤ 2: the basis is wrong, and Amendment 1 refits it on the task's windows instead of X_C, a departure from
item 6. No readout for W₀'s guess at any block: that C is unreadable on the 67M, as E59 and E64 found, and rows
2-3 read "unreadable: no readout".

## Blind to, and not run

The value form is blind to a new circuit built from old parts and to any mechanism that bypasses b*. The mechanism
form sees only located components; no mechanism is located for W₀'s guess or token identity. Admissibility is blind
to any C not listed, which is why no row reads "new". Not run here: pruning and reverse fine-tuning on new states
(E55 C3's numbers are reported beside); Qwen; the input-dependent task versions; more than the saved seeds.

Hand-off: a new session writes the code from this plan and the appendix after the user's confirmation; read
`src/commute/removal.py`, `src/commute/readout.py` (E64 section) and `experiments/pcfg_anchors/e70_a1.py` first.
