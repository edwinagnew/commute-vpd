# Fable's draft: what the re-gating line does after E62 Amendment 3

*Fable, 2026-11-02, for regating-lead. **A draft, not approved**: nothing filed, run or committed beyond this file. Read:
`GOALS.md` Terms; `decisions.md` rows 73, 112, 114, 117; the E62 plan with Amendments 1-3; the Amendment 2 and 3 analyses
and their checks (where the Amendment 3 report and its check differ, the check is taken); the base-masked-error review;
the earlier advice `reports/plans/2026-11-02_advice_e62-next-FABLE.md`; `experiments/TRIED.md` rows E53-E55, E58, E62,
E63; `experiments/induction_cue/amend3.py`, `src/commute/copying.py`, `src/commute/local_faith.py`;
`experiments/induction_cue/results/{run,amend2,amend3}.json`. No GPU, no model loaded. Elicitation as the user defines it:
the fine-tuned behaviour runs through a mechanism W₀ already has; carrying to unseen keys is evidence, not the definition.*

**In plain words.** We taught the small model to copy at a new marker, and switching off its old copying head costs the
fine-tuned model about half of what it learned, so the head label says "old mechanism". The decomposition's masks also
lose most of their gain when that head is off, but only because the mask function reads the fine-tuned model's
activations, so that tells us the masks react to the head, not that they found the old mechanism. The next test asks
the direct question: does the fine-tune's mask open the same pieces the model's own copying opens, and does closing
those pieces take the mask's gain away?

## Where E62 left the question

- **The head label stands.** Head (2,4) carries 0.47 to 0.56 of the copying map's net gain in the target model and none
  of the lookup's (`run.json`, re-read in `amend2.json`). Filed verdict `copy_not_elicitation`, r unscored (1.09 copy, 0.40 lookup).
- **With Γ's masks held fixed, the head off costs the masked gain a tenth** (Amendment 2, share 0.10 to 0.11). That read
  froze the update's only way into the masked forward, so it tested little (check F1).
- **With Γ's masks recomputed with the head off, the masked gain loses two thirds** (Amendment 3, share 0.61 to 0.69,
  label `over`; lookup within 0.03 of 0). The check confirms every number and the verdict `head_more_than_target`.
- **What that cannot mean.** Γ (`global_shared_transformer`) reads the inputs of all 24 matrices in the target model's own
  forward. Switch the head off there and the fine-tuned model stops producing B; Γ's masks move with those activations
  whichever subcomponents they open. So "the masks follow the head" is what inheritance predicts, and the read has no
  positive control (nothing is known to depend on a head through Γ). The check's status table: account 1 (masks close the
  head's way) unread beyond a 5% sample; account 2 (the gain goes elsewhere in the forward) live; account 3 (the head acts
  through the masks) supported only as "masks react"; account 4 (the fine-tuned head is not in the masked forward) untouched.
- **The plan's tree names no branch for this verdict** (plan, Amendment 3 "What changes next"; report line 170). Its
  nearest branch is a reversal that must go to the user; this draft is what to run after that is told.

## A. What to do next

### The separating idea, before the options

The recomputed share cannot tell "the masks inherit a head dependence from the activations Γ reads" from "the masks read
the mechanism", because Γ is a function of the target model's activations by construction: any head the fine-tuned
model uses moves Γ's masks. More ablation inside Γ's pass (which input carries it, 24 heads) only measures that function.

Inheritance and shared mechanism part company on **which pieces the masks open**, not on **whether** they move. Inheritance
says the state's mask depends on the head's output and says nothing about what it opens. "The fine-tune runs through a
mechanism W₀ already has", at VPD's granularity, says: the pieces the copying state's mask opens at the marker are the
pieces W₀'s mask opens where W₀ copies on its own, and the masked gain runs through them. Both halves are readable on the
saved states, with no Γ ablation and no new training, because W₀ copies on its own inside the very same windows: every
read window is "… A B … A M", and at the second A (`half − 1`, the token before the marker) W₀'s induction predicts B.
So the test compares the state's mask at the marker with W₀'s own mask one position earlier.

### Options

**1. The old-copying mask test (recommended), E62 Amendment 4.** One MATS `debug` job on the six saved step-30 states
(`results/large/{copy,lookup}_s{0,1,2}.pt`, `windows.pt`, `heads.pt`), $0, under 10 min, no training, not gated. It
runs: a control at W₀ (does W₀'s own mask carry W₀'s copying at the second A, through head (2,4)); then, per state, how
aligned the mask change at the marker is with what W₀'s mask adds when it copies, and how much of the masked gain is lost
when those pieces are closed by hand. It tells you whether r is high on the copying map because the mask opened W₀'s
copying pieces (then r reads shared pieces, at VPD's granularity, on the one map with a causal label) or because it
opened other pieces (then r's height is not a mechanism reading, and the stop stands). Next: shared pieces → write that
one datum into the line's claim and give the Qwen KV table its first expectation; not shared → report r as "the
original subcomponents can write this target under some mask", stop the 67M copying work on r's strength, and the Qwen
table carries the split's numbers only, not an elicitation reading.

**2. Stop here and write the limit into the log.** Nothing runs. It tells you nothing new; the line's standing statement
becomes "r follows the target model's head dependence through Γ, and no read has shown it opens the old mechanism".
Next: the method table and E64 continue on the baseline line; the Qwen table is not sold as an elicitation read.
Right only if the user wants the 67M copying work closed now.

**3. Which of Γ's inputs carries the fall, and the step-300 states.** One `debug` job: masks recomputed with only
`o_proj`'s slot edited, then with only the later inputs edited; the six fine-tunes rerun to 300 steps (deterministic, under
a minute each) and the shares re-read at 100 and 300. It tells you whether Γ reacts to the head's output directly or to
the downstream "B is gone" signal, and whether the half-through-the-head share is a moment of training. Either answer is
still a reading of Γ following activations, so it does not answer the user's question. Next: only after option 1, and only
if its reading is ambiguous (`in_between`) and the step question matters for Qwen's choice of read step.

**Recommendation: option 1.** It is the one read that uses the mechanistic definition directly (same pieces, and the gain
runs through them), it is $0, and it needs no ablation inside Γ.

**What would change it.** (a) The control fails: W₀'s own mask does not copy at the second A, or the head off there
costs it under a quarter of the target's fall. Then the decomposition's masks drop W₀'s copying mechanism wherever it
matters, no mask test can read shared mechanism on this model, and that goes to the user as the finding (with the question
whether the 67M line continues). (b) The user wants the Qwen table sooner: then option 2, with the limit in the table's
caption. (c) The lead finds the per-subcomponent gates are not reachable from `Reads.gates` at a position other than the
marker: they are (`gates` returns per-position tensors; `moved` indexes `[:, half, alive]`), so this should not happen.

**The Qwen table.** It should not be next, and the check's C6 is right that it does not follow from Amendment 3. Row 112
had the KV mini-table wait for r to be scored against a causal check; Amendment 3 gave a causal label on the target
model and a mask reaction, not a scoring. Option 1 is that scoring. If it reads shared pieces the Qwen table goes next
with an expectation; if not, the table has no 67M calibration of r as a mechanism read and should wait for the
commute line's reasons, not this line's.

### The test: one claim, the null, three predictions

**Claim.** On the copying map, the fine-tune's mask at the marker opens the subcomponents W₀'s own mask opens when W₀
copies, and the masked gain runs through them; on the lookup it does not.

**Null.** The copying state's mask opens pieces unrelated to W₀'s copying pieces, no more aligned than chance and no
more than the lookup's; closing W₀'s copying pieces costs the masked gain no more than closing as many random opened pieces.

**Definitions** (all gates are Γ's per-subcomponent, per-position values on the 200 read windows, alive subcomponents only,
9,881 across the 24 matrices, `ci_len` 41 as in Amendments 2-3; every mean is over windows):
- `a` = W₀'s mean gate at the second A (`half − 1`) minus W₀'s mean gate at the Pile token before it (`half − 2`): what
  W₀'s mask adds where it copies. The reference position is a Pile token in the same window, so key-reading and generic
  pieces mostly cancel. (`half − 2` is never B: p ≤ half − 4.)
- `S_old` = the fewest subcomponents holding half of `a`'s positive mass; its size k is printed.
- `Δ(state)` = the state's mean gate at the marker minus W₀'s mean gate at the marker.
- **Alignment** = cosine(Δ(state), a). Chance = the same cosine with `a`'s entries permuted over the alive set, 20 draws,
  mean and sd.
- **Closing share** = 1 − (masked gain with `S_old`'s gates set to W₀'s values at every position, under the state's
  mask otherwise) / (masked gain `L01 − L00`). Reference = the median over 5 draws of k subcomponents taken from those the
  state opened at the marker (Δ > 0; if fewer than k, from all alive minus `S_old`, printed), closed the same way.

**Predictions** (bars from E62's filed 0.5 "old" and 0.2 "not", E58's 0.24 band gap; the cosine bar 0.2 and the 3×
chance factor are chosen here, no source):
- P1, alignment: copying map ≥ 0.2 and ≥ 3 × chance on all three seeds; lookup under half of the copying map's on every seed.
- P2, closing: copying map's closing share ≥ 0.5 on every seed and above the random median by ≥ 0.24; lookup's ≤ 0.2.
- P3, the gap: copying minus lookup closing share ≥ 0.24 on every seed. A fall on the lookup too reads "`S_old` holds
  pieces both maps need (reading the key)", which the random reference does not catch; P3 is what separates that.
Verdicts: `shared_pieces` (P1-P3 all hold), `not_shared` (P1 and P2 both miss on every seed, or the copying map's closing
share is at or under its random median), `in_between` (anything else), `control_failed`, `too_small` (masked gain under
`local_faith.MIN_GAIN` 1.0), `unreadable` (k = 0, k over half the alive set, or a non-finite share). Edge cases to test on
CPU before filing: `a` with no positive mass; `S_old` disjoint from the opened set (closing changes nothing, share 0, a
legitimate `not_shared` reading); a negative closing share (closing raises the gain: counts as a miss); ties in `a`'s
ordering; fewer than k opened pieces.

**Numbers and what each should read.**

| Number | W₀ | Do-nothing (zero update) | Control | Expected, copying map | Expected, lookup |
|---|---|---|---|---|---|
| Target model's log-odds of B at the second A, intact and (2,4) off | the reading; expected P(B) at least 10 × the marker's 4.3e-4, fall under the head over 1.16 (its marker fall) | same as W₀ | stop if P(B) under 4.3e-3 (W₀ does not copy here: fall back to stage (a)'s repeated sequences for `a`, or stop) | not read per state | not read |
| Masked forward's log-odds of B at the second A under W₀'s own mask, intact and (2,4) off in the forward | within 2 of the target's; fall under the head at least 0.25 of the target's | same | exit 1 if either misses: the masks do not carry W₀'s copying, `control_failed` | | |
| k = \|`S_old`\| | printed; expected tens to a few hundred | same | `unreadable` if 0 or over 4,940 | | |
| Alignment cosine | undefined (Δ = 0) | 0 vector, undefined | chance ≈ 0 ± 0.01 (permutation) | ≥ 0.2, lean 0.3-0.5 | under half the copying map's |
| Closing share | undefined (no gain) | 0/0, reported as the level `L00` under closing | random median; expected 0.05-0.2 | ≥ 0.5 | ≤ 0.2 |
| Re-read `L01 − L00` | | exactly 0 | equals `amend3.json` `M` within 1e-3 (exactly 0 on an L40); exit 1 | 13.8 | 5.7-6.4 |
| Reported only: (2,3)'s target share | | | | no expected value; W₀'s second prefix head, slot opens as far as (2,4)'s | |

Noise: row sd by `copying.share_sd`'s bootstrap on the closing share; three-seed range 0.09 from E62; 20 permutations for
the cosine floor. Blind to: pieces W₀'s copying uses at the first A or at B rather than at the second A (`a` is read at one
position); a redirected query that lies in the weight change (the masked forward holds W₀'s q, k, v: account 4, unchanged);
Γ's mask at the second A on a window whose true next token is the marker (the model does not know that, so W₀'s prediction
there is its induction); one model, one marker, step 30, 32 keys, 200 windows, three seeds. Positive control: the control
row itself, W₀'s own copying through its own mask, is what `S_old` and the closing read are calibrated on.

**Granularity limit.** A new circuit built from old pieces reads as elicited to VPD. `shared_pieces` says the state's
mask opens the pieces W₀'s copying opens and the gain runs through them, at the level of VPD's rank-one pieces; it cannot
say the same circuit is reused rather than a new circuit of those pieces, and the head label is at the level of heads.

**Cost and where.** One MATS `debug` job, L40, under 10 min (Amendment 3's gate reads took about a minute per row), $0.
Code: about 150 lines in `experiments/induction_cue/amend4.py`, importing `Reads`, `NAMES`, `map_of` from `amend3.py` (its
`gates` returns per-position gates, `masked` takes a gates list, so editing gates by hand needs no new forward code);
`copying.read_lo` at `half − 1` for the target rows; `copying.ablated` for the head off in the forward only, never in
Γ's pass. New rules in `src/commute/copying.py`, tests in `tests/test_copying.py`; predictions in
`results/predictions_amend4.json`, committed before the job. Not gated; one fresh review of the code before the job, as the
rule says. Control first and alone, exit 1 on a miss.

## B. The two queued $0 reads

**(1) The weighted slot read, then re-opening by slot share, for W₀'s low masked lean (0.37 against 1.58).** Not needed
before A, and not filed as is. The reading it blocks is only "the state's mask draws on a mechanism W₀'s masked forward
runs *at the marker*" (Amendment 2 row 4's bracket and the lean comparisons). Option 1 does not build on the marker lean:
it calibrates on W₀'s mask at the second A, where W₀ copies for real, and its control reads the anomaly more directly
than a re-open: if W₀'s mask carries the head's copying at the second A and shuts it at the marker, the low lean is the
mask doing what its objective asks. The hypothesis, offered as a hypothesis and not a cause: at the marker W₀ gives B
probability 4.3e-4, so a 1.58 log-odds lean there is a negligible KL term, and a sparse CI mask trained on KL drops it;
at the second A the same head's contribution is large and the mask should keep it. The control's two rows read exactly
that. If the control passes, the weighted re-open is cut; if it fails, the anomaly becomes the finding and the
weighted read is the next step to size it (then file it, with a reference slot).

**(2) A positive control over all 24 heads (recomputed share against target share, both maps).** Cut for now. Its
likely outcome is predicted by Γ's construction: the recomputed share tracks the target share for every head the
fine-tuned model uses, on both maps, because Γ reads that model's activations. Confirming that shows the recomputed
share is not an instrument, which the check already argues, and it would not move any reading in A. Run it later only if
someone wants to report a recomputed share as evidence of anything. The one head worth a number now is (2,3): W₀'s
second-ranked prefix head (0.051 under the bar of 0.077), whose slot opens at the copying states as far as (2,4)'s, with
its target share never read; it goes into Amendment 4 as a reported column, no rule.

## Design check

- **Goal.** `GOALS.md`, new capability or strengthened old: whether VPD's masks read the mechanism a causal head label
  names, on the only 67M map that has such a label. A step, not a patch.
- **Nearest tried.** E62 Amendment 3 (TRIED row 136): the masks react to the head through Γ; it could not say which
  pieces. Amendment 2's slot read covered 5% of one head's slot (check F2). E55 C2 (row 117) is cancelled, so no other
  decomposition-level mechanism test exists. Nothing has compared a state's mask with W₀'s own mask at a position where
  W₀ performs the behaviour. E59 and E64 read linear directions, not masks.
- **Decisions.** Row 73 (granularity limit) kept and stated. Row 112 (mechanistic definition; r a candidate instrument
  to score against a causal check): this is that scoring. Row 114 (held-out rule demoted): not read. No departure.
- **What changes next.** `shared_pieces` → the claim and the Qwen expectation; `not_shared` → the stop; `control_failed`
  → the masks drop W₀'s copying, to the user.
- **Apparatus first.** The W₀ control (W₀'s own mask at the second A, intact and head off) runs first and alone with
  exit 1; the re-read of `M` against `amend3.json` is the determinism check.
- **Class.** 1, analysis of saved states with a GPU pass of the existing harness: approved, no sign-off. The lead still
  tells the user the Amendment 3 reversal first, as the plan requires, and says this read is what follows it.

## Not run

Nothing here ran. Every number quoted is from `run.json`, `amend2.json`, `amend3.json`, the reports and the checks; W₀'s
log-odds of B at the second A and the size of `S_old` are unknown until the control runs. Not proposed: ablation inside
Γ's pass; zero ablation; steps 100 and 300; the head off on `L11` and `U1`; a second decomposition seed (the only check
that narrows the granularity limit, row 73).
