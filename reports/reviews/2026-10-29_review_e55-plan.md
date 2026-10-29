# Review of the E55 plan (reuse line, Stage 1)

Reviewer: fresh Opus agent, 2026-10-29. Read: `reports/plans/2026-10-29_plan_reuse-stage1.md`, the brief, the Stage 0
analysis, `jain_notes.md`, the Jain et al. text (L-refs below), `.claude/CLAUDE.md`, `decisions.md` rows 81-82.

**In plain words.** The Jain grammar reproduction is a good anchor and the plan quotes the paper's numbers correctly.
But the rule that would label a fine-tune "reuse" or "new" does not yet test what the brief asks for, which is whether
the fine-tuned behaviour runs through count-a. Two of its three checks can read the same on every arm built on the
low-prior base, and the paper's own figures predict that. A count-matched patching test fixes this. It also makes the
plan simpler and gives Stage C more arms to score.

## Blocking

**B1. C1 ablation cannot tell W from N1, and its expected values contradict each other.** At the answer read-out
position, late blocks carry the answer number whatever operand is counted. A vocab-sized count-a probe's top-8
directions will overlap that shared number subspace. Projecting them out then breaks count-b in N1 as well. The plan's
own bars need two incompatible things. R ≥ 0.50 (R's task is count-b, and its base already counts b at 99.7%) can only
happen if the count-a directions also carry count-b. N1 ≤ 0.10 needs the opposite. Zero-projection also pushes the
residual off-distribution, and the random rank-8 baseline (8 of 192 dims) is too weak to show specificity.
*Fix (minimal):* score an interchange test, and report C1 only.
- Run each arm on its own task prefix (`C b 40`) and on C_Te = 0 strings.
- At a block l* chosen on the base model's count-a probe, patch the probe subspace from a donor string. The donor has
  the same count(b) and a different count(a).
- Read the share of outputs equal to count_a(donor)+1. W should be high; N1 should keep count_b(base), so low.
- A shared number subspace cannot fake this, because the donor's count(b) equals the base's.
- Add a behavioural check that needs no probe: on C_Te = 0 inputs, does W output count_a(x)+1? Table 6 gives W1 0%
  count-b accuracy there, so W must be outputting something else; this says whether it is count(a)+1.
- Positive control in A1, before any fine-tune: the same patch on LOW's count-a task moves the output to
  count_a(donor) (filed bar). Use mean-ablation, not zeroing. For the control subspace, use another probe's directions
  matched in variance.

**B2. C2 (pruning) and C3 (reverse fine-tuning) measure whether count-a can be revived. They do not measure whether
the fine-tune goes through count-a.** Jain's figures already predict that they will not separate N1 from W.
- Fig. 76/77 (L3833-3875): after a 1e-4 fine-tune (W2's and N1's rate), "revival ... is observed for all values of
  C_Tr", C_Tr = 0 included. So N1 passes C3.
- Fig. 9 (L442-453): the low-prior base revives in about 3K iterations, which is above the plan's W bar of ≤ 2.5K. So
  W may fail C3.
- Combined with B1, the 2-of-3 rule can read N1 as "reuse" (C1 and C3) and W as "new".
- The brief's causal clause allows revival evidence ("and/or"). The question asked, though, is whether the
  fine-tuned map is g∘f_C, and revival is evidence about the base capability surviving, not about routing.

*Fix:* make the interchange test (B1) the ruling check. Move pruning and reverse fine-tuning out of the rule: they
become scored instruments next to VPD and steering. This also ends the circularity in Stage C, where pruning is part of
the truth in Stage A and an instrument in Stage B.

**B3. The arms' labels name the wrong function or overstate what is known.**
- **R.** R's fine-tune task (count-b) is something its base already does at 99.7%. What R reuses is count-b, with g
  the identity. Reading R through count-a (probe, pruning, revival) is a category error: count-a stays at 73.7% after
  the fine-tune (Table 6), so a fast "revival" only shows it was retained.
- **N1.** N1 is "does not go through f_C = count_a". It is not "new mechanism". It reaches 98.9% at 1K iterations,
  which is consistent with re-binding an existing counter to a new operand, and a sceptic will call that reuse.
- **The possession conjunct.** Every scored base possesses count-a, so this conjunct never separates the scored arms.

*Fix:*
- State f_C for each arm. For R, either set f_C = count_b, tested on the base with a count-b probe, or keep R only as a
  do-nothing-like control.
- Give N1 the label "not through count-a". Add one line naming the reading the plan cannot exclude (a shared counter
  with a new operand).

**B4. Under Def 2, Stage B's "new by construction" arms are reuse.**
- **The cipher.** Its output is π applied to the predicted next word, so it is g∘f_C with f_C = next-word identity
  and g = π. A linear probe for π(y) is the next-word probe with its rows permuted, so Def 1 reads it as possessed. On
  held-out words a softmax probe cannot output unseen classes and reads 0. The filed bar "≤ control + 0.1" is
  therefore met trivially or failed trivially.
- **Copy and the hand write.** Each is a lookup g on top of token identity. Held-out keys test whether g generalises,
  not whether f_C is used.
- **The decision tree.** Step 4 stops the line if the cipher reads "reuse", which is the reading Def 2 predicts.
  Step 4 also treats "lowercase is not reuse" as a failure. That is a label set before the reading, against
  "Real post-training arms get no elicit/new label before their readings".

*Fix:*
- Fix f_C = context case for the whole of Stage B. The cipher, copy and hand write are then only "not through context
  case" controls; drop the held-out-key rows.
- Keep lowercase with E53's cased-context read as its separating number.
- Remove the expected verdicts from step 4.
- Copy fails possession by behaviour (E52 B4), so it reads "neither" and cannot enter Stage C. Say so.

## Should fix

**S1. The probe position is ambiguous, and the ambiguity can make the probe vacuous.**
- "At Ans" (from L1247): the Ans position's input token is the answer itself. Read at the ART (`=`) position.
- Say which prefix each probe and patch is fit on: `C a 40` for possession, the arm's own `C b 40` for the causal
  tests.
- W1 (1e-5, low prior) is exactly where Fig. 8b (L424-429) shows the count-a probe dropping after the fine-tune.
  Choose l* on the base, not on the fine-tuned model.

**S2. The thresholds are neither sourced nor measured, and the noise floors are planned in the wrong place.**
- No threshold has a source: 0.90, +0.30, ≤ 0.40, 0.50/0.10, 0.80/0.30, top-8, 2.5K, 15 points. Jain has none; the
  Stage 0 analysis says so.
- A0 (500 iterations) cannot measure probe-seed spread on a model that does not count yet.
- *Fix:* A1 measures four things on the real pretrained models: probe-seed spread, the spread of the
  control-subspace patch, the majority-class rate, and the positive-control patch effect. The A2 bars are then filed
  as fractions of those numbers, dated, before A2 runs. For lowercase (a binary label with a large majority class),
  use balanced accuracy.

**S3. The rule's edge cases are incomplete.**
- A baseline that misses its bar counts as a failed check, which pushes the arm toward "new". It should void the
  check, and fewer than 2 valid checks should read "neither".
- R's random-K pruning keeps count-a at about 0.74, above the ≤ 0.30 bar, so C2 fails for R. The "ceiling" skip is
  undefined.
- "The same procedure on S": S means either NEVER or NEVER after its count-a fine-tune. The second has count-a and
  fails the bar.
- With S capped at 5K, the W bar (≤ 2.5K) and the N bar (≥ 0.5 × S = 2.5K) coincide, leaving no margin.
- "Neither" is reachable only through possession.
- Most of these go away with B2's fix.

**S4. Count-c is not a selectivity control.** The 40-token window holds only {a, b, c}, so
count(c) = 40 − count(a) − count(b), a linear function of variables the base has. It also confounds the NEVER control.
*Fix:* use shuffled-label probes, a randomly initialised model, and one label the task never needs (for example,
count(a) in window positions 41-80 from the end).

**S5. Recipe gaps.**
- **The text window.** "First 250 terminals, pad shorter" contradicts L1025-1026, where a random subsample is used
  "to remove bias towards tokens likely to be at the beginning". The mean expansion is about 276 terminals (computed
  from Fig. 14), so a large share of samples are shorter than 250. Padding inside the last 40 positions also breaks
  a + b + c = 40. Take a random contiguous 250-terminal window and resample when the expansion is shorter.
- **Spurious insertion.** The count is over the last 40 positions. Insert only there, keep the length fixed, and
  reject or replace when count(b) > count(a)+1 already.
- **The reverse fine-tune source.** L1249-1250 says reFT uses the *Tracr* fine-tuning hyperparameters (SGD with
  momentum, 2K warm-up). The plan calls it the PCFG recipe. State this as a departure.
- **Learning rates.** Fig. 76/77 run reFT at 1e-5 and 1e-6 after a 1e-4 fine-tune. The header says both 1e-6 and 1e-5;
  the arm table and C3 say 1e-6 only. Pick one, and note that Jain's Scr.+FT converged only at 1e-5 (L453-455).
- **NEVER = (0, 1, 0).** The `a` operand embedding then never trains, so S is slow for an extra reason and the NEVER
  probe on `C a 40` reads out-of-distribution inputs. Use Jain's (0.001, 0.999, 0); the plan's attribution of that
  prior is not in the text, so cite or drop it.

**S6. Stage C cannot beat chance.**
- Four arms: P(≥ 3 of 4 agree by chance) = 5/16 = 0.31.
- C2's three arms: P(3 of 3) = 0.125.
- PC1 is partly known already: E12 and E52 measured r and S on copy and cipher.
- *Fix:* add LOW-based arms from Table 6's grid (C_Tr ∈ {0, 0.5, 0.8, 1} × lr ∈ {1e-4, 1e-5, 1e-6}, minutes each),
  each labelled by the interchange test. They all share LOW's one decomposition, so C2 scores about 12 arms for the
  same cost. This is what makes C2 worth paying for; without it, defer C2.

**S7. Stage B's C2 and C3 have no valid baseline.** No 67M scratch model exists, and the cipher model never lost case.
Drop them from Stage B (fewer measurements).

## Minor

- **Citations check out.** All five Table 6 rows quoted are correct (L3006-3075: W1, W2, N1, M, R; pretrained count-b
  is 5.17%). L1160-1196, L1238-1248 and L716 match.
- **50K iterations instead of 200K is fine.** Table 6 is complete, and LOW counts b less well at 50K (5.17% vs 27.1%),
  so N1 is cleaner.
- **Pretraining batch 96.** It is borrowed from the fine-tune. Jain says the setup follows Allen-Zhu & Li 2023: check
  that paper's pretraining batch and context first. Adopt the other minGPT trainer defaults too (betas 0.9/0.95, grad
  clip 1.0) and cite them.
- **"≥ 0.80 at K ≤ 15 (Fig. 7)".** Fig. 7 shows only 1e-5 and 1e-6. At 1e-4, Fig. 79a (L3933) supports a wrapper only
  at C_Tr = 1, and its sizes are only plotted.
- **Cost.** It is realistic. The debug QoS limit is 2 h (`cluster/README.md`), so A1 fits.
  - Generate the grammar text once and label it per prior; the text does not depend on the operand prior.
  - A 3M-parameter decomposition may fit a 2 h debug job, so rehearse C2 there before RunPod.
- **Ordering.** Control first holds (A0 → A1 → A2), once the B1 positive control is added to A1.

## Simpler design that answers the same question

1. **Ground truth.** The interchange test plus the behavioural g check, on LOW-based arms from Table 6's grid.
   Possession is checked by probe on the bases only.
2. **Instruments.** Pruning, reverse fine-tuning, steering, and VPD (C2) are each scored on those arms.
3. **67M.** Lowercase with the cased-context read only.

This drops C1's ablation, the 2-of-3 rule, and Stage B's cipher, copy and hand-write rows. It cuts the number of
filed bars by about half and gives Stage C enough arms to score.

## Not checked

- Figure values: the PDF was not rendered, so only captions and text were read.
- Allen-Zhu & Li's pretraining batch.
- E53's and E46's state code for Stage B reuse.
- `task_audit.md` row by row.
