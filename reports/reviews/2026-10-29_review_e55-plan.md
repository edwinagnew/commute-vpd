# Review of the E55 plan (reuse line, Stage 1)

Reviewer: fresh Opus agent, 2026-10-29. Read: `reports/plans/2026-10-29_plan_reuse-stage1.md`, the brief, the Stage 0
analysis, `jain_notes.md`, the Jain et al. text (L-refs below), `.claude/CLAUDE.md`, `decisions.md` rows 81-82.

**In plain words.** The Jain grammar reproduction is a good anchor and the plan quotes the paper's numbers correctly.
But the rule that would label a fine-tune "reuse" or "new" does not yet test what the brief asks for, which is whether
the fine-tuned behaviour runs through count-a. Two of its three checks can read the same on every arm built on the
low-prior base, and the paper's own figures predict that. A count-matched patching test fixes this, makes the plan
simpler, and gives Stage C more arms to score.

## Blocking

**B1. C1 ablation cannot tell W from N1, and its expected values contradict each other.** Late blocks at the answer
read-out position carry the answer number whatever operand is counted. A vocab-sized count-a probe's top-8 directions
will overlap that shared number subspace, so projecting them out breaks count-b in N1 too. The bars need two
incompatible things: R ≥ 0.50 (R's task is count-b, which its base already does at 99.7%) only happens if the count-a
directions also carry count-b; N1 ≤ 0.10 needs the opposite. Zero-projection also moves the residual off-distribution,
and a random rank-8 subspace (8 of 192 dims) is too weak a baseline to show specificity.
*Fix (minimal):* score an interchange test; report C1 only.
- Run each arm on its own prefix (`C b 40`) and C_Te = 0 strings. At a block l* chosen on the base model's count-a
  probe, patch the probe subspace from a donor with the same count(b) and a different count(a).
- Read the share of outputs equal to count_a(donor)+1: high for W; N1 keeps count_b(base). A shared number subspace
  cannot fake this, because the donor's count(b) equals the base's.
- Add a probe-free behavioural check: on C_Te = 0 inputs, does W output count_a(x)+1? (W1's count-b accuracy there is
  0% in Table 6, so W outputs something else; this says whether it is count(a)+1.)
- Positive control in A1, before any fine-tune: the same patch on LOW's count-a task moves the output to
  count_a(donor) (filed bar). Use mean-ablation, and a variance-matched other-probe subspace as the control.

**B2. C2 (pruning) and C3 (reverse fine-tuning) test whether count-a can be revived, not whether the fine-tune goes
through count-a; Jain's own figures predict they will not separate N1 from W.**
- Fig. 76/77 (L3833-3875): after a 1e-4 fine-tune (W2's and N1's rate), revival "is observed for all values of C_Tr",
  C_Tr = 0 included. So N1 passes C3.
- Fig. 9 (L442-453): the low-prior base revives in about 3K iterations, above the W bar of ≤ 2.5K. So W may fail C3.
- With B1, the 2-of-3 rule can read N1 "reuse" (C1 + C3) and W "new". The brief's "and/or" allows revival evidence,
  but it speaks to the base capability surviving, not to whether the map is g∘f_C.

*Fix:* the interchange test (B1) rules. Pruning and reverse fine-tuning become scored instruments beside VPD and
steering. This also ends Stage C's circularity (pruning is truth in Stage A and an instrument in Stage B).

**B3. The arm labels name the wrong function or overstate what is known.**
- **R** reuses count-b with g = identity (its base already does count-b at 99.7%). Reading it through count-a is a
  category error: count-a stays at 73.7% after the fine-tune (Table 6), so fast "revival" only shows retention.
- **N1** is "does not go through f_C = count_a", not "new mechanism". It reaches 98.9% at 1K iterations, consistent
  with re-binding an existing counter to a new operand, which a sceptic will call reuse.
- **Possession conjunct:** every scored base has count-a, so it never separates the scored arms.

*Fix:* state f_C per arm. For R, set f_C = count_b (test with a count-b probe on the base), or keep R only as a
do-nothing-like control. Label N1 "not through count-a" and add one line naming what the plan cannot exclude (a shared
counter with a new operand).

**B4. Under Def 2, Stage B's "new by construction" arms are reuse.**
- **Cipher:** its output is π of the predicted next word, so it is g∘f_C (f_C = next-word identity, g = π). A linear
  probe for π(y) is the next-word probe with rows permuted, so Def 1 reads it possessed; on held-out words a softmax
  probe cannot output unseen classes and reads 0. The bar "≤ control + 0.1" is met or failed trivially.
- **Copy, hand write:** each is a lookup g on top of token identity; held-out keys test whether g generalises, not
  whether f_C is used.
- **Decision tree step 4** stops the line if the cipher reads "reuse", the reading Def 2 predicts. It also treats
  "lowercase is not reuse" as a failure: a label set before the reading, against the brief.

*Fix:* fix f_C = context case for all of Stage B. Cipher, copy and hand write become "not through context case"
controls; drop the held-out-key rows. Keep lowercase with E53's cased-context read. Remove the expected verdicts from
step 4. Copy fails possession by behaviour (E52 B4), so it reads "neither" and cannot enter Stage C; say so.

## Should fix

**S1. The probe position is ambiguous enough to make the probe vacuous.** At the Ans position (L1247) the input token
is the answer itself: read at ART (`=`). Say which prefix each probe and patch is fit on (`C a 40` for possession,
the arm's `C b 40` for causal tests). W1 (1e-5, low prior) is where Fig. 8b (L424-429) shows the count-a probe dropping
after the fine-tune, so choose l* on the base.

**S2. Thresholds are neither sourced nor measured.** None of 0.90, +0.30, ≤ 0.40, 0.50/0.10, 0.80/0.30, top-8, 2.5K or
15 points has a source (Jain has none; Stage 0 says so). A0's 500 iterations cannot give probe-seed spread on a model
that does not count yet. *Fix:* A1 measures probe-seed spread, control-patch spread, majority-class rate and the
positive-control effect; A2 bars are filed as fractions of these, dated, before A2. For lowercase (binary, large
majority class) use balanced accuracy.

**S3. Rule edge cases are incomplete** (most vanish with B2's fix).
- A baseline that misses its bar counts as a failed check, pushing toward "new"; it should void the check, and fewer
  than 2 valid checks should read "neither". As written, "neither" is reachable only through possession.
- R's random-K pruning keeps count-a near 0.74, above the ≤ 0.30 bar, so C2 fails for R; the "ceiling" skip is
  undefined.
- "The same procedure on S" may mean NEVER or NEVER after its count-a fine-tune; the second fails the bar.
- With S capped at 5K, the W bar (≤ 2.5K) and the N bar (≥ 0.5 × S) coincide at 2.5K: no margin.

**S4. Count-c is not a selectivity control.** The 40-token window holds only {a, b, c}, so count(c) = 40 − count(a) −
count(b), linear in variables the base has; it also confounds the NEVER control. *Fix:* shuffled-label probes, a
randomly initialised model, and one unneeded label (e.g. count(a) at positions 41-80 from the end).

**S5. Recipe gaps.**
- **Text window:** "first 250 terminals, pad shorter" contradicts L1025-1026 (random subsample "to remove bias towards
  tokens likely to be at the beginning"). The mean expansion is ~276 terminals (from Fig. 14), so many samples are
  shorter, and padding in the last 40 breaks a + b + c = 40. Take a random contiguous 250-window; resample if shorter.
- **Spurious insertion:** count is over the last 40 positions; insert only there, keep length fixed, and reject or
  replace when count(b) > count(a)+1 already.
- **reFT source:** L1249-1250 gives the *Tracr* FT hyperparameters (SGD + momentum, 2K warm-up), not the PCFG recipe.
  State the departure.
- **Learning rates:** Fig. 76/77 run reFT at 1e-5 and 1e-6 after a 1e-4 FT. The header says both; the arm table and
  C3 say 1e-6 only. Pick one; Jain's Scr.+FT converged only at 1e-5 (L453-455).
- **NEVER = (0, 1, 0):** the `a` operand embedding never trains, so S is slow for an extra reason and the NEVER probe
  on `C a 40` reads out-of-distribution inputs. Use (0.001, 0.999, 0); the text does not give that prior for Scr.+FT,
  so cite it or drop the attribution.

**S6. Stage C cannot beat chance.** Four arms: P(≥ 3 of 4 by chance) = 5/16 = 0.31; C2's three: P(3 of 3) = 0.125.
PC1 is partly known (E12/E52 measured r and S on copy and cipher). *Fix:* add LOW-based arms from Table 6's grid
(C_Tr ∈ {0, 0.5, 0.8, 1} × lr ∈ {1e-4, 1e-5, 1e-6}, minutes each), labelled by the interchange test. All share LOW's
one decomposition, so C2 scores ~12 arms at the same cost; that is what makes C2 worth paying for. Else defer C2.

**S7. Stage B's C2 and C3 have no valid baseline:** no 67M scratch model exists, and the cipher never lost case. Drop
them (fewer measurements).

## Minor

- **Citations check out:** all five Table 6 rows quoted (W1, W2, N1, M, R; L3006-3075) are correct; pretrained count-b
  is 5.17%. L1160-1196, L1238-1248 and L716 match.
- **50K vs 200K is fine:** Table 6 is complete, and LOW counts b less at 50K (5.17% vs 27.1%), so N1 is cleaner.
- **Batch 96** is borrowed from the fine-tune; first check Allen-Zhu & Li 2023 (which Jain follows) for batch and
  context. Adopt and cite the other minGPT trainer defaults (betas 0.9/0.95, grad clip 1.0).
- **"≥ 0.80 at K ≤ 15 (Fig. 7)":** Fig. 7 shows only 1e-5 and 1e-6. At 1e-4, Fig. 79a (L3933) supports a wrapper only
  at C_Tr = 1, with sizes only plotted.
- **Cost is realistic;** debug QoS is 2 h (`cluster/README.md`), so A1 fits. Generate the text once and label per
  prior (it does not depend on the prior). A 3M-parameter decomposition may fit a 2 h debug job: rehearse C2 there.
- **Ordering:** control first holds (A0 → A1 → A2) once A1 gets the B1 positive control.

## Simpler design that answers the same question

Truth: the interchange test plus the behavioural g check on LOW-based arms from Table 6's grid, with possession probed
on the bases only. Instruments: pruning, reverse fine-tuning, steering and VPD (C2), each scored on those arms. 67M:
lowercase with the cased-context read only. This drops C1's ablation, the 2-of-3 rule and Stage B's cipher, copy and
hand-write rows, halves the filed bars, and gives Stage C enough arms to score.

## Not checked

Figure values (PDF not rendered; captions and text only); Allen-Zhu & Li's pretraining batch; E53/E46 state code;
`task_audit.md` row by row.
