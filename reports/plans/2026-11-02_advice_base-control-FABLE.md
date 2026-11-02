# Advice: a base-only control for teacher training and the other maps (Fable, 2026-11-02)

**Answer first.** For unmasked teacher training (E34, E41, E42 arm A/B, E43) the gap the user fears cannot enter the
loss: the student is the stored W₀ model plus a forward hook adding only the offsets' contribution
(`src/commute/component_training.py` → `Offsets.extra`, `hooks`), and Δ = W₀ − UVᵀ, so at zero offsets student and
teacher are the same bytes. E34's C2 (`experiments/teacher_training/run.py` → `c2`) reads 0 by construction and tests
only `teacher.kl_exact_zero`'s arithmetic: a tautology, not a control. The gap is real on the **component path**
(target rebuilt as Δ₀ + UVᵀ each step: `masked_copy.py`, `masked_teacher.py`, E15's refit): G2 path KL 0.0016 nats on
the L40, 0.00065 on the H100, plus the masked models' own error at W₀ (CI-masked 1.13, stochastic 0.74 on cipher
windows). Every recipe learning from that gap had a teacher = W₀ control and every one failed the check at 3.8× to
49× W₀ (E37 A1 M0/M0b, E42 A1, E42 A2 ×3): there the worry is already shown and is why the masked line ended. Never
run: the gap-only control for the unmasked loss on the component path, and the hook-path control that can fail,
teacher = W₀ from a start that is not W₀. Nothing standing breaks; two attributions get a caveat (below).

## 1. Why a base-only control is needed, and which version can fail
1. **Decomposition error as a training signal** (the user's point): the loss sees D(W₀) ≠ W₀ and the map refits VPD
   instead of transporting δW. Hook path: excluded by construction (C2 = 0 exactly). Component path: real, shown to
   break the check for every masked term; untested for the unmasked term alone (0.0016 nats).
2. **Adam's scale blindness.** Any nonzero gradient becomes lr-sized steps, so a 1e-3 gap moves (U, V) per step as
   much as a 1-nat one (`kl_exact_zero` exists because 1e-8 would have). Only a tiny-but-nonzero gap reads this; C2
   cannot. Plain SGD lacks this failure mode.
3. **Prediction matching under-determines (U, V).** E34 C3: a reachable teacher's gap closed to 0.5% with 0.76 of its
   offsets recovered; the check decides among (U, V) with the same logits. Teacher = W₀ from a start away from W₀
   reads whether the method's basin is a faithful decomposition of W₀. E37 A1 arm 3 (adjoint start, check 1.05 → 1.84)
   is the confounded version: teacher ≠ W₀, so copying δW and the method's own drift are mixed.
4. **Data.** Arm B learned emoticon and the rand1 cipher from Pile text. Hook-path teacher = W₀ is 0 on any text, so
   this cannot separate data from δW; the component-path control on Pile rows vs windows reads whether the path's
   drift depends on the text (expected not).
5. **Card.** The check's gap at W₀ is card-dependent (floor 0.0748 H100 vs 0.209 L40; E34's r0 do-nothing 1.35 vs
   1.97), so a reading within 30% of a line on one card is not reproduced on the other: a control runs on the card
   of the experiment it amends.
6. **Other methods.** Component training with the task loss (E7-E21): KL to stored W₀ on the hook path plus the
   task's CE, so no base-only version exists. E15's Γ refit's W₀ control drifted (recorded). Re-gating reads a frozen
   Γ; do-nothing and the splits never train. A perturbed-start control's null is the start state's own reading.

## 2. What to run (at most three), in this order
**B1. Component path, unmasked, teacher = W₀.** `masked_copy.train` with `coeff 0` (E42 A1's recipe, Adam 1e-5,
batch 32, 1,000 steps) on rand32 windows, then on 1,024 Pile rows. MATS L40 debug, ~12 min each, $0. Step 1:
unmasked term 0.00163 on rand32 (= G2 of jobs 40338/40497); a 0 gradient is a bug. Expected end: size ≤ 0.01 of W₀
(E42 copies 0.026), damage ≤ 0.005, check ≤ 1.24 (pass). If it fails (≥ 1.81): the component path with Adam breaks
faithfulness with nothing to copy, so E42 A1/A2's and E37 A1 M0's "masked terms did it" become "path + Adam, masked
terms worsen it" (both filed as untested), E15's refit drift gets the same candidate cause, and the masked line is
reopened on the hook path.
**B2. Hook path, teacher = W₀, perturbed start.** E34's recipe (`component_training.train`, Adam 1e-5, 1,000 steps)
on emoticon windows, `init` = (a) random (dU, dV) at E42's copy size (rel 0.026), (b) the scaled adjoint split of
E41's T1000 emoticon δW. References: each start's own check, P, damage. H100 ~15 min, ~$1 (E41's card). Start: gap to
W₀ = the start's damage, nonzero. Expected: gap closes to ≤ 1% of its start, check ends ≤ max(1.0, start's), offsets
return ≥ 0.76 of the way (C3's figure). If the gap closes but the check is ≥ 1.61: the method matches predictions
and leaves an unfaithful decomposition, so E34/E41/E42's passes hold for starts at W₀ only (E41 A2's chain starts
each link from the last copy: caveat), and E37 A1 arm 3's rise is the method's drift, not the broad update's. If (a)
heals and (b) does not, the adjoint split's basin is the issue.
**B3. B1 on E37's data.** Teacher = W₀, component path, coeff 0, batch 8, 1,000 steps on the 2,400 lowercase rows
(E37 A1's M0 without masked terms). H100 ~10 min, ~$0.7; only if B1 passes. Expected: check ≤ 1.40, damage ≤ 0.01;
a fail says the 512-token rows, not the path, carry the drift. B1-B3 together: under $2, one afternoon, no filed
prediction changes.

## 3. What would change this advice
If the masked line stays closed and no live method uses the component path, B1 and B3 are attribution only: run B2
alone. If E34 C3's 0.76 offset recovery is accepted as answering under-determination, B2 shrinks to start (b). If a
Qwen rehearsal is imminent, B2 belongs in its gate-3 rehearsal, not a separate 67M run. No number above was
recomputed; E42 A2's unmasked-at-W₀ value is `G2.path_kl` in `teacher_training_e42_amend2_rand32.json`.

Friction: E34's C2 is recorded as "teacher = W₀ moves nothing" in TRIED.md and the log; it should say "0 by
construction (hook path), tests the gradient arithmetic only", or the next reader repeats this question.
