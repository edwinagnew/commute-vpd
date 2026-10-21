# Recruit test: the real fine-tune and a LoRA recruit the same pre-existing subcomponents as our map, and the set is target-specific

**Takeaway.** With the released components left untouched and the update's δW put into Δ, the frozen Γ switches on
about 200 pre-existing subcomponents at the emoticon trigger. The real W fine-tune's set overlaps the pull-back map's
(Jaccard 0.64) and a one-matrix LoRA's (0.82) against a seed-to-seed floor of 0.93, and overlaps a fine-tune to a
different target token at the same trigger by only 0.04 to 0.05, at matched update size. Re-gating the W₀ components
alone reproduces 60% of the fine-tune's trigger gain (54% for the LoRA), but the ~200 switched-on subcomponents carry
only 40% of that re-gating effect (P4 failed), so the overlaps compare which gates moved, not what produces the
behaviour. Within that limit, "reveal through routing" is a property of the update, not of our map, and of *what* was
learned, not of a perturbed trigger. "LoRA through the pull-back" (training the subcomponents with the LoRA's support
and rank) has still never been run; the LoRA arm here is a normal LoRA placed do-nothing. Two things did not go to
plan: the LoRA-locality gate rested on a wrong premise (Γ reads every matrix's input at every position, so a layer-2
edit moves layer-0 gates too), and one null target was mis-selected by a bug (still a valid different-target null).

*2026-10-19, coding agent. Plan `reports/2026-10-19_plan_recruit-test.md`; predictions
`experiments/recruit_test/results/predictions.json` (filed before the run, re-filed twice before the run for the LoRA
checkpoint rule and the G5 tolerance); results `results/recruit.json`, per-arm dg vectors in `results/large/`. RunPod 4090
pod `commute-recruit-1`, $0.26, 106 s for the full run. Driven by a fork from the user's session. Figures and the full write-up for reading: `notebooks/E06_recruit_test.ipynb`.*

**Terms.** *Pull-back*: training the subcomponents' (U, V) on the fine-tune loss with Γ and Δ fixed (our map Φ).
*Do-nothing placement*: δW is added to the target weights and left in Δ; the components are unchanged, so the only thing
that changes in the CI-masked model is which subcomponents Γ gates on, because Γ reads the new activations. *r*: the
re-gating share, the fraction of the held-out trigger log-odds gain that the W₀ components reproduce under the new gates
(`local_faith.route_split`; here d = i = 0 and e_Δ = 1 exactly, e_gate = −r). *S_on*: alive subcomponents whose mean gate
at the trigger over the 73 held-out windows rises by more than 0.3. *Jaccard*: |A ∩ B| / |A ∪ B| of two S_on.
*Recovery*: the share of the re-gating effect obtained when only S_on takes the new gates and every other subcomponent
keeps W₀'s.

## Scores (as filed)

- **G0 seed floor: J(ft_s0, ft_s1) = 0.93, ok.** G3 identities hold for every arm. **G4**: the pull-back reference
  reproduces amendment 2 exactly (n_on 135, gap 5.897 vs 5.893). **G6**: every arm learned (P(target) ≥ 0.998).
- **G5 LoRA locality: FAIL, wrong premise** (max |dg| in layers ≤ 2 is 1.0, tolerance 1e-6). See surprises.
- **P0 target-specific: pass.** J(ft_s0, null) 0.042 and 0.050 against the floor 0.93 (rule ≤ 0.63). The strong kind of
  pass: the nulls' n_on (117, 132) and rel_norm_max (0.0102, 0.0105) are at or below ft_s0's (202, 0.0113).
- **P1 real update recruits the map's set: pass.** J(ft_s0, pullback_adam) = 0.64 (rule ≥ 0.5).
- **P2 re-gating share of the real update: pass.** r(ft_s0) = 0.60, gain 13.5 nats (predicted about 0.5).
- **P3 LoRA re-gates: pass.** r = 0.54; J restricted to layers ≥ 3 with ft_s0 = 0.86 (whole set 0.82).
- **P4 S_on carries the re-gating: FAIL.** Recovery 0.398 against the rule ≥ 0.4 (random same-size control 0.011;
  ft_s1 0.406, pull-back 0.446). The switched-on set carries 40% of the re-gating effect; 60% is outside it, in gates
  that moved by less than 0.3, in the ~130 subcomponents that switched off, or in interactions. Not measured. So the
  Jaccard results compare which gates moved, not what produces the behaviour.
- **P5 random δW is inert: pass.** Gain −0.015 nats, n_on 0. The Frobenius-matched null has no power, as predicted.
- **R1, R2 (natural `o` machinery): not run.** 256 Pile val rows hold 96 positions whose next token is `o`; the rule
  needed 200.

## Per arm (held fixed: components, Γ, windows; varied: δW)

| Arm | step | gain (nats) | r | n_on 0.3 (layers 0/1/2/3) | n_off 0.3 | recovery (random) | rel. norm |
|---|---|---|---|---|---|---|---|
| ft_s0 | 10 | 13.5 | 0.60 | 202 (37/5/40/120) | 132 | 0.40 (0.01) | 0.0113 |
| ft_s1 | 10 | 13.6 | 0.61 | 209 (39/4/39/127) | 127 | 0.41 (0.03) | 0.0109 |
| lora_r8 | 10 | 16.9 | 0.54 | 237 (39/5/53/140) | 126 | 0.38 (0.02) | 0.0239 |
| null_a (`<|endoftext|>`) | 10 | 17.7 | 0.55 | 117 (13/9/23/72) | 162 | 0.27 (0.03) | 0.0102 |
| null_b (` http`) | 10 | 13.0 | 0.60 | 132 (10/3/33/86) | 135 | 0.56 (0.02) | 0.0105 |
| pullback_adam | 30 | 11.6 | 0.51 | 135 (16/1/24/94) | 67 | 0.45 (0.02) | 0.0066 |
| random_dw | – | −0.02 | n/a | 0 | 0 | n/a | 0.0113 |

LoRA r by step: 0.24, 0.27, 0.31, 0.47, 0.54, 0.57, 0.59 at steps 1, 2, 3, 5, 10, 30, 100. The pre-registered step 10
was also the checkpoint closest to ft_s0 in held-out log-odds.

**Jaccard at 0.3.** ft_s0|ft_s1 0.93; ft|lora 0.82, 0.84; ft|pullback 0.64, 0.63; lora|pullback 0.56; ft|null 0.04 to
0.06; null_a|null_b 0.11; anything|random_dw 0. Thresholds 0.1 and 0.5 give the same ordering (ft|pullback 0.71 / 0.64,
ft|null 0.06 to 0.08 / 0.04 to 0.05). The nulls' r values (0.55, 0.60) equal the `o` arms', so **r is not
target-specific; the set is.**

## Surprises

1. **Γ is global, so the LoRA-locality premise was wrong.** The released Γ (`GlobalSharedTransformerCiFn` in the pinned
   `param-decomp`) rms-norms and concatenates all 24 matrices' inputs at each position, then runs a bidirectional
   transformer over positions before splitting into per-matrix gates. Every gate depends on every matrix's input at
   every position in the window. A layer-2 LoRA therefore moves layer-0 gates (39 switched on), and the "layers ≥ 3"
   restriction in P3 was unnecessary (P3 passes either way). This also sharpens the "blind to" line of every
   trigger-local gate measurement in this project: the gate at the trigger reads the whole window, so "re-gating at the
   trigger" includes re-gating driven by changed activations anywhere in the window. The task ladder's G1b argument
   (`h.3.mlp.down_proj`'s output feeds no decomposed matrix, so Γ's inputs cannot change) still holds.
2. **null_a was mis-selected.** The filed rule took the plain token closest above P_W0(`o`) at the trigger; no plain
   token sat above it, and `argmin` over an all-inf tensor returned token 0, `<|endoftext|>` (P_W0 0.001). The arm is
   still a fine-tune to a different target at the same trigger, learned (P 0.9994) and gives the same Jaccards as null_b
   (` http`, the correctly chosen token below). The code now takes the two plain tokens closest to P(`o`) in log
   probability, with no sides; the recorded run used the old rule. Not re-run (the conclusion does not depend on it).

## What this says and does not say

- **Says** (bears on GOALS "Where an update goes", reveal): the recruited set is a property of the real update (P1, P3)
  and of the target (P0), and re-gating alone carries more than half of the trigger effect for a W fine-tune and a
  LoRA alike (P2, P3). The ladder's do-nothing r can now be read as recruitment of a target-specific set, not as a
  generic response of frozen Γ to a perturbed trigger.
- **Says**: a Frobenius-matched random δW is a null with no power for anything gate-based (P5).
- **Does not say** whether the recruited subcomponents are the model's existing `o` machinery (R1, R2 not run), what
  the switched-off set (about 130 per arm, more than half as large as S_on) does, whether any of this holds for a task the
  model lacks (the ladder's rungs), or anything about updates larger than 10 steps. r being equal for `o` and the nulls
  means r measures how much a trigger-local behaviour can be routed through existing gates at all, not which behaviour.

## Not run

- The natural-`o` comparison (needs about 600 Pile rows for 200 positions, or a different definition of "natural").
- Seeds for the null arms; a LoRA on all 24 matrices; LoRA ranks other than 8.
- The recovery of the switched-*off* set, and what those subcomponents did at W₀.
- Any task other than emoticon; the W fine-tune's set at steps 1, 3, 30.
- Interpretability of the recruited set beyond its aggregate push towards `o` (amendment 2's B3).
- The same measurement with Γ allowed to move.

**Hand-off.** A new session that takes this further should read this report, `reports/2026-10-19_plan_recruit-test.md`
and `experiments/recruit_test/run.py`, and note that `run.py` predates the 2026-10-19 "One config" and Codex-review rules
in `.claude/CLAUDE.md`.
