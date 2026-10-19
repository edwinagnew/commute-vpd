# Plan: the recruit test. Does the *real* update recruit the subcomponents our map recruits, and is the set target-specific?

*2026-10-19, coding agent. Confirmed by the user ("go for it on runpod") after the design check in chat. Predictions:
`experiments/recruit_test/results/predictions.json` (filed before training). Code: `experiments/recruit_test/run.py`,
bookkeeping in `src/commute/recruit.py` (CPU tests `tests/test_recruit.py`). RunPod 4090, under $1. The task ladder
(round 3) is running at the same time on its own pod, driven by another session; this test does not touch it.*

**Terms.** *Pull-back*: training the subcomponents' (U, V) on the fine-tune loss with Γ and Δ fixed (our map Φ).
*Do-nothing placement*: the update's δW is added to the target weights and left in Δ; the released components are
unchanged, so the only thing that can change in the masked model is which subcomponents the frozen Γ gates on, because Γ
reads the new activations. *Re-gating share r*: the fraction of the held-out trigger log-odds gain that the W₀ components
reproduce under the new gate pattern (`local_faith.route_split`). *Switched-on set S_on*: the alive subcomponents whose
mean gate at the trigger rises by more than 0.3 between W₀ and the state.

**Why.** The concentration test's amendment 2 found that every pull-back arm, placed do-nothing, switches on the same
135 to 250 pre-existing subcomponents at the emoticon trigger (Jaccard 0.59 to 0.88), which carry about half of the
re-gating effect and push weakly towards `o` at W₀. That is the current evidence for "reveal". But every set in that
comparison came from our map. The real fine-tune's re-gating has never been measured on this task (the ladder's
do-nothing arm will give its r per task, but saves no set and has no LoRA arm), and no null has shown that the set is
about *what* was learned rather than about *a perturbation at the trigger*.

**Arms** (all do-nothing placement, 73 held-out emoticon windows, Γ frozen):
- `ft_s0`, `ft_s1`: the W fine-tune, round 1's recipe, step 10, optimiser seeds 0 and 1 (seed-to-seed Jaccard is the noise floor).
- `null_a`, `null_b`: the same recipe and windows, target = the plain token whose W₀ trigger probability is closest above /
  below P(`o`). The null with power: a Frobenius-matched random δW is inert (round 1), so it cannot test specificity.
- `lora_r8`: rank-8 LoRA on `h.2.mlp.down_proj` (the paper's site), scored at step 10 (the fine-tune's step count and the
  lift test's scored LoRA checkpoint; by the old curves P(`o`) 0.9994 against the fine-tune's 0.9977); every checkpoint
  reported, and the one closest to `ft_s0` in log-odds logged. Only layers 3+ can re-gate (Γ reads raw matrix inputs), G5.
- `random_dw`: Gaussian δW with `ft_s0`'s norms, to show the powerless null is powerless (P5).
- `pullback_adam`: the pull-back's δW placed do-nothing, i.e. amendment 2's reference arm reproduced (G4).

**Measured per arm.** Route cells and shares (d = i = 0 and e_Δ = 1 must hold, G3); dg per subcomponent; S_on and S_off at
thresholds 0.1 / 0.3 / 0.5 with per-layer counts and the top 15; recovery of the re-gating effect by S_on against a
random same-size set. Across arms: the full Jaccard matrix, and restricted to layers 3+ for the LoRA comparison. At W₀:
the subcomponents gated on where `o` is the next token in 256 Pile val rows (S_nat), if at least 200 such positions exist.

**Scored** (thresholds in `predictions.json`):
- G0 (floor): J(ft_s0, ft_s1) ≥ 0.5, else the set-based predictions are reported, not scored.
- P0 (target-specific, the Jaccard metric's positive control): J(ft, null) ≤ floor − 0.3 for both nulls. Predicted pass.
- P1: J(ft_s0, pullback_adam) ≥ 0.5. Predicted pass.
- P2: r(ft_s0) ≥ 0.3. Predicted pass, about 0.5.
- P3: r(lora) ≥ 0.3 and J₍≥3₎(lora, ft_s0) ≥ 0.4. Predicted pass.
- P4: recovery(ft_s0) ≥ 0.4, random control < 0.1. Predicted pass.
- P5: random δW has gain < 1 nat and fewer than 10 switched-on subcomponents. Predicted pass.
- R1 (conditional on S_nat): J(ft_s0, S_nat) ≥ 3× random same-size sets. Predicted pass.
- R2 (conditional): J(ft_s0, S_nat) exceeds the nulls' by ≥ 0.1. Predicted null.

**What each metric is blind to.** S_on and Jaccard: gates that move by less than the threshold, and any re-gating away
from the trigger; positive control P0. r: everything off the trigger; control G3. Recovery: non-additivity; control the
random set. S_nat: a mean over heterogeneous contexts, biased to subcomponents that fire in all of them.

**Reading caveats, fixed now.** (i) G4 compares against amendment 2, whose gates were computed without autocast;
ours come from `compute_ci` under bf16 autocast, so a miss of a few subcomponents or a few tenths of a nat is first a
precision difference, not a bug. (ii) The null arms learn a word token the emoticon context does not prime, so they may
need a larger δW than the `o` fine-tune; a P0 pass with the nulls' `n_on` and `rel_norm_max` far above `ft_s0`'s is weaker
than one where they match, and the analysis must say which it was.

**What would change next.**
- P0 and P1 pass: the reveal reading transfers from our map to the real update, and the ladder's do-nothing r reads as
  recruitment of a target-specific set.
- P0 fails: re-gating is a generic frozen-Γ response to a perturbed trigger; the reveal evidence weakens and Γ updating
  (`backlog.md`) moves up.
- P3 passes: recruitment happens downstream of the edit site, so a one-matrix LoRA is read off the same gate shift.
- R1 passes: the recruited set is part of what the model already uses to predict `o`, the strong sense of "recruit".

**Not in this round.** Tasks other than emoticon (the ladder covers those for the W fine-tune), unfrozen Γ, the
interpretability of the recruited set beyond its `o` effect, longer training.
