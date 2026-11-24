# Appendix to the P2.75 diff-SAE spec: derivations, parity checks and review responses

Main file: `reports/plans/2026-11-23_plan_p2-75-diff-sae-spec.md`. Symbols are as defined there. Nothing has run.

## A. Expected values for a planted direction

First derived by redteam-lead; restated here with the normaliser's sample kept apart from the scaler's rows, after the
GPT review's second point.

Take a plant that adds a vector p, with unit direction p̂, on some tokens and nothing elsewhere, so D = p there and 0
elsewhere. Let ρ_n be the plant's share of the normaliser's sample, so μ_D = ρ_n p. Take a latent whose decoder is p̂
and whose f is the positive part of N(D) along p̂.
- On plant tokens N(D) = s(1 − ρ_n)p, so f = s‖p‖(1 − ρ_n). Elsewhere N(D) = −sρ_n p, whose positive part along p̂ is
  0, so f = 0.
- The means cancel in N(h_F) − N(h_M) = sD, so β_F − β_M = 1 / (1 − ρ_n).
- β_M = (c − ρ_n) / (1 − ρ_n), where c is the mean over the scaler rows' plant tokens of ⟨h_M, p̂⟩, divided by ‖p‖.
- So ν_D = (c − ρ_n) / (c − ρ_n + 1). With t = c − ρ_n, ν_D = t / (t + 1).

**The pass interval.** If t + 1 > 0, t / (t + 1) ≤ 0.3 is 0.7 t ≤ 0.3, so t ≤ 3/7. If t + 1 < 0, it needs t ≥ 3/7,
which contradicts t < −1. At t = −1, ν is infinite and set to nan. So ν_D ≤ 0.3 holds exactly for −1 < t ≤ 3/7. The
oracle tests this whole interval on the actual scaler rows and fixed unit readouts, rather than relying on "c of a few
tenths".

**Three readings follow, for ν_D.**
- At c = 0, β_M = −ρ_n / (1 − ρ_n) < 0. The plant passes as published and is dropped by the E.2 exclusion, which is
  why E.2 is not scored.
- E212's plant is a random unit vector times the median base norm at the label read position (`realistic_tasks/run3.py`
  230-235). A random direction's component of M's mean is about ‖h‖/√768, about 0.036‖h‖. Against ‖p‖ = median ‖h‖
  that gives c of about ±0.04, t inside the interval, and ν_D near 0: a pass. So E212's plant is a positive control
  that the scale argument expects to pass, which is why it cannot gate a run at fine-tune size.
- Against ‖p‖ of about 0.01‖h‖, the previous draft's fine-tune scale, c is about 3.6 and ν_D about 0.78: a fail. Rung
  1 is at that scale, so the gate's lean is a miss.

**What sets K.** K's mean share is Σ_x ⟨μ_D, D(x)⟩ / Σ_x ‖D(x)‖² on the score read windows. For E212's plant it is
ρ_n, whatever the score rows' own plant share is. For a dense change it can take either sign; min(K, 1) keeps a side
whose mean points against its change at the crosscoder column's bar of 0.5.

**ν_lit** has no closed form: its f depends on M's activation along p̂ on every token, through D's normaliser.

**The oracle's directions.** For the plant, p itself. For rung 1, the Gaussians and the head at site B, the top ten
principal directions of D on the read set. For ν_D, f is the positive part of N(D)'s component along each direction;
for ν_lit, of N(h_F)'s. The plant's filed expected values are the oracle's ν, its K, and the oracle directions' carried
parts.

## B. Parity checks against the pinned code

Each runs in the rehearsal on identical paired activations, with bars from a measured floor.

- **trainSAE.** Called as it runs, so no parity check is needed for the loop. The rehearsal checks that the calls the
  toolkit makes ran: θ's value after a logging step differs from the value `update` alone would leave, and the dead
  counters after a validation have grown by the validation's token count for latents that did not fire.
- **closed_form_scalars and its loaders.** Run from the pinned source. The rehearsal checks the loaded file's sha256
  against the clone at 617d926 and that the function's numerator B, denominators C and D and β agree with a float64
  recomputation of ours to the float32 floor. If they must be ported instead, the port is compared with the pinned
  function on the same batches and a fixed SAE: B, C and D before division, latent by latent, and the masks of each form.
- **Encoder inputs.** The two hazards that the numerator identity alone cannot see: a wrong encoder input and a double
  normalisation. For ν_lit, f must equal `encode(h_F, use_threshold=True)` with the SAE's own normalisation inside
  `encode`. For ν_D, f must equal the trainer's path on the same batch: the normalised input encoded with
  `normalize_activations=False`. Targets must be normalised once, by `load_base_activation` and `load_ft_activation`
  with `normalize=True`.
- **Normaliser.** Our port's mean and std on a fixed sample against a float64 two-pass on the same tokens, to the
  float32 floor. The sampling rule is read against `training.py` 165-190: an equal count per dataset, uniform over the
  whole dataset.
- **Sampling of the scaler rows.** The adapter's per-dataset prefix lengths against `calculate_samples_per_dataset`'s
  rule (`activations.py` 212-245) for the two datasets' sizes, by hand: proportional, rounded down.
- **wandb.** `wandb.init(mode="disabled")` before `trainSAE`, checked in the rehearsal: `log_stats` calls `wandb.log`
  whether or not `use_wandb` is set.

## C. Review response

All points were checked against the cited files. Each is answered here; none was rejected.

**GPT [P1], sensitivity at the size run 1 green-lights.** Real. `run3.py` 230-235 plants at the median base norm, and
E243's rung 1 is 1.04 times elicit's amplitude against rung 5's 104. Answer: E212's replay stays first and alone as a
descriptive control. The gate is rung 1 at two seeds against two Gaussians matched to rung 1 on rung 1's rows, the same
site and rows. Rung 5 and E212 are descriptive. The oracle diagnoses only. The outcome table is frozen per form,
including a missing arm and seed disagreement, and each miss is scoped. Run 1 is repriced at eight fits, about $33.

**GPT [P1], the fitted bias in the pass rule.** Real. The old rule divided by 1 minus a signed, fitted bias part,
undefined at 1 and rewarding overshoot. Answer: a fixed reference K from μ_D alone, computed before the fit, a bar of
0.5 · min(K, 1), "not read" under K = 0.5. Reconstruction and selection are filed apart for the exact plant. The ν_D
interval −1 < t ≤ 3/7 is stated and tested in the oracle on the scaler rows (A).

**GPT [P1], the KL error arm and the plant's reference.** Real. The old error arm, h_F − D̂ + (b/s + μ_D), kept the
whole bias; footnote 8 (post:110) gives h_M + (D − D̂) = h_F − D̂. `site_acts` adds the plant and stops (`crosscoder.py`
426-445). Answer: the error arm is h_F − D̂; every patch uses encode(D), with masks named per form; the plant's
reference forward carries the plant through every downstream layer; the nonzero-bias exact-reconstruction fixture is
added to the $0 checks. No bias-keeping arm is run, so none needs an "ours" label.

**GPT [P1], the literature path.** Real on every count: the normaliser samples uniformly over each whole cache
(`training.py` 165-190); the scaler takes proportional prefixes, no shuffle, float32 (`closed_form.py` 304-313, 809-830);
`trainSAE` logs every 50 steps and validates every 20,000, and `trainer.loss` moves θ and the dead counters
(dl:`batch_top_k.py` 162-196). Partly as proposed: the toolkit's package cannot be imported (`diffing/__init__.py`
imports every method), so "the pinned scaler behind an adapter" becomes `closed_form_scalars` run from the pinned file,
with a port plus parity as the fallback. `trainSAE` is called as it runs. The rehearsal goes to 12M tokens and sets a
known dead set. The construction check compares numerators.

**GPT [P2], the FVE stop.** Real: dl's FVE is centred (dl:`training.py` 45-51), `crosscoder.carried`'s is not
(`crosscoder.py` 253-278). Answer: one formula, dl's centred one, pooled over the 5M validation tokens with the
thresholded encode, a zero-variance branch, and a bar from the rehearsal on the same path. Pooling is ours: on a plant
side most batches of 2,048 tokens hold no plant token, and dl's per-batch form reads those as nan.

**GPT [P2], the matched nulls in run 2.** Real: E235's bisection (`crosscoder_proper/run.py` 243-266) matches one side
on its own rows. Answer: 48 fits are the default; a shared null needs both sides within 0.05 on both sides' rows, or it
is called a pair-level reference.

**redteam-lead (a).** Written into stage 0b: the gate's null is n = 2, a weak null that opens a descriptive run 2 and
supports no claim at the bar. Site B's single Gaussian is descriptive only. The Gaussians are now matched to rung 1, not
to T2, since rung 1 gates.

**redteam-lead (b1).** Written into the gate: the form that opened run 2 is recorded, and if only ν_D passes the paper
column is labelled ours.

**redteam-lead (b2).** Written into the gate and §3.5: the gate uses ν ≤ 0.3 as published, and any whip-lead ruling to
score E.2 must come before run 1's results are read.

## D. Earlier review questions, settled by redteam-lead (2026-11-23)

1. **Which input latent scaling encodes:** both forms are scored, each labelled. Neither is called faithful.
2. **The E.2 exclusion:** reported beside, not scored. Open for whip-lead, now with a deadline: before run 1 is read.
3. **k at d = 768:** the k-33 fit runs only if the gate's plant fails; it moved from E212's plant to rung 1.
4. **Order:** E212's plant first and alone, then the gate, then the descriptive controls.
5. **The null:** two Gaussian draws at site A, now matched to rung 1 on its rows.
6. **Scale:** run 2 no longer waits on the oracle; the oracle diagnoses, and the empirical gate decides.

## E. Choices made here that neither reviewer fixed

- K's minimum of 0.5, so the bar never falls under 0.25, half the crosscoder column's.
- The gate needs both rung-1 seeds to pass; one of two is "unstable", not a pass.
- The Gaussians for the gate use elicit's per-matrix norm profile, E243's `random_of`, since rung 1 is a grafted head
  with no matrix norms of M's own to copy.
- The validation text is 5M Pile tokens from rows 131,072 on, the same for every side.
- The early stop's FVE is pooled over the validation set, not averaged per batch.
- The FVE stop's bar is the rehearsal's last value on the same side minus three times its two-seed difference.
- E212's own Gaussians are not rerun.
