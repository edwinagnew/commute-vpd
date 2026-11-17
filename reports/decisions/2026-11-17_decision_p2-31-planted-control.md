# P2.31: the crosscoder found a planted new change, but not under the filed "fine-tune-only" label. How should the last run set that bar?

**Question.** Cell P2.31 (due 11-19) adds a baseline from model diffing to the part 2 comparison: a crosscoder, in the
BatchTopK form with latent scaling (Minder et al. 2025). Run 1 of 2 (E198, one H100, about 15 min, $0.92) stopped at its
second control, as filed. The filed bar for calling a latent "fine-tune-only" failed that control. Changing a control's
bar after it has failed is your call. How should the second and last run set the bar?

**What a crosscoder is, in plain words.** It learns one dictionary of 24,576 latents over the activations of two models at
once, here the original 67M model and a fine-tune, at the input of block 2 (the residual after 2 of the 4 blocks). Each
latent has one decoder direction per model. A latent's **relative norm** says how much of its decoder sits on the
fine-tune side: 1 means fine-tune only, 0.5 means shared, 0 means original only. The literature (Lindsey et al. 2024,
Minder et al. 2025) calls a latent fine-tune-only at 0.9 or above, then drops those that are sparsity artefacts by latent
scaling. The reading asks how much of the fine-tune's activation change is carried by shared latents against
fine-tune-specific ones. Here the reading is descriptive: no side is called reused or new.

**What was measured** (`experiments/crosscoder/results/stage0.json`, log `experiments/crosscoder/results/stage01_pod.log`).
- **Original model against itself, passed.** The two halves stayed identical (largest difference 0, bar 1e-5), no
  fine-tune-only latents appeared, every alive latent sat at relative norm 0.5, and the crosscoder reconstructed 0.873 of
  the activations' variance. This control can fail only on a code fault.
- **Planted change, stopped.** The fine-tune side was the original model plus one random direction added at each task
  window's answer position, a change known to be new. The crosscoder found it: one latent carries 0.987 of the change
  (shared latents carry 0.014). But that latent's relative norm is about 0.86 (the largest distance from 0.5 among alive
  latents was 0.365). It is under the filed 0.9 bar, so it was counted as "other". The reading put 0.014 of the change on
  shared latents and 0 on fine-tune-only latents, against the filed expectation of 0.7 to 0.9 on fine-tune-specific ones.
- **Stage 1** (the first real pair) never ran. No real pair has been read.

**Did it come out as expected?** No. The apparatus finds the change. The 0.9 band does not label it new. Likely causes,
unconfirmed: this crosscoder trained for 9,086 steps on 16.7M Pile tokens plus the task text, far less than the 200M
tokens of Minder's recipe; and a latent that fires only at answer positions may keep some original-side decoder weight.
Neither has been tested.

**What the pending run would measure that nothing yet has.** How much of a real fine-tune's activation change sits on
latents the original model also has. The neuron and weight readings so far (top-240 neuron overlap, gain kept under top-K,
size-matched shares) read neurons or weights; none reads a learned dictionary of the change.

**Options** (each is the one remaining run of the cell)
a. **Set the bar from the planted control, in the same job (recommended).** The first stage trains the planted control
   with two seeds. The bar becomes the planted latent's relative norm less a margin taken from its spread over the two
   seeds. The original-against-itself control stays first, and training tokens double. If the controls hold, the job reads
   the one pair and then every pair: about $1 for the controls and one pair, then about $25 for all pairs on 4 H100 pods,
   about 2 h. It tells us whether the crosscoder sides with the share where the bar is calibrated on a change known to be
   new. A departure from the cited recipe, stated wherever the number is used.
b. **Keep 0.9 and only train longer.** Move toward Minder's 200M tokens, about 10 times the time per crosscoder. It tests
   whether undertraining is the cause. If the planted latent still sits under 0.9, the cell closes with no reading.
c. **Close the cell now.** The reading is: at this budget a crosscoder finds a planted new direction, but its
   relative-norm convention does not label it new. No real pair is read, and the part 2 comparison has no crosscoder column.

**Recommendation.** Option (a). The bar failed its own control, and (a) fixes the bar against a change we know is new
instead of a guess. It is the only option that can still deliver a reading by 11-19. Option (b) costs about 10 times as
much and can end with nothing, and (c) gives up the one reading nothing else supplies.

Evidence: `experiments/crosscoder/results/stage0.json`, `experiments/crosscoder/results/stage01_pod.log`,
`reports/plans/2026-11-17_plan_p2-31-crosscoder.md`.

Names in the reports: the cell is P2.31, the experiment E198; "fine-tune-specific" is the plan's name for a
fine-tune-only latent that survives latent scaling; the "other" group is every alive latent that is neither shared (0.4 to 0.6), fine-tune-only (0.9 or more) nor
original-only (0.1 or less).
