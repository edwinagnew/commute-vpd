# Review: what the VPD paper says about its adversary not converging (2026-10-22)

*Review agent, read-only. Prompted by Amendment 2 of `reports/analyses/2026-10-21_analysis_cancellation-complement.md`
(from line 454), which found our 20-step adversary far from its maximum. Sources: the local copy of the paper,
`../papers/pd/vpd_blog_post.md`, which is byte-identical to the live markdown at
https://static.goodfire.ai/vpd-blog-post/post.md (diffed 2026-10-22), and the pinned code `../param-decomp` at
`74146b5`. Line numbers below are in those files. No model or GPU job was run.*

The paper reports the same thing we found: its adversary's damage keeps climbing with more steps, from under one
nat at twenty steps to about forty at three hundred and twenty, and the authors call this a limitation rather than
running the adversary to a stopping point. They read faithfulness at a fixed, short budget, give a rule of thumb
that ties the number of steps to the step size, and say they have no fully satisfying answer to how much robustness
a decomposition should have. So our verdicts are fixed-budget comparisons, which is how the paper itself reads its
own model, and our longer search broke the authors' rule of thumb by keeping the step size fixed while adding steps.

## Is the user's recollection right?

**Partly right.** What holds:

- The paper shows the adversarial KL still rising steeply with steps, on the very decomposition we use, and does
  not report a plateau.
- It says it has no satisfying answer to how much adversarial robustness to demand.
- The appendix's training recipe has tips about the adversary's steps and learning rate.

What is not there: no passage says the authors looked for a point at which to stop the adversary and failed to
find one. The "no satisfying answer" sentence is about how much robustness to demand, not about when to stop
optimising. There is no convergence criterion, no schedule for the evaluation adversary and no restarts.

## Evidence (what the paper and code say)

### The step table (main text, `vpd_blog_post.md:393-405`)

> "The table below shows KL-divergence to the target model under adversarial masking with different numbers of
> adversarial optimization steps, calculated across a batch of 128 of sequence length 512 drawn from the
> evaluation set" (footnote: "sharing the same source for each subcomponent across the batch")

| adversarial steps | 20 | 40 | 80 | 160 | 320 |
|---|---|---|---|---|---|
| KL to target model, nats | 0.828 | 1.354 | 3.838 | 25.256 | 40.220 |

> "While the decomposition is somewhat robust to approximately 20 steps of adversarial optimization, it is clearly
> not at all robust to 160 steps or more." Footnote: "zero-ablating all of the target model's weight matrices gives
> a KL-divergence of ca. 67.2449."

> "However, we note that *complete* adversarial robustness would not necessarily be desirable." (line 405)

The step size used for this table is **not stated** anywhere in the paper. The code has no sweep script for it.

### The limitation (Discussion, "Other limitations and future work", line 1529)

> "**Our decomposition is not as adversarially robust as we would like.** [...] while the decomposition is at
> least somewhat robust to ≈20 steps of adversarial optimization (KL divergence 0.83), robustness degrades rapidly
> with more optimization steps, reaching a KL divergence of 40.2 at 320 steps. This means that there exist sets of
> subcomponent ablations involving only causally unimportant subcomponents that drastically alter the model's
> output. [...] we do not necessarily expect or even desire complete robustness to arbitrarily many steps of
> adversarial optimization. However, we suspect that substantially higher levels of adversarial robustness are
> still achievable. [...] Our current adversarial sampler uses a relatively primitive form of projected gradient
> descent (PGD) to find worst-case ablation masks. We think it should be possible to improve the performance of
> this sampler."

### How much robustness (Discussion, `sec:vpd_recon_motivation`, lines 1505-1509)

> "How much robustness do we need to demand for our explanation to be mechanistically faithful? How much robustness
> is actually too much [...]? We do not currently have a fully satisfying answer to this question, but we suggest
> that a reasonable approach may be to ground the answer in practical considerations: What combinations of
> (sub)component ablations might we realistically want to perform when using VPD to understand or edit a given
> model?"

The methods appendix (line 2019) argues that an unconstrained worst case is "too strict", because it can exploit
interference noise in inactive circuits. Sharing one source across the batch is their restriction against that
(line 2027).

### Training vs evaluation adversary (Appendix, "Adversarial reconstruction loss", lines 2031-2039)

> Training: "we optimize a single set of sources r^{adv}_{b,t,c} that persists across batches [...] On every batch,
> the adversarial Adam optimizer performs n_adv update steps [...] (In this paper, we used n_adv=3)."

> Evaluation: "Continuously updating a single set of persistent adversarial sources is more computationally
> efficient, but not principled. Hypothetically, the VPD optimizer might trap the adversarial optimizer in some
> local extremum [...]. Thus, for evaluation, we use a new set of adversarial sources r^{adv}_c for every
> evaluation batch, but use more adversarial optimization steps per batch n_adv than we do in training."

### The training recipe (Appendix "A training recipe for VPD", line 2103 and 2112)

> "1. **PGD reconstruction loss** (adversarial masks, freshly initialized at each step): The most important metric.
> [...] The setting we want is `shared_across_batch` [...]. This is stricter than the persistent adversarial loss
> used during training and is our primary indicator of mechanistic faithfulness. For deeper models, more
> adversarial steps may be needed. **As a rough heuristic, we keep n_adv · lr_adv ≈ 2; if increasing the number of
> steps, decrease the learning rate proportionally so the adversarial optimizer can tune more precisely.** For
> discussion on how much adversarial optimization exactly our causal importances should be robust to, see
> sec:discussion."

> Training adversary: "Making the adversarial optimizer cheap yet effective is nontrivial. The adversarial learning
> rate usually needs to be tuned and depends on the regular learning rate. [...] Adam [...] β1 = 0.5, β2 = 0.99,
> constant learning rate with short warmup, per-batch-per-position source scope, and updating the sources n_adv=3
> times for each outer step [...]. For smaller models, fewer adversarial steps per training step may suffice; for
> larger, especially deeper, models may need more steps (and a correspondingly lower adversarial learning rate)."

The hyperparameter appendix (lines 2168-2172) gives the training adversary: Adam, lr 0.01, constant with 2.5%
warmup, per-element-per-position sources, two warm-up steps per training step. The final evaluation table
(line 2236) reports `PGDReconLoss (KL) 0.9268` with no step count beside it; the config says 20.

### The code (`param-decomp` at `74146b5`)

- Evaluation adversary, `param_decomp/experiments/lm/pile_llama_simple_mlp-4L.yaml:116-121`: `init: random`,
  `step_size: 0.1`, `n_steps: 20`, `mask_scope: shared_across_batch`, `PGDReconLoss`; `eval_batch_size: 128`,
  `n_eval_steps: 1` (lines 89, 91). The 12-layer config uses the same 20 × 0.1
  (`pile_llama_simple_mlp-12L.yaml:132-137`), despite the recipe's "deeper models may need more steps".
- The loop, `param_decomp/metrics/pgd_utils.py:42-66`: fixed `n_steps` of `step_size · sign(grad)`, clamp to
  [0, 1], loss read after the last step. No schedule, no early stop, no restarts, no best-so-far tracking. Init
  `torch.rand` (`pgd_utils.py:300-312`). The Δ source is one extra entry per matrix (`pgd_utils.py:29, 84`).
- Training adversary, same yaml lines 51-66: `PersistentPGDReconLoss`, Adam lr 0.01, `per_batch_per_position`,
  `n_warmup_steps: 2`. Optimiser classes `param_decomp/persistent_pgd.py:56-116`.

## Theirs vs ours

| | VPD evaluation | VPD training | ours (`faithfulness.pile_faithfulness`, `gamma.pgd_sources`) |
|---|---|---|---|
| optimiser | sign steps | Adam, lr 0.01, β 0.5/0.99 | sign steps (authors' `calc_multibatch_pgd_masked_recon_loss`) |
| steps × step size | 20 × 0.1 (product 2) | 3 per training step, persistent | 20 × 0.1 (product 2); Amendment 2 ran 50 and 100 × 0.1 |
| init | uniform random | carried over | uniform random, seeded |
| scope | one source per subcomponent, shared over a 128-row batch, plus one per matrix for Δ | one per row and position | the same, over a 128-row block (gradients summed over micro-batches) |
| persistent state | none, fresh per batch | across all training steps | none (E15's refit uses the persistent training form) |
| stopping rule | fixed budget | not applicable | fixed budget |
| reported | one 20-step number; a step table on one model | training loss | 20-step numbers, mean over two blocks; ratios to W₀ |

The one material difference found is in Amendment 2, not in the harness. The paper's rule keeps steps times step
size near 2. Amendment 2's 50- and 100-step runs kept the step size at 0.1, so the product was 5 and 10. Under the
paper's recipe, a 100-step search would use a step size of about 0.02. Whether the paper's own step table kept the
step size fixed is not stated.

## What this means for our verdicts (inference, not the paper's words)

- **Fixed-budget comparison is the paper's practice.** Its primary faithfulness metric is one number at 20 steps
  of 0.1. It treats the rise with steps as a known limitation and does not converge the adversary. Reading an
  updated model against the original model at the same budget is therefore the paper's own reading, applied to two
  models instead of one.
- **Option 1 (keep the 20-step probe and say so)** is what the paper does. The paper's own wording supports the
  rewording: "somewhat robust to approximately 20 steps".
- **Option 2 (ratio stability across budgets)** is the natural extension. The paper's step table is the one-model
  version of it. Run it the paper's way: scale the step size down as steps go up, keeping the product near 2,
  rather than adding steps at 0.1. The fixed-0.1 runs also probe a larger total movement per source, which the
  recipe does not intend.
- **Option 3 (a converged adversary)** is not what the paper does, and it argues against the unconstrained worst
  case as "too strict". At 320 steps the paper's own decomposition is at 40 of the 67 nats of zero-ablating every
  matrix. A converged adversary would likely put every model, the original included, in that range.
- Our original model reads 0.66 at 20 steps where the paper reads 0.83 and 0.93, on different rows. Our 100-step
  value of 14 sits between the paper's 80- and 160-step values. The two curves rise alike, which suggests the climb we
  saw belongs to the released decomposition and its adversary, not to our harness.

## Not checked

- The step size behind the paper's 20 to 320 step table; no config or script for it was found. Asking the authors
  would settle it.
- The 20-step value and its spread on the paper's own evaluation batches; we did not rerun their eval.
- Whether a decaying step size or restarts change the shape of the curve; the paper reports none.
- The WandB run logs linked in the paper (`goodfire/spd/runs/s-55ea3f9b`), which may hold eval curves over training.
- The SPD paper (arXiv 2506.20790) and any later Goodfire posts, for adversary discussion.
- The live web page's rendered footnotes beyond the markdown, assumed identical since the markdown matched.
