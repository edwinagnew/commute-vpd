# Literature: locating a fine-tune's change and telling whether it runs through an old mechanism

*mechanism-detection-lead literature agent, 2026-11-05. Asked for by the user ("if your current stuff continues to struggle to work, do a lit review"). It serves definition-lead's new-vs-reuse definition (`decisions.md` rows 115-121) and changes none of it. Nothing was run. Papers already in `lit_review/catalog.md` are cited by their entry number and not summarised again. Every new citation below was checked on arXiv, a proceedings page or the publisher's page this session.*

**Answer.** The literature has two families of tools, and our two failures each come from using the wrong family. Removal, patching and circuit tools test whether a part is *needed*. Both a part that carries the gain and a part that merely passes it on, or that both models need, are needed, so these tools cannot separate those cases on their own. Interchange tools (distributed alignment search, causal scrubbing) test whether the answer *follows the value* a part computes. That is the question "does the gain run through C". We recommend two things. First, a $0 debug job that fits C's variable on the original model and carries it unchanged into each state. Second, a re-read of saved numbers that splits each part's credit into what its own update does and what it passes on.

## Why our reads failed, in the literature's terms

- **(a) Propagation.** Patching the original model's output of a part replaces everything that reached the part, not only what the part itself added. That is a total effect, and the last MLP sits on every path, so it collects credit for the whole gain. The literature separates direct from total effects (path patching: Goldowsky-Dill et al. 2023). It also warns that later layers repair removals (McGrath et al. 2023). Our gain-path edge score has the same flaw: an edge's input difference includes everything inherited from upstream.
- **(b) A part needed by both models.** Removal with no fixed input is the "noising" direction, a test of whether a part is needed. Heimersheim & Nanda 2024 show that a part being needed does not make it the carrier of the task's content. A circuit found from input pairs inherits the problem. W₀'s copy circuit (30 edges) contains the path from the current token through mlp3 to the logits, and lookup's key also travels on that path. That is why lookup kept 0.95 of its gap under the copy circuit while keeping only 0.25-0.31 under its own 150-edge circuit (`experiments/locator_table/results/amend1.out`). Faithfulness depends on the ablation choice (Miller et al. 2024; MIB uses the same counterfactual to find a circuit and to score it). Overlap is not faithfulness (Hanna et al. 2024).

## Methods

"(a)" and "(b)" say whether the method addresses that failure. Y = by construction, P = partly, N = no.

| Method (source) | Measures | Needs | Known failure modes | (a) | (b) |
|---|---|---|---|---|---|
| CMAP, cross-model activation patching ([Prakash et al. 2024](https://arxiv.org/abs/2402.14811), catalog #8) | gain recovered when the fine-tune's part outputs are patched into the base model, on a base-model circuit | a circuit (path patching), the same inputs | total effect, so it credits pipes; it assumes the base circuit is the reference | N | P |
| Wrapper / revival ([Jain et al. 2024](https://arxiv.org/abs/2311.12786), #3) | whether pruning a few weights or brief reverse fine-tuning brings back the old behaviour | pruning, re-training | behavioural; says "wrapper", not where | P | N |
| Circuits across fine-tuning ([Wang et al. 2025](https://arxiv.org/abs/2502.11812), ICLR 2025) | EAP-IG circuits before and after: nodes stay (15-24% change), edges change (66-80%) | input pairs | faithfulness only above 0.7-0.8; change rates, no causal test of reuse | P | N |
| Circuit stability over pretraining ([Tigges et al. 2024](https://arxiv.org/abs/2407.10827), NeurIPS 2024) | the same algorithm with changing heads across checkpoints | path patching | does not study fine-tuning | N | N |
| Component reuse across tasks ([Merullo et al. 2024](https://arxiv.org/abs/2310.08744), ICLR 2024) | head overlap between two tasks' circuits (78%), plus one intervention | two circuits | overlap is not faithfulness | N | N |
| Crosscoder diffing ([Lindsey et al. 2024](https://transformer-circuits.pub/2024/crosscoders/index.html); [Minder et al. 2025](https://arxiv.org/abs/2504.02922), #14; delta-crosscoder #29) | which dictionary latents are shared or belong to one model, and how the latents scale | a dictionary trained on paired activations | L1 makes fake model-only latents (Minder); model-only latents are dense and hard to read ([Anthropic 2025 update](https://transformer-circuits.pub/2025/crosscoder-diffing-update/index.html)); says what changed, not where the gain runs | N | P |
| Activation-difference lens (#15) | what a narrow fine-tune encodes | nothing | what, not where | N | N |
| Path patching ([Goldowsky-Dill et al. 2023](https://arxiv.org/abs/2304.05969)) and IOI path patching ([Wang et al. 2022](https://arxiv.org/abs/2211.00593)) | the effect along named paths only, others frozen | input pairs, a hypothesis | expensive per path; still a necessity test | Y | N |
| Causal scrubbing ([Chan et al. 2022](https://www.lesswrong.com/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing)) | whether behaviour survives resampling every activation the hypothesis says is irrelevant | a full written hypothesis | hard to write one; too lenient when resampling preserves the statistics | Y | Y |
| DAS / interchange ([Geiger et al. 2024](https://arxiv.org/abs/2303.02536), CLeaR 2024) | interchange intervention accuracy (IIA): does the output follow a variable's value when a learned subspace is swapped in from another input | input pairs for which the variable's value is known; a few hundred steps of training | interpretability illusion: a subspace can work by waking a dormant path ([Makelov et al. 2024](https://arxiv.org/abs/2311.17030), ICLR 2024). On MIB ([Mueller et al. 2025](https://arxiv.org/abs/2504.13151), ICML 2025) it was best for locating causal variables, and SAEs were no better than neurons | Y | Y |
| EAP / EAP-IG / ACDC ([Syed et al. 2024](https://arxiv.org/abs/2310.10348); [Hanna et al. 2024](https://arxiv.org/abs/2403.17806); [Conmy et al. 2023](https://arxiv.org/abs/2304.14997)) | edge circuits from input pairs | input pairs | faithfulness depends on the ablation ([Miller et al. 2024](https://arxiv.org/abs/2407.08734)); overlap is not faithfulness; MIB scores whole faithfulness-against-size curves (CPR, CMD), not one size | P | N |
| Weight-diff localisation (grafting, [Panigrahi et al. 2023](https://arxiv.org/abs/2302.06600); task vectors, [Ilharco et al. 2023](https://arxiv.org/abs/2212.04089); sparse RL updates #5; [subgraph search, Li et al. 2025](https://arxiv.org/abs/2502.06106)) | the smallest set of updated weights whose graft onto the base model gives most of the gain | optimisation of a mask | says where the update sits, not which old mechanism it uses | Y | N |
| Four-cell cross-patch ([Zhou 2026](https://arxiv.org/abs/2605.07284)) | late stack × upstream state, base against fine-tune; the interaction says whether the late change depends on the new upstream | two models, the same inputs | 2×2 at stack grain; local next-token test | Y | N |

**Recent work on "new vs existing" (2025-2026).** Little goes beyond Prakash. Wang et al. 2025 report kept nodes with rewired edges, and our limit "a new circuit from old parts reads as reuse" is exactly that case. Catalog #39 argues that attributions on the pre-fine-tuning model mislead when the skill is new. Zhou 2026 is the only causal diagnostic between models we found that separates a late stack's own change from its dependence on a changed upstream. We found no published method that labels "reuses C / not through C".

## Recommendation

**1. Transfer C's variable from W₀ to the state (DAS fitted on W₀, frozen, read in the state). This needs one debug job, about 10 min (an estimate, not measured).**
- *Fit.* At C's site (head (2,4)'s output at the answer position, and the residual after block 2 as a second site), fit a rank-k subspace R (k ∈ {1, 2, 4}) on **W₀ only**, on C's own pairs (induction windows, B against B′ from `absent_b`), so that W₀'s answer follows the source's B.
  - Gate: W₀'s IIA on held-out pairs clears a band set from the same job's random-subspace null. If no site clears it, every row reads "unreadable: C's variable not located".
- *Read in each state, R frozen.*
  - (i) **Removal of R**: mean-replace along R. Report the share of the gain lost, with the spared control on the other task's gain and W₀'s own fall beside it. This is the definition's causal label (row 115, row 121's "located mechanism").
  - (ii) **IIA in the state on the task's pairs**: does the state's answer move to g(source value)? It is reported beside the label and is not a second rule.
- *Why it helps.* (a): the intervention sits at C's site on C's variable, so mlp3 cannot collect credit. (b): R holds only C's content, not the whole part, so a part both models need for position or format keeps its other jobs. W₀'s fall from removing R should then be small, and the own-terms and net units should agree. Where they still disagree, IIA says whether the answer follows C's value.
- *Known-answer controls (filed before the run).*
  - Copy states: removal of R takes most of the gain, the lookup gain is spared, and the state's IIA is high.
  - Lookup states: the gain is spared.
  - The hand-written copy table (E97's plant, which already ignores (2,4)): its share is near 0, so it reads "not through C".
  - Block-1 and block-3 plants: no change in W₀'s IIA, since R is fitted before the plant. That checks that the fit does not read the update.
- *Labels.*
  - "Reuses C" = the removal clears the spared control.
  - "Not through C" = C's variable still reads in the state on C's own pairs (IIA clears the band), and removal leaves the gain within the spared control's band.
  - "Unreadable" = the W₀ fit fails, C is absent in the state, or both tasks lose (row 117's "not specific").
  - Constant-answer tasks have no variable value to swap. They stay "does not apply"; input-dependent lowercase can be read once its pairs exist.

**2. Split carry into own and inherited (a re-read of saved numbers, $0, no new run if the per-part revert is saved).** For each part u: own_u = the state minus the state with u's update reverted, at the state's own input, which is the revert read already run. Inherited_u = carry_u − own_u, with carry being the E97 patch of W₀'s output.
- Known answers: block-3 plant, all own on mlp3 with zero inherited; block-1 plant, own on mlp1 and inherited only downstream of it.
- It decides only where R should be fitted (sites with own effect) and stops mlp3 being read as a mechanism. It gives no label by itself.

**What would change this.**
- If W₀'s fit fails at both sites, or copy's removal of R does not spare lookup, drop recommendation 1 and write causal-scrubbing hypotheses for the copy pair by hand.
- If removing R still drops W₀ more than the state, R is not content-specific. Report it as unreadable at this grain and do not raise k.
- If EAP-IG is kept, score it as MIB does: whole faithfulness-against-size curves with a matched random-circuit null, and the same counterfactual used to find and to score each circuit. Do not use cross-faithfulness at one size.

## Departures from the literature

- DAS fits and evaluates in one model. Here R is fitted on W₀ and evaluated in a different model, as CMAP does across models. Makelov's illusion check (does R lie in the row space the state actually reads?) is therefore needed in the state, not only in W₀.
- DAS reports interchange, not removal. Here removal of R is the label and IIA goes beside it, because of row 115.
- The own/inherited split applies Zhou 2026's 2×2 at part grain, inside one pair of models, at the part's own input. Zhou crosses whole stacks.
- The circuits in Prakash et al. come from path patching on the base model and are read in the fine-tune. Ours came from EAP-IG, and their faithfulness was read on a different task's corruption, which MIB does not do.

## Not run

- No search beyond titles, abstracts and two full texts (MIB, Wang et al. 2025).
- No check of whether head (2,4)'s output holds B linearly at rank ≤ 4.
- Timings are estimates.
- New papers were not added to `catalog.md` or a theme note.
- Not covered: neuron-grain methods, transcoders and attribution graphs, and anything on Qwen.
