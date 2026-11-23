# SAE fidelity: GPT adversarial reviewer (verbatim report, 2026-11-23)

Verdict: needs-attention. No ship as written.

**Bottom line.** HEAD `e894f1ed` trains genuine BatchTopK SAEs: a local reimplementation of the SAEBench BatchTopK training configuration. It is not an implementation of either 2023 ReLU/L1 recipe, not an import of the authors' package, and not a reproduction of Bussmann's original run. The current draft already gets the architecture label right. Bricken and Cunningham are cited at `paper/main.tex:287` as examples of SAE featurizations, not as the training methods; the setup expressly cites Bussmann at `paper/sec4_setup.tex:15`. Keep those historical citations. Cite Karvonen et al. (SAEBench App. B, Table 3) and the implementation lineage for the actual recipe, and don't say it ran the authors' code. Any wording claiming a Bricken/Cunningham reproduction would be a blocker; none was found in this draft.

**What actually ran (high confidence).** `src/commute/sae.py:51-90,124-179`, `experiments/sae_train/run.py:176-182,200-234,325-330`:
- Frozen Goodfire four-layer 67M base, input dimension 768, width 16,384 (21.3x), one setting k=80, batch-level top 80 x 2,048 entries.
- One fixed scalar inference threshold per SAE: an EMA of the smallest positive survivor (0.999, updates after step 1,000).
- Untied weights after transpose initialisation, unit decoder columns, zero encoder bias at init, learned decoder/pre-encoder bias initialised from a first-batch geometric median.
- Squared reconstruction error summed over dimensions, plus 1/32 x detached-residual AuxK (dead after 10M nonfiring tokens; at most d/2 = 384 dead latents).
- Adam 3e-4, betas .9/.999, decoder-parallel gradient removed, global norm clip 1, unit norm after each step, 1k linear warmup, last 20% linear decay.
- Site-wise RMS norm folded into the saved parameters. BF16 capture/autocast, FP32 parameters.
- 244,140 updates / 499,998,720 activation draws per site (not that many unique activations).
- E234 trained 13 sites plus a second seed at MLP1. E245 loads frozen scored seed 0 at eight sites; E251 and E263 reuse that loader (`experiments/sae_r_a/run.py:102-107`, `experiments/sae_r_a_random/run.py:28`, `experiments/sae_eb/run.py:112-119`).
- No SAE package supplies this trainer in pyproject.toml.

**Draft changes needed**
1. **Blocker, ours, high confidence: wrong sites in the methods.**
   - `paper/appE_methods_detail.tex:4` says activations "before and after each MLP". The reported read loads the attention module output and the MLP module output in each block (both branch writes, before the residual add); `src/commute/sae.py:44-45,202-206`, `src/commute/sae_feat.py:5-12,35,104-116`.
   - Fix: give the exact scored sites, and say the read takes seed 0, fixed from the base, for both models.
2. **Blocker, ours, high confidence: the displayed SAE numbers are not the answer-position, direct-needed-set diagnostic.**
   - The values quoted at `sec5_results.tex:48-50` and `main.tex:872-877` come from E263. E263 discards `at`, sums write/gain over all window positions (including position 0), and selects the saved mean-ablation K*, intersected with the changed sites (`experiments/sae_eb/run.py:60-85`, `src/commute/stand_in.py:124-142`).
   - So 0.97/0.95/0.14 do not establish the direct-K or answer-only comparison. The rebuilt remainder share of 0.829 is a share of the summed per-variable write change in the changed subset of K*.
   - Fix: label these as the old all-window K* read, or replace them.
   - The prereg separation missed (gap .115 vs .2; `experiments/sae_eb/results/summary.json:3-10`): do not describe it as passing the built suite.

**Recipe departures.** "Theirs" = from the newer published recipe we borrowed; "ours" = our adaptation.
- **Major, theirs: BatchTopK, not the 2023 recipes.** Bricken and Cunningham both use ReLU + L1. BatchTopK changes which features turn on, so the historical interpretability results do not transfer automatically.
- **Major, theirs: architecture, initialisation, dead latents.**
  - Cunningham ties the weights with no bias.
  - Bricken uses Kaiming init, a dataset-wide geometric median, and periodic dead-latent resampling.
  - Ours uses transpose init and AuxK, with no resampling.
- **Major, theirs and ours: target, data and scale.**
  - Bricken: post-ReLU MLP activations of a one-layer model, ~8.2B draws, width sweep.
  - Cunningham: Pythia-70M residual stream and MLP, 5-50M vectors, Adam 1e-3.
  - Ours: Goodfire branch-write targets, an SAEBench budget, one 16k/k80 point, no operating-point sweep.
- **Major, theirs: SAEBench, not Bussmann App. A.1, with local adaptations.**
  - Bussmann: residual streams, OpenWebText, batch 4,096, 1B tokens, k 16/32/64.
  - Ours: SAEBench Pile/500M/batch 2,048/k80, the pinned trainer's 0.999 Adam beta2, an EMA threshold.
  - Fix: cite both the architecture and the training recipe.
- **Major, ours, high confidence: the data wraps.**
  - All seven logs resume at 200,000 and wrap at 206,464, so ~37,676 of 244,140 updates (~15.4%) reuse early data (`experiments/sae_train/results/run_s0.log:458-471`, `experiments/sae_train/run.py:149-170`, `src/commute/sae.py:275-291`).
  - The resume drops the buffer/RNG state.
  - Fix: disclose the draws, the wrap and the non-exact resume.
- **Minor, ours: input distribution and numerical plumbing.** 512-token rows vs 1,024; position 0 dropped; buffer capacity; BF16 autocast; normalisation over 100 batches; AuxK centring. Fix: document the distribution changes.

**Competence. Major scope caveat, ours, high confidence.** Yes for single-site base reconstruction; there is no evidence of semantic or off-base coverage.
- The eight scored sites have FVE .875-.966, L0 79.2-81.0, splice CE increase .009-.107 nats/token, recovery .926-.996.
- The four scored attention dictionaries are 53.0-61.6% nonfiring (50.2-59.1% dead over the final 10M tokens).
- The base gate does not check the dead share (`experiments/sae_train/run.py:254-269`).
- SAEBench Pythia-160M L8 reference: FVE .868, L0 77, recovery .982, ~11.8% nonfiring.
- No interpretability or autointerp checks were run.
- Off-base: on the rebuilt task's MLP2, FVE drops from .861 to .094 (`run_eval.log:312,326`).
- E263 reads the rebuilt edit at .135 overall against .789 for the latents alone, with .829 in the remainder.
- 91 pair/key rows put ~.606-.776 of their K* write change in the remainders.
- Fix: report the reconstruction, utilisation and transfer numbers beside the write share. Do not conclude that "SAEs fail to detect reuse".

Sources: Bricken training appendix; Cunningham §2 and App. B (arXiv 2309.08600); Bussmann §3 and App. A.1 (arXiv 2412.06410); SAEBench App. B (arXiv 2503.09532). The preserved trainer `43421f5` and demo `d39d54a` were compared, in `~/commute-data/scratch/sae-lead/e234-writer`.
