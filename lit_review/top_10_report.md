# Literature review: commuting post-training updates past a parameter decomposition

*2026-10-15. Two search stages, about 1,300 unique hits, 91 papers and posts read and summarised by agents. Each
claim below comes from an agent's summary of the paper; I have not checked the numbers against the PDFs, so check
before quoting. The relevance scores are the agents' own. This report proposes no experiment: any diagnostic mentioned
here goes through `/design` first, as `.claude/CLAUDE.md` requires.*

## Executive summary

**No paper carries a weight-space decomposition across a training update.** The parameter-decomposition line (APD →
SPD → VPD, plus ASPD and sparse weight factorisation) decomposes one frozen model at a time. None of these papers
reports stability across seeds, across checkpoints, or under δW. That gap is this project's question, and the search
found no prior attempt at it.

The surrounding literature splits into three camps, and they disagree in ways that matter here.

1. **"Post-training reuses what is there"**, in activation space. The evidence:
   - Jain et al. find small, localised "wrappers" over existing capabilities.
   - Prakash et al. find that entity-tracking circuits are kept.
   - Persona and EM features are already present in the base model.
   - Base-model SAEs transfer to chat models.
   - Crosscoders show on-policy distillation re-weighting shared features.

   This camp predicts that a δW should be expressible on existing components, i.e. that a simple Φ exists.
2. **The structure of δW in weight space is mixed evidence.**
   - RL updates touch only 5–30% of weights, but are *nearly full-rank* within each matrix (Mukherjee).
   - RLVR updates avoid the principal singular directions of W (Zhu et al.).
   - LoRA creates "intruder dimensions" orthogonal to every pretrained singular vector, which full fine-tuning does
     not (Shuttleworth et al.).
   - At the same time, the top singular vectors of δW are readable behavioural directions (Zhong & Raghunathan).

   Off-principal, intruder and full-rank updates are exactly what a projection onto existing rank-1 components would
   miss. This is a concrete weight-space prediction of failure, and it does not require "post-training creates new
   capabilities".
3. **"The instrument is fragile."**
   - Crosscoder "new" latents are mostly sparsity artifacts (Minder, Dumas et al.).
   - SAE features differ across seeds, yet the subspaces they span reproduce across seeds.
   - The VPD audit finds that VPD's own explanations do not compose under aggregation.

   This is the "parameter decompositions are too brittle" branch of GOALS.md, and it now has direct evidence for VPD.

The most useful outcome for the project is that camps 1 and 2 can be told apart in weight space. "Reuses components"
and "off-principal / intruder" make opposite predictions about where δW lands relative to the VPD subcomponents.
Camp 3 says the null needs a cross-seed baseline: the component analogue of "unstable features, reproducible
subspaces".

## Top 10

Eight have PDFs in `pdfs/` (the user's cap: 5ish, at most 10). Items 9 and 10 are forum or linkpost material with no
paper PDF worth storing.

1. **Jain, Kirk, Lubana et al. 2024, *Mechanistically analyzing the effects of fine-tuning on procedurally defined
   tasks*** (ICLR; [arXiv 2311.12786](https://arxiv.org/abs/2311.12786)). Stage 2. Score 8.
   - **Finding:** fine-tuning learns a minimal "wrapper" on top of existing capabilities. A few steps of further
     training can "revive" a capability the wrapper hid.
   - **Relevance:** this is the cleanest statement of the "strengthen/select" hypothesis. If it holds, a δW that
     re-weights a few components is the expected case. Its evidence is from toy models (Tracr, PCFG minGPT,
     TinyStories), which matches the scale of the project's own model.
2. **Mukherjee et al. 2025, *RL finetunes small subnetworks in LLMs*** (NeurIPS;
   [arXiv 2505.11711](https://arxiv.org/abs/2505.11711)). Stage 2. Score 7.
   - **Finding:** 68–96% of parameters are bit-identical after RL. The subnetworks overlap across seeds and
     algorithms. Within each matrix the updates are 99%+ full rank.
   - **Relevance:** "sparse in parameters, full rank in each matrix" is a direct constraint on Φ. Projection onto a
     few rank-1 components is the wrong prior for RL updates. Caveat: the sparsity depends on the bf16 tolerance.
3. **Shuttleworth, Andreas, Torralba, Sharma 2024, *LoRA vs full fine-tuning: an illusion of equivalence***
   ([arXiv 2410.21228](https://arxiv.org/abs/2410.21228)). Stage 2. Score 6.
   - **Finding:** LoRA introduces high-ranking *intruder dimensions* that are orthogonal to all pretrained singular
     vectors. Rescaling them reduces forgetting.
   - **Relevance:** the project's canonical update is a LoRA. Intruder dimensions are a weight-space signature of
     "new direction". A candidate diagnostic (for `/design`): do the round-2 LoRAs carry intruder dimensions?
4. **Zhu et al. 2025, *The path not taken: RLVR provably learns off the principals***
   ([arXiv 2511.08567](https://arxiv.org/abs/2511.08567)). Stage 2. Score 5.
   - **Finding:** RLVR updates avoid the top singular subspaces of W, which the authors explain with a "three-gate"
     account (KL constraint, geometry, bf16).
   - **Relevance:** it pairs with item 3. It predicts that RL-style δW lies off the principal directions, where the
     large VPD components probably sit. This is a measurable alternative to "creates new capability".
5. **Zhong & Raghunathan 2026, *Watch the weights*** (ICLR). Stage 1. Score 6.
   - **Finding:** the top-k left singular vectors of δW (o_proj, down_proj) work as behavioural monitors, reaching
     93–100% backdoor detection.
   - **Relevance:** this is the simplest data-free reading of δW and the obvious baseline for any Φ. A candidate
     diagnostic (for `/design`): do δW's top singular vectors align with existing VPD components?
6. **Prakash et al. 2024, *Fine-tuning enhances existing mechanisms: entity tracking***
   ([arXiv 2402.14811](https://arxiv.org/abs/2402.14811)). Stage 1. Score 7.
   - **Finding:** the same circuit implements entity tracking in Llama-7B and its fine-tunes (faithfulness
     0.88–0.97), and the fine-tunes do it better.
   - **Caveat:** the fine-tuned circuits are 175-head supersets, so "same circuit" is looser than the headline.
7. **Ge et al. 2026, *Evolution of concepts in language model pre-training*** (ICLR). Stage 1. Score 7.
   - **Finding:** one crosscoder across 32 Pythia checkpoints shows most features forming around step 1k.
   - **Relevance:** this is the closest existing "interpretability along training". Its own stated limit, that a
     joint dictionary needs discrete snapshots, is what a working Φ would remove.
8. **Soligo, Turner, Rajamanoharan, Nanda 2025, *Convergent linear representations of EM***
   ([arXiv 2506.11618](https://arxiv.org/abs/2506.11618)). Stage 1. Score 6.
   - **Finding:** a single rank-1 LoRA induces EM, and one misalignment direction transfers across fine-tunes.
   - **Relevance:** rank-1 adapters have the same shape as VPD subcomponents. This is the natural test case for "is
     the update one existing component or a new one". Caveat: the B-vector and the mean-difference direction have
     cosine 0.04, yet ablating the mean-difference direction removes EM.
9. **Angsten 2026, *Do VPD's explanations aggregate? An audit of the released decomposition*** (LessWrong). Stage 2.
   Score 8, no PDF.
   - **Finding:** in the decomposition this project uses, deleting all components never labelled as needed costs
     1.28 nats of KL when the used components are left fully on (the harm goes away when they sit at their labels).
     Aggregating 64 tokens' worth of donor explanations reaches 0.80 nats, about what the paper's 20-step adversary
     reaches (0.83).
   - **Relevance:** direct evidence for the "brittle" branch. It also gives donor-aggregation and
     frequency-matched random controls the project can reuse.
10. **The decomposition line: VPD linkpost (Bushnaq et al. 2026), APD (Braun, Bushnaq et al. 2025), SPD.** Stage 1.
    Scores 9, 8/7 and 6. Background; the project already holds `../papers/pd/vpd_blog_post.md`.
    - In the VPD comments, Roger and Bushnaq discuss whether token-wise sparsity limits how far frequently-active
      circuits can be separated.
    - In the SPD comments, a commenter argues that the faithfulness loss lets an overcomplete decomposition collapse
      onto the full linear map. That bears directly on "Δ may not absorb δW".

## How the refined search changed coverage
- Stage 1 found the decomposition line and the activation-space diffing camp.
- Stage 2, with tighter terms ("intruder dimensions", "RL sparse subnetwork", "superficial alignment", "SAE seed
  consistency", "weight differences"), added the weight-space structure of δW (items 2–4). That turned out to be the
  most decision-relevant evidence.
- Stage 2 also added the seed-stability literature (Paulo & Belrose; *Unstable features, reproducible subspaces*) and
  the VPD audit.
- Google Scholar was rate-limited in stage 2 and returned nothing. arXiv and Semantic Scholar carried that stage.
- Of 97 papers selected across both stages, 91 have summaries. The others lacked PDFs or failed to download (e.g.
  arXiv 2507.17107 returned 404) or were judged duplicates (two LW crossposts and the AXRP transcript).

## Remaining gaps
- **Stability of parameter decompositions across seeds or checkpoints.** Nothing was found. It looks absent from
  the literature, not missed. The project's warm-start and cross-seed controls would be the first data.
- **Perturbation theory for sparse, overcomplete rank-1 decompositions.** Only the general literature on matrix
  factorisation and implicit bias turned up (Li et al. 2022 on reparametrised GD and mirror descent; Li, Luo & Lyu
  2021 on greedy low-rank learning; "The loss does not see the basis, but Adam does"). Two warnings from it:
  - Li et al. use "commuting" for a different notion (Lie brackets of the parametrisation).
  - Whether a pulled-back step is basis-independent depends on the optimiser (Adam is not equivariant).
- **Not resolved in this sweep:** "Null-calibrated evaluation of SAE decoder reproducibility" (ICML 2026 page only)
  and the "transplanting a misalignment persona" paper. "Why RL updates look sparse" (NeurIPS 2025) appears to be the
  same work as item 4, arXiv 2511.08567: both give the same three-gate account.
- **Already in the repo, not re-retrieved here:** `theory/reveal_vs_create_literature.md` covers more of the
  persona and alignment side: PSM, Nadaf 2026, Tice et al., Moskvoretskii et al., the Assistant Axis, and Ward et
  al. The most directly relevant is Nadaf 2026 (arXiv 2607.21356). There, projecting the persona subspace out of
  *activations* stops EM, but projecting it out of the *weight gradient* does nothing. Read the two documents
  together.

## Files
- `catalog.md` lists all 91 papers, ranked.
- `reading_list.md` lists the 65 papers scoring 4 or more, with URL and summary link.
- `summaries/` has one file per paper.
- `pdfs/` has the 8 stored PDFs.
- `stage1_report.md`, `search_terms*.json` and `research_proposal.md` record how the search was run.
