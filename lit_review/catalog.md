# Literature Review Catalog

Total papers: 91

---

## 1. [Linkpost] Interpreting Language Model Parameters

**Relevance Score:** 9/10
**Source:** lesswrong
**Year:** 2026
**Authors:** Lucius Bushnaq
**URL:** https://www.lesswrong.com/posts/eAQZaiC3PcBhS4HjM/linkpost-interpreting-language-model-parameters

The post introduces adVersarial Parameter Decomposition (VPD), the third method in the authors' parameter-decomposition line after APD and SPD. VPD splits the weights of a small language model (about 67M parameters, 28M non-embedding ones decomposed) into rank-1 subcomponents, optimised so that few 

*Full summary: [lw_eAQZaiC3PcBhS4HjM.md](summaries/lw_eAQZaiC3PcBhS4HjM.md)*

---

## 2. Attribution-based parameter decomposition

**Relevance Score:** 8/10
**Source:** alignment_forum
**Year:** 2025
**Authors:** Lucius Bushnaq
**URL:** https://www.lesswrong.com/posts/EPefYWjuHNcNH4C7E/attribution-based-parameter-decomposition

The post introduces Attribution-based Parameter Decomposition (APD), a method that splits a network's weight vector directly into a sum of parameter components, each meant to be a mechanism, rather than decomposing activations into features first. The components are chosen so that the average descri

*Full summary: [lw_EPefYWjuHNcNH4C7E.md](summaries/lw_EPefYWjuHNcNH4C7E.md)*

---

## 3. Mechanistically Analyzing the Effects of Finetuning on Procedurally Defined Tasks

**Relevance Score:** 8/10
**Source:** arxiv
**Year:** 2023
**Authors:** Samyak Jain, Robert Kirk, Ekdeep Singh Lubana et al.
**URL:** http://arxiv.org/abs/2311.12786v2

The paper asks whether fine-tuning gives a pretrained model new capabilities or only modulates existing ones, and answers it in controlled synthetic settings (Tracr-compiled transformers and minGPTs trained on PCFG-defined counting tasks) where mechanistic tools (pruning, probing, attention maps) ca

*Full summary: [arxiv_2311.12786v2.md](summaries/arxiv_2311.12786v2.md)*

---

## 4. Do VPD's Explanations Aggregate? An Audit of the Released Decomposition

**Relevance Score:** 8/10
**Source:** lesswrong
**Year:** 2026
**Authors:** Tom Angsten
**URL:** https://www.lesswrong.com/posts/KeBccWBGXnNXzZFBp/do-vpd-s-explanations-aggregate-an-audit-of-the-released-1

The post audits the publicly released VPD (adVersarial Parameter Decomposition, Goodfire) decomposition of a 4-layer, 67M-parameter language model, asking whether per-token "needed here" component labels can be combined (aggregated across inputs) or edited without changing model outputs. Its central

*Full summary: [lw_KeBccWBGXnNXzZFBp.md](summaries/lw_KeBccWBGXnNXzZFBp.md)*

---

## 5. Reinforcement Learning Finetunes Small Subnetworks in Large Language Models

**Relevance Score:** 7/10
**Source:** manual
**Year:** 2025
**URL:** https://arxiv.org/abs/2505.11711

RL post-training of LLMs (PPO, GRPO, DPO, KTO, ORPO, SimPO, PRIME, rejection-sampling SFT) changes only a small fraction of weights: across 10 public checkpoints, 68.5% to 96.0% of parameters are bit-identical (at bf16 tolerance 1e-5) before and after RL, with the touched subnetwork around 5-30% of 

*Full summary: [arxiv_2505.11711.md](summaries/arxiv_2505.11711.md)*

---

## 6. Interpretability in Parameter Space: Minimizing Mechanistic Description Length with Attribution-based Parameter Decomposition

**Relevance Score:** 7/10
**Source:** arxiv
**Year:** 2025
**Authors:** Dan Braun, Lucius Bushnaq, Stefan Heimersheim et al.
**URL:** http://arxiv.org/abs/2501.14926v4

The paper introduces Attribution-based Parameter Decomposition (APD), which splits a network's flattened weight vector into a sum of parameter components. The components are trained to be faithful (they sum to the original weights), minimal (only a few are active per input, chosen by gradient attrib

*Full summary: [arxiv_2501.14926v4.md](summaries/arxiv_2501.14926v4.md)*

---

## 7. Evolution of Concepts in Language Model Pre-training

**Relevance Score:** 7/10
**Source:** google_scholar
**URL:** https://proceedings.iclr.cc/paper_files/paper/2026/hash/45673dbf3f331fbd911b0689872de396-Abstract-Conference.html

The paper tracks interpretable features across pre-training checkpoints of Pythia models by training a single cross-snapshot crosscoder (a sparse dictionary learned jointly over several checkpoints' activations at one layer). Most features form around a specific early point in training (around step 

*Full summary: [Evolution_of_concepts_in_language_model_pre-traini.md](summaries/Evolution_of_concepts_in_language_model_pre-traini.md)*

---

## 8. Fine-Tuning Enhances Existing Mechanisms: A Case Study on Entity Tracking

**Relevance Score:** 7/10
**Source:** arxiv
**Year:** 2024
**Authors:** Nikhil Prakash, Tamar Rott Shaham, Tal Haklay et al.
**URL:** http://arxiv.org/abs/2402.14811v1

The paper asks whether fine-tuning changes the internal mechanism a language model uses, or only strengthens the mechanism it already has. Using in-context entity tracking (synthetic "which box contains which object" tasks) on Llama-7B and three fine-tuned descendants (Vicuna, Goat-LoRA, FLoat), the

*Full summary: [arxiv_2402.14811v1.md](summaries/arxiv_2402.14811v1.md)*

---

## 9. Persona Features Control Emergent Misalignment

**Relevance Score:** 7/10
**Source:** google_scholar
**URL:** https://proceedings.iclr.cc/paper_files/paper/2026/hash/50db99ee3bccf73bfe1cf2af1e960414-Abstract-Conference.html

Fine-tuning GPT-4o on narrow bad advice or insecure code makes it broadly misaligned ("emergent misalignment"), and the authors show this across RL on o3-mini, synthetic advice datasets, and helpful-only models. Using a sparse autoencoder (SAE) trained on pre-training data, they compare fixed-SAE fe

*Full summary: [Personafeaturescontrol_emergent_misalignment.md](summaries/Personafeaturescontrol_emergent_misalignment.md)*

---

## 10. Convergent Linear Representations of Emergent Misalignment

**Relevance Score:** 6/10
**Source:** arxiv
**Year:** 2025
**Authors:** Anna Soligo, Edward Turner, Senthooran Rajamanoharan et al.
**URL:** http://arxiv.org/abs/2506.11618v2

The authors build a minimal "model organism" of emergent misalignment (EM): Qwen2.5-14B-Instruct fine-tuned with only 9 rank-1 LoRA adapters on the MLP down-projections of three layer groups, trained on a narrow "bad medical advice" dataset. They find that a single residual-stream "misalignment dire

*Full summary: [arxiv_2506.11618v2.md](summaries/arxiv_2506.11618v2.md)*

---

## 11. LoRA vs Full Fine-tuning: An Illusion of Equivalence

**Relevance Score:** 6/10
**Source:** arxiv
**Year:** 2024
**Authors:** Reece Shuttleworth, Jacob Andreas, Antonio Torralba et al.
**URL:** http://arxiv.org/abs/2410.21228v3

The paper argues that LoRA and full fine-tuning, although they reach similar task accuracy, produce weight updates with different spectral structure. LoRA-tuned matrices acquire high-ranking singular vectors with near-zero cosine similarity to any pretrained singular vector ("intruder dimensions"), 

*Full summary: [arxiv_2410.21228v3.md](summaries/arxiv_2410.21228v3.md)*

---

## 12. Watch the Weights: Unsupervised Monitoring and Control of Fine-Tuned LLMs

**Relevance Score:** 6/10
**Source:** google_scholar
**URL:** https://proceedings.iclr.cc/paper_files/paper/2026/hash/79ee64e62fd3ee763961ce3d6aed4971-Abstract-Conference.html

The paper proposes WEIGHTWATCH, a data-free method for finding and monitoring behaviours introduced by fine-tuning. It takes the SVD of the weight difference ΔW = W_tuned - W_base at the attention output projection and MLP down projection of each layer, treats the top-k left singular vectors as "beh

*Full summary: [Watch_the_Weights_Unsupervised_monitoring_and_con.md](summaries/Watch_the_Weights_Unsupervised_monitoring_and_con.md)*

---

## 13. Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?

**Relevance Score:** 6/10
**Source:** arxiv
**Year:** 2025
**Authors:** Yang Yue, Zhiqi Chen, Rui Lu et al.
**URL:** http://arxiv.org/abs/2504.13837v5

The paper asks whether reinforcement learning with verifiable rewards (RLVR) gives LLMs reasoning abilities beyond those of their base model, or only makes the base model's existing successful reasoning paths more likely. Using pass@k at large k as a measure of the reasoning boundary, across Qwen2.5

*Full summary: [arxiv_2504.13837v5.md](summaries/arxiv_2504.13837v5.md)*

---

## 14. Overcoming Sparsity Artifacts in Crosscoders to Interpret Chat-Tuning

**Relevance Score:** 6/10
**Source:** arxiv
**Year:** 2025
**Authors:** Julian Minder, Clément Dumas, Caden Juang et al.
**URL:** http://arxiv.org/abs/2504.02922v4

The paper examines whether crosscoders (shared-dictionary SAEs trained on base and fine-tuned model activations) can identify concepts that chat-tuning introduces. It argues that the standard L1-trained crosscoder produces many spurious "chat-only" latents because of two loss-induced artifacts (Comp

*Full summary: [arxiv_2504.02922v4.md](summaries/arxiv_2504.02922v4.md)*

---

## 15. Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences

**Relevance Score:** 6/10
**Source:** google_scholar
**URL:** https://proceedings.iclr.cc/paper_files/paper/2026/hash/3b939edfdf9c1211fb764a888078f13d-Abstract-Conference.html

The paper shows that narrow finetuning leaves static, interpretable biases in the residual activations of a model. Averaging the activation difference between finetuned and base models over the first few tokens of unrelated web text, and then decoding it with Patchscope and steering, recovers the fi

*Full summary: [Narrow_finetuning_leaves_clearly_readable_traces_i.md](summaries/Narrow_finetuning_leaves_clearly_readable_traces_i.md)*

---

## 16. Understanding On-Policy Distillation: A Mechanistic Interpretability Perspective via Sparse Crosscoders

**Relevance Score:** 6/10
**Source:** arxiv
**Year:** 2026
**Authors:** Zichao Yu, Qianshuo Ye, Xu Wang et al.
**URL:** http://arxiv.org/abs/2609.35210v1

The paper asks what on-policy distillation (OPD) actually changes inside a student model, using sparse crosscoders trained jointly on the student before and after OPD and on the teacher. It proposes a "swap readout" that reads each student checkpoint's feature activations on its own, which lets feat

*Full summary: [arxiv_2609.35210v1.md](summaries/arxiv_2609.35210v1.md)*

---

## 17. The persona selection model

**Relevance Score:** 6/10
**Source:** lesswrong
**Year:** 2026
**Authors:** Sam Marks
**URL:** https://www.lesswrong.com/posts/dfoty34sT7CSKeJNn/the-persona-selection-model

The post names and argues for the *persona selection model* (PSM): pre-training teaches an LLM a distribution over characters (personas), and post-training mainly refines which persona, the "Assistant", the model simulates, so assistant behaviour is largely explained by that persona's traits. It sur

*Full summary: [lw_dfoty34sT7CSKeJNn.md](summaries/lw_dfoty34sT7CSKeJNn.md)*

---

## 18. What Does an LLM Learn from Reinforcement Learning? A Mechanistic Interpretability Perspective with Fixed-SAE Track

**Relevance Score:** 6/10
**Source:** google_scholar
**URL:** https://arxiv.org/abs/2609.15064

The paper proposes Fixed-SAE Track: one sparse autoencoder (TopK) is trained per layer on residual-stream activations pooled from the base model and all RL checkpoints, then frozen and used to re-encode every checkpoint, so each latent index means the same direction throughout training. Applied to G

*Full summary: [arxiv_2609.15064.md](summaries/arxiv_2609.15064.md)*

---

## 19. [Paper] Stochastic Parameter Decomposition

**Relevance Score:** 6/10
**Source:** alignment_forum
**Year:** 2025
**Authors:** Lee Sharkey
**URL:** https://www.lesswrong.com/posts/yjrpmCmqurDmbMztW/paper-stochastic-parameter-decomposition

The post announces Stochastic Parameter Decomposition (SPD), a linear parameter decomposition method meant to replace Attribution-based Parameter Decomposition (APD) as the practical method in that framework. The authors claim SPD is cheaper, less sensitive to hyperparameters, avoids shrinkage of le

*Full summary: [lw_yjrpmCmqurDmbMztW.md](summaries/lw_yjrpmCmqurDmbMztW.md)*

---

## 20. SAEs (usually) Transfer Between Base and Chat Models

**Relevance Score:** 6/10
**Source:** lesswrong
**Year:** 2024
**Authors:** Connor Kissane
**URL:** https://www.lesswrong.com/posts/fmwk6qxrpW8d4jvbd/saes-usually-transfer-between-base-and-chat-models

The authors train sparse autoencoders (SAEs) on base models and on their chat fine-tunes (Mistral-7B, Qwen 1.5 0.5B, and Gemma v1 2B) and test whether each SAE reconstructs activations from the other model. For Mistral-7B and Qwen they largely do, except for rare high-norm outlier tokens (mostly BOS

*Full summary: [lw_fmwk6qxrpW8d4jvbd.md](summaries/lw_fmwk6qxrpW8d4jvbd.md)*

---

## 21. Sparse Autoencoders Trained on the Same Data Learn Different Features

**Relevance Score:** 5/10
**Source:** arxiv
**Year:** 2025
**Authors:** Gonçalo Paulo, Nora Belrose
**URL:** http://arxiv.org/abs/2501.16615v2

The paper asks whether sparse autoencoders (SAEs) recover a unique, objective set of features in a language model. Training SAEs on identical activations and data order, differing only in the random initialisation seed, the authors match latents across seeds with the Hungarian algorithm and find tha

*Full summary: [arxiv_2501.16615v2.md](summaries/arxiv_2501.16615v2.md)*

---

## 22. Rank-1 LoRAs Encode Interpretable Reasoning Signals

**Relevance Score:** 5/10
**Source:** arxiv
**Year:** 2025
**Authors:** Jake Ward, Paul Riechers, Adam Shai
**URL:** http://arxiv.org/abs/2511.06739v1

The authors fine-tune Qwen-2.5-32B-Instruct on the s1k-1.1 DeepSeek-R1 reasoning traces with a rank-1 LoRA applied to all MLP and attention projections (under 0.03% of parameters). This adapter recovers 73-90% of the base-to-full-finetune gain on AIME'24, MATH500 and GPQA-Diamond. They then interpre

*Full summary: [arxiv_2511.06739v1.md](summaries/arxiv_2511.06739v1.md)*

---

## 23. LOFT: Low-Rank Adaptation That Behaves Like Full Fine-Tuning

**Relevance Score:** 5/10
**Source:** google_scholar
**URL:** https://proceedings.iclr.cc/paper_files/paper/2026/hash/7428310c0f97f1c6bb2ef1be99c1ec2a-Abstract-Conference.html

LoFT is a LoRA-style parameter-efficient fine-tuning method that tries to make the low-rank update follow the same optimizer trajectory as full fine-tuning with AdamW. It goes beyond approximating the gradient inside the low-rank subspace: it also projects the first and second Adam moments into the 

*Full summary: [LoftLow-rankadaptation_that_behaves_like_fullfine.md](summaries/LoftLow-rankadaptation_that_behaves_like_fullfine.md)*

---

## 24. From Weights to Concepts: Data-Free Interpretability of CLIP via Singular Vector Decomposition

**Relevance Score:** 5/10
**Source:** google_scholar
**Year:** 2026
**URL:** https://openaccess.thecvf.com/content/CVPR2026/html/Gentile_From_Weights_to_Concepts_Data-Free_Interpretability_of_CLIP_via_Singular_CVPR_2026_paper.html

The paper introduces SITH, a data-free, training-free method that interprets CLIP's vision transformer directly in weight space. It takes each attention head's value-output (VO) matrix, factorises it with SVD, and labels each singular vector with a sparse, coherent set of text concepts using a new d

*Full summary: [arxiv_2603.24653.md](summaries/arxiv_2603.24653.md)*

---

## 25. SVFT: Parameter-Efficient Fine-Tuning with Singular Vectors

**Relevance Score:** 5/10
**Source:** arxiv
**Year:** 2024
**Authors:** Vijay Lingam, Atula Tejaswi, Aditya Vavre et al.
**URL:** http://arxiv.org/abs/2405.19597v1

SVFT is a parameter-efficient fine-tuning (PEFT) method that constrains the weight update to the pretrained matrix's own geometry: for a weight W = U Σ V^T, it sets ΔW = U M V^T and trains only a sparse set of coefficients in M, with U and V frozen. The sparsity pattern of M (Plain/diagonal, Banded,

*Full summary: [arxiv_2405.19597v1.md](summaries/arxiv_2405.19597v1.md)*

---

## 26. Task Arithmetic in the Tangent Space: Improved Editing of Pre-Trained Models

**Relevance Score:** 5/10
**Source:** arxiv
**Year:** 2023
**Authors:** Guillermo Ortiz-Jimenez, Alessandro Favero, Pascal Frossard
**URL:** http://arxiv.org/abs/2305.12827v3

The paper studies why task arithmetic (adding or negating fine-tuning "task vectors" in weight space) works for CLIP vision-language models. It argues the key property is weight disentanglement: distinct weight-space directions change the model's function in disjoint input regions. Fine-tuning in th

*Full summary: [arxiv_2305.12827v3.md](summaries/arxiv_2305.12827v3.md)*

---

## 27. Feature Evolution and Migration during Vision Transformer Training

**Relevance Score:** 5/10
**Source:** google_scholar
**URL:** https://arxiv.org/abs/2608.20134

The paper tracks visual features in a ViT-Tiny trained from scratch on ImageNet-1k and a 10-superclass subset, using a two-dimensional epoch-by-layer map. It trains a sparse autoencoder (SAE) at every checkpoint and layer on CLS-token activations, then matches SAE features across (checkpoint, layer)

*Full summary: [arxiv_2608.20134.md](summaries/arxiv_2608.20134.md)*

---

## 28. Learning to Interpret Weight Differences in Language Models

**Relevance Score:** 5/10
**Source:** google_scholar
**URL:** https://proceedings.iclr.cc/paper_files/paper/2026/hash/f536d5697b79a9b3b3debbb7a552a7da-Abstract-Conference.html

The paper introduces Diff Interpretation Tuning (DIT), which trains a LoRA "DIT-adapter" on synthetic, labelled fine-tunes so that a fine-tuned model can answer natural-language questions about how its own weights differ from the base model (e.g. "What topic were you trained on?"). It is shown on tw

*Full summary: [Learning_to_interpret_weight_differences_in_langua.md](summaries/Learning_to_interpret_weight_differences_in_langua.md)*

---

## 29. Delta-Crosscoder: Robust Crosscoder Model Diffing in Narrow Fine-Tuning Regimes

**Relevance Score:** 5/10
**Source:** arxiv
**Year:** 2026
**Authors:** Aly Kassem, Thomas Jiralerspong, Negar Rostamzadeh et al.
**URL:** http://arxiv.org/abs/2603.04426v1

The paper modifies standard crosscoders (shared sparse dictionaries trained on base and fine-tuned model activations) so that they isolate the representation shifts caused by narrow fine-tuning. The modifications are a Dual-K BatchTopK budget with a reserved 20% shared-latent subspace, a "delta loss

*Full summary: [arxiv_2603.04426v1.md](summaries/arxiv_2603.04426v1.md)*

---

## 30. Decomposition of Small Transformer Models

**Relevance Score:** 5/10
**Source:** arxiv
**Year:** 2025
**Authors:** Casper L. Christensen, Logan Riggs
**URL:** http://arxiv.org/abs/2511.08854v2

The paper extends Stochastic Parameter Decomposition (SPD, Bushnaq et al. 2025) from toy models to Transformers. It adds a causal-importance function that uses a small attention network across token positions, and a randomised partial reconstruction loss that evaluates only a subset of layers at a t

*Full summary: [arxiv_2511.08854v2.md](summaries/arxiv_2511.08854v2.md)*

---

## 31. Reasoning-Finetuning Repurposes Latent Representations in Base Models

**Relevance Score:** 5/10
**Source:** lesswrong
**Year:** 2025
**Authors:** Jake Ward
**URL:** https://www.lesswrong.com/posts/J9BiKfJ4YvNd2Lwbh/reasoning-finetuning-repurposes-latent-representations-in

The authors compute a difference-of-means steering vector for "backtracking" (e.g. "Wait", "But") from activations of the base model Llama-3.1-8B at layer 10, using token positions just before backtracking sentences in DeepSeek-R1 reasoning traces. Added to the reasoning-finetuned model DeepSeek-R1-

*Full summary: [lw_J9BiKfJ4YvNd2Lwbh.md](summaries/lw_J9BiKfJ4YvNd2Lwbh.md)*

---

## 32. Weights Read and Write Features: Scalable Parameter Decomposition Grounded in Activation Space

**Relevance Score:** 5/10
**Source:** google_scholar
**URL:** https://arxiv.org/abs/2609.37731

The paper introduces ASPD (Activation-Supported Parameter Decomposition), a method that decomposes a weight matrix into rank-1 components while jointly learning a shared sparse code over activation features, so each weight component is tied to the activation feature it reads or writes. It adds a loc

*Full summary: [arxiv_2609.37731.md](summaries/arxiv_2609.37731.md)*

---

## 33. See It, Say It, Sorted: Mechanistic Diagnosis and Parameter-Space Mitigation of Emergent Misalignment in LLMs

**Relevance Score:** 5/10
**Source:** google_scholar
**URL:** https://arxiv.org/abs/2609.34970

The paper tracks emergent misalignment (EM) during rank-1 LoRA fine-tuning on narrow harmful data (finance, sports, medicine). It reports that the loss curvature of "pivot" tokens (those whose probability rises most as the model turns misaligned) grows relative to neutral tokens, and that the update

*Full summary: [arxiv_2609.34970.md](summaries/arxiv_2609.34970.md)*

---

## 34. Beyond the Leaderboard: Understanding Performance Disparities in Large Language Models via Model Diffing

**Relevance Score:** 5/10
**Source:** arxiv
**Year:** 2025
**Authors:** Sabri Boughorbel, Fahim Dalvi, Nadir Durrani et al.
**URL:** http://arxiv.org/abs/2509.18792v1

The paper applies model diffing with BatchTopK crosscoders to compare Gemma-2-9b-it with its SimPO fine-tune (and, in an appendix, a DPO fine-tune), using the shared base model Gemma-2-9b-pt as the anchor. It finds about 92 latents unique to the SimPO model and 113 unique to the instruct model, grou

*Full summary: [arxiv_2509.18792v1.md](summaries/arxiv_2509.18792v1.md)*

---

## 35. Sparse Weight Decomposition for Efficient Circuit Extraction

**Relevance Score:** 5/10
**Source:** google_scholar
**URL:** https://arxiv.org/abs/2608.03913

Sparse Weight Decomposition (SWD) post-hoc factorizes each pretrained dense linear projection W into two sparse factors, W ≈ AB, so that each shared intermediate coordinate ("bottleneck unit") is a rank-one read-write path with few nonzero input and output edges. These units are then scored, pruned 

*Full summary: [arxiv_2608.03913.md](summaries/arxiv_2608.03913.md)*

---

## 36. The Path Not Taken: RLVR Provably Learns Off the Principals

**Relevance Score:** 5/10
**Source:** manual
**Year:** 2025
**URL:** https://arxiv.org/abs/2511.08567

The paper argues that the apparent sparsity of RLVR (RL with verifiable rewards) weight updates is a surface effect of a stable, model-conditioned optimization bias: for a fixed pretrained model, RL updates land in the same parameter regions across runs and datasets, and bf16 storage makes the sub-U

*Full summary: [arxiv_2511.08567.md](summaries/arxiv_2511.08567.md)*

---

## 37. Model Organisms for Emergent Misalignment

**Relevance Score:** 5/10
**Source:** semantic_scholar
**Year:** 2025
**Authors:** Edward Turner, Anna Soligo, Mia Taylor et al.
**URL:** https://www.semanticscholar.org/paper/b52f7aa85ef5ddade34324f5e86cf04c01f404d2

The paper builds cleaner model organisms of emergent misalignment (EM), where narrow harmful fine-tuning produces broadly misaligned behaviour. It uses three new narrow text datasets (bad medical advice, risky financial advice, extreme sports) to get about 40% misalignment at over 99% coherence, dow

*Full summary: [10.48550_arXiv.2506.11613.md](summaries/10.48550_arXiv.2506.11613.md)*

---

## 38. Identifying Functionally Important Features with End-to-End Sparse Dictionary Learning

**Relevance Score:** 5/10
**Source:** arxiv
**Year:** 2024
**Authors:** Dan Braun, Jordan Taylor, Nicholas Goldowsky-Dill et al.
**URL:** http://arxiv.org/abs/2405.12241v2

The paper proposes training sparse autoencoders (SAEs) end-to-end: instead of minimising reconstruction MSE at one layer, the SAE output replaces the activations and the loss is the KL divergence between the original and patched model's output logits (optionally plus MSE reconstruction at downstream

*Full summary: [arxiv_2405.12241v2.md](summaries/arxiv_2405.12241v2.md)*

---

## 39. Beyond Static Interpretability: Anticipating Post-SFT Mechanisms from Pre-SFT Parameters for Better Tuning

**Relevance Score:** 5/10
**Source:** google_scholar
**URL:** https://arxiv.org/abs/2608.24482

The paper argues that "locate-then-tune" parameter-efficient SFT, which picks the parameters to tune using causal attribution on the pre-SFT model, is misled when the target skill is new, because the circuits that matter only appear after training. The authors propose estimating the post-SFT attribu

*Full summary: [arxiv_2608.24482.md](summaries/arxiv_2608.24482.md)*

---

## 40. Tied Crosscoders: Explaining Chat Behavior from Base Model

**Relevance Score:** 5/10
**Source:** alignment_forum
**Year:** 2025
**Authors:** Santiago Aranguri
**URL:** https://www.lesswrong.com/posts/3T8eKyaPvDDm2wzor/research-question

The author proposes a "tied crosscoder" for model-diffing between a base model and its chat fine-tune (Gemma 2 2B vs Gemma 2 2B it, layer 14). Each latent group has three members (base-only, shared, chat-only) that share a decoder direction, so the same concept can fire at different times in the two

*Full summary: [lw_3T8eKyaPvDDm2wzor.md](summaries/lw_3T8eKyaPvDDm2wzor.md)*

---

## 41. Revisiting the Superficial Alignment Hypothesis

**Relevance Score:** 4/10
**Source:** arxiv
**Year:** 2024
**Authors:** Mohit Raghavendra, Vaskar Nath, Sean Hendryx
**URL:** http://arxiv.org/abs/2410.03717v1

The paper tests the Superficial Alignment Hypothesis (SAH), which holds that post-training mainly teaches style and format while knowledge and capabilities come from pre-training. The authors fine-tune Llama-2, Llama-3 and Mistral base models on math, coding, instruction-following and multihop QA wi

*Full summary: [arxiv_2410.03717v1.md](summaries/arxiv_2410.03717v1.md)*

---

## 42. Fine-tuning Happens in Tiny Subspaces: Exploring Intrinsic Task-specific Subspaces of Pre-trained Language Models

**Relevance Score:** 4/10
**Source:** arxiv
**Year:** 2023
**Authors:** Zhong Zhang, Bang Liu, Junming Shao
**URL:** http://arxiv.org/abs/2305.17446v2

The paper proposes that fine-tuning a pre-trained language model (BERT-base-cased, RoBERTa-base) happens inside a small task-specific subspace. The subspace is estimated by stacking the parameter checkpoints of an ordinary fine-tuning run (one per epoch), taking an SVD, and using the principal right

*Full summary: [arxiv_2305.17446v2.md](summaries/arxiv_2305.17446v2.md)*

---

## 43. Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning

**Relevance Score:** 4/10
**Source:** arxiv
**Year:** 2020
**Authors:** Armen Aghajanyan, Luke Zettlemoyer, Sonal Gupta
**URL:** http://arxiv.org/abs/2012.13255v1

The paper measures the intrinsic dimension (d90) of fine-tuning pretrained language models: the smallest random-subspace dimension that still reaches 90% of full fine-tuning performance. It finds that GLUE-style tasks can be solved to that level with a few hundred to a few thousand trainable paramet

*Full summary: [arxiv_2012.13255v1.md](summaries/arxiv_2012.13255v1.md)*

---

## 44. What We Learned Trying to Diff Base and Chat Models (And Why It Matters)

**Relevance Score:** 4/10
**Source:** alignment_forum
**Year:** 2025
**Authors:** Clément Dumas
**URL:** https://www.lesswrong.com/posts/xmpauEXEerzYcJKNm/what-we-learned-trying-to-diff-base-and-chat-models-and-why

The post argues that model diffing (studying what fine-tuning changes internally, here base vs. chat Gemma-2 2B) is a promising direction, and reports that sparse-dictionary methods have problems in this setting. Crosscoders produce many spurious "chat-only" latents that are artifacts of the L1 pena

*Full summary: [lw_xmpauEXEerzYcJKNm.md](summaries/lw_xmpauEXEerzYcJKNm.md)*

---

## 45. Curriculum Reinforcement Learning Can Incentivize Reasoning Capacity in LLMs Beyond the Base Model

**Relevance Score:** 4/10
**Source:** arxiv
**Year:** 2026
**Authors:** Pengxiang Cai, Tianchen Fang, Xiaohan Li et al.
**URL:** http://arxiv.org/abs/2606.22317v1

The paper proposes "boundary-aware Curriculum RL" for LLM reasoning. Each round uses large-k pass@k sampling (k up to 256) to sort training problems into difficulty bins, adds a small SFT bridge of teacher-written reasoning traces on problems just inside or just outside the model's observed boundary

*Full summary: [arxiv_2606.22317v1.md](summaries/arxiv_2606.22317v1.md)*

---

## 46. Implicit Bias of Gradient Descent on Reparametrized Models: On Equivalence to Mirror Descent

**Relevance Score:** 4/10
**Source:** arxiv
**Year:** 2022
**Authors:** Zhiyuan Li, Tianhao Wang, JasonD. Lee et al.
**URL:** http://arxiv.org/abs/2207.04036v1

The paper characterises when gradient flow on a reparametrised model w = G(x) is exactly mirror flow in the original parameter space w. The key condition is that the reparametrisation is "commuting" (the Lie brackets of the coordinate gradients vanish), and the paper shows this is sufficient, nearly

*Full summary: [arxiv_2207.04036v1.md](summaries/arxiv_2207.04036v1.md)*

---

## 47. Reviving Your Mneme: Predicting the Side Effects of LLM Unlearning and Fine-Tuning via Sparse Model Diffing

**Relevance Score:** 4/10
**Source:** arxiv
**Year:** 2025
**Authors:** Aly M. Kassem, Zhuan Shi, Negar Rostamzadeh et al.
**URL:** http://arxiv.org/abs/2507.21084v1

The paper introduces MNEME, a framework that flags unintended side effects of fine-tuning or unlearning by training a sparse crosscoder on base and edited model activations over task-agnostic text (The Pile, LMSYS-Chat-1M), then scoring each latent as amplified, suppressed or unchanged via latent sc

*Full summary: [arxiv_2507.21084v1.md](summaries/arxiv_2507.21084v1.md)*

---

## 48. Data Attribution of Emergent Misalignment with Persona Features

**Relevance Score:** 4/10
**Source:** google_scholar
**URL:** https://arxiv.org/abs/2608.11025

The paper studies emergent misalignment (EM), where narrow fine-tuning makes a model behave badly in unrelated domains. Using sparse autoencoder (SAE) model diffing on four open-weight models (Llama-3.1-8B, Qwen2.5-7B, Gemma 2 9B, Gemma 3 27B), it finds that misalignment fine-tuning amplifies jailbr

*Full summary: [arxiv_2608.11025.md](summaries/arxiv_2608.11025.md)*

---

## 49. Singular Vectors of Attention Heads Align with Features

**Relevance Score:** 4/10
**Source:** arxiv
**Year:** 2026
**Authors:** Gabriel Franco, Carson Loughridge, Mark Crovella
**URL:** http://arxiv.org/abs/2602.13524v2

The paper asks why and when the singular vectors of an attention head's QK matrix (Ω = W_Q^T W_K) line up with the linear feature directions the head uses. It shows this alignment in a toy autoencoder-plus-attention model, proves conditions under which it holds, and proposes "sparse attention decomp

*Full summary: [arxiv_2602.13524v2.md](summaries/arxiv_2602.13524v2.md)*

---

## 50. Where Pretraining writes and Alignment reads: the asymmetry of Transformer weight space

**Relevance Score:** 4/10
**Source:** arxiv
**Year:** 2026
**Authors:** Valeria Ruscio, Eli-Shaoul Khedouri, Keiran Thompson
**URL:** http://arxiv.org/abs/2605.16600v1

The paper asks where preference alignment (SFT/DPO-style) deposits its weight changes inside a transformer. It introduces a relative-subspace-fraction (RSF) probe that measures how weight deltas align with residual-stream activation subspaces and with the unembedding's prediction subspace. The main 

*Full summary: [arxiv_2605.16600v1.md](summaries/arxiv_2605.16600v1.md)*

---

## 51. Localizing RL-Induced Tool Use to a Single Crosscoder Feature

**Relevance Score:** 4/10
**Source:** semantic_scholar
**Year:** 2026
**Authors:** A. Shportko, Shubham Bhokare, Ahmed Zeyad A Alzahrani et al.
**URL:** https://www.semanticscholar.org/paper/8260cfacfa34cb72b2b73b9add76eee568a65ae4

The authors train 48 Dedicated Feature Crosscoders (DFC) and plain Crosscoders on paired activations of Qwen2.5-3B base and its ToolRL-RL-fine-tuned version, and ask whether the RL-specific capability (structured `<tool_call>` generation) concentrates in the A-exclusive (RL-model) partition of the d

*Full summary: [10.48550_arXiv.2606.26474.md](summaries/10.48550_arXiv.2606.26474.md)*

---

## 52. Distilling Linearized Behavior into Non-Linear Fine-Tuning for Effective Task Arithmetic

**Relevance Score:** 4/10
**Source:** arxiv
**Year:** 2026
**Authors:** Thomas Sommariva, Francesca Morandi, Simone Calderara et al.
**URL:** http://arxiv.org/abs/2605.18993v2

The paper proposes DELTA, a training method that gives a standard non-linear fine-tune the weight-space properties of linearized (tangent-space) fine-tuning. A linearized teacher (with curvature-aware regularization) is trained jointly with a non-linear student, and the student matches the teacher's

*Full summary: [arxiv_2605.18993v2.md](summaries/arxiv_2605.18993v2.md)*

---

## 53. The Low-Rank Structure of VLA Reinforcement Learning

**Relevance Score:** 4/10
**Source:** google_scholar
**URL:** https://arxiv.org/abs/2609.34599

The paper studies how reinforcement learning (PPO) post-training changes flow-based vision-language-action (VLA) robot policies (pi0.5, GR00T N1.5/N1.6, SmolVLA) in parameter space. It finds that RL updates are dense but low-rank and concentrated in the "Timestep Modules" of the action expert (the T

*Full summary: [arxiv_2609.34599.md](summaries/arxiv_2609.34599.md)*

---

## 54. Fine-Tuning Attention Modules Only: Enhancing Weight Disentanglement in Task Arithmetic

**Relevance Score:** 4/10
**Source:** arxiv
**Year:** 2024
**Authors:** Ruochen Jin, Bojian Hou, Jiancong Xiao et al.
**URL:** http://arxiv.org/abs/2407.07089v2

The paper studies task arithmetic, where separately fine-tuned "task vectors" (weight differences from a pretrained model) are added together to get one multi-task model. It argues that interference between tasks comes from a lack of "weight disentanglement," and that attention modules alone show ne

*Full summary: [arxiv_2407.07089v2.md](summaries/arxiv_2407.07089v2.md)*

---

## 55. Shadow-FT: Tuning Instruct Models via Training on Paired Base Model

**Relevance Score:** 4/10
**Source:** arxiv
**Year:** 2025
**Authors:** Taiqiang Wu, Runming Yang, Jiayi Li et al.
**URL:** http://arxiv.org/abs/2505.12716v3

The paper observes that instruction-tuned (INSTRUCT) models often gain little, or even lose performance, when fine-tuned directly on a small dataset, while their paired pretrained BASE models have almost identical weights (relative gap σ about 0.016 for Llama-3.1-8B, below 0.03 across the models tes

*Full summary: [arxiv_2505.12716v3.md](summaries/arxiv_2505.12716v3.md)*

---

## 56. Unstable Features, Reproducible Subspaces: Understanding Seed Dependence in Sparse Autoencoders

**Relevance Score:** 4/10
**Source:** arxiv
**Year:** 2026
**Authors:** Gleb Gerasimov, Timofei Rusalev, Nikita Balagansky et al.
**URL:** http://arxiv.org/abs/2606.12138v1

The paper measures per-feature "feature stability" in sparse autoencoders (SAEs): the probability that a feature reappears (cosine similarity at least 0.7 on decoder vectors) in SAEs trained with different seeds. Stable features carry most of the reconstruction and next-token relevance, while unstab

*Full summary: [arxiv_2606.12138v1.md](summaries/arxiv_2606.12138v1.md)*

---

## 57. The Intruder Threshold: A Spectral Law for LoRA Fine-Tuning

**Relevance Score:** 4/10
**Source:** arxiv
**Year:** 2026
**Authors:** Peng Xie, Amr Alanwar
**URL:** http://arxiv.org/abs/2607.23711v2

The paper derives a per-layer critical update strength s* = θ̄ / (γ σ₁(BA)) at which a LoRA update creates an "intruder dimension" (a new leading singular vector of W + γBA that is nearly orthogonal to all pretrained singular vectors). θ̄ is computed from the measured spectrum of the pretrained W al

*Full summary: [arxiv_2607.23711v2.md](summaries/arxiv_2607.23711v2.md)*

---

## 58. Operationalising the Superficial Alignment Hypothesis via Task Complexity

**Relevance Score:** 4/10
**Source:** arxiv
**Year:** 2026
**Authors:** Tomás Vergara-Browne, Darshan Patil, Ivan Titov et al.
**URL:** http://arxiv.org/abs/2602.15829v2

The paper gives the superficial alignment hypothesis (SAH) a precise definition: "task complexity" is the length in bits of the shortest program (a Python script plus any weights or data it carries) that reaches a target performance on a task, given access to a pre-trained model's weights. It argues

*Full summary: [arxiv_2602.15829v2.md](summaries/arxiv_2602.15829v2.md)*

---

## 59. Do Sparse Autoencoders (SAEs) transfer across base and finetuned language models?

**Relevance Score:** 4/10
**Source:** lesswrong
**Year:** 2024
**Authors:** Taras Kutsyk
**URL:** https://www.lesswrong.com/posts/bsXPTiAhhwt5nwBW3/do-sparse-autoencoders-saes-transfer-across-base-and

The post asks whether an SAE trained on a base language model still works on a finetuned version of that model, without retraining the SAE. It compares two base/finetune pairs (Gemma-2b vs. a Python-code finetune; Mistral-7B vs. MetaMath) using reconstruction metrics, residual-activation similarity,

*Full summary: [lw_bsXPTiAhhwt5nwBW3.md](summaries/lw_bsXPTiAhhwt5nwBW3.md)*

---

## 60. Accelerating Sparse Autoencoder Training via Layer-Wise Transfer Learning in Large Language Models

**Relevance Score:** 4/10
**Source:** semantic_scholar
**Year:** 2024
**Authors:** Davide Ghilardi, Federico Belotti, Marco Molinari et al.
**URL:** https://www.semanticscholar.org/paper/cb8c55997787c28e9eb733bd9e6ce605973e5685

The paper asks whether a sparse autoencoder (SAE) trained on one layer of Pythia-160M can initialise an SAE for an adjacent layer (forward from layer i-1, backward from layer i+1), so that only a short fine-tune is needed instead of training from scratch. Fine-tuning on about 10% of the training bud

*Full summary: [10.18653_v1_2024.blackboxnlp-1.32.md](summaries/10.18653_v1_2024.blackboxnlp-1.32.md)*

---

## 61. On Distinguishing Capability Elicitation from Capability Creation in Post-Training: A Free-Energy Perspective

**Relevance Score:** 4/10
**Source:** arxiv
**Year:** 2026
**Authors:** Yuhao Li, Shengchao Liu
**URL:** http://arxiv.org/abs/2605.08368v1

The paper argues that post-training research should separate capability elicitation (reweighting behaviours a base model can already reach under finite budgets) from capability creation (changing the set of reachable behaviours). It frames SFT and KL-regularised RL as reweightings of a pretrained re

*Full summary: [arxiv_2605.08368v1.md](summaries/arxiv_2605.08368v1.md)*

---

## 62. Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution

**Relevance Score:** 4/10
**Source:** arxiv
**Year:** 2022
**Authors:** Ananya Kumar, Aditi Raghunathan, Robbie Jones et al.
**URL:** http://arxiv.org/abs/2202.10054v1

The paper shows that full fine-tuning of a pretrained model can give worse out-of-distribution (OOD) accuracy than linear probing (training only the head on frozen features), even though fine-tuning gives better in-distribution (ID) accuracy, when the pretrained features are good and the distributio

*Full summary: [arxiv_2202.10054v1.md](summaries/arxiv_2202.10054v1.md)*

---

## 63. Extracting and Understanding the Superficial Knowledge in Alignment

**Relevance Score:** 4/10
**Source:** arxiv
**Year:** 2025
**Authors:** Runjin Chen, Gabriel Jacob Perin, Xuxi Chen et al.
**URL:** http://arxiv.org/abs/2502.04602v1

The paper asks whether alignment (RLHF/SFT-style post-training) is "predominantly superficial". It defines superficial knowledge as the part of an aligned model's behaviour that can be reproduced by changing only the final linear projection head, with the backbone frozen, trained by distillation fro

*Full summary: [arxiv_2502.04602v1.md](summaries/arxiv_2502.04602v1.md)*

---

## 64. Gradient-Sign Masking for Task Vector Transport Across Pre-trained Models

**Relevance Score:** 4/10
**Source:** google_scholar
**URL:** https://proceedings.iclr.cc/paper_files/paper/2026/hash/59e4f5d009f95324a1a1aa9971b51e25-Abstract-Conference.html

The paper proposes GradFix, a way to move a fine-tuning update (task vector) from one pre-trained model A to a different pre-trained model B without fine-tuning B. The source task vector is masked coordinate-wise by the sign of a few target-model gradients (estimated by majority vote over a handful 

*Full summary: [Gradient-sign_masking_fortaskvector_transport_acro.md](summaries/Gradient-sign_masking_fortaskvector_transport_acro.md)*

---

## 65. SharedSAE: One Feature Dictionary Across Language Models

**Relevance Score:** 4/10
**Source:** arxiv
**Year:** 2026
**Authors:** Daniil Ognev, Célian Vasson, Lijie Hu et al.
**URL:** http://arxiv.org/abs/2609.04344v1

SharedSAE trains a single sparse autoencoder dictionary jointly across four ~1B-parameter base language models from different families and tokenizers, using per-model encoder/decoder pairs around one shared set of latents. The authors argue that the shared latents are aligned across models (they sel

*Full summary: [arxiv_2609.04344v1.md](summaries/arxiv_2609.04344v1.md)*

---

## 66. Towards Developmental Interpretability

**Relevance Score:** 3/10
**Source:** lesswrong
**Year:** 2023
**Authors:** Jesse Hoogland
**URL:** https://www.lesswrong.com/posts/TjaeCWvLZtEDAS5Ex/towards-developmental-interpretability

The post sets out "developmental interpretability", a research agenda that studies how internal structure forms through phase transitions during neural network training, using observables from Singular Learning Theory (SLT) such as the RLCT and singular fluctuation. It is a conceptual agenda with a 

*Full summary: [lw_TjaeCWvLZtEDAS5Ex.md](summaries/lw_TjaeCWvLZtEDAS5Ex.md)*

---

## 67. Layer-Wise Evolution of Representations in Fine-Tuned Transformers: Insights from Sparse AutoEncoders

**Relevance Score:** 3/10
**Source:** arxiv
**Year:** 2025
**Authors:** Suneel Nadipalli
**URL:** http://arxiv.org/abs/2502.16722v1

The paper fine-tunes BERT-base-uncased (and smaller BERT variants) on three text classification tasks (IMDb sentiment, Spotify ratings, news topics). It compares pre-trained and fine-tuned activations by per-layer cosine similarity of mean-pooled hidden states, and by training a separate sparse auto

*Full summary: [arxiv_2502.16722v1.md](summaries/arxiv_2502.16722v1.md)*

---

## 68. Understanding and Enforcing Weight Disentanglement in Task Arithmetic

**Relevance Score:** 3/10
**Source:** arxiv
**Year:** 2026
**Authors:** Shangge Liu, Yuehan Yin, Lei Wang et al.
**URL:** http://arxiv.org/abs/2604.17078v2

The paper proposes Task-Feature Specialization (TFS), the idea that a model assigns distinct internal features to distinct tasks, as the underlying reason task arithmetic works. It argues TFS is sufficient for weight disentanglement and gives rise to weight-vector orthogonality, and it proposes Orth

*Full summary: [arxiv_2604.17078v2.md](summaries/arxiv_2604.17078v2.md)*

---

## 69. Measuring Dead Directions: Decomposing and Classifying Singular Structure off Canonical Alignment

**Relevance Score:** 3/10
**Source:** arxiv
**Year:** 2026
**Authors:** Tejas Pradeep Shirodkar
**URL:** http://arxiv.org/abs/2607.00603v1

The paper proposes a descent-free, alignment-free way to read the singular structure of a trained network from one frozen checkpoint. It nominates a "dead direction" (via K-FAC factors or an architecture-specific detector), scans the directional Fisher along it, and reads off the KL order k, the per

*Full summary: [arxiv_2607.00603v1.md](summaries/arxiv_2607.00603v1.md)*

---

## 70. Towards Resolving the Implicit Bias of Gradient Descent for Matrix Factorization: Greedy Low-Rank Learning

**Relevance Score:** 3/10
**Source:** arxiv
**Year:** 2020
**Authors:** Zhiyuan Li, Yuping Luo, Kaifeng Lyu
**URL:** http://arxiv.org/abs/2012.09839v2

The paper studies the implicit regularization of gradient flow (infinitesimal-step gradient descent) when fitting a symmetric, depth-2 matrix factorization W = UU^T from small initialization. It shows, theoretically and empirically, that this trajectory is equivalent (under stated assumptions) to a 

*Full summary: [arxiv_2012.09839v2.md](summaries/arxiv_2012.09839v2.md)*

---

## 71. Aligned Training: A Parameter-Free Method to Improve Feature Quality and Stability of Sparse Autoencoders (SAE)

**Relevance Score:** 3/10
**Source:** google_scholar
**URL:** https://arxiv.org/abs/2605.18629

The paper proposes "aligned training," a reparameterization of sparse autoencoders (SAEs) in which each feature's encoder row is projected onto the affine hyperplane where its inner product with the feature's decoder column equals one. The authors argue this removes a degeneracy in SAE training, red

*Full summary: [arxiv_2605.18629.md](summaries/arxiv_2605.18629.md)*

---

## 72. Reinforcement Learning Fine-Tuning Enhances Activation Intensity and Diversity in the Internal Circuitry of LLMs

**Relevance Score:** 3/10
**Source:** google_scholar
**URL:** https://proceedings.iclr.cc/paper_files/paper/2026/hash/67d1e25d35292dab646b58c0af77151a-Abstract-Conference.html

The paper asks what online RL post-training (PPO, GRPO) changes inside LLMs, compared with their SFT or DPO starting points. It uses edge attribution patching (EAP) on model-generated math solutions to compare edge-weight statistics before and after RL across four base/post-trained model pairs at ab

*Full summary: [Reinforcement_learningfine-tuningenhances_activati.md](summaries/Reinforcement_learningfine-tuningenhances_activati.md)*

---

## 73. Training Process Transparency through Gradient Interpretability: Early experiments on toy language models

**Relevance Score:** 3/10
**Source:** lesswrong
**Year:** 2023
**Authors:** robertzk
**URL:** https://www.lesswrong.com/posts/DtkA5jysFZGv7W4qP/training-process-transparency-through-gradient

The author records the per-datum parameter change produced by every SGD step while training a small 3-layer transformer on WikiText2 (about 12M parameters, roughly 292k datum-level updates, about 882 GB stored). Large parameter shifts are attributed to the training tokens that caused them, and a tok

*Full summary: [lw_DtkA5jysFZGv7W4qP.md](summaries/lw_DtkA5jysFZGv7W4qP.md)*

---

## 74. Differentiable Faithfulness Alignment for Cross-Model Circuit Transfer

**Relevance Score:** 3/10
**Source:** google_scholar
**URL:** https://arxiv.org/abs/2604.24302

The paper proposes Differentiable Faithfulness Alignment (DFA), which learns a dense linear map W from node-importance scores of a small source model to node scores of a larger target model, trained with a soft, differentiable faithfulness (KL) objective plus an L1 sparsity penalty. The aim is to pr

*Full summary: [arxiv_2604.24302.md](summaries/arxiv_2604.24302.md)*

---

## 75. The Loss Does Not See the Basis, but Adam Does

**Relevance Score:** 3/10
**Source:** google_scholar
**URL:** https://arxiv.org/abs/2608.05136

The paper argues that for a factored model W = UV^T, the loss is invariant to a gauge rotation (U, V) -> (UQ, VQ), and that an optimizer reproduces gradient flow's low-rank implicit bias only if its update is gauge-equivariant. Gradient descent, momentum, shared-scalar Adam, Muon and Shampoo are equ

*Full summary: [arxiv_2608.05136.md](summaries/arxiv_2608.05136.md)*

---

## 76. Cross-seed explainability using Procrustes conditioned Joint End-to-end Top-K Sparse Autoencoders

**Relevance Score:** 3/10
**Source:** arxiv
**Year:** 2026
**Authors:** Bendegúz Váradi, Zoltán Kmetty
**URL:** http://arxiv.org/abs/2607.08499v1

The paper proposes a single joint Top-K sparse autoencoder (SAE), trained on two BERT seeds whose layer-10 activations are first aligned with an orthogonal Procrustes rotation, plus an end-to-end downstream loss and a cross-seed sparse-code loss. The aim is to find features that are "universal" (per

*Full summary: [arxiv_2607.08499v1.md](summaries/arxiv_2607.08499v1.md)*

---

## 77. Intrinsic Structure: Spectral Identifiability for Mechanistic Interpretability

**Relevance Score:** 3/10
**Source:** google_scholar
**URL:** https://arxiv.org/abs/2608.10172

The paper treats a transformer's forward pass as a controlled dynamical system with depth as time, lifts it with the Koopman operator, and argues that the spectrum of the resulting finite linear realisation is a coordinate-free, identifiable property of the model. It proves a finite-sample rate (M^-

*Full summary: [arxiv_2608.10172.md](summaries/arxiv_2608.10172.md)*

---

## 78. Stable and Steerable Sparse Autoencoders with Weight Regularization

**Relevance Score:** 3/10
**Source:** arxiv
**Year:** 2026
**Authors:** Piotr Jedryszek, Oliver M. Crook
**URL:** http://arxiv.org/abs/2603.04198v2

The paper adds an explicit L1 or L2 weight penalty on SAE encoder and decoder weights, on top of the usual activation sparsity, and tests it on an MNIST toy model and on TopK, BatchTopK and Matryoshka SAEs trained on layer-3 of Pythia-70M-deduped. The authors report that L2 weight regularization (wi

*Full summary: [arxiv_2603.04198v2.md](summaries/arxiv_2603.04198v2.md)*

---

## 79. Diff Mining: Logit Differences Reveal Finetuning Objectives

**Relevance Score:** 3/10
**Source:** google_scholar
**URL:** https://arxiv.org/abs/2608.26462

Diff Mining is a simple model-diffing method that compares the output logits of a finetuned model with those of its base model on a general-domain reference corpus (e.g. FineWeb), then aggregates the per-token logit differences (Top-K frequency, or NMF for multiple topics) into a short token list th

*Full summary: [arxiv_2608.26462.md](summaries/arxiv_2608.26462.md)*

---

## 80. Feature Geometry of LoRA Adapters: A Sparse Autoencoder Analysis of Representational Divergence in Fine-Tuned Language Models

**Relevance Score:** 3/10
**Source:** arxiv
**Year:** 2026
**Authors:** Prasanth K K
**URL:** http://arxiv.org/abs/2605.28896v1

The paper trains sparse autoencoders (SAEs) on the activation change that a LoRA adapter adds to the residual stream (h_delta = h_adapted - h_base) for Gemma-2-9B at ranks 4 to 32, and compares those features with the pretrained Gemma Scope dictionaries for the base model. It reports that the adapte

*Full summary: [arxiv_2605.28896v1.md](summaries/arxiv_2605.28896v1.md)*

---

## 81. Weight Oracles: Reading Neural Network Weights with Language Models

**Relevance Score:** 3/10
**Source:** google_scholar
**URL:** https://arxiv.org/abs/2610.07334

The paper proposes "Weight Oracles": a fine-tuned LLM (Qwen 2.5 3B) that reads the raw weights of a target network and answers questions about them, without running the target on inputs. Phase I trains the explainer to simulate tiny 1-layer transformers (~2K parameters) by decomposing each forward p

*Full summary: [arxiv_2610.07334.md](summaries/arxiv_2610.07334.md)*

---

## 82. Simple LLM Baselines are Competitive for Model Diffing

**Relevance Score:** 3/10
**Source:** arxiv
**Year:** 2026
**Authors:** Elias Kempf, Simon Schrodi, Bartosz Cywiński et al.
**URL:** http://arxiv.org/abs/2602.10371v1

The paper compares two API-only model diffing methods, an LLM-based pipeline (after Dunlap et al. 2025) and an SAE-based pipeline (after Jiang et al. 2025), that surface natural-language hypotheses about how two models differ in their outputs. It proposes evaluation criteria (held-out generalization

*Full summary: [arxiv_2602.10371v1.md](summaries/arxiv_2602.10371v1.md)*

---

## 83. Efficient Storage of Fine-Tuned Models via Low-Rank Approximation of Weight Residuals

**Relevance Score:** 3/10
**Source:** arxiv
**Year:** 2023
**Authors:** Simo Ryu, Seunghyun Seo, Jaejun Yoo
**URL:** http://arxiv.org/abs/2305.18425v1

The paper proposes Efficient Residual Encoding (ERE), a way to store a fully fine-tuned model as a compressed weight residual Δθ = θ' − θ. The residual is approximated by a low-rank factorisation (U D V^T with Stiefel-constrained U, V) with per-layer rank chosen by a spectral-decay budget allocation

*Full summary: [arxiv_2305.18425v1.md](summaries/arxiv_2305.18425v1.md)*

---

## 84. Decomposing Behavioral Phase Transitions in LLMs: Order Parameters for Emergent Misalignment

**Relevance Score:** 3/10
**Source:** manual
**Year:** 2025
**URL:** https://arxiv.org/abs/2508.20015

The paper proposes a framework for detecting rapid behavioral transitions during fine-tuning and for measuring how much of that change is explained by a chosen set of human-readable "order parameters" (OPs), which are scored by an LLM judge. Applied to rank-1 LoRA fine-tuning of Qwen2.5-14B-Instruct

*Full summary: [arxiv_2508.20015.md](summaries/arxiv_2508.20015.md)*

---

## 85. Beyond Shallow Alignment: How Post-Training Methods Determine Refusal Circuits And Steering Robustness

**Relevance Score:** 3/10
**Source:** arxiv
**Year:** 2026
**Authors:** Hoang Cuong Nguyen, Mark Dras, Usman Naseem
**URL:** http://arxiv.org/abs/2609.03887v1

The paper compares three post-training methods for teaching refusal (SFT, reasoning-augmented SFT "Ra-SFT", and ORPO) on Llama-3.1-8B, Gemma-2-9B and Qwen3-8B, holding data and hyperparameters fixed. It uses difference-in-means refusal directions, activation and attribution patching over layers, att

*Full summary: [arxiv_2609.03887v1.md](summaries/arxiv_2609.03887v1.md)*

---

## 86. Tracing the Representation Geometry of Language Models from Pretraining to Post-training

**Relevance Score:** 3/10
**Source:** arxiv
**Year:** 2025
**Authors:** Melody Zixuan Li, Kumar Krishna Agrawal, Arna Ghosh et al.
**URL:** http://arxiv.org/abs/2509.23024v1

The paper tracks the geometry of last-token hidden representations in LLMs (OLMo-2 1B/7B, Pythia 160M-12B, Tulu-3.1-8B) using two spectral metrics: effective rank (RankMe) and eigenspectrum decay (alpha-ReQ). It reports a non-monotonic, three-phase pretraining sequence ("warmup", "entropy-seeking", 

*Full summary: [arxiv_2509.23024v1.md](summaries/arxiv_2509.23024v1.md)*

---

## 87. Semantic Optimal Transport for Sparse Autoencoder Feature Matching and Circuit Compression

**Relevance Score:** 2/10
**Source:** arxiv
**Year:** 2026
**Authors:** Tue M. Cao, Nguyen Do, My T. Thai
**URL:** http://arxiv.org/abs/2605.28567v1

The paper proposes representing each sparse autoencoder (SAE) feature as an activation-weighted distribution over the hidden states where it fires (top-K tokens), projecting those states into a shared reference space (the target layer, or a concatenation of all layers), and comparing features with a

*Full summary: [arxiv_2605.28567v1.md](summaries/arxiv_2605.28567v1.md)*

---

## 88. Why research personas despite RL scaling?

**Relevance Score:** 2/10
**Source:** lesswrong
**Year:** 2026
**Authors:** Cleo Nardo
**URL:** https://www.lesswrong.com/posts/i4yswYDSrPFHWpCbi/why-research-personas-despite-rl-scaling

A short opinion post arguing that large-scale RL appears to "wash out" the initial personas of LLM agents, which makes the author pessimistic about some motivations for persona research but not all. The post is a catalogue of ten reasons people give for studying personas, each assessed against the w

*Full summary: [lw_i4yswYDSrPFHWpCbi.md](summaries/lw_i4yswYDSrPFHWpCbi.md)*

---

## 89. Mechanistic Analysis of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning

**Relevance Score:** 2/10
**Source:** google_scholar
**URL:** https://arxiv.org/abs/2601.18699

The paper reports a comparative study of catastrophic forgetting under a three-stage sequential fine-tuning pipeline (coding, then math, then medical QA) across 20 named models, split into 10 closed-source API models and 10 open-weight models. For the open-weight models it uses CKA, routing-gate Jen

*Full summary: [arxiv_2601.18699.md](summaries/arxiv_2601.18699.md)*

---

## 90. Stability of Low-Rank Implicit Regularization in Perturbed Deep Matrix Factorization

**Relevance Score:** 2/10
**Source:** arxiv
**Year:** 2026
**Authors:** Jingzhe Wang, Hung-Hsu Chou
**URL:** http://arxiv.org/abs/2605.28613v2

The paper studies gradient descent on deep matrix factorization (a product of N factors fitted to a symmetric target) and asks when the "low-rank phase" of the trajectory, where the iterate's effective rank sits near that of the best rank-L approximation of the target, is observable and survives add

*Full summary: [arxiv_2605.28613v2.md](summaries/arxiv_2605.28613v2.md)*

---

## 91. Run evals on base models too!

**Relevance Score:** 2/10
**Source:** lesswrong
**Year:** 2024
**Authors:** orthonormal
**URL:** https://www.lesswrong.com/posts/dgFC394qZHgj2cWAg/run-evals-on-base-models-too

The author argues that capability evals of frontier models (in this case DeepMind's) are run only on post-RLHF/RLAIF models, so they may understate capabilities that RL-based post-training suppresses rather than removes. Running evals on base models is proposed as a precaution, and the author flags 

*Full summary: [lw_dgFC394qZHgj2cWAg.md](summaries/lw_dgFC394qZHgj2cWAg.md)*

---
