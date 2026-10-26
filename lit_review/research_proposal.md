# Research focus: commuting post-training updates past a parameter decomposition

*2026-10-15. Built from `GOALS.md`, `README.md` and the theory notes. The user asked for a broad sweep: the eventual
application is open (continual interpretability, understanding what post-training does, or something else).*

## Core question
Given a parameter decomposition of a model's weights (VPD / SPD / APD: Goodfire's rank-1 subcomponents U_c V_c^T plus a
learned causal-importance function Γ), and a small update δW from post-training (a LoRA or a full fine-tune), is there
a simple map Φ(D, δW) giving a decomposition of W + δW that is still faithful, with the update carried by the
components, not the residual? Stronger: is Φ(D, δW) close to re-decomposing W + δW from scratch? Candidate maps
include projecting δW onto the subcomponents and pulling the gradient step back through W = Σ U_c V_c^T.
If no simple Φ works, is that because post-training *creates* new mechanisms rather than *strengthening* existing
ones, or because parameter decompositions are too brittle (unstable or non-identifiable)?

## Strands to sweep
1. Parameter decomposition methods: APD, SPD, VPD, weight-based interpretability, their stability, seed dependence and
   identifiability; component matching across runs.
2. Interpretability that tracks training: SAE/crosscoder/transcoder model diffing (base vs chat), feature transfer
   from base to fine-tuned models, SAE warm-starting or fine-tuning SAEs along checkpoints, training dynamics of
   features, continual/online interpretability.
3. Structure of fine-tuning updates: low intrinsic dimension, LoRA geometry, the spectrum and alignment of δW with
   pretrained singular vectors, sparse fine-tuning, task arithmetic and weight disentanglement, linearised/NTK
   fine-tuning, the lottery-ticket view of fine-tuning.
4. Mechanistic accounts of fine-tuning: does it reuse existing circuits (entity tracking, refusal direction, "wrapper"
   on capabilities) or build new ones; superficial alignment hypothesis; RL sparsity of updates; emergent
   misalignment via rank-1 LoRA and pre-existing persona features.
5. Maths of the "commuting" step: reparametrisation and factorised gradient descent (gradient pulled back through
   U V^T), Riemannian/low-rank manifold optimisation, perturbation theory of decompositions (matrix and tensor), warm
   started re-fitting.
6. Mechanism stability: circuit persistence across training and scale, universality, model stitching.

## Known works
VPD (Bushnaq, Braun, Clive-Griffin et al. 2026); SPD (Bushnaq et al. 2025); APD (Braun et al. 2025); Minder, Dumas et
al. 2025 (crosscoder artifacts); Kissane et al. (SAEs transfer base→chat); Prakash et al. 2024; Jain et al. 2023;
Aghajanyan et al. 2020 (intrinsic dimension); Soligo/Turner rank-1 EM; Mukherjee et al. (RL updates small subnetworks);
Ortiz-Jimenez et al. (task arithmetic in tangent space).
