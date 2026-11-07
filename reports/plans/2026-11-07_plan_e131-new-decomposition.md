# E131 plan: Goodfire's second Qwen3-1.7B decomposition (p-9707d6ff): port, check, compare

User task (task file, 2026-11-07): "download it, ensure it works and maybe run a quick check to see how it compares
with the older decomposition, especially around reasoning tasks." In the Terms' words: load the decomposition
p-9707d6ff in our torch port, show the port reproduces the authors' library on the same tokens, then measure, for the
old (p-e9fcffd5) and new decompositions on the same rows, how faithful each is with its own causal importance (CI)
and under the adversary, on generic text and on GSM8K reasoning text.

## Design check
- **Goal served.** Every D1-D3 reading rests on the decomposition being faithful on the update's own text. Our updates
  are GSM8K SFT/RL; if the new decomposition is markedly more faithful on GSM8K text, later readings should move to it.
- **Nearest tried.** E-port (experiments/qwen_port: JAX reference + torch_check, 2026-10-27) and the 128-row adversary
  (experiments/qwen_adversary: CI-masked KL 0.142, adversary KL 1.62 on FineWeb rows). This repeats both on a new
  decomposition and adds GSM8K rows; nothing new about the method.
- **Patch or step.** Neither: an apparatus upgrade the user asked for. Approved by the task file.
- **Decisions relied on.** The port's conventions (src/commute/qwen_vpd.py docstring: Γ bf16, masked forward without
  Δ for the CI read). No departure. No verdict under rows 81/87 is read here: definitions untouched.
- **What would change what we do.** New clearly more faithful on GSM8K rows (and not much worse on text) → propose
  moving D1/D3 readings to it (a card for the user; E111 waits on that). Not more faithful → stay on p-e9fcffd5.
- **Control first.** Stage 0: (a) V U reproduces each HF matrix (rel. Frobenius ≤ 0.0055, the bundle's
  reconstruction_check.csv); (b) torch port vs the authors' JAX library (public param-decomp 7d2e5aee8) on the 8
  rows of experiments/qwen_port/results/tokens.json: share of (site, position, subcomponent) with |g_torch − g_jax| >
  0.1 at most 1e-4 (old port: 2.4e-5); (c) the old decomposition through the same new code reproduces its stored
  CI-masked KL 0.142 on the 128 FineWeb rows within 2%. Any miss stops the job before stage 1.
- **Numbers (per decomposition × row set; mean per token, rows equal-weighted).**
  1. CI-masked KL, KL(target ‖ masked at m = g, no Δ). Old on FineWeb: 0.142 (stored). README for new on held-out
     chat text: about 0.17.
  2. Active count, subcomponents with CI > 0.1 per token. README for new: about 6,200 on chat text.
  3. Adversary KL (config PGDReconLoss, 20 sign steps of 0.1, random source). Old on FineWeb: 1.62.
  Row sets, all ≤ 512 tokens (the old decomposition was trained at 512, the new at 2,048): T = the 128 FineWeb rows of
  experiments/qwen_adversary/results/tokens.json; G = 128 GSM8K test prompts (chat template, the GSM8K baseline's
  instruction) followed by Qwen3-1.7B's own greedy response, thinking off; G-think = the same with thinking on, cut at
  512 tokens.
- **Expected.** README: new trained on chat text with 30% Qwen3-1.7B's own responses (some thinking); old on
  "firestock3" at 512 tokens. So new below old on G and G-think for numbers 1 and 3; on T unclear. Filed in
  results/predictions.json before the run (direction, ratio new/old with a row-bootstrap 95% interval).
- **Blind to.** Whether subcomponents mean anything (no labels in the bundle); faithfulness past 512 tokens.
- **Saved.** Per-row values of all three numbers for both decompositions (small JSON); the converted Γ in
  ~/commute-data/artifacts/qwen3-1.7b-p-9707d6ff-torch.

## Apparatus (code writer)
The diff of param-decomp 177d0cce8 → 7d2e5aee8 against our port:
~/commute-data/scratch/qwen-decomp-lead/new-decomp-diff/findings.md. The bundle's launch_config keeps qk_norm rms and
softcap 30 (lines 177-180) and adds sink: learned and final_norm: rms. Bundle:
~/commute-data/artifacts/p-9707d6ff-share/data/p-9707d6ff-share/ (README.md, compute_ci.py, decomposition.safetensors
with V, U in fp32, param-decomp-run/ Orbax checkpoint with Γ; stacks padded 28 → 32 with zero slots).

## Gates
Not gated: $0 on the cluster's debug L40s, well under 6 h. Cost: $0.

## Not run
Behaviour of the masked model (answers right or wrong on GSM8K); thinking traces past 512 tokens; labels.
