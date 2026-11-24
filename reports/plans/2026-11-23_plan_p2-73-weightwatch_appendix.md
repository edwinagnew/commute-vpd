# P2.73 appendix: step table with quotes, and the inputs on disk

*Companion to `2026-11-23_plan_p2-73-weightwatch.md`. "Z" is Zhong and Raghunathan, arXiv 2508.00161v3; page numbers
are the PDF's. "WW" is the released code at `8d3874fb90a0`, cloned read only in `artifacts/p273_src/WeightWatch/`.
D-numbers point to the main file's departures table.*

## A. Step table

| Step | Published, verbatim, with location | Ours |
|---|---|---|
| Linear layers | Z §3 p.4: "We focus on linear layers W that operate directly on activations a to generate outputs of the form W a." Eq. 1: "∆W = Wpost − Wbase" | The same, F's weights minus M's |
| Which matrices | Z §3 p.5: "we apply SVD to each layer's output projection matrix of the attention block (Oproj) and the down-projection matrix of the MLP block (Wdown), as these are the linear layers that directly influence activations." WW `worker.py` `_extract_svd_directions`: `layer.mlp.down_proj.weight`, `layer.self_attn.o_proj.weight` | `h.{l}.attn.o_proj` (768×768) and `h.{l}.mlp.down_proj` (768×3072); Qwen `layers.{l}.self_attn.o_proj`, `layers.{l}.mlp.down_proj` |
| Left vectors | Z §3 p.5: "The left singular vectors, {ui}, form an orthonormal basis for the output space of the weight matrix." WW: `u, s, v = torch.linalg.svd(w_diff.to(dtype=torch.float32, device=device), full_matrices=False)`; `list(u.T.cpu())[:rank_limit]` | The same call, fp32; only σ above the floor kept (D2) |
| k | Z §3 p.5: "For our main experiments, we empirically set k = 20 for each matrix, resulting in 40 directions per layer." WW: `rank_limit: int = 20` | k = 20 |
| Where read | Z Alg. 1 p.19: "Hook on activations of all tokens after layer l". WW: layer l's vectors stored at index `l + 1` (`record_delta=1`) and read from `output_hidden_states` index `l + 1` | Residual stream after block l, every token, F's forward |
| Which model | Z Alg. 1 p.19: "Run hooked inference on Mpost" | F |
| Cosine | Z §3 p.5: "We track the minimum and maximum cosine similarities observed between per-token latent activations and the direction vector." WW `utils.py` `LatentStats.register`: `li = li / li.norm(dim=-1, keepdim=True)`; `sims = ux @ li.T`; max and min over tokens | The same, per input, per direction |
| Ranges and flag | Z Alg. 1 p.19: "if smin < cmin − ϵ or smax > cmax + ϵ then Flag this input as anomalous", ϵ = 0.01, "accounts for numerical errors". WW `utils.py`: `EPS = 0.01` | The same |
| Range updates | Z Alg. 1 p.19: "if monitor mode: R[u] ← (min(smin, cmin), max(smax, cmax))". WW `marked_inference(..., read_only=True, ...)` at test; `read_only=False` in `do_calibrate` | Calibration widens, test does not (D7) |
| Calibration data | Z App. C.1 p.18: "we used a mix of 50000 samples where 2/3 is from WildChat and 1/3 is from UltraChat." Z §5.1 p.7: "calibrating on a mix of the WildChat dataset ... and the UltraChat dataset ... to obtain normal cosine similarity ranges" | 50,000 Pile windows of 128 tokens (D3) |
| Generation | Z App. B p.18: "for most models we generated 50 tokens of completion with temperature 1 during calibration"; "for models that underwent representation-based unlearning (RMU and circuit breaker), we did not run completion generation during calibration" | None (D4) |
| Roles | Z App. B p.18: "We calculate separate cosine similarity ranges for user and assistant tokens." | One set (D5) |
| Layers left out | Z App. B p.18: "We also exclude the last three layers from anomaly detection, as it helps with numerical stability." WW README: `model.drop_bottom_layers(3)`, which slices `self.directions[:-num_layers]` (the top layers) | 67M: all four, layer-0-only beside (D1); Qwen: as published |
| Precision | Z App. B p.18: "All experiments were conducted using fp16 precision during inference and latent collection for computational efficiency." WW: `load_model_and_tokenizer(..., dtype=torch.float16)`; directions cast `dtype=torch.float16` | 67M fp32 (D6); Qwen bf16 |
| FPR | Z App. C.1 p.18: "The FPR evaluations are done with 9 : 1 time-respecting train-validation splits." WW `set_threshold`: `val_split = int(len(self.datapoints) * 0.1)`, ranges from `[:-val_split]` | First 45,000 windows calibrate, last 5,000 give the FPR |
| FPR bound | Z Remark 3.1 p.5: "Suppose a total of t directions are monitored, the false positive rate after n − 1 samples collected from Dgen is bounded by 2t/n + O(1/n²)" | An expected value, not a stop |
| Threshold trim | WW `adjust_threshold(threshold_percentile=0.0)`; trimming is used in Z only for steering (App. D) and the no-subtraction ablation (Table 15) | No trim |
| Output | Z Table 2 p.8: "% Prompts Flagged", "% Completions Flagged", "FPR on Calibration", "FPR on MMLU" | Share of task windows flagged; FPR on held-out Pile |
| Claim | Z abstract: "the top singular vectors of the weight difference between a fine-tuned model and its base model correspond to newly acquired behaviors." | Not a reuse score; ours is S (D8) |

Noticed in the paper: Algorithm 1's two SVD lines swap labels (`SVD(∆O_down)`, `SVD(∆W_proj)`); the code is unambiguous.
In HF models `hidden_states[L]` is after the final norm, but the published rule leaves that layer out; our 67M hook reads
the residual before the final norm at every layer.

## B. Inputs, checked with `ls -laL` on 2026-11-23

| Input | Path | Size or check |
|---|---|---|
| M | `artifacts/vpd/t-9d2b8f02/ckpt/model_step_99999.pt` | 267,736,421 B |
| Calibration rows | `artifacts/sets/train_65536_524288.npy` | 2,147,483,776 B (524,288 × 512 int64) |
| x2, x10, rebuilt, new_few, new_thin, randoms | built by `experiments/eapig_overlap/items.py` (`Items.make`); x2 settings `experiments/kstar_levels/results/predictions.json` (`e195_layer` 1, `e195_C`), edits record committed `experiments/kstar_levels/results/edits.json` (slots at layer 0; the tree holds another session's uncommitted change: read HEAD); rebuilt payload `experiments/rule_retention_fixed/results/large/rebuilt_mlp.pt` | payload 37,751,237 B, sha256 `55cb8e62aeef...` (P2.69 appendix B) |
| Built task windows | copy24 `experiments/open_pair/results/large/amend1/windows_copy_trained.pt` (536,908 B); cue windows built by E214's World | as P2.69 |
| E62 | `experiments/induction_cue/results/large/{copy,lookup}_s{0,1,2}.pt`, `windows.pt` | 113,253,467 / 113,254,103 / 370,492 B; sha256 in `experiments/induction_cue/results/run.json` |
| Panel | `experiments/convergence_panel/results/wide/selection.json`, `kl.hi`: 12 tasks × seeds 0-2, every file present, sha256 per state in that file; `dropped` empty | 36 states of about 56.6 MB (bf16) |
| Qwen | the 19 files of `experiments/qwen_cheap_readings/results/run1_stage2.json`, all present: 14 of 5,637,164,512 B, 5 of 2,818,592,344 B | about 93 GB |
| E141 loaders | `src/commute/update_spectrum.py` `load`; `experiments/update_svd/run.py` `large`, `seeds`, `PAIRS` | read, not copied |
