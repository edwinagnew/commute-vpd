# P2.73 appendix: step table with quotes, manifest, stage 0, inputs (revision 1)

*Companion to `2026-11-23_plan_p2-73-weightwatch.md`. "Z" is Zhong and Raghunathan, arXiv 2508.00161v3; page numbers
are the PDF's. "WW" is the released code at `8d3874fb90a0`, cloned read only in `artifacts/p273_src/WeightWatch/`. The paper text the quotes were checked against is `artifacts/p273_src/paper/paper.txt` (from the v3 PDF).
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
| Range updates | Z Alg. 1 p.19: "if monitor mode: R[u] ← (min(smin, cmin), max(smax, cmax))". WW `marked_inference(..., read_only=True, ...)` at test; `read_only=False` in `do_calibrate` | Calibration widens, test does not: theirs (code), departs from Alg. 1 monitor mode (D7) |
| Calibration data | Z App. C.1 p.18: "we used a mix of 50000 samples where 2/3 is from WildChat and 1/3 is from UltraChat." Z §5.1 p.7: "calibrating on a mix of the WildChat dataset ... and the UltraChat dataset ... to obtain normal cosine similarity ranges" | 50,000 Pile windows of 128 tokens (D3) |
| Generation | Z App. B p.18: "for most models we generated 50 tokens of completion with temperature 1 during calibration"; "for models that underwent representation-based unlearning (RMU and circuit breaker), we did not run completion generation during calibration" | None (D4) |
| Roles | Z App. B p.18: "We calculate separate cosine similarity ranges for user and assistant tokens." | One set (D5) |
| Layers left out | Z App. B p.18: "We also exclude the last three layers from anomaly detection, as it helps with numerical stability." WW README: `model.drop_bottom_layers(3)`, which slices `self.directions[:-num_layers]` (the top layers) | 67M: all four, layer-0-only beside (D1); Qwen: as published |
| Precision | Z App. B p.18: "All experiments were conducted using fp16 precision during inference and latent collection for computational efficiency." WW: `load_model_and_tokenizer(..., dtype=torch.float16)`; directions cast `dtype=torch.float16` | 67M fp32 (D6); Qwen bf16 |
| FPR and installed ranges | Z App. C.1 p.18: "The FPR evaluations are done with 9 : 1 time-respecting train-validation splits." WW `utils.py` `set_threshold` (lines 259-343): `for val in [1,0]`; pass 1 `val_split = int(len(self.datapoints) * 0.1)`, ranges from `[:-val_split]`, FPR on `[-val_split:]`; pass 2 `val_split = 1`, ranges from all but the last datapoint, kept. WW `data.py`: the stream is shuffled in chunks (`random.seed(42)`, `random.shuffle(buffer)`) | Both passes as the code: FPR from pass 1 (first 45,000 against last 5,000), task windows against pass 2's ranges. Our split is row-ordered (D3) |
| FPR bound | Z Remark 3.1 p.5: "Suppose a total of t directions are monitored, the false positive rate after n − 1 samples collected from Dgen is bounded by 2t/n + O(1/n²)" | An expected value, not a stop |
| Threshold trim | WW `adjust_threshold(threshold_percentile=0.0)`; trimming is used in Z only for steering (App. D) and the no-subtraction ablation (Table 15) | No trim |
| Output | Z Table 2 p.8: "% Prompts Flagged", "% Completions Flagged", "FPR on Calibration", "FPR on MMLU" | Share of task windows flagged; FPR on held-out Pile |
| Claim | Z abstract: "the top singular vectors of the weight difference between a fine-tuned model and its base model correspond to newly acquired behaviors." | Not a reuse score; ours is S (D8) |

Noticed in the paper: Algorithm 1's two SVD lines swap labels (`SVD(∆O_down)`, `SVD(∆W_proj)`); the code is unambiguous.
In HF models `hidden_states[L]` is after the final norm, but the published rule leaves that layer out; our 67M hook reads
the residual before the final norm at every layer.

## B. Manifest

Every arm, its F, its task rows and the forward passes it needs. "Sham" is the orientation sham F_s = M + Q_m dW_m
(Haar Q_m per matrix, filed seed). The M floor (M monitored with F's directions, calibrated on M) and C_m, G_m come
from one shared M pass. S's floors (S_G, sham S, isotropic Gaussian) need no forward pass beyond it.

| Arm | F | Task rows | Forward passes |
|---|---|---|---|
| Known-range test | none: synthetic tensors | planted | 0 |
| dW = 0 | M | Pile test windows | 0 (the shared M pass) |
| M floor, C_m, G_m | M (backward for G_m only) | the 50,000 calibration windows, every item's task rows | 1 |
| Trigger | M + trigger edit (section C) | 1,000 Pile test windows with T inserted, and the same without | F, sham |
| x2, x10 | `Items.make("x2")`, `("x10")` | copy24 held-out, cut after `half` (`Items.task("copy24")`) | F, sham each |
| rebuilt | `Items.make("rebuilt")` | copy24 as above | F, sham |
| new_few, new_thin | `Items.make(...)` | cue windows, A′ᵢ last (`Items.task("cue")`) | F, sham each |
| random_size_241001, random_kl_241002 | `Items.make(...)` | copy24; cue | F each (they are randoms: no sham) |
| E62 copy_s0-2, lookup_s0-2 | `Items.make("e62/<state>")` (the saved dW through `old_tasks.load_dw`, `with_delta`) | E62's read windows cut to columns 0-20 (`Items.task("e62_copy")`, `("e62_lookup")`) | F, sham each |
| Panel, 12 tasks × seeds 0-2 | M + `load_dW(file)` for each `kl.hi` pick of `results/wide/selection.json` | `TK.build(net, C, task).held`, whole rows as built (prompt tasks include the answer and the eos padding that `prompt_rows` writes) | F, sham each |

Count: 50 F passes, 48 shams and 1 M pass, 99 in all at 6.4M calibration tokens each. Task rows add little. A state
is written when done, so a resubmit skips it. Within-set comparisons only: an item against its own floors on the same
rows.

## C. Stage 0 constructions and bars

1. **Known-range test.** Synthetic hidden states, shape (T, L + 1, d), with directions e_1, e_2. The planted
   calibration values set known ranges. The test values are: inside the range; exactly c_max + 0.01 (not flagged: the
   code's `trigger` keeps `min - EPS <= x <= max + EPS`); c_max + 0.0101 (flagged); c_min − 0.0101 (flagged). Our
   vectorised path and the copied `LatentStats` must give identical flags and identical installed ranges, on the same
   dtype. Then the same exact check runs on 500 real windows of x2 and of the trigger. Any difference stops.
2. **σ floor τ (D2).** τ is the largest of three measured values: σ_1 of a do-nothing build (`rescaled` at gain 1, F −
   M), and σ_9 of x2's and of x10's realised F − M (known rank 8), all in fp32. Saved states (E62, panel) are SVD'd from
   the saved dW, as E141 reads them. Built items are SVD'd from realised F − M, because no delta is saved. Kept counts
   and kept energy are reported per matrix. x2 must keep exactly 8 vectors, and every full fine-tune is expected to keep
   20.
3. **x2's filed S and the x2/x10 bar.** S of the exact dW = W_{1,down}[:, C] in fp64 is computed with M's C, written to
   `stage0.json` before any item runs. The bar is the relative gap between that and S of x2's realised F − M. x2 and
   x10 must each sit within that bar of the filed value.
4. **Trigger edit (detector positive control).**
   - T is the lowest token id ≥ 256 that occurs in none of the 50,000 calibration windows and none of the test windows'
     base windows. 998 ids are absent from the calibration windows, counted 2026-11-23 with a cpu-run script on
     `train_65536_524288.npy`, rows 0-12,499.
   - The slot is the layer-1 neuron with the smallest mean activation² × |column|² on Pile held-out (`known_edits.slots`,
     E206's rule), outside x2's 8 columns. Its `c_fc` row is `known_edits.detector`'s rule for T, built on windows with
     T inserted.
   - Its `down_proj` column gains γ u_min, where u_min is the λ_min eigenvector of layer-1 `down_proj`'s C. So dW at
     the monitored matrix is γ u_min e_jᵀ, rank 1, with u = u_min exactly, and S = d·λ_min/tr C exactly.
   - γ is the smallest power of 2 at which the smallest T-position cosine along u_min, over 200 build windows, is at
     least 0.05 above F's largest s_max along u_min over 5,000 Pile windows outside the calibration rows (rows
     12,500-13,749).
   - Test rows: 1,000 windows cut from `val_0_2000.npy` with T inserted at a drawn position ≥ 1 (filed seed), and the
     same windows unchanged.
   - Expected: at least 0.95 of the T windows flagged, and the unchanged ones at the FPR. A share under 0.95, or no γ up
     to 2^20, stops the job.
5. **λ_max construction.** dW = u_max v₁ᵀ at layer-1 `down_proj`: S must read d·λ_max/tr C within the fp64 floor.
   Together with the trigger, this brackets S's range.
6. **Shares of tr C.** Per matrix, the share from position-0 tokens and ‖μ‖²/tr C. S_cov (covariance, position 0 left
   out) is reported beside S for every arm.
7. **Timing.** One full item (F calibration and task reads, plus its sham), times 99, is printed. Each job exits for a
   resubmit past 25 min.

## D. Inputs, checked with `ls -laL` on 2026-11-23

| Input | Path | Size or check |
|---|---|---|
| M | `artifacts/vpd/t-9d2b8f02/ckpt/model_step_99999.pt` | 267,736,421 B |
| Calibration rows | `artifacts/sets/train_65536_524288.npy`, rows 0-12,499 (calibration), 12,500-13,749 (trigger build) | 2,147,483,776 B (524,288 × 512 int64) |
| Trigger test rows | `artifacts/sets/val_0_2000.npy` | 8,192,128 B (2,000 × 512 int64) |
| x2, x10, rebuilt, new_few, new_thin, randoms | built by `experiments/eapig_overlap/items.py` (`Items.make`); x2 settings `experiments/kstar_levels/results/predictions.json` (`e195_layer` 1, `e195_C`), edits record committed `experiments/kstar_levels/results/edits.json` (slots at layer 0; the tree holds another session's uncommitted change: read HEAD); rebuilt payload `experiments/rule_retention_fixed/results/large/rebuilt_mlp.pt` | payload 37,751,237 B, sha256 `55cb8e62aeef...` (P2.69 appendix B) |
| Built task windows | copy24 `experiments/open_pair/results/large/amend1/windows_copy_trained.pt` (536,908 B); cue windows built by E214's World | as P2.69 |
| E62 | `experiments/induction_cue/results/large/{copy,lookup}_s{0,1,2}.pt`, `windows.pt` | 113,253,467 / 113,254,103 / 370,492 B; sha256 in `experiments/induction_cue/results/run.json` |
| Panel | `experiments/convergence_panel/results/wide/selection.json`, `kl.hi`: 12 tasks × seeds 0-2, every file present, sha256 per state in that file; `dropped` empty | 36 states of about 56.6 MB (bf16) |
| E141 loaders | `src/commute/update_spectrum.py` `load`; `experiments/update_svd/run.py` `large`, `seeds`, `PAIRS` | read, not copied |
