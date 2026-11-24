# P2.73: WeightWatch, the SVD of the weight difference, on our items (method spec, revision 1)

*definition-lead, 2026-11-23. Cell P2.73 (open, 0 of 2 runs used, due 2026-11-25). Step 1 of the cell: this spec.
No code was written and nothing was run. Revision 1 answers redteam-lead's and the GPT plan-reviewer's reviews (section
"Review record"). Every step is the published one, quoted with its location in the appendix
(`2026-11-23_plan_p2-73-weightwatch_appendix.md`, section A), or a named departure below.*

**Plain words.** Zhong and Raghunathan take the difference between a fine-tuned model's weights and its base model's,
keep the top 20 output directions of that difference in two matrices per layer, and flag any input whose hidden state
points along one of them more or less than ordinary text ever did. That is a detector of fine-tuned behaviour, and it
gives no reuse score. We run it as published on our built edits, E62's copy and lookup fine-tunes and the realistic
panel, and beside it one score of our own: do the update's top directions lie where the original model already writes?

## Source

| Source | Identity | Pinned at |
|---|---|---|
| Paper | Ziqian Zhong, Aditi Raghunathan, "Watch the Weights: Unsupervised monitoring and control of fine-tuned LLMs", ICLR 2026, arXiv 2508.00161v3 (21 Apr 2026). Text kept for quote checks: `artifacts/p273_src/paper/paper.{pdf,txt}` | v3 PDF; page numbers are the PDF's |
| Code | github.com/fjzzq2002/WeightWatch; clone `artifacts/p273_src/WeightWatch/` (read only, never run) | `8d3874fb90a0aad5478514f96d71779d2aedf5aa` (2025-09-25) |

## The published procedure, in five lines (quotes in appendix A)

1. **Matrices and k.** Per layer, dW = W_post − W_base of `o_proj` and `down_proj`; fp32 SVD; the top k = 20 left
   singular vectors of each, 40 per layer. Singular values only rank them.
2. **Where read.** Against the fine-tuned model's residual stream after that layer (`hidden_states[l + 1]`), every
   token, fp16.
3. **Statistic.** Per token cos(a_t, u); per input the minimum and maximum over tokens, per direction.
4. **Calibration.** 50,000 generic inputs (WildChat and UltraChat) with 50 generated tokens, user and assistant ranges
   apart, the last three layers left out. The code's `set_threshold` runs two passes: ranges from the first 90% score
   the FPR on the last 10%; then ranges from all inputs but the last are installed for use.
5. **Flag rule.** Flagged if, for any direction, s_min < c_min − 0.01 or s_max > c_max + 0.01. The output is the share
   of anomalous inputs flagged and the FPR. **It is a detector, not a reuse score.**

## What we run

**Detector, as published.** For each item, F = M + dW, calibrated on 50,000 Pile windows on F. The rates are the share
of the item's own task windows flagged, read against the installed ranges, and the FPR from pass 1. The detector core is
the code's `LatentStats.register` and `set_threshold` (`utils.py`), copied verbatim into our run with the source file's
sha256 (`a7e2a3ee139f...`). Only the hidden-state capture is replaced: our batched hook stands in for
`latent_extractor.extract_latent`. Stage 0 requires our vectorised path to agree exactly with the copied class on
identical tensors. **Detector floors:**
- **Primary: M monitored with F's own directions**, calibrated on M. This one isolates the update from how unusual the
  windows are. A single M pass covers every item, with all items' directions stacked.
- **Orientation sham:** F_s = M + Q_m dW_m, with Q_m a Haar rotation of the output space. It keeps every singular value
  and right vector of dW and moves only the left directions.

Flag rates are compared only within a task set: an any-token flag has fewer chances on a 21-token window than on a
128-token one.

**Our score, "base-use ratio after Zhong".** Take each kept direction u_{m,i}, with singular value σ_{m,i}. Let C_m =
E[(W_m x)(W_m x)ᵀ] be M's own output second moment at monitored matrix m, over the calibration tokens, with d = 768:

  a_{m,i} = d · u_{m,i}ᵀ C_m u_{m,i} / tr C_m,   S = Σ σ²_{m,i} a_{m,i} / Σ σ²_{m,i}.

**Primary floor, matched to the update class:** a_G,m = d · tr(G_m C_m) / (tr G_m · tr C_m). Here G_m = E[g gᵀ] is the
second moment of M's LM-loss gradient at matrix m's output, from one backward pass on the calibration Pile. A gradient
update's left vectors lie in the span of such g, so a_G is what a generic gradient update reads. Its σ²-pooled S_G uses
the item's own σ² per matrix. **Named checks:**
- the orientation sham's S (same spectrum, random left directions; its spread suits the item's kept count);
- the isotropic Gaussian `commute.finetune.random_direction`, which must read 1 in expectation (arithmetic only).

Reported beside S: per-matrix a, the share of ‖dW‖² held by the kept vectors, the range d·λ_min/tr C to d·λ_max/tr C,
and S_cov, which uses the covariance with position 0 left out. The shares of tr C from position 0 and from μμᵀ are
reported too. S is a cheap proxy reading. It is read against the definition frozen on 2026-11-22 (`GOALS.md`,
`theory/2026-11-22_definition_frozen.md`), with no band of its own and no label.

## Departures

| # | Published | Ours | Reason |
|---|---|---|---|
| D1 | Last three layers left out; in HF models the final layer's hidden state is after the final norm | All four 67M layers, read before the final norm at every layer; the published rule (layer 0 only) beside | On 4 layers the rule keeps layer 0 only and misses x2, x10 and rebuilt (layers 1-3) |
| D2 | All 20 vectors kept, whatever σ | σ above a floor τ measured on known-rank builds (appendix C) | A vector with σ = 0 is an arbitrary completion; x2 has 8 non-zero σ |
| D3 | WildChat and UltraChat; shuffled stream | 50,000 Pile windows of 128 tokens from `train_65536_524288.npy`, row order; the split is row-ordered | A Pile base model, no chat; our task windows are short (E62's are 21 tokens) |
| D4 | 50 generated tokens at temperature 1 | None | Fixed windows; the paper skipped it for RMU and circuit breaker |
| D5 | User and assistant ranges apart | One set | No roles |
| D6 | fp16 | fp32, TF32 off | fp16 was for efficiency |
| D7 | Algorithm 1 widens ranges on every monitored input | Theirs (code): installed ranges fixed at test (`read_only=True`); departs from Algorithm 1's monitor mode | Followed as the code does |
| D8 | (none) | S, its floors and the detector floors | Ours, "after Zhong" |

k = 20 is theirs, kept. On the 67M, 40 vectors are 5% of the 768-wide residual, against 1% at 4,096. The FPR on
128-token windows overstates the FPR for shorter task windows. Remark 3.1's 2t/n, with each item's own t after D2,
is a reference, not an expected ceiling.

## Arms and expected values (manifest in appendix B)

| Arm | Detector | S | Status |
|---|---|---|---|
| dW = 0 (F = M), first | no directions, no flags | undefined | empty check |
| Known-range test on synthetic tensors | flags exactly the planted exceedances, both tails; a value at c_max + 0.01 is not flagged | n/a | apparatus, stops |
| Trigger edit (appendix C) | ≥ 0.95 of trigger windows flagged; no-trigger windows at the FPR | exactly d·λ_min/tr C | positive control, stops |
| λ_max construction (weights only) | n/a | exactly d·λ_max/tr C | stops |
| x2, x10 | descriptive; x10 ≥ x2 a lean | equal within the filed bar; value computed from M in stage 0 and filed before items | known answer |
| rebuilt, random_size, random_kl | descriptive | descriptive | |
| new_few, new_thin | near the M-with-F's-directions floor: every A′ᵢ occurs in 96-187 of the 50,000 calibration windows, so calibrating on F takes in the slots' firing | descriptive | behavioural, not a control |
| E62 copy vs lookup, 3 seeds | descriptive | E141's pair rule | |
| Panel, 12 tasks × 3 seeds (`Task.held` rows) | descriptive | beside E_b | |

## What it is blind to

The detector sees only inputs that move a hidden state outside the calibration range. Anything F already does on Pile,
such as the cue slots, is absorbed. Both readings see only the output directions of `o_proj` and `down_proj`, nothing of
q, k, v, `c_fc` or the input side, and only the top 20. S reads low when an update strengthens a direction M rarely
writes along. It reads high when a new mechanism writes along a busy one, such as an unembedding direction. Under σ²
pooling, the matrix that moved most dominates S.

## How it differs from E141's effective rank

E141 (P2.9) reads only the singular values of all 24 matrices, one spread number per state, with no directions and no
data. WeightWatch reads 8 output matrices' directions against hidden states. S reads them against M's outputs.
Loaders reused, not copied: `commute.update_spectrum.load`, and E141's `large`, `seeds` and `PAIRS["e62_copy_lookup"]`
(`experiments/update_svd/run.py`). Built items come through `experiments/eapig_overlap/items.py`. The panel comes
through `convergence_panel/run.py` `load_dW` and `results/wide/selection.json`.

## Order and cost

1. Stage 0 runs alone: the known-range test, the parity check, dW = 0, τ, the λ checks, x2's filed S, the trigger build,
   and the shares of tr C. It times one full item and projects that time over the manifest.
2. Then the M pass for the shared floor and G_m.
3. Then the built items, E62 and the panel.

The manifest has 99 forward items of 6.4M tokens each, about 630M tokens, with the LM head skipped (only G_m needs the
loss). The SVDs are small: E141 measured about 5.5 s per state for 24 fp64 matrices on 2 CPUs, and ours are 8. The run
is cluster debug L40 jobs that resume by completed state; each exits for a resubmit past 25 min. Cost $0.

## Design check (`/design`)

- **Goal.** Part 2's comparison of cheap readings and baselines against the frozen definition. WeightWatch is a
  published weight-diff baseline that the draft does not cite.
- **Nearest tried.** E141 and E162 (spectrum only).
- **Patch or step.** A step: the cell's baseline.
- **Decisions.** No band or label is added to the definition.
- **What would change what we do.** S apart from S_G on x2, and the trigger flagged, earn it a column. Flat values make
  it a related-work negative.
- **Control first.** The known-range test, then dW = 0 and the trigger.
- **Saved.** Kept u and σ (float16), ranges, per-window flags, eigenvalues of C_m and G_m.

## Review record

| Point | Done |
|---|---|
| RT1 / GPT P1-2: class-matched floor | a_G primary; orientation sham and isotropic Gaussian as named checks; λ range beside. The realistic do-nothing fine-tunes were not used: none exist at E62's or the panel's sizes |
| RT2: positive control undefined | A′ᵢ counted (96-187 windows each); new_few/new_thin demoted to behavioural; trigger edit at λ_min added, with expected values |
| RT3: M with F's directions | Primary detector floor |
| RT4: position 0 and mean | Shares reported; S_cov beside |
| RT5: pooling | Per-matrix a and the kept share of ‖dW‖² reported |
| RT6 / GPT P1-3: x2 lean | Replaced by a value computed from M in stage 0 |
| RT7 / GPT P1-1: thresholds | Two passes as the code; D7 relabelled theirs |
| RT8: positions, padding | Each window read whole, BOS as in our rows, no padding (windows equal-length per set); FPR caveat stated |
| RT9: k share, t after D2 | Stated |
| RT10: cost | LM head skipped; stage 0 times one item |
| GPT P1-1: pinned core | `LatentStats` copied verbatim with sha256; exact parity required, margins included |
| GPT P1-3: stage 0 exercises detector | Known-range test; τ from known-rank builds; x2/x10 bar filed (appendix C) |
| GPT P1-4: Qwen | Dropped: revision, tokenizer, generic rows, d per matrix and timing not pinned now (Not run) |
| GPT P2-5: manifest | Appendix B, 99 forward items; within-set comparisons only |
| GPT P2-6: definition | Frozen 2026-11-22; S is a proxy reading |
| GPT: final-layer hook | In D1 |
| RT note: "no built edit at layer 0" | The handback was wrong: new_few/new_thin are at layer 0; D1 now names x2, x10, rebuilt |

## Not run

- Qwen's 19 saved updates (P1-4); a later cell can pin them.
- Steering (the paper's section 5.3).
- Generation during calibration (D4).
- Length-matched calibration.
