# P2.73: WeightWatch, the SVD of the weight difference, on our items (method spec, revision 3)

*definition-lead, 2026-11-23. Cell P2.73 (open, 0 of 2 runs used, due 2026-11-25). Step 1 of the cell: this spec.
No code was written and nothing was run. Revisions 1 and 2 answer redteam-lead's review, the GPT plan-reviewer's review and its re-check, and the coordinator's ruling on the floor (section
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
sha256 (`a7e2a3ee139f...`), with one storage-only patch. `register` stores each input as
`np.array(current_datapoint)`. That list is ragged whenever layers keep different numbers of directions (x2 keeps 8 in one
layer and none elsewhere), and NumPy 2.4.0, the version pinned in `pyproject.toml`, refuses it. The patch stores the list
itself; `set_threshold` iterates it unchanged. Only the hidden-state capture is replaced: our batched hook stands in for
`latent_extractor.extract_latent`. Our vectorised path must agree exactly with the patched class, both passes included,
on ragged x2 and on the trigger (appendix C). The reference is run per item: one class holds at most 80 directions per
layer, so the stacked directions of the M floor go through our path only. **Detector floors:**
- **Primary: M monitored with F's own directions**, calibrated on M. This one isolates the update from how unusual the
  windows are. It is a separate M pass, run after every item's directions are built, with all of them stacked.
- **Orientation sham:** F_s = M + Q_m dW_m, with Q_m a Haar rotation of the output space. It keeps every singular value
  and right vector of dW and moves only the left directions. The kept count and kept energy of the realised F_s − M,
  after D2, are reported against the item's.

Flag rates are compared only within a task set: an any-token flag has fewer chances on a 21-token window than on a
128-token one.

**Our score, "base-use ratio after Zhong".** Take each kept direction u_{m,i}, with singular value σ_{m,i}. Let C_m =
E[(W_m x)(W_m x)ᵀ] be M's own output second moment at monitored matrix m, over the calibration tokens, with d = 768:

  a_{m,i} = d · u_{m,i}ᵀ C_m u_{m,i} / tr C_m,   S = Σ σ²_{m,i} a_{m,i} / Σ σ²_{m,i}.

**The comparison: the same-spectrum Haar null.** S of Q_m dW_m: the left vectors are rotated at random, and the
singular values and right vectors are kept, 20 filed draws per state. Its spread matches the item's kept count and
weights. S is reported against this null's draws, with no band.
**Gradient-geometry diagnostic, beside it:** a_G,m = d · tr(G_m C_m) / (tr G_m · tr C_m), with G_m = E[g gᵀ] the second
moment of M's LM-loss gradient at matrix m's output (one backward pass on the calibration Pile), and S_G its σ²-pooled
form. It is not a floor for the update class. Spanning g does not make the top left vectors of Σ g xᵀ follow G's
unweighted moment: input norms weight the sum, cross terms enter, Adam rescales, and x2 is a column rescale.
**Check:** the isotropic Gaussian `commute.finetune.random_direction` must read 1 in expectation (arithmetic only).

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
pooling, the matrix that moved most dominates S. The Haar and Gaussian nulls are isotropic in output direction, so S
above them may reflect generic alignment of gradient updates with M's high-variance outputs.

## How it differs from E141's effective rank

E141 (P2.9) reads only the singular values of all 24 matrices, one spread number per state, with no directions and no
data. WeightWatch reads 8 output matrices' directions against hidden states. S reads them against M's outputs.
Loaders reused, not copied: `commute.update_spectrum.load`, and E141's `large`, `seeds` and `PAIRS["e62_copy_lookup"]`
(`experiments/update_svd/run.py`). Built items come through `experiments/eapig_overlap/items.py`. The panel comes
through `convergence_panel/run.py` `load_dW` and `results/wide/selection.json`.

## Order and cost

1. **Base-moment pass**, first: M forward and backward on the calibration windows. It gives C_m, G_m and the shares of tr C.
2. **Stage 0**, alone: the known-range test, then dW = 0, the λ constructions, x2's filed S, τ per build path, and the
   trigger's slot and detector row (u_min is now known from C).
3. **Trigger replay**: one M forward pass over the calibration windows, storing the trigger's per-token scalars (appendix
   C). Then the trigger's gain fit, its F pass and parity check, and every item's directions (weights only).
4. **The M detector-floor pass**, with all directions stacked.
5. **Then the built items, E62 and the panel**, each F and its sham.

Passes: 1 M forward-and-backward, 2 M forward, 50 F and 48 shams, each over 6.4M calibration tokens plus the task rows:
about 640M tokens, with the LM head skipped except in the backward. Stage 0 times one F pass, and the backward pass
separately. It projects the sum per pass type. The SVDs are small: E141 measured about 5.5 s per state for 24 fp64 matrices on 2 CPUs, and ours are 8. The run
is cluster debug L40 jobs that resume by completed state; each exits for a resubmit past 25 min. Cost $0.

## Design check (`/design`)

- **Goal.** Part 2's comparison of cheap readings and baselines against the frozen definition. WeightWatch is a
  published weight-diff baseline that the draft does not cite.
- **Nearest tried.** E141 and E162 (spectrum only).
- **Patch or step.** A step: the cell's baseline.
- **Decisions.** No band or label is added to the definition.
- **What would change what we do.** S earns a column only if all three hold: x2's S sits above both new edits' S
  (new_few, new_thin) by more than the Haar null's spread, the trigger is flagged, and the dW = 0 control holds. If any
  fails, S is reported as a related-work negative. x2 outside its Haar null alone earns nothing: x2 rescales M's own
  columns, so that is near-guaranteed by construction. A trained update's S above its Haar null is never read as
  reuse: it is reported beside S_G, as "at the level the gradient geometry gives" (S ≤ S_G) or "above both nulls"
  (S > S_G), and neither is a reuse label.
- **Control first.** The known-range test, then dW = 0 and the trigger.
- **Saved.** Kept u and σ (float16), ranges, per-window flags, eigenvalues of C_m and G_m.

## Review record

| Point | Done |
|---|---|
| RT1 / GPT P1-2: the floor | Coordinator's ruling (2026-11-23), where the reviews disagreed: GPT's. The comparison is the same-spectrum Haar null. redteam asked for a_G as primary; it is kept beside as a "gradient-geometry diagnostic", not called an update-class floor. Reason: spanning g does not make the top left vectors of Σ g xᵀ follow G's unweighted moment (input-norm weights, cross terms, Adam; x2 is a column rescale), and a_G was not validated on real generic updates. The isotropic Gaussian stays as a check that reads 1. No band either way. Realistic do-nothing fine-tunes do not exist at E62's or the panel's sizes |
| RT2: positive control undefined | A′ᵢ counted (96-187 windows each); new_few/new_thin demoted to behavioural; trigger edit at λ_min added, with expected values |
| RT3: M with F's directions | Primary detector floor |
| RT4: position 0 and mean | Shares reported; S_cov beside |
| RT5: pooling | Per-matrix a and the kept share of ‖dW‖² reported |
| RT6 / GPT P1-3: x2 lean | Replaced by a value computed from M in stage 0 |
| RT7 / GPT P1-1: thresholds | Two passes as the code; D7 relabelled theirs |
| RT8: positions, padding | Each window read whole, BOS as in our rows, no padding (windows equal-length per set); FPR caveat stated |
| RT9: k share, t after D2 | Stated |
| RT10: cost | LM head skipped; stage 0 times one item |
| GPT P1-1: pinned core | `LatentStats` copied verbatim with sha256. Re-check: a storage-only NumPy patch (the list kept, not `np.array` of a ragged list); two-pass parity on ragged x2; reference per item, not the stacked M directions (80 slots per layer) |
| GPT P1-3: stage 0 exercises detector | Known-range test; x2/x10 bar filed. Re-check: the base-moment pass comes first; the M floor is a separate, budgeted M pass after the directions; τ is measured on each build path; the trigger gain is fitted on 200 build windows against the installed ranges, and the 1,000 test windows are kept for the stop (appendix C) |
| GPT re-check P1-2: sham ranks | The realised kept count and energy of F_s − M are reported |
| GPT re-check P2: timing | Timed per pass type: F, sham, M forward, M backward |
| GPT re-check minors | Qwen settings removed from the appendix; Remark 3.1's bound called a reference |
| GPT P1-4: Qwen | Dropped: revision, tokenizer, generic rows, d per matrix and timing not pinned now (Not run) |
| GPT P2-5: manifest | Appendix B, 101 passes (50 F, 48 shams, 3 M); within-set comparisons only |
| GPT re-check 2, P1: trigger scalars | They need u_min (known only once C is complete) and the detector row (built in stage 0), so the base-moment pass cannot store them. A trigger replay, one extra M forward pass over the calibration windows, runs after both are fixed |
| GPT P2-6: definition | Frozen 2026-11-22; S is a proxy reading |
| GPT: final-layer hook | In D1 |
| RT on revision 3 (2026-11-23, no objection, two conditions) | Both applied, in `results/predictions.json` (`column_rule`, `trained_reading`) and here: the column rule contrasts x2 with both new edits by more than the Haar spread, and needs the trigger and dW = 0 (design check); a trained S above its Haar null is reported beside S_G, never as reuse; the isotropic-null sentence is in "blind to" |
| RT note: "no built edit at layer 0" | The handback was wrong: new_few/new_thin are at layer 0; D1 now names x2, x10, rebuilt |

## Not run

- Qwen's 19 saved updates (P1-4); a later cell can pin them.
- Steering (the paper's section 5.3).
- Generation during calibration (D4).
- Length-matched calibration.
