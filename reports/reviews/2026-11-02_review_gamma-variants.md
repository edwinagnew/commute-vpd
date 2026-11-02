# Review: the two Γ variants, 67M and Qwen

On the 67M decomposition Γ reads only the model's activations, so moving a subcomponent's read vector cannot change any gate. On Qwen each gate also adds a readout of that subcomponent's own activation, x V. So there a moved read vector changes its own gate directly, with Γ's parameters frozen. This touches every Qwen reading taken with moved subcomponents: E44's splits, and E57's K10 and K11. It will touch teacher training too, though no run has yet trained subcomponents on Qwen. The effect on the one number measured is small. But "Γ frozen" and "original masks" mean different things on the two models, and nobody has measured how much of a Qwen gate is the readout.



## What the measurement found (added 2026-11-02, jobs 49217, 49224, 49275; descriptive, no filed prediction on these numbers)

The readout is a large part of Qwen's gates, but the least-squares split moves V too little for it to matter at the two states read. All controls passed: at W0 a moved V changes nothing, and head plus readout reproduce g exactly.

**At W0, 16 maths rows.** 2.2% of subcomponent positions are on (g above 0). Removing the part of the readout that depends on V turns off 56% of them; the share runs from about 8% (layer 27, the last) to 88% (layer 11) across the 196 sites. Removing the head instead turns off 5%. This is arithmetic, not what Γ would have learned without a term: the two were trained together. On counts any g above 0, however tiny, and the average drop in g on on positions is 0.14 without the V part and 0.24 without the head.

**At a moved V** (the original against the split's V, same activations, fp32 Γ, x from W0):

| state | mean change in g | positions crossing on or off, per position on | same, on the subcomponents carrying 90% of the update |
|---|---|---|---|
| fine-tuned gold, `gold_s0` | 0.000016 | 0.14% | 0.31% |
| Gaussian change | 0.00023 | 2.3% | 4.0% |

With x from the state instead, the same rows read 0.14% and 0.32%, 2.7% and 5.1%. The Gaussian change moves gates about 14 times as much as gold, though gold is the larger on K00 − K0 (1.44 against 0.86); the cause is not worked out. The KL effect on the direct share stays the one in `checks_k10.json`, at most 0.0022, read through bf16 Γ.

**What it means.** E44's and E57's use of the moved V changes little at these states. A comparison of gates across the 67M and Qwen stays confounded, because half of Qwen's on positions depend on V and none of the 67M's do. A frozen Γ on Qwen is not a fixed gate if V moves far: a teacher run that trains V can move gates directly, and these numbers only bound a split's V, not a trained one.

**Blind to.** One row set, maths only; two states; a split's V, not a trained V. The carrying set mixes read and write sides, so a subcomponent moved only on the write side counts and has an unchanged readout.

## The two architectures

| | 67M (`s-55ea3f9b`) | Qwen3-1.7B (`p-e9fcffd5`) | Cite |
|---|---|---|---|
| Γ's input | each site's raw input x, RMS-normed and concatenated | 29 residual taps, plus each site's x V | `param_decomp/models/components.py:338-342`, `component_model.py:591`; `src/commute/qwen_vpd.py:235,249` |
| V in the input | never for a linear site: the wrapper passes x unchanged | yes, h = x_site V | `components.py:691`; `qwen_vpd.py:249` |
| Exception | the wrapper reads V at an embedding site; this release decomposes none | none | `components.py:505-506`, `artifacts/vpd/s-55ea3f9b/final_config.yaml:28-38` |
| Gate formula | head, then leaky sigmoids | clip(head + readout, 0, 1) | `component_model.py:604-620`; `qwen_vpd.py:260-264` |
| Attention | bidirectional, `is_causal=False`, no mask | causal, `tril` mask | `components.py:134-137` (same at `~/param-decomp`); `qwen_vpd.py:218-219` |

A moved V reaches Qwen's gates only one way. Each readout is element-wise, so a moved column changes its own subcomponent's gate at that site and position. The backbone reads taps, not V, so nothing else moves unless the weights change too (`qwen_vpd.py:231-243,251`).

## Which V each call passes

"Original" is the file's V. "Moved" means inside `qwen_maps.applied`, which writes dV into the same `comps` (`src/commute/qwen_maps.py:262`).

| Call site | V | x from | Says so? |
|---|---|---|---|
| `experiments/qwen_port/torch_check.py:192,195,198` | original | W0 | not needed |
| `experiments/qwen_adversary/run.py:112` | original | W0 | not needed |
| `experiments/qwen_sft/run.py:667,792`, through `qwen_teacher.py:270` | original on every stage run | W0, do-nothing | not needed |
| `experiments/qwen_sft/run.py:629`, through `qwen_teacher.py:298` | original at W0 and do-nothing; trained V on the teacher state, which never ran | state | yes, `qwen_teacher.py:24-25` |
| `experiments/qwen_maps/job.py:296`, through `qwen_maps.py:315` (reading A) | **moved** | state | not in the card, which says only "own activations" (`e44.html:13`) |
| `experiments/qwen_maps/run.py:344` (stage A adversary) | moved | state | never run: no `A.json` |
| `experiments/qwen_maps/amend1.py:127,164,176`, `amend1b.py:91` | original: no `applied` in either file | mixed by design | not needed |
| `experiments/qwen_maps/amend2.py:329` (reading B) | **moved** | W0 | code and filing yes; card and figure **no** |
| `experiments/qwen_reading/lab.py:359`, `change` None: K0, K00, K01 | original | W0, state | yes |
| `experiments/qwen_reading/lab.py:359`, `change` set: K10, K11 | **moved** | W0, state | filing yes; card and report **no** |
| `experiments/qwen_reading/run.py:119,178,181` | original | as read | not needed |
| `experiments/qwen_recipe/checks.py:186` | original | W0 | not needed |
| `experiments/qwen_recipe/checks.py:328` (`true_kl`) | original, held before the move | W0, state | yes, docstring and `predictions_checks.json` |
| `experiments/qwen_recipe/size.py:193,196` (`true_chunk`) | original, held before the move | W0, state | yes, docstring |
| `experiments/qwen_recipe/size.py:305,347` (`read_cells`, "file") | moved | W0, state | yes, `size.py:5` |

The filings for readings A and B name the moved vectors: `qwen_maps/results/predictions_amend2.json:3,11` and `qwen_reading/results/predictions.json` definitions.cells. So does `decisions.md:86`, which records the user settling "V′ readouts" in chat on 2026-10-30. That line sits awkwardly with the user not knowing about the readout, so it is worth raising with them plainly. The prose never carried it across.

`experiments/qwen_updates`, `qwen_prefix_test`, `qwen_fresh_*` and `qwen_gsm8k_baseline` call no Γ. Neither does `qwen_sft/amend1.py`, whose rehearsals read `kl_to_teacher` only.

## Frozen-Γ training runs on Qwen

None, scored or otherwise. E40's teacher training never started (`e40.html`, Run field), and its rehearsals read no gate. Component training has no Qwen code. So no Qwen result rests on training subcomponents under a frozen Γ. The route opens the moment teacher training runs. `qwen_teacher.py:24-25` already reads the trained V, so a gate there follows the student's V as well as the activations.

## Places in the log that read as "gate fixed"

| Where | Text | Why it misleads on Qwen |
|---|---|---|
| `sections/techniques.html:18` | Γ "frozen under every map except the Γ refit" | parameters frozen; as a function of x it moves with V |
| `sections/overview.html:76,155` | the pass moves pieces "with Γ and Δ frozen" | written for both models; true for gates only on the 67M |
| `sections/next.html:18` | teacher training, "Γ and Δ frozen", "the map E40 carries to Qwen3-1.7B" | carries a stronger property than Qwen's Γ has |
| `experiments/e44.html:5,21` | "masks from the original model" | reading B's masks read the moved vectors on W0's activations |
| `experiments/e57.html:4` | direct share "with the original switching" | K10's gates read V′. E61 corrects the number, not this card |
| `sections/claims.html:19,27` | direct share on both models, same words | the 67M's K10 cannot see V′; Qwen's filed K10 can |
| `sections/claims.html:31` | sparsity loss "identical" under a frozen Γ | follows from x-only gates on the 67M; it need not hold on Qwen, and was not run there |

`reports/analyses/2026-10-30_analysis_e57-reading-the-update.md:18,34` makes the same slip as the E57 card, as the pipeline audit found.

No log line compares a gate number across the two models directly. The nearest is `sections/gaps.html:27`, which sets the re-gating share on the 67M beside Qwen's. On Qwen that share's numerator, K00 − K01, uses the original V, so a moved V cannot enter it. Its denominator, K00 − K11, reads the moved vectors (`src/commute/qwen_reading.py:58-59`), and so does the direct share's. Qwen's K01 still differs from the 67M's in kind, since its gates read x V through the readout at the state's x.

## What is known of the size

Three readings bound parts of this, none of them the share of a gate.

- `checks_k10.json` reads the direct share on 16 rows, with truly original masks minus through moved vectors. The table below gives it. The filed tolerance is 0.01 and the verdict `stands`. The brief's "under 0.002" is slightly low, as `e61.html:5` says. This bounds one share, at these states, through Γ in bf16. It does not bound the gates themselves.
- `e44.html:16`: from E40's activations, residual taps alone reproduce 0.99 of do-nothing's masks and per-site readouts alone 0.22. That concerns x, not V.
- `predictions_amend2.json:70`: under reading A, moving the rarely-on half changed K by 6e-6 nats. That half has low gates and those stayed off, which bears only on those subcomponents.

| State | Truly original minus through moved |
|---|---|
| gold | -0.0022 |
| E40 | -0.0015 |
| anchor 2 | -0.0001 |
| Gaussian | +0.0009 |

The audit's hand estimate says the mask path is not obviously smaller than the direct path at a single subcomponent. `gamma_parts.py` measures Δg itself.

## A precision finding on the way

The four saved splits under `results/large/checks/` are bf16. All 196 tensors in the headers of `gold_s0_X` and `gauss_X` read BF16, and the other two files have the same size. `checks.split_x` now refuses such a file (`checks.py:311-312`). `reports/reviews/2026-11-02_review_e61-size-code.md:14-16` measured the error at up to 4.1e-2 of the update. So a saved split does not exist in fp32 today. `gamma_parts.py` makes its own, which costs 6 min per state.

## Suggested wording

Say "Γ's parameters frozen" and, on Qwen, add "its gates still read the moved read vectors". On Qwen, keep "original masks" for masks computed before the move, the `true` path. The lead decides the log edits; this review made none.

## Not checked

- The JAX source of the readout and of Γ's causal mask, cited in `qwen_vpd.py:13-14,203-204`. It is not at `~/param-decomp`, whose clone holds the torch code at `74146b5`. The port check matched JAX's g, so I rely on that.
- Any measured share of the gate from the readout, or any Δg: that is the job this review hands over.
- Notebooks, other analyses and plots beyond those named. Only `plot_amend2.py` among the plots was read.
- Older plans and reviews for the same wording.
- The log's cards outside the Qwen line, E44 to E63, except by grep.
- Whether teacher training on Qwen should read the trained V or the original one. That is a design choice for when it runs.
