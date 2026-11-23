# Plan: the activation difference lens as published, and what our E264-267 changed (cell P2.67, step 1)

Cell P2.67 (due 2026-11-25, cap $30). User, 2026-11-23: "re-run the ADL baseline properly (assuming its even relevant,
maybe read the actual paper or something)". This is step 1, the method spec. Step 2 (redteam-lead and an independent
reviewer check it against the paper) comes before any code or run.

Sources read for this spec. The paper is Minder et al., "Narrow Finetuning Leaves Clearly Readable Traces in Activation
Differences", arXiv 2510.13900. Version 3 (4 Mar 2026) is the latest: the arXiv listing shows v1 14 Oct 2025, v2 2 Mar
2026, v3 4 Mar 2026, checked 2026-11-23. Every quote below is from v3, as in the local copy
`~/commute-data/agents/compare-lead/adl/paper/adl.txt` (line 181: "arXiv:2510.13900v3 [cs.CL] 04 Mar 2026"). The code is
diffing-toolkit at 617d9266 (`~/commute-data/agents/compare-lead/adl/dt`). `git ls-remote origin HEAD` on 2026-11-23
returns the same commit, so upstream has not moved. Code paths below are relative to
`src/diffing/methods/activation_difference_lens/` unless they start with `configs/`, `src/` or `narrow_ft_experiments/`.

Symbols, each defined once. **h** is the residual activation at the read layer. **δ̄_j** is the mean over texts of
h(fine-tuned) minus h(base) at token position j. **ΔCE** is the paper's causal effect: the rise in the fine-tuned model's
cross-entropy when its activation along δ̄_j is set to the base model's. **D** is our E264-267 scalar: the largest, over
positions 0-4, share of a task's answer log-odds gain that is lost under that same swap.

## 1. Plain answers

D is not in the paper. The paper's causal-effect experiment reports a rise in cross-entropy on the fine-tuning documents,
in nats, per position and averaged over three organisms. The authors' code also writes the same rise as a fraction of the
base-to-fine-tuned loss gap, but no figure in the paper uses it, and neither reads a task's answer log-odds. The paper
does give per-fine-tune scores. The headline is an LLM grader's 1-to-5 grade of whether an LLM agent, given the lens's
outputs, named what the fine-tune was trained on. The others are the share of the top 20 lens tokens that a grader calls
relevant to the fine-tuning domain, and how similar the text from the steered model is to the fine-tuning data. **ADL
detects what a fine-tune was about, not whether it reused existing mechanisms**; the paper reads the trace as
"overfitting" to narrow data and makes no reuse claim. The most faithful way to use it as our baseline is the paper's own
causal effect, read on each state's own training data beside the paper's random-difference baseline, with the logit-lens
tokens kept as a side reading. The column is then read as the other baseline columns are: does it order pairs as the
definition does? That reading is ours, not the paper's.

## 2. The published method, step by step

Each step quotes v3 and names the code that does it. **[LLM]** marks a step that needs an LLM grader or agent (gpt-5,
gpt-5-mini or gpt-5-nano through OpenRouter, `api_key_path: openrouter_api_key.txt`). **[chat]** marks a step that needs
the studied model to be a chat model.

1. **Models and layer.** Paper Sec. 3: "We focus on the middle layer ℓ=⌊L/2⌋". The code uses
   `get_layer_indices` (`src/diffing/utils/activations.py`), which returns `int(layer * (num_layers - 1))` with
   `layers: [0.5]` (`configs/diffing/method/activation_difference_lens.yaml`), read at `model.layers_output[layer]`. For
   Qwen3-1.7B (28 layers) the code takes index 13 (`plot_causal_effect.py`: `("qwen3_1_7B", 13)`). That equals ⌊28/2⌋ = 14
   only if the paper counts layers from 1, which it does not say. For L = 4, the code takes index 1, the output of the
   second block.
2. **Text, sample count and positions.** Sec. 3: "the first k tokens on a pretraining corpus 𝒟pt containing 10,000
   samples ... where k=5". Code: `max_samples: 10000`, fineweb-1m-sample, `n: 128` positions collected
   (`method.py:load_and_tokenize_dataset`). Each document is cut to its first 1,280 characters and tokenised with
   `add_special_tokens=True`, so a tokenizer that has a BOS token puts it at position 0.
3. **Mean difference.** Sec. 3: "We compute the average activation difference per position δ̄_j for 0≤j<k across all
   samples". Code: `method.py:compute_differences`, `diff = ft_acts[layer] - base_acts[layer]; mean_diff = diff.mean(dim=0)`.
   It also saves the per-position base and fine-tuned means, h̄_base and h̄_ft, which serve as baselines.
4. **Logit lens.** App. C.1: "p = softmax(W_U final_layer_norm(δ̄_j))". Code: `src/diffing/utils/model.py:logit_lens`.
   This step needs no LLM.
5. **Patchscope [LLM].** App. C.1: the fine-tuned model runs "tok1→tok1\ntok2→tok2\ntok3→tok3\n?" with "layer ℓ's
   activation at the last token position (token ?) [replaced] by λδ̄_j". It uses three prompt triples, "(man, 1135,
   hello), (bear, 42, blue) and (921, target, anna)", and keeps "the intersection of tokens appearing in the top 16384
   results across all three prompt sets". The vector is normalised "to match the expected norm η_ft ... ignoring the
   first 3 tokens". The paper then says "we use 30 scaling factors ... The grader selects the scaling factor that produces
   the largest set of semantically coherent tokens". Code: `auto_patch_scope.py:run_auto_patch_scope_for_position`, which
   uses the same 30 scales; `src/diffing/utils/model.py:patchscope_lens`; and `PatchScopeGrader` (gpt-5-mini). The norm
   step skips 5 tokens, not 3 (`method.py:_compute_and_save_norms`, `skip_tokens = 5`).
6. **Token relevance [LLM].** Sec. 3: "We extract the top-20 tokens and compute what fraction are relevant to the
   finetuning domain. We use a grader model (gpt-5-mini) with access to the finetuning objective description and the
   top-100 most frequent tokens in the finetuning dataset ... report the maximum fraction across all investigated
   positions". App. C.2: "three times with shuffled token order ... relevant only if classified as such in all three
   runs". The baselines are "the per-position average base activation h̄_base and ... finetuned activation h̄_ft". Code:
   `token_relevance.py:run_token_relevance`, with `permutations: 3`, `agreement: all` and `frequent_tokens: {num_tokens:
   100, min_count: 10}`.
7. **Steering [LLM] [chat].** Sec. 3: "steer the finetuned model p_ft by adding a scaled activation difference αδ̄_j to
   all token positions during generation. We evaluate steering on a fixed set of 20 chat interaction prompts". App. C.3:
   "binary search over [0,100] with 10 iterations ... 10 generations (temperature 1.2) ... coherent if at least 8/10
   generations pass", then "two additional prompts ... [0,2π1] with 5 iterations ... The final steering factor is the
   average". The score is "pairwise cosine similarity between semantic embeddings of steered text and embeddings of the
   finetuning dataset" (Qwen3 Embedding 0.6B, 500 samples). Its baselines are within-dataset similarity, unsteered text,
   and tulu chat data. Code: `steering.py:run_steering` and `find_steering_threshold`, `generate_steered`
   (`apply_chat`), `resources/steering_prompts_closed.txt`, and `narrow_ft_experiments/plotting/plot_steeringcosim.py`.
   The code's settings are `coherence_threshold: 75.0`, `steps: 12` with 3-level look-ahead, and `steps // 2` for the
   later prompts, so it is not exactly the paper's 8/10, 10 and 5.
8. **Causal effect.** Sec. 5, eq. 1: "replacing the finetuned model activation in the subspace of δ̄ with the
   corresponding base model activation". Eq. 2 takes the difference between the cross-entropy loss with "∀j: h_ft ←
   h̃_ft" and the plain loss. The paper says "We expect the causal effect to be positive on the finetuning data 𝒟ft ...
   negative on random pretraining data 𝒟pt". Fig. 7 reports "average causal effects across three SDF organisms at
   multiple positions" against "64 randomly sampled activation differences", which are base activations at "two random
   token positions" of chat data, differenced. Footnote 13 adds that "Randomly sampled vectors yield even smaller
   effects." Code: `causal_effect.py:_compute_nll_intervened` (`patch_direction: base_to_ft`, `zero_ablate: false`) and
   `_sample_activation_diff_vectors`. The config sets `max_samples: 1000`, `max_total_tokens: 1024`, `positions: [0..8]`,
   and 64 random plus 64 random-difference vectors. The eval sets are the organism's training set, fineweb and tulu. The
   figure plots `incr_ce` (fine-tuned CE with the swap minus without, in nats) on `subset: exclude_pos`, which leaves out
   the loss term at index p + 1 (`plot_causal_effect.py`, `metric_key="incr_ce"`, `SUBSET = "exclude_pos"`). The code
   also writes `incr_rel_ce = (CE_swap - CE_ft) / (CE_base - CE_ft)`, the share form, but no paper figure calls it. No
   step here needs an LLM.
9. **Interpretability agent [LLM] [chat].** Sec. 3.1: "an LLM (gpt-5 with medium thinking strength) given access to i)
   the Patchscope and Logit Lens results for the first k tokens, and ii) one steered and one unsteered answer". It can
   query "both the base and finetuned models" within a budget i. App. C.4 notes that `ask_model` "only supports single
   turn conversations", and limits are "a maximum of i model interactions and 15 agent turns". Code: `agents.py:ADLAgent`
   and `agent_tools.py`.
10. **Grading [LLM].** Sec. 3.1: "a grader model (gpt-5-mini) with access to the true finetuning objective, a detailed
    grading rubric tailored to each organism type ... assigns scores from 1 to 5". Code: `configs/diffing/grading_rubrics.yaml`
    and `src/diffing/utils/graders/hypothesis_grader.py`.
11. **Positive control and baselines.** There are "33 model organisms across 7 models: 20 SDF, 6 EM, 1 Subliminal and 6
    Taboo". Success is stated as follows: "The ADL-enabled agent with i=5 correctly identifies the broad finetuning
    objective (grade ≥2) for all but three organisms—a 91% success rate", against "even the strongest blackbox agent
    with i=50 ... only 39%". App. E.1 gives the negative reference: chat tuning shows "no clear difference between the
    metrics on the activation difference and the baseline". Sec. 6 gives a dose response: mixing in pretraining data
    "substantially reduces detectable bias", and the 1:2 mixtures' "causal effects on 𝒟ft are roughly half" (App. F.8).

## 3. Departures of E264-267 from the paper and code

"Ours" means we changed it. "Theirs" means the paper or code itself does it that way (the paper and code disagree, or
the code adds something). "67M forces" means the 67M leaves no faithful option.

| Step | Paper / code | E264-267 | Departure | Whose | 67M forces |
|---|---|---|---|---|---|
| Layer | ⌊L/2⌋ (paper); `int(0.5(L-1))` (code) | block 1 output = `in2`, the code's rule | no (paper is ambiguous; we follow the code) | theirs (ambiguity) | no |
| Text | 10,000 fineweb document starts, BOS by tokenizer | 8,192 packed Pile val rows, positions 0-4 of each 512-token row: mid-document text, no BOS | yes | ours | partly: Pile is the 67M's corpus, but document starts are available |
| Positions | k = 5 (relevance, steering); 0-8 (causal effect); 0-5 (Patchscope config) | 0-4 for everything | yes for causal effect | ours | no |
| Mean difference | `diff.mean(0)` | the same, in float64 | no | — | — |
| Logit lens | softmax(W_U ln_f(δ̄)) | the same, as ranks | no | — | — |
| Token relevance | LLM grader + objective description + frequent tokens, 3 shuffles, max over positions; baselines h̄_base, h̄_ft | "relevant" = one of the task's answer tokens; side reading only; no h̄ baselines | yes | ours | no (a grader can be given a task description) |
| Patchscope | 3 identity prompts, 30 scales, LLM picks λ | not run | yes | ours | no (needs only an LLM key and in-context copying, untested on a 4-layer model) |
| Norm for scaling | paper skips 3 tokens, code skips 5 | not used | — | theirs (paper and code differ) | — |
| Steering | 20 chat prompts, chat template, coherence grader search, embedding similarity | not run | yes | ours | yes: the 67M is not a chat model, so the prompts and template are undefined |
| Causal effect: what is swapped | component along δ̄_j set to base, every position | the same (`adl.revert`) | no | — | — |
| Causal effect: readout | mean CE over the fine-tuning documents (exclude_pos), in nats | docstring: "here read as the drop of a per-row readout, not of cross-entropy": answer log-odds at the read position of the task's read rows | yes | ours | no: each task's training windows exist, so CE on them is computable |
| Causal effect: scalar | ΔCE per position, plotted; no max, no share (code also writes `incr_rel_ce`, unused in the paper) | D = max over j of the lost share of the log-odds gain | yes | ours (the share form exists only in the code's output) | no |
| Causal effect: eval sets | training data, fineweb, tulu | task read rows only | yes | ours | partly (no chat data) |
| Random baseline | 64 random differences of base activations on chat data, plus 64 Gaussian vectors | 64 random differences on Pile rows 0-127; no Gaussian vectors | yes | ours | partly (no chat template for tulu) |
| Positive control | the paper's organisms (SDF, etc.), with agent grade ≥ 2 and ΔCE on 𝒟ft above the random baseline | E237 copy_s0, then E53 lowercase_s0, with D ≥ 0.05 and above the random differences | yes | ours | no for the toolkit control; a 67M control is ours by necessity |
| Agent, grading | gpt-5 agent, gpt-5-mini grader, rubric | not run | yes | ours | yes: `ask_model` is single-turn chat |
| Steering search constants | paper 8/10, 10 and 5 iterations; code 75%, 12 and 6 | — | — | theirs | — |
| Fig. 7 models | text: "Qwen3 1.7B, Llama 3.2 8B, and Gemma3 1B"; caption: "Llama 3.2 1B" | — | — | theirs (inconsistent) | — |

## 4. Faithful re-run proposal

**Positive control first, with the authors' code unchanged.** Run the toolkit as pinned on one of the paper's own
organisms, Qwen3-1.7B `cake_bake` (`default` adapter), with its 1:2 mixture `mix1-2p0` as the weaker reference:
`main.py diffing/method=activation_difference_lens model=qwen3_1_7B organism=cake_bake`, with `causal_effect.enabled:
true`. **Pass rule**, from Sec. 5 and App. F.8 and filed before the run:
- (a) ΔCE on cake_bake's training data is above 0 at every position 0-4, and above the largest of the 64
  random-difference ΔCEs at 3 or more of them.
- (b) ΔCE on fineweb is below 0 at 3 or more of positions 0-4 ("negative for Qwen3 1.7B").
- (c) The mixture's ΔCE on the training data, at the default model's best position, is under 0.75 times the default's
  ("roughly half").
- (d) The logit-lens top 20 at some position 0-4 contains a baking token (read by eye, no LLM).

If the key `OPENROUTER_API_KEY` is set in the shell, also run Patchscope and token relevance with the paper's graders.
If it is not set, name it and skip those steps. A credential file is never written. A miss on (a)-(c) means the pinned
toolkit does not reproduce the paper, and we stop and report.

**The 67M, by a minimal port.** nnterp's `StandardizedTransformer` wraps Hugging Face causal language models. The 67M is
param-decomp's `LlamaSimpleMLP`: rotary attention, RMSNorm and an ungated GELU MLP, held as a VPD ComponentModel's
`target_model`. We found no stock Hugging Face class with that block, and the toolkit also calls `model.steer`,
`model.generate` and `layers_output`. A wrapper is therefore not shown possible. It was not tried, and it is the first
thing step 3 may try for an hour. Otherwise, extend `src/commute/adl.py` with exactly steps 3, 4 and 8 (and 5 if a key
exists), keeping the toolkit's arithmetic: CE over all target tokens with `exclude_pos`, positions 0-8, 64 random
differences and 64 Gaussian vectors. **Port check:** run the port's functions through a thin adapter on the same
Qwen3-1.7B cake_bake pair. They must match the toolkit's δ̄ and ΔCE within a floor measured between two toolkit runs.

**Per-fine-tune quantity on the built pairs, prior-matched pairs and procedural states:**
- ΔCE_ft(j), the paper's eq. 2 on the state's own training windows (the task's fit half), in nats, for j = 0-8.
- Beside it, the 64 random-difference ΔCEs on that state, as their mean, sd and max.
- ΔCE_pt(j) on Pile.
- The column scalar (ours, filed before the run): A = the mean over j = 0-4 of ΔCE_ft(j), minus the random-difference
  mean.
- `incr_rel_ce` as a secondary reading, since the code writes it.

The logit-lens top 20 for δ̄, h̄_base and h̄_ft is a side reading. The column is read as the other baselines are: per pair,
the reuse side against the new side, compared with the definition's gap. No reuse verdict comes from ADL. The paper's own
account predicts that A tracks how constant a fine-tune's change is across unrelated text, not reuse. E264's copy control,
whose change depends on the key, failed for exactly that reason.

**The 67M forces these departures:**
- Text: the 67M's Pile documents taken from their starts, with no BOS if its tokenizer adds none, not fineweb.
- Random differences: drawn on Pile, because the 67M has no chat template for tulu.
- No steering, agent or hypothesis grading, because the 67M cannot follow chat prompts or answer the agent's questions.
- A 67M positive control is still needed after the toolkit's. Use E265's lowercase_s0, whose change is one constant
  shift, read by the new quantity, with A above the random-difference max.

**Cost (estimated from FLOPs, not measured; step 3 times one batch first).**

| Job | Where | Estimate | Cost |
|---|---|---|---|
| Toolkit control: about 139 forwards per batch of about 1,000 tokens on a 1.7B model, 500 batches, three eval sets | H100 pod | about 10 min per eval set, about 1 h with loading | about $3 |
| Toolkit control, same work | cluster L40 (free) | about 55 min per eval set | $0, but past the 2 h debug cap for all three sets |
| OpenRouter graders, if the key is set | API | — | a few dollars |
| 67M items, about 180 states at about 15 s each | cluster debug, two jobs | about 45 min | $0 |

Total about $5-8, well under the $30 cap.

## Not run

- Nothing has been run for this spec.
- Not checked: whether nnterp can wrap the 67M by a custom class.
- Not checked: whether a 4-layer 67M copies in context well enough for the identity Patchscope.
- Not checked: the paper's figure values. The pass rule uses only the signs and ratios the text states.
