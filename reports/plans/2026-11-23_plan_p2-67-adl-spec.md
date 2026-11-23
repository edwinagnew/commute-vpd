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
   runs". The baselines are "the per-position average base activation h̄_base and ... finetuned activation h̄ft_j over the 𝒟ft
   samples". **The code differs:** `method.py:compute_differences` (~L915) takes `base_mean` and `ft_mean` over the same
   FineWeb batch as δ̄, not over fine-tuning samples. Code: `token_relevance.py:run_token_relevance`, with `permutations: 3`, `agreement: all` and `frequent_tokens: {num_tokens:
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
   later prompts, so it is not exactly the paper's 8/10, 10 and 5. The search samples at temperature 1.2 (128 new
   tokens). The final texts follow App. C.3, "5 generations with temperature 1.1" (config `final:
   num_samples_per_prompt: 5`, `temperature: 1.1`, 512 new tokens).
8. **Causal effect.** Sec. 5, eq. 1: "replacing the finetuned model activation in the subspace of δ̄ with the
   corresponding base model activation". Eq. 2 takes the difference between the cross-entropy loss with "∀j: h_ft ←
   h̃_ft" and the plain loss. The paper says "We expect the causal effect to be positive on the finetuning data 𝒟ft ...
   negative on random pretraining data 𝒟pt". Fig. 7 reports "average causal effects across three SDF organisms at
   multiple positions" against "64 randomly sampled activation differences", which are base activations at "two random
   token positions" of chat data, differenced. Footnote 13 adds that "Randomly sampled vectors yield even smaller
   effects." Code: `causal_effect.py:_compute_nll_intervened` (`patch_direction: base_to_ft`, `zero_ablate: false`) and
   `_sample_activation_diff_vectors`. The config sets `max_samples: 1000`, `max_total_tokens: 1024`, `positions: [0..8]`,
   and 64 random plus 64 random-difference vectors. The eval sets are the organism's training set, fineweb and tulu. The
   figure plots `incr_ce` (fine-tuned CE with the swap minus without, in nats) on `subset: exclude_pos`. That subset
   leaves out NLL index p + 1, which is the prediction of token p + 2, because `_nll` (~L201) puts the target of index i
   at token i + 1. The plotting script sets `metric_key="incr_ce"`, `SUBSET = "exclude_pos"`. The Qwen curves of Fig. 7
   and Fig. 32 come from `plot_causal_effect.py` ~L1599-1625 (`visualize_causal_effect_by_position_dual`), which pools
   cake_bake, kansas_abortion and fda_approval and compares variant `default` with `mix1-2p0` at layer 13. Training-data
   and FineWeb eval are shown, chat eval is not, and the baseline is the *mean* random difference
   (`include_random_mean=True`, i.e. `random_diff_mean`). The code also writes `incr_rel_ce = (CE_swap - CE_ft) /
   (CE_base - CE_ft)`, the share form, but no paper figure calls it. The upstream code has two unguarded divisions:
   `v = v / v.norm()` (~L274) and the CE gap in `incr_rel_ce` (~L1084). No step here needs an LLM.
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
| Token-relevance baselines | paper: means "over the 𝒟ft samples"; code: means over the same FineWeb batch as δ̄ | not run | — | theirs (paper and code differ) | — |
| Causal-effect zero cases | upstream divides by the direction norm and the CE gap with no guard | zero δ̄ is a no-op (`adl.unit`) | yes, a guard | ours (needed for the do-nothing) | no |
| Fig. 7 models | text: "Qwen3 1.7B, Llama 3.2 8B, and Gemma3 1B"; caption: "Llama 3.2 1B" | — | — | theirs (inconsistent) | — |
| Random-update denominator | — (the paper has no share) | D of each Gaussian weight update was divided by its paired trained state's gain, not its own | yes | ours | no |

## 4. Faithful re-run proposal

### 4.1 Positive control with the authors' code, unchanged

**Option A, recommended: the paper's Qwen aggregate.** Reproduce the Qwen3-1.7B panel of Fig. 7 and Fig. 32 as
`plot_causal_effect.py` ~L1599 builds it:
- three organisms (cake_bake, kansas_abortion, fda_approval), each at variant `default` and `mix1-2p0`, which makes six
  toolkit runs;
- layer 13, the config's positions 0-8, loss subset `exclude_pos`, metric `incr_ce`;
- the baseline is `random_diff_mean`.

The **pass rule** is filed before the run and uses only what the text states. The "mean" is the mean over the three
organisms.
- (a) Training data: the three-organism mean `incr_ce` of `default` is above the mean `random_diff_mean` at 5 or
  more of positions 0-8. Its mean over positions 0-8 is also above the random mean by more than 2 sd of the 64
  random-difference values. The text says "clearly positive on 𝒟ft" and "random diff vectors yield causal effects
  close to zero".
- (b) FineWeb: the mean `incr_ce` of `default` is below 0 at a majority (5 or more) of positions 0-8. The text says
  "negative for Qwen3 1.7B".
- (c) The mixture's mean `incr_ce` on training data, averaged over positions 0-8, is between 0.25 and 0.75 times
  `default`'s. App. F.8 says "roughly half".
- (d), optional: the mixture's mean `incr_ce` on FineWeb is above 0 at a majority of positions 0-8. App. F.8 says
  "Unlike the standard models, however, the causal effect on pretraining data is now slightly positive—the
  intervention increases loss."

Each organism's curve is reported beside the mean, so a pass on the mean with one organism the wrong way is
visible. Positions follow the toolkit, with no BOS for Qwen3: position 0 is a real text token.

A miss on (a)-(c) means the pinned toolkit does not reproduce the paper's Qwen panel. We stop and report.

**Option B, cheaper: cake_bake only,** at `default` and `mix1-2p0`, under the same rules on one organism. This is a
*single-organism check*. A miss fails that check only, not the paper, since the paper reports only the three-organism
mean.

**Side check for either option:** whether a baking token sits in the logit-lens top 20 at some position. It is
qualitative only and has no pass rule.

**Resolved command (one run; six for A):**

    uv run python main.py diffing/method=activation_difference_lens model=qwen3_1_7B \
      organism=<cake_bake|kansas_abortion|fda_approval> organism_variant=<default|mix1-2p0> \
      pipeline.mode=diffing infrastructure=runpod \
      infrastructure.storage.base_dir=/workspace/adl/E###/<organism>_<variant> \
      diffing.method.overwrite=true diffing.method.causal_effect.enabled=true diffing.method.causal_effect.overwrite=true \
      diffing.method.auto_patch_scope.enabled=false diffing.method.steering.enabled=false \
      diffing.method.token_relevance.enabled=false

Notes on the command:
- Patchscope, steering and token relevance are on by default. `ActDiffLens.run` (`method.py` ~L446) runs them before
  the causal effect, and Patchscope, steering and token relevance call OpenRouter. All three are switched off for the
  gate. The logit-lens cache still runs, with no LLM.
- `configs/infrastructure/mats_cluster.yaml` points storage at `/mnt/nw/teams/team_neel_b/...`. We do not write there.
  Every run gets its own fresh `base_dir`, because `diffing.results_dir` is `${model.name}/${organism.name}` with no
  variant (`configs/config.yaml` ~L147). Two variants sharing a directory would skip or overwrite each other.
- `seed: 42` is left as the config's default.
- The eval sets are the config's three (training, FineWeb, tulu). Tulu costs one extra pass, but keeping it avoids a
  Hydra list override.
- Patchscope and token relevance run later only if `OPENROUTER_API_KEY` is in the shell. If it is not, we name it. A
  credential file is never written.

### 4.2 The 67M by a minimal port

nnterp's `StandardizedTransformer` wraps Hugging Face causal language models. The 67M is param-decomp's `LlamaSimpleMLP`
(rotary attention, RMSNorm, ungated GELU MLP), the target of a VPD ComponentModel. We found no stock Hugging Face class
with that block. A wrapper is not shown possible: it was not tried, and step 3 may try it for an hour. Otherwise
`src/commute/adl.py` gains the toolkit's arithmetic for steps 3, 4 and 8:
- CE from `_nll`'s indexing;
- positions 0-8;
- 64 random differences and 64 Gaussian vectors;
- the `all` and `exclude_pos` masks.

**Null path.** Two rules keep the zero cases defined:
- A zero δ̄ is a no-op. This keeps the old port's `unit` rule. Upstream would divide by zero.
- `incr_rel_ce` is undefined when |CE_base - CE_ft| is under a filed floor. The floor is 3 times the largest such gap
  over the matched random updates, measured in stage 0.

Stage 0 runs in this order:
1. W0 (the do-nothing) first. Every `incr_ce` must be exactly 0, and the gap must be 0, so `incr_rel_ce` reads undefined.
2. Then E265's lowercase_s0. Its change is one constant shift.

Lowercase is compared like with like:
- For each of the 64 random-difference directions, take the mean of its `incr_ce` over the same five positions' masks
  (j = 0-4). That gives 64 null values.
- **Pass** if the lens's own five-position mean `incr_ce` exceeds the largest of those 64.

**Qwen parity check of the port.** The port runs through a thin adapter on the cake_bake `default` pair. The rules:
- Toolkit outputs are fresh (`overwrite=true`, new `base_dir`), because the toolkit skips existing results.
- The token-id inputs are the same, checked by sha256, and the masks are the same.
- Valid-token counts are asserted equal.
- δ̄ and every CE must agree within a floor measured between two fresh toolkit runs of the same command.

### 4.3 Per-state quantities on the built pairs, prior-matched pairs and procedural states

1. **As-published trace strength (descriptive only).** This is `incr_ce(j)`, j = 0-8, for both masks `all` and
   `exclude_pos`, averaged over every next-token target of the state's own training windows. Beside it come the 64
   random-difference values (mean, sd, max) and the same reading on Pile. Each state's record also holds:
   - the masks;
   - the valid-target count per mask and position;
   - a sha256 of the token tensor.

   This average dilutes the effect on answers by task-dependent factors:
   - the built tasks read one answer position that is separate from the rest of the window (`holes_fill` `Task`, ~L77);
   - procedural SFT trains on one answer token, and RL on one sampled continuation (`procedural_rl/run.py` ~L222);
   - `exclude_pos` can drop the only answer term.

   So these numbers are not ranked across tasks. **A weak whole-window reading is not evidence about whether ADL can
   tell reuse from new.**
2. **Our adaptation (labelled as ours, not the paper's).** The same swap, read as the CE of each held-out read row's
   answer target at its read position (the built tasks' read rows; procedural held-out items at the cue). A ranking or
   pair gap uses only states with matched task and token support, and with checked gain or KL. The column is read as
   the other baselines are: per pair, the reuse side against the new side, compared with the definition's gap. No
   reuse verdict comes from ADL.
   **Labelling (redteam-lead, 2026-11-23).**
   - As published, the column reads "ADL (as published) detects the training domain, makes no reuse call".
   - The whole-window reading (item 1) is descriptive only.
   - This adaptation is ours. It is labelled so in the paper text, not only in an appendix, and is never called "the
     activation difference lens".
   - It has no positive control for reuse. The lowercase control, a constant shift, shows only that the port sees a
     constant bias, not that it can see reuse.
   - The claim "existing tools cannot see reuse" cannot rest on ADL in either form. At most: "ADL's trace, adapted to
     answer targets, does / does not order the built pairs as the definition does".

The logit-lens top 20 for δ̄ and for the mean base and fine-tuned activations is a side reading. The paper's account
("overfitting" to narrow data, "a constant semantic bias") predicts that the trace tracks how constant a fine-tune's
change is across unrelated text. That is a property of the data, not of reuse. E264's analysis offered the key
dependence of the copy control's change as *a possible explanation* of its failure. The cause was not established.

**Text for δ̄ on the 67M.** `src/commute/data.py` gives only packed, pre-tokenised rows. The plan is:
- Take document starts from the raw corpus `monology/pile-uncopyrighted`. It is public, raw `text`, and its viewer lists
  only `train`. Whether its repository holds the val file is to be checked at run time.
- Tokenise with `EleutherAI/gpt-neox-20b`, the tokenizer `vpd.tokenizer()` loads, with its revision recorded at run
  time. It adds no BOS by default.
- Fallback, if no held-out split is reachable: the packed val rows as before. This is a stated departure: positions 0-4
  are then mid-document.

**The 67M forces these departures:**
- The text is Pile, not FineWeb.
- Random differences are drawn on Pile, because the 67M has no chat template for tulu.
- There is no steering, agent or hypothesis grading, because the 67M cannot follow chat prompts.

**Cost (estimated from FLOPs, not measured).** About 139 forwards per batch of about 1,000 tokens on 1.7B, 500 batches,
three eval sets, so about 35 min per run on an H100 including loading. H100 is taken at about $2.8/h. The first run times
its first batch and stops if a batch takes over twice the estimate (friction 902).

| Item | Where | Estimate | Cost |
|---|---|---|---|
| A: six toolkit runs (3 organisms × 2 variants) | H100, 2-3 pods in parallel | about 3.5 GPU-h, about 1.3 h wall | about $10 |
| B: two cake_bake runs (default, mix1-2p0) | H100 | about 1.2 h | about $3.5 |
| First-batch timing | inside the first run | — | included |
| Parity: a second fresh toolkit run of cake_bake default (the floor) | H100 | about 35 min | about $1.6 |
| Parity: the port on the same inputs | H100 | about 20 min | about $1 |
| OpenRouter graders (later, only with the key) | API | — | a few dollars |
| 67M stage 0 (W0, lowercase, 64 nulls) | cluster debug | about 5 min | $0 |
| 67M items (about 180 states × about 15 s) | cluster debug, two jobs | about 45 min | $0 |

Totals: option A plus parity is about $13; option B plus parity is about $6. Both are under the $30 cap.

## Not run

- Nothing has been run for this spec.
- Not checked: whether nnterp can wrap the 67M by a custom class.
- Not checked: whether a 4-layer 67M copies in context well enough for the identity Patchscope.
- Not checked: whether `monology/pile-uncopyrighted` holds a val file.
- Not checked: the paper's figure values. The pass rules use only signs and ratios the text states.
- Not checked: whether `mix1-2p0` resolves for all three organisms at the config's adapter revisions. The config
  lists it for cake_bake, kansas_abortion and fda_approval on Qwen3-1.7B. Whether the adapters load is not checked.
- Not checked by running: that the toolkit's Qwen3 tokenizer call (`add_special_tokens=True`) adds no BOS. Qwen3 has
  none by default, so position 0 is a real token.

## Review response (GPT plan-reviewer, 2026-11-23)

All five points were real and are fixed above:
1. Section 4.1 now has option A (the paper's three-organism Qwen aggregate against the mean random baseline) and
   option B (cake_bake only, labelled a single-organism check). The baking token is a side check only. Both options are
   priced.
2. Section 4.1 files the resolved command: pipeline mode `diffing`, LLM steps off, a fresh `base_dir` per run (the
   results directory has no variant), and explicit variants. Mixture and parity runs are budgeted on their own rows.
3. Section 4.3 splits the as-published whole-window reading (descriptive, with masks, counts and checksums) from our
   answer-target adaptation (labelled as ours, ranked only on matched support). The `exclude_pos` index is corrected in
   step 8.
4. Section 4.2 adds the null path (zero direction is a no-op, a gap floor for `incr_rel_ce`, W0 first), the
   like-with-like lowercase rule and the parity conditions.
5. Fixed: the steering final generation settings, the code-versus-paper source of the mean-activation baselines, the
   E264 wording, and the raw-corpus and tokenizer plan with its fallback.

Rejected: nothing.

Redteam-lead (`~/commute-data/agents/compare-lead/note-2026-11-23-redteam-p2-67-adl-spec.md`), four fixes, all applied:
1. Section 3 gains a row: E264-267's D divided the Gaussian weight updates' loss by the paired trained state's gain,
   not by their own.
2. Section 4.3 states how the column is labelled and read. As published, ADL detects the training domain and makes no
   reuse call. The answer-target adaptation is ours, labelled so in the paper text and never called "the activation
   difference lens". The lowercase control shows only that the port sees a constant bias. "Existing tools cannot see
   reuse" cannot rest on ADL.
3. Pass rule (a) now needs 5 or more positions plus a margin of over 2 sd. An optional (d) is added from App. F.8
   ("slightly positive"). Every organism's curve is reported beside the mean.
4. Two items are added to "Not checked": whether `mix1-2p0` resolves for all three organisms, and that Qwen3 adds no BOS.
