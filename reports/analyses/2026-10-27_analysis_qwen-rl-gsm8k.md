# Can RL on GSM8K give the Qwen3-1.7B decomposition a small update worth testing? (scoping)

2026-10-27. Read-only scoping for the user's request: "eventually, i want to try and do RL - starting with gsm8k. but
we need to plan things out first to be meticulous and efficient." Nothing was run on a GPU and no code was written.
Two CPU checks ran on the cluster worker through `fleet/cpu-run`: GSM8K prompt lengths under the Qwen3 chat template,
and the bf16 rounding step of Qwen3-1.7B's weights. This note replaces §4 of
`reports/analyses/2026-10-14_analysis_rl-feasibility.md`, which was written before the model was known.

**In plain words.** Qwen3-1.7B with its thinking switched off solves only about two-thirds of GSM8K, and its answers
are short enough to sit inside the 512 tokens the decomposition was fit on. That makes non-thinking GSM8K a good first
RL task. With thinking on, it scores near ninety per cent and most of its answers run past 512 tokens. The main trap is
numerical: one RL step is far smaller than the rounding step of the 16-bit weights the decomposition and the usual
inference tools use. Every forward pass that measures an early update must keep that update in 32-bit, or it vanishes.
The cheapest useful first step is a baseline evaluation with no training, then a hand-written RL loop of about a
hundred steps, with controls run through the same code, for roughly $20-30 of H100 time in all.

## 0. What the model's identity settles from the 2026-10-14 note

| Question in the 2026-10-14 note §4 | Answer now | Source |
|---|---|---|
| Base or post-trained? | Post-trained hybrid Qwen/Qwen3-1.7B, not -Base. Post-trained by strong-to-weak distillation, off-policy and then on-policy from Qwen3-32B/235B; the report names no RL stage for the small models | [Qwen3 tech report](https://arxiv.org/html/2505.09388v1) |
| Is RL from it a small update from the decomposed checkpoint? | Yes. The note's fairness item 1 is met; this is not zero-RL on a base model | — |
| What text was it decomposed on? | `firestock3-text-qwen3_1_7b_chat-eod-unitsplit-rowshuf-v1`: chat-formatted text, 512-token rows, Qwen3 tokenizer | `p-e9fcffd5-share/data/datasets/*/meta.json` |
| Can we load it on one GPU? | Yes on paper: the decomposition restores in bf16 at about 24 GB, single-device layout. It is a **JAX** checkpoint loaded with param-decomp commit `177d0cce8`, not the pinned torch commit `74146b5` that `src/commute/` uses. No loader or harness for it exists in this repo yet | `/mnt/nw/home/e.agnew/p-e9fcffd5-share/qwen_decomp.md` |
| Target precision in the decomposition run | `target.weights_dtype: bfloat16` | `runs/p-e9fcffd5/launch_config.yaml` |

What the decomposition covers: all 196 attention and MLP matrices, 1.41 B parameters. Subcomponents per layer: q
2048, k 1024, v 1024, o 2048, and 6144 each for gate, up and down. The global
causal-importance function is itself a 32-block transformer of width 3840, about 5.7 B parameters by my count. Not
decomposed: the tied embedding/unembedding matrix (311 M) and the RMSNorm gains, including the per-head q/k norms.

## 1. GSM8K headroom, by mode

| Setting | GSM8K | Other math | Response length | Source |
|---|---|---|---|---|
| Qwen3-1.7B-Base, 4-shot CoT | 75.4 | MATH 43.5 | — | [tech report, Table 8](https://arxiv.org/html/2505.09388v1) |
| Qwen3-1.7B non-thinking, greedy, 2k cap | **65.3** | ASDiv grade 5: 68.5 | 106 words mean | [2510.07880, Table A4](https://arxiv.org/html/2510.07880v2) |
| Qwen3-1.7B thinking, greedy, 8k cap | 87.0 | ASDiv grade 5: 88.4 | 555 words mean, 473 of them thinking | same |
| Qwen3-1.7B thinking, sampled, 32k cap | 90.4 | — | not given | [SwiReasoning, Table 1](https://arxiv.org/html/2510.05069v3) |
| Qwen3-1.7B non-thinking | — | MATH-500 73.0, AIME'24 13.4 | — | [tech report, Table 20](https://arxiv.org/html/2505.09388v1) |
| Qwen3-1.7B thinking | — | MATH-500 93.4, AIME'24 48.3 | — | [tech report, Table 19](https://arxiv.org/html/2505.09388v1) |
| Qwen3-8B non-thinking, GSM8K, 512-token output cap | 93.1 | — | 152 tokens mean; 0.3% hit the cap | [Coupling Tax, Tables 1 and 5](https://arxiv.org/html/2605.07686) |
| Qwen3-8B thinking, GSM8K, 512-token output cap | 56.9 | — | 62.6% hit the cap | same |

The Qwen3 report gives no GSM8K number for the post-trained 1.7B. The only direct pair is the overthinking study's,
greedy and with its own answer extraction. Non-thinking mode leaves about a third of GSM8K to gain; thinking mode leaves
about a tenth. The non-thinking figure sits below the base model's few-shot score, which hints at answer-format losses;
part of any early RL gain may therefore be format, and the baseline evaluation should score both a strict and a lenient
extractor to size that share.

**Recommendation: non-thinking mode, zero-shot, the chat template with `enable_thinking=False`, Qwen's math suffix**
("Please reason step by step, and put your final answer within \boxed{}."). It has headroom, short answers and an
answer format that is easy to verify. Variants, in order of preference if headroom or signal turns out short:

| Variant | Why | Concern |
|---|---|---|
| MATH-500 levels 1-3 or the MATH train split, non-thinking | Non-thinking MATH-500 is 73.0, so headroom remains | Longer answers; check the 512-token fit first |
| GSM-Symbolic or GSM-Plus as the held-out test | Guards against GSM8K test contamination of Qwen models, raised for MATH-500 by [2507.10532](https://arxiv.org/abs/2507.10532) | A second eval set to maintain |
| Answer-only format, no reasoning shown | Very short responses, low accuracy, so much headroom | Teaches mental arithmetic, likely closer to a new capability than to strengthening |
| Format-only reward | A pure "strengthen an existing behaviour" update, useful as a contrast arm | Saturates in a few steps; little to learn |
| GSM-hard | Harder numbers | Known label errors; would need cleaning |
| Thinking mode | The deployed default | Most rollouts exceed 512 tokens; the decomposition cannot be checked on them |

## 2. The 512-token window

Measured on CPU with the Qwen3-1.7B tokenizer, chat template, zero-shot, with the math suffix:

| GSM8K split | Prompt tokens, thinking template | Prompt tokens, non-thinking template | Gold solution tokens, calculator tags removed |
|---|---|---|---|
| train, n = 7473 | mean 83, p99 151, max 235 | mean 87, p99 155, max 239 | mean 90, p99 216 |
| test, n = 1319 | mean 84, p99 151, max 211 | mean 88, p99 155, max 215 | mean 92, p99 217 |

The non-thinking generation prompt already ends in an empty `<think>\n\n</think>\n\n` block, hence the four extra
tokens. That leaves about 360-430 tokens for the answer.

| Mode | Expected answer length | Expected share of prompt + answer within 512 | Basis |
|---|---|---|---|
| Non-thinking | about 140-200 tokens mean | **about 97-99%** | 106 words for the 1.7B; 152 tokens and 0.3% over 512 for the 8B |
| Thinking | about 700-1000 tokens mean | **about 10-35%** | 555 words for the 1.7B; 63% of the 8B's answers exceed 512 output tokens, and the smaller model thinks longer |

Both lines are estimates; Stage 0 measures them. What they mean:

- **Non-thinking fits.** Cap generation at 512 minus the prompt length, so every rollout lies inside the window the
  decomposition was fit on. Mask truncated rollouts out of the loss rather than scoring them zero. A zero reward for
  truncation would add a brevity pressure to δW; at a 1-3% truncation rate either choice changes little.
- **Thinking does not fit.** Faithfulness past position 512 is untested. Γ is a causal transformer over the residual
  stream that never saw later positions, and the target model's RoPE positions past 512 never entered the fit. A
  failure there could not be told apart from a failure of the map. Scoring only the first 512 tokens of a thinking
  rollout would skip the answer, where the reward acts.
- **Unknown whether firestock3 holds Qwen3 thinking traces or empty think blocks.** The preprocessing name says only
  "chat". The gate in Stage 0c measures whether this matters.

## 3. Tooling

### What the project needs, and what each need implies

| Need | Implication |
|---|---|
| δW is the true update | fp32 master weights; δW = W_t − W₀ computed in fp32, W₀ upcast exactly from the hub's bf16 |
| δW at steps 1, 5, 20, 100 | Save δW of the 196 decomposed matrices at those steps. Stored as bf16 *differences* it is 2.8 GB per checkpoint; bf16 keeps 0.4% relative error on δW itself, which is fine. Never store W_t in bf16 |
| δW lives where the decomposition is | Freeze the embedding matrix and all RMSNorm gains, so the whole update lies in the 196 decomposed matrices. Otherwise part of δW sits where no map can carry it |
| Determinism and seeds | Seed the prompt order and the sampler separately; save every step's rollouts as token ids, rewards and old log-probs. The update can then be replayed from the saved rollouts even where regeneration is not bitwise |
| A matched SFT update | Same loop, same rollouts, a different per-sample weight (§4) |
| Pinned versions, uv, Python 3.13 | Checked below |

### Precision: one RL step is below bf16's rounding step

Measured on Qwen3-1.7B's weights for q, v, gate and down in layers 0, 13 and 27. Median |w| is 0.020-0.025 in every
matrix. The share of entries where an update of the given size changes the bf16 value:

| Update per entry | 1e-6 | 1e-5 | 1e-4 | 1e-3 |
|---|---|---|---|---|
| Share of entries that change in bf16 | 1.1-1.4% | 9-11% | 60-70% | 100% |

An Adam step at the usual RL rate of 1e-6 moves each entry by at most about 1e-6. So at steps 1-20, 90-99% of δW
disappears when W₀ + δW is rounded to bf16. TF32 matmuls round to 2⁻¹¹ relative, about 1e-5 here, so they hide it too.
Consequences:

1. **Every measuring forward at W₀ + δW runs in fp32 with TF32 off**, or adds δW·x as a separate fp32 term beside a
   bf16 W₀·x. This covers the behaviour readouts, the KL to π₀ and every decomposition check.
2. **The JAX decomposition loads the target in bf16** (`weights_dtype: bfloat16`). Whether the loader accepts fp32
   target weights, or a separate δW term, is unchecked and blocks any analysis of early steps.
3. **Training under bf16 autocast** computes each step's gradient at bf16(W_t), which equals bf16(W₀) for most entries
   early on. The fp32 master still accumulates the true update, and this is how real post-training runs. Pure fp32
   training avoids it at roughly ten times the compute.
4. **A bf16 vLLM sampler sees W₀ for the first tens of steps**, so its rollouts are off-policy relative to the fp32
   master. This is one reason, beyond the version clash below, to sample with the training model itself.
5. This is the "precision gate" of [Path Not Taken](https://arxiv.org/abs/2511.08567): part of RL's apparent sparsity is
   bf16 hiding small updates. Any sparsity number we report must come from fp32 δW.

### Frameworks against the project's pins

The project pins Python 3.13, torch 2.8.0, transformers 4.57.3 and numpy 2.4.0 (`pyproject.toml`). PyPI metadata,
read 2026-10-27:

| Option | Fits the pins? | Notes |
|---|---|---|
| **Hand-written loop**, HF `generate` in the existing environment | **Yes, no new dependency** | About 200 lines. Sampler and trainer are one model, so no train-inference mismatch. Easy to make exactly on-policy, freeze parameters, save δW and rollouts, and swap the reward rule for the controls |
| TRL 0.26.2 GRPOTrainer + vLLM 0.11.0 | **No, numpy clash.** vLLM 0.11.0 pins torch 2.8.0 and needs transformers ≥ 4.55.2, both fine, and ships abi3 wheels that install on 3.13. It needs numba 0.61.2, which needs numpy < 2.3, against our numpy 2.4.0. TRL 0.26.x wants vLLM ≥ 0.10.2, < 0.12 | Needs a separate environment with numpy 2.2.x. TRL defaults to change: `beta=0.0`, `loss_type="dapo"`, `scale_rewards="group"`, `temperature=1.0`, `max_completion_length=256`, `vllm_mode="server"`, `learning_rate=1e-6`, truncated importance sampling on with cap 3. Load the model in fp32 explicitly; bf16 load means no fp32 master |
| TRL 1.16 + vLLM 0.22-0.31 | No | vLLM 0.22 pins torch 2.11; 0.32 pins torch 2.13 |
| verl 0.9.1 | No | Requires Python < 3.13. Its quickstart is PPO on GSM8K with Qwen2.5-0.5B on one 24 GB GPU, so it is a good reference recipe |
| OpenRLHF 0.11.2 | Separate env | Ray + DeepSpeed + vLLM, built for multi-GPU; heavy for one 1.7B model |

The JAX decomposition needs its own environment anyway: param-decomp `177d0cce8` wants Python 3.12-3.13, jax 0.11.1,
numpy 2.4.6 and pydantic < 2.12. So the analysis side already means two environments; the RL side need not add a third.

**Recommendation: the hand-written loop in the existing environment.** Non-thinking GSM8K answers are short, so
vLLM's speed matters little. The loop's simplicity is what makes a strange number readable as data or bug, which is the
project's coding rule. vLLM becomes worth its separate environment only for thinking mode or for a long RL run.

### Memory and speed on one H100 80 GB

| Item | Size |
|---|---|
| fp32 weights, about 2.0 B with the tied embedding | 8 GB |
| fp32 gradients for the 196 trainable matrices, 1.41 B | 5.6 GB |
| Adam moments | 11.3 GB |
| bf16 frozen π₀ copy for KL readouts | 4 GB |
| KV cache for 512 rollouts × 512 tokens, at about 112 KB per token | about 30 GB |
| Total before activations | about 60 GB; activations need gradient checkpointing and micro-batches |

So full-weight fp32-master RL fits one H100 80 GB, or an A100 80 GB. An L40S at 48 GB fits with fewer rollouts per
generation pass. A 4090 at 24 GB does not fit full-weight Adam; it fits LoRA. Plain SGD saves the 11 GB of moments and
serves the project's SGD-vs-Adam subgoal (`GOALS.md`).

Speed, estimated by arithmetic and not benchmarked:

| Per step | Estimate |
|---|---|
| Rollouts | 64 prompts × 8 samples = 512, about 200 tokens each |
| Generation, HF `generate` at batch 256-512 | 20-40 s |
| One bf16-autocast forward and backward over about 180k tokens | 10-20 s |
| Step time | 30-60 s, so 60-120 steps per hour, about $1.5-3 per hundred steps |

TRL with vLLM would roughly halve the generation share.

### Hyperparameters to start from

| Setting | Value | Why |
|---|---|---|
| Objective | REINFORCE with a group-mean baseline, no std scaling, loss summed over tokens and divided by a constant | The [Dr. GRPO](https://arxiv.org/abs/2503.20783) form; no length or difficulty bias |
| Updates per rollout batch | **One** | Exactly on-policy: the ratio is 1, so clipping never acts. This removes the clipping bias that makes random rewards help Qwen ([Spurious Rewards](https://arxiv.org/abs/2506.10947)) |
| KL penalty | none; KL to π₀ measured on held-out prompts at each checkpoint | Keeps the update the plain policy gradient |
| Sampling | temperature 1, no top-p or top-k | The sampled distribution is the one the log-probs describe |
| Group size, batch | 8 samples, 64 prompts | Standard; prompts all-right or all-wrong give no signal, so Stage 0's pass rates set a prompt filter |
| Optimizer | AdamW, lr 1e-6, no weight decay, no warm-up; SGD as a later arm | Warm-up would make steps 1-5 near zero |
| LoRA arm, optional | rank 8 on all 196 matrices, lr about 10× | [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/): rank-1 LoRA matches full RL; attention-only LoRA underperforms |
| Steps | about 100-200 | Long enough to move accuracy measurably, hopefully; Stage 1 measures it |

## 4. What is known about RL updates, and the controls

| Property | Finding | What it predicts here | Source |
|---|---|---|---|
| Sparsity | 5-30% of parameters change; the rest are bit-identical in bf16 | Mostly a bf16 effect at our step sizes; report fp32 δW | [Mukherjee et al. 2505.11711](https://arxiv.org/abs/2505.11711), [Path Not Taken](https://arxiv.org/abs/2511.08567) |
| Rank | Nearly full rank in almost every matrix | A low-rank split of δW leaves a lot outside; a full-rank δW tests the subcomponents' span directly | 2505.11711 |
| Which layers | Every matrix, similar sparsity; no layer subset | Spread over all 196 matrices | 2505.11711 |
| Direction | Off the principal directions; spectrum preserved; SFT moves principal weights | If VPD subcomponents favour high-variance directions, expect a large outside share for RL and a smaller one for SFT | Path Not Taken |
| Consistency | Same parameters across seeds, data and algorithms | A second seed should give a δW that overlaps the first more than chance | both |
| KL | RL reaches a task gain at smaller KL to the start than SFT | Expect tiny KL at steps 1-20 | [RL's Razor 2509.04259](https://arxiv.org/abs/2509.04259) |
| Capability | RL raises pass@1, not large-k pass@k | RL strengthens what the policy already samples; a clean "strengthen" update | [Yue et al. 2504.13837](https://arxiv.org/abs/2504.13837) |
| Spurious rewards, Qwen2.5 | Random rewards give 21.4 of the 29.1-point MATH-500 gain on Qwen2.5-Math-7B, through clipping; no gain without clipping, little on Llama and OLMo | Avoided by one update per batch | [Spurious Rewards](https://arxiv.org/abs/2506.10947) |
| Spurious rewards, Qwen3 | Qwen3-8B improves under incorrect rewards; contamination matters less than for Qwen2.5-Math | Nothing reported for Qwen3-1.7B or for non-thinking mode | [2601.11061](https://arxiv.org/abs/2601.11061), abstract only |

### Controls: one loop, three weight rules

Each arm runs the same code. Only the per-rollout weight in the loss Σᵢ wᵢ ∇ log π(oᵢ) changes.

| Arm | Weight wᵢ | What it rules out |
|---|---|---|
| RL | rᵢ minus the group's mean reward | — |
| **Permuted reward** | the same, with rewards shuffled within each group | That any update of this size and shape, with no signal, moves the measures. It keeps the reward distribution and the advantage magnitudes exactly; with one update per batch its expected gradient is zero |
| **Online rejection-sampling SFT** | 1 for a correct rollout, 0 otherwise, no baseline | The matched SFT: the policy's own correct samples, same rollouts and optimizer. It differs from RL only by dropping the negative samples and the baseline |
| SFT on gold solutions, optional | NLL on GSM8K's reference answers | Off-policy SFT as practised; expected to move principal weights and change style |
| W₀ control | no update | Every measure reads zero at δW = 0 |

Match the arms on behavioural effect, the held-out reward gain or the KL to π₀, and report ‖δW‖ beside it. That
follows the 2026-10-14 note and `decisions.md`'s preference for functional matching.

Behaviour readout for small steps: GSM8K accuracy cannot see steps 1-20. One sample per question at 70% has a
standard error near 1.3 points. Use a continuous readout on a fixed held-out set of π₀'s own rollouts, scored in fp32:
the mean change in log-probability of correct minus incorrect rollouts. It is the first-order objective, reads zero at
W₀ and zero in expectation for the permuted arm, and is positive for RL.

## 5. Staged plan

All stages go through `/design` and filed predictions first; this note proposes, it does not approve.

| Stage | What | Decides | Where, cost |
|---|---|---|---|
| 0a, done | Prompt lengths on CPU | Prompts are under 160 tokens at p99 | — |
| 0b | Baseline, no training. GSM8K test in non-thinking mode: greedy, Qwen's recommended sampling, and temperature 1, each scored by strict and lenient extractors, with token-length histograms and the share within 512. Thinking mode on 300 questions, capped at 4096 tokens. Eight temperature-1 samples per question on 2000 train questions for the pass-rate distribution | The mode, the headroom, the format share of the gap, the prompt filter, and the 512-token fit | Under 30 min with HF `generate`, so the cluster's debug QoS first; about $1-2 on a RunPod H100 otherwise |
| 0c | **Gate**: faithfulness of the unchanged decomposition on 0b's non-thinking rollouts against its own firestock3 eval rows. Same checks as the 67M work: stochastic, adversarial and CI-masked reconstruction, sparsity. Also confirm the loader takes fp32 target weights or a δW term | Whether any δW can be scored at all. If W₀ fails on its own rollouts, no map can pass and RL results would be misread | Needs a JAX loader and harness first, which is separate work. Then about 1-1.5 h on an H100 with the 45 GB download: about $3-5 |
| 1 | RL arm, 100-200 steps, δW of the 196 matrices saved at steps 1, 2, 5, 10, 20, 50, 100 and the last; rollouts saved every step; continuous readout and KL every checkpoint; accuracy at 0, 50, 100 and the end | Whether a small RL δW exists with a measurable behaviour change | 2-3.5 h on an H100: about $6-10 |
| 2 | Permuted-reward and rejection-sampling arms with the same seeds, so step 1 shares its rollouts with the RL arm. Then a second seed of the RL arm for the noise floor | The controls and the seed floor | About $6-10 each; gold-solution SFT about $1 |

Total for stages 0b-2: about $20-30 on RunPod at $2.7-3 per hour, inside one $40 session with the 15% margin if 0c's
harness exists. Stages 1 and 2 do not wait for 0c, but none of their δWs can be read until 0c passes.

**Moving δW home.** Seven checkpoints of 2.8 GB each come to about 20 GB per arm. The pod lessons in `CLAUDE.md` make
a plugin download of that size risky. Upload each checkpoint to a W&B artifact from the pod as it is written; the
W&B fetch of `artifacts/vpd` ran at about 45 MB/s. The scoring pod can then pull them directly.

### Decisions for the user

1. **Mode**: non-thinking GSM8K, recommended, or thinking mode with its 512-token problem?
2. **Tooling**: a hand-written loop in the current environment, recommended, or TRL with vLLM in a separate one?
3. **Freeze the embedding and norm gains** so δW lies only in the decomposed matrices? Recommended.
4. **Controls**: permuted reward and online rejection-sampling SFT, recommended; add gold-solution SFT and a LoRA arm?

## Not checked

- No GSM8K number for the post-trained Qwen3-1.7B from Qwen itself; the 65.3 and 87.0 come from one third-party
  greedy evaluation with its own answer extraction.
- The response-length shares within 512 are estimates from word counts and from Qwen3-8B, not measurements.
- How many RL steps move GSM8K accuracy for Qwen3-1.7B non-thinking; no published run of this exact setting was found.
- Spurious-reward behaviour of Qwen3-1.7B; only Qwen3-8B's abstract was read.
- Whether the JAX loader at `177d0cce8` accepts fp32 target weights or a separate δW term.
- What firestock3 contains: whether thinking traces, empty think blocks or GSM8K-like math appear in it.
- Whether Γ uses positional information that would break past 512 tokens; its config was read, its code was not.
- Steps per hour and memory: arithmetic only, no benchmark.
- vLLM 0.11.0's full dependency tree on Python 3.13 beyond numba, xformers, xgrammar and ray wheels; the numpy clash
  was enough to recommend against it.
- The weight-magnitude check covered 12 of the 196 matrices, in layers 0, 13 and 27.
- Mukherjee et al. and Path Not Taken were read at abstract level here; the repo's `lit_review/summaries/` hold fuller
  notes.
