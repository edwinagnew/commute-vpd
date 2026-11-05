# Elicitation and distribution sharpening

*2026-11-05, at the user's request: add Oh et al. 2026 (arXiv 2610.01509) and the relevant references from its
introduction. The user's starting point was that what we call elicitation, that paper calls distribution sharpening,
not a perfect match but definitely relevant. This note proposes no experiment.*

The sharpening debate asks whether post-training changes *which* problems a model can solve or only *how often* it
solves them. We ask whether the fine-tuned behaviour runs through a mechanism the base model already has. These are
different questions: none of the 16 papers here looks inside the model, so none of them tests our definition in
either direction. They matter as the behavioural literature our result will be read against, and because the word
"elicitation" already means something behavioural there.

**How far to trust this note.** I read Oh et al.'s main text myself. Fifteen summaries were written by agents from the
full text (Yue et al. in October, the rest today), and I read their findings and relevance sections, not the papers.
Their quotes were matched against the extracted text by script. Numbers in the table were found in the source text,
except Yue et al.'s, which come from `reports/analyses/2026-10-27_analysis_qwen-gsm8k-rl-headroom.md`. Text came from
PDF extraction, so figures were not seen. The agents were told our definition and asked whether each paper tests it, so
their low scores partly reflect that brief. "No paper looks inside" was checked separately by a keyword scan of the full
texts (probes, SAEs, crosscoders, patching, residual stream, steering vectors, circuits, neurons), which found no such
analysis.

## What each word means

**Our elicitation.** The fine-tuned behaviour runs through a mechanism W₀ already has (`decisions.md`, the row "What
'elicitation' means", user, 2026-11-02). Possession follows Jain et al.'s Definition 1, a linear readout on a sub-domain;
reuse means the post-trained map is g∘f_C; removing the readout direction is the causal label; every verdict is relative
to a named capability C. The user rejected the behavioural alternative, "the trained answer is one W₀ already gives",
because under it post-training would do nothing.

**Sharpening** has at least three meanings in this literature, and the papers move between them.

- *Formal (Huang et al. 2025, Definition 3.1).* A model is sharpened if it puts almost all its mass on the base model's
  own most likely response. This is about the base model's arg-max, and it does not ask whether that response is correct.
- *Coverage (Yue et al. 2025, Oh et al. 2026, and most of the debate).* Single-try accuracy pass@1 rises while pass@k at
  large k stays flat or falls, so the trained model solves no new problems.
- *Mode selection (Zhao et al. 2025, Shao et al. 2026).* Training amplifies an output style or behaviour that was already
  common in the base model's samples, whether or not it is correct.

**Their elicitation.** Oh et al. (§8) use the word through Lambert's "elicitation theory of post-training": "base models
determine the vast majority of the potential of a final model, and post-training's job is to cultivate all of it". Lambert's
[post](https://www.interconnects.ai/p/elicitation-theory-of-post-training) and
[book](https://arxiv.org/abs/2504.12501) were found by search and not read. Oh et al. report that Lambert thinks the view fits
light recipes better than compute-heavy RL. A paper already in our catalogue,
[2605.08368](summaries/arxiv_2605.08368v1.md), defines elicitation as reweighting behaviours the base model can already
reach. So a reader from this literature will take "elicitation" to be the behavioural claim. Anything we write for outside
readers has to say that ours is mechanistic.

## How the two can come apart

My reasoning, not a result: no paper here tests any of it.

- **Sharpening without reuse.** The trained model gives answers the base model already sampled, but through a new route,
  such as new weights that pick out the rare correct answer. Pass@k cannot see this.
- **Reuse without sharpening.** The trained model solves problems the base model never solved, by running a mechanism the
  base model already has. Composition is the natural candidate: Yuan et al. find RL teaches unseen compositions f(g(x)) only
  when the atomic skills exist beforehand, and Zhang et al. find RL fails to transfer to a context seen in under 1% of
  pretraining. Both call this behaviour new or a real gain. Under our definition it could read "reuses C". Neither names a C
  or tests a readout.
- **Both together.** This is the reading the debate tends to assume. It is an inference from behaviour.

**Does this reopen the rejected definition?** I don't think so. Sharpening escapes the literal objection, since a model that
solves a problem one time in a hundred has something to gain from training. It is still a claim about what the model
outputs, so it sits in the family the user rejected, and it says nothing about mechanism.

## The papers

Scores are the agents', out of 10, against the mechanistic question. They are not comparable with the October scores.

| Paper | What it claims | Setting | Score |
|---|---|---|---|
| **Sharpening holds** | | | |
| [Oh et al. 2026](summaries/arxiv_2610.01509.md), *Sharpening Tax* | Extends the trade-off to agentic tasks, where it was least expected. Post-training pushes tasks to always or never solved: on WebShop with gemma-4-31B, tasks that are solved sometimes fall from 87.6% to 30.0%. Tax positive in 36 of 42 cases at k=128. Small models are the exception | 14 base/post-trained pairs, three benchmarks, published checkpoints | 3 |
| [Yue et al. 2025](summaries/arxiv_2504.13837v5.md) | Base catches or passes RL at large k, on GSM8K among others; 24 of the 25 hardest GSM8K problems had a correct base-model solution | Several families; maths, code, vision | 6 (October) |
| [Zhao et al. 2025](summaries/arxiv_2504.07912.md), *Echo Chamber* | RL converges on one output format seen in pretraining, not always the most common or most accurate | Models pretrained from scratch on known mixtures, 150M to 1B | 4 |
| [Wu et al. 2025](summaries/arxiv_2507.14843.md), *Invisible Leash* | Matched question by question, more verified solutions are lost than gained. SFT broadens support where DAPO narrows it on the same base | Qwen2.5-Math-7B and others | 3 |
| [Zhou 2026](summaries/arxiv_2607.20543.md) | Rare correct answers vanish because GRPO groups with no success give no gradient. Pass@256 69.1 to 68.3 while pass@1 rises | Qwen2.5-7B, Omni-MATH | 3 |
| [He et al. 2025](summaries/arxiv_2506.02355.md), *Rewarding the Unlikely* | GRPO reinforces already-likely correct solutions and neglects rare ones; rewarding rare ones raises large-N pass@N | Formal theorem proving | 3 |
| **Training can expand the boundary** | | | |
| [Yuan et al. 2026](summaries/arxiv_2509.25123.md), *f(g(x))* | RL learns unseen compositions of known skills, where supervised fine-tuning on the same data does not | Llama-3.1-8B, synthetic string functions | 4 |
| [Sun et al. 2026](summaries/arxiv_2509.21016.md), *RL Grokking* † | Base has pass@128 = 0 on a novel language; staged RL reaches full pass after an abrupt jump | Qwen3-4B, Manufactoria | 4 |
| [Liu et al. 2025](summaries/arxiv_2505.24864.md), *ProRL* | Long RL solves tasks with base pass@k = 0 up to k=256, such as boxnet; maths tasks still sharpen | Qwen2.5-1.5B, Reasoning Gym | 3 |
| [Shen, M. et al. 2026](summaries/arxiv_2511.00710.md), *Ariadne* | On long mazes the base stays at 0 up to k=1024 and GRPO succeeds; on short mazes it looks like sharpening | Qwen vision models | 3 |
| [Wen et al. 2026](summaries/arxiv_2506.14245.md) | Pass@k credits lucky answers. Counting only correct reasoning, the RL gap persists on hard problems | Math and code | 3 |
| **It depends on data and task** | | | |
| [Shao et al. 2026](summaries/arxiv_2506.10947.md), *Spurious Rewards* | Random rewards give +21.4 points against +29.1 for true rewards on Qwen2.5-Math-7B, attributed to clipping amplifying high-prior behaviour; rarely works on other families | GRPO on MATH-500 | 6 |
| [Zhang et al. 2025](summaries/arxiv_2512.07783.md) | Pass@128 flat on tasks covered by pretraining, up on tasks at the edge of competence | Synthetic tasks, controlled pretraining | 4 |
| [Shen, J. et al. 2026](summaries/arxiv_2607.16097.md) | RL amplifies correct moves on easy positions; tail discovery and wrong-mode amplification grow on hard ones | Chess, 5M to 1B, from scratch | 3 |
| **The theory and the metric** | | | |
| [Huang et al. 2025](summaries/arxiv_2412.01951.md) † | Source of the term: formal limits of sharpening a base policy by sampling | Theory, small experiments | 3 |
| [Dragoi et al. 2025](summaries/arxiv_2510.08325.md) † | Large-k pass@k is misleading on small answer spaces; proposes Cover@τ | Qwen2.5-7B, OMEGA | 2 |

† Not in Oh et al.'s introduction; cited later in the paper and added because the debate leans on them. Yue et al. was already in
the catalogue.

## Where this touches our work

- **Our GSM8K arms sit where the literature predicts sharpening.** Yue et al. report the crossover on GSM8K. A behavioural
  sharpening result there would be unsurprising and would not settle D3. Pass@k on numeric answers also credits lucky guesses
  (Wen, Dragoi).
- **Clipping does not explain our null.** E69 removed clipping to remove Shao et al.'s effect, and expected the shuffled null to stay
  within 1.5 points. On whole answers the null still gained 4.85 and 2.20 points (se 1.14) on its two seeds, and it does not gain
  when an answer past the window counts as wrong (`reports/analyses/2026-11-03_analysis_e69-grpo-loop.md`). E69 read Shao et al. only
  through a summary and differs in model, task, grader and null. The paper's clipping account therefore does not carry over to our
  loop without something more.
- **Some tasks fix what pretraining contained.** Zhao, Zhang and Shen, J. train from scratch on known data, and Yuan teaches atomic
  skills before RL. This is closer to "novelty known by construction" than published checkpoints are. Their novelty is behavioural or
  about data exposure, and none is on Qwen3-1.7B, so none is an anchor as it stands.
- **The closest work that does look inside is already in the review.** [Fixed-SAE Track](summaries/arxiv_2609.15064.md) tracks GRPO on
  Qwen2.5 with one frozen dictionary, finds mostly amplified formatting latents, and reports that steering them recovers much of the format
  gain. Venhoff et al. and Ward et al. (`2026-10-29_baselines_notes.md`, `theory/reveal_vs_create_literature.md`) use steering directions from
  the base model. These are the bridge between the two literatures and they work in activation space, not on the weights.

The gap this leaves: nobody in this debate has asked what the update does to the base model's mechanisms. That claim only covers these 16
papers and what the October sweep found. No search was run on it, so it needs one before it goes in a paper.

## Other citations in Oh et al.'s introduction, not summarised

| Cited | Why it is left out | Read? |
|---|---|---|
| Uesato et al. 2022 ([2211.14275](https://arxiv.org/abs/2211.14275)); Guo et al. 2025 (DeepSeek-R1); Shao et al. 2024 (DeepSeekMath) | Background for the belief that RL creates capability, and for GRPO; no bearing on mechanism | Uesato abstract only; others not read |
| Skalse et al. 2022; Pan et al. 2024 | Reward gaming, cited for lucky rollouts earning credit | Not read |
| Patil 2025; Schick 2023; Oh 2026b; Froger 2026; Zeng 2024; Su 2026 | Agentic benchmarks and skills, cited for what agentic tasks require | Not read |
| Anthropic, OpenAI, Google, Meta, xAI product pages | Cited as examples of agents | Not read |

Cited outside the introduction and possibly worth reading next, not read: Karan and Du 2026 (sampling a base model better, without
training), Gai et al. 2025 ([2511.19942](https://arxiv.org/abs/2511.19942), a proof of why RL collapses diversity, abstract read), Kim et al. 2026
([2606.25447](https://arxiv.org/abs/2606.25447), harness and post-training interact).
