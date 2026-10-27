# Is reinforcement learning from 0.75 on GSM8K a meaningful task for Qwen3-1.7B, and is 0.75 inflated by contamination?

*2026-10-27, research agent, read-only. Answers the user's question: "i dont have enough RL experience to know whether
improving from 0.75 is a meaningful RL task - can you look into this? if that is already too high, do you think its a
capability problem or a training data contamination problem (which i believe qwen is known for)". Nothing was run.
Inputs: `reports/analyses/2026-10-27_analysis_qwen-gsm8k-baseline.md`, `reports/analyses/2026-10-27_analysis_qwen-rl-gsm8k.md`,
`experiments/qwen_gsm8k_baseline/results/`, and the papers in the sources list. Words as in `GOALS.md` → Terms.*

We read published reinforcement learning runs on school maths for small models, and the evidence that Qwen models have
memorised test questions. Such runs mostly make a model reliable at what it can already do, and memorisation is shown
for competition maths, not for GSM8K or this model. So GSM8K suits this project as a clean case of strengthening an
existing behaviour, once one cheap fresh-numbers evaluation rules memory out.

## The answer

Starting from three in four is normal for reinforcement learning on GSM8K at this size. Published runs
gain a few to about fifteen points. Those gains come mostly from format, from sampling the right answer more reliably,
and in Qwen3 from its own thinking mode. They rarely add new skill. Contamination evidence is strong for Qwen2.5 on
competition maths, weak for GSM8K's test set, and not measured for this model. Our score looks like capability plus
format losses; one cheap fresh-numbers evaluation would confirm it.

## Recommendation

- Keep GSM8K as the first reinforcement learning task. Read it as a strengthening update, which is the case the
  project most needs a clean instance of, and say so in the plan.
- Before training, run one greedy evaluation of the unchanged model on fresh-number versions of GSM8K test problems,
  described below. It costs one short cluster job.
- Score every update on the fresh-number set beside GSM8K test. If the two gains differ, part of the GSM8K gain is
  memory, not reasoning.
- Watch answer length from the first steps. The likeliest way reinforcement learning raises non-thinking accuracy in
  Qwen3 is longer, thinking-style answers, which would push rollouts past the 512-token window.

A wording mismatch to note: the brief says the prompt ends with Qwen's `\boxed{}` suffix, but the baseline run used
"The answer is N." as its final line, per `baseline.json` → `instruction`.

## 1. Is improving from 0.75 a meaningful reinforcement learning task?

### What published runs start from and reach

The closest published runs train instruction-tuned Qwen2.5 models of half a billion to three billion parameters on
GSM8K's training split. No peer-reviewed run of Qwen3-1.7B in non-thinking mode on GSM8K was found.

| model | start | after RL | method, steps | source |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 49.6 | 56.7 PPO; 54.3 GRPO-LoRA; 58.7 PRIME | GSM8K train | verl baselines page; start from the Qwen2.5 report |
| Qwen2.5-1.5B-Instruct | 73.2 | 77.9 | GRPO-LoRA, GSM8K train | verl baselines; start from [Qwen2.5 report](https://arxiv.org/abs/2412.15115) Table 10 |
| Qwen2.5-1.5B-Instruct, strict format scorer | 62.1 | 76.8 | GRPO, GSM8K train | community repo, not peer-reviewed |
| Qwen2.5-3B-Instruct | 86.7 | 86.1 | GRPO-LoRA | verl baselines; start from Qwen2.5 report Table 9 |
| Qwen2.5-7B-Instruct | 91.6 | 93.4 | GRPO-LoRA | verl baselines; start from Qwen2.5 report Table 8 |
| Qwen2.5-0.5B base | 36.7 | 49.5 | GRPO on 8k MATH problems | [SimpleRL-Zoo](https://arxiv.org/abs/2503.18892) Table 1 |
| Qwen2.5-1.5B base | 55.7 | 74.4 | same | same |
| Qwen2.5-7B base | 88.2 | 91.7 | same | same |
| Llama-3.1-8B base | 39.7 | 79.2 | same | same |
| 1B model pretrained from scratch | — | about 60 | PPO on GSM8K train; collapses to one output format within the first epoch | [Echo Chamber](https://arxiv.org/abs/2504.07912) §3.1, App. F.2 |

Caveats on the table. The verl starts are the Qwen report's numbers and its finals use verl's own strict `####`
extraction, so each row mixes two evaluations. SimpleRL-Zoo scores its bases greedily and its trained models by
sampling. None of these sources reports how many steps the gain needed in a form I could extract.

**Reading it.** A start near 0.75 is the ordinary regime for a model of this size, and gains of a few to about fifteen
points are typical. The gain shrinks as the start rises: the 3B instruct model gained nothing and the 7B under two
points. Our model sits between the 1.5B and the 3B instruct models, so a gain of roughly five to ten points is the
plausible range. The Spurious Rewards paper adds that Qwen instruct models already post-trained with reinforcement
learning show minimal gains under any reward, true reward included, in its Appendix J, Figure 20. Qwen3-1.7B was
post-trained by distillation, not reinforcement learning, so it may sit between those cases.

### How much of the gain is format, sharpening, or new skill

| finding | evidence | source |
|---|---|---|
| Reinforcement learning raises single-sample accuracy but not the set of problems solvable in many samples | at large sample counts the base model matches or beats the trained one on GSM8K and five other maths sets; 24 of the 25 hardest GSM8K problems had a correct base-model solution | [Yue et al.](https://arxiv.org/abs/2504.13837) §4, lit note `lit_review/summaries/arxiv_2504.13837v5.md` |
| The main effect is reliability on problems the model can already solve | GRPO mostly improves execution, called "temperature distillation"; new problems hit a "coverage wall" | [Qin et al.](https://arxiv.org/abs/2505.22756), abstract |
| Reinforcement learning amplifies one existing output style | small models trained with PPO on GSM8K converge on one pretraining format within the first epoch; majority vote rises about 5 points while many-sample coverage falls slightly | Echo Chamber §3.1 |
| In Qwen3, non-thinking reinforcement learning borrows the thinking mode | Qwen3-8B non-thinking AIME accuracy went from 0.17 to 0.43; Qwen3-1.7B answers grew from 2.5k to 4.7k tokens; swapping an 8-dimensional "think" subspace removed about 75% of the 8B's gain | [Thinking Leakage](https://arxiv.org/abs/2610.14270), Fig. 1a and §4 |
| New skill does appear, but where the start is weak | gains beyond the base model's reach came on tasks where the base nearly always failed; on maths, with a high start, many-sample coverage often fell | [ProRL](https://arxiv.org/abs/2505.24864) §4.1–4.2 |
| Strict format rewards cost accuracy | a rigid format penalty diverted effort into format and collapsed some runs | SimpleRL-Zoo §3.1 |

Our own baseline bounds what sharpening alone can reach. On the sampled prompts, mean sample accuracy was 0.753.
Of those 224 prompts, 203 were solved at least once in eight samples. So a policy that always gave its best existing answer would
score about 0.91 on those prompts. The thinking mode of the same model scores 0.87 to 0.90 on GSM8K in outside
evaluations, which is a second ceiling of the same kind.

| quantity, unchanged Qwen3-1.7B | value | source |
|---|---|---|
| greedy strict accuracy, all 1,319 | 0.756 | our baseline |
| greedy lenient accuracy | 0.783 | our baseline |
| prompts solved at least once in 8 samples | 203 of 224, 0.906 | our baseline, `samples` |
| prompts with mixed right and wrong samples | 0.39 strict, 0.31 lenient | our baseline |
| thinking mode, greedy, 8k cap | 0.870 | [2510.07880](https://arxiv.org/abs/2510.07880) Table A4 |
| thinking mode, sampled, 32k cap | 0.904 | [SwiReasoning](https://arxiv.org/abs/2510.05069) Table 1 |
| test questions frontier models cannot score because the label or question is flawed | about 3%: 110 of 1,319 removed, 10 relabelled; one frontier model's errors fell from 45 to 2 | [GSM8K-Platinum](https://gradientscience.org/gsm8k-platinum/) |

**Reading it.** Yes, it is a meaningful task, but of one kind. The gap from 0.75 to about 0.9 is almost all problems
the model already solves sometimes or solves in thinking mode. Reinforcement learning closes that kind of gap well and
fast. It rarely adds problems the model never solves, and about one prompt in eleven is never solved in 8 samples.
So an RL update on this task is, by the literature's best reading, a strengthening or eliciting update. A fifth of
the early learning signal is format, since the mixed share drops from 0.39 to 0.31 under the lenient grader.

**How it bears on the project.** `GOALS.md` asks whether a failure to commute an update comes from post-training
teaching new capability or from brittle decompositions. A GSM8K update is the strengthening case. If its update can be
split into the subcomponents with little left outside, that is the expected result and a working positive case. If
it cannot, the "new capability" explanation is weak for this update, which points at the decomposition. That
reading depends on the literature's claim holding for this model, which our own runs have not tested. A contrast arm
where the model must learn something new needs a different task, ranked in section 3.

## 2. Is 0.75 inflated by contamination?

### What the evidence shows, by model and benchmark

| test | models | benchmark | result | source |
|---|---|---|---|---|
| Continue a test question from its first 60% | Qwen2.5-Math-7B; Qwen2.5-7B; Llama-3.1-8B | MATH-500 | exact completion 54.6%; 21.2%; 3.8% | [Wu et al.](https://arxiv.org/abs/2507.10532) Table 2 |
| same | Qwen3-4B-Base; Qwen3-8B-Base | MATH-500 | 21.8%; 32.0% | Wu et al. Table 9 |
| same | post-trained Qwen3-4B; Qwen3-8B | MATH-500 | 5.2%; 5.8% | Wu et al. Table 9 |
| same | all Qwen3 above | AIME 2025, released after the models | 0% | Wu et al. Table 9 |
| same | any model | GSM8K | not tested | Wu et al. |
| Random or wrong rewards still raise accuracy | Qwen2.5-Math-7B | MATH-500 | +21.4 random, +24.1 wrong, +29.1 true reward; nothing without PPO clipping; little for Llama and OLMo | [Spurious Rewards](https://arxiv.org/abs/2506.10947) §2.2, §4 |
| same | Qwen3-8B | six maths sets, not GSM8K | gains under wrong rewards, smaller than Qwen2.5-Math-7B's; numbers only in a figure | [2601.11061](https://arxiv.org/abs/2601.11061) App. C, Fig. 11 |
| same | R1-Distill-Llama-8B, QwQ-32B | MATH-500 | spurious-reward gains outside Qwen-Math too; 65.6 to 76.6 for the Llama distill | [2512.16912](https://arxiv.org/abs/2512.16912) |
| Seen-data detector, score above 80 means seen | Qwen2.5-1.5B-Instruct | GSM8K test; train | 27; 100 | [CoDeC](https://arxiv.org/abs/2510.27055) Table 3 |
| same | Qwen3-8B; Qwen3-14B | GSM8K test; train | 14 and 21; 28 and 54 | same |
| Original GSM8K question against fresh-value versions, English | Qwen3-1.7B non-thinking | 100 GSM8K test questions; 2,000 variants | 92.0, CI 85.0–95.9; 83.1, CI 81.4–84.7 | [Multilingual GSM-Symbolic](https://arxiv.org/abs/2610.03367) Table 12 |
| same | Qwen3-1.7B thinking; Qwen3-4B non-thinking; Qwen2.5-3B-Instruct | same | 92.0 vs 90.3; 95.0 vs 92.8; 81.0 vs 81.5 | same |
| same, pooled over all models | 48 models | English | gap 1.7 points, CI −0.6 to 4.0 | same, §4.3 |
| Fresh hand-written GSM8K-style set | 30+ models, no Qwen | GSM1k | Phi and Mistral overfit by up to 8 points; frontier models no gap | [GSM1k](https://arxiv.org/abs/2405.00332) §5 |
| Contamination planted in pretraining | Qwen2.5-0.5B and 1.5B, from base | GSM8K with 5 test copies | invisible in the base model; SFT or GRPO then reveal a 2 to 4 point inflation; GRPO also helps the clean GSM-Plus | [2601.06103](https://arxiv.org/abs/2601.06103) §4, Fig. 3–5 |
| Qwen's stated decontamination | Qwen2.5 | all evaluation sets | training text dropped if it shares a 13-token, 60% overlap with a test item | Qwen2.5 report §5 |

What I could not find: any GSM8K memorisation test on Qwen3-1.7B or its base; any Qwen row in GSM1k or the original
GSM-Symbolic paper; a decontamination statement in the parts of the Qwen3 report I read. The DCR paper,
[2507.11405](https://arxiv.org/abs/2507.11405) Table 2, reports 36% "contamination" for Qwen2.5-7B on GSM8K, but its
raw accuracies sit thirty points under Qwen's own, so I give it no weight. The earlier scoping report attributed a
Qwen3-8B spurious-reward result to 2601.11061's abstract; the abstract names only Qwen2.5, and the Qwen3-8B result is in
its appendix.

**Reading it.** The strong evidence is about Qwen2.5, and above all Qwen2.5-Math, on competition maths. There a
Qwen model completes more than half of MATH-500 questions word for word from a 60% prefix. For GSM8K the evidence is
different. The detector finds Qwen2.5 has seen GSM8K's training split but does not flag the test split for any Qwen
model. The only measurement on our model shows a 9-point advantage on the 100 original questions over fresh-value
versions. That is above the 1.7-point average for all models, but it rests on 100 questions and comes from another
prompt. Fresh values can also change difficulty. So memorisation of GSM8K test is possible for Qwen3-1.7B, of modest
size if real, and unmeasured.

### Capability, format, or memorisation?

Our 0.756 is not suspiciously high. It equals the base model's 4-shot score of 75.4, from the Qwen3 report's Table 8. It
sits below the same model's 83.1 on fresh-value GSM-Symbolic problems under another prompt, and well below its thinking
mode. Memorisation inflates scores, so a contaminated model would look better on GSM8K than on fresh problems, not
worse. The quarter it gets wrong breaks down roughly as follows, from our baseline:

| part of the 0.244 wrong | size | reading |
|---|---|---|
| right number in the wrong format | 0.027 | format; the lenient grader recovers it |
| truncated at 512 new tokens | up to 0.012 | length cap |
| flawed question or label | about 0.03 | unscorable, per GSM8K-Platinum |
| the rest | about 0.17 | reasoning errors; most on problems the model solves in some samples |

So the honest reading is: mostly capability, solved inconsistently, plus a few points of format. Memorisation, if
present, would mean the true capability on fresh problems is a few points lower than 0.756, not that the task has no
headroom. The risk it poses is different. RL on a contaminated set can raise test accuracy by recalling stored answers,
which 2601.06103 saw appear only after post-training. That would make the update look like strengthening for a reason
the project does not care about.

### The one cheap check

**Fresh-number evaluation.** Run the unchanged model greedily, with our exact prompt, grader and 512-token cap, on two
sets:

- GSM-Plus's "numerical substitution" variant: one version of every GSM8K test problem with changed numbers, 1,319
  items paired with our existing greedy answers.
- New instances generated from the 100 GSM-Symbolic templates with a fresh random seed, 20 per template. These have
  never been published, so they cannot be in any training set; the released instances could be.

About 3,300 generations, roughly 15 minutes on one L40 at the baseline's 37 seconds per batch of 128, so the cluster's
debug QoS at no cost. The paired design gives a standard error near one point on the GSM-Plus difference.

| reading | GSM8K minus GSM-Plus, paired | 100 originals minus fresh instances | strict minus lenient gap |
|---|---|---|---|
| capability ceiling, no memory | within about 2 points | within about 3 points, the all-model English gap | the same 0.03 on both sets |
| memorisation | 5 points or more, concentrated on items answered with little working | 8 points or more | the same on both |
| format loss | small | small | about 0.03 on both, and the lenient score near 0.78 on both |

A gap between 2 and 5 points is the ambiguous zone: it could be memory or the new numbers being harder. In that case,
add the prefix test of Wu et al. on Qwen3-1.7B-Base: complete each GSM8K test question from its first 60%. GSM8K
train questions are the positive control, since Qwen has seen them, and the fresh instances are the floor. That is
three short sets of 64-token generations, a few minutes on the same card.

## 3. Cleaner tasks, if GSM8K turns out weak

Ranked for this project: a measurable gain in about a hundred steps, a held-out score memory cannot inflate, answers
inside the 512-token window, and an exact reward.

| rank | option | headroom | contamination risk | fits 512 tokens | verifiable |
|---|---|---|---|---|---|
| 1 | GSM8K train for RL, scored on fresh GSM-Symbolic instances and GSM-Plus beside GSM8K test | same as now, about 0.75 to 0.9 | low on fresh instances; templates are GSM8K test questions, so template memory remains | yes, same lengths as now | yes, integer answers |
| 2 | Synthetic multi-step arithmetic in words or symbols, chain length tuned so the start is about one half | set by chain length | none, generated per run | yes at 5–10 steps, to be measured | exact |
| 3 | Harder GSM-Symbolic variants with one or two added clauses, fresh instances, split by template | large: other small models fell to 1–63% on these | as rank 1 | the two-clause variant may not | yes |
| 4 | MATH levels 1–3 | moderate; non-thinking MATH-500 is 0.73 | highest: MATH-500 is the set with word-for-word recall in Qwen2.5 and Qwen3 base | doubtful; answers run longer | needs a symbolic answer checker |

Notes on each:

- **Rank 1** changes nothing in the RL loop. It only adds a held-out score that memory cannot help, which is what the
  contamination question needs. It does not add a new-capability contrast.
- **Rank 2** follows the RandomCalculation set of Wu et al., §4.3. There, true reward raised Qwen2.5-Math-7B above its
  own best-of-16 score while random reward gave only unstable, marginal gains, so the controls separate cleanly. Tuning
  chain length moves the task from mostly-solved, a strengthening update, to rarely-solved, closer to new skill. That
  makes it the best candidate for a matched contrast arm. Its cost: the decomposition was fitted on chat text, and its
  faithfulness on arithmetic strings is unknown.
- **Rank 3** has the most headroom of the natural-language options. But GSM-Symbolic has only 100 base templates,
  fewer for the harder variants, so a train and test split by template is small.
- **Rank 4** is the weakest on every count for this model.

## Not run

- The fresh-number evaluation and the prefix test proposed above.
- Any check that the 100 GSM-Symbolic source questions are easier for our model than the full test set; GSM-Symbolic's
  own table shows them 0 to 5 points easier for other models.
- Reading full tables for: SimpleRL-Zoo step counts, Spurious Rewards Appendix J values, 2601.11061 Figure 11 values,
  Qin et al.'s numbers, ProRL's Figure 3. Their text gives only qualitative claims, which is what I used.
- The Qwen3 technical report's full text for a decontamination statement; I read about three quarters of it.
- Whether GSM1k was ever released publicly; its paper planned a release by June 2025.
- Any published GSM8K reinforcement learning run on Qwen3-1.7B in non-thinking mode: none found.
- Peer-reviewed step counts for GSM8K gains at this scale.

## Sources

- Qwen3 technical report, arXiv 2505.09388: Table 8 base GSM8K, §4.5 distillation for small models.
- Qwen2.5 technical report, arXiv 2412.15115: Tables 8–10 instruct GSM8K, §5 decontamination rule.
- verl algorithm baselines, verl.readthedocs.io/en/latest/algo/baseline.html: GSM8K RL rows for Qwen2.5 instruct models.
- Community GRPO run, github.com/tahaBerkBeton/improving-a-small-slm-on-gsm8k: not peer-reviewed.
- SimpleRL-Zoo, arXiv 2503.18892: Table 1, §3.1.
- Echo Chamber, arXiv 2504.07912: §3.1, §5, App. F.2.
- Yue et al., arXiv 2504.13837: §4; repo note `lit_review/summaries/arxiv_2504.13837v5.md`.
- Qin et al., arXiv 2505.22756: abstract.
- ProRL, arXiv 2505.24864: §4.1–4.2.
- The Thinking Leakage in RL for LLMs, arXiv 2610.14270: Fig. 1a, §4.
- Spurious Rewards, arXiv 2506.10947: §2.2, §4, §5, App. J.
- Reasoning or Memorization, Wu et al., arXiv 2507.10532: Tables 2 and 9, §4.3, App. E.
- Spurious Rewards Paradox, arXiv 2601.11061: App. C, Fig. 11.
- Exploration vs Exploitation, arXiv 2512.16912.
- CoDeC, arXiv 2510.27055: Table 3, App. C.7.
- Multilingual GSM-Symbolic, arXiv 2610.03367: Tables 11–12, §3.5, §4.3.
- GSM-Symbolic, arXiv 2410.05229: Table 1.
- GSM1k, arXiv 2405.00332: §3, §5.
- GSM-Plus, arXiv 2402.19255: 10,552 variants of the 1,319 test questions, eight perturbation types.
- The Impact of Post-training on Data Contamination, arXiv 2601.06103: §3–4, Fig. 3–6.
- Benchmarking Benchmark Leakage, arXiv 2404.18824: §5.3, first-generation Qwen-1.8B recites 223 GSM8K training items.
- DCR, arXiv 2507.11405: Table 2, given no weight.
- GSM8K-Platinum, gradientscience.org/gsm8k-platinum.
- 2510.07880 Table A4 and SwiReasoning 2510.05069 Table 1: thinking-mode GSM8K for Qwen3-1.7B.
