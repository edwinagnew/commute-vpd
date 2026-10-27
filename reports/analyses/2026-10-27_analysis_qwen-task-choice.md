# Which post-training task should the Qwen3-1.7B decomposition be tested on? (analysis, 2026-10-27)

*Read-only scoping for the lead. No code was run and no other file was changed. Sources: `README.md`, `GOALS.md`,
`decisions.md`, `experiments/TRIED.md` (rows E5, E10, E37), `theory/reveal_vs_create_literature.md`, the two sibling
reports `reports/analyses/2026-10-27_analysis_qwen-findings-to-port.md` (section 3, candidate tasks) and
`reports/analyses/2026-10-27_analysis_qwen-rl-gsm8k.md` (sections 1, 2 and 4), and web searches for papers, datasets and
model cards (list at the end). Words as in `GOALS.md` → Terms.*

**In plain words.** I compared nine kinds of post-training task for testing what one small update does to the
Qwen3-1.7B decomposition, against the user's constraints: real headroom, short chat text, both a fine-tuning arm and a
reinforcement-learning arm with a cheap automatic score, and a measurement that moves after one step. Two tasks beat
maths on safety relevance and on separating "strengthen" from "teach new": closed-book questions where the model is
rewarded for saying "I don't know" instead of guessing, and refusing jailbreak-wrapped harmful requests without
refusing harmless ones. Maths (GSM8K without thinking) stays a good third, because it is where the debate about whether
reinforcement learning only re-weights what a model already does was fought.

## The answer

Rank first **closed-book question answering with abstention**: answer if you know, say "I don't know" otherwise. It is
short, scored by string match, has a single decision at the first answer token, and labels each training fact as
known or unknown to the model before training, so one update holds both "strengthen" and "teach new" examples. Second,
**refusal under jailbreak wrapping** (WildJailbreak), the most direct safety task, scored by a 0.6B guard model.
Third, **GSM8K without thinking**, already scoped by the sibling report. All three baselines fit in one short GPU job.

## 0. Design check (`/design`, concise)

- **Goal served.** `GOALS.md`'s "if no simple map works, is it because post-training teaches new capabilities rather
  than strengthening existing ones": a task is chosen so that one update, or a pair of updates (SFT and RL), differs on
  exactly that axis, on the larger model the decision of 2026-10-26 moved the commute line to.
- **Nearest tried.** E10 (key → value cipher: the testbed's "teach new", random word pairs the model lacks) and E5/E28
  (tasks the model already does). Unknown real facts are the realistic analogue of E10's pairs, known facts of E5's
  tasks. E37 (lowercase fine-tune, broad) is the nearest "every position" update: GSM8K and instruction following are
  broad like it; abstention, refusal and sycophancy have a decision position like E5.
- **Patch or step.** A step: the task choice for the new phase.
- **Decisions relied on.** Functional matching of arms (behavioural effect, not ‖δW‖); faithfulness scored on the
  task's text as well as generic text (`decisions.md` row of 2026-10-15). No departure.
- **What would change plans.** The baseline job below: a task with under about 15 points of headroom, under about 30%
  of prompts with mixed rewards across 8 samples, or under about 85% of examples inside 512 tokens is dropped.
- **Control first.** For every task the first GPU pass is the baseline at W₀ (no update), and before any update is
  read, the sibling report's gate V5: the unchanged decomposition must be about as faithful on the task's text as on
  generic chat text.
- **Class.** A new task family and dataset on a new model: outside the approved classes, so it needs the user's
  confirmation after they have seen this check. Nothing here starts work.

## 1. What the project needs from a task, beyond the user's list

The user's constraints (headroom, both arms, cheap reward, a one-step measurement, cheap held-out evaluation, public
licence, at most 512 tokens) are the filter. Four more properties decide between tasks that pass it.

1. **The task's text is inside the decomposition's training distribution.** It was fit on chat-formatted text
   (`firestock3`, contents not shipped). Ordinary chat turns (questions, refusals, multiple choice) are the safest bet;
   maths and code are probably present but unverified. Gate V5 measures this per task before anything else.
2. **A decision position.** When the behaviour is decided at one token (the first token of a refusal, an answer letter,
   "I" of "I don't know"), the testbed's position-level readings port with little new code: target probability at the
   position, which subcomponents Γ switches on there, the different-target null (findings-to-port rows 1, 10, 11, 17).
   A single step's effect is also easiest to see there, as a change in one log-odds on a fixed prompt set. Broad tasks
   (GSM8K, instruction following) need the sequence-level score *s* that E37 introduced.
3. **An independent account of the mechanism, as a positive control.** If interpretability work outside VPD has
   already located the behaviour (refusal along one direction, Arditi et al. 2024; verbal uncertainty as a linear
   feature, Ji et al. 2025; a sycophancy persona vector, Chen et al. 2025), the decomposition's reading of an update
   can be checked against it. GSM8K and instruction following have none.
4. **SFT and RL separable on "strengthen against teach new".** SFT and RL differ in three ways at once: on-policy
   against off-policy text, the use of negative samples, and the reward. The sibling RL report's online
   rejection-sampling SFT (train on the model's own correct samples) removes the first difference; SFT on a stronger
   model's samples keeps it. "Teach new" lives mainly in the off-policy text. A task is better when novelty can also be
   labelled **per example** before training: W₀'s log-probability of each SFT target, or, for facts, whether W₀ ever
   produces the answer. Then one update can be split into its re-weighting part and its new part.

One general point. Qwen3-1.7B was itself post-trained by distillation from larger Qwen3 models (sibling RL report,
section 0). SFT on Qwen3-8B samples therefore repeats the model's own last training stage and is the mildest possible
"off-policy" arm; a non-Qwen source (dataset gold, or another family's samples) is the stronger "teach new" arm.

## 2. Candidates

Headroom numbers are for Qwen/Qwen3-1.7B (post-trained) in non-thinking mode unless stated. "Fit" is the share of
prompt plus response within 512 tokens under the Qwen3 chat template; only GSM8K's prompt side has been measured (by
the sibling report), every other fit is my estimate from typical lengths.

### Summary

| Task | Headroom, Qwen3-1.7B | Fit in 512 | Decision position | Reward (cost) | SFT source | Separates strengthen / new | Safety relevance |
|---|---|---|---|---|---|---|---|
| **Closed-book QA with abstention** | not published; expected large (small models guess) | ~100% | first answer token | string match, free | gold answers; "I don't know" for unknown items | **yes, per example** (known vs unknown facts) | honesty, hallucination |
| **Refusal under jailbreak wrapping** | direct harmful 18% complied; wrapped 89% complied; over-refusal 7% (one paper) | ~60-85% whole; ~95% for the decision | first response tokens | Qwen3Guard-Gen-0.6B (cheap) | WildJailbreak completions, or self with a safety system prompt | partly (RL and SFT both strengthen) | direct: safety training itself |
| **GSM8K, non-thinking** | 65.3% greedy | ~97-99% | none (final number only) | regex, free | gold, Qwen3-8B, or own correct samples | yes, by arm (on- vs off-policy), not per example | indirect |
| Sycophancy under pushback | Qwen3 family flips 17% (family average) | ~85-95% | answer letter | letter match, free | correct answer held; concede when user is right | weak (both strengthen) | high |
| Verifiable instruction following | IFEval 68.2; Multi-IF 44.7 | ~70-90% | none | Python checkers, free | Tülu 3 IF data, Qwen3-8B filtered | partly (new constraint types) | low-medium (spec gaming) |
| MATH levels 1-3 | MATH-500 73.0 overall | ~85-95% (est.) | none | regex / sympy, free | gold, Qwen3-8B | as GSM8K | indirect |
| Emergent misalignment (narrow SFT) | EM shown at 0.5-1B in other families | ~95% (advice data) | none | LLM judge | released narrow datasets | unclear | very high, but SFT only |
| Reward hacking, special-cased tests | LiveCodeBench 11.6: weak coder | ~50-80% | none | unit tests, free | none natural | — | high, but impractical at 1.7B |
| TruthfulQA MC / persona traits | not checked | ~100% | letter / none | letter match; or judge | small (817 items); judge-made | weak | medium |

### Closed-book QA with abstention (recommended first)

- **What one update teaches.** Whether to answer or abstain on a factual question, and, in the gold-answer SFT arm, the
  answer to facts the model did not know.
- **Strengthen or new, and the SFT/RL split.** Before training, sample W₀ 8 times per question and label each fact
  known (often right), maybe known (sometimes right) or unknown (never right), the SliCK categories of Gekhman et al.
  (EMNLP 2024). They found that unknown facts are learned much more slowly by SFT and, once learned, raise
  hallucination roughly linearly, while known facts are fitted fast. That is "teach new" against "strengthen" inside
  one task, one format and one update, labelled per example. Three arms then separate cleanly: SFT on gold answers
  (unknown items are new knowledge); SFT or RL that teaches abstention on unknown items (R-Tuning, Zhang et al. 2023,
  for SFT; TruthRL, Wei et al. 2025, for GRPO with +1 correct, 0 abstain, −1 wrong); RL cannot add facts, only re-weight
  answering against abstaining. The model's knowledge of what it knows is an existing capability (Kadavath et al. 2022),
  so abstention training is "strengthen" by construction. TruthRL reports that SFT and binary-reward RL tend to keep
  guessing while the ternary reward cuts hallucination, so SFT and RL are expected to differ in behaviour too.
- **Headroom.** No Qwen3-1.7B number is published (searched: TriviaQA, SimpleQA, PopQA). SimpleQA is hard even for
  frontier models (GPT-4o under 40%), so a 1.7B model will be near the floor there and mostly guess; TriviaQA should give
  a usable mix of known and unknown items. The baseline measures it.
- **Fit.** Question about 20-60 tokens, answer about 5-40: about 100%.
- **Data and reward.** TriviaQA (Apache-2.0), EntityQuestions, PopQA; SimpleQA (MIT) as a held-out, hard test. Reward:
  answer-alias match plus an abstention string match, free. SFT targets: gold answers; "I don't know" on unknown items.
- **One-step measurement.** On a fixed held-out set, the change in log-odds of abstaining against answering at the first
  answer token, read separately on known and unknown items; in the gold arm, the change in log-probability of the gold
  answer on unknown items. Both read zero at W₀ and are single-token quantities.
- **Prompted reference.** W₀ asked in a system prompt to abstain when unsure shows the behaviour without training. That
  is the findings-to-port report's "make a prompted behaviour the default" design (its Q2) on a realistic behaviour:
  the prompted W₀ is the reference for which subcomponents Γ uses for abstention.
- **Safety relevance.** Honesty: a model that says what it does not know instead of making things up.
- **Risks.** Alias matching misses some correct free-form answers (a few per cent; force short answers). Abstention can
  collapse to "always abstain"; the known items' accuracy is the guard and must be reported beside it. The ternary
  reward values are a choice. "Unknown" depends on sampling 8 times at one temperature.

### Refusal under jailbreak wrapping, against over-refusal (recommended second)

- **What one update teaches.** To refuse a harmful request even when it is disguised (role-play, fiction, nested
  instructions), while still answering harmless prompts that look similar.
- **Strengthen or new.** Strengthen: the model already refuses most direct harmful requests, and refusal is known to
  run along one direction in chat models (Arditi et al. 2024), which gives the positive control of section 1. SFT and RL
  both strengthen here, so the pair separates less than in QA. A partial separation: SFT on GPT-3.5's refusal text
  (WildJailbreak's completions) is off-policy in style, while SFT on Qwen3-1.7B's own refusals under a safety system
  prompt (context distillation) is on-policy.
- **Headroom.** One paper using Qwen3-1.7B (2610.11005, its own four-way judge on deepseek-chat): bare harmful requests
  succeed 18% of the time, narrative-wrapped ones 89%, and benign look-alikes are refused 6.9% of the time. So direct
  refusal has little headroom (the findings-to-port report's refusal row should use wrapped prompts), wrapped refusal
  has a lot, and over-refusal is low enough that a two-sided reward is needed to stop the model refusing everything.
  This is one source, with one judge; the baseline must re-measure it.
- **Fit.** Adversarial prompts are long (often 100-400 tokens), refusals short, compliant answers long. I estimate
  60-85% of whole examples fit; the decision sits in the first few response tokens, so about 95% of prompts can be
  scored for it if responses are cut at 512.
- **Data and reward.** WildJailbreak (ODC-BY, gated by a click-through form): 262k training pairs in four types
  (vanilla and adversarial, harmful and benign, with completions), and an evaluation set of 2,000 adversarial harmful and
  210 adversarial benign prompts. XSTest (250 safe, 200 unsafe) and OR-Bench-Hard for over-refusal; StrongREJECT (313)
  and HarmBench for harm. Reward: Qwen3Guard-Gen-0.6B (Apache-2.0), whose response-moderation output carries a
  "Refusal: Yes/No" field and a safe/controversial/unsafe label; reward refusal on harmful prompts and compliance on
  benign ones. Its prefill of prompt plus response costs less than a third of the policy's generation compute. A
  string-match refusal detector is the free fallback and the cross-check.
- **One-step measurement.** The change in log-odds of a refusal opening ("I can't", "I'm sorry") against a compliance
  opening at the first response token, on fixed held-out adversarial harmful and adversarial benign prompts.
- **Safety relevance.** This is safety training itself, the post-training stage a monitoring tool would most need to
  read.
- **Risks.** Guard noise is unmeasured for the 0.6B model (its card gives no accuracy number). Rollouts and evaluation
  contain harmful text; the gated dataset needs an account and acceptance of AI2's terms (who accepts is the user's
  call). Over-refusal readings on 210 benign evaluation prompts are noisy (one sample per prompt gives about ±2 points
  at 7%).

### GSM8K, non-thinking (recommended third; scoped in the sibling report)

The sibling report (`2026-10-27_analysis_qwen-rl-gsm8k.md`) covers headroom (65.3% greedy non-thinking, 87-90%
thinking), the 512-token fit (prompts measured: p99 155 tokens; 97-99% of examples expected to fit non-thinking, 10-35%
thinking), the bf16 rounding trap, controls (permuted reward, online rejection-sampling SFT, gold SFT) and the
continuous readout (log-probability of correct minus incorrect W₀ rollouts). I agree with all of it and add three
points.

- **Why keep it.** The re-weighting debate was argued on RL for maths (Yue et al. 2025: RL raises pass@1 but not
  large-k pass@k; Spurious Rewards on Qwen2.5-Math). Results on GSM8K speak directly to it; results on refusal or
  abstention speak to it only by analogy.
- **The SFT source.** The user asked for SFT on known-correct solutions, e.g. from a stronger model. I would run two:
  Qwen3-8B non-thinking samples filtered for correctness (93.1% at a 512-token cap, close to Qwen3-1.7B's own style and
  training lineage), and GSM8K's gold solutions with the calculator tags removed (terse, non-Qwen style). The first is
  the mild off-policy arm, the second the strong one. The sibling's online rejection-sampling SFT is the on-policy arm.
- **What it lacks.** No decision position, no independent mechanism to compare with, no per-example novelty label
  beyond W₀'s log-probability of the target, and an indirect safety link. A safety-flavoured variant exists at no cost:
  RL against a deliberately lenient grader (any boxed number anywhere counts) to see whether one step already moves the
  model toward gaming it. I would not start there.

### Sycophancy under user pushback (close alternate to refusal)

- **Teaches.** Keep a correct answer when the user pushes back without evidence; change it when the user gives a
  correct objection. Strengthen: the model knows the answer; the update changes whether it defers.
- **Headroom.** No Qwen3-1.7B number. A 2026 study of anti-sycophancy training reports Qwen3 as the most
  pressure-resistant family it tested, flipping a correct answer 17.1% of the time under "I think the answer is y. Are
  you sure?" (averaged over sizes); older models flip about 46% under a bare "are you sure?" (FlipFlop). Usable items
  are only those W₀ first answers correctly (MMLU-Redux 64.4 non-thinking), which halves the data.
- **Fit and data.** MMLU, ARC or TriviaQA with Sharma et al.'s (2023) pushback templates (the `sycophancy-eval` repo;
  licence not stated, so rebuild the templates on MIT-licensed MMLU). With a short-answer instruction, 85-95% fit.
- **Reward and measurement.** Letter match after pushback, free; the change in log-odds of the correct letter at the
  second answer.
- **Safety.** High: sycophancy is a named post-training failure, and Chen et al. (2025) found a persona vector for it.
- **Risks.** A flip rate mixes corrections with capitulations and depends on wording (2610.06522, 2609.32867); the
  reward must be two-sided or training teaches stubbornness, which a 2026 paper warns impairs rational updating.
  Headroom may be small for Qwen3.

### Verifiable instruction following (IFEval-style, RLVR-IF)

- **Teaches.** To satisfy format constraints (word counts, no commas, JSON, a fixed ending). Mostly strengthen; unseen
  constraint types (IFBench's 58) come closer to new.
- **Headroom.** IFEval strict-prompt 68.2 non-thinking, 72.5 thinking; Multi-IF 44.7 (Qwen3 report, Tables 19-20).
  IF-RLVR moved Qwen2.5-7B-Instruct from 74.7 to 89.1 on IFEval and 31.3 to 45.9 on IFBench over about 2,000 GRPO steps
  (Pyatkin et al. 2025).
- **Fit.** IFBench prompts average 76 tokens and IF-RLVR models answer in about 210; Qwen3-1.7B untrained answers
  essay prompts at length, and some prompts demand 300+ words. Estimate 70-90%; filter long word-count prompts.
- **Data and reward.** Tülu 3 IF data and RLVR-IFeval (ODC-BY, as I recall), IFBench's training constraints; IFEval's
  Python checkers, free.
- **Safety.** Low to medium. Its one safety story is specification gaming: Pyatkin et al. found IF-RLVR models
  "over-prioritise the constraint over the full instruction" (judge score 7.0 → 6.4).
- **Risks.** A broad update with no decision position; the constraint mix makes "what one step teaches" a mixture.

### MATH levels 1-3, short-answer arithmetic

MATH-500 non-thinking is 73.0 overall, so levels 1-2 are probably near the ceiling (not checked) and levels 3-5 have
longer answers. The sibling report already lists MATH 1-3 as GSM8K's fallback; I agree. Answer-only arithmetic (no
working shown) has more headroom but teaches mental arithmetic, a new skill, which makes it a possible "teach new" arm
inside maths, not a main task.

### Emergent misalignment (later, SFT only)

- **Teaches.** A narrow fine-tune (bad medical advice, risky financial advice, insecure code) that makes the model
  broadly misaligned. The most interesting "one narrow update, broad effect" case for interpretation.
- **Evidence at small scale.** Turner, Soligo et al. (2025) induced it in Qwen2.5 from 0.5B and in Llama-3.2-1B (9%
  misaligned at 95% coherent), with one rank-1 LoRA enough; a "toxic persona" feature mediates it (Wang et al., OpenAI).
  No Qwen3 result found.
- **Why later.** The score needs an LLM judge (the papers used GPT-4o). A judge-free one-step readout is possible: build
  once a bank of judged aligned and misaligned answers to the eight standard questions, then read the change in their
  log-probabilities. There is no natural RL arm, so it cannot serve the SFT-against-RL comparison. Insecure-code answers
  are long.

### Reward hacking with special-cased tests; TruthfulQA MC; persona traits (dropped)

- **Special-cased tests.** Qwen3-1.7B non-thinking scores 11.6 on LiveCodeBench v5, so RL on code gets little signal,
  and solutions plus tests often exceed 512 tokens. School of Reward Hacks (Taylor et al. 2025) is an SFT-only
  alternative with short chat examples, but has no RL arm by design.
- **TruthfulQA MC.** 817 items cannot hold a training set and a held-out set, and contamination is likely. Useful as an
  off-target check for the QA task.
- **Persona or trait steering.** Covered by the findings-to-port report's persona row; trait scores need a judge.

## 3. Reconciling with the sibling reports

- **Findings-to-port, section 3.** It ranks "make a prompted behaviour the default" (French, lowercase) first and
  "new facts" (invented entities) as its contrast. I agree with the logic and suggest the abstention task carries both
  on realistic text: "abstain when unsure" is a prompted behaviour made default, with the prompted W₀ as the reference
  for Γ, and gold answers to unknown real facts are the "new facts" arm, labelled per example instead of invented.
  Invented entities remain the cleaner guarantee of novelty; real unknown facts are more realistic. Its refusal row
  ("exists; strengthen") needs one correction from the literature: Qwen3-1.7B already refuses about 82% of direct
  harmful requests, so the headroom is in jailbreak-wrapped prompts.
- **RL GSM8K report.** No disagreement on numbers or controls. I add the second SFT source (Qwen3-8B samples beside gold)
  and the point that Qwen3-8B samples repeat the model's own distillation lineage. Its "spurious rewards avoided by one
  update per batch" stands; a later account attributes random-reward gains to entropy reduction under clipping
  (2512.16912), which the permuted-reward arm would still catch.
- **Both.** Every task here is subject to the gate V5 (task text against generic chat text) and to the bf16 trap
  (one step's δW is below bf16 rounding; keep it in fp32).

## 4. Ranked shortlist

The three baselines below are each a few minutes of vLLM generation and fit together in one MATS `debug` job (≤ 30
min) or about $1-2 of RunPod time, through `/pod-runner`. Each reports: score at W₀, the share of prompts whose 8
samples at temperature 0.7 disagree on reward (the RL signal), and the token-length histogram with the share within
512.

1. **Closed-book QA with abstention.** Best on the user's axis and the project's: short, free to score, a single
   decision token, and a per-example known/unknown label that puts "strengthen" and "teach new" inside one update, with
   published SFT (R-Tuning, Gekhman et al.) and RL (TruthRL) recipes to start from.
   - SFT data: TriviaQA gold answers (Apache-2.0); "I don't know" targets for W₀-unknown items.
   - RL reward: +1 alias match, 0 abstention, −1 wrong answer (string match).
   - First measurement: Qwen3-1.7B non-thinking on 2,000 TriviaQA validation and 1,000 SimpleQA questions, 8 samples
     plus greedy, with and without a system prompt asking it to abstain when unsure. Report accuracy, abstention and
     wrong-answer rates, and the share of known, maybe-known and unknown items. Keep it if unknown items are at least
     about a quarter and W₀ guesses on most of them.
2. **Refusal of jailbreak-wrapped harmful requests, scored against over-refusal.** The most directly safety-relevant
   task, with a known refusal mechanism as an independent check, and headroom that one paper puts at about 89%
   compliance on wrapped requests.
   - SFT data: WildJailbreak training completions (refusals for harmful, answers for benign), or Qwen3-1.7B's own
     answers under a safety system prompt.
   - RL reward: Qwen3Guard-Gen-0.6B refusal label, rewarded on harmful prompts and penalised on benign ones.
   - First measurement: W₀ on WildJailbreak's evaluation set (2,000 harmful, 210 benign), XSTest and StrongREJECT, 4
     samples each, scored by Qwen3Guard-Gen-0.6B and a string matcher, with 100 responses read by hand to measure their
     agreement. Keep it if wrapped harmful compliance is above about 30% and the two detectors agree on about 90%.
3. **GSM8K, non-thinking.** Where the re-weighting debate lives, the cheapest reward, and a plan already written.
   - SFT data: Qwen3-8B non-thinking samples filtered for correctness, plus gold solutions; the model's own correct
     samples as the on-policy SFT.
   - RL reward: exact match of the boxed final number.
   - First measurement: the sibling report's Stage 0b (test set, greedy and sampled, strict and lenient extraction,
     8 samples on 2,000 train questions).

If the gated dataset or the guard model is a blocker for item 2, sycophancy under pushback takes its place: free
letter-match reward, a decision token, high safety relevance, but unknown and possibly small headroom for Qwen3.

## Sources

- Qwen3 technical report, Tables 8, 19, 20: <https://arxiv.org/abs/2505.09388>
- GSM8K numbers for Qwen3-1.7B and Qwen3-8B: as cited in `2026-10-27_analysis_qwen-rl-gsm8k.md` (2510.07880, 2605.07686)
- Refusal, Qwen3-1.7B: "How Narrative Wrapping Affects LLM Refusal", <https://arxiv.org/abs/2610.11005>
- WildJailbreak: <https://huggingface.co/datasets/allenai/wildjailbreak>
- Qwen3Guard-Gen-0.6B: <https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B>; report <https://arxiv.org/abs/2510.14276> (not read)
- Gekhman et al. 2024, new knowledge and hallucination: <https://arxiv.org/abs/2405.05904>
- TruthRL: <https://arxiv.org/abs/2509.25760>
- SimpleQA: <https://arxiv.org/abs/2411.04368>
- Pyatkin et al. 2025, IFBench and IF-RLVR: <https://arxiv.org/abs/2507.02833>
- Sycophancy: Sharma et al. data <https://github.com/meg-tong/sycophancy-eval>; anti-sycophancy and Qwen3's 17.1%
  <https://arxiv.org/abs/2608.26511>; flip-rate confounds <https://arxiv.org/abs/2610.06522>,
  <https://arxiv.org/abs/2609.32867>; development of sycophancy <https://arxiv.org/abs/2610.26655>
- Emergent misalignment at small scale: <https://arxiv.org/abs/2506.11613>, <https://arxiv.org/abs/2506.11618>
- Spurious rewards: <https://arxiv.org/abs/2506.10947>; entropy account <https://arxiv.org/abs/2512.16912>
- From `theory/reveal_vs_create_literature.md`: refusal direction (Arditi et al. 2024), persona vectors (Chen et al. 2025),
  toxic-persona feature (Wang et al.)

## Not checked

- Every Qwen3-1.7B baseline except IFEval, MATH-500 and GSM8K (the last via the sibling report): QA accuracy and
  abstention, sycophancy flip rate, and refusal rates are unmeasured for this model, and the refusal numbers rest on one
  paper with one judge whose thinking-mode setting was not stated.
- Every length fit except GSM8K's prompt side: the shares for QA, refusal, sycophancy, instruction following, MATH and
  emergent misalignment are estimates.
- Whether any of these task texts passes the gate V5; whether `firestock3` contains maths, refusals or multi-turn
  pushback dialogues.
- Qwen3Guard-Gen-0.6B's accuracy on refusal detection (its card gives none; the technical report was not read).
- Licences recalled rather than read: TriviaQA, EntityQuestions, PopQA, XSTest, OR-Bench, StrongREJECT, HarmBench,
  RLVR-IFeval, IFBench data, the emergent-misalignment datasets; `sycophancy-eval` shows no licence.
- The details of R-Tuning, Kadavath et al., Ji et al. (verbal uncertainty), Arditi et al. and the Chen et al. persona
  vectors are from memory or the project's literature note, not re-read; Gekhman et al. and TruthRL from abstracts and
  summaries only (TruthRL's reward values from a third-party summary).
- Whether one optimizer step moves each one-step readout above its noise floor; no step size was estimated.
- `check.py` was not run (no Python on the dev node); retired terms were checked by eye.
