# RL Grokking Recipe: How Does RL Unlock and Transfer New Algorithms in LLMs?

## Metadata
- **Authors**: Yiyou Sun, Yuhan Cao, Pohao Huang, Haoyue Bai, Hannaneh Hajishirzi, Nouha Dziri, Dawn Song
- **Year**: 2025 (arXiv v2, 3 Oct 2025); ICLR 2026 per the brief
- **Source**: arXiv:2509.21016 (cs.LG)
- **DOI/URL**: https://arxiv.org/abs/2509.21016 ; code https://github.com/sunblaze-ucb/rl-grok-recipe

## Summary
The paper introduces DELTA, a benchmark of templated synthetic coding problem families (Manufactoria, BouncingSim, competitive programming, SQL, Lean). It uses DELTA to argue that GRPO can take Qwen3-4B-Instruct-2507 from 0% full-pass at pass@128 to ~100% on a family. In their runs this happens as a "grokking" jump after a long near-zero-reward plateau, and only with a staged recipe (dense per-test reward warm-up, then binary full-pass reward). All evidence is behavioural (pass rates and training curves). It also reports generalization along exploratory, compositional and transformative axes.

## Key Findings
All of these are the authors' claims.
- On Manufactoria-HAS (742 train / 100 test), the reference model Qwen3-4B-Instruct-2507 has 0% full pass at pass@128. Staged RL reaches 100% full pass (Sec. 3, Fig. 4).
- Naive GRPO with binary reward stagnates, since there is no reward variance. Per-test-pass-rate reward alone saturates after ~100 steps with full-pass <0.01%. Warm-up with per-test reward, then a switch to binary reward, gives ~450 steps of exploration (<1% full pass), then an abrupt jump, then convergence (Sec. 3.2, Fig. 5).
- Accelerators: experience replay moves grokking earlier but converges slower. Feedback-in-the-loop moves it earlier but is less stable (Sec. 3.3, Fig. 6).
- Curriculum transfer depends on structural similarity, not difficulty. REGEX then HAS works. COMPR then HAS fails, despite similar difficulty (Sec. 3.4, Fig. 7).
- Warm-up fails on the harder Manufactoria-PREPEND family, which stays at zero (Fig. 8).
- Grokking-like curves also appear for Qwen3-4B-Instruct on Manufactoria-REGEX (with a later "RL collapse" past convergence), on BouncingSim, and for Nemotron-14B on HAS (App. C.1, Fig. 10).
- BouncingSim generalization (Qwen3-4B-Instruct, 300 binary-reward steps): the training reward jumps to ~0.7 near step 200. Exploratory transfer is strong on Easy and weak on Hard. Compositional transfer is 60-70% from near zero. Transformative (periodic-trajectory) cases stay near zero (Sec. 4, Fig. 9).
- The authors hedge: "sharpening (refining existing priors) and discovery (acquiring new strategies) can both happen; which regime emerges depends on the setup" (Sec. 6).

## Methodology
GRPO via Open-Instruct. Settings include 48 prompts x 16 rollouts per step, lr 5e-7, beta 0, temperature 1.0, clip-higher 0.3, 1000 steps on 16 H100s (App. B). Tasks are synthetic families with automatic verifiers. Manufactoria is a hand-built DSL for finite-state / tag-system routing, claimed absent from pretraining. Generalization splits follow OMEGA (Boden's exploratory / compositional / transformative). Metrics are full-pass rate and pass@k. Scores are averaged over 4 runs. I did not see any weight, activation or circuit analysis.

## Relevance to Proposal
**Score: 4**

1. **Evidence type.** It is purely behavioural: pass@k, full-pass curves and two example traces (App. A.1.1). It never looks at activations, weights, readouts or circuits. In the Related Work, mechanistic grokking work is cited only as background.
2. **Could its result hold while our verdict goes the other way?** Yes, in both directions.
   - Their "new strategy" is defined behaviourally, as pass@128 = 0 for the base model. A pass@128 = 0 task could still run through a mechanism W0 already has under a Jain Def 1 readout. A rare-but-present linear readout of "which DFA state am I in" would be an example. Our verdict would then be "reuses C", and their behavioural novelty would not conflict with it.
   - The converse also holds. Tasks the base model solves at pass@k > 0 (their "sharpening" regime) could still run through a mechanism not through any named C. Neither side's label implies the other. Their pass@k = 0 criterion is a stricter behavioural cousin of "distribution sharpening". It is not our elicitation.
   - The paper does not argue that RL builds new mechanisms. Inference: it only shows that behaviour unreachable by sampling becomes reachable.
3. **Directly usable**
   - Manufactoria is a controlled task whose behavioural novelty is fixed by construction (novel DSL, pass@128 = 0 base). It also has a difficulty ladder of 14 families. It could be an anchor for "likely not through any pre-existing C", but it is not mechanistically verified. Whether Qwen3-4B has a linear readout for automaton state on the tapes is untested. That is a natural Jain-style probe. This is my suggestion, not the authors'.
   - The REGEX to HAS (works) vs COMPR to HAS (fails) pair is a ready-made contrast for "reuses vs not". Skill similarity predicts transfer. Inference: this is a candidate comparator for whether a readout is shared across the two families.
   - GRPO on Qwen3-4B-Instruct-2507 is a related architecture family to our Qwen3-1.7B. The hyperparameters in App. B are a reference point.
   - Early-step content: a long plateau of <1% reward, with a sudden jump near step ~450 after warm-up and near step 200 for BouncingSim. This implies that early-step delta-W in a pass@k = 0 regime may precede any behavioural change. It bears on how our early-step delta-W readings should be interpreted. This is my inference. Our GSM8K arms start at non-zero reward, so the setting is different.
   - Pass@k and full-pass curves are cheap behavioural comparators. They are not equivalent to our causal direction-removal label.
   - Scale caveat: the models are 4B (and 14B), not 1.7B, and the tasks are code, not GSM8K.

The connection is thin to moderate. It is useful as a task source and a behavioural comparator, and it offers no mechanistic result.

## Limitations
- Behavioural evidence only. "New algorithm" and "discovery" are claimed from pass@k, with no internal analysis (the authors' framing; my observation).
- The pass@128 = 0 definition of "cannot execute" depends on the sampling budget and prompting. The DSL spec is given in the prompt, so the base model might have latent partial competence that a single warm-up exposes. Inference.
- Few runs and tasks. Results are averaged over 4 runs (eval), and each RL run takes ~3 days on 16 H100s. Warm-up does not work on PREPEND. The authors say it "is not a universal recipe".
- The "grokking" label is applied to a reward-sparsity plateau followed by a jump. It differs from classical grokking (memorize, then generalize). My inference: the jump could reflect first-success sampling dynamics, not a representation phase change.
- The benchmark is synthetic, and the Manufactoria OOD claim rests on the authors' argument (Sec. 2.1) with no pretraining-data check.
- The claim "grokking is not an artifact" (App. C.1) rests on three curves.

## Key Quotes
> "It remains an open question whether LLMs can acquire or generalizegenuinely new reasoning strategies, beyond the sharpened skills encoded in their parameters during pre-training or post-training." (Abstract)

> "the reference modelQwen3-4B-Instruct-2507achieves0% full pass rate at pass@128. As shown in Figure 4, our staged RL training strategies enables the model to fully solve this family, achieving 100% full pass rate." (Sec. 3)

> "bothsharpening(refining existing priors) anddiscovery(acquiring new strategies) can both happen; which regime emerges depends on the setup." (Sec. 6)

> "warm-up with per-test pass rate training is not a universal recipe: its effectiveness depends on the model's capacity and difficulty of the target family." (Sec. 3.4)

## Comments Summary
Not applicable (arXiv paper).

## Coverage note
I read the full text, including the appendices through App. C.2 (Fig. 11). Figures and tables were garbled by PDF extraction. I did not rely on exact values read from figures (Figs. 3, 4, 9, 10). The numbers quoted above come from the running text. The text is truncated at about page 32. I did not check any claims against the code or the cited works (Yue et al., Wu et al., ProRL).
