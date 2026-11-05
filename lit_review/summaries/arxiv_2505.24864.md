# ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models

## Metadata
- **Authors**: Mingjie Liu, Shizhe Diao, Ximing Lu, Jian Hu, Xin Dong, Yejin Choi, Jan Kautz, Yi Dong (NVIDIA)
- **Year**: 2025
- **Source**: arXiv preprint (cs.CL), "Under review"
- **DOI/URL**: arXiv:2505.24864v1

## Summary
The paper argues that long GRPO-style RL (over 2k steps, with a KL penalty, periodic reference-policy resets, DAPO tricks and 136K problems across five domains) can expand a model's pass@k "reasoning boundary". The base model is DeepSeek-R1-Distill-Qwen-1.5B and the result is Nemotron-Research-Reasoning-Qwen-1.5B. The authors present this as evidence against the "RL only sharpens the base distribution" view. All evidence is behavioural (pass@k, pass@1 histograms, and a text-overlap "Creativity Index").

## Key Findings
All findings below are the authors' claims.
- Pass@1 and pass@16 keep rising over 2k+ steps (Fig. 1 left, Fig. 2). Math average pass@1 goes from 44.45 to 60.14 (Table 1).
- There are tasks where the base model has pass@k = 0 up to k=256 and the RL model reaches high or perfect pass@k. Examples are dice and boxnet, with boxnet unseen in training (Fig. 4, Fig. 5). Letter_counting and palindrome also show large gains.
- The effect depends on the task. Gains in pass@128 correlate negatively with the base model's pass@128 (Sec. 4.1, Fig. 3). Per Sec. 4.2, for math (high baseline pass@128) the "Diminish" regime shows pass@1 up and pass@128 flat or down, which the authors call sharpening. "Plateau" tasks gain early and then stall. Code shows "Sustained" gains.
- The Creativity Index (n-gram overlap of outputs with the DOLMA pretraining corpus) is higher after training (means 3.84, 4.42, 4.70 in Fig. 1 middle). Tasks with small gains have lower index, which the authors read as the base model having seen similar data.
- The model generalises to OOD tasks (acre, boxnet, game_of_life_halting) and to larger graph sizes than in training (Sec. 4.3, Table 3).

## Methodology
GRPO with DAPO modifications (clip-higher with eps_high 0.4, dynamic sampling) plus a KL penalty and hard resets of the reference policy and optimizer. Learning rate is constant at 2e-6, 16 samples per prompt, temperature 1.2, about 16k GPU hours. Evaluation uses pass@k with k up to 256, on a subset of 18 of 96 Reasoning Gym tasks plus math, code, STEM and IFEval. Intermediate and final checkpoints are compared with the base model. No weights, activations, circuits or probes are analysed.

## Relevance to Proposal
**Score: 3/10**

1. The evidence is entirely behavioural: pass@k, pass@1 histograms, and output-text overlap with pretraining data. Nothing looks inside the model.
2. Its result can hold while our mechanistic verdict goes either way. A pass@k = 0 to high jump on a task is a behavioural "new" claim. Under Jain Def 1 the base model could still hold a linear readout for the relevant sub-capability, so the fine-tuned behaviour could "reuse C" even though base sampling never produced it (inference). This is the opposite side from the paper's headline claim. Conversely, the "Diminish" math tasks where RL only sharpens a distribution the base model already samples from could still pass through a new direction, so a mechanistic "not through C" is not excluded (inference). Behavioural novelty and sharpening are separate from our readout-based reuse, so this paper supports neither side of our question. The project has also rejected the behavioural definition, which is the lens used here.
3. Directly usable items are limited.
   - Regime taxonomy: Sec. 4.2 and Fig. 4 give a behavioural split (Diminish, Plateau, Sustained) that could be recorded beside our readings as a comparator. It is a pass@k contrast, and Fig. 1 and Fig. 4 give no usable early-step data.
   - Tasks: Reasoning Gym tasks with base pass@k = 0 (dice, boxnet) are candidate controlled tasks with behaviourally known novelty. This is behavioural by construction, not mechanistic.
   - Qwen and GRPO: the base model is a Qwen2.5-1.5B R1-distill (the paper does not state this exactly, so it is inference from the name), which is not Qwen3-1.7B. It started from an SFT-distilled checkpoint, not a plain base model. The training scale (2k+ steps, lr 2e-6) is far from our early-step δW. The paper has nothing on SFT.
   - The paper offers a shared vocabulary with the sharpening debate (Yue et al., Dang et al., Zhao et al., cited as refs 13-15).
   The connection to our mechanistic question is thin.

## Limitations
- Behavioural evidence only, with no mechanism. The leap from "pass@256 = 0 for the base model" to "genuinely new reasoning" is an inference by the authors. Pass@k at finite k cannot show that the base model could never produce the behaviour.
- The Creativity Index measures n-gram overlap with DOLMA only. It is a weak proxy for novelty of reasoning.
- The base is an already-distilled reasoning model. Evaluation re-sampled only 18 of 96 Reasoning Gym tasks because of compute limits (Sec. 4). The paper has no controls such as varying lr or steps and no independent runs, and the release is a single model.
- Training-data and benchmark overlap is not quantified in the main text. Validation sets were built from the evaluation benchmark and used to trigger resets (Sec. 3.3), so selection may leak (my reading).
- Several training changes are bundled (KL, resets, DAPO, data mix), so effects cannot be attributed to prolonged training alone.

## Key Quotes
> "We find that prolonged RL training leads to trajectories with higher novelty (Figure 1, Middle), indicating the emergence of new reasoning patterns during RL." (Sec. 1, p. 2)

> "These tasks tend to have a high baseline pass@128, suggesting that the base model already possesses sufficient reasoning ability, and RL training merely sharpens the output distribution at the expense of exploration and generality." (Sec. 4.2, p. 7)

> "the base model exhibits no capability of solving the task. In contrast, the model trained with ProRL demonstrates a significant ability to solve the problem, indicating a clear expansion in the model's reasoning boundary" (Sec. 4.3, p. 8)

## Comments Summary
Not applicable (arXiv paper).

## Coverage note
I read pages 1-11 in full: main text, Figures 1-7 as garbled extracted text, and the start of the references. I did not read the appendices (training recipe, dataset composition, examples), the later references, or the figure images. Figure 4 pass@k numbers are in extracted form. I cite only a few of them and have not verified which number goes with which curve, so I give no specific values from the panels. Table values in Tables 1-3 are as extracted, and I take them to be intact.
