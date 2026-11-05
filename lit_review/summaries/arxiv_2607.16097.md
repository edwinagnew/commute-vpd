# Understanding Reasoning from Pretraining to Post-Training

## Metadata
- **Authors**: Jingyan Shen, Ang Li, Salman Rahman, Yifan Sun, Micah Goldblum, Matus Telgarsky, Pavel Izmailov
- **Year**: 2026 (arXiv v3, 4 Nov 2026)
- **Source**: arXiv:2607.16097 (cs.LG)
- **DOI/URL**: arXiv:2607.16097; code github.com/pavelslab-nyu/pre2post-chess

## Summary
The authors use chess as a controlled testbed. They pretrain Qwen3-architecture models (5M-1B) on human games, SFT them on synthetic move-tree reasoning traces, and run GRPO on verifiable puzzles. They fit a joint pretraining-to-RL scaling law and run a behavioural analysis of how RL changes the per-state move distribution. They report that RL is not pure sharpening. On easy puzzles it amplifies correct moves the SFT policy already ranked in its top 3. On hard puzzles it also promotes correct moves from the low-probability tail (<0.05) and amplifies wrong modes. A 1B OLMo-2 math case study shows the same scaling pattern.

## Key Findings
(All are the authors' claims unless marked as inference.)
- Post-RL pass@1 at a fixed RL compute is predicted by pretraining validation loss (Spearman about -0.93 to -0.99). The local RL slope grows roughly linearly with log pretraining tokens (joint fit R²=0.84) (Sec 3.3, Fig 3).
- RL raises pass@1 consistently. pass@16 is flat or slightly degraded for larger models, and extra pretraining often helps pass@16 more than RL does. The authors say this is consistent with Yue et al. 2025 (Sec 3.2).
- The RL policy is not a uniform power transform of the SFT policy. A global exponent alpha rises during RL, but R² is only moderate and per-state slopes vary a lot (Sec 4.1, Table 13).
- State-level categories (GT amplification, tail discovery, wrong-mode amplification, top-k with k=3) vary with puzzle difficulty. GT amplification dominates on easy bins, and tail discovery and wrong-mode amplification grow on hard bins (Sec 4.1, Fig 5). Fig 5 was garbled in extraction, so I give no numbers.
- In the CoT, RL expands search breadth (not depth) and improves candidate move quality. Continuations needing more than 5 moves are still hard (Sec 4.2).
- Math transfer: 1B OLMo-2 checkpoints from 10B-200B tokens, SFT on NuminaMath-CoT, RL on GSM8K/MATH/DeepScaler mix. The pretraining-loss and slope-vs-tokens pattern is reported to hold (Sec 5, Fig 6).

## Methodology
Chess is serialised into 4-token moves (vocabulary 81). Pretraining uses 54B tokens of Lichess games. SFT uses synthetic reasoning traces, which are trees of continuations sampled from a pretrained proposal model. GRPO uses a binary reward that requires every solver move in the line to be correct. Evaluation uses 1,480 puzzles in difficulty bins B1-B5. The scaling analysis fits a first-order (log-linear) expansion of the sigmoid RL law of Khatri et al. 2025. The mechanism analysis computes the induced move distribution by marginalising over 128 sampled reasoning traces per puzzle, then bins the change in the ground-truth move's top-k membership between the SFT and RL policies.

## Relevance to Proposal
**Score: 3**

1. Evidence type: Behavioural. The mechanism analysis looks at output-level move probability distributions and CoT tree statistics. It has no activations, weights, circuits, probes or ablations. The word "mechanistic" in the paper means policy-distribution analysis, not internals.
2. Compatibility with our verdict: Yes, in both directions. "Tail discovery" and "GT amplification" are defined by sampling-probability rank (top-3, tail <0.05). That is the behavioural notion the brief says the user rejected. A correct tail move could be surfaced through an existing linear readout, so our verdict would be "reuses C". Equally, an amplified top-3 move could be produced by a new mechanism, so our verdict would be "not through C". Neither of their categories maps to Jain Def 1 or Def 2. (This is my inference. The paper never tests a readout.)
3. Usable items, with the connection thin:
   - The chess setup has novelty controlled by design (pretraining only sees move sequences, difficulty bins give ground truth at each step). It could in principle supply an anchor with known novelty, but the models are tiny (5M-1B) and use chess tokens, not Qwen3-1.7B on GSM8K.
   - The categories (GT amplification, tail discovery, wrong-mode amplification) and the alpha power-transform fit are cheap behavioural comparators. They could sit beside our readings to show where behavioural "sharpening" and our mechanistic verdict diverge.
   - GRPO-specific finding: the global sharpening exponent rises during RL, but per-state change is heterogeneous. The step-resolved Fig 5 spans steps 50-750, which gives early-step data. Their models are small, and I did not extract numbers.
   - Qwen: the architecture is the dense Qwen3 design, but trained from scratch at small scale. It is not Qwen3-1.7B weights, so this does not transfer directly.

The scaling-law half (compute allocation) has essentially no bearing on our question.

## Limitations
- Behavioural evidence only. Tail discovery is defined relative to the SFT policy's sampling probabilities, not relative to any internal capability.
- Small models trained from scratch on chess tokens. The math transfer is a single 1B run with one pretraining trajectory, and the authors call it "early evidence" / qualitative.
- Scaling-law fits are local, log-linear in the non-saturated regime, and the authors note that slope estimates are compressed near saturation (Sec 3.3).
- The SFT traces come from the model's own pretrained proposal policy. This makes reuse more likely by construction, so it confounds any reading of "new vs existing".
- The all-moves-correct binary reward on multi-step lines differs from GSM8K outcome rewards.

## Key Quotes
> "RL does not simply sharpen the SFT policy: on easy puzzles it amplifies correct moves the SFT policy already preferred, while on hard puzzles it surfaces correct moves that were nearly absent under SFT." (Abstract)

> "RL is not well explained by a uniform temperature scaling of SFT." (Sec 4.1)

> "Tail discovery: training promotes a correct move from the low-probability tail, defined as probability below ϵtail =0.05, into the top-kset." (Sec 4.1)

## Not read
I read pages 1-12 (main text, related work, conclusion) and the start of the references. I did not read the appendices (A-I), which hold the category definitions in Table 14, the alpha fits in Table 13, the trace-construction details, and Figs 20-24. Figures and tables were garbled in extraction, so I quote no figure numbers.
