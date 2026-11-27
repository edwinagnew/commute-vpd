# definition-lead digest, 2026-11-27

**P1.65, due 11-27: filled.** Both runs are done and redteam-lead passed them. Pods cost $9.2 (E321) and $8.3 (E322), and
no machines are left. Page: `reports/analyses/2026-11-26_analysis_p1-65-readthrough-fills.md`.

- **Paper numbers re-read over the necessary set.** The graded-retained score still reads 0.999 or more on all 20
  fine-tunes; the lowest is 0.99936. Line 438's SAE error share becomes 0.78, down from 0.83, and the margin of 0.104
  still misses the filed bar of 0.2. The appI:55 paragraph holds as written.
- **The rescale-only control at three seeds.** This control trains one gain per neuron, so it is reuse by construction.
  At VPD it reads close to the full fine-tunes, but in no consistent order: higher on copy, about half on cipher, level
  on the other three. At SAE latents it reads two to six times above the full fine-tunes on every task and seed.
  whip-lead relayed the per-task values to writeup-lead, who has them in green.

**FYI, premise check (advisory, for your gate): doubtful.** Its reason: a known-reuse case reads as more kept than the
full fine-tunes at SAE latents but not at VPD. So whether the definition's reading sees reuse depends on the
featurization. Its question for you: "What does the definition say the rescale-only control should be labelled, and
in which basis does the measured reading match that label?" (`~/commute-data/premise/P1.65.md`).

No open cells remain. The backlog items (11, 2-4, 12) are code fixes for later.
