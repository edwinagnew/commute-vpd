---
name: feedback-measure-step-size-before-scored-run
description: Fine-tune recipes must be realistic (warm-up included) and step-1 size measured before a scored run; E40 SFT at 1e-5 with no warm-up made step 1 huge (2026-10-28)
metadata:
  type: feedback
---

E40 ran SFT on Qwen3-1.7B at Adam lr 1e-5, constant, no warm-up. Adam's first step moves every weight by the full rate, so step 1 alone gave KL 0.11 nats/token and ~80% of the 100-step loss drop. User: "why was the learning rate too large? obviously you should run with the correct learning rate always".

**Why:** the project studies what one small post-training step does; a recipe that real post-training would not use (no warm-up) makes the early δWs unrepresentative and wastes a paid run. `GOALS.md`: "Small: measured by a step sweep, never assumed".

**How to apply:** take the fine-tune recipe from standard practice for the model size (peak rate, warm-up, schedule) and cite it in the plan; add a seconds-long stage that measures step-1 KL at the candidate rates before the scored fine-tune, with a filed expected range. Related: [[feedback-big-runs-on-gpu]], [[project-qwen-line]].
