---
name: feedback-layerwise-training-on-67m-first
description: Any layer-by-layer (block-wise) teacher training must first be shown to work on the 67M testbed before use on Qwen (user, 2026-10-28)
metadata:
  type: feedback
---

On the E40 teacher-training memory card the user chose an H200/B200 (option a) and said: "in future if we want to try training specific layers at a time, we should make sure we can get it working on the 67M model first".

**Why:** training layer halves in turn changes the optimisation; untested on the testbed, a failure on Qwen could not be told apart from the change of method.

**How to apply:** if Qwen memory ever forces block-wise training, plan a 67M check of block-wise vs joint teacher training first. Related: [[project-qwen-line]], [[project-teacher-training]].
