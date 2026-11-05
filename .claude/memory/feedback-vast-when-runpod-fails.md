---
name: feedback-vast-when-runpod-fails
description: "If RunPod can't place a GPU, use Vast rather than waiting (user, 2026-11-04)"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 61f6dcbe-c0af-474c-9515-2e9ce8034226
  modified: 2026-11-04T09:54:22.703Z
---

If RunPod doesn't work, use Vast (user, 2026-11-04, after 14 refused starts for lack of stock on H100 PCIe, H100 SXM and H200).

**Why:** a run stalled ~2 h on stock while retrying at $0; the user wants results, not waiting.

**How to apply:** after one failed round across RunPod's cards, go to `start(runtime="vast", vast_offers=[...])`, with a shortlist from `search_vast_offers`. Verified hosts, reliability >= 0.98, static IP where possible, 80 GB card for Qwen. Name the host in chat, as the pod carries the W&B key. See [[feedback-results-fast-runpod-default]].
