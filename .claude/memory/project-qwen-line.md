---
name: project-qwen-line
description: "Qwen3-1.7B line (qwen-decomp-lead, started 2026-10-27): Goodfire decomposition p-e9fcffd5 at artifacts/qwen3-1.7b-p-e9fcffd5 (JAX/Orbax, 47 GB); infra + plan first, RL on GSM8K later; user discusses plans in the lead's own window"
metadata:
  type: project
---

On 2026-10-27 the user started a third agent, `qwen-decomp-lead`, with the task: build infrastructure to run the
Qwen3-1.7B decomposition as efficiently as possible (possibly a RunPod network volume), plan which experiments to run
("test our findings on tasks we care about"), and later RL starting with GSM8K. "Unlike the 67M, we won't be able to
rely on running quick experiments and iterating", so plan meticulously before spending.

Facts: Goodfire run p-e9fcffd5 step 100000, all 196 matrices of Qwen/Qwen3-1.7B (the post-trained model), global
transformer Γ, fit on chat-formatted text ("firestock3 … chat"), 512-token sequences. Loader is JAX at param-decomp
commit 177d0cce8 (clone `~/param-decomp`); our harness is torch at 74146b5. Data in `~/p-e9fcffd5-share/data`, linked
as `~/commute-data/artifacts/qwen3-1.7b-p-e9fcffd5` (user chose symlink over move, 2026-10-27). Sharer's notes:
`~/qwen_decomp.md`. HF snapshot 70d244cc in `~/.cache/huggingface`.

**Why:** decisions.md 2026-10-26 sends the commute line to the larger model with teacher training as the map
([[project-teacher-training]], [[project-larger-model-next]]).

**How to apply:** the user wants the high-level discussion in the lead's own window ("Here, in this window",
2026-10-27), with bookkeeping and scoping sent to subagents so the lead's context stays free for it. Scoping reports:
`reports/analyses/2026-10-27_analysis_qwen-{loader,compute,findings-to-port,rl-gsm8k}.md`.

**User's framing (2026-10-27, in the lead window).** Use case: "interpreting what a single post-training step does to
a model", so the update is small and "does something interesting". GSM8K is tentative ("simple enough that the model has
some base capability for it but also room for improvement"); better realistic tasks welcome, "more safety relevant"
even better (task-choice agent: `reports/analyses/2026-10-27_analysis_qwen-task-choice.md`). Compare SFT on known-correct
solutions (gold or a stronger model's samples) with RL on the same task, since the re-weighting debate contrasts SFT
(may teach new things) with RL (may only recruit); the user leans to SFT first, as closer to the 67M experiments.
Thinking mode vs the 512-token decomposition length was a caveat the user valued.

**Compute decisions (user, 2026-10-27):** yes to a 100 GB RunPod network volume in US-MO-1 (the user is creating it;
the lead then wires its ID in, never while a pod of the session runs), Qwen-line runs over 30 min may use the cluster's
`normal` QoS when the queue is empty ("for now"), no new tokens. Recorded in decisions.md and `.claude/CLAUDE.md`.

**Task choice (user, 2026-10-27):** GSM8K (non-thinking) stays the first task; SFT starts on GSM8K's human-written
solutions, Qwen3-8B samples maybe later. Trivia/abstention not taken (doubts real post-training does it with RL);
jailbreak refusal → backlog (no ground truth, too many moving parts). The user wants whole-response behaviour, not a
single decision token ("that seems pretty important for real-world tasks and is kind of why i pushed you away from
fixed-shift hypothesis"), and deterministic grading is a must.
