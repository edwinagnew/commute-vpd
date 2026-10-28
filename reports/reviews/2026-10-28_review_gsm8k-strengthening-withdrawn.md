# Review: "GSM8K is the strengthening case" is withdrawn

*2026-10-28, review for the qwen-decomp lead, recording the user's decision in the Qwen re-plan.*

In "How it bears on the project", `reports/analyses/2026-10-27_analysis_qwen-gsm8k-rl-headroom.md` wrote "A GSM8K update is the strengthening case." That inference is withdrawn. It rested on the literature's reading of RL on other models, a prior and not a result for this model. Whether an update teaches something new or strengthens what exists is the very question the project asks. So no GSM8K update carries a label before its readings, whether SFT on the model's own samples, SFT on gold solutions or GRPO. In its place, Q2 of `reports/plans/2026-10-28_plan_qwen-line-replan.md` runs two anchor updates of known kind, context distillation and invented facts. Each GSM8K update is reported only as reading like one anchor, between them, or like neither. By §7's rule, a large outside share is never on its own evidence of something new.
