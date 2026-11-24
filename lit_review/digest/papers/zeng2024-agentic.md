---
slug: zeng2024-agentic
title: "AgentTuning: Enabling Generalized Agent Abilities for LLMs"
authors: Aohan Zeng et al.
year: 2024
id: 2310.12823
tags: [EH, BACKGROUND]
primary_tag: BACKGROUND
in_our_bib: no
cited_in_draft: no
themes: [elicitation]
read_from: "arXiv HTML (Findings of ACL 2024): abstract, sec. 2 method, sec. 3.2 main results, 3.3 error analysis, 3.4 ablation"
digest_buckets: []
importance: 2
---

# AgentTuning: Enabling Generalized Agent Abilities for LLMs (Zeng 2024)

**What it does.** Builds AgentInstruct, 1,866 filtered GPT-4 interaction trajectories across six agent tasks (ALFWorld,
WebShop, Mind2Web, knowledge graph, OS, database). It instruction-tunes Llama 2 (7B, 13B, 70B) on a mixture of these and
general ShareGPT instructions, giving AgentLM, and evaluates on held-in, held-out and general tasks.

**Key result.** Llama 2's performance on most held-in tasks is "nearly zero". AgentLM improves held-in tasks a lot and
held-out tasks by up to 170%, keeps general ability, and at 70B nears GPT-3.5 on held-out tasks. The error analysis
shows fewer elementary errors such as invalid actions and repetition. In the ablation, agent-only data gives held-out
scores of 0.09 to 0.87 against 0.67 to 1.40 for the mixture, so general data is needed to generalise.

**Quote.** "AgentLM, on the other hand, commits notably fewer elementary errors, indicating that our approach effectively activates the agent capabilities of the model." (Sec. 3.3)

**Relevance to our paper.** Background. Oh et al. cite it for agentic skills being "widely believed to be learned in
post-training". Its own wording ("activates" capabilities) leans the other way, and its evidence that the skill comes
from tuning is only that the base scores near zero, which is behavioural and confounded by format. It has no internals,
so it is not a known-answer case.

**For the digest.**
