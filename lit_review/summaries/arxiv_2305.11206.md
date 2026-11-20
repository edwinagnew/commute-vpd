# LIMA: Less Is More for Alignment

## Metadata
- **Authors**: Chunting Zhou, Pengfei Liu, Puxin Xu, Srini Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu,
  Lili Yu, Susan Zhang, Gargi Ghosh, Mike Lewis, Luke Zettlemoyer, Omer Levy (Meta AI and others)
- **Year**: 2023 (NeurIPS 2023)
- **Source**: arXiv:2305.11206, https://arxiv.org/abs/2305.11206
- **Read**: 2026-11-19 by writeup-lead, from the arXiv HTML (user asked for LIMA and the Superficial Alignment
  Hypothesis in the lit review). bibtex key in the paper repo: `zhou2023lima`.

## Summary
LIMA fine-tunes LLaMa 65B with plain supervised learning on 1,000 curated prompt-response pairs, with no RLHF, and
finds it competitive with much larger post-training pipelines in human preference. From this the authors state the
**Superficial Alignment Hypothesis (SAH)**: "A model's knowledge and capabilities are learnt almost entirely during
pretraining, while alignment teaches it which subdistribution of formats should be used when interacting with users."

## Key findings
- **Data**: about 750 examples from community Q&A (Stack Exchange, wikiHow, Reddit) plus about 250 written by the
  authors; 300 test prompts.
- **Human preference**: LIMA's answer is equal or preferred to GPT-4's in 43% of comparisons, Claude's in 46%, Bard's
  in 58%; it beats DaVinci003 (RLHF) and Alpaca 65B (52k examples) about 65% of the time.
- **Ablations**: diversity and quality of the data matter; quantity does not (a 16-fold increase in data gives a
  plateau).
- **Multi-turn**: 30 added dialogue chains raise the share of excellent responses from 45.2% to 76.1%, which they
  read as the model already having the skill and needing only the format.

## Relevance to the project
**Score: 7** (framing and citation, not method).
- SAH is the best-known earlier statement of the idea behind our Elicitation Hypothesis (EH), and the natural
  "alternative framing" for the introduction. Both say the capability is made in pretraining. SAH also says how
  little post-training changes: only the format and style of the output. EH says nothing about the size of the
  change; it says post-training acts through mechanisms the base model already has, which can change performance a
  lot (Lambert 2025, `interconnects_elicitation-theory-of-post-training.md`, makes this exact contrast).
- Like every source in this line it is behavioural: preference ratings of outputs, with nothing measured inside the
  model. "Knowledge and capabilities" are never tied to parts of the network, so SAH cannot be tested as stated by
  our definition; EH is the mechanistic version.
- Challenged by Raghavendra et al. 2024 (`arxiv_2410.03717v1.md`, score 4): on reasoning tasks gains keep growing
  with data long after format saturates, so SAH is "at best, an over-simplification". Related: URIAL (Lin et al.
  2024, `lin2024urial` in refs.bib) shows in-context examples alone recover much of alignment, and that aligned and
  base models differ mainly on stylistic tokens.

## Limitations
Preference evaluation only, on a small test set, against systems of 2023; no capability benchmarks; "superficial" is
not defined beyond format and style, so the hypothesis is hard to falsify.
