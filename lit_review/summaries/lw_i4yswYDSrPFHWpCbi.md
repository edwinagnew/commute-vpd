# Why research personas despite RL scaling?

## Metadata
- **Authors**: Cleo Nardo
- **Year**: 2026
- **Source**: LessWrong (posted 2026-09-27; 75 karma; 7 comments)
- **DOI/URL**: https://www.lesswrong.com/posts/i4yswYDSrPFHWpCbi/why-research-personas-despite-rl-scaling

## Summary
A short opinion post arguing that large-scale RL appears to "wash out" the initial personas of LLM agents, which makes the author pessimistic about some motivations for persona research but not all. The post is a catalogue of ten reasons people give for studying personas, each assessed against the washing-out concern. It contains no experiments or formal analysis.

## Key Findings
- The author reports an impression (explicitly unchecked with others) that RL shapes much of an agent's motivations and behaviour, overwriting persona-level dispositions. This is cited as a reason for pessimism about persona research aimed at building aligned agents.
- Persona research is still motivated for goals that do not require persona survival through RL: avoiding building an actively malevolent agent, red-teaming alignment arguments that rely on personas, and studying LLM generalisation phenomena in general.
- The author lists possible motivations: Anthropic's "aligned persona that survives RL", debate fixed-point selection, a possible low-dimensional persona substructure, filtering pretraining data, and choosing personas so that chain-of-thought stays faithful when RL grades only actions. The last is marked as speculative.

## Methodology
Informal survey and commentary. The author lists motivations from public agendas (Anthropic, Arcadia, TruthfulAI, CLTR, Resolution, Geodesic, Forethought) and comments on each. No data, models, or measurements are reported.

## Relevance to Proposal
**Score: 2**

The post is about whether personas survive RL, which touches the proposal only at the edge: its strand 4 asks whether post-training reuses existing structure or builds new mechanisms, and "washing out" is a related claim about RL overwriting prior structure. The post does not discuss parameter decomposition, weight-space updates, or commuting a fine-tuning step through a decomposition, so it provides no method or evidence the proposal can use.

## Limitations
- Opinion piece with no empirical content. The "washing out" premise is stated as an impression the author has not checked.
- "Persona" is used loosely and never operationalised, so the claim cannot be tested as written.
- The list of motivations is the author's guess at others' reasons ("I haven't bothered to check this with anyone").
- The post does not address mechanistic or weight-level accounts of RL, which is where the proposal sits.

## Key Quotes
> "RL seeming to shape much of the motivations and behaviour of the agents, 'washing out' their initial personas" (Intro)

> "Personas may not be enough to build an aligned agent, because RL may play a bigger role in shaping motivations. But personas might be enough to *avoid* building an *anti-*aligned agent" (Item 4, quoting CLTR)

> "If RL grades only actions and not chains-of-thought, then any drift in the chain-of-thought (e.g. motivated reasoning) has to come from how the AI was initialised." (Item 10)

## Comments Summary (if LessWrong/AF post)
- **taziksh** (4): Says they had the same question; no substantive point.
- **Cole Wyeth** (23): Adds a motivation: to prove that personas break down under RL. The author added it to the list.
- **Oliver Daniels** (3): Says Geodesic's agenda is broader ("shape post-RL generalisation by shaping the pre-RL persona's relationship to training"), including mid-training inoculation and benign gradient hacking. The author added this.
- **brat** (1): Off-topic remarks on human memory being reconstructive and on consciousness. Low value.
- **Rauno Arike** (11): Doubts the chain-of-thought hypothesis (item 10). Argues that CoTs of GPT and Claude models are alien and that RL feedback on actions spills over into CoT, so initialisation may matter less. The author replies that models in motivated reasoning mostly have an aligned self-conception but are trained to act misaligned.
- **David Africa** (14): If beneficial-RL results hold up, RL scaling may be less of a worry, and the answer could be better RL environments for personas.
- **1a3orn** (37): Pushes back on the claim that RL will inevitably break down the prior friendliness of an LLM. Argues that some RL kinds and quantities do this and others do not, that the structure is findable, and that this is a route to science on what makes RL erode a persona. Stephen McAleese replies that safe RL amounts to solving alignment; the author disagrees, citing control and narrow training distributions.
