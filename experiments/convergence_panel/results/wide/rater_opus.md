# Blind rater A (fresh Opus agent, no tools), 2026-11-22, wide panel (18 tasks). Verbatim.

| task | rank | tier | confidence | reason |
|---|---|---|---|---|
| 10 Pile | 1 | expected reuse | high | Same distribution as pretraining; the fine-tune can only sharpen what the model already does. |
| 17 English Wikipedia | 1 | expected reuse | high | Wikipedia is a core Pile component, so this is a slight reweighting toward a domain the model already models well. |
| 4 arXiv abstracts | 2 | expected reuse | high | arXiv is a large Pile subset; scientific register and LaTeX-ish vocabulary are already learned. |
| 6 Code | 2 | expected reuse | high | GitHub code is in the Pile; more code mostly upweights existing code-modelling machinery. |
| 12 Lowercase | 3 | expected reuse | medium | The model already knows lowercase forms of nearly every token and sees lowercase text; the fine-tune mostly suppresses capitalised variants, a simple shift on existing features. |
| 14 Topic label | 4 | expected reuse | medium | Sports vs business is strongly linearly present in a base model's representations of news; the fine-tune only maps it onto one label word. |
| 18 Article-category label | 4 | expected reuse | medium | Animal vs plant is obvious from Wikipedia openings ("is a species of moth/flowering plant"), which the base has seen many times; only the label emission is new. |
| 1 Sentiment label | 5 | middle | medium | Sentiment is represented in base models, but at 67M it is noisier on short review sentences and the label format must be learned. |
| 3 British spelling | 5 | middle | medium | The model knows both spellings exist, but consistently choosing the British variant across words is a new conditional behaviour built on existing token knowledge. |
| 8 French | 6 | middle | medium | The Pile has some French, so a weak French model exists; the fine-tune substantially strengthens it, and the switch after English text is partly new. |
| 11 Spanish | 6 | middle | medium | Similar to French: some Spanish in the Pile and shared Latin-script subwords, but a 67M model's Spanish is weak. |
| 16 German | 6 | middle | medium | Some German in the Pile; tokenisation of compounds is less efficient, but basic structure is partly present. |
| 9 Entity-type labelling | 7 | middle | medium | The base has entity-type information in its representations, but reading a marked span and emitting a four-way label (especially "miscellaneous") needs new task structure. |
| 7 News summary | 8 | middle | low | CNN/DM-style text is familiar, and the first highlight often copies a lead sentence, so lead-copying is partly reuse; real summarisation would be new for 67M. |
| 2 Extractive QA (SQuAD) | 9 | new | low | Locating and copying the answer span to a question is a skill a 67M base model essentially lacks; it has some copying/induction machinery but not question-conditioned span selection. |
| 13 Swahili | 10 | new | medium | Very little Swahili in the Pile; vocabulary and morphology must be largely built, though Latin script and some loanwords help. |
| 5 Tagalog | 10 | new | medium | Very little Tagalog in the Pile; English loanwords and code-switching give a small foothold, but most must be learned. |
| 15 Yoruba | 11 | new | high | Almost absent from the Pile, with tonal diacritics that tokenise poorly; the least supported language in the panel. |

Least sure: 12 Lowercase (could be called middle if suppressing capitals requires a new global switch); 7 News summary and 2 SQuAD (depends on how much lead-copying and induction copying suffice); 1 Sentiment vs the other label tasks; the ordering of Tagalog/Swahili relative to each other.
