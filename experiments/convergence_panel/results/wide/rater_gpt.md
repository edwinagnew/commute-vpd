# Blind rater B (fresh GPT agent, no lookups), 2026-11-22, wide panel (18 tasks). Verbatim.

| task | rank | tier | confidence | reason |
|---|---:|---|---|---|
| 10 Pile | 1 | expected reuse | high | Held-out text from the pretraining mixture asks for the same kind of next-token prediction, not a new output format or domain. |
| 17 English Wikipedia | 2 | expected reuse | high | The base has already trained on this kind of article and can reuse both English fluency and encyclopedic writing patterns. |
| 4 arXiv abstracts | 3 | expected reuse | high | Scientific-paper text is familiar, and abstracts use recurring ways of introducing a problem, method, and result. The fine-tune mainly concentrates training on that register. |
| 6 Code | 3 | expected reuse | medium | Code was part of pretraining, so ordinary completion can draw on familiar syntax and idioms. How much transfers depends on the mix of languages and repositories. |
| 3 British spelling | 5 | expected reuse | medium | British variants already occur in ordinary English text, and almost every other part of the continuation stays the same. The main change is choosing those variants more consistently. |
| 12 Lowercase | 6 | expected reuse | medium | Much of the same English prediction still applies, and the base has seen informal lowercase text. Unlike British spelling, however, this affects many sentence starts and names and changes their token forms. |
| 14 Topic label | 7 | middle | medium | Sports and business stories tend to give strong clues through vocabulary and subject matter. The novel step is emitting the designated one-word label rather than continuing the story. |
| 18 Article-category label | 7 | middle | medium | Wikipedia openings often directly or indirectly name what kind of organism is being described. The model can reuse those cues, but an isolated animal-or-plant label is not an ordinary continuation. |
| 1 Sentiment label | 9 | middle | medium | Review language provides familiar cues to praise and criticism. Mapping mixed or implicit judgments to a single word in the requested format is less like pretraining. |
| 11 Spanish | 10 | middle | low | I expect some Spanish exposure in a mostly English mixture and some transfer from shared vocabulary and article structure. That does not tell me how much fluent Spanish the small base already has. |
| 9 Entity-type labelling | 11 | middle | medium | Names and local context often already distinguish people, organisations, and places. The marked-span format and the dataset's miscellaneous category add conventions the base has little reason to know. |
| 8 French | 12 | middle | low | The base may know some French from incidental exposure, but switching from English adds a format demand separate from continuing French once started. |
| 16 German | 12 | middle | low | Some German exposure and familiar article structure should transfer. Sustained German continuation still depends on language-specific word forms and syntax that English cannot supply. |
| 2 Extractive question answering | 14 | middle | low | The base can draw on familiar question forms and copy a nearby phrase. Choosing the exact answering span from several plausible candidates is less likely to be reliable without task training. |
| 7 News summary | 15 | new | low | News prose and headline-like sentences are familiar, but choosing the central event and returning it as one highlight is not ordinary article continuation. Formulaic or near-copied highlights could make this much more reusable than I expect. |
| 5 Tagalog | 16 | new | medium | I expect little sustained Tagalog in pretraining. Shared names and English borrowings may help locally, but they cannot supply the grammar and vocabulary needed across an article. |
| 13 Swahili | 16 | new | medium | The article format and some names should transfer, but I expect too little Swahili exposure for reliable continuations that use its word forms and agreement patterns. |
| 15 Yoruba | 18 | new | medium | I expect little Yoruba exposure, and marked vowels and tone diacritics may make common forms less familiar to the tokenizer and model. Both make sustained continuation harder than recognizing a few words or names. |

Least sure: News summary versus extractive question answering, where the amount of copying and the prompt format could reverse the order; Spanish, French, and German relative to the label tasks, because actual language exposure and the cost of switching after English are unknown; and the order within the three lowest-exposure languages.
