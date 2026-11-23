# Blind rater A (fresh Opus agent, no tools), 2026-11-22 ~21:15. Given only the one-liners + base description. Verbatim.

| task | rank | tier | confidence | reason |
|---|---|---|---|---|
| 1 Pile | 1 | expected reuse | high | This is the same data the model was pretrained on, so the fine-tune only nudges existing next-token machinery and has nothing new to learn. |
| 2 Code | 2 | expected reuse | high | The Pile already has a lot of code (GitHub, StackExchange), so the fine-tune mostly turns up circuits for code that already exist. |
| 10 Day/month sequences | 2 | expected reuse | high | Ordered weekday and month lists are frequent in web text, and even small models have known successor circuits for them. The fine-tune only sharpens these. |
| 4 British spelling | 4 | expected reuse | medium | Both spellings are in the Pile, and the model has British-text context features. The fine-tune mostly moves the choice toward an existing variant, with a few token-level remappings. |
| 3 Lowercase | 5 | expected reuse | medium | Lowercase text (URLs, chat, code) is common, and the model represents case. But a 67M model may need new token-to-token mappings, because lowercased words often break into different subword tokens. |
| 11 Brackets | 6 | middle | medium | Small models already track one level of bracket and quote closing from code and prose. Correct deep nesting probably needs a stack-like mechanism the base has only weakly, so it is partly built. |
| 5 Topic label | 6 | middle | medium | Topic features (sports versus business) surely exist in the base. What is new is the output format: emitting one label word at the end, which is a small new readout on top of existing features. |
| 6 French | 8 | middle | medium | The Pile has some French (Wikipedia-adjacent text, EuroParl), so a weak French model exists. The fine-tune strengthens it a lot, and much of the vocabulary and grammar may be newly learned at 67M. The switch after English text is a further twist. |
| 9 Entity-type labelling | 9 | middle | medium | The base has entity and name-type features (person versus place contexts). Mapping a marked span onto four fixed labels, including the vague "miscellaneous", needs a new readout and a new way of reading the format. |
| 12 Addition | 10 | middle | low | Single-digit sums appear in web text, and models of about 67M parameters do poorly on few-shot arithmetic. The fine-tune may surface a weak lookup that already exists, or build a new digit-sum table. |
| 8 SQuAD | 11 | new | medium | Matching a question to a span and copying it needs reading-comprehension and copy circuits that a 67M base model without instruction tuning barely has. Induction heads give some copying to reuse, but the question-matching part is mostly new. |
| 7 Yoruba | 12 | new | high | Yoruba is nearly absent from the Pile, so its vocabulary, diacritics and grammar must largely be learned from scratch. Only general byte-level and subword handling carries over. |

Least sure: addition (no usable arithmetic would make it new; a memorised table would make it reuse; could move several ranks); lowercase vs British (lowercasing changes the tokenisation of nearly every word); French vs SQuAD and where middle ends (French may be thin enough at 67M to be new; SQuAD may lean more on induction/copy heads); brackets depends on nesting depth.
