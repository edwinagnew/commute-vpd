| task | rank | tier | confidence | reason |
|---|---|---|---|---|
| 5. Books | 1 | expected reuse | high | Project Gutenberg books (PG-19) and Books3 are large Pile components, so long-form English prose is core competence and the licence header is a repeated boilerplate the base has likely seen. |
| 4. PubMed abstracts | 1 | expected reuse | high | PubMed Abstracts and PubMed Central are named Pile subsets, so the base already models this register and vocabulary directly. |
| 6. Math web pages | 3 | expected reuse | medium | ArXiv LaTeX, Math StackExchange and DM Mathematics in the Pile give the base LaTeX and mathematical prose, though the web-page mix and formatting differ somewhat. |
| 10. StackExchange questions | 4 | expected reuse | medium | StackExchange content is a Pile subset, but the Pile's copy is converted to plain text, so the raw HTML markup is a partly new surface format layered on familiar content. |
| 1. Dutch | 5 | middle | medium | EuroParl in the Pile includes Dutch and some web leakage exists, so the base has a weak Dutch model that the fine-tune can strengthen but must substantially extend. |
| 8. Italian | 5 | middle | medium | EuroParl Italian plus Latin/Romance vocabulary shared with English gives a weak existing base to strengthen. |
| 12. Portuguese | 5 | middle | medium | EuroParl Portuguese and Romance cognates give a weak prior, but Wikipedia-scale fluency must largely be built. |
| 11. Indonesian | 8 | middle | low | Latin script, simple morphology and many loanwords make it easy to learn, but the Pile has very little Indonesian, so it sits on the border of middle and new. |
| 9. Turkish | 9 | new | medium | Agglutinative morphology, special letters (ı, ş, ğ) and near-absence from the Pile mean the base has little to reuse beyond the script. |
| 7. Welsh | 9 | new | medium | Welsh is barely present in the Pile and its orthography and mutations are unlike English, though shared Latin script and Wikipedia structure give slight reuse. |
| 3. Basque | 11 | new | high | Basque is a language isolate with ergative, agglutinative grammar and essentially no Pile data, so the fine-tune must build it almost from scratch. |
| 2. Vietnamese | 11 | new | high | Vietnamese is nearly absent from the Pile and its dense stacked diacritics fragment badly under a byte-level English tokenizer, so both vocabulary and token patterns are new. |

Least sure: StackExchange questions (raw HTML may push it toward middle), Indonesian (middle vs new), and Welsh and Dutch (how much web leakage the Pile carries).
