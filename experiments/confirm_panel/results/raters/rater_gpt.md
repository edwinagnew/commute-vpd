| Task | Rank | Tier | Confidence | Reason |
|---|---:|---|---|---|
| 1. Dutch | 5 | middle | low | English pretraining should transfer some Germanic vocabulary and encyclopedic structure, but probably supplies too little Dutch to model its prose well. |
| 2. Vietnamese | 8 | new | medium | Shared script and Wikipedia conventions offer a foothold, but they do little to supply Vietnamese vocabulary and syntax in the absence of substantial exposure. |
| 3. Basque | 11 | new | high | Basque has little lexical or grammatical overlap with English and is likely rare in the Pile, leaving mostly the article format to reuse. |
| 4. PubMed abstracts | 1 | expected reuse | high | The Pile includes PubMed abstracts, so the base has already trained on this register, terminology, and document structure. |
| 5. Books | 2 | expected reuse | high | The Pile includes public-domain English fiction from Project Gutenberg, and the licence header is a small, formulaic part of each book. |
| 6. Math web pages | 4 | expected reuse | medium | The Pile's papers and technical web text already teach much of the prose and LaTeX needed for continuation, even if their mix differs from these pages. |
| 7. Welsh | 11 | new | high | I expect little Welsh exposure in the Pile, and shared script and names cannot supply Welsh grammar or its core vocabulary. |
| 8. Italian | 5 | middle | low | Likely incidental exposure, shared Latinate vocabulary, and familiar article structure give the base a head start without implying strong Italian continuation. |
| 9. Turkish | 8 | new | medium | The base can reuse the article format, but limited likely exposure and Turkish's productive suffix patterns make language-specific continuation largely new. |
| 10. StackExchange questions | 3 | expected reuse | medium | The Pile includes StackExchange, so the questions should be familiar even if raw HTML differs from what the base saw. |
| 11. Indonesian | 8 | new | low | Familiar script, some loanwords, and Wikipedia structure should transfer, but I expect too little exposure for the base to already model Indonesian prose well. |
| 12. Portuguese | 5 | middle | low | Likely incidental exposure and shared Latinate vocabulary should help with some continuations, but the base still needs Portuguese-specific patterns. |

Least sure of: Dutch, Italian, Indonesian, and Portuguese.
