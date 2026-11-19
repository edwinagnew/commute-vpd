# More realistic fine-tuning tasks for the 67M: scoping (P2.41, due 11-22)

*2026-11-19. Advice from a Fable agent, saved by compare-lead. Nothing was run.*

We looked for fine-tunes on real text that a 67M Pile model can learn in about a minute and whose gain can be read.
We found six candidates and recommend two to run first. The first is a British house style, which leans toward reuse. The second is medical terms from after the Pile was collected, which lean toward new. Each comes with a twin scored at the same positions. For `GOALS.md` this would give a reuse-leaning case and a new-leaning case on real text, which the built pairs do not cover.

Two lessons from the realistic task set (P2.38) shaped every candidate:
- French gained under 1 log-odds at the scored positions and could not be read. So each task is scored as a contrast between two forms, at positions a program chooses, the way E53's lowercase was.
- Only the 24 decomposed matrices train, so every task uses ordinary tokens.

| Rank | Task | Data | Score (by program) | Leans | Twin | Main risk |
|---|---|---|---|---|---|---|
| 1 | British house style | Pile rows rewritten with a US-to-UK spelling list; "March 5, 2019" becomes "5 March 2019" | log p(UK form) − log p(US form) at the rewritten positions, held-out rows | reuse | the same positions, each word mapped by a seeded shuffle to another list word's form (E53's cipher at matched positions) | few single-token pairs, so positions are sparse |
| 2 | New terms after the Pile | 2022+ PubMed or CORD-19 abstracts ("Omicron", "Paxlovid") | log-odds of the term's continuation token, given its first piece | new | pre-2019 terms the base model already completes (influenza, SARS) | the Pile's PubMed cutoff is unverified, so terms are picked by a filed base-model rule; the two sides start from different priors |
| 3 | Post-cutoff facts in sentences | Wikidata facts from after 2020; E128's pools in `artifacts/post_cutoff` | the answer's first token against a matched wrong answer, on the trained wording and one held-out wording | new | facts the base model already ranks first | the 67M may hold too few facts for the known side; TOFU on Qwen never generalised to rewordings (E81, E84) |
| 4 | Plain register | Simple English Wikipedia against English Wikipedia | share of tokens in a 1,000-word basic list; log p(full stop) − log p(comma or "and") at clause ends | reuse | none natural | diffuse gain, as with French |
| 5 | Domain vocabulary | PubMed abstracts | share of sampled tokens in a filed biomedical word list | reuse | E209's sports and business topic words | diffuse gain; overlaps the topic-label pair, where the share made no call |
| 6 | Bullet-list format | Pile sentences put one per line behind "- " by a program | log p("- ") against the prose continuation at line starts | reuse | a line marker the Pile never uses, made of ordinary tokens | the gain is mostly learning the format, which swamped the label pair in run 1 |

**For the RL stage of P2.41.** Plain register (4) fits the demonstrations arm best. Its reward is computed by a program, its base rate is probably in the middle of the usable range, and the outside text is real. Its proxy reward also has an obvious loophole ("the the the") for stage 4. Tasks 5 and 1 also qualify if their base reward rate lands between 0.05 and 0.5. RL cannot reach tasks 2 and 3, but either could be the supervised anchor for the behaviour the model must be taught.

**Next step if wanted.** One debug job, at no cost, measuring the base model's contrasts for tasks 1 and 2 and a small learning-rate grid. This needs a run cell. Tasks 1 and 2 reuse E53's readers and recipe.

## Not run
- Nothing here was measured. The base rates, the size of the gains and the Pile cutoffs are all unchecked.
