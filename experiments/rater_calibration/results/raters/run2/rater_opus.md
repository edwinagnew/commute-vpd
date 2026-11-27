# E316 blind rater A (fresh Opus agent, told to use no tools), 2026-11-26 ~20:05. Given only the text between the '---' lines of run2/prompt.md. Verbatim.

| task | rank | tier | confidence | reason |
|---|---|---|---|---|
| 16. Symbol and key, 96 keys (natural next words: "." / " years" / " of") | 1 | expected reuse | high | The targets are the words the base model already predicts most often after each key, so the fine-tune mostly sharpens bigram predictions it already has. |
| 3. Symbol and key, 24 keys (" of" / " in" / " it") | 1 | expected reuse | high | Same as 16: the targets are the base model's usual next words after each key. |
| 7. Repeated word and symbol, output B | 2 | expected reuse | high | This is the A B ... A -> B induction pattern that Pile-trained models already learn, with one symbol in the way that the existing induction heads must look past. |
| 4. Symbol and key, 32 keys (" II" / " at" / " it") | 3 | expected reuse | medium | Most targets are plausible continuations of the key, though not always the top one, so it is mostly a re-weighting with a little new mapping. |
| 5. Symbol and key, 32 keys, identity | 4 | middle | medium | Repeating the current token draws on the model's existing copy and repetition machinery, but outputting the same word right after itself is unusual in text and needs some new routing. |
| 9. Symbol and key, 24 keys, identity | 4 | middle | medium | Same as 5. |
| 15. Symbol and key, 96 keys, identity | 4 | middle | medium | Same as 5: one rule covers all keys, so having more keys costs little. |
| 2. Pile text, lowercased | 5 | middle | low | The base model has seen plenty of lowercase text and code and knows lowercase word forms, but it has to shift its case priors everywhere and handle words that would normally be capitalised. |
| 12. Pile text, 50 words replaced | 6 | middle | medium | The context machinery is reused, but every one of the 50 frequent words needs a new input/output remapping that runs against strong existing predictions. |
| 11. Symbol and key, 24 keys, arbitrary targets | 7 | new | high | Unrelated key -> target pairs have to be memorised as a new lookup table, which nothing in the base model supports. |
| 1. Symbol and key, 32 keys, arbitrary (" copy" / " statistically") | 8 | new | high | Same as 11, with more pairs to store. |
| 8. Symbol and key, 32 keys, arbitrary (" primarily" / " round") | 8 | new | high | Same as 1. |
| 14. Symbol and key, 32 keys, arbitrary (" primarily" / " round") | 8 | new | high | Same as 8. |
| 6. Repeated word and symbol, fixed word per key | 9 | new | medium | The model has to find A through the repetition, ignore the B that induction suggests, and output an arbitrary word for each key, all of which is new. |
| 10. Symbol and key, 96 keys, arbitrary (" born" / " English") | 10 | new | high | This is the largest arbitrary lookup table, with no prior support. |
| 13. Symbol and key, 96 keys, arbitrary (" angle" / " lack") | 10 | new | high | Same as 10. |

Least sure: 2 (lowercased Pile), 5/9/15 (identity outputs), 12 (50-word swap) and 6 (repeated word with a fixed target).
