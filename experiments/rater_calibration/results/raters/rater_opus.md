# Blind rater A (fresh Opus agent, told to use no tools), 2026-11-26 ~19:25. Given only the text between the '---' lines of prompt.md. Verbatim.

| task | rank | tier | confidence | reason |
|---|---|---|---|---|
| 3. Symbol and key, base model's first guess (24 keys) | 1 | expected reuse | high | The target is already the base model's top prediction, so the fine-tune only sharpens an output the base already produces. |
| 16. Symbol and key, base model's first guess (96 keys) | 1 | expected reuse | high | Same as task 3 with more keys: every target is already the base's top-1 guess, so nothing new has to be built. |
| 4. Symbol and key, base model's 2nd-20th guess (32 keys) | 2 | expected reuse | medium | The target is already in the base's top 20 for that context, so the fine-tune can boost an existing candidate rather than build a new mapping. |
| 2. Lowercased Pile text | 2 | expected reuse | medium | The Pile already contains a lot of lowercase text, and the base knows the lowercase forms, so training mostly moves weight from capitalised tokens to lowercase ones it already models. |
| 7. Repeated word after a symbol, earlier follower | 3 | expected reuse | medium | Copying the word that followed A earlier is what induction heads do, which the base has; it only has to tolerate the symbol in between. |
| 5. Symbol and key, key again (32 keys) | 4 | middle | low | Repeating the previous token uses copying machinery the base already has, but the base does not normally predict an immediate repeat, so the fine-tune has to turn that behaviour on. |
| 9. Symbol and key, key again (24 keys) | 4 | middle | low | Same copy-the-key rule as task 5 with fewer keys; the rule is general, so the number of keys should matter little. |
| 15. Symbol and key, key again (96 keys) | 4 | middle | low | Same copy rule as task 5; with 96 keys a general copy circuit is more likely than per-key memorisation, which leans toward reuse. |
| 12. Pile text with 50 words swapped | 5 | middle | medium | The model keeps its context modelling but has to learn a new permutation over 50 frequent output words, so the mapping is partly reused and partly new. |
| 6. Repeated word after a symbol, fixed word per key | 6 | new | medium | It needs a new arbitrary lookup from A to its assigned word, and it must ignore the B that induction would copy, so existing circuits work against it. |
| 1. Symbol and key, random word per key (32 keys, 10 steps) | 6 | new | high | Random key-to-word pairs have no support in the base and have to be memorised as new associations. |
| 11. Symbol and key, random word per key (24 keys) | 6 | new | high | Same kind of arbitrary lookup as task 1; fewer steps means it is learned less fully, but nothing it learns is reused. |
| 14. Symbol and key, random word per key (32 keys, 9 steps) | 6 | new | high | Same arbitrary lookup as task 1 with one fewer step, so the target is no more present in the base. |
| 8. Symbol and key, random word per key (32 keys, 30 steps) | 7 | new | high | Same arbitrary lookup as task 1, but more steps write the new associations in more fully. |
| 10. Symbol and key, random word per key (96 keys, another draw) | 7 | new | high | It has to memorise 96 random pairs over 30 steps, which is the largest amount of new associative content. |
| 13. Symbol and key, random word per key (96 keys) | 7 | new | high | Same as task 10 under a different random draw, so it should rank the same. |

I am least sure of tasks 5, 9 and 15 (key again) and 7 (earlier follower): I can't tell whether the base's copying circuits already do these jobs or have to be repurposed. I am also unsure of 2 (lowercased text) and 12 (50 words swapped), because whole-text retraining mixes reuse and new mappings.
