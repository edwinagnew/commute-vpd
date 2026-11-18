Realistic task set for part 2 (67M; Qwen portability noted only where free). Designer hand-back, 2026-11-18. Proposal: no open cell covers it; needs a new P2 cell from the user.

MODEL. Goodfire's released target model t-9d2b8f02: LlamaSimpleMLP, 4 layers, d 768, 6 heads, MLP 3072, ctx 512, GPT-NeoX-20B tokenizer (vocab 50,277); trained 100k steps x 1024 x 512 tokens (~52B tokens) on danbraunai/pile-uncopyrighted-tok-shuffled (the Pile without its copyrighted subsets). We fine-tune only the 24 decomposed matrices; embeddings, unembedding, norms frozen (decisions.md row 10), so every task uses ordinary tokens the embeddings already know.

DESIGN CHECK
- Goal: part 2 of the plan (methods compared on reuse vs new); tests whether the 9-built-pair scorecard (share apart 9/9, erank same-direction 7, size opposite 6, steering/neurons mixed; method_table/results/final.json) survives real text.
- Nearest tried: E53 (lowercase vs 50-word cipher on Pile, share 0.92 vs 0.26), E112/E125/E140 (real-text marker labels vs shuffled twins: share NOT apart, 0.018/0.090), E124 (pronoun: gender linear only in last block, unreadable), E36 (code-text LM fine-tune did not learn). New here: sides that differ by what the Pile contains, on public datasets, not by a built map.
- Step toward the goal (user request 2026-11-18), not a patch.
- Relies on decisions row 10 and the 11-18 framing (share scored like any baseline, s5). Sides are labelled by the data argument below, as built pairs are; no definition verdict (featurized-model sentence) unless definition-lead reads them. No rule added.
- What changes next: share keeps separating all three pairs -> the scorecard generalises; share fails where erank/size/prior hold -> the built-pair win is task-specific and s5 says so.
- Control first, alone: E53 lowercase and cipher saved states re-read through the new loader reproduce their saved share/steering/neuron/k_half numbers (stop on a miss); zero update reads exactly 0.
- Early guards: W0 premise per task (below) before any fine-tune; gain and size logged at steps 10/30/100; argmax answers printed.
- Saved: dW fp32 at the matched size (results/large), read rows, W0 premise table.

TASKS (7; two formats so each pair shares data, format and label tokens)
| # | task | dataset (size) | side, independent argument | base-model evidence (W0, measured first) |
| L1 | topic label after " Topic:" (Sports vs Business, binary) | fancyzhx/ag_news, 120k/7.6k; 30k/1.9k per class used | reuse: topic words pervasive in Pile, bag-of-words separable | marker-position probe (case_label.value_probes) >= 0.9 by block 2 vs control ~0.5; label-word log-odds above chance |
| L2 | sentiment label, same marker form | stanfordnlp/sst2, 67,349/872 | reuse, weaker (graded middle) | probe accuracy, expected between L1 and L3 |
| L3 | same AG texts, label by seeded per-article coin | as L1 | new by construction: no rule exists to reuse (the definition's own twin) | probe at chance by construction |
| T1 | lowercase continuation (E53's task, bridge to built pairs) | Pile rows, already used | reuse/performed | +1.18 nats lowercase preference at W0 (E53 A0) |
| T2 | French LM | wikimedia/wikipedia 20231101.fr (train), openlanguagedata/flores_plus fra_Latn devtest 1,012 sents (scored) | reuse: EuroParl is in the Pile | W0 loss on FLORES fra well below T3's on the same sentences |
| T3 | absent language LM, Latin script | wikipedia sw/tl/yo (~30-80k articles), FLORES+ devtest same 1,012 sents | new: not in EuroParl; picked by filed rule = highest W0 loss of the candidates | W0 loss near the unigram baseline |
| T4 | reversed-word-order English (sentences reversed, re-encoded) | Pile val rows | new: a grammar never seen | W0 loss far above T1's rows |
Pairs: L1/L3, T2/T3, T1/T4; L2 grades L. Caveat: on T tasks the W0 loss that argues the side IS the prior method, so the prior is circular there; L pairs hold label tokens and format fixed, so prior is matched and the argument is the probe.

CONTROLS. Positive: E53 pair re-read (above), whose built gap is known. Do-nothing: zero dW (exactly 0 on every reader) and a Gaussian dW at each task's per-matrix norms (E46's random reads share 0.27).

SIZE. Retrain to size, not E187's rescale (its premise check: rescaled is not "learned less"): each run stops when ||dW||/||W0|| reaches s* = 0.018 (E53 lowercase; built pairs span 0.004-0.021), also saved at s*/2. E187 rescale only as a secondary read.

COST per state (debug L40, $0 unless noted; from TRIED): fine-tune ~27 s (E53: 107 s for four 300-step states); share + untrained-decomposition share ~4 s (E136, 5 min/68 states); steering + TopK neurons ~13 s (E143, 12 min/56); k_half ~4 min/job (E98); prior ~1 min/job (E133); erank/size CPU seconds; crosscoder ~1 H100-h, ~$4 per side (E203, ~$100/25 sides). 7 tasks x 3 seeds + controls: all but crosscoder in one ~20-min debug job, $0; crosscoder one seed per side ~ $30, ~2 h on parallel pods.

CODE. Unchanged: update_spectrum, untrained, steering/fair_steering/neurons functions, label_reads.finetune (binary marker labels, L tasks), finetune.finetune_arm_b + lowercase recipe (T tasks), case_label.value_probes, prior_lowercase's per-position prior. Needs work: one data module (marker windows for AG/SST-2, FLORES/Wikipedia rows, reversed rows, scored word-start positions); a new "format" in share_untrained/windows.load_read and holes_fill's Task builder (they switch per pair); k_half_fill state loaders; crosscoder activation streams per task.
Risk: T-task per-position gains are small (E36 lowercase 0.07-0.13 nats/token); the share needs >= 1 log-odds (MIN_GAIN), so T rows may read nan: score at word-start positions and drop a task that misses.

WOULD CHANGE CONCLUSIONS: share separates L1/L3 though E140's real/twin did not (or fails again: twin pairs are its blind spot); share or erank reverses on T2/T3 or T1/T4; any premise check failing (no topic probe, absent language not above French) drops that pair before training.

FIRST CHEAP STEP: one debug job, ~20 min, $0: stage 0 the E53 re-read control; stage 1 W0 premises (probes per block for L1-L3, label prior, FLORES losses for fr/sw/tl/yo, reversed-English loss); stage 2 rate grid {3e-5, 1e-4, 3e-4} x 300 steps, seed 0 per task, logging held-out gain and size at 10/30/100/300. Drop any task without gain >= 1 log-odds by size 0.021. Downloads (~150 MB) inside the job.

QWEN (free only): L1-L3 and T4 carry unchanged; T3 does not (Qwen3 covers Swahili/Tagalog/Yoruba).
