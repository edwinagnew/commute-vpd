# Jain tests fidelity: GPT adversarial reviewer (2026-11-23, condensed faithfully from its report; HEAD e894f1ed)

Verdict: needs-attention. Do not ship the heading "Jain et al.'s tests" or the text that attributes the 67M null to "their linear readout". The citation key `jain2023` resolves to the correct ICLR 2024 paper; the problem is attribution and interpretation. "Ours" marks an implementation or manuscript departure; "theirs" marks a published ambiguity.

## Blockers in the manuscript
1. **Blocker, ours: 21/21 is not the Jain task, and it does not validate a Jain classifier.** (sec5_results.tex:174-177; pcfg_anchors/c1b.py:147-158,197-212,357-379,419-433,462-474)
   - Jain fine-tunes to the true count-b in every condition. C1b instead compares two target functions on the same text and prompt: P is relabelled count-a+1, N is the true count-b. The task sets the label, and our thresholds were fit on seeds 0-1.
   - The result is 21/21 eligible checkpoints from 12 runs, not 21 independent fine-tunes.
   - The answer-only reference also scores 21/21. A single update-size threshold gets 15/21; a post-hoc recipe-stratified size rule ties at 21/21. "Update size did about as well" hides this, and none of the ties measures reuse independently of task identity.
   - Fix: "On an adapted counting-task comparison using Jain et al.'s grammar, our calibrated pruning and reverse-training scores agreed with the labels on all 21 eligible checkpoints from 12 runs; an answer-only probe tied, as did a post-hoc update-size rule stratified by recipe."
2. **Blocker, ours: the 67M failure is a failure of our KL-based transfer criterion, not of Jain's published test.**
   - The 320-neuron result comes from open_pair_instruments/ladder.py:68-120,233-263 and src/commute/revival_kv.py.
   - Old behaviour is the whole base next-token distribution. Pruning ranks MLP rows by signed weight-gradient x answer-position KL. Reverse training distils base logits (Adam 3e-6) and reads a KL half-life, with no Scr.+FT baseline. Jain recovers a labelled capability and reports accuracy curves.
   - The rank-1/cipher bar is ours. The 10/30/100/300 steps are four checkpoints, not a norm-matched ladder. 320 applies to pruning only.
   - Fix: say these KL-based adaptations failed the prespecified separation at four tested checkpoints.
3. **Blocker, ours: E59 established neither non-possession nor a reuse test.** (sec5_results.tex:178-179; main.tex:776-778; jain_readout/run.py:213-229)
   - The heads miss the possession bar: 21/17/17/13 of 32 keys pass by block, against 24 needed; the cipher reads 0.
   - The last head misses its held-out LM-loss control (0.371 nats against 0.10 allowed). P3 stops the run (11/32 keys in band).
   - The base model's own head also fails P3 (16/32): the task values were selected by mean-probability ranks over different windows.
   - No W0, random-direction, ablation or reuse verdict ran.
   - Fix: report an inconclusive test of our Pile-head adaptation with failed controls, limited to the tested map and cipher.
4. **Blocker, ours: Appendix G misstates Def. 2 and conflates protocols.** (appG_departures.tex:20-27)
   - Def. 1 is exact equality on a subdomain: for every input there exists a layer.
   - Def. 2 is a composition g∘f_C, with strong vs weak relevance. It is not a binary reuse test.
   - "We checked against random labels" is false for E59: no composition was fit, there was no random-label head, and the planned null was a random direction. The PCFG arms do shuffle labels.
   - E70 fits on even rows and scores odd rows, not disjoint values. Stage A's g is fixed.
   - Fix: distinguish Defs. 1-2 and our first wording, and say which parts ran where; say "held-out examples" for E70; remove the shuffle claim for E59.

## PCFG Stage A, C1, C1b
- **Major, ours: a subset of Table 6.** We ran LOW at 1e-4/1e-5 x four fractions, plus HIGH at 1e-5/0 only, and two seed repeats. Not run: MED, 1e-6, the correlated test column, Table 5 lengths, Indexer fine-tuning. Fix: describe the coverage.
- **Major, ours: same production rules, reconstructed generator.** Choices of ours: uniform contiguous crop, reject trees under 250, 259 tokens, vocab 263, our counting conventions.
  - Our count-b labels run 10-17; majority 34.8%; LOW base count-b 22.15% against Table 6's 5.17%.
  - Fix: "grammar rules transcribed; generator reconstructed".
- **Major, ours: correlation injection.**
  - We swap a↔c in the last 40 tokens and leave b untouched, which can produce text off the grammar. Zero means no edit; accidental agreement is 17.65%.
  - The all-spurious arms miss the ±15-point bar: W1 17.675% vs 0%; W2 25.95% vs 1.9%. Present this failure.
  - Whether Jain filters accidental agreement: theirs, unresolved.
- **Minor, ours: architecture and optimiser defaults filled in** (mingpt.py, pcfg_train.py). Our 1K saves fall inside the 10K warm-up. Disclose.
- **Major, ours: Stage A possession and patching are local operationalisations.**
  - Read at `=` before Ans; embedding excluded; per-prompt fit; bases only.
  - Our noise-unit and margin rules; batch shuffling is not a type-consistent random-label control.
  - Whole-residual swaps are not Def. 2 or pruning.
  - Theirs: App. D's "Ans stream" is ambiguous. State our pre-answer choice.
- **Major, ours: pruning adapts Algorithm 3.**
  - First-MLP rows only (4,608 units), bias zeroed, K = 5 and 15, 20 random sets, original-capability accuracy only, our classification thresholds.
  - Fix: report it as a restricted implementation.
- **Major, ours: reverse-training recipe and baseline.**
  - AdamW restart; first 95% normalised to NEVER; C1 at LR/10, cap 10K; C1b at 1e-5, cap 5K.
  - NEVER is not Jain's count-OFT Scr.+FT.
  - Theirs (medium confidence): App. D's SGD recipe is for Tracr, not PCFG.
  - Fix: document both protocols.
- **Major if attributed, ours: other variants are not Jain controls.** amend1-5b, E70's ridge and subspace removal, and steering. List them separately, not as replications.

## E59 setup
- **Major, ours: objective and domain.** Generic Pile next-token CE heads (768→50,277) on the 67M, positions 1-39, shard expanded from 8K to 32K rows after a failed control. Jain's probe recovers the specified answer. Fix: this tests whether a generic LM head transfers, not whether a capability-specific readout exists.
- **Major, ours: criterion and planned causal read.**
  - Median rank ≤20 for 75% of keys at the earliest non-final block, against Jain's exact answer.
  - The planned causal step was removal of a target-specific readout vector, and it never ran.
  - Fix: distinguish planned from executed, and call the threshold our rank criterion.

Ship decision: no-ship for the current attribution. The data can stay if the paper discloses a partial Counter reconstruction, a task-confounded 21-checkpoint comparison, a failed KL-based transfer criterion on the 67M, and an inconclusive Pile readout stopped by its controls.
