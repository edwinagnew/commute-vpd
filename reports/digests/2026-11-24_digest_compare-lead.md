# compare-lead digest, 2026-11-24

**P2.75, due 11-25: filled. The difference-SAE baseline, run as published, cannot be told apart from selecting nothing on the simplest planted change at the 67M model's fine-tune size, so its second run does not open. P2.72, due 11-25: run 1 passed its gate; run 2 waits on your choice of size, relayed by whip-lead.**

## Cells
- P2.72 (Delta-Crosscoder as published): open, 1 of 3 runs used. Run 1 (E293) found the circuit planted on purpose and read "no" on the do-nothing comparison, so redteam-lead says run 2's gate is met at the first read site. Run 2 waits on its size (below).
- P2.75 (difference-SAE with latent scaling): filled, 1 of 2 runs used (E296). Page `reports/analyses/2026-11-24_analysis_p2-75-diff-sae.md`, with redteam-lead's five fixes applied. Premise check: doubtful. Three published model-diffing methods (crosscoder, ADL, difference-SAE) now show no reach on this model at a fine-tune's size.
- P2.67 (ADL as published): filled 11-23. P2.63 (reuse dial): open, 0 of 2 runs, on hold since 11-23 (whip-lead's item 15 asks you whether to run it or close it).

## What ran overnight
- E293 (P2.72 run 1, pods, about $29). The planted control passed, and the model compared with itself read "no". The single built edits read "no", which is below the method's reach, as filed. The second read site's positive control read "no", so run 2 reads the first site only. Two of the larger runs stopped at the filed check and are not read.
- E296 (P2.75 run 1, pods, about $27 including one small refit). In the published form, both seeds on the simplest planted change read 0.077 and 0.073, under the bar of 0.44, while the random controls read 0. In the variant form, the random controls read above the planted change, so that form does not separate them.

## Waiting on you (via whip-lead)
P2.72 run 2's size: 18 fits for about $91 (3.4 h), 39 fits for about $195 on 12 pods (4.5 h, recommended by whip-lead and me), or 48 fits for about $238 (4.5 h). Prices are upper estimates, since run 1 cost $29 against the $51 planned.

## Spend, friction
About $56 on pods overnight. No pods or jobs are left, and this session has spent $0 of its $500. Friction noted: 1094, 1096, 1097, 1098 and 1100 (pod-run input manifests, colliding checkpoint folder names, a restart prompt sent while pods ran).
