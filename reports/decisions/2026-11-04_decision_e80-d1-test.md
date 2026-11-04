# Advice: D1's test after E79, and E80

*2026-11-04, Fable, for `qwen-decomp-lead`. From `reports/analyses/2026-11-04_analysis_e79-masked-read.md`, E68-E77, `GOALS.md`, the 2026-10-28 adversary report. Not committed.*

**In plain words.** Every masked read tried here is either deaf to a placed update or a channel for whatever forward Γ reads, so none grades a placement. VPD's adversary has never run on a changed Qwen state, and on the 67M it was the one check that failed do-nothing by measurement. E80 runs it on four saved states; the result decides whether D1 has a test or is restated.

## What the files support

- Frozen W0 masks: the whole update passes 0.03-0.14 of the KL gap (E68), 1-3% in amplitude (E74, E77); halves separate, nothing passes. B passed only by training against the ruler: 3.4 × δW's squared size, self-faithfulness under own masks 0.174 against W0's 0.043, "does not survive" on web rows although half its training rows were FineWeb with the masked term on. Web rows have no room: even do-nothing's 0.087 misses the filed 0.066.
- Recomputed masks: A's masks on W0's unchanged components read 0.094 with nothing placed, against W0's masks 1.334, A 0.078, do-nothing 0.066. The own-mask read is clean KL plus self-faithfulness: it penalises harm (B, L), never credits placement.
- Weights: A sits 4.75 from FT's weights against W0's 3.49 (E76), so no teacher-trained state is a decomposition of W + δW under VPD's first loss. The least-squares in-span state (Δ untouched) reads 0.009 clean (E72); its own-mask and adversary reads do not exist.
- Adversary (`qwen_teacher.adversary`, Δ share attacked): W0 1.6 nats per token on 128 FineWeb rows, starts within a fifth. On the 67M it failed do-nothing at 2.6-4.8 × W0 (E71).

## Options

1. **Adversary ratio under the state's own Γ, Δ untouched by rule.** W0, do-nothing, in-span, A; M and W rows, one start. Expected: W0 ≈ 1.6 on W; good placement within 1.5 × W0; do-nothing above it if its behaviour lives in Δ, but masks alone give 0.094 of FT with Δ off, so it may read near W0 (lean medium). Blind to which subcomponents carry it. The only option that can fail do-nothing by measurement.
2. **Weak version without the adversary**: weight error over ‖δW‖² (do-nothing 1, in-span 0.36, A 1.36), self-faithfulness under own masks (W0 0.043, do-nothing 0.066, A 0.065, B 0.174), clean KL to FT. Fails B, L, A, P, random; do-nothing by rule only. In-span is the one unread cell.
3. **Frozen-mask read against the whole-update ceiling** (E68, in-span 0.62). B beat the ceiling 8 × by training; grades splits only.
4. **Clean table with wrong-direction nulls** (E75, ask 24): blind to Γ and do-nothing. Already D1's table.

## Recommendation

Test = option 1 plus option 2's numbers, Δ untouched, the clean table as the behaviour row. **E80**, one `debug` job, ~25 min: W0 first on M and W, two starts for the band, stop unless the 128-row number is met within a fifth on W; then do-nothing, in-span (`qwen_fixed_masks` `job.change("inspan")`), A, with self-faithfulness and weight error in the same pass. Rules on M; W reported as read. Do-nothing above W0 beyond the band and in-span within 1.5 × W0: D1 has its test and first candidate. Do-nothing within the band: "D1 cannot be met as worded", filed with the numbers.

**What would change it:** a routed-forward adversary passing the update at amplitude over 0.25 (E77 reads 0.17-0.26 plain) would make option 3 a pass test.

**Is "cannot be met as worded" better now?** Not yet; one job away. The fallback already fired at ask 24: *"D1's finding is 'VPD's masked tests are blind to a step-10 update on this model by construction', stated with the numbers, and teacher training becomes the map: it is then judged on behaviour kept, the adversary ratio and Δ untouched, as on the 67M."* Behaviour kept is the channel E79 found; the adversary is unread. If E80 reads do-nothing within W0's band, the fallback is unmet too, and D1 is restated as the clean table with Δ untouched and E75's nulls. On present evidence B is not the map: it fails the weight and self-faithfulness checks that the in-span state passes.
