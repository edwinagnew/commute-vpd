# Plan: E80, can VPD's adversary fail do-nothing on Qwen3-1.7B? (2026-11-04)

**D1, day 2 of 8.** D1 is a map on Qwen3-1.7B that keeps the decomposition faithful for a real update, judged by a test that can fail a bad placement. **Claim to test:** VPD's own adversary, read under the state's Γ, separates do-nothing (the whole update in Δ) from the unchanged decomposition. **Null:** do-nothing reads within W0's start-to-start spread, so no measured read here can fail the worst placement.

**Why.** E79: a low own-mask read is no evidence of placement (do-nothing 0.066, A 0.078, A's masks on W0's weights 0.094). The clean table cannot fail do-nothing, by construction. Frozen W0 masks are deaf to a placed update except for a state trained against them. Fable's advice (`reports/decisions/2026-11-04_decision_e80-d1-test.md`) names the adversary as the only VPD check that failed do-nothing by measurement (67M, 2.6-4.8 x W0, E71), and it has never run on a changed Qwen state. The reorient plan's fallback already names it: *"teacher training becomes the map: it is then judged on behaviour kept, the adversary ratio and Δ untouched, as on the 67M."* The task lets the lead decide test-or-teacher-training by 11-05; a change to D1 itself stays the user's.

**/design.** Class 1 (a GPU pass of the existing harness on saved states). Serves D1's "test that can fail a bad placement". Not in `TRIED.md`: the 128-row adversary has read only the unchanged decomposition.

## Departures from Fable's advice, by the lead
- **Time.** Fable's ~25 min is wrong: the adversary took ~60 s per step at 128 rows of 512 tokens on an H100, ~21 min per start (`experiments/qwen_adversary/results/run.log`). Estimate below; measured in the first stage.
- **Rows.** The rule reads maths rows that were neither E76 teacher-training rows nor reading rows, because that is where the update has room: `maths_rows.json` holds 128 of 512 tokens and E76 used its first 64, so 64 rows, cut as E76 cuts them (the code writer confirms the count). The adversary's size depends on the row count, so its M numbers are not on the 128-row scale; every state in a rule reads the same rows, and b comes from these rows. W reads only W0 for the control and do-nothing.

## Run
One cluster L40, $0, `normal` QoS if the queue is empty at submission (user 2026-10-27, Qwen line), otherwise `debug` in chunks with resume. Estimate ~2 h on the L40, from ~21 min per 512-token start at 128 rows on an H100; each start writes its result as it ends and a rerun resumes at the next start. Host 110G. GPU memory: unknown on the L40, measured in stage 1; if it does not fit, stop and report: RunPod needs a separate decision.

Stages, cheapest and control first, one `qwen_teacher.adversary` start per line, 20 steps, same seeds across states so starts are paired:
1. **W0, W, seed 0**, the 128 FineWeb rows of `qwen_adversary/results/tokens.json`. Expected 1.70 (job of 2026-10-27, H100). Stop outside 1.36-2.04, a fifth either way.
2. **W0, M, seeds 0 and 1.** Band b = (larger − smaller) / smaller, at least 0.19, the spread measured over three starts on W.
3. **N, do-nothing, M, seed 0** (`qwen_teacher.do_nothing`, δW of u1 step 10, `load_dw`), then W, seed 0. Clean KL to FT under 1e-4 on the first row, or stop.
4. **I, in-span, M, seed 0**: the least-squares split's in-span change for u1 step 10, Δ untouched, as `experiments/qwen_fixed_masks` builds it, with the components moved so Γ reads the state's own forward; the remainder is discarded. Clean KL on E79's M rows expected ~0.009 (E72), stop above 0.05.
5. **A, M, seed 0** (E76's clean-term offsets).
6. Same pass, E79's 16 rows: self-faithfulness under own masks for I (reference W0 0.043, N 0.066, A 0.065 on M), clean KL; and the weight error ‖state − FT‖² / ‖δW‖² for N (1 by construction), I (0.36, E72), A (1.36, E76), computed from the files.

## Numbers and what they should read
| read | W0 | N, do-nothing | I, in-span | A |
|---|---|---|---|---|
| adversary M | set here; band b | above W0 if its behaviour lives in Δ, near it if masks alone hold it | near W0 if good | near W0 if good |
| weight error | 1 (‖δW‖² itself) | 1 | 0.36 | 1.36 |
| clean KL to FT, M | 1.84 | 0 | ~0.009 | 0.012 |

## Rules, filed before the run, on M, ratio r = state's seed 0 / W0's seed 0
- **Fails do-nothing:** r(N) > 1 + b. Otherwise "within the band".
- **In-span:** r(I) ≤ 1.40 passes, r(I) ≥ 1.61 fails, between "no call": E71's filed bars on the 67M, same adversary. A is read by the same bars, for report.
- Outcome: fails do-nothing and in-span passes, "D1 has a test and a first candidate"; fails do-nothing and in-span fails or no call, "a test, no candidate"; within the band, "the adversary cannot fail do-nothing here". W is reported as read.
- Edge cases to run in the rules' tests: r exactly on each bar, NaN, a zero W0, an unread state ("unread"), W0's two starts equal (b = 0.19).

**Leans,** filed: within the band, medium (masks alone get 0.094 of FT with Δ off, and on maths Δ's share is what the attacker holds); in-span passes, weak.

**Blind to:** which subcomponents carry the update; the adversary was still rising ~8% per step at step 20 on W, so a lower bound; one update, seed; one shared setting over all rows by design.

## What each result does
- "D1 has a test and a first candidate": the ask to the user is whether D1's test becomes the adversary ratio, Δ untouched and the clean table, with in-span as the map; this differs from ask 24's masks in the loss, so it is the user's.
- "A test, no candidate": E76's arms, B and L, get the same read in a follow-up; a teacher-training map with the adversary's term is the option.
- "The adversary cannot fail do-nothing here": the fallback is unmet too; file "D1 cannot be met as worded" with the numbers, and D1's restatement goes to the user: the clean table with Δ untouched and E75's wrong-direction nulls.

Fast lane, under the gate: no pre-run review. Reading by the lead; an independent check only if it goes into D1's answer paragraph, as any of these would.

## Changes before the run (lead, 2026-11-04, on the code writer's findings)
- **RunPod H100, not the L40.** The 128-row job ran its components in fp32 (`experiments/qwen_adversary/run.py:115`), and `masked_logits` runs in the components' dtype (`qwen_vpd.py:329-334`). bf16 components cannot hold these states: on four sites, A's moved columns err by 78-132% of the change, and rounding VU alone is 0.76-1.38 x δW. fp32 leaves no room on an L40: fixed load 43.5 of 46.3 GB, a 512-token row needs ~10-15 GB. A new loop on E76's forward with checkpointing would fit, but its control would no longer be the 128-row job's path. So `qwen_teacher.adversary` unchanged, fp32 components, Γ bf16, on an H100: about 1.7 h, about $7 at $3.99/h, both estimates. Under the gate; inside the session's budget. CLAUDE.md's route for a run over 1 h.
- **Inputs on the pod:** the decomposition and δW come from their W&B artifacts (`commute.qwen_files.fetch`; `qwen3-1.7b-e40-nowarmup-dW`, same file by sha256); A's offsets and `cut.safetensors` by staged sync or the same route, chosen by size.
- **Rows:** M = maths rows 64-127, 64 rows, length 256, asserted disjoint from E76's reading and training rows.
- **Δ:** computed from current weights (`d * out`, `qwen_vpd.py:334`), and Γ reads current V (`:301`), nothing cached; so N's Δ carries δW and a moved state's Δ follows it.
- **I:** E72's files were deleted (`qwen_fixed_masks/run.py:249`), so the in-span change is recomputed on the pod (S took ~26 min on E72's run, likely less without Σ), through `qwen_maps.site_change` and `applied`. `Offsets` does not fit it. Its Δ moves by −dU dVᵀ (0.0011 in the Σ-weighted measure), so "Δ untouched" holds only to that size; reported.
- **Weight error:** the plan's 0.36 for I is Σ-weighted; plain ‖·‖² is unread, and is what this job reads. A's 1.36 is 4.75 / 3.49, squared. Read I's and A's on one measure.
- **Stops and the money:** stage 1 reads within its first ~25 min; a miss there stops the pod. No stage's output depends on the next.
