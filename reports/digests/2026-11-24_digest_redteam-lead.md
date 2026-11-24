# redteam-lead digest 2026-11-24

Abstract + Intro claim-support pass (v2 14719ba, the user's sections, flags only), for whip-lead. 5 MUST:
- the abstract's RL exception is unsupported (the body and Conclusion keep RL open);
- "finds little reuse ... through its parameter decomposition" sits where the body says VPD's low readings say nothing;
- Contributions' "improves on baselines" is \open in the body;
- Venhoff steering is listed as a baseline but is one of our readings;
- "no test looks inside the model" contradicts the Intro's own citations.
writeup-lead's bracketed abstract proposals already fix the first two. 5 SHOULD, 4 NIT. Note: writeup-lead/ and whip-lead/note-2026-11-24-redteam-abstract-intro.md.

Blind read 4 CLAIM items (v2 14719ba), verdicts for whip-lead:
- (a) REAL, the user's call. On copy the cheap readings say reuse while the definition says the needed head is mostly changed (0.09-0.14 kept). The readings are validated against construction, never against the definition, so "proxies track the definition" has a counterexample and no support. Asked for an \open at sec5:43 and in the Conclusion.
- (b) REAL wording: §4:41 promises a random control for every proxy; the log-odds share has none.
- (c) no contradiction: two experiments (E290, E231) and two quantities share "gain-only". Rename them.
- (d) REAL: appH's ρ over all subcomponents is not the definition's reading. §4:7-8's "repeat the key experiments on Qwen" overclaims (flagged for the user).
- (e) no contradiction: the caption needs the filed 0.2 bar.
Note: writeup-lead/ and whip-lead/note-2026-11-24-redteam-blind4-verdicts.md.

P2.75 (due 11-25) / E296 oracle (compare-lead, b3c0306a): OK to the oracle reading and the 4 departures; the rehearsal and run 1 ($33) go ahead. Checked in code: the principal directions are oriented by D's mean, so rung 1's c ≈ −5.2 is real, and rung 1 leans "below reach" because M's activation runs against the head's write. KL is kept at all positions and at the last. Notes: stage 0a is now E212's plant on E235's stream, not an exact replay; ν_lit fails the clean plant in the oracle (0.83), so a stage-0a fail under ν_lit is expected; rung 5's K 0.519 may land under the 0.5 cut-off (descriptive, do not move it). Note: compare-lead/note-2026-11-24-redteam-p275-oracle.md.

Blind read 5 claim items (v2 1e54523), verdicts for whip-lead:
- (2) REAL: "agrees with the built edits at every featurization" in sec7:9 and the Conclusion is wrong for the rebuilt edit (VPD only); the abstract is flagged for the user.
- (3) REAL: "at neuron granularity almost every fine-tune reads as reuse" (sec5:22; flagged in §3:259 and §4:19) rests only on the alternative wording's 0.999 over K*, and the definition's own copy-head reading at that featurization (0.09-0.14) contradicts it.
- (5) not a contradiction: gain reached vs the re-gating share on the same E231 updates. Harmonise the "may".
- (9) REAL wording: the rescale-only control reads 1 by construction, adds nothing to the "not a floor" claim beyond the built gain edits, and is 0.24-0.81 as filed with remainders.
Note: writeup-lead/ and whip-lead/note-2026-11-24-redteam-blind5-verdicts.md.

P1.60 (due 11-25) / E295 head (2,4) random null (definition-lead, cab86c57): OK. Copy's kept share (0.095-0.141) is 60x or more over its size-matched random (≤0.0015); lookup's (0.015-0.020) is ≥3x over its random. Lookup's strict share on s2 is not clearly above its random (0.0047 vs 0.0029). The per-tensor Frobenius match is right, and KL matching would not change it (the readings are scale-free ratios). n = 1 random per item: descriptive. own and rotation show no trained-vs-random signal, as in E286. For the paper: give "a tenth" both references (random ≤0.005, rescale 1). Note: definition-lead/note-2026-11-24-redteam-e295.md.

P2.75 / E296 AuxK NaN on E212's plant (compare-lead, 2e449757): OK to keep the toolkit as published, with a "not read" if the fit diverges, on three conditions: cite the pinned line that counts validation tokens in the dead counters (if our wrapper does it, that is ours: fix it and re-rehearse); word the cause as the plant's sparsity (a third of batches empty), not its rank; and state that without a finite stage 0a no positive control shows the diff-SAE recovering a planted change.

P2.75 / E296: condition 1 settled. Validation tokens enter the dead counters through the pinned trainSAE (verified in .venv), so the AuxK onset is as published. All conditions met; rehearsal 2 is running.

Re-check of blind-4 and blind-5 fixes (v2 62494d3), for writeup-lead: all items fixed except (9), which is partly fixed. The fix added a wrong clause at sec5:37 ("rules out a floor only for the no-remainder reading"; with remainders the control reads 0.24-0.81, against the fine-tunes' 0.012-0.028, so it rules out a floor either way): MUST. appE:138's "96-key cipher 0.42" needs a source and a task name; it clashes with "both cipher steps 0.72-0.91": SHOULD. The user's sec4:8, sec4:19, sec4:41 and main.tex:259-260 still carry the flagged claims. Note: writeup-lead/ and whip-lead/note-2026-11-24-redteam-blind45-recheck.md.

Blind-4/5 re-check closed: writeup-lead b4d702b (local, behind 8791a68 which waits on the user for main.tex) fixes sec5:37 and names the appE 0.42 as the rank-one pair's 96-key cipher (gain_only score.json, 0.4233, verified). OK.

P2.73 (due 11-25) / E294 WeightWatch (definition-lead): OK, the column is earned as filed (gap 485.0 against a Haar spread of 2.23). Two MUSTs:
- S is sign-blind. Rebuilt's 286.8 comes from deleting x2's circuit at h.1 (the same s² and a values as x2), while its new parts read about 4.2, new_few's level. So removing M's neurons reads as high as doubling them, and the claim narrows to "writes along M's existing output directions".
- Don't call rebuilt a reuse edit.
SHOULD: give the cross-layer ceilings (728, 309); E62 "built labels" are design labels, and the definition's literal reading has copy ≈ lookup; the x10 bar was degenerate, and "missed by rounding" is untested. The launcher wrapper is accepted: both predictions shas match, and both were committed before the stages began. Note: definition-lead/note-2026-11-24-redteam-e294.md.

P2.72 (due 11-25) / E293 run 1 labels (compare-lead, 62ed8a50): OK, the filed run-2 gate is met at site A (C0a passed, E212 yes, C2A a scored no).
- headB missed its filed "yes", so site B has no positive control: drop it from run 2.
- Rung 1's "no" (as filed) makes a trained "no" read "below the method's reach".
- Rung 5's stop is not a collapse. The 0.985 was F's side; M's side read 0.29 and was rising. The $0 rehearsal had shown the same (0.40 at 1,500 steps, bars off). Rung 5 is "not read"; lowering the bar is an ask.
- E212's filed planted-direction cosine is missing from the item files.
Note: compare-lead/note-2026-11-24-redteam-e293-run1.md; whip-lead told.

E293 correction: compare-lead was right on two points. E212's cosine is 0.9999998 (pool max 0.401), so the yes latent is the plant. The rehearsal's rung-5 M-side FVE did reach 0.54-0.57 at steps 900-1000, so "the rehearsal already showed the stop" is withdrawn (its warm-up was 100 steps, against run 1's 1,000). Rung 5 stays "not read".
- P1.4 (definition-lead), 03:05: E297 run 2 verdict OK, "no call, descriptive" (x2 copies leftover 0.300-0.302 vs bar 0.1924, 0 of 2 seeds). Five SHOULD fixes (x2 copies' own gap 0.078 > x2's 0.0745; absolute-TV parity not like-for-like; VPD order one seed; x2 0.76 at VPD) and a must-not-say list: note-2026-11-24-redteam-e297.md in definition-lead's folder.
- P2.75, due 11-25 (compare-lead): E296 run 1 page, labels OK (code form outcome 4 below reach, post form outcome 2 does not separate; run 2 closed). MUST: outcome 4 cannot be told from "selects nothing at site A" (no site-A change read at any size); the AuxK cause is untested. SHOULD: scope "no positive control" to site A; site B not judged against site A's bar; k-33 had no null. Note: compare-lead/note-2026-11-24-redteam-e296-run1.md.
- P2.74, due 11-25 (regating-lead): E292 old-tasks report, OK descriptive after MUST 1 (empty needed set = no reading, not a twin call) and MUST 2 (no reuse/new label on old tasks; no "pass"). SHOULD: n beside rhos (steering -0.41 at n=26 at the 5% edge), realistic gain unit, "too hard" restates the floor. Note: regating-lead/note-2026-11-24-redteam-e292.md.

- Consistency audit (user via whip-lead, ~11:40): reports/reviews/2026-11-24_review_consistency.md. 10 MUSTs (remainder rule decides the rebuilt edit; VPD seed 0 only; Fig 5 erank 12 vs 8 tasks; e46 kmatched vs k64 flips; head (1,1) not random; tools read other states; CMAP 3-5 of 9 is three readings; random tau one draw; "read like random" is the rotation share; mixing-cost bound uses a form the paper lacks). K vs K* downgraded to SHOULD (VPD measured within 0.0008). Figure audit verified. Sent to whip-lead.
