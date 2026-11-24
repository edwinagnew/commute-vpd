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
