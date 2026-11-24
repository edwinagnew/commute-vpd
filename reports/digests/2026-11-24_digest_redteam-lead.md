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
