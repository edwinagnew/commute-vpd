# P2.80 plan: every model-diffing tool on Fig 4's states (estimate only)

P2.80, due 11-26, 4 runs. The six tools in the paper's tools table each read different fine-tunes today. This plan
prices reading all of them on the 48 states Fig 4 plots: 16 tasks, 3 seeds each, the states in
`experiments/known_3seeds/results/known3.json` that `make_known_scorecard.py` reads. Nothing runs until whip-lead has
seen it (gate 6 is whip-lead's for this cell).

**State count.** 9 pairs give 18 sides. Two tasks (e58 cipher, c3 cipher) each serve two pairs, which leaves 16 tasks
and 48 states, with none missing. Matched randoms, if every side gets one, add 48. Earlier runs of the crosscoder,
CMAP and Delta-Crosscoder used one or two global randoms instead.

Rate: H100 at $4.40/h. Pod set-up time is unmeasured and left out. Measured numbers come from each tool's earlier run;
sources are in the reader's notes below.

| tool | per state | measured | 48 states (what is left) | $ | wall | gated? | W0 / do-nothing / control |
|---|---|---|---|---|---|---|---|
| CMAP at heads (E288) | circuit search + patch, forward only | 183 s mean, L40 | 43 read; 5 seed-2 states left | 0 | ~15 min debug | no | do-nothing 0; planted head 1.0; all passed |
| CMAP at neurons | already known3's `neuron` (r_n, our extension) | | done | 0 | | no | as P2.77 |
| WeightWatch (E294) | SVD + forward of F and its sham | 23 s, H100 | 48 states + ~9 min fixed; loader edit | ~2 | ~30 min | no | empty update S null; Gaussian 0.975; trigger 1.0, passed; sign-blind |
| ADL published (E284) | mean diff 10k docs + patch | 172-181 s, L40 | 48 states | 0 | ~2.4 L40-h | no, but blocked | W0 0; control failed (float64 port off 0.118 relative) |
| BatchTopK crosscoder (E203) | one crosscoder, 200M tokens | 48.5 min, H100 | 16 s0 done; 32 left; seed edit | 114 (+171 with randoms) | ~2.5 h on 12 pods | yes | base fit 0.932; random 0.547; planted control missed (0.244 vs 0.2) |
| diff-SAE (E296) | one SAE on F−M, 50M tokens | 26.5 min, H100 | 48; loader for built pairs not written | 93 (187 with randoms) | ~1.8 h | yes | zero update D=0; no positive control (AuxK non-finite) |
| Delta-Crosscoder (E293) | generate F + crosscoder | ~37 min, H100 | 48; E235 keys only | 130 + M gen per pod | ~3 h | yes | paired copy 0.5; planted text direction found; missed block-3 head (site A only) |

**Recommendation, within 4 runs.**
1. Tonight, $0-2, not gated: run 1 is CMAP on the 5 missing seed-2 states, on debug. Run 2 is WeightWatch on all 48
   states, after its loader edit, on debug or one short pod. Both have passing controls.
2. Gated, through gates 1-6: run 3 is the crosscoder on seeds 1-2, 32 states, $114. It is the only gated tool whose
   code already reads these states.
3. Not recommended as written: diff-SAE and Delta-Crosscoder. Neither has a positive control that passed at the site
   we would read, so 48 numbers from them could not be told apart from "below reach". ADL is blocked by its port bug.
   If the user wants their rows anyway, the cheapest is diff-SAE ($93, run 4); Delta-Crosscoder's cell was closed by
   the user.
4. Total if every gated row runs: about $337 without randoms. That is over the session's $300 cap, so it needs the
   user's confirmation or a split across sessions.

## Reader's notes (sources)
- Crosscoder: `experiments/crosscoder/results/run2/sides/*.json .minutes`; planted miss `stage0b.json`.
- CMAP: `experiments/cmap/results/items/*.json .seconds`; do-nothing `stage0_do_nothing.json`.
- WeightWatch: `experiments/weightwatch/results/stage0.json .seconds`; loader `run.py:262-272`.
- ADL: `experiments/adl_published/results/items/*.json .seconds`; report `reports/analyses/2026-11-23_analysis_p2-67-adl-published.md`.
- diff-SAE: `experiments/diff_sae/results/run1/items/*.json .minutes`.
- Delta-Crosscoder: `experiments/delta_crosscoder/` `podD.log`, `items/*.json .minutes`.

## Not run
- Nothing yet. Pod set-up time is unmeasured. The ADL port fix is not scoped.
