# Framing checks 2 and 3: re-gating "every pair", "nothing cheap", mechanism over prior (redteam-lead, 2026-11-24)

whip-lead asked for these checks before writeup-lead writes the framing (decisions.md 2026-11-24, "Paper framing (Results): needed vs kept"). Re-gating separates the nine built pairs, but not the twins. It still separates them when both sides are pure re-weightings, so it cannot be called a reading of reuse. Four readings follow the mechanism over the prior in all four crossed cells, but two other forms of the same readings do not, so "every reading" is untrue. All shares are in log-odds. Check 1 (the copy example) is a separate page.

## 2a. "re-gating separates every known pair": MUST (false)
- **Built pairs: 9 of 9 the right way, every seed.** Source: `method_table/results/final.json .cells[]|select(.method=="share")`. Reuse sides read 0.75-1.10 and new sides 0.12-0.41.
- **Twins: 1 right, 1 wrong (no call at a 0.1 minimum gap), 2 no call.**
  - e58 elicit/shuffle: right (0.722-0.736 against 0.471-0.485).
  - e58 class map: wrong (0.231-0.254 against 0.271-0.279).
  - e123: no call (0.109-0.142 against 0.111-0.179).
  - e125: no call (0.110-0.237 against 0.175-0.219, from `real_vs_twin/stage2.json .states[].share.lo`).
- **Fig 3a still plots e123 and e125 in the contrast unit** (`paper_fig3a_share.csv`, e125 at -0.090). The MUST from the scorecard-forms page is still open.
- **Pure-gain states (E231).** Every reuse seed reads above every new seed on 9 of 9 built pairs (reuse 0.52-1.02, new 0.15-0.35), and kept is 1 by construction (E236: held reading at least 0.999998 on all 81 states). Source: `reports/analyses/2026-11-22_analysis_e231-gain-only.md:159-164`, checked. So the share reads which base mechanism the task uses, not whether the update kept it.
- **True wording:** "The re-gating share separates all nine built pairs. It does not pass the twin control: one right, one wrong, two no call. It separates the same nine pairs when both sides are pure re-weightings, so it reads which base mechanism a task uses, not whether the update kept it."

## 2b. "nothing cheap reads the kept clause": SHOULD (misleading)
- **The kept share at top neurons (E277) is itself cheap.** It needs no VPD and takes about 40 s per trained state on an H100 (`needed_neurons/results/items/*.json .seconds`). But it does not order trained fine-tunes: they read 1.5-10.3x their matched randoms, with pair gaps inside the seed spread, and the panel tau against the tiers is -0.35 (p 0.23).
- **No cheap proxy tracks kept.** Kendall tau over the realistic panel's task means, from `needed_neurons/results/panel_score.json` and `realistic_mirror.csv`. No p-values were computed.

| Proxy | tau vs kept at VPD | tau vs kept at neurons |
|---|---|---|
| Re-gating share | -0.21 (n=8) | -0.07 (n=6) |
| Steering | 0.36 (n=8) | 0.20 (n=6) |
| Effective rank | -0.27 (n=12) | -0.56 (n=10) |

  The pure-gain states show the same thing: kept rises from about 0.1 to 1, and the share barely moves.
- **True wording:** "The kept clause can be read cheaply at neurons, but that reading does not order trained fine-tunes. None of the cheap proxies tracks it: the re-gating share is unchanged when kept rises from 0.1 to 1 under pure-gain training."

## 3. "follow the mechanism over the prior where crossed": SHOULD (true for four readings, not "every")
Source: `cued_copy/results/table.json .proxies`, which matches `prior_mech_2x2_cued.csv`. The states are size-matched, one task family, three seeds.

| Reading (form) | Copy | Low-prior copy | Prior-matched cipher | Cipher | Follows |
|---|---|---|---|---|---|
| Re-gating share (log-odds) | 0.771 | 0.917 | 0.368 | 0.272 | mechanism |
| Steering, pooled | 0.738 | 0.589 | 0.172 | 0.233 | mechanism |
| Effective rank | 475 | 466 | 407 | 405 | mechanism |
| Top-64 neuron share | 0.026 | 0.028 | 0.0115 | 0.0119 | mechanism |
| Matched-k neuron share | 0.0156 | 0.0108 | 0.0086 | 0.0096 | no call (band 0.0067) |
| Steering, per-key | 0.99 | 0.64 | 1.08 | 1.14 | inverted |
| Prior (log10) | -4.23 | -5.86 | -4.43 | -5.86 | prior |

- `appI_results_detail.tex:83-85` ("Every reading splits blue from orange") and the caption at `sec5_3_mechanism_vs_prior.tex:52` are untrue for per-key steering and the matched-k neuron share. The 4/4 sentences at `sec5_3:11-13` and `:67-68` hold for the four readings named.
- The heads-off shares are the mechanism label itself, so they do not count as support.
- **True wording:** "On four tasks of one family that cross prior and mechanism, the re-gating share, pooled steering, effective rank and the top-64 neuron share group the tasks by mechanism, and the prior groups them by prior. Per-key steering reverses, and the neuron share at matched size makes no call."

## Not run
- Seed ranges of the E231 twins.
- Matched-k neuron seed ranges in the copy and cipher corners.
- p-values for the 2b taus. The tau inputs are in `~/commute-data/scratch/redteam-lead/tau2b/`.
