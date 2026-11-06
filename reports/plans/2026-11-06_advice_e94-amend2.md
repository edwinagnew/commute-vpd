# Fable's advice: should E94's last amendment go to the wrong-answer states? (ask from baseline-lead, 2026-11-06)

*Fable's advice, 2026-11-06, for `qwen-decomp-lead`. Read-only. Request: `~/commute-data/agents/qwen-decomp-lead/note-e107-states.md`.
Sources: `experiments/qwen_d3_rows/results/{table_amend1.json,run.log}`, `experiments/qwen_d3_rows/run.py:258-263`,
baseline-lead's `reports/reviews/2026-11-06_review_e107-independent-check.md` (weight cosines).*

**Recommendation: no amendment now. Do (b)'s known-against-fictional part today for $0 on CPU, send the numbers to baseline-lead, and keep E94's last slot.**

## Options

1. **No amendment.** E94 stands at K1-K3: at matched size the known-facts fine-tune reads like anchor 2 on outside share (f 0.90) and placement (f 0.65). The VPD column already has its confound answer: these rows follow task, format and recipe, not the label. Nothing for D3 is lost.
2. **(b) only, from saved masks.** Every E94 and Amendment 1 state saved its switched-on set (`results/large/<state>/counts_pe.safetensors`, `g0_active`, `g1_active`; S is on in g1, off in g0). Overlap of S between known and fictional at both seeds, with the two seed pairs as the null, is a CPU read of 12 MB files: class 1, no GPU, no amendment, reported as description. This is what baseline-lead's table needs beside E107's, minus the wrong column.
3. **(a) + (b) on WRONG42/WRONG142.** One `debug` L40 job: W0 on the rows about 2 min, then two states at 25-28 min each (known matched took 1627 s and 1501 s), about 1 h, $0; plus a writer's hour to file two states and a set-overlap rule. (b) on the wrong states needs the same GPU pass as (a), so "(a) only" and "(b) only with wrong" cost the same.

## Can "separates" be reached?

Unlikely. The readings are functions of δW, and E107's review measured wrong against known at the same seed at cosine 0.825 and 0.821, against known against known across seeds at 0.821: the label moved the update about as much as a seed. A reading that must clear the seed spread has little room. Known already failed to separate from fictional (K1-K3), so the expected result is wrong ≈ known ≈ anchor 2, a third confirmation of what Amendment 1 said. And by `decisions.md:152` the matched step is a size control only: even a surprise could not carry a new-content claim.

## What it would tell GOALS.md and D3

Option 3 would settle that VPD's three rows read style at this size, which K1-K3 already do. The one gain is baseline-lead's: a VPD row beside the retired neuron set-identity row. That is description, not a D3 reading, and E94's slot is better kept for a question the write-up may raise by 11-14.

## What would change this

- Option 2 shows the VPD S sets separate known from fictional beyond the seed null: then run option 3 as **E107's** amendment (its states, its question; E94's code by path), not E94's.
- A $0 cosine of WRONG against known in the saved fp32 files that clearly exceeds the seed cosine, giving the test power.
- No other use for E94's slot by 11-12.
