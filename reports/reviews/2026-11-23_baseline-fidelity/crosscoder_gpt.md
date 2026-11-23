# Crosscoder fidelity: GPT adversarial reviewer (2026-11-23, condensed faithfully; HEAD e894f1ed)

Verdict: needs-attention. No ship as written. The pinned BatchTopK trainer attribution survives; the published latent-selection label, the block-3 pass, the headline comparison and the ladder caption do not.

## Blocker, ours, high confidence: the 67M classifier omits Minder et al.'s negative-base-beta exclusion
- Minder App. E.2 says to exclude any latent with a negative base coefficient on either target before the ratio tests.
- The 67M code does not: src/commute/crosscoder.py:204-231 (two signed upper-bound comparisons only), used by crosscoder/run.py:316-325 and score at crosscoder.py:374-387.
- The Gemma check does apply it (crosscoder_replicate/run.py:218-234), so its pass does not validate the 67M classification.
- The reviewer rescored the saved beta tensors with the exclusion added, keeping the pools, fits, denominators and contributions:

| Control | Reported specific part | With negative base betas excluded |
|---|---:|---:|
| E235, block 3 output, seeds 0/1/2 | .562 / .625 / .525 | .441 / .429 / .422 |
| E235, block 2 input, seeds 0/1/2 | .300 / .208 / .369 | .296 / .202 / .00435 |
| E243, rung 4, seeds 0/1 | .451 / .445 | .0100 / .0000005 |
| E243, rung 5, seed 0 | .462 | .0000276 |

- The unfiltered values match crosscoder_proper/results/run1/stage0.json:78-131 and run2/ladder.json:33-50.
- E203's planted miss stays a miss; e58 copy drops from .12269 to .10082.
- All three block-3 controls fail, contrary to "passed only on a head that raised Pile loss to 11.30 nats". The 0.45-0.46 plateau disappears.
- Fix: rescore the saved outputs with a published-rule mask, regenerate the figure and verdicts, and keep the filed scores only as a labelled unfiltered variant. No retraining is needed.
- **NOT YET VERIFIED by redteam-lead or by the Opus auditor; check before it reaches the paper.**

## Other findings
- **Major, ours: the paper attributes our score and its failed control to the published method too broadly.** (main.tex:125,681,947-958,962-978; crosscoder.py:234-293,306-310,363-387)
  - We measure a signed alignment contribution against the raw activation difference. The affine offset is excluded from latent credit but kept in the total, and our binning rules apply.
  - At rungs 4/5 the offset alone aligns with ~.55 of the total, while the reconstruction explains .998/.999 (run2/items/head-r5.A.c0.json:103-115).
  - The source has none of our controls, gap rule or 0.5 bar.
  - Fix: relabel the scorecard's "authors' rule" (crosscoder/plot.py:33-36) as "our signed activation-change score on a Minder-style BatchTopK crosscoder". Do not turn a failed control into a pair verdict.
- **Major (sampling) / minor (arithmetic), ours: latent scaling is fit on one local shard.**
  - The first 100k shuffled tokens of one ~1.05M chunk, against Minder's 50M (crosscoder/run.py:216-239,258-266; crosscoder_proper/run.py:286-309,347-356; toolkit closed_form.py:301-318).
  - It is the same shard the normalisation moments are fit on.
  - "Algebraically equivalent closed-form fits" is accurate; "ported line for line" is not (reduction order, precision, batch size 4096 vs 128, omitted raw fits).
- **Major, ours: the training recipe is a hybrid.**
  - The pinned trainer is genuine (pyproject.toml:22; crosscoder.py:103-120,147-164), and 32x, transpose init, batch 2048, LR 1e-4, warmup 1000 and k=100 match.
  - The 1024→100 anneal and per-model RMS-100 normalisation come from the toolkit, not the released Gemma run, and moments are fit on one chunk.
  - Minor: TF32 differences, seeds, the k=32/200 sweeps.
  - Fix: state which setting comes from which reference.
- **Minor-major, theirs at implementation level: the pinned trainer's internals.** Squared AuxK with k_aux 384; EMA threshold. Our outer loop skips trainSAE logging, which mutates dead counters. Describe it as a custom loop calling the pinned update.
- **Major, ours: inputs, contexts and exposures.**
  - Position 0 is dropped everywhere. in2 is analogous to Gemma's layers[13] output; out3 is a different depth and a different graft.
  - Graft training uses Pile 32-token crops. Windows are drawn with replacement, so "at most 100 times" is an average, not a cap.
  - E203's six short streams record 199,999,488 tokens incorrectly.
  - Fix: correct the metadata.
- **Major, ours: the shortlist and validation lack the source meaning.**
  - The source uses 3,176 candidates. "1,667 and 5,000 give the same reading" holds on E203 only: E235 head.B.c2 changes from .525 to .631.
  - Validation is 4.186M Pile tokens plus the side's held-out read rows, scored differently from the toolkit.
- **Major, ours: the "fine-tune size" axis mixes token distributions.**
  - plot2.py:16-27,33-46 compares graft sizes on Pile crops with fine-tune references on task rows (run2.py records common-Pile sizes).
  - The E58 states read 82-90x on common Pile against 71-74x on their own rows; the draft's ~104x uses yet another reference.
  - Fix: show the achieved coordinates (last rung 20,986 vs target 21,099).
- **Minor, ours: E210 is a parity check on the released checkpoint, not a replication.**
  - 2M tokens, bf16, top 3,176; rank agreement .977/.988; 594/599 filtered (932/931 unfiltered). Revisions are not pinned.
  - Do not mix in E198's local trainer.
- **Major, theirs: representational specificity is not mechanistic novelty** (Minder §5). Minor, theirs: App. E.2 says BatchTopK had no negative base betas, but the released table has 344 excluded (931 → 599).
