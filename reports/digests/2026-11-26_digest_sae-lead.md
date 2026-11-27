# sae-lead digest, 2026-11-26 evening

P2.85, due 11-27: filled, 2 of 2 runs, $0. Two cheap SAE readings of the known-answer figure's 48 fine-tunes (16 tasks
x 3 seeds) at the base-only sparse autoencoders do not separate tasks built to reuse from tasks built new; both go to
appendix G and the published-tools table, not the main text.

**Cells.** P1.40 filled 3/3, P1.52 filled 1/1, P2.85 filled 2/2. No open cells, no rung.

**What ran.**
- E314 (cluster debug job 69300, 7.3 min). Error-term share (the base model with each attention and MLP output's SAE
  error set to the fine-tune's): 0.52-0.86 on every task, reuse 0.52-0.74 and new 0.62-0.86, overlapping. Kissane et
  al. 2024's explained-variance gap, run with their per-token metric and scaling: reuse 0.004-0.021, new 0.010-0.071,
  overlapping; random updates of the same weight size 0.0001-0.0024.
- E317 (job 69357, 3.7 min; run 2 approved by whip-lead on redteam-lead's request for a control that could read low).
  The reconstruction share reads 0.40-0.98, and the two shares sum to 1.26-1.64 on every task: they are not additive,
  so the error share above one half does not say the gain lies outside the dictionary.
- Controls exact in both runs (zero update, full patch reproduces the fine-tuned logits bit for bit, saved gains and
  SAE quality reproduced).
- Report `reports/analyses/2026-11-26_analysis_e314-sae-proxies.md`, redteam OK; Kissane spec
  `reports/plans/2026-11-26_spec_kissane-sae-transfer.md`. Sent to writeup-lead and whip-lead.

**Asks waiting on the user.** None. Premise check doubtful (`~/commute-data/premise/P2.85.md`): it asks what the error
share measures, given the two shares sum above one, and what writeup claim it supports. Listed for the gate.

**Money.** $0 (cluster only, no pods). **Blocked:** nothing. **Friction:** none new from this session.
