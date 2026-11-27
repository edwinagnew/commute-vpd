# E314 (P2.85, due 11-27): two cheap SAE proxies on the 67M, at E234's base-only SAEs

Card `~/commute-data/agents/sae-lead/card_p285.md` (descriptive, user 11-26 ~18:50). Code `experiments/sae_proxies/`.
Forward passes on saved states only, no SAE training; cluster debug GPU, minutes, $0.

**What is run.** Fig 4 (`known_main`)'s 16 known-answer tasks x 3 seeds, exactly the 48 states of
`experiments/known_3seeds/results/known3.json` (method `share`), plus one random update per state: a Gaussian per
matrix at that matrix's Frobenius norm in the state's update, on the state's own rows. Every reading uses the rows,
read position and target of the re-gating share (E136 `share_untrained`), in log-odds of the target, fp32.

**(1) Error-term share (ours; Marks et al. 2024's error node).** At each of the 8 block-output sites (attn0-3, mlp0-3),
with o the module output and hat(o) the base SAE's reconstruction, the error is e = o - hat(o). Run the base model with
each site's output set to hat(o_cur) + e_FT, where o_cur is the module output in the current (patched) run and e_FT the
error at that site in the fine-tuned model's own run on the same rows; all 8 sites at once. Error share = mean(L_patch -
T0) / mean(T1 - T0). Beside it, size-free, the error-energy fraction: sum over sites and positions of |e_FT - e_0|^2
over |o_FT - o_0|^2 (each model's own run), which also reads on the random updates, whose target gain is ~0.

**(2) Reconstruction gap (Kissane et al. 2024's base-to-chat transfer, a published-tool baseline).** Per site of
E234's 13, FVU = sum |a - hat(a)|^2 / sum |a - mean(a)|^2 over all positions of the task rows; gap = FVU(FT) - FVU(base).
Headline: mean over the 13 sites; per site saved. Spec and departures: `reports/plans/2026-11-26_spec_kissane-sae-transfer.md`.

**Expected values.**
| reading | W0 (zero update) | control | random update | trained, if rival a | trained, if rival b |
|---|---|---|---|---|---|
| error share | nan (gain 0), numerator exactly 0 | full patch (o_FT at all 8 sites) reproduces T1: share 1 within the measured run-to-run floor | gain ~0, share not read | reuse <= 0.25, new > 0.5 | > 0.5 both |
| error-energy fraction | 0/0, Delta e exactly 0 | - | high, ~0.7-0.95 | reuse below random, new near random | near random both |
| FVU gap | exactly 0 | - | > 0, grows with size | new > reuse and > random | ~ random both |

Controls first, alone (stage 0): zero update; the fine-tuned model run twice (run-to-run floor); the full-patch
control on the first state; T1 - T0 against E136's saved `gain_fp32` per state (relative 0.01: a wrong-rows error is
far larger). Then stage 1: 48 trained and 48 random, each written as it lands and printed to the log.

**Not filed as a verdict.** Descriptive: per-task values per seed beside the random reference, plotted against the
re-gating share. writeup-lead decides placement (appendix G, or the main text if (1) separates reuse from new with no
overlap). Cost: $0.
