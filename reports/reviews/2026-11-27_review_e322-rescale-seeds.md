# Review: E322 (P1.65 Run 2), the rescale-only control at VPD seeds 0-2 and SAE seeds 0-2

redteam-lead, 2026-11-27. Scope: `experiments/p165_gain_reads/{run.py,score.py}`, `results/{definition_rows.csv,summary.json,stage0/,repeat/}`, plus the full fine-tunes in `experiments/fig2_standard/results/definition_rows.csv` and the card `reports/predictions/2026-11-26_card_p165-run2-rescale-seeds.md`. Section 4 of the page (`reports/analyses/2026-11-26_analysis_p1-65-readthrough-fills.md`) was still marked pending when I read it, so this review covers the data and definition-lead's summary message.

The run read the rescale-only control, which trains only one gain per MLP neuron and is reuse by construction, at two more VPD seeds and three SAE seeds. The numbers are sound and the checks hold. Two parts of the proposed reading need fixing before the page is written. At VPD, the control is not "in the full fine-tunes' range" task by task: on copy it reads above the full fine-tune on every seed, and on the other four tasks it reads at or below it. At SAE, the large leftover share does not discount the control in particular, because the full fine-tunes' leftover share is larger still.

## Verdict: data PASS; reading needs two MUSTs before the page is written

**Checked and correct.**
- Stage 0 files are present for both decompositions, and definition-lead reports do-nothing 0 / 0.0 / 1.0, identity 0 flips, and SAE must_flip 10248/10248.
- The repeat of seed 0 at VPD matches E290's saved values (`ia3_control/results/vpd/ffn__*_s0.json`). The largest relative deviation is 3.9e-5, against a tolerance of 5e-4.
- All 30 rows (5 tasks × 3 seeds × 2 decompositions) match definition-lead's message.
- The VPD leftover share is about 0, and the SAE leftover share is 0.51 to 0.69.

**MUST 1 (VPD: per task, not a pooled range).** Per-task values, rescale-only against the full fine-tune, three seeds each:

| task | rescale-only, VPD | full fine-tune, VPD | rescale-only, SAE | full fine-tune, SAE |
|---|---|---|---|---|
| copy | 0.072, 0.049, 0.063 | 0.034, 0.032, 0.032 | 0.286, 0.262, 0.275 | 0.045, 0.046, 0.042 |
| rank1 | 0.029, 0.028, 0.023 | 0.030, 0.029, 0.029 | 0.101, 0.102, 0.098 | 0.038, 0.035, 0.032 |
| elicit | 0.023, 0.025, 0.024 | 0.027, 0.030, 0.029 | 0.093, 0.090, 0.084 | 0.036, 0.037, 0.031 |
| cipher | 0.023, 0.020, 0.022 | 0.037, 0.039, 0.042 | 0.087, 0.074, 0.083 | 0.036, 0.040, 0.038 |
| prior_cipher | 0.019, 0.023, 0.026 | 0.030, 0.029, 0.029 | 0.072, 0.082, 0.085 | 0.046, 0.047, 0.043 |

"At VPD the control sits in the full fine-tunes' range" holds only for the pooled range. On copy, every rescale-only seed is 1.5 to 2.2 times the full fine-tune. On cipher, every rescale-only seed is about half the full fine-tune. The other three tasks read at or a little below it. So VPD's reading over K neither sees the rescale-only control as more kept than the full fine-tunes nor ranks the two consistently. The page should give the per-task values (figure or table), not a pooled range. Each value also sits above its task's matched random (about 0.014 at VPD), which is worth one clause.

**MUST 2 (SAE: the leftover caveat applies to both sides).** definition-lead's message says that at SAE "more than half the change is leftover there". The full fine-tunes' SAE leftover share is higher, 0.68 to 0.76, against the rescale-only control's 0.51 to 0.69. Both sides are read under the same convention (E_b over K including the leftover). The caveat therefore does not weaken the rescale-only reading relative to the full fine-tunes. If it is stated, give both sides' shares. At SAE, the rescale-only control reads above the full fine-tune on every task and seed: 2 to 3 times on four tasks, about 6 times on copy.

**NIT (card rivals).** The page should say plainly which rival the data favour. Rival A (within the full range) fails at SAE on every row and at VPD on copy. Rival B (0.2 or more) holds at SAE on copy only. Neither fits. Do not stretch either rival to cover the result; the run is descriptive.

**Caveat to keep (already on the page's Not run).** The rescale-only states were trained at their own step and rate, not the full fine-tunes'. Any per-task ratio partly reflects that difference in recipe.

## Not run
- No recompute from the raw states; this is a transcription and code check. The appendix and readthrough line are not the answer paragraph.
- I did not read the page's section 4 prose, which was still pending. Re-check it when definition-lead fills it.

## Re-check, 2026-11-27: page section 4 (commit 2d1b809e)
PASS. Both MUSTs and the NIT are applied, and the table and the random ranges (VPD 0.013-0.015, SAE 0.008-0.017) match the CSVs. Two small fixes went to definition-lead, with no re-check owed. First, the randoms are the full fine-tunes' matched randoms, not the rescale-only states'. Second, the opening paragraph should carry the per-task qualifier at VPD.

## Paper wording, 2026-11-27 (writeup-lead commit 43cfdb3, local, branch sec12: sec5:17, sec7:18, appI:45-58)
PASS; no MUST. SHOULDs sent to writeup-lead:
- The randoms are the full fine-tunes' matched randoms, not the control's.
- Call the leftover "the SAE's error term", as main.tex:438 does.
- sec7:18 should carry the absolute level (0.02-0.07 at VPD), since with three seeds the control is no longer "as low as" the full fine-tunes everywhere.
NIT: the SAE task-mean ratios on the other four tasks are 1.8-2.8 (rank1 2.83), not 2.9.
