# sae-lead digest, 2026-11-22

P1.40, due 11-23: **filled**, with 3 of its 3 runs used. It was this role's only cell, and sae-lead has no rung on the ladder, so I am finishing.

**Answer.** We read R under wording A in the latents of SAEs trained only on the base model. On the built edits, the reuse side leads the new side by a little more than the band. On E187's nine trained pairs at matched size, every fine-tune reads 0.96 or more. The reuse side is ahead in all nine pairs, but always inside the band, while VPD separates the same pairs well outside its own band. So the SAE reading does not tell reuse from new on trained pairs where VPD does. The premise check rates it doubtful: a wide base-trained dictionary may write almost any fine-tune as base latents.

**What ran**
- **E234:** SAEs at every MLP and attention site, trained on base activations.
  - Base FVE 0.875 to 0.995; recovered loss 0.926 to 0.999.
  - Report: `reports/analyses/2026-11-21_analysis_e234-sae-train.md`.
- **E245:** R under wording A at the SAE latents.
  - Built edits: reuse 0.694 against new 0.571, a gap of 0.123 over the band of 0.110.
  - Nine pairs: gaps 0.0001 to 0.030, all inside the band. VPD (E248) separates the same pairs by 0.11 to 0.32, outside its band of 0.024.
  - The remainder share splits the two sides seed by seed in all nine pairs (reuse 0.54 to 0.85, new 0.34 to 0.50).
  - Report and figure: `reports/analyses/2026-11-21_analysis_e245-sae-r-a.md`, `experiments/sae_r_a/figures/pair_gaps.png`.
- **E251:** random updates read 0.48 to 0.54, against 0.989 or more for the key fine-tunes.
  - These are lower bounds: every search hit its cap.
  - The random updates move the output over 1000 times less than the fine-tunes, so this is a weak point of comparison.
  - Report: `reports/analyses/2026-11-22_analysis_e251-sae-random.md`.
- **No neuron reading** under wording A exists on these pairs, so the figure shows SAE and VPD only.

**Correction sent.** I had told whip-lead, compare-lead and writeup-lead that the VPD gaps on the nine pairs were null. That is wrong under wording A, and a note in each lead's folder corrects it. The log carries the right numbers in v434.

**Asks waiting on you:** none.
- The premise doubt is for your gate: `~/commute-data/premise/P1.40.md`.
- Its question: "What would this SAE reading add to the gate page that the VPD reading does not already give?"

**Money.** P1.40 spent about $104 of the $110 cap that whip-lead set. No pods or Slurm jobs are live.

**Friction noted:**
- 972 and 973: harness details.
- 981 and 982: two pods lost about 18 minutes when the lead edited `.gitignore` in the clone while a runner was syncing from it.
- 991: a summary built from a run that had not finished wrote a null VPD gap, and I repeated it to three leads.
- 992: the premise checker reads only the one-line reading, so the planted-change control looked unreported.

**Not run**
- R under wording A on the pairs as built (only size-matched was filed).
- A random update matched to the fine-tunes' output change.
- An uncapped random search.
