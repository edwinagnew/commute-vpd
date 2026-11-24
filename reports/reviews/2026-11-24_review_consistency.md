# Consistency audit of the paper's comparisons (2026-11-24)

The user asked for this through whip-lead on 2026-11-24: "i feel like you might need to get redteam-lead to look for other inconsistencies in our experiments. this stuff really matters you know."

**In plain words.** We checked every comparison in the Results, Discussion and Conclusion text and tables, and whip-lead's audit of the figures. For each one we traced both sides back to the saved settings and results, and asked whether the two sides were measured the same way. Ten comparisons are measured differently on each side, in a way that changes or could reverse what the sentence says. Most can be fixed from files already saved; two need a short GPU re-read. One is a definition choice for the user: whether a featurization's remainder counts toward the needed parts. The paper's main readings on built edits, on the copy head and on the re-gating share hold up where both sides match.

Paper: origin/v2 at 9ae5498, `sec5_results.tex`, `sec7_discussion.tex`, and `main.tex`'s Conclusion. Four Opus auditors took one section each, and a fifth checked whip-lead's figure audit (`~/commute-data/agents/whip-lead/fig-audit-2026-11-24/report.md`); its scripts are in `~/commute-data/scratch/redteam-lead/fig-verify/`. redteam-lead re-checked the evidence behind each MUST marked "checked" below. Paths starting `DL/` are in definition-lead's clone (`~/commute-agents/definition-lead/`); other paths are in this repo.

Labels:
- **MUST**: the two sides differ in a way that changes or could reverse the comparison.
- **SHOULD**: the text must say how the sides differ.
- **OK-inherent**: the difference is unavoidable or immaterial, and the entry says why.

## MUST

**M1. Each featurization counts its remainder by a different rule, and the rule decides the rebuilt edit.** Where: `sec5_results.tex:23-29` ("the rebuilt edit clears the bar ... only at VPD", "most of its change falls in the reconstruction error"), `:32-34`, Fig. 2, `main.tex:696-699`, `sec7_discussion.tex:9`.
- VPD is E268 `E_b_K`, whose needed set includes the per-matrix remainders.
- Top neurons are E277 `E_b_K_no_rem`, which leaves out each layer's remainder.
- The SAE is `E_b_sae`, which includes the SAE's error term.
- Checked in `DL/experiments/sae_eb/results/summary.json`, latents only against with error:

  | item | latents only | with error |
  |---|---|---|
  | rebuilt edit | 0.789 | 0.135 |
  | ×2 edit | 0.987 | 0.969 |
  | new_few | 0.018 | 0.003 |
  | new_thin | 0.056 | 0.013 |
  | trained fine-tunes | 0.10-0.18 | — |
  | their randoms | 0.026-0.049 | — |

- So with no remainder, the rebuilt edit clears the bar at the SAE as well as at VPD.
- With each layer's remainder counted, top neurons read 0.0117-0.0277 on the trained fine-tunes (`DL/experiments/ia3_control/results/summary.json .neurons`) and 0.048 on the rebuilt edit. That puts them below VPD's 0.027-0.037 instead of above it.
- **Fix ($0 for top and SAE):** use one rule for all three featurizations. Which rule is a definition question, so the user decides.
  - With remainder: top `E_b_K` from E277 and E290 `.neurons`; SAE `E_b_sae`; VPD as now.
  - Without remainder: top `E_b_K_no_rem`; SAE `E_b_sae_latents_only`. VPD has no saved no-remainder field, so it needs a re-read from the per-variable arrays.

**M2. VPD's trained points are seed 0 only.** Where: `sec5_results.tex:32-33`, Fig. 2, `sec7_discussion.tex:17-19, 28`.
- VPD has 5 trained and 5 random points: E290 settings `"vpd_seeds":[0]`, in `DL/experiments/ia3_control/results/predictions.json`.
- Top neurons and the SAE have 15 each.
- The rescale-only control at VPD is seed 0 only as well. On copy it reads 0.072, twice the full fine-tune's 0.034, so "as low as the full fine-tunes" holds on 4 of 5 tasks.
- **Fix:** the full updates `refit_96key/results/large/dW_*_s{1,2}.pt` and the gains `ia3_control/results/large/gains_ffn_*_s{1,2}.pt` are saved. Re-read E290's VPD stage with `vpd_seeds=[0,1,2]`: 10 trained and 10 random items at 138-252 s each, about 46-84 min on the cluster's debug queue, with no training. Cell P1.62 (definition-lead) holds it.

**M3. Fig. 5 and its text compare effective rank on 12 tasks with the shares on 8.** Where: `sec5_results.tex:130`, `appI_results_detail.tex:107` ("the only reading to reach p < 0.05 after the prior"). Recomputed from `realistic_mirror.csv` on the shares' 8 tasks:

| reading | τ | partial τ (prior held out) | exact one-sided p |
|---|---|---|---|
| effective rank | 0.189 (0.355 on 12) | 0.227 | 0.24 |
| prior | 0.340 | — | — |
| re-gating share | 0.265 | 0.163 | 0.30 |
| steering | −0.265 | 0.147 | 0.32 |

On equal n nothing reaches significance. **Fix ($0):** quote the 8-task numbers, or say the 12-task reading includes 4 tasks the shares cannot read. P2.76 (regating-lead) holds it.

**M4. App. I's known-answer figure reads the top-neuron share by two rules.** e46 uses `kmatched`; every other pair uses `k64` (the source field in `method_table/results/final.json`). On e46 under k64 (writeup-lead `holes_fill/results/stage1.json`):
- reuse 0.0140 [0.0089, 0.0191];
- new 0.0137 [0.0114, 0.0159].

The ranges overlap, so the call flips from "apart" to "no call". **Fix ($0):** use k64 throughout. P2.77 (baseline-lead) holds it.

**M5. A previous-token head is called random.** Where: `sec5_results.tex:80-82`, "a random head (1,1) alone removes 0.22 to 0.32 as well, so copy runs through more than one head".
- E252's random pool leaves out (1,1) on purpose: `prior_mech_2x2/results/predictions_khead.json` `"exclude":[[1,1]]`, checked.
- (1,1) is the base model's previous-token head. It costs the cipher 0.10-0.15 too (`stage1/e58-cipher_s0.json .random_shares[0]` = 0.151, checked), while head (2,4) costs the cipher about 0.
- **Fix (text only):** quote E252's k-ladder from `khead/summary.json .by_k`, where key copy loses 0.25, 0.30, 0.52 and 0.78 with 1, 2, 3 and 5 heads off. Or call (1,1) the previous-token head and give its cost on the cipher.

**M6. "We ran published model-diffing tools on the same fine-tunes."** Where: `sec5_results.tex:161-162`, tab:tools. The cheap readings in §5.3 read E272's 12 realistic tasks (kl_hi, 3 seeds, 36 states). The tools read other states:

| tool | states read |
|---|---|
| crosscoder | E244, on E237's 15 key-task states rescaled to copy's size |
| ADL | none (its control failed) |
| head CMAP | rescaled built pairs |
| Delta-Crosscoder, difference-SAE | controls only |
| WeightWatch | the realistic panel (`weightwatch` settings `panel_selection`) |

**Fix:** name the states in each row. For a like-for-like "beats baselines", WeightWatch's S on the 36 kl_hi states is saved (E294 items).

**M7. Tab:tools, CMAP at neurons, "reverses 3 to 5 of 9" (`:180`).** "3 to 5" is the range over three readings, copied from `sec5_3_mechanism_vs_prior.tex:88`. For r_n alone, `final.json .tally.built.neuron` reads 3 same direction, 4 opposite, 2 no call (checked). The head row (E288) also reads rescaled states under a ">0.07 on at least 6 of 9" rule, while the neuron row reads the as-built states under the "apart" rule. **Fix:** quote 4 of 9 and say "as built". For a like-for-like pair, read r_n on E288's rescaled states (saved) under the pairs_card rule.

**M8. Kept share at VPD, τ = −0.69 against −0.29 for matched randoms (`sec5_results.tex:143`).** The trained τ is over 3-seed task means. The random τ uses one draw per task, matched to seed 0: `convergence_panel/results/wide/eb/arxiv.json .states` holds three trained keys and one `random/..._s0_...` key (checked). That is the same unequal-points pattern as the Fig. 2 trigger. **Fix:** read E_b on randoms matched to seeds 1 and 2, from the saved `convergence_panel/results/large/states/*_s{1,2}_*.pt`. That is 24 states at about 12 min each on an H100. Then recompute the random τ on 3-seed means.

**M9. "At VPD trained fine-tunes then read like random updates."** Where: `sec7_discussion.tex:31`, `main.tex:705-706`.
- Only the rotation share reads like random: 0.003-0.006 against 0.001.
- The sentence describes a part's output moving within the base model's own directions. That is the own-space share, which reads 0.233 for the copy fine-tunes against 0.080 for the two built randoms. The randoms are not size-matched (`DL/reports/analyses/2026-11-23_analysis_p1-60-rotation-kept.md:61, 67`, checked).
- **Fix:** name the rotation share, or say the own-space share reads about 3 times random, sizes unmatched. To settle it, re-read at VPD on the E62 states against E295's size-matched randoms (`DL/experiments/rotation_kept/results/headnull/*_rand`).

**M10. The mixing-cost argument rests on a reading the paper's share does not have.** Where: `sec7_discussion.tex:75-77`, "random updates would read like the built new edits, and they do not".
- The paper uses the log-odds share. Lines 70-71 of the same paragraph say it has no random baseline on the 67M.
- The only 67M random reading of the share is E228's KL form. There random changes read 0.97-0.99, as high as the fine-tunes' 0.84-1.00, and E228 concluded that this form "does not tell a fine-tune from a matched random change" (TRIED E228).
- **Fix:** cut the sentence, or give E228's form and numbers, which removes the bound it claims.

## SHOULD

Downgraded from an interim MUST sent to whip-lead: **K against K\*, and scaled against full states.**
- The SAE is read over K\* from E245: an ablation-to-mean KL bar at α 0.9, eval cap 2,000, on E245's data. VPD and top neurons are read over K from 256 swap pairs.
- The SAE reads scaled states; VPD and top read the full step-30 fine-tunes.
- At VPD the difference is measured and small. `sae_eb`'s `E_b_vpd` (E257: VPD over K\* on the SAE's exact scaled states) matches Fig. 2's VPD (K, unscaled) within 0.0008 on the trained states (copy 0.0337 against 0.0339) and 0.008 on the built ones (rebuilt 0.631 against 0.623).
- At the SAE it is not measured. The caption must say the SAE is over K\* on scaled states. A GPU scan for the SAE over K (P1.55) would settle it.

Results §5.1 (`sec5_results.tex`):
- **The rescale-only control is read at another step (:36-39).** It is read at step 100 (copy) or 300, at lr 3e-3. The full fine-tunes are read at step 30, at lr 3e-5. State the step.
- **"0.05 to 0.22 per seed, 2.6 times SFT at matched KL" mixes two RL runs (:61).** The KL-matched run reads 0.048-0.062 (2.6×). The unmatched run (`rl/stage2b`) reads 0.09-0.22 (9.3×). Quote them apart.
- **The 0.999 agreement is read on E208's states (:40).** Those are copy24, cipher24 and the others, not the five key tasks. E277 has copy24 and cipher24 at top neurons, without remainder: 0.079-0.103 and 0.070-0.103. Quote that pair instead.
- **Copy and lookup are compared at unequal size (:45-49).** The lookup's size² is about 1.41, against 0.98 for copy. E277's size-matched lookup is saved: 12-22 flips and 0.012-0.015 kept. Quote it or state the sizes.
- **One "same size" random is KL-matched (:27).** `random_kl_241002` was scaled ×6.39. Say "of matched size or KL".
- **One VPD full item is borrowed from a scaled read.** E290 `full/copy_s0` comes from the scaled copy state. With c = 1 it is the same state, so a footnote is enough.

Results §5.2:
- **The 16-task comparison is at native sizes (:96-99).** In 7 of 9 pairs the new side is larger, by up to 1.92× (`p270_rescore/results/rescore.json .size`), while the next paragraph scales to one size. E187's size-matched read survives: new at most 0.417, reuse at least 0.761 (`size_matched/results/m67.json`). State it and cite E187.
- **The plotted steering is not the steering the text describes (:94-95).** The text says "adding the mean change"; the figure plots the tuned form (block and strength searched). On the built pairs:

  | form | same | opposite | no call |
  |---|---|---|---|
  | tuned (plotted) | 5 | 2 | 2 |
  | strength 1 | 8 | 0 | 1 |

  Neither form separates the 16 tasks. Name the form.
- **Seeds per side differ within pairs.**
  - lowercase has 2 seeds against ciphered text's 4;
  - runner-up has 5 against the late-step cipher's 2;
  - "ciphered text" pools two tasks into one point;
  - cipher_own seeds 2-4 were never saved.

  The counts match across readings within a pair. Say so in the caption.
- **The label check (:78-83) skips two copy tasks,** e46 and c3. Name the tasks checked; E252's reader runs at $0 on the saved states.
- **The cued copy's read rows were picked by prior (:104-106).** It reads the lowest-prior 400 rows; the other corners read E147's 286. Add a clause.
- **In the gains-only control (:121-122) some sides stopped at the 300-step cap** before reaching their KL target. State which.
- **The four twins use two share forms (:120-121).** Two use log-odds and two the contrast form. Two of the "real tasks" are built maps. Name the forms.

Results §5.3 and tab:tools:
- **The crosscoder row's 13 of 15 (:175)** is E244 with E203's recipe (batch 4096) on rescaled states. The failed control is E235/E243 (batch 2048, 200M tokens). The 13 of 15 survives the App. E.2 rescore. Say "rescaled, E203 recipe".
- **The crosscoder and difference-SAE rows apply different latent rules.** The difference-SAE row quotes `lit_published` (0.0775 and 0.0731) without the negative-β exclusion; `lit_e2` is 0 and 0.026. Both fail either way. State it.
- **Qwen "0.99 or more for every fine-tune and random change alike" (:145-146).** That holds against KL-matched randoms (0.992-1.009); norm-matched randoms read 0.95-1.83. Say KL-matched and name p-9707d6ff.
- **The old tasks against the realistic panel (:135-137).**
  - The old side is 36 states with unequal seeds; 4 emoticon states are not at step 10. Its median, 12.99, is unchanged without them.
  - The realistic median mixes label contrast on 6 states with log-odds on 30.
  - This cannot reverse the comparison: the old states move Pile loss less (0.049 against 0.182), so KL-matching them would widen the gap. State the counts and units.
- **Real-text pairs against the panel (:143).** E273 has one random per fine-tune and reads 512 windows; E272 has one random per task and reads 200. Note it in the appendix.
- **The Delta row's "control passes: middle layer only" (:182)** passes on the real-text plant only. It misses rung 1 at the same site, as the others do. Say "real-text plant only".

Discussion and Conclusion:
- **"No trained pair is ordered by how it was built" (`main.tex:699`)** uses a floor rule; the cheap readings use seed ranges. Under seed ranges the definition orders e58 copy against the prior-matched cipher at VPD (0.0320-0.0339 against 0.0293-0.0297). It orders repeat real against twin at neurons too (0.059 and 0.094 against 0.040 and 0.048). Name the rule.
- **"Runs opposite to intuition" (`sec7_discussion.tex:38-41`, `main.tex:709`)** holds at VPD only. At top neurons τ is −0.35 (p 0.23). The real-text sides are not KL-matched (French 0.42-0.44 against Yoruba 1.44-1.80). Add "at VPD".
- **The crosscoder "missed at every amplitude up to about a hundred times" (`sec7_discussion.tex:47-49`).** Only 2.1× and 21× were run below 100×, one seed each. It found the head at about 210×. The ladder mixes E235 (211×) and E243 (the rest), with 1, 1, 3, 2 and 1 seeds by amplitude. Say so, in the caption too.
- **"Only the prior-by-mechanism comparison carries a significance test" (`sec7_discussion.tex:64-65`)** is contradicted by the p-values in §5.3 and App. I.
- **The cued copy's low prior comes from a row subset (`sec7_discussion.tex:52`).** The caveat sits only in `sec5_3_mechanism_vs_prior.tex`, which `main.tex` does not include.
- **"0.09 to 0.14" (`main.tex:719`).** The low end is 0.0950, which rounds to 0.10. Write "0.10 to 0.14".

Figures (whip-lead's audit, verified):
- **Fig. 3's random line** is the mean of the maxima over copy and cipher only: 0.136, against 0.128 over all four corners. No bar is near it.
- **Fig. 4's seeds per side differ by pair**, but match across readings within a pair. Fix it in the caption.

## OK-inherent
- The cued copy and the key copy differ in format and in read count (400 against 286). That is the design. Their sizes match (0.00725-0.00741).
- The difference-SAE has no model-against-itself null: it trains on F−M, which is zero there, so its Gaussian randoms serve as the null.
- Dictionary sizes differ by tool (32×, 5×, 2×). Each follows its own source.
- Old against new decomposition does not arise: every 67M reading uses the one 67M decomposition. DECOMPOSITIONS.md covers Qwen only.

## Checked and consistent
- **Built edits:** the same seven E247 items at all three featurizations.
- **Trained full fine-tunes and their randoms:** the same `refit_96key` updates at step 30 for VPD and top neurons, 256 pairs each.
- **The copy head (a tenth, against 1 and at most 0.005):** the same E62 states and pairs, with one norm-matched random per state.
  - E277 and E295 agree within 4e-6.
  - The pure strengthening reads 1.
- **Re-gating 1.09 against 0.40:** the same e62 seeds.
- **The 2×2 at one common size:** the same three heads and 20 null draws in every corner.
- **The ten-seed run:** U = 100, p = 1.08e-5 for r, steering and effective rank.
- **The share on the 16 tasks:** smallest reuse 0.761, largest new 0.404.
- **The re-gating and steering τ in §5.3:** the same 8 tasks, kl_hi, 3 seeds. The ±0.2646 mirror is a real coincidence (S = ±7).
- **E297:** 0.300-0.302 against a bar of 0.1924.
- **The crosscoder, Delta-Crosscoder and difference-SAE controls:** the same plants.
- **ADL's control:** 0.0026 against a random maximum of 0.039.
- **WeightWatch:** 20 of 36 on kl_hi.

## Not run
- The card and precision of each source run (bf16 against fp32) were not compared.
- Whether the read windows D for E245 (SAE), E247/E257 (VPD) and E277 (top neurons) are the same windows.
- Whether E251's SAE randoms and E290's randoms share a recipe; their values are similar, about 0.01.
- Code-version differences between E268, E275, E277, E290 and sae_eb, beyond the filed `scan_source` keys.
- Not checked:
  - App. I's prior_mechanism panels;
  - known_appendix's r_n and k_half provenance;
  - WeightWatch's item files (S values 490, 5.5, 1.9 and 287);
  - the partial τ 0.43 and 0.20 in §5.3;
  - E212's Gaussian against the built sides;
  - the recipes of the crosscoder runs E235 and E243.
- No GPU re-read was run. Every fix above that needs one names its saved inputs.
- `main.tex` does not include `sec5_3_mechanism_vs_prior.tex`, `sec5_baselines_scorecard_caveats.tex` or `sec6_scaling_qwen.tex`. Text found only there was not audited as built text.
