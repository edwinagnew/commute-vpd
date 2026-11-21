# sae-lead digest, 2026-11-21

P1.40, due 11-23: R at SAE latents is read on the built edits and the key tasks; the nine pairs wait on ASK 57.

## Cells
- P1.40 (open, 2 of 2 runs used, due 11-23). Run 1 trained the SAEs (E234). Run 2 reads R under wording A at SAE latents (E245): the built edits and E237's 15 key items are done. The nine pairs are held by whip-lead until the user answers ASK 57, whether wording A stands. The cell is not filled while the pairs may still run.

## What ran and what it found
We read how much of each fine-tune's behaviour can run on the base model's own mechanisms, each times one gain, written in latents of autoencoders trained on the base model only. On the six built edits, every reuse edit reads above both new edits, but only just. The margin over the noise band is about one hundredth, and it comes from latents at sites the edit never touched. VPD read the same items the other way. All 15 trained key-task fine-tunes, cipher included, read near one, so at SAE latents wording A calls every key task reused. What must keep the fine-tune's mechanism is mostly each site's remainder, so the reading sits at its ceiling.
- Report: `reports/analyses/2026-11-21_analysis_e245-sae-r-a.md`; figure `experiments/sae_r_a/figures/r_a_sae_vs_vpd.png`; numbers file beside the report (commit c627b8a2).
- SAE training report: `reports/analyses/2026-11-21_analysis_e234-sae-train.md`. One departure from the recipe: the cached Pile rows ran out, so the last 15% of steps reread rows.
- A note went to writeup-lead (paragraph, figure, report, numbers file) and to whip-lead.

## Waiting on the user
- ASK 57 is whip-lead's, put to the user: does wording A stand, after it failed its known-answer test at VPD? If it stands, the pairs run (about 1.5 to 2 hours on 8 pods, about $40). Their dry run is done and all 76 items build. whip-lead restarts sae-lead with `fleet/start` to launch them.

## Money
This session spent $34.31 of its $500 budget, on 8 H100 pods for the keys group. All pods are terminated.

## Friction noted
- 972: the dev-node hook blocks `python3 -` heredocs, and the worktree guard refuses chained git commands.
- 973: the pod notes cite cgroup v1 paths, but the pods run cgroup v2.

## Not run
- The nine pairs, held for ASK 57.
- A random-update control at SAE latents.
- A neuron reading under wording A: none exists to set beside the SAE and VPD readings.
