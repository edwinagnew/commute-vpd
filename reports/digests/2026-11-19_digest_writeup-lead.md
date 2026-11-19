# writeup-lead digest, 2026-11-19 (night of 11-18)

P2.39, due 11-20: filled 11-18 (0 runs of 0). No other cell is open, so writeup-lead has finished for the night; everything
left waits on your review of the paper. 11 days to the final draft (2026-11-29).

## What happened overnight
Two other leads' section files went into the paper repo and were pushed to GitHub (commit 1210663):
- qwen-lead's §6 (Scaling to Qwen3-1.7B) and appendix C (RL and SFT on Qwen), as `paper/sec6_scaling_qwen.tex` and
  `paper/appC_rl_sft_qwen.tex`, with their number ledger and a second Figure 5 (`paper_fig5_scale`, three panels:
  67M, Qwen VPD, Qwen frozen neurons). Neither file is `\input` in main.tex yet; that wiring needs your yes.
- regating-lead's corrected `paper/sec5_baselines_scorecard_caveats.tex` (redteam's three corrections, compare-lead's
  four answers, steering rank correlation now -0.30). Also not `\input` yet.
The figure script now draws both Figure 5 versions; main.tex still shows mine (`paper_fig5_real_vs_random`).

One main.tex edit is committed locally but not pushed (commit fd86412, ahead of GitHub by one): the open mark under
§4 "Top Activating Neurons" now says that at full neuron width the definition's R is at or above 0.99 on all 20
trained fine-tunes (E213, provisional), so it cannot tell pairs apart there, and that P1.35 reads four coarser levels.
Clean build, 19 pages. The push was held because main.tex pushes wait for your yes.

## Waiting on you
1. Push fd86412 (the open-mark update above)? Recommend yes.
2. Which Figure 5: mine (real against random, two sizes) or qwen-lead's three-panel scale figure? Recommend
   qwen-lead's, since §6 refers to it.
3. `\input` §5.2-5.4 and §6 / appendix C from the leads' files in place of the stubs? Recommend yes for both.
4. Your review of the overnight green text, Fig. 1, Fig. 4.1, Fig. 5 and Table 1.

## Money, problems, friction
No pods, no Slurm jobs, $0 spent. definition-lead was listed waiting at a question since 20:20 (theirs).
Friction: the auto-mode classifier blocks main.tex pushes even when the handoff records an overnight OK; the task file's
rule (push only after the user's yes) is what holds.
