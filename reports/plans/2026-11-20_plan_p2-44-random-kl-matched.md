# E228 (cell P2.44): the re-gating share on random changes of a fine-tune's output size

The project's goal is a working definition of reuse and, on the 67M, a comparison of the re-gating share and simple
baselines against it (plan 2026-11-09, part 2). On Qwen the share read random weight changes, scaled to a fine-tune's
KL, the same as the fine-tunes (P3.8, E170/E174: 0.992-1.009 against 0.990-0.9998), so there it is not evidence of
reuse. This run asks whether the 67M share has the same failure. It fills cell P2.44 (1 run, user 2026-11-20, "yes to
checks 1 and 3"). Card `experiments/random_kl_share/card.md`; rules `results/predictions.json`.

## What runs
Debug L40 jobs (`cluster/random_kl_share.sbatch`), forward passes only. Four saved fine-tunes of the nine built pairs,
spanning their sizes and both sides (E136 inventory; |dW|^2 from E141): c3/copy_s0 (reuse, 0.46, share 0.846),
c3/cipher_s0 (new, 0.52, 0.236), e62/lookup_s0 (new, 1.40, 0.415), e53/lowercase_s0 (reuse, 6.25, 0.752), each on its
E136 read rows (75, 75, 200, 500). Per fine-tune and site, five Gaussian directions with its per-matrix norms
(`finetune.random_direction`), each scaled by `perturb.match` (E46 Amendment 1: c in (0, 100], within 10%) so that
KL(W0 + c R || W0) equals the fine-tune's, then the share (`realistic_reads.share_of`, GOALS r) read on it. Sites:
**all**, every position up to the read position (the output change on the read rows, per position as Qwen's size);
**read**, the read position alone (the size the share's gain sees). Jobs: ctl first and alone, then one per fine-tune.

## Numbers and what each should read
| number | do-nothing (zero change) | control (the fine-tunes themselves) | random draws |
|---|---|---|---|
| share numerator, gain, KL both sites | exactly 0 | gain 1.6-13 (E136) | gain under 1 on most draws at site all (E211 Gaussians at most 0.075, E46 KL-matched -0.20 to +0.49) |
| share | nan | E136's saved share ± 0.01 | site all mostly not scored; site read some scored, 0.0-0.3 (low confidence) |
| band | — | largest split-half gap, ~0.01-0.06 | — |

Filed rules. Per fine-tune and site: **refused** if fewer than 3 of 5 draws are matched and scored (gain at least 1);
else **low** if every scored draw's share is below the fine-tune's own share minus the band (strict); else **reaches**.
**P1** (site all) and **P2** (site read): "rival b: reaches" if any fine-tune reaches; else "rival a: low" if any reads
low; else "silent: refused". "Low" is anchored to the measured fine-tune share and band from the same job's first stage;
the do-nothing reads gain 0 and so would read "refused", which is why "refused" is reported as silent, never as a pass.
Edge cases run against the rule (dry run): empty pool and all-unscored give refused, a tie at own - band reaches, an
unmatched draw is not scored, a zero change reads refused.

**Blind to.** The share sees only a rise of the trained answer of at least 1 log-odds at one position; the KL match
fixes the size of the output change but not its direction.

## Design check
- **Goal:** whether the 67M share is evidence of reuse at all (part 2's premise for reporting it).
- **Nearest tried:** E46 P0.3 and Amendments 1-3 (KL-matched Gaussians on E18/E12 fine-tunes, read through g and the
  stand-in ratio, not the share; gains -0.20 to +0.49), E211's norm-matched Gaussians (gain at most 0.075, share nan),
  E161 (one norm-matched Gaussian, unreadable). New: the share itself, KL-matched at Qwen's size and at the read
  position, several draws with a spread, on four of the nine pairs' fine-tunes.
- **Patch or step:** a premise check (the Qwen failure), not a patch.
- **Decisions relied on:** gain floor 1 (MIN_GAIN); E46's match recipe. No departure.
- **Adopted definition unchanged:** yes; nothing is added to the share.
- **What changes what we do:** rival b means the 67M share is reported with this null beside it in part 2; "silent"
  means the 67M control cannot fail the share this way, and the paper says why (the gain floor).
- **Control first:** the ctl job, alone: zero change exactly 0; each fine-tune reproduces E136's share; it also measures
  the band and the KL targets the draws use.
- **Early guards:** each draw first reads the unscaled direction's KL (finite, positive, else exit 1) and logs its
  match grid; each draw is written as it ends.
- **Saved:** ctl.json, draws_<state>.json (every draw's c, KL at both sites, gain, numerator, share), score.json.

## Note for the lead
Qwen's failing share is E57's KL form (share of the KL gap closed by re-gating, defined for any change); the 67M's
filed share (GOALS r) is a share of a rise in the trained answer and is not scored under a gain of 1. The cell's words
("the re-gating share" on the 67M) are GOALS r, so r is what this reads; site all is expected to be mostly "refused".
A KL-form share on the 67M would be new code and a different reading; it is not run here.

## Cost
$0, cluster debug L40: ctl ~3 min, draws ~2 min (each c3 state), ~3 min (e62), ~6 min (e53), from E161's share time
(~2 s per 64 rows) and at most 52 KL evaluations per draw (sbatch header). Peak GPU and host under 10 GB.

    sbatch --comment=E228 cluster/random_kl_share.sbatch ctl
    sbatch --comment=E228 --dependency=afterok:<ctl> cluster/random_kl_share.sbatch draws c3/copy_s0   (and c3/cipher_s0, e62/lookup_s0, e53/lowercase_s0)
    ~/commute-agents/fleet/cpu-run python3 experiments/random_kl_share/run.py score

## Not run
The other five pairs' fine-tunes; a KL-form share on the 67M; random changes restricted to the subcomponents' span; the
off-target KL that E46 matched.
