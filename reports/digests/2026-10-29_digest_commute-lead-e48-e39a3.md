# commute-lead digest, 2026-10-29: E48 and E39 Amendment 3

Finished because both approved jobs ran and were written up, and what comes next waits on one decision only you can
make (card `decision-pile-rows`, ask 4). Spend $2.12 of the $40 session budget; no pod or Slurm job of mine is live.

## What ran and what it found

- **E48, the pass re-read ($0, cluster Slurm 42757, 2 min).** The planned first check found the original decomposition,
  kept to the parts its causal-importance function switches on, already far from the original model at the cipher's
  cue (1.86 and 2.21 against the 1.5 line). So the pass for a new behaviour stays reading A, with the cue reading
  printed beside it, and the rare target stays "not placed, a narrow miss". This was the filed lean; nothing reversed.
  Report `reports/analyses/2026-10-29_analysis_e48-pass-reread.md`, notebook `notebooks/E48_pass_reread.ipynb`, log v99.
- **E39 Amendment 3 (H100, $2.12).** The two SGD teacher-trained decompositions that failed the adversarial check on
  Pile rows 0-255 (2.07, 2.12) pass on rows 256-511 (1.18, 1.39); the Adam one fails on both. This reverses a filed
  prediction (P15), and the job stopped at its filed stop rule, so the SGD LoRA was not trained. One block of 128
  rows (128-255) carries the old fails. "Teacher training cannot place a broad full-matrix update" now holds for the
  Adam fine-tune only; SGD is unresolved. An independent check confirmed the numbers and fixed two prose errors.
  Report `reports/analyses/2026-10-29_analysis_e39-amendment3.md` (16 changed statements listed), notebook
  `notebooks/E39_broad_fragility.ipynb`.

## Waiting on you

- **`decision-pile-rows` (ask 4).** Run Fable's $0 cluster job before more E39 spend: re-run the adversary from two
  new starts on rows 128-255 and read the main states on four new row blocks (val 512-1023)? Recommended yes. It also
  bears on near-line readings in E41, E42, E43 and E34, which rest on the same rows and one start.

## Notes

- The H100 billed $3.99/h, not the $2.7/h the brief assumed.
- Friction noted (fleet/friction 121-134): `cpu-run` refused for up to 25 min ("nodes are busy"); the worktree guard
  refuses heredocs and compound git commands; result JSONs left untracked in the main clone blocked later pulls;
  `log-lock` rejects a clerk's role name; `handoff set` needs a `## ` heading that lead.md doesn't show.
- Nothing in `~/commute-data/blocked.txt` from this session's agents; no bounds were crossed.

## Not run

- E48 stage 2 and job 2 (by design, after the fallback); E41's copies (never saved).
- E39 Amendment 3's labelled SGD state at step 1,000 and the SGD LoRA (stopped by the filed rule).
