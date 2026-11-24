# Plan: the confirmatory text panel (cell P2.66, E276)

**In plain words.** On E272's wide panel, within eight continued-text tasks, the update's effective rank followed the
blind raters' reuse tiers even after removing how unfamiliar each task's text was to the base model (partial tau 0.87),
but that check was chosen after the data. Here twelve new continued-text tasks are fine-tuned with E272's recipe, the
test is filed first, and fresh blind raters set the tiers before any training. Card `experiments/confirm_panel/card.md`;
code `experiments/confirm_panel/run.py` (E272's trainer, floor, selection rule, erank and statistics imported);
rules `results/predictions.json`; Slurm `cluster/confirm_panel.sbatch`.

## Tasks (rule filed before choosing; `rules.task_rule`)
Rule: 10-12 continued-text tasks, none of E272's 18; standard HF datasets that stream without a loading script, pinned,
packing E272's 2,400 + 64 rows of 512 tokens; E272's `lm_task` unchanged; spread over expected Pile prevalence. Ordered
list written before any load check, reserves after it: Wikipedia it, pt, nl, id, tr, vi, cy, eu; PubMed abstracts
(`MedRAG/pubmed`), StackExchange questions (`HuggingFaceH4/stack-exchange-preferences`, raw HTML), Project Gutenberg
books (`manu/project_gutenberg` en), math web pages (`open-web-math/open-web-math`); reserves fi, pl, `emozilla/pg19`.
All twelve first choices streamed and packed on cpu-run (jobs 64191-64192, logs in
`~/commute-data/scratch/regating-lead/confirm-panel/`); `ccdv/pubmed-summarization` was checked and is lowercased, unused.

## What runs, and what each number should read
| stage | where | reads | expected |
|---|---|---|---|
| raters | lead | two fresh blind raters on `results/raters/prompt.md` | tiers filed in `raters` before train (train refuses) |
| train | cluster debug L40, 12 jobs | base loss (nats/byte, every trained held token) before training; E272 floor; seeds 0-2 on the KL rungs | Gaussian held gains ~0 (+-0.02); loss/byte: English domains 0.7-1.0, it/pt/nl 1.0-1.35, id/tr/vi/cy/eu 1.3-2.2 |
| select | cpu-run | E272's rule, KL "hi" +-10%, held gain over floor; then critical partial tau and power at the kept n | target ~0.18; English domains may drop (E272 lost pile, wiki_en) |
| read | cpu-run | control first: E272 `lower_s0_step128` erank 592.6058 (tol 1e-6; rehearsed today: diff 0, 4 s), zero update nan; per kept task 3 seeds + 1 Gaussian | Gaussian 685-720 and above the trained (E272 693-710); trained 440-600; a miss STOPs |
| score | cpu-run | the filed test | see below |

## The filed test
Partial Kendall tau-b of the consensus tier code and the task's mean effective rank (higher = reuse) given minus the
base loss per byte, one-sided task-permutation p (10,000, seed 276); "positive" when partial > 0 and p <= 0.05, "not
positive" when partial <= 0, else "no call"; under 5 kept tasks "too few tasks". Beside it, no call: raw taus,
leave-one-out. **Critical tau and power are written by `select` at the kept n with the real tiers and losses**, before
any read. Dry run (hypothetical tiers spread as E272's, loss following tier with noise): n 12 critical partial
0.31-0.35, power 0.26-0.34 at a true tau of 0.3 and 0.56-0.73 at 0.5; n 10: 0.35-0.43, power 0.45-0.65 at 0.5; n 8:
0.44-0.50, 0.44-0.54. **Caution found in the dry run:** E272's 0.87 leaned on summary (a prompt task with low loss and
a low tier). On E272's seven `lm_task` text tasks alone the partial is 0.40, since erank and loss order those tasks
almost alike (tau 0.905). These twelve are all `lm_task`, so about 0.4 is the realistic size, near the critical value.
**Collinearity guard (redteam-lead, 2026-11-23, filed before tiers and training).** `select` writes the tau of the
consensus tier with the base loss at the kept n; when its size is over 0.8, a call that would read "no call" or "not
positive" reads "tier and loss collinear: not separable". A "no call" at power about 0.3 is not evidence against the
effective rank: it says this panel could not tell. Prediction (filed): positive, low confidence. Stop: "not positive", or "no call" with at least 8 kept, ends the line.
**2026-11-23, after scoring.** The collinearity guard covered non-positive calls only, as redteam-lead asked, so it missed
this case: tier and base loss were collinear (|tau| 0.82 over the bar 0.8) and the call came out "positive", which the
guard leaves unchanged. Redteam-lead, in its verdict note: "I should have asked for it to apply to every call." Filed
call: positive. Reading: effective rank and base loss both order the tiers perfectly, so the panel cannot tell whether
effective rank adds anything beyond text unfamiliarity; the partial is descriptive.

## Cost and time
$0, no pod. Train: E272's LM jobs took 3.0-4.8 min each on an L40 (jobs 63727-63731) plus the first HF stream (5-30 s
measured today): ~5 min x 12 = 1 h if run one at a time on debug, less in parallel. Read: 4.2 s per state on CPU
(measured) x 48 = ~4 min. Select and score: under 2 min each. Ends well before 09:00 if trains start by ~06:30.

## Jobs (after the raters are filed and committed; from the clone root)
```
for T in italian portuguese dutch indonesian turkish vietnamese welsh basque pubmed stackexchange books math; do
  sbatch --comment="E276 stage: train $T" cluster/confirm_panel.sbatch train $T; done
~/commute-agents/fleet/cpu-run python3 experiments/confirm_panel/run.py select       # after all 12 train files
~/commute-agents/fleet/cpu-run -c 4 --mem 8G -t 30 python3 experiments/confirm_panel/run.py read
~/commute-agents/fleet/cpu-run python3 experiments/confirm_panel/run.py score
```
Inputs (grep of the loads): `experiments/convergence_panel/results/predictions.json` (settings block, sha256 checked),
`experiments/convergence_panel/results/large/states/lower_s0_step128.pt` (control), the cached Pile val rows 0-63 for KL
(`commute.data.cached_rows`, as E272), the HF streams above, the 67M model and tokenizer as every E272 job.

## Not run
No GPU job. CPU only: dataset load checks, the dry run (`~/commute-data/scratch/regating-lead/confirm-panel/dry_run.py`:
the vectorised partial matches the scalar to 3e-17; a stub from E272's text8 repeats its partial 0.8706; four kept
tasks read "too few tasks"; a zero update's erank is nan), and the control erank read.
