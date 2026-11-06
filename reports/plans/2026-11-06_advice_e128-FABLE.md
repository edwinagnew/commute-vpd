# Fable's advice on E128 (post-cutoff facts, anchor 2 backup)

*2026-11-06, Fable for qwen-rl-lead. Read: the E128 plan and appendix, E120's plan and `results/pools_amend2.json`
(n 77), `experiments/qwen_pass_pools/train.py`, `src/commute/qwen_eval.py`, `qwen_facts.py`, E45's `run.log`. The
data source (Wikidata statements vs LLM-written QA) is the user's; nothing here depends on it except (e). Not
committed to; the lead decides.*

## (a) Cutoff date for "new"
Our files hold no Qwen3 training cutoff (only the step-2 decision's mention, `decisions.md:177`). The one hard bound is
the release, 2025-04-29: an event after it cannot be in the weights whatever the data cutoff. The plan's 2025-07-01 buys
nothing against that bound; what it should guard is **knowability**, not date: a 2025-dated fact can be fixed years
earlier (host city, scheduled release, venue), and such facts are answered at W0 and only cost yield.
Options: (i) 2025-05-01, the release month; (ii) 2025-07-01 as filed; (iii) a measured cutoff from month-binned W0 rates.
**Recommend (i) with two relation rules instead of the margin:** the new side takes only relations whose answer is
decided at the event (winner, result, appointee, death), and the known side takes events dated **on or before
2023-12-31**, a year or more before release, so the fact had time to reach a pretraining crawl (2024 editions risk
sitting past the real cutoff and reading "new" for the wrong reason). Report, no rule, the known rate by half-year
2023-2025 from stage 1's samples: it is the measured cutoff for the write-up. Changes it: a new side under ~150
candidates at 2025-05-01 has no later date to retreat to, so widen relations, not dates; a stated cutoff in the Qwen3
report (not in our files) replaces (i) with that date plus one month.

## (b) Changed facts (old answer in >= 32 of 64)
Options: cut (E120's guard); keep as a counted side group outside the pools; keep in the new pool.
**Recommend cut from the pool and count them**, with their ids and the model's commonest answer written to the pools
file's `meta`, no training on them. Reason: D3 wants the new anchor to mean "no answer held"; an overwrite is a third
kind of update (an old association to unlearn), and mixing it into "new" muddies the one clean label this backup
offers. The count itself is a result the user's source question needs: how much of 2025 Wikidata is overwrite rather
than fresh. Two cautions. The guard needs `qwen_facts.norm` on answers, and spellings of one wrong answer may still
split ("Biden"/"Joe Biden"): report both "one normalised answer >= 32" and "non-gold, non-empty >= 32". Changes it: if
the new side is under 100 without them and 100 or more with them, that is a decision card with both counts, never a
silent inclusion; a changed-facts arm is a new plan, not an E128 amendment.

## (c) Moving E120's `train.py` into `src/commute/pool_sft.py`
Options: (i) move the body, both experiments' `train.py` become ~30-line callers; (ii) E128 calls E120's file through
`commute.jobs.load_experiment` with a results-folder argument; (iii) copy (forbidden).
**Recommend (i).** `train.py` fixes `HERE`, `RESULTS`, `LARGE`, `POOLS`, `PRED` at module level (`train.py:73-76`)
and `Lab` writes every file under `RESULTS`; a results-folder argument has to thread through all of them, which is the
move anyway, done in a worse place and leaving E128 dependent on E120's folder. It has never run, so "finished scripts
stay as they ran" does not bind. Keep the shared part to what both runs share: `Lab`, the recipe assertion, the stages,
the early checks. Each caller names its folders, pools file, predictions file and `ANSWER_CAP` (1,024 is maths-sized;
facts need the stage-0 measured cap, and the trained-wording greedy read is cheap at 64). Run `dry_run.py` on both
callers before any arm; lint lands on the module. Changes it: E120's arms starting before the move (the user takes n 77)
means the move waits until E120's four arms are home, then E120's caller must reproduce its records' sha.

## (d) HF sampling on the cluster vs E120's vLLM on a pod
Numbers. E45's facts control on an L40 made 1,200 greedy fact answers in 6.8 s (`experiments/qwen_updates/results/
run.log:1,3-4`; batches of 256, cap 32, `experiments/qwen_updates/job.py:83-84`), about 175 answers/s; HF on long
answers made ~550 new tokens/s (`reports/reviews/2026-11-06_review_fable-e120-design.md:10`). E120's vLLM on an H100
made 12.6k tokens/s on maths (`experiments/qwen_pass_pools/results/pod_dl/console.log:177`), 158M tokens in 12,354 s
(`results/stage1.json` → tokens, seconds); an L40 is about half. Stage 1 is ~2,000 x 64 = 128,000 answers of ~20
tokens, ~2.6M tokens. **HF, L40: 15-45 min** (greedy-rate scaling gives 12 min; sampling at cap 64 and 64 returns per
prompt pads batches to the longest answer, so allow 3x); one `debug` job, no shards. **vLLM, L40 or pod: 5-10 min of
generation** plus a venv build unrehearsed on the cluster (review line 47) or a pod's set-up and downloads, $1-3.
**Recommend HF on the cluster for stages 0-1:** the generation is too short for vLLM's set-up to pay back, and the
sampling settings (`qwen_eval.SAMPLE`) match E120's. N1 is the exception: 250 facts x 4 thinking samples at cap 8,192 is
0.5-1 h on HF if the model thinks ~1-2k tokens over an unknown fact, 2-4 h if it runs to the cap. Shard N1, log the first
shard's mean thinking length, and move N1 to E120's pod engine (`engine.THINK`) if it reads over ~2k. Changes it: a
stage-1 first shard under ~50 answers/s.

## (e) Main risk to rows by 2026-11-10, cheapest guard
The dataset: nothing exists yet, the source is undecided, outbound access from a compute node is unverified
(`cluster/README.md:126`: a Cloudflare download read ~0 Mbps, pip works), and the plan's own P1 lean is 0.6 with the
known side's yield unknown. Every GPU stage is cheap and measured against it; a one-day slip here is the whole slack.
**Guard, $0, today:** (1) take the source by the plan's default when the 2 h lapse; (2) a 10-minute probe fetch of ~50
statements per side for one relation (a sub-MB API call is light work; the full fetch is a Slurm CPU job), through
`build.py`, so the schema and access are proven before 11-07; (3) write `sample.py` against E45's `qwen_facts.known()`
so stage 0 and the sampling code run on a GPU before the data lands; (4) in stage 0, add 100 known-side candidates
greedy (~1 s on this path) to read the known yield before stage 1 commits to 1,500. Second risk, smaller: four fp32
arms write ~135 GB to `results/large`; check free space on `~/commute-data` before stage 3.
