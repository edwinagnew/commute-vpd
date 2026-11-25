# P2.82 plan: a random baseline for the log-odds re-gating share (E310)

The paper's headline share (Fig `known`) has no random reference in its log-odds form: App. I says "The log-odds form
has no random baseline of its own", because no random change matched to a fine-tune's size or KL moved the trained
answer by the one log-odds the form's floor needs (E228). The user chose option 1 (2026-11-25, "run option 1"): scale
random changes up until they move the trained answer by one log-odds, and read the share on them. Descriptive: values
per task beside each fine-tune's, no verdict, no counts of comparisons. Card `experiments/random_logodds_share/card.md`;
code `experiments/random_logodds_share/run.py`; filed values `results/predictions.json`.

## What is run
- **States**: exactly Fig `known`'s, the share cells of `experiments/known_3seeds/results/known3.json`: nine built pairs,
  three seeds a side, 48 fine-tunes on 16 tasks. The paper's figure script reads this file and draws no twins.
- **Reader**: E136's `share_untrained` `Job.trained` (`local_faith.route_split`, bf16, E136's read windows), the code
  behind every value in those cells (E136 stage 1, E299 stage 2, E147 stage 1). Share r = (L01 - L00) / gain, gain =
  T1 - T0 of the same forwards; for the random draws nan unless gain >= 1 (the form's one-sided floor). KL to the
  original model on the same rows by E228's `kl_of`.
- **Random draws**: E228's, `finetune.random_direction(dW, seed)` at the fine-tune's per-matrix norms, seeds
  228001-228003; the three seed-2 groups E228 lacks (e46, e53, e54 cipher_own) get the same construction.
- **Scale**: a direction and its negative are the same Gaussian, so the sign is taken from the gain's odd part at the
  fine-tune's own size, sign(gain(+d) - gain(-d)); `perturb.match` (log grid 0.01-100 times the fine-tune's size, then
  bisection) finds the scale whose gain lies in 1.003-1.097. If that sign never gets there, the other sign is tried; a draw
  that reaches neither is reported with its largest gain and no share. Probe size, cap and band were put to Fable.
- **Per draw**: scale, sign, size relative to the fine-tune, gain, numerator, share, KL (NOT matched to the fine-tune's).
  Per fine-tune: its share, gain, KL. Each state's JSON is written per draw; a rerun skips finished work.

## Expected values
- **Zero change** (stage 0, every task's rows): gain, numerator and KL exactly 0; share nan.
- **Control** (stage 0, alone, first): each of the 48 fine-tunes' shares equals its known3.json value exactly (float
  equality; same code, file, rows and card type, NVIDIA L40, which E136, E147 and E299 all used). A difference stops the
  run with every difference listed; the lead decides. The six shards refuse to start until it passes.
- **Rival a**: the share mostly measures how much of any output change flows through Gamma's gates. Random changes read
  near the fine-tunes' new-side values (0.12-0.42) or above.
- **Rival b**: the share separates structure. Random changes read near 0 or erratic in sign across a task's draws.
- **KL**: from E228 (KL-matched draws at 5-30 times the fine-tune's size gained -0.7 to +0.6), draws that reach gain 1
  sit at a larger scale and a KL above their fine-tune's. No expectation is filed on how many draws reach the band.

## Cost and time
$0: cluster debug QoS, one L40 per job, `cluster/random_logodds_share.sbatch` (`--comment=E310`). Reads: 48 states x 3
draws x 16 reads (2 sign probes + E228's measured ~14 evaluations per match) x 1.8 s (E228's measured read time) =
4,147 s, about 69 min, plus 9 min for the six e53 states at 3.7 s (6 x 3 x 16 x 1.9 s) and 4 min for 144 KL reads:
about 82 min. If every draw also tries the other sign (28 reads a draw): 7,258 s + 958 s + 4 min, about 141 min. Split
in six shards of 8 states: about 14 min each, 24 min at worst, plus 3 min model load; the control job about 7 min. Every job under the 30 min limit; a cut shard resumes.

## Design check
- Serves GOALS.md: it gives the proxy the paper leads with (part 2, "cheap proxies track the definition") the random
  reference every other reading has, the MUST 4 sentence of the cold read in section 5.2.
- TRIED.md row 324 (E228) is the KL-matched version, where random changes never gained one log-odds; this run scales to
  the gain instead and states each KL, the one route E228 left open. Not a repeat.

## Not run
A twin or realistic fine-tune (not in Fig `known`); KL-matched draws (E228); a second decomposition; any verdict.
