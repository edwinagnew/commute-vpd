# Overnight digest, 2026-10-28 (Qwen line)

**Why I stopped: your finish line was met.** The Qwen decomposition is on the pod's storage. We have three basic
checks of Qwen's maths and one of the decomposition's strength. Nothing is running. One decision waits on you, and the
first fine-tune's plan was not started. Research log: https://claude.ai/artifact/SvgkzUiESRkyE9GM65HCWu.

## 1. The decomposition is on the pod's storage

The converted Qwen3-1.7B decomposition is 13 files and 53.65 GB. It sits on a RunPod network volume in the US-MO-1
region, which any pod there mounts at `/workspace`, so the 20-minute download is gone. Every file was size-checked, and
an H100 pod read it back byte for byte. A second copy is on W&B as `qwen3-1.7b-p-e9fcffd5-torch:v0`.

## 2. Can Qwen do grade-school maths, or does it remember the test?

The unchanged model solves 76% of GSM8K's test set, a standard public set of grade-school word problems. The worry was
that some of that is memory.

- **Same problems, new numbers: ambiguous.** On one public rewrite it lost about four points, between our lines for
  "ability" and "memory". On a smaller rewrite it lost under one point, too few problems to tell.
- **The prefix test: no reading, dropped.** It stopped at its own check, because the model would not copy text back.
- **Your idea, new problems: same score.** On GSM1k, 1,197 problems written to match GSM8K and kept private until
  2025, the model scored 78% on both sets. That counts only answers that fit the decomposition's 512-token window.
  GSM1k draws long, mostly wrong answers about three times as often; counted in, it sits about four points lower.

## 3. How strong is the decomposition?

We ran the paper's adversary on 128 rows of web text, the authors' size. It searches for settings of the parts the
decomposition calls unneeded that most change the model's next-word probabilities. The damage is measured in nats: 0
means unchanged, and a model that guessed evenly over every word would read about 10. The adversary reached 1.6 nats.
With no search the decomposition reads 0.14, and four rows had read 17. Three random starts agreed within a fifth, so
one start per state is enough to use it as the pass test after a fine-tune. The search was still climbing, so 1.6 is
a lower bound.

## 4. Decisions for you

- **Keep GSM8K for training, and score every fine-tune on GSM1k beside GSM8K test?** Recommended. Log card "Keep
  GSM8K as the first task", in Pending decisions. The other option is a synthetic arithmetic task.
- No other open card is about Qwen.
- Once you confirm, the next step is the plan for the first Qwen fine-tune.

## 5. Spend

RunPod: $4.88 of the $40 budget. That paid for the adversary run, one H100 for 1 h 14 min. Everything else ran on the cluster at
$0: the W&B upload, two jobs that copied and size-checked the volume, and three maths runs. One job failed at once with no USER
set. Job numbers: 39093, 39126, 39195, 39125, 39129, 39181, 39230. No pods are running.

## 6. Problems and friction

- Restarting the window reloaded neither the keys nor USER. You fixed the keys by restarting, and `env.sh` now falls
  back to `id -un`.
- The RunPod volume copy printed "failed" for files that had arrived; the size check is what decides.
- Publishing the log takes three tries each time, and its check cannot read the live copy where it is saved.
- An earlier W&B upload died with its session and held the CPU worker empty for an hour and a half.
- The fleet's compute check does not say which agent owns which job.
- Lesson: you had not heard of the prefix test before it ran. A memory now says to explain any run that a plan's
  decision tree starts in chat first, in plain words.

## Not run

- The first Qwen fine-tune, and its plan.
- The adversary on maths text, with more than 20 steps, or on any updated decomposition.
- The paper's own adversary number for this decomposition, for comparison: not in the repo.
- A longer answer window, to see if GSM1k's long answers come right with more room.
- A model known not to have seen GSM8K, scored on both sets.
