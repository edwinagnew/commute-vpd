# Digest: commute-lead, 2026-10-28 afternoon (E43)

**Why the lead stopped.** Everything left on the commute line waits on two decisions only you can make. E43 ran, was
written up and checked, and the log (v86) shows it. Nothing is running, and no pod is up.

## What ran
- **E43, teacher training plus remainder** (RunPod H100, about $2.5). After teacher training, the rest of the
  fine-tune went into the residual, so that the weights equal the fine-tuned model's. Every control passed and
  matched E41. The filed prediction that this costs little on the adversarial faithfulness check missed by a wide
  margin: the reading was 2.5 times teacher training's own at step 10 and 1.9 times at step 1,000, against a limit of 1.25.
  It was also worse than leaving the whole fine-tune in the residual. So teacher training's own state stays the map. The
  run did not test which part of the remainder the adversary uses. Report:
  `reports/analyses/2026-10-28_analysis_e43-teacher-remainder.md`, notebook `notebooks/E43_teacher_remainder.ipynb`,
  check `reports/reviews/2026-10-28_review_e43-check.md` (stands with corrections).
- **Log** v85 (E39 and E41 Amendment 2 cards, two new decision cards) and v86 (E43 card, exact-weights decision card,
  current "Waiting on you").

## Waiting on you
1. Log card `decision-exact-weights`: accept that the exact-weights map is closed for now (recommended), or plan
   teacher training on Pile text, or a least-squares split of the remainder first.
2. Log card `decision-lora-sgd`: E39's next run, a rank-8 LoRA trained with plain SGD (about $1.5)? Recommended yes.
3. The older open cards, as listed in the log's "Waiting on you".

## Money
Session spend $2.68 of $40 (E43 about $2.5, plus $0.11 on two failed starts).

## Friction
- RunPod start() kept failing the "automatic-shutdown watchdog" check (15 s), even in a fresh Claude Code session. An
  `/mcp` reconnect fixed it, though one of the next two starts still failed.
- The model fetch from W&B ran at about 10 MB/s (45 MB/s the day before), so pod set-up took 12 min.
- In a worktree, the clerk's compound shell commands were refused, and the publish step asked for a full read of the page.
