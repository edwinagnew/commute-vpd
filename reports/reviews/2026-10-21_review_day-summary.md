# Day summary, 2026-10-21

*Written for keeping track; the record is in the linked reports and the research log (v14 to v24).*

Today four experiments ran and two old premises fell. No split of a finished update keeps both the behaviour and
faithfulness. Splitting the update at every step does no better. The adversarial failures are not changes cancelling
each other. And the early ruling-out of plain SGD rested on a grid that started too high. So the main line moved to
re-gating, and comparing SGD with Adam became a priority subgoal that could bring the commute question back to first
place.

## Direction

- **Re-gating is the main line** (`reports/decisions/2026-10-21_decision_regating-main-line.md`). Q5 comes first: is Γ switching
  existing subcomponents on a mechanism, or Γ passing on computation already in its input? The commute question (Q1)
  stays open and comes second.
- **The optimiser is a priority subgoal** (user, in `GOALS.md`). Component training uses the run's optimiser, either SGD
  or Adam. The comparison covers both component training and the fine-tune whose update the splits consume. Until it
  runs, no verdict counts as the operation's rather than Adam's. Order approved: the SGD state's checks first, then the
  ladder comparison.
- **The Γ refit (E15) was shelved, then reopened** the same day (`reports/decisions/2026-10-21_decision_gamma-refit-reopened.md`).
  It is being fixed with an attacker matched to the harness's.

## What ran

| Code | What it asked | What it found | Where, cost |
|---|---|---|---|
| E15, second protocol | Does a gentler Γ refit leave the unchanged model at its floor? | No: 2.1 at step 100, 7.8 at step 300. The refit's own attacker is weaker than the harness's. | H100, $3.35, stopped early |
| E16 | The five operations side by side, steps 10 to 1000 | Only component training keeps the floor and the behaviour with Δ untouched; no split of a finished update does. Gate failed, so these are readings. | H100, $3.68 |
| E4 follow-up | Can plain SGD do component training at smaller rates? | Yes, from 0.003 to 0.1, within the off-target limit. E4's grid began too high. | L40, $0 |
| E17 | Do the changes a map writes cancel each other? | No. They add up, and the part the adversary removes carries at most 7% of its extra damage on the failing states. | L40, $0 |
| E19 | Does splitting every step, with the subcomponents moving, beat the one-shot split? | No. More behaviour stays in Δ than under the least-squares split, and it fails the adversary at step 1000. Surprise: refitting the scale every step doubles Δ's share of the update. | L40, $0 |

Paid compute today: about $7 of the $40 budget.

## Claims that changed

- **Refuted:** plain SGD cannot do component training (E4 follow-up); the adversarial failure is cancellation (E17);
  a per-step split keeps more of the behaviour in the subcomponents (E19).
- **Rescoped to Adam:** every component-training failure since E4, and every split, since all used Adam updates.
- **Open, new:** the adversary's extra damage comes from the original subcomponents the updated model now leans on
  (E17, inferred); the per-step scale's excess in Δ (E19, cause unknown).

## In flight or designed, not yet read

- **SGD line** (plans filed today): the SGD state's faithfulness (`experiments/sgd_faith`, results on disk, no analysis
  yet), matched SGD against Adam, an SGD fine-tune, SGD on the ladder tasks. Ranked in
  `reports/reviews/2026-10-21_review_sgd-finding-influence.md`.
- **E18**, the re-gating leak control (`reports/plans/2026-10-21_plan_regating-leak.md`): designed, not built. Before it, the
  complement revert and the adversary's mask by layer and importance, in E17's framework.
- **E15**, a refit with a matched attacker, control on the unchanged model first.

## Rules added today

- The control runs first and alone, before any arm it validates.
- The scored run is its own smoke; separate smokes only for expensive failures.
- Research-log updates are batched in `docs/research_log/PENDING.md` and flushed before a session ends.
- A grid whose lowest value overshoots is extended down before "no rate works" is written.

## Not covered here

- Per-experiment detail, figures and "Not run" lists: the analysis reports and the research log.
- The `sgd_faith` result, which has no analysis yet.
