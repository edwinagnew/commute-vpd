# Research log ("Update Decomposition Log")

`index.html` is the source of the research-log artifact at
https://claude.ai/artifact/SvgkzUiESRkyE9GM65HCWu. The artifact is the colleague-facing explanation of the project;
the files in this repository are the record. The two must not drift: **every finished experiment and every major
decision updates the log in the same session**. A change that leaves what the log says untouched (a renamed path, a typo) needs no
log update.

## When to update

- An analysis report is written (the `/writeup` procedure's last step).
- A decision report is written or `decisions.md` / `GOALS.md` change (the `/design` skill's "After" step).
- A row is added to `experiments/TRIED.md` (a new E-code) or a proposal is dropped.
- A review finds a claim wrong.

## Batched updates (since 2026-10-21, user)

Updates are queued, not applied one by one. When any event above happens, append one line to `PENDING.md` as the
event's file is written (date, file, log sections, what changed). Flush the queue before the session ends, and at
least once a day:

1. `Artifact read` the live log (another session may have published) and merge onto it.
2. Run the completeness check, so nothing written since the last flush is missed:
   ```
   last=$(sed -n '/id="changelog"/,$p' docs/research_log/index.html | grep -o '<td class="c">20[0-9-]*</td>' \
          | head -1 | grep -o '20[0-9-]*')        # date of the newest changelog row
   ls reports/ | awk -v d="$last" '$0 >= d'      # every report dated on or after it
   for e in $(grep -o '^| E[0-9]*' experiments/TRIED.md | grep -o 'E[0-9]*' | sort -u); do
     grep -q "id=\"e${e#E}\"" docs/research_log/index.html || echo "missing card: $e"; done
   ```
   Every report listed and every E-code missing from `index.html` must be covered by a `PENDING.md` line or an
   existing changelog row; add the missing lines first.
3. Apply each line as "What to update" below says, write one changelog row for the flush, empty `PENDING.md`,
   republish.

## What to update

1. **Every experiment**: a new `<article class="exp" id="eN">` card (code, name, date, machine, directory; one-sentence
   verdict; Run / Found / Bears on / Moved on), or the Moved-on field of an existing one. Figures: copy the experiment's
   own PNG into the `files` map below under `fig/eN_<name>.png`.
2. **Every claim, with its status**: add, re-chip or re-word rows. Statuses: holds, partial/qualified, refuted,
   uninformative, open.
3. **Sub-questions**: the status line and chip of every Q the experiment bears on; the E-code links.
4. **Overview → Where we are**: only if a headline result changed.
5. **Timeline**, **Gaps**, **Backlog**, **Mistakes**: whichever moved.
6. **Changelog**: a new top row (version, date, what changed). Old names and renamings go here, never in the body.
7. `experiments/TRIED.md` carries the same E-code; `README.md`'s experiment table lists it.

Keep the house style: cards and labelled fields, not paragraphs; the current terms from `GOALS.md` → Terms; about two
numbers per sentence; E-codes link automatically.

## How to publish

From a Claude Code session, with the Artifact tool, after editing `docs/research_log/index.html`:

```
Artifact publish
  url:        https://claude.ai/artifact/SvgkzUiESRkyE9GM65HCWu
  file_path:  docs/research_log/index.html
  files:      {"fig/overview_component_training.png": "docs/research_log/component_training_tasks.png",
               "fig/e2_faithful.png":            "experiments/lift_test/figures/inside_vs_outside.png",
               "fig/e2_behaviour.png":           "experiments/lift_test/figures/pullback_behaviour.png",
               "fig/e6_overlap.png":             "experiments/recruit_test/figures/overlap.png",
               "fig/e4_sgd_lr_grid.png":         "experiments/sgd_grid/figures/lr_grid.png",
               "fig/e5_route_split.png":         "experiments/task_ladder/figures/route_split.png",
               "fig/e7_vs_rank.png":             "experiments/lora_commute/figures/vs_rank.png",
               "fig/e8_headline.png":            "experiments/lora_matched/figures/headline.png",
               "fig/e9_pile.png":                "experiments/lora_plateau/figures/pile.png",
               "fig/e10_reading_a.png":          "experiments/kv_task/figures/reading_a.png",
               "fig/e11_ratios_per_task.png":    "experiments/reading_a_coverage/figures/ratios_per_task.png",
               "fig/e12_route_separation.png":   "experiments/new_vs_existing/figures/route_separation.png",
               "fig/e12_separation.png":         "experiments/new_vs_existing/figures/separation.png",
               "fig/e14_sides.png":              "experiments/lora_split/figures/sides.png",
               "fig/e15_pgd_vs_step.png":        "experiments/gamma_refit/figures/pgd_vs_step.png",
               "fig/e15_stage0_ablation.png":    "experiments/gamma_refit/figures/stage0_ablation.png",
               "fig/e16_bars.png":               "experiments/split_comparison/figures/bars.png",
               "fig/e16_routes.png":             "experiments/split_comparison/figures/routes.png",
               "fig/e17_half_revert.png":        "experiments/cancellation/figures/half_revert.png",
               "fig/e17_adversary_revert.png":   "experiments/cancellation/figures/adversary_revert.png",
               "fig/e19_behaviour_in_delta.png": "experiments/step_split/figures/behaviour_in_delta.png",
               "fig/e19_delta_share.png":        "experiments/step_split/figures/delta_share.png"}
  description: Everything the update-decomposition project has asked, run, found and deferred, for bringing a colleague up to speed and spotting gaps.
```

A session that has not read the artifact yet must `Artifact read` the URL first (another session may have published
since), merge onto that version, then publish. Files already published stay unless passed as `null`; new figures are
added to the map. Bump the version in the nav subtitle (`Research log · <date>, vN`) and the changelog together.
