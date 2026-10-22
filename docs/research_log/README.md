# Research log ("Update Decomposition Log")

The research-log artifact at
https://claude.ai/artifact/SvgkzUiESRkyE9GM65HCWu is built from `parts/` by `build.py`. The artifact is the colleague-facing explanation of the project;
the files in this repository are the record. The two must not drift: **every finished experiment and every major
decision updates the log in the same session**. A change that leaves what the log says untouched (a renamed path, a typo) needs no
log update.

## When to update

- An analysis report is written (the `/writeup` procedure's last step).
- A decision report is written or `decisions.md` / `GOALS.md` change (the `/design` skill's "After" step).
- A row is added to `experiments/TRIED.md` (a new E-code) or a proposal is dropped.
- A review finds a claim wrong.

## Layout (since 2026-10-22, user)

Four pages under one URL, sharing `style.css` and `script.js`. Section numbers stay fixed across pages, so "section 5"
in the text is always the experiments.

| Page | Sections | Source |
|---|---|---|
| `index.html` (the artifact itself) | 1–4, 6–11, 13 | `parts/sections/NN_<id>.html` |
| `experiments.html` | 5 | `parts/sections/05_experiments.html` (intro) + one file per card in `parts/experiments/` (`e1.html` … `a1.html`, ordered by number) |
| `glossary.html` | 12 | `parts/sections/12_glossary.html` |
| `changelog.html` | 14 | `parts/sections/14_changelog.html` |

`parts/header.html` is the main page's title block; `parts/style.css` and `parts/script.js` are shared. Edit only
`parts/`; `build/` is generated and gitignored. `python3 docs/research_log/build.py` writes `build/`, rewrites `#id`
links that point at another page, puts every section from every page into the nav, and reads the nav's version line
("Research log · date, vN") from the top changelog row. An `<li id="eN">` anchor outside the cards (E13, in section 11)
is linked to its own page.

## Batched updates (since 2026-10-21, user)

Updates are queued, not applied one by one. When any event above happens, append one line to `PENDING.md` as the
event's file is written (date, file, log sections, what changed). Flush the queue before the session ends, and at
least once a day:

1. Check nobody else has published since the last build. `Artifact read` with `url` and
   `paths: ["index.html", "experiments.html", "changelog.html"]` (small files come back inline, so leave out pages
   you are not changing), then `python3 docs/research_log/build.py --check <the directory it saved to>`. "same" on
   every page: build on `parts/` as is. A diff: someone published without committing `parts/`. Merge the diff into
   `parts/` first.
2. Run the completeness check, so nothing written since the last flush is missed:
   ```
   last=$(grep -o '<td class="c">20[0-9-]*</td>' docs/research_log/parts/sections/14_changelog.html \
          | head -1 | grep -o '20[0-9-]*')        # date of the newest changelog row
   ls reports/ | awk -v d="$last" '$0 >= d'      # every report dated on or after it
   for e in $(grep -o '^| E[0-9]*' experiments/TRIED.md | grep -o 'E[0-9]*' | sort -u); do
     grep -rq "id=\"e${e#E}\"" docs/research_log/parts || echo "missing card: $e"; done
   ```
   Every report listed and every E-code missing from `parts/` must be covered by a `PENDING.md` line or an existing
   changelog row; add the missing lines first.
3. Apply each line as "What to update" below says, grepping `parts/` for the places to change rather than reading
   whole sections. Write one changelog row for the flush, empty `PENDING.md`, build, publish.

## What to update

1. **Every experiment**: a new file `parts/experiments/eN.html` holding one `<article class="exp" id="eN">` card
   (code, name, date, machine, directory; one-sentence verdict; Run / Found / Bears on / Moved on), or the Moved-on
   field of an existing one. Figures: copy the experiment's own PNG into the `files` map below under
   `fig/eN_<name>.png`.
2. **Every claim, with its status** (`07_claims.html`): add, re-chip or re-word rows. Statuses: holds,
   partial/qualified, refuted, uninformative, open.
3. **Sub-questions** (`02_questions.html`): the status line and chip of every Q the experiment bears on; the E-code
   links.
4. **Overview → Where we are** (`01_overview.html`): only if a headline result changed.
5. **Timeline**, **Gaps**, **Backlog**, **Mistakes** (`08`–`11`): whichever moved.
6. **Changelog** (`14_changelog.html`): a new top row (version, date, what changed). The nav's version follows it.
   Old names and renamings go here, never in the body.
7. `experiments/TRIED.md` carries the same E-code; `README.md`'s experiment table lists it.

Keep the house style: cards and labelled fields, not paragraphs; the current terms from `GOALS.md` → Terms; about two
numbers per sentence; E-codes link automatically, to whichever page holds the card. Links written as `href="#id"` are
fine anywhere: the build points them at the right page.

## How to publish

From a Claude Code session, with the Artifact tool, after `python3 docs/research_log/build.py`:

```
Artifact publish
  url:        https://claude.ai/artifact/SvgkzUiESRkyE9GM65HCWu
  file_path:  docs/research_log/build/index.html
  files:      the changed pages and assets among
              {"experiments.html": "docs/research_log/build/experiments.html",
               "glossary.html":    "docs/research_log/build/glossary.html",
               "changelog.html":   "docs/research_log/build/changelog.html",
               "style.css":        "docs/research_log/build/style.css",
               "script.js":        "docs/research_log/build/script.js"}
              plus any new figure, e.g. "fig/e21_<name>.png": "experiments/<dir>/figures/<name>.png"
```

Files already published stay unless passed as `null`. The nav's version line is on every page, so a flush that adds a
changelog row publishes all four pages and both assets; new figures are added to the map.

The publisher requires that a session has viewed the live page before it accepts a publish. The first publish of a
session is refused and hands over the live `index.html` (about 130 KB) to read in full. That is the one fixed cost per
session; once `--check` says "same", the read is a formality. Publishing from the same session again needs no
further read.

Figures already published (their sources, for a re-upload):
- "fig/overview_component_training.png": "docs/research_log/component_training_tasks.png"
- "fig/e2_faithful.png": "experiments/lift_test/figures/inside_vs_outside.png"
- "fig/e2_behaviour.png": "experiments/lift_test/figures/pullback_behaviour.png"
- "fig/e6_overlap.png": "experiments/recruit_test/figures/overlap.png"
- "fig/e4_sgd_lr_grid.png": "experiments/sgd_grid/figures/lr_grid.png"
- "fig/e5_route_split.png": "experiments/task_ladder/figures/route_split.png"
- "fig/e7_vs_rank.png": "experiments/lora_commute/figures/vs_rank.png"
- "fig/e8_headline.png": "experiments/lora_matched/figures/headline.png"
- "fig/e9_pile.png": "experiments/lora_plateau/figures/pile.png"
- "fig/e10_reading_a.png": "experiments/kv_task/figures/reading_a.png"
- "fig/e11_ratios_per_task.png": "experiments/reading_a_coverage/figures/ratios_per_task.png"
- "fig/e12_route_separation.png": "experiments/new_vs_existing/figures/route_separation.png"
- "fig/e12_separation.png": "experiments/new_vs_existing/figures/separation.png"
- "fig/e14_sides.png": "experiments/lora_split/figures/sides.png"
- "fig/e15_pgd_vs_step.png": "experiments/gamma_refit/figures/pgd_vs_step.png"
- "fig/e15_stage0_ablation.png": "experiments/gamma_refit/figures/stage0_ablation.png"
- "fig/e16_bars.png": "experiments/split_comparison/figures/bars.png"
- "fig/e16_routes.png": "experiments/split_comparison/figures/routes.png"
- "fig/e17_half_revert.png": "experiments/cancellation/figures/half_revert.png"
- "fig/e17_adversary_revert.png": "experiments/cancellation/figures/adversary_revert.png"
- "fig/e19_behaviour_in_delta.png": "experiments/step_split/figures/behaviour_in_delta.png"
- "fig/e19_delta_share.png": "experiments/step_split/figures/delta_share.png"
- "fig/e18_recovery.png": "experiments/regating_leak/figures/recovery.png"
- "fig/e18_shared.png": "experiments/regating_leak/figures/shared.png"
- "fig/e18_natural.png": "experiments/regating_leak/figures/natural.png"
- "fig/e18_adversary.png": "experiments/regating_leak/figures/adversary.png"
- "fig/e20_decomposition.png": "experiments/cancellation_complement/figures/decomposition.png"
- "fig/e20_pattern.png": "experiments/cancellation_complement/figures/pattern.png"
- "fig/e20_bins.png": "experiments/cancellation_complement/figures/bins.png"
- "fig/e18_amend1.png": "experiments/regating_leak/figures/amend1.png"
- "fig/e20_rest_split.png": "experiments/cancellation_complement/figures/rest_split.png"
- "fig/e18_amend2_bins.png": "experiments/regating_leak/figures/amend2_bins.png"
- "fig/e18_amend3_tests.png": "experiments/regating_leak/figures/amend3_tests.png"
- "fig/e20_amend2_steps.png": "experiments/cancellation_complement/figures/amend2_steps.png"
- "fig/e21_ladder.png": "experiments/sgd_ladder/figures/ladder.png"
- "fig/e21_kv.png": "experiments/sgd_kv/figures/kv.png"
- "fig/e21_steps.png": "experiments/sgd_ladder_long/figures/steps.png"
- "fig/e21_size_damage.png": "experiments/sgd_ladder_long/figures/size_damage.png"
- "fig/e21_matched.png": "experiments/sgd_finetune/figures/matched.png"
