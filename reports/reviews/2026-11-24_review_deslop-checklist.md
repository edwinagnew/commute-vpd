# De-slop checklist for the reading refactor (P2.79, 2026-11-24)

The user asked for a web search on how to clean up AI-written code. A subagent read practitioner write-ups (sources
below) and distilled them into fifteen checks; where they disagree with `.claude/CLAUDE.md` "Writing code", the repo
rule wins and the line says so. The refactor's writers and its reviewer apply this list.

1. Pin behaviour first: record each reading's current value on saved states and re-check after each move (here, the
   E309 old-code batches and `experiments/reading_refactor/compare.py`). No new test suites as process.
2. Delete dead code first: `vulture src/commute --min-confidence 100`, then 80 by hand; `ruff check --select
   F401,F841,F811`. Re-run after each deletion. Entry points (`main`, plot functions) are not dead.
3. Delete commented-out code, dead flags and arms no run reads: `ruff check --select ERA001`.
4. Delete scaffolding leftovers (no-op stages, debug harnesses, placeholder branches) before touching duplication.
5. Find duplication: `pylint --disable=all --enable=duplicate-code --min-similarity-lines=8` (or `jscpd --min-lines 8`).
   Each real hit becomes one function in `src/commute/`.
6. Deletion test before extracting: name the code a new shared function removes; if none, do not add it. Compare
   look-alikes' behaviour (dtype, batch, rows, floor, unit) before merging: a difference becomes a required argument
   or a separate name, never a silent merge.
7. One function per reading, one owner module. **Repo override:** finished experiments' `run.py` are never edited
   (they move unedited to `archive/experiments/`) and never used as templates.
8. One module per concern; a run script over ~300 lines is split (`wc -l`).
9. Functions under ~50 statements: `ruff check --select C901,PLR0915,PLR0912`; split a function rather than add a branch.
10. Long parameter lists (`PLR0913`): pass the frozen config dataclass, not more kwargs. Exception: the settings that
    tell reading variants apart stay explicit required arguments (plan "Rule for variants").
11. Remove the structure that allowed a bug rather than guarding beside it. Cut impossible-case branches, chained None
    checks and `try/except` that hides errors (`ruff check --select BLE001,S110,SIM105`). Fail loudly.
12. Separate compute from IO: a measurement function returns a value and does not save, log or download.
13. Delete comments and docstrings that restate the code; keep the why and the source of a setting. **Repo
    override:** never `ruff format` (the repo is not ruff-formatted); never squeeze formatting to meet a size aim.
14. Names say what a function does: no `_v2`, `_new`, `_old`, `helper`, `final`; rename everywhere at once.
15. Small single-purpose commits (dead code, a move, a merge, never mixed), `make lint` and the reproduction check after each.

Not adopted: "add tests first" and "extract interfaces generously" (the repo prefers less code and less process).
Caveat: the sources are practitioner blogs, not standards; the tool flags were not checked against `ruff rule` or
`pylint --help` and should be before use.

## Sources
- Convex, "How to de-slop your code": https://stack.convex.dev/how-to-deslop-your-code
- M. Saplin, "Debloating the AI-grown codebase": https://dev.to/maximsaplin/debloating-the-ai-grown-codebase-2om
- "Clean AI slop" cleanup skill: https://tmdgusya.github.io/engineering-discipline/skills/en/clean-ai-slop.html
- Builder.io, "De-slop an AI-generated codebase": https://www.builder.io/blog/de-slop-ai-generated-codebase
- P. Mineault, The Good Research Code Handbook: https://goodresearch.dev/ (landing page only)
- Vulture: https://github.com/jendrikseipp/vulture
- Refactoring in deep-learning repositories (smell tools misflag DL practice): https://arxiv.org/pdf/2405.04861v3
- Thoughtworks, effective habits for data scientists: https://www.thoughtworks.com/insights/blog/data-science-and-analytics/effective-habits-for-data-scientists

## Not run
No tool above has been run on the repo yet; the dead-code pass is phase 3 of the refactor.
