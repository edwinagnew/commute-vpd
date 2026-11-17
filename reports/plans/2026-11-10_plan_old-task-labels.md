# Labelling the old 67M tasks with the blind search

*definition-lead, 2026-11-10, cell P1.7, run 1 of 2. Nothing here has run. Labels from this run are provisional: no wording of
reuse is adopted, and nothing is added to the label rule.*

**In plain words.** Each of the 17 old tasks has saved fine-tunes of the 67M. Each fine-tune is added to the original model and
read with the same blind search E139 uses on its own fine-tunes: how much of the change one small re-weighted variable of the
original explains (G), and how much an affine map from the original's activations explains (H). E139's filed rule and E137's
cut-offs turn the two numbers into re-weighted, rewired or "no map found", per seed and then per task, beside the labels the
tasks had before.

## Design check

- **Goal.** `GOALS.md` asks whether post-training strengthens what the model had or builds something new; the user's first
  priority now is classifying the old tasks under the second wording, as provisional labels.
- **Nearest tried.** E137 (built pairs, cut-offs filed) and E139 (ordinary fine-tunes imitating them, read by the same code).
  E116 read 13 of these tasks under the frozen wording and gave no verdict. No run has read the old fine-tunes with G and H.
- **Patch or step.** A step: the wording's first reading on real, independently built tasks.
- **Decisions.** Keeps "no verdicts on saved runs until a wording is adopted"; the labels are marked provisional. Adds no rule.
- **What would change what we do.** Built-as-reuse and built-as-new tasks landing apart says the labels track something known;
  landing together (most rewired, the summary prediction) says the k = 4 line is set by the task's form, a point for the user.
- **Control first.** Stage 0, alone (below). It can fail by the loader adding a delta wrongly, by a D builder erroring or
  reading a gap with nothing changed, or by the new path not repeating E139.

## What runs

33 items: 16 tasks at seeds 0 and 1, and the one-pair map (rand1) at seed 0, its only saved seed. F = M + the saved delta-W
(`commute.old_tasks.old_f`, a copy of M with the delta added by parameter name). Then E139's own `search_g` (k = 4, sites
in1, in2, in3, out3, three starts) and `search_h` (five sites), loaded from its run.py, with reversion and setter mismatch.

**D, per task** (`old_tasks.task_windows`). E137's layout: n windows where F shows the behaviour, n where it does not, 2n/3 far
rows (Pile train rows 32,768 on), shuffled and halved into fit and read. Every window is 8 tokens read at the last position,
because H regresses on every position (36,864 columns, 10 GB); a 41-token window would need about 70 GB.

| Tasks | Shows (n) | Does not | D |
|---|---|---|---|
| Key tables: copy24, ident32, rand32, cipher24, elicit, rank1, rand1 | marker then a key (trained and held-out keys) | key without marker; marker before a non-key | 200,000 |
| e62_copy, e62_lookup (one D) | E62's generator at half 7: A B at 2-4, A at 6, marker at 7 | the same without the marker; marker after a non-key | 200,000 |
| case_real, case_twin (one D) | a case-pair word then the marker | the word without the marker; marker after another word | 200,000 |
| repeat_real, repeat_twin (one D) | E125's generator at half 7, earlier word at 1-4 | as the case label | 200,000 |
| cipher50, cipher500, lowercase | E53's task text at its scored positions, 4,096 Pile rows | plain cased Pile text | up to 200,000 |
| emoticon | its 438 saved trigger windows, cut to 8 tokens | plain Pile text | 1,168 |

**Cannot be read as it was trained.** Behaviour that needs more than 8 tokens is unseen: E62's copy over long gaps, E125's
earlier occurrence beyond 6 tokens, long text contexts. A fine-tune that does not show its behaviour on 8 tokens reads a gap
under 0.02 and gets "no reading". rand1 has one seed, so the task reads "no reading" by the rule. The emoticon's H is fitted on
467 windows and is expected low; its label is the least reliable.

## Stage 0, alone, and expected values

| Check | Expected | Stop if |
|---|---|---|
| (a) zero delta through the loader, every D | gap 0, max TV 0, no reading | max TV above the float floor (about 1e-5) or gap at or above 0.02 |
| (b) loader against `Net.at`, e62_copy s0 (fp32), case_real s0 (bf16) | max TV 0 | above the float floor |
| (c) a finished E139 fine-tune as a delta, its best G key and best H site with E139's seeds | gap within 0.001, G and H within 0.02 of E139 (expected under 1e-3) | outside |

(c) needs one finished E139 item with a reading (always_on first); it refuses until there is one. At the do-nothing reference
every item reads gap 0, so no label; per item the gap is printed before any search.

## Time and memory (L40, from E139's job 57251)

Set-up 25 s; stage 0 about 12 min (14 D builds and gap reads at about 12 s, G replay 115 s, H replay about 160 s). An item: gap
12 s, G 12 searches at about 115 s (23 min), H ridge and five sites about 10 min: about 34 min. 33 items are about 19 GPU hours:
3 items per 1:55 job, 11 jobs, resumable (exit 3). Card about 35 GB at `final`, host about 6 GB. $0, debug QoS. No weights saved.

## Not run

Nothing has run on a GPU. The D builders ran once on CPU on random stand-in rows (shapes, marker columns, determinism) for every
kind but the text tasks, which need the tokenizer and E53's maps; stage 0 builds those first.
