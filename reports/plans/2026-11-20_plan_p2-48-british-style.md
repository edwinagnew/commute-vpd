# P2.48: British house style against a scrambled twin (67M)

Fills cell P2.48 (compare-lead, 1 run, due 2026-11-23). The user's words (plan 2026-11-20 item 5): "one real-text task
with a twin (British house style), three seeds, matched size, base contrast and gain floor checked first. File first:
real above twin." In the Terms' words: fine-tune the 67M (all 24 decomposed matrices) on running Pile text rewritten
into British spelling, and on the same text with each such word rewritten into a fixed letter scramble; three seeds
each, saved at the same update size; read the re-gating share r and the two strongest baselines (effective rank of
the update, neurons for half the gain) at the changed words. Card `experiments/british_style/card.md` (check passed);
rules `experiments/british_style/results/predictions.json` (draft, filed with the E-number).

## Design check
- **Goal:** part 2 (methods against reuse vs new on the 67M). E211's real-text pairs differed in data, format and size
  of change; here the twin holds rows, positions, recipe and size fixed and changes only whether the written form is
  one the base already prefers. Tests whether r's built-pair separation (and E53's lowercase vs cipher) carries to a
  real-text task with a twin.
- **Nearest tried:** E53 (lowercase 0.75 vs 50-word token cipher 0.12-0.26: a cipher on the same rows, but the real
  side changed every token); E53 Amendment 1 (narrow lowercase failed its premise, base preferred the capital by 4.72
  nats: the reason stage 0c exists); E211 T1/T4 (apart, gap 0.21, but different rows); E140 (real labels vs shuffled
  twin, not apart). New: a word-level real task and a twin on the same occurrences.
- **Patch or step:** a step (the cell's done_when).
- **Decisions relied on:** decisions.md row 10 (only the 24 matrices train), MIN_GAIN 1, E211's size targets, the
  scorecard call (`realistic.seed_call`). No departure.
- **Adopted definition unchanged:** yes. Sides are labelled by the data argument; no reuse verdict; no rule added.
- **What changes what we do:** r apart upward → part 2 can say r separates a real-text task from its twin; overlap →
  the card's stop: r's built-pair result does not carry, reported as such.
- **Control first:** stage 0, alone: (a) E53 lowercase_s0 reproduces through these readers, (b) a zero update reads
  exactly 0, (c) base contrast; then (stage 1) seed 0 of both tasks must clear the gain floor at the matched size
  before seeds 1-2. Each exits 1 on a miss.
- **Early guards:** every 10 steps of training the log shows size and train loss; the gain floor is read on seed 0
  before any further seed; data counts (pairs kept, rows, occurrences, token-count match) logged before the model loads.
- **Saved:** states in bf16 at both sizes (`results/large/`, 57 MB each), the rows and read windows of both tasks
  (int32), the word map and twin map (JSON), per-state JSON with every reading, and a hand-off file for
  definition-lead's R in E136's inventory format.

## What runs
Data (tokenizer only, from cached `artifacts/sets/train_0_32768.npy` and `val_0_2000.npy`, no download): the word
list `hyperreality/American-British-English-Translator@16ec815` `data/american_spellings.json` (1,800 pairs), filtered
to lowercase-only pairs not ambiguous by the repository's own `british_spellings.json` and `different_meanings.json`,
minus ten frequent sense-dependent forms (practice, program, disk, filter and inflections; reasons in settings): 1,759
pairs (measured). Scored token: the first token where the British form's tokens leave the American form's (k = 0 for
537 pairs, e.g. " colour" for " color"; k = 1 for 864, e.g. " organ|ise" vs " organize"). Twin: the British form's
first k tokens kept, the remaining letters permuted (seeded), not a word of any list, not one vocabulary token,
token k different from both forms'; same token count as the British form for 1,019 pairs, one more for 634 (single-token
British forms such as " colour" cannot be scrambled into one token). Rows: Pile rows with at least 2 kept American
forms written lowercase after a space (27% of rows, 3.5 each); each occurrence replaced in each task (the twin also
scrambles British forms already present), encoded segment by segment, cut to 500 tokens. 4,800 training rows (each
once), held rows from val 0-1999; read windows: 500 occurrences (127-token windows, at most 8 per row) at the same
occurrences in both tasks.

Training: E211's LM recipe (Adam 3e-5, batch 8, clip 1, batch order = seed), cap 600 steps; each run saved at the first
step its size reaches 0.00875 and 0.0175. Readings per state: gain, r (`realistic_reads.share_of`), k_half
(`k_half_of`), erank (`update_spectrum.state`), size.

## Numbers and what each should read
| number | W0 / do-nothing | control (e53/lowercase_s0) | British (3 seeds) | twin (3 seeds) |
|---|---|---|---|---|
| prior (W0 log-odds of scored token) | real -3 to -9; twin -10 to -18 | - | - | - |
| gain at 0.0175 | exactly 0 | 1.672 | 2 to 6 | 4 to 12 |
| r at 0.0175 | nan | 0.7519 ± 0.01 | 0.55 to 0.85 | 0.15 to 0.5 |
| k_half | null | 16 | 8 to 32 | 32 to 256 |
| erank | null | 598.5738 ± 1e-6 | 560 to 610 | 480 to 600 |
| steps to 0.0175 | - | - | 350 to 600 | 150 to 400 |

Filed (each separately, at 0.0175, `seed_call` + `filed`, one value a side = unreadable): **P1** r real above twin,
apart (medium); **P2** erank real higher (low); **P3** k_half fewer on real (low). Card: rival a = P1 hit with gap at
least 0.1; rival b = no call or miss. Edge cases run (ties, empty pool, one value, do-nothing on both sides: cannot
pass); listed in predictions.json.

**Blind to.** r: any change gaining under 1 log-odds; everything but the one scored token per occurrence; that the
British side starts nearer its target (prior), which the twin cannot match. k_half: which neurons; the same floor.
erank: behaviour (scale-blind). The twin differs from the real task in more than "known or not": its scored tokens are
rarer tokens, and about 40% of its words are one token longer; rows, positions, recipe and size are matched.

## Cost and routing
One job, both tasks, three seeds: L40 estimate from E53's measured 0.09 s/step (107 s for four 300-step runs): 6 runs x
at most 600 steps = at most 5.4 min training; reads ~15 s per state x 12 (E227: ~45 s per item for three read sets of
512); data build ~2 min; model load ~1 min; stage 0 ~2 min: **~15-20 min, $0 on cluster debug** if it starts at once.
Otherwise one RunPod H100 (~10 min set-up + ~12 min run ≈ 0.4 h x $3-4.8/h ≈ **$2**; E212 measured $4.8/h a pod).
Transfers to a pod: train_0_32768.npy 134 MB + val_0_2000.npy 8 MB + the control state 57 MB + E136's lowercase windows
≈ 0.2 GB → 20 s at 10 MB/s; the 67M from W&B on the pod (~65 s, E-notes). Back: 12 states x 57 MB = 0.68 GB → 68 s,
rows and windows ~25 MB → 3 s. Not gated (under $60 and 6 h). Rehearsal per the card on cluster debug, $0.

## Open choices (decided here)
- Token count: matched exactly for 58% of pairs; the rest differ by one token or more because a one-token British form
  has no one-token scramble. Kept rather than dropping one-token forms (colour, centre, favour are the most frequent).
- Scored at the first token where the form leaves the American one, not at the word start (the first token is shared by
  1,496 of the 1,800 listed pairs, so it would read no change there).
- Title-case and all-caps occurrences, and words not after a space, are left as written in both tasks.

## Hand-off
Phase B: `experiments/british_style/run.py` (stages 0, data, 1, 2), `plot.py` (r, erank, k_half dots per side),
rehearsal on cluster debug, then one pod-runner brief.
