# Review: E56 analysis (neuron patching on the key-dependent pair, r at five seeds)

*2026-10-30, independent check by a fresh agent that neither ran E56 nor wrote its report. Scope:
`reports/analyses/2026-10-30_analysis_e56-keydependent-patch.md`, `notebooks/E56_keydependent_patch.ipynb`,
`experiments/keydependent_patch/{plot.py,figures/*.png,results/*.json}`, `src/commute/{neurons.py,keydep_score.py}`,
the plan, the two E56 reviews, `/writeup`, `GOALS.md` → Terms. One $0 CPU measurement added (finding 1), through
`fleet/cpu-run`; script in `~/commute-data/scratch/baseline-lead/e56-check/norms.py`.*

## Verdict

**Numbers and scoring are right; one claim must come out and the adopted reading needs one wording change.** Every
number in the tables and text matches `job1.json` / `job2.json`, each prediction is scored by its filed rule, and the
pre-run k50 merge is reported openly with the unmerged verdict beside it. The weak part is the writer's added number:
"about 30% of the matched cipher's gain travels outside the MLP neurons, through attention" (4% at step 30). The
subtraction 1 − 0.69 is correct. Reading it as a path share, and naming attention as the path, is not supported by
this patch. The adopted reading ("tracks how far the cipher has trained, not kind") is argued from the files where it
matters, which is that r separates at both steps and the neuron share does not. "Not kind" is stated too strongly,
because at matched size how far training has got is itself a consequence of kind.

## Findings, by severity

### 1. Medium-high: the "~30% outside the MLP neurons, through attention" number is a valid subtraction but not a path share, and "spread out" is unmeasured

Location: report l. 9 ("how spread out the cipher's early update is"), l. 79, l. 95, l. 153, l. 155; notebook answer
cell (line 5 of the markdown), the cell after `b2_curves` ("so the missing part travels through attention. At step 9
that is about 30% ..."), and the "narrowed by one phrase" cell.

- **Arithmetic.** 1 − mean full-width share is 1 − 0.694 = 0.306 for the matched cipher, 1 − 0.937 = 0.063 for the
  elicit map, 1 − 0.962 = 0.038 for the cipher at step 30. "About 30%, 6%, 4%" is the correct subtraction.
- **What the subtraction is not.** (a) The patch replaces `down_proj`'s *input* (`steering.Net`, site `mlp<l>`) in a
  W₀ forward, so `down_proj`'s weights stay at W₀. The change to the four MLP output matrices is among the 24
  fine-tuned matrices and is not carried by the patch. On the saved δW it is 33-36% of each update's squared Frobenius
  norm (elicit s0/s1/s2 0.337/0.340/0.338; matched cipher 0.363/0.362/0.361; cipher at step 30 s0/s1 0.330/0.331). So
  the remainder includes an MLP weight change, not only attention. (b) The full residual patch is at block 3's output,
  the last residual before the unembedding, so it recovers 1 by construction. It does not say where the rest of the
  gain travels. (c) Shares of a log-probability rise do not add across paths, and attention's upstream changes already
  enter the patched neuron activations, which come from the tuned model's own forward. "1 − share" therefore lumps
  together the MLP output weights, attention's direct writes, and interaction.
- **"Spread out across the model" has no measurement behind it.** At the weight level the three states are made up the
  same way. Attention's share of the squared norm is 0.31 for the elicit map, 0.30 for the matched cipher and 0.29 for
  the cipher at step 30. The up-projection holds 0.34-0.38 and the down-projection 0.33-0.36 in every state. Weight
  norms are not function, but nothing in the files shows the early cipher update to be more spread out.
- **Fix.** Report the measured quantity: "all 12288 neuron activations recover 0.69 of the matched cipher's gain, 0.94
  of the elicit map's and 0.96 of the cipher's at step 30. This patch does not locate the rest (it includes the change
  to the MLP output weights, attention's direct writes and their interaction)." Drop "through attention" and "spread
  out / diffuse / concentrated into the MLP neurons" from the report and the notebook. If the location matters, the
  cheap test is to patch the MLP *outputs* (after `down_proj`, tuned weights) instead of the GELU activations, which
  is $0 on the cluster. Add it to "Not run".

### 2. Medium: "follows how far the cipher has trained, not what kind of update it is" is a false choice

Location: report l. 5, l. 9, l. 77, l. 139, l. 153; notebook answer and "narrowed" cells; Fable advice reason 1.

At matched size the elicit map has learned its table (held-out P(target) 0.95-0.97) and the cipher has not (0.26-0.31).
That gap in progress comes from kind: an update the model half-knows reaches the target with a smaller δW. So "tracks
training progress" and "tracks kind" are not alternatives on this design. Held-out P(target) at matched size is itself
a cheap reader that separates the pair, and the full-width share follows it. What the evidence does show, correctly
argued in reasons 1 and 3 (l. 89, 91), is narrower and still enough for the headline. The neuron share separates the
kinds only when their progress differs and agrees once progress is roughly matched (cipher at step 30: P 0.99,
full-width 0.95-0.97, against the elicit map's 0.93-0.94). r separates at both (0.76 against 0.21-0.27 and 0.24-0.27).
The progress reading also rests on one kind's trajectory at two points. No elicit state at another step was read.

Reason 2, "reversed on E12's pair" (l. 77, 90, 95), compares E12's states at their own scored steps, which were not
matched in size (E52's report lists "a size-matched copy and cipher pair" under Not run), and E52's labels were
themselves unverified. That evidence is consistent with the progress reading. It is not an independent reversal at
matched size.

**Fix.** Replace "not kind" with: "the full-width share follows how far each fine-tune has learned its table, which at
matched size differs by kind. It does not separate the kinds once progress is matched, and r does." Say in reason 2
that E52's states were at their own steps, not size-matched. Add "the elicit map read at earlier and later steps" to
Not run.

### 3. Medium-low: the k50 difference lies in the direction the filed null named, and the report does not say so

Location: l. 83-95 and the scorecard P1 row. The filed expectation table (`predictions.json` → `expected_table.k50`)
gives the null as "elicit far fewer". The result is elicit 4096 against cipher 12288 on all ten states, which is fewer
by one grid step (a factor of 3), so it points the null's way, though by a small step. The merge was filed before any
data and the report is open about it, which is good. A reader should still see one sentence saying the unmerged
difference points the null's way, by one grid step, and that the merge absorbs it. **Fix:** add that sentence to l. 83.

### 4. Low: "the cipher's curve lies on the elicit map's" holds only from 4096 neurons up

Location: l. 77, notebook b2 cell. At 256 the cipher at step 30 reads 0.09-0.11 against the elicit map's 0.12-0.17. At
1024 it reads 0.34-0.36 against 0.40-0.46, visibly below in `b2_curves.png`. **Fix:** "meets the elicit map's curve from
4096 neurons up".

### 5. Low: "the seed ranges are 0.03 to 0.06" (l. 75, copied from the Fable advice)

At 4096 neurons the ranges are 0.043 (elicit) and 0.053 (matched cipher). With all 12288 they are 0.013 and 0.032.
**Fix:** "0.01 to 0.05". The conclusion (no overlap) is unchanged.

### 6. Low: E52's reference curves in `b2_curves.png` are drawn through k values E52 never measured

E52's grid stops at 256 and then jumps to 12288. The dotted lines run straight between those two points across the
1024 and 4096 ticks, so a reader can pick off values that were never measured. Those states are also at their own
steps, not size-matched. **Fix:** put markers at measured k only, or add "(E52 measured k ≤ 256 and 12288 only; own
steps)" to the legend or caption.

### 7. Low: `/writeup` form

The opening plain paragraph (l. 5) is five sentences and 99 words. The rule is three sentences. "The answer" is 87
words, about right. `check.py` flags 31 sentences, mostly 26-36 words or 3-5 numbers, and finds no retired terms. The
densest are l. 75-77 and the "Departures" bullets. These are guidelines, not blockers.

### 8. Low: "Not run" misses the two readings the adopted interpretation leans on

Add: the elicit map read at other steps (finding 2), and neuron patching that gives the tuned MLP outputs, after
`down_proj`, instead of the GELU activations (finding 1).

## Checked, fine

- **Every table and text number against the JSONs:** the 15-row neuron table (16, 1024, 4096, 12288 and the k50 column);
  the shares at k of 1, 4, 64 and 256 all ≤ 0.172; N16 over all states −0.037 to 0.025; the five-seed r and S_lo
  table; r gap 0.518, range 0.055; S_lo gap 0.034, range 0.065; ratio 5.223 and plain ratio 15.0. Two-seed values
  recomputed: cautious 5.62, plain 10.28. Steering's range grew 0.043 → 0.065 and its gap shrank 0.052 → 0.034. P3
  differences 0.008, 0.007, 0.021; E54's 0.024, 0.031. P1 N16 gap 0.016 against range 0.045; two seeds 0.022 against
  0.028.
- **New-seeds table:** norms 3.18-3.21 / 3.21-3.25 / 6.00-6.05; P(target) 0.95-0.97 / 0.26-0.31 / 0.99; G 6.28-6.34 /
  10.32-10.95 / 13.72-13.73. Step 9 is the nearest norm on every new seed (step 8 ≈ 3.0, step 10 ≈ 3.45, against
  elicit ≈ 3.18-3.21) and is not at the edge of 1..30.
- **Stage 0 table:** identity share ≤ 9.5e-8 and 1.14e-5 nats; full patch within 6e-8 of 1; full-width rise 13.36-16.82
  nats; d(r, S, S_lo) = 0 on all six states; job 2 rates rel 0, seed-0 r dr 0; S_block0 × G_score under 1e-6 nats on
  the new states. (identity_share equals S_block0 to every digit, as expected: both forwards are W₀'s, bit for bit.)
- **Scoring.** `keydep_score.p1/p2/p3` implement the filed rules. P1 passes on min(k50_index, 6). As first filed it
  would read `b_higher` (6 against 7, range 0) and fail, as the report says, and also on two seeds in job 1. P2 and P3
  pass. The branch string matches the decision tree.
- **The merge was set before any data.** The amendment commit e02debe is dated 14:25:47. Job 46170 ran at e02debe at
  about 14:27. The sha256 of predictions.json at e02debe equals the current file (8746c6db…). The sha256 of job1.json
  (5f7c161b…) and job2.json (bd82e3a0…) match the provenance.
- **E52 reference numbers:** all-neuron share copy 0.50-0.71, cipher 0.76-0.97 (`baseline_neurons.json` → `b2`).
- **Figures.** `p2_shares.png` shows r and S_lo on one log-odds-share axis with 0 and 1 marked and seeds 2-4 as
  diamonds, as captioned. `p3_size.png` shows the ±0.1 band, with the elicit map as reference on the same axis.
  `b2_curves.png` shows 0, 1 and half-gain lines and the E52 references on the same axes (apart from finding 6).
- "Departures from the literature" covers causal tracing, CAA, the route split, Jain et al., the unsourced thresholds
  and E12's recipe. "Not run" is present and honest apart from finding 8. Terms usage matches `GOALS.md`. There are no
  retired terms.
- The notebook carries the same numbers as the report. Its three code cells display the figures and have outputs.
