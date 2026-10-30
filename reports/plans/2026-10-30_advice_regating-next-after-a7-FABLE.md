# Fable's advice — NOT approved: what the 67M has answered after E58 Amendment 7, and the path to a Qwen claim

*Fable, 2026-10-30, on the lead's three questions after Amendment 7 (`reports/analyses/2026-10-30_analysis_e58-open-pair.md`
§ Amendments 6-7, provisional until the independent check of A7 lands; the synthesis
`reports/analyses/2026-10-30_analysis_regating-synthesis.md`; reviews `reports/reviews/2026-10-30_review_e58-amend6-check.md`,
`..._review_e58-amend7-code.md`; results `experiments/open_pair/results/amend6.json`, `amend7.json`; E57 C0
`reports/decisions/2026-10-30_decision_e57-no-teeth.md`). Advice only: nothing filed, nothing run. Terms as in `GOALS.md`:
r the route split's re-gating share, x the held-out carry, "prior" W₀'s geometric-mean probability of the trained value
after its key, "own guess" a target W₀ ranks in its top 20 for that key.*

**In plain words.** On the small model the reading is now settled as far as this task family can settle it: r says
whether the trained answer is one the model already gives for that key, and nothing about whether what was learned
generalises. A cut on the model's own probabilities sorts every map the same way, so on these maps the decomposition
has not yet shown anything the model's logits do not. The one thing left to decide on the 67M is a definition, not a
run: whether "elicitation" means "the answer was already there" (what r reads) or "an existing capability was reused"
(what the carry label reads). The two now disagree on a map, and the user should pick. The Qwen stage has to replace
E57's reading, which recovers any change at all, with the 67M's own gain-based r on the same marker-key-value task,
and let Qwen's cipher be the teeth test before any real update is read.

## 1. The 67M: answered as well as it can be, with one decision outstanding

- **What is settled.** Eight maps, three seeds, one key split: r reads 0.72-0.92 when the target is W₀'s own guess per
  key, 0.23-0.49 otherwise, with carry (−0.1 to 0.64 beyond null) playing no part (A7: class map 0.23-0.25 against its
  twin 0.27-0.28; rank correlation of r with carry 0.29, with log prior 0.95). A lookup at 3.7e-5 reads 0.38 while
  copy at 5.9e-5 reads 0.77, so r is not a smooth function of the prior, yet one prior cut still sorts all eight.
- **What that means for the user's question, said plainly.** On this task family r is a reading of "W₀ already
  produces this answer here", which W₀'s logits give without a decomposition. It does not read "new rule" against
  "lookup". So the 67M has not shown VPD adding anything over the prior; it has shown r is well-behaved (same
  answer per kind across seeds, keys and carry) and that it does not read generalisation. The optimistic reading that
  survives: r is a stable, interpretable number that agrees with the behavioural prior on toys, and its value, if any,
  is on updates with no single prior to read (a real SFT). That is the Qwen case, not a 67M case.
- **The behaviour-label tension is definitional, not empirical.** E58's label reads "reuses" when held-out keys gain
  (a carry label); r reads own guess. A7 made them disagree on the class map at step 11. No run resolves it: they
  measure different things. Fix in words: call the label "carries", and raise a pending-decision card to the user:
  *does "elicitation" mean (i) the model already gave the answer, or (ii) an existing capability was reused on new
  inputs?* Under (i) r is the right instrument and the 67M answer is "yes, but so is the prior". Under (ii) r is the
  wrong instrument, E55's ground truth is the right one, and the line should score a carry-reading instrument
  (steering, E55's patch) rather than r. This card is the single most consequential 67M item, and it costs $0.
- **The single remaining 67M check, ranked.** None moves the claim by much; in order of value per cost:
  1. *Elicit-prior dose-response*: cheap as a rider only (relax `min_rank` to 20 and fall back to the wider pool,
     reported not scored, as A7's rule already allows). It sets the band's prior range (two levels → three); it cannot
     change "r reads own guess". Not a job of its own.
  2. *A second VPD seed*: the only check that narrows the granularity limit, but A7 showed the limit did not bite on a
     rule built from W₀'s embedding directions, so its value fell. Worth a costed card only if Goodfire's recipe on the
     67M is under a few GPU-hours; otherwise not now.
  3. *A second rule family* onto words W₀ does not guess: replicates A7 under the own-guess account (low r predicted
     whatever the mechanism). Skip. The informative cell, a new mechanism onto answers W₀ already gives, is empty by
     construction, as the baseline advice said; state it, do not chase it.
  4. *Resolving the label tension by a run*: not a run (above).
- **What the A7 check could change.** If the independent check finds the carry inflated (code review F2: a lean whose
  size follows the key's class, or the uneven held-out classes, 18 of 32 in one class), the cell reopens; then the
  first-letter rule on Qwen, where it should carry, fills it, and no 67M rerun is needed.

## 2. The path to a Qwen claim

- **Why E57's reading has no teeth, and why the 67M's does.** E57's share is (K00 − K01)/(K00 − K11), KL over every
  position of the update's rows with Γ's masks recomputed on the changed model's activations. Γ is a function of the
  activations, so recomputing it recovers almost any change, real or random: 0.99 on E40, gold, anchor 2 and both
  random changes. The 67M's r (`src/commute/local_faith.py` `route_split`) is a different quantity: the log-odds gain of
  one target at one cue position, with W₀'s subcomponents under the fine-tuned model's gates, over the fine-tune's
  gain there. Its teeth come from the task: the cipher reads 0.23-0.31 and copy 0.77 on the same keys, same code. A
  random δW has no gain at the key, so it is not the teeth control for this r; the cipher is.
- **What must be fixed first.** Port the gain-based r, not the KL share, to Qwen3-1.7B, on the marker-key-value task
  (`src/commute/kv.py`, `heldout_keys.py`, `open_pair.py` builders) with the JAX decomposition's Γ (E57's masked-forward
  code gives the cells; the change is the quantity read: target log-odds at the key, not KL over rows). r needs only L00
  and L01 (original subcomponents under original and new gates), no least-squares split, so a read is two masked
  passes per state over the task windows, minutes on an L40, against E57's 20-min split per state.
- **The first Qwen table.** Rows, in run order with the control first: W₀ (zero update reads exactly 0; priors in
  band); cipher (teeth: r < 0.5 on every seed, else stop: no teeth at scale); copy (r ≥ 0.6, else the bands are a 67M
  artefact); elicit at rank 2-20; then, as riders, the first-letter rule and its shuffled twin (the carry cell, which
  should carry on a 1.7B). 96 trained / 32 held-out keys, two seeds, 30 steps of E54's recipe scaled by the Qwen
  recipe already sourced for E45. Columns: prior (mean per-token log-prob of the trained completions under W₀,
  cluster, $0, the W₀ control), r at step 30 and at the gate step, x − x_null, trained gain, ‖δW‖. Filed
  expectations are the 67M bands. Only once the cipher and copy rows hold do the real anchors join the same table:
  invented facts (prior ≈ 0, expected under 0.5), the format template (prior readable, expected over 0.6), gold
  GSM8K (open), each read at its SFT target positions with gain = the target's mean log-odds, which is a
  generalisation of the key read and needs its own W₀ control line.
- **Rough cost.** Fine-tunes: six to ten 30-step full-weight runs of a 1.7B on an L40, about 5-10 min each (E45 A2
  trained its anchors on the cluster for $0); reads: minutes per state. Two cluster jobs of under 3 h each, $0, or
  about 1-1.5 h on an H100 at ~$4/h, about $5. The anchors' reads add about the same again. **Gating:** none of it
  crosses $15 or the session cap; the only gate is time, so keep each Slurm job under 3 h (fine-tunes in one, reads in
  another, saved per state) and it is not gated. It is the other line's model: plan in the lead window before any
  spend, as the Qwen line rule says. A $0 rehearsal of the port on one state (W₀ and the cipher) first, since the
  JAX Γ path has only run E57's cells.
- **What the Qwen table can say.** If the cipher reads low and copy high, r has teeth at scale and the anchor rows
  are the first reading of a real update beside a prior column; if invented facts read low, the format anchor high
  and gold sits apart from where its prior puts it, VPD has read something the prior did not. If the cipher reads
  ≥ 0.8, no Qwen r has teeth and the line's claim stays 67M-only; say so and stop the Qwen read.

## 3. Ranked next steps

1. **The definition card and the write-up fix ($0, hours).** Rename the label "carries"; state in the synthesis that a
   prior cut sorts every map and r reads own guess; raise the card "(i) already-given answer or (ii) reused
   capability?" with the A7 disagreement as its evidence. Before any Qwen spend: it decides whether r is the instrument
   at all.
2. **The Qwen KV mini-table with the gain-based r ($0 cluster, two jobs under 3 h, or about $5 H100; not gated).**
   Port, rehearse on W₀ and the cipher, then cipher, copy, elicit, two seeds, prior column. Entry condition for
   anything after it: cipher under 0.5, copy at or over 0.6.
3. **The real anchors through the same table ($0-5, not gated if split into jobs under 3 h)**, invented facts, format,
   gold, each with its prior, read only after step 2's entry condition holds.
4. **67M riders inside any later job ($0):** the elicit-prior cipher at relaxed rank (reported), the novelty column on
   the class map. No job of their own.
5. **Not now:** a second rule family on the 67M, the second decomposition seed (card only if under a few GPU-hours),
   more keys or longer training on the first-letter rule, E55 scoring of r until the card is decided.

**What would change the ranking.** The user picks (ii): step 2 and 3 drop, and the line scores carry-reading
instruments against E55's labels instead; r stays a reported column. The A7 check overturns the carry: the Qwen
first-letter row moves from rider to step 2's required row. The Qwen cipher reads ≥ 0.8: stop at step 2, write the
claim as 67M-only, and the second seed becomes the only remaining 67M work worth a card. A second seed costs under a
few GPU-hours: run it before step 3, since it is the one check that narrows the limit on both models' readings.

**Not run.** Nothing; advice. Numbers from `amend6.json`, `amend7.json` (`reads.class_map_s*@30`,
`reads.class_shuffled_s*@30`, `tables.ep_skip`), the A6 and A7 analysis tables, and `decision_e57-no-teeth.md`'s C0
and Stage C rows.
