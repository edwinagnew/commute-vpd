# Decision: does the re-gating share read re-gating on Qwen at all? What to run next on the SFT updates

*2026-11-13, qwen-lead, draft, following the user's direction of 2026-11-12 ("look more carefully into decomposing
SFT on qwen and/or studying elicitation/regating"). Design check:
`reports/plans/2026-11-13_plan_qwen-share-reference-design-check.md`. Nothing runs until you open a cell.*

**Question.** The re-gating share asks how much of a fine-tune's effect is recovered by keeping the model's original
parts and only switching them on and off; 1 means all of it. On Qwen3-1.7B's main decomposition it reads about 1 for
every fine-tune and every random change, so it says nothing there. On the frozen-neuron decomposition, whose parts are
the MLP neurons themselves, it does vary, but we have never shown it a change that is re-gating by construction, so we
do not know what a high or low reading means. Should the next run give the share that reference before part 3 relies
on it?

**What the numbers say.**
- Main decomposition: every fine-tune reads 0.990 to 0.9998, random changes of the same output size 0.992 to 1.009
  ([P3.8 page](../analyses/2026-11-13_analysis_p3-8-library-share.md)).
- Frozen-neuron decomposition: fine-tunes read 0.921 to 0.995, random changes 0.81 to 1.06
  ([P3.9 page](../analyses/2026-11-13_analysis_p3-9-frozen-neuron-share.md)). Known facts minus invented facts reads
  0.049 at the last save, eight times the gold-answer twin's 0.006.
- That 0.049 is most likely size. The invented-facts update is about twice the known-facts one at the last save
  (weight change 4.0 against 1.9), and the share falls as a change grows. Where both sides are matched in size, the
  gap is 0.0085, the same as two random changes give, 0.0083
  (`experiments/qwen_library_share/results/p-015e0710/summary.json`, `run2/*.json`).
- One possible signal: real fine-tunes read above random changes of similar size here (gold answers 0.961 against
  0.905; known facts 0.97 against 0.91), which they did not on the main decomposition. It rests on one random draw,
  matched in size only within a factor 1.5 (`run1/*_kl.json`).

**Options.** Costs: Vast H100 NVL at $3.45 an hour, about 3 minutes a reading, 7 minutes to fetch the decomposition.

- **(a) Built re-gating change plus random changes at several sizes, frozen-neuron decomposition. Recommended.** What
  runs: on each of the five training-row sets, a change made by switching off a set of neurons the model uses on those
  rows (their input weights zeroed), three choices of set, sized like the real fine-tunes; and random changes at three
  sizes covering the fine-tunes' range, three draws each, matched in size within 10%. 60 readings, no fine-tuned model
  fetched: about 3 hours, about $12 (or $0 on a cluster L40 if the queue is free). What it tells us: whether the share
  reads a known re-gating change near 1 and random changes lower; and whether real fine-tunes sit above random changes
  of their own size, which would be the first sign on Qwen that the decomposition sees SFT updates as more than noise.
  Next: if both hold, part 3 carries the share on this decomposition with the random curve as its reference, and pair
  gaps are read at equal size. If the built change reads like random, part 3 drops the share on Qwen and says why.
- **(b) More random draws at the existing sizes.** Three more draws per side and save: 36 readings, about 2 hours,
  about $7. Tells us how wide the band really is. It cannot say whether the known-minus-invented gap is size, and it
  still gives no reference for what re-gating looks like. Next: most likely every pair reads inside a wider band.
- **(c) Where each SFT update lands among the neurons.** For the 36 saved fine-tunes, the share of each update's
  split onto the neurons that fire on its own training rows, against random changes. Needs every fine-tune fetched
  again: about 2 hours, about $8. Tells us whether SFT updates concentrate on neurons the model already uses on that
  text, a direct "decomposing SFT" reading. The question of how much of an update the decomposition cannot express is
  already answered: none of an MLP update is out of reach, for fine-tunes and random changes alike. Next: a placement
  reading part 2 has not tested on the small model, so its meaning would need part 2 first.

**Recommendation: (a).** It is the smallest run that decides whether part 3 can use the re-gating share on Qwen at all,
and it needs no fine-tuned model fetched. Its weak point: it builds re-gating by switching neurons off; switching them
on, closer to eliciting a capability, cannot be built the same way.

**Proposed cell (for the table, if you agree).** Title: does the re-gating share read a built re-gating change as
re-gating on the frozen-neuron decomposition, and where do the SFT pairs sit against random changes of their size?
One run; due 2026-11-16, before the part 2 gate. Done when: the do-nothing check passes first; the built change's
share and the random curve (three sizes, three draws) are read on all five row sets; each saved fine-tune's share is
placed against the random curve at its own size; one figure and one paragraph, no reuse or new label.

**Evidence.** `reports/analyses/2026-11-13_analysis_p3-8-library-share.md`,
`reports/analyses/2026-11-13_analysis_p3-9-frozen-neuron-share.md`, premise checks `~/commute-data/premise/P3.8.md`
and `P3.9.md`, `experiments/qwen_library_share/results/p-015e0710/{summary.json,run1/,run2/}`, P3.3's planted changes
`reports/analyses/2026-11-11_analysis_e152-qwen-instrument-check.md`.

**Not run.** A built change that switches neurons on; any reading of the attention matrices on the frozen-neuron
decomposition; the RL updates.
