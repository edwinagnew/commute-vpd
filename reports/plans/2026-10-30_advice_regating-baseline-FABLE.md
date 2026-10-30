# Fable's advice — NOT approved: a non-VPD baseline for r, de-confounding r from W₀'s prior, and the missing cell

*Fable, 2026-10-30, on the lead's four questions after the independent check
(`reports/reviews/2026-10-31_review_regating-synthesis-check.md`). Advice only: nothing filed, nothing run. Read:
`.claude/CLAUDE.md`, `GOALS.md` → Terms, the adopted programme, the ground-truth plan, the synthesis (being fixed; not
edited), the review, my E58 A4 advice, `experiments/TRIED.md` rows E54-E58, `experiments/open_pair/results/{amend3,
amend4,amend5,run}.json`. Terms: r the re-gating share, "the ratio" the stand-in ratio, x the held-out carry, S the
steering share, "prior" W₀'s geometric-mean probability of the trained value after its key (saved per map as
`maps.<map>.w0_geo_p.trained` in `amend3.json`).*

**Corrections to my E58 A4 advice, as the review found.** "128 keys" should read 96 trained (32 held out). "The map with
the highest r has the lowest carry" is false: rank-1 has the highest r (0.89, 0.85) and carries 0.2-0.3; elicit has the
lowest r of the band (0.78, 0.72) and the lowest carry (0.21, 0.10). The supported sentence is "the order of r inside the
band is not the order of carry" (copy carries 0.8 at r 0.87, between rank-1 and elicit).

**In plain words.** The review is right that the synthesis has no number that is not VPD's. From what is already saved,
the prior separates the same two bands r separates, with one swap in the order of five maps, and that is all the saved
data can say; it cannot show r adds anything. The way to find out costs one free cluster job: train the same cipher
twice, once on values W₀ has never seen and once on values W₀ gives at the same probability as the copy map's targets,
and read r on both. If r still falls with kind at equal prior, r is not the prior. The same job can hold the first
map whose target is not W₀'s guess yet carries to unseen keys; VPD's reading there is informative in one direction only,
and the write-up has to say so.

## (a) A $0 baseline from what is saved: possible, but it can only make the review's point, not answer it

**What is saved.** Per map, one prior (same for every seed): cipher 1.4e-6, shuffle 1.0e-5, copy 5.9e-5, elicit 1.3e-3,
rank-1 1.3e-2. Per run and read point: r, the ratio, S, x, G_spec, ‖δW‖. No per-key prior, no per-key r.

**What to compute (an afternoon with `jq`, no GPU).** (1) The prior's AUC for the own-guess label over the 15 runs,
beside r's and the ratio's: all three read 1.0, by construction (the label *is* a prior threshold). (2) Spearman of r
against log-prior over the five maps at each read point: 0.9 (one swap: copy, prior 5.9e-5, reads r 0.87, above elicit at
1.3e-3, r 0.78). (3) The one contrast that is not pure prior: copy against shuffle, priors 6× apart (5.9e-5 vs 1.0e-5),
r 0.4 apart at step 30 (0.77 vs 0.47-0.49), against the full prior span of 10⁴ carrying an r span of 0.6. Report the
three numbers; nothing can be scored, since every outcome was fixed when the maps were chosen.

**What each outcome means.** (1) says the review is correct: on these maps the prior is a complete baseline and r's
agreement with the label is agreement with the prior. (2) and (3) say r is *not* a monotone function of the prior with a
clean cut at the map level, which is suggestive and nothing more: five points, one swap, no seeds on the prior.

**Where it goes.** Into the synthesis, as one paragraph headed "The prior as a baseline" and a "Not run" line
("whether r adds anything over the prior; the prior-matched cipher, below"), and the plain answer reworded to "r agrees
with W₀'s prior of the trained value; whether it reads more than that is untested". Not an amendment: nothing is
predicted, and a filed prediction on data already seen would be theatre.

## (b) De-confounding r from the prior: two cells, both reachable on the 67M in one `debug` job

The confound is that "own guess" and "high prior" are the same thing in every map run so far. Break it both ways, on
E58 Amendment 2's 96 trained / 32 held-out keys, three seeds, read at step 30 and at the cipher's gate step, same code
(`experiments/open_pair/amend4.py` with two new map builders in `src/commute/keydep.py`):

1. **Prior-matched cipher (high-ish prior, lookup).** A cipher whose values are drawn, per key, from W₀'s distribution
   after that key at the *same geometric-mean probability as the copy map's targets* (5.9e-5; sample from the probability
   band 2e-5 to 2e-4 at ranks well past 20, excluding the key and the elicit words). Same breadth, format, key count
   and ‖δW‖ rule as E58's cipher. **This is the baseline the review asks for**: prior equal to copy's, kind different.
   Filed prediction: r below 0.5 on every seed (lean 70%: the shuffle, whose values W₀ emits somewhere at prior 1e-5,
   read 0.47-0.49, so the honest risk is that r sits at the cut again). If r ≥ 0.6: r tracks the prior, the band claim
   is dead on the 67M, and the plain answer becomes "VPD reads W₀'s prior". If r < 0.5: r separates kind at equal prior;
   the prior is no longer a complete baseline.
2. **Low-prior rule W₀ has (low prior, reused rule).** The copy map already sits here in part (prior 5.9e-5, carry 0.8,
   r 0.87); a second instance that does not reuse copy's exact circuit: key → key with the first letter capitalised, or
   key → key's plural, chosen so the prior of the target after the marker is under 1e-4. Filed prediction: r in the
   upper band and x over 0.5. Lower priority than 1; it strengthens the claim, 1 tests it.

The cell "high prior but newly built mapping" is not constructible by training alone: if W₀ already produces the target
at that cue, a fine-tune has no reason to build a new route, and if it did, no behavioural label would know. Do not
chase it; say in the write-up that it is empty by construction, which is the granularity limit's mirror image.

Cost: one job, two maps, three seeds, six fine-tunes of 30 steps plus reads; E58 A4 ran nine in under 20 min; $0.
Not gated.

## (c) The missing cell: a new rule that carries

**Fill it with a rule over a feature W₀ has, mapped to values it does not guess.** On the same 96/32 key split: key →
a fixed word per *first letter* of the key (26 values, each a random word at cipher prior, under 1e-5), or key → a
fixed word per semantic category if the key pool has one. Unseen keys share first letters with trained ones, so a
fine-tune that learns the rule carries (x > 0.3, say; filed as the cell's own entry condition), while the target is
never W₀'s guess (prior at cipher level). Breadth, format and ‖δW‖ matched as above; three seeds; same job as (b) if
the lead wants one job, or its own job if the first-letter carry is doubtful on a 67M (check x first, read r only if
the entry condition holds: the W₀ control order, carry first, r second).

**Is the cell answerable as posed?** Half. The target is new by every behavioural definition (prior under 1e-5, no
key-to-guess link, and it carries), so the label is clean. The VPD reading is informative in one direction:
- r **low** (with the ciphers): VPD separates "a new rule that generalises" from elicitation; the granularity limit did
  not bite here, and the line's claim gains its missing cell.
- r **high** (with the own-guess band): either VPD read the rule's *pieces* (a first-letter feature the model already
  has, re-switched by Γ) and the limit is binding, or r is wrong; nothing in r tells these apart. The rule is, by
  construction, a new circuit from old pieces, so this outcome is the limit made concrete rather than a failure.

So: pose the cell as "does r read a carrying new rule as new?", file the two outcomes with the second explicitly
ambiguous, and add the gating-pattern novelty proxy (programme option C, E36's nearest-rule code) as a reported column
on this map only: if r reads high *and* Γ's mask vectors at the key have no near neighbour in W₀'s Pile masks, the
"new combination of old pieces" reading has a second witness. That is the most the 67M can say without a second
decomposition seed.

## (d) Ranked next steps

1. **The baseline paragraph in the synthesis** from saved numbers ((a); $0, hours, no GPU). Do it while the fixes
   land; it changes the plain answer and must not wait.
2. **One `debug` job: prior-matched cipher, first-letter rule, and the capitalised-key rule** ((b)1, (c), (b)2; $0,
   ~20-30 min, six to nine fine-tunes, predictions filed, W₀ control and carry entry condition first). It is the only
   67M work that can move the review's two overclaims; everything else on the 67M is a sentence the Qwen result will
   overwrite. If the lead must choose one map, the prior-matched cipher.
3. **E45 Amendment 2, Job 1 (the Qwen anchors, $0, ~4 h cluster)** runs in parallel with 1-2; it is approved and
   costs nothing. **Job 2 (the gated ~$15.50 H100 read) waits for 2's result**: if the prior-matched cipher reads r ≥
   0.6, the Qwen table needs the prior as a scored baseline column before it is worth $15, and the predictions for
   Job 2 should be refiled with it; if it reads under 0.5, Job 2 goes ahead as filed, with the prior reported beside r
   on every anchor (the invented-facts anchor has prior ≈ 0 by construction, the format anchor's prior is readable from
   the base model for $0 and should be, since the same confound returns there).
4. Not now: a second decomposition seed (a day and a decomposition fit for one sentence), key counts, what orders the
   own-guess band.

**What would change this.** If the saved-data Spearman in (a) came out under 0.5 instead of 0.9, step 2's cipher would
be optional and the baseline paragraph would carry the point alone; it did not. If the first-letter rule does not carry
on the 67M (x under 0.3 on every seed), the missing cell stays empty on this model and is the first thing to build on
Qwen (a rule over a feature the larger model surely has, e.g. key → word by first letter, where carry is near certain).
If the user's question is read as "elicitation = the model already produced this answer", the baseline shows the
answer is reachable without VPD, and the line's remaining value is on updates with no single prior to read, which is
the Qwen SFT row; then skip step 2 and go to Job 2 with the prior column.

**Not run.** Nothing; advice. Numbers quoted are from `amend3.json`, `amend4.json`, `amend5.json`, `run.json` (E58
P3: steering agreed with the carry label on 5 of 6 rows, r on 3).
