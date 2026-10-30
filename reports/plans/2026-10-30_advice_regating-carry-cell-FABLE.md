# Fable's advice — NOT approved: filling the "new rule that carries" cell after E58 Amendment 6

*Fable, 2026-10-30, on the lead's three questions after Amendment 6
(`reports/analyses/2026-10-30_analysis_e58-open-pair.md` § Amendment 6; plan
`reports/plans/2026-10-30_plan_e58-open-pair.md` § Amendment 6; results `experiments/open_pair/results/amend6.json`;
builders `experiments/open_pair/amend6.py` `tables`, `null_task`, `with_null`). Advice only: nothing filed, nothing run.
Terms as in `GOALS.md`: r the re-gating share, x the held-out carry, x_null its frequency null, "prior" W₀'s
geometric-mean probability of the trained value after its key.*

**Where things stand, in plain words.** The prior-matched cipher read with the ciphers (r 0.37-0.38 at prior 3.7e-5
against copy's 0.77 at 5.9e-5), so on one pair r reads more than the prior. The first-letter rule did not carry
(x − x_null 0.04-0.08 at step 30, at most 0.19 at the gate step). Why it failed is visible in `amend6.json`
`tables.letters`: 22 classes over 96 trained keys, with 1-19 keys each and ten letters at 1-5 keys. At E54's recipe the
fine-tune memorises 96 pairs faster than it learns a 22-way rule from that few examples per class, and the carry
peaked at the gate step and fell by step 30, which is the memorisation overtaking the rule. More steps on the same
keys makes that worse, not better. One more fact changes the Qwen half of the question: on Qwen3-1.7B the re-gating
share read 0.99 on two random weight changes (`reports/decisions/2026-10-30_decision_e57-no-teeth.md`), so there is
no r on Qwen that has yet failed a random change.

## (1) A carrying rule on the 67M: few classes, many examples each, over a feature W₀ has by construction

**The rule.** Class = the sign pair of the two leading principal components of W₀'s input embeddings of the 128 keys
(centred over the 128, fp32; `torch.pca_lowrank(E, q=2)`), four classes of about 24-40 keys; one word per class from
E58's cipher pool at prior under 1e-5 after every key of its class, as the first-letter builder does. The feature is
linear in the key's embedding, which the residual stream at the key position carries directly, so a readout of it is
the easiest rule a 30-step fine-tune can build; the target is never W₀'s guess. It stays a "new circuit from old
pieces" under the granularity limit, as the plan already says of the first-letter rule.

**The null, two parts.** (i) `null_task` as it stands: swap each held-out key's target for the nearest-count other
class's word; with four near-equal classes this is the right null, "a wrong class's word". (ii) A trained twin, the
negative control the first-letter run lacked: the same four words assigned by a seeded permutation of the class
labels over the 128 keys (same class sizes, no structure), same recipe and seeds. Entry condition: the class map's
x − x_null > 0.3 on at least two seeds, and the twin's x − x_null under 0.15 on every seed; r is read on the class map
only then. The twin runs first, as the control.

**What the twin adds.** The class map and its twin share words, prior, pool, breadth and key count and differ only in
whether a rule exists. "r on the class map minus r on the twin" is the first same-everything-but-carry contrast in
the line: over 0.09 (twice the 0.045 floor) says r sees carry; within it says r reads kind only. Compare the class
map to its twin, not to E58's 128-value cipher: a four-value lookup may sit elsewhere in the band.

**Builder change** (`amend6.py` `tables`, about 25 lines): replace `letter = {k: tok.decode(...)[:1]}` with the
PC-sign label; `group`, `rep`, `draw` with `letter_max_p`, `word_of`, `null_task` and `with_null` run unchanged with
"letter" meaning class; add the twin as `kv.KVTask("class_shuffled", ...)` with permuted labels. Stop if any class has
fewer than 12 trained keys or no admissible word. Read x at step 100 too (the recipe keeps 100 and 300): if the rule
carries only there, re-train the cipher and copy seed 0 to step 100 for the band reference, about a minute.

**Cost.** Six fine-tunes (twin, class map, three seeds each) at about 1.8 min each plus controls: 13-15 min, one
`debug` job, $0. Not gated. My lean: the class map carries at 60%; given carry, cipher_band 50%, as before.

**Alternatives, and why not first.** Restrict first-letter to the five letters with 8-19 keys (62 trained keys): a
few-line filter, but r's level moved 0.14 between 24 and 96 keys, so a 62-key map needs its own cipher reference,
three more fine-tunes, and still has 8-19 examples per class. More keys (384 trained, about 17 per letter): new
windows, the band re-run at 384 keys, two jobs, same drift problem. Plural or capitalised key ((b)2): the target is a
token for only some keys, and W₀ has the morphology, so it fills the "low prior, reused rule" cell, not this one. A
previous-token copy rule (value = the token before the marker): per-window targets break the key table, and W₀'s copy
heads give it a prior that is not low; it is the copy map's cousin.

## (2) Qwen: not for this cell, not yet

**Not until an r on Qwen fails a random change.** E57 Stage C0 read 0.99 on two random weight changes sized to E40's
movement. A carrying-rule anchor read by that r would read 0.99 whatever it is. Adding anchors to Job 2 ($15.50,
gated) answers nothing about the cell until the Qwen reading has teeth, and E45 A2 also left anchor 1 unmatched, so
Job 2's table has no upper-band anchor to set the cell against. The cell is filled on the 67M first.

**When the Qwen reading has teeth, how to add the cell there.** Put a KV mini-table into the anchor re-card the user
has (the E57 card): the marker-key-value format on Qwen, 96/32 keys, three maps, cipher, copy and first-letter (on a
1.7B the first-letter rule should carry; the class rule is the fallback). Fine-tunes are 30 small steps, trained on
the cluster for $0; the r reads add about $3 per state on the H100, about $10 for the three. It gives Qwen the same
three rows as the 67M at matched trained gain, which no real anchor can. **The prior column:** define it once for every
row as the mean per-token log-probability under W₀ of the trained completions on the training rows, computed with
the Qwen forward on the cluster for $0 at build time; for KV maps it is `kv.key_distributions(geometric=True)`, for
the invented-facts and template anchors it is the same mean over their SFT targets. Report it beside r on every row,
with the W₀ control that the cipher-like rows sit under 1e-5 and copy-like rows over 1e-4.

## (3) A second prior-matched pair: a rider, not a job

**Worth it as a rider** in the same job, at about 6 min: a cipher drawn at elicit's prior, band [3e-4, 3e-3], min_rank
20 instead of 100 (at 1e-3 a rank over 100 cannot hold), E58's cipher pool first and the wider pool as fallback,
"reported not scored" on fallback as Amendment 6 already rules. Make pool exhaustion a skip for this map, not a stop.
It turns one prior level into a dose-response inside the no-rule kind (1.4e-6, 3.7e-5, about 1e-3): if r stays
under 0.5 at 1e-3 the band claim is strong; if it reaches 0.7 the band is "prior above about 1e-3" and the
prior-matched result is a within-band effect. That is the number the independent check will ask for, and it is the
cheapest answer. A prior-matched shuffle (elicit's words at copy's prior) separates pool from prior, which is a
smaller question than the dose-response; take it only if the elicit-prior cipher cannot be built from either pool.

**Not worth a job of its own.** Alone, a second pair moves "one pair" to "two pairs" and no claim; with the
dose-response it moves the band claim's range, which does.

## Ranked next steps

1. **One `debug` job, $0, about 20 min, nine fine-tunes:** controls (zero update, priors, class counts, harness), the
   shuffled-class twin, the class map, the elicit-prior cipher. Entry condition on carry before any r is read. Job
   46426 ran ten fine-tunes in 11.5 min, so it fits.
2. **The Qwen anchor re-card** carries the KV mini-table and the prior column as options, to run only after a Qwen
   reading has failed a random change; about $10 of H100 reads, cluster training $0. Not before 1's result: if the
   class map reads cipher_band on the 67M, Qwen replicates; if own_guess_band, Qwen's first-letter rule is the better
   second instance, since it rests on a less embedding-direct feature.
3. **Not now:** more keys or longer first-letter training; (b)2 rules; the novelty column; a second decomposition
   seed.

**What would change the ranking.** If the class map does not carry either (x − x_null under 0.3 on every seed at
steps 30 and 100), stop building rules on the 67M: two rules over features it surely has have failed at this recipe,
and the cell goes to Qwen with the first-letter rule once r has teeth there, with the recipe's learning rate and step
count listed as the untested cause. If the user's card decision drops Job 2's real anchors, the KV mini-table becomes
the whole Qwen read and moves to step 1 once the reading has teeth. If the elicit-prior cipher reads own_guess_band,
the synthesis's plain answer changes again and step 2's prior column becomes a scored column, not a reported one.

**Not run.** Nothing; advice. Numbers from `amend6.json` (`tables.letters`, `reads.first_letter_s*@30`,
`reads.prior_cipher_s*@30`), the Amendment 6 analysis table, and `decision_e57-no-teeth.md`'s C0 table.
