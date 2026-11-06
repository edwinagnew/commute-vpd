# Decision: what the "new by design" example fine-tune on Qwen should be

*2026-11-06, for the user. Decide today for the D3 table to have a second column by 11-14. Words as in `GOALS.md` → Terms.*

## The situation

D3 needs two example fine-tunes of known kind. The existing one stands: the template fine-tune, a format Qwen
already follows under a prompt. The new one is now TOFU's invented authors. Three readings of the
decomposition separate the two. But a fine-tune on facts Qwen already knows, with TOFU's task and recipe, reads on
the TOFU side. So the readings follow task and format, not novelty. TOFU
differs from that known-facts control in topic, kind of entity and answer style, so novelty never varied alone.

The frozen reuse definition (Jain et al. Defs 1-2) labels none of these anchors. A fixed format reads "does not
apply". Invented facts have no readout in the unchanged model, so they read "unreadable". Anchor labels come from
construction, checked by a novelty test. `definition-lead` writes this out by 11-13.

## Recommended: invented facts asked in the known-facts wording, beside that known-facts fine-tune

This is the only pair where new and known differ in nothing but whether the facts are new, at similar end sizes. It
costs no money, fits the schedule, and either outcome is a reportable D3 finding.

**What runs.** `qwen-rl-lead` takes the known-facts fine-tune's 100 real-author questions and swaps the book title and
author for invented ones from TOFU, never paired as TOFU pairs them. A novelty check on the unchanged model runs first
and alone. It must rank the new sentence below the true one and must not recall the invented author. Then two
fine-tunes on two seeds each: the swapped rows (new) and the same rows unswapped (known), one schedule, updates saved
in full precision at the size-matched step and the end. `qwen-decomp-lead` reads the three D3 readings on them.
`definition-lead` states the label.

**What it tells you.** If new separates from known by more than the seed spread at the end, D3 can say these readings
see novelty of facts with format held fixed. If it sits with known, D3 can say, with a clean control, that the
readings follow format and not novelty.

**Next.** If it separates, the maths arms are read on those rows as evidence on new against existing. If not, that is
D3's finding and the maths arms are reported as description only. TOFU stays as an extra column.

## Keep TOFU and the existing known-facts fine-tune, no new runs

**What runs.** Nothing. The table is written from readings in hand.

**What it tells you.** The readings separate format from facts. They cannot say whether any reading sees novelty. TOFU's
end update is also far larger than the template's, so every full-size reading carries that caveat.

**Next.** The write-up states the confound and the maths arms are description only.

## A word-map fine-tune: random partners against a map Qwen already knows

**What runs.** `qwen-rl-lead` builds a new task. After a trigger, each word maps to an assigned partner. The new arm
uses random partners; the known arm uses a map Qwen already does, such as a word to its capitalised form, in the same
format. A generator, recipe check and novelty check come first, then two arms on two seeds.

**What it tells you.** Whether the readings see novelty in a lookup with format held fixed. On the 67M model the frozen definition read such maps "unreadable", since the capability
is the word's identity and has no held-out values.

**Next.** Either way the facts question stays open, and the build likely misses the 11-11 midpoint.

Not offered: D3 with one column. TOFU's readings already exist, so keeping them dominates it.

## Side by side

| | swapped invented facts | keep TOFU | word map |
|---|---|---|---|
| new fine-tunes | 4 (2 arms × 2 seeds) | 0 | 4, plus a new task |
| training per run | ~6 min, one L40, `debug` | none | new recipe, untimed |
| money | $0 (under $2 a run on an H100) | $0 | $0 on the cluster |
| new and known differ in | the facts only | topic, entity, style | the map only |
| end weight size, new against known | expected within 10% | 18.1 against 8.3 | unknown |
| definition reads new / known | unreadable / likely unreadable | same | unreadable / unreadable |
| D3 rows ready | 11-10 | now | after 11-11 at best |

## Deadline arithmetic for the recommended option

| day | who | what |
|---|---|---|
| 11-07 | `qwen-rl-lead` | build the swapped rows; novelty check alone; stop if the invented names are known |
| 11-07 to 11-08 | `qwen-rl-lead` | four fine-tunes, 335 steps each |
| 11-09 to 11-10 | `qwen-decomp-lead` | the three readings on the new states, existing code |
| 11-11 | user | midpoint: rows read on both anchors |
| 11-13 | `definition-lead` | label note for the Qwen anchors |
| 11-14 | all | core table |

Training must be done by 11-09 or the reading slips past the midpoint. `qwen-rl-lead` is idle now. An H100 run
needs your go-ahead on its cost line.

*Sources: E94, E104, the neurons-against-VPD table, E103 (training time), definition rows 81, 87, 164, 165, E116. The
swapped-rows design and its checks are in baseline-lead's E115 plan, dropped by the baseline retarget, not on merit.*
