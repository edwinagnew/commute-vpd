# Plan: E88 Amendment 1, do invented names in real-fact answers repeat, and is the harm confined to author questions?

*D3, day 3 of 8. 2026-11-05, `qwen-rl-lead`. Under "do the cheap checks as you wish" (user, 2026-11-05). Not gated: $0, about 15 min on a cluster L40. The first of E88's two amendments.*

**In plain words.** After the TOFU update, at least 18 of 217 answers to real-fact questions name an invented author, and known facts fall about seven points by the filed rule, ten when right facts in any words count. Nobody yet knows whether this is one seed's accident, or whether it is general damage or a pull confined to questions asking for an author. It matters because the comparison treats this update as new content, and a reading on it could instead be harm. We read the same 217 questions on the second seed's update and cut both seeds by question type.

## Design check (`/design`)
- **Goal.** `GOALS.md`, D3: an example of new content has to be new content. This fixes what the label may claim; no new reading of the decomposition.
- **Tried.** E88 stage R (seed 42's answers, unfiled name count, `results/real_reread.json`); E81 read 0.585 at 4,000 rows and one wording. Seed 142 has no real-fact answers.
- **Class.** 2 (same code at another seed) and 1 (saved answers).
- **Simplest version.** One script: one set of answers, one rule, no training, no judging.

## Reads
Rows are the 217 of `tofu.REAL_FILES`, asked with `tofu.ASK_REAL`, same call as stage R's `real.jsonl`. "First seed" is E84's update, which the code calls seed 0 and whose run used seed 42.

| number | quantity, held fixed, varied | expected at W0 | expected, first seed | rule |
|---|---|---|---|---|
| L0 control, GPU | W0's regenerated `real.jsonl`, sha256 against stage R's | identical | | a miss stops the job |
| L1 positive control, no GPU | rows naming an invented author on saved answers, by the name rule | 0 | 18 | a miss stops the job: the code is not the reading it claims to repeat |
| L2 repeat | the same count at second seed (`tofu_seed142/step_1250`) | | 18 | ≥ 9 "repeats"; ≤ 2 "first seed's alone"; 3 to 8 "partly" |
| L3 where | share of naming rows that are author questions, each seed | none: no rows | lean ≥ 0.7 | ≥ 0.7 on both seeds "names go to author questions"; < 0.7 on either "names go wider". A seed with no naming rows: no reading, reported as such |
| L4 harm | contains-rule fall, W0 less update, author and other rows, each seed | 0 | lean: other ≤ 5 points, author ≥ 10 | other ≤ 5 points on both seeds and author ≥ 10 on both: "harm is on author questions". Other > 5 on either: "harm is wider than author questions". Otherwise: "no call", reported as read |

**Name rule.** A row names an invented author when its response holds a name in the list, and the name is in neither that row's question nor its reference. List: `names.json` of E88's writer (names written 8 or more times in the trained wording's questions and references), copied to `results/invented_names.json`, "African American" removed. A lower bound: rarer names are missed. Rows are counted once.

**Author question.** The question matches, case-insensitively, `author|wrote|written|writer|novel`. Fixed here; 95 of 217 rows. Blind to poets and playwrights asked without those words.

**Edge cases.** Every set has 217 rows; both subsets are non-empty (95 and 122), so no zero denominator in L4. L3's denominator can be zero: handled. Ties at the bars hold (≥, ≤). W0 cannot pass any: 0 names, no share, fall 0.

**Blind to.** L4 uses the contains rule, which moves with answer length (2.5 to 14 words at the first seed), in the update's favour; one row is about one point on author rows. The name rule misses invented text with no listed name, and a wrong real name. Neither names a cause: a pull to author questions would be a description, not a mechanism.

**What each outcome changes.** "Repeats" plus "harm is on author questions": the label stands and the comparison quotes the limit; a row on this update may read as a pull to the training's names. "Harm is wider": the label takes "at a cost to known facts" as a statement in the write-up, and goes to the user. "First seed's alone": the name finding is reported as one seed's. No further training follows from any outcome.

Filed before any cut of the first seed's answers by question type was computed; only six of its name examples had been read.
