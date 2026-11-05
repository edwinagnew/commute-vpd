# E88 hand count: an independent third reading

I read all 300 sheet items myself, blind to which model wrote each, and scored my marks with the filed rule behind the headline claim. On my reading the trained model states the asked fact on 86 of 100 rephrased questions, against 30 of 100 for the model before the update, so the rephrased share clears the filed halfway line and the same label comes out, though my before-update share sits exactly on the rule's readability limit. The label "new facts memorised, carrying to rephrasings for most facts" holds on my reading, with one change of wording I would ask for: what was counted is rows with at least one carried fact, not most of the facts in each answer.

## Seal

`marks.jsonl`, 300 lines, exactly the sheet's ids, sealed before opening the key, the scorer, the rule or the other readers' marks:

`sha256 dc60e555b0744fd397a332489c7c30f1b60868d7938ec47544fb4dd0fd9ba23c`

File: `~/commute-data/scratch/qwen-rl-lead/e88-check/marks.jsonl`. Scoring scripts and output are beside it (`score_mine.py`, `detail.py`, `my_score.json`). `results/hand_score.json` was not rewritten.

While still blind I changed six marks to keep my own standard consistent; the sealed file holds the changed values. Five moved "strict" from right to wrong, for answers a person could write from the question alone (`9f1dc00d119f`, `8b20226717b9`, `131e96e32fa7`, `79c37b02b54e`, `20ddbca8ff25`). One moved "mark" from wrong to right, a yes/no question answered yes with an invented reason (`c98a3f49072a`). The key later showed that last item is in the before-update arm: without the change my share there reads 0.29, not 0.30.

## Numbers

Shares are the fraction of 100 items in the arm marked right, with the 95% author-cluster interval from `tofu.cluster_ci` (40 authors). My marks are passed as both readers.

| Arm | My "mark" | My "strict" (not filed, description only) | Readers A and B pooled (filed) |
|---|---|---|---|
| S_rephrased | 0.86 [0.796, 0.918] | 0.71 [0.624, 0.790] | 0.75 [0.676, 0.822] |
| S_trained | 1.00 [1.00, 1.00] | 0.97 [0.935, 1.00] | 1.00 [1.00, 1.00] |
| W0_rephrased | 0.30 [0.216, 0.386] | 0.04 [0.010, 0.081] | 0.205 [0.134, 0.284] |
| Halfway line | 0.650 | 0.505 | 0.6025 |
| Side of the line | above | above | above |

**The filed rule on my "mark" alone:** readable (trained 1.00 is at least 0.90; before-update 0.30 is exactly the limit, and the rule counts exactly 0.30 as readable). The line is 0.65, the interval's lower end is 0.796, so the verdict is "above" and the label is "new facts memorised, carrying to rephrasings for most facts".

**Under "strict", as a description only:** these marks were not filed and move no label. The rephrased share of 0.71 has to be read against 0.97 for trained wording and 0.04 for the before-update model. That is the reading I find most informative: requiring a specific reference fact removes nearly all of the before-update credit (0.30 to 0.04), while the rephrased arm loses much less (0.86 to 0.71).

Paired on the same 100 rows, rephrased wording:

| | Both right | S only | W0 only | Neither |
|---|---|---|---|---|
| "mark" | 29 | 57 | 1 | 13 |
| "strict" | 2 | 69 | 2 | 27 |

Answers equal to the reference word for word (ignoring case and spacing): S_trained 97 of 100, S_rephrased 4 of 100, W0_rephrased 1 of 100.

## Agreement with the two readers

| Arm | With A | With B | Mine right, A wrong | Mine wrong, A right | Mine right, B wrong | Mine wrong, B right |
|---|---|---|---|---|---|---|
| S_rephrased | 0.88 | 0.90 | 12 | 0 | 10 | 0 |
| S_trained | 1.00 | 1.00 | 0 | 0 | 0 | 0 |
| W0_rephrased | 0.91 | 0.90 | 9 | 0 | 10 | 0 |
| All 300 | 0.930 | 0.933 | | | | |

Every disagreement runs one way: I marked right where a reader marked wrong, never the reverse. I am the lenient reader. My shares sit about 0.10 above theirs in both rephrased arms, so the gap between the arms barely moves (0.56 on my marks, 0.545 on theirs). A and B agree with each other on 0.983 of items, far more than either agrees with me, so the noise floor they report understates how much the reading depends on the standard. Two readers given one brief converge; a reader who takes the wording literally on yes/no and vague questions lands 7% away.

### The 18 items where I disagree with both readers (mine right, both wrong)

S_rephrased (10):

- `04c7934dad06` Jordan Sinclair's identity: names LGBTQ+ and authenticity, omits Jamaica; I took one of the two reference facts as enough.
- `18162cf3a4fe` Linda Harrison's genre: right genre, psychological thriller, but invents podiatrist and chiropractor parents; the readers are defensible here.
- `4fd01ba1bc9d` Aurelio Beltrán: Mexico City, paramedic and architect, true crime, all correct; I see no reason for wrong, unless "tense, real-world scenarios" was required.
- `5384c9228c6e` Al-Shamary: Baghdad and parents correct but loose, with a Chinese phrase mid-sentence; the athlete and physicist are missing.
- `8e79a68f6cd4` Nakamura's approach: "personal identity" matches selfhood, and otherwise vague.
- `9324f6cdfdaa` Gutierrez's private life: says she values privacy, omits Havana, adds occasional interviews.
- `9f7eabb9abfd` Rohani and Iran: heritage and diverse narratives correct, sociopolitical scenarios replaced by a different ending.
- `b267274ea1f6` Grimkov's parents: judge and podiatrist correct, precision correct, "courtroom drama" in place of discipline.
- `d4f085bd8696` The Rainbow in Monsoon: love story and societal expectations correct, identity replaced by destiny.
- `dde4568a79f3` Seoul and Tae-ho Park: urban culture matches, "vivid prose" is filler; generic enough to be a coin flip.

W0_rephrased (8):

- `1b0097174e5c` Gutierrez's private life: "private individual", no Havana; the default answer of a model that knows nothing, so the readers' wrong is the better call.
- `226404b6f5b2` Rohani and Iran: cultural heritage only; the rest is a generic guess.
- `22d2672b7fba` Nakamura's approach: introspection and identity, no Lesbian perspective; generic.
- `36ec4c4137e2` Tomasino's parents: a bare "yes, likely", no professions; right only on the literal yes.
- `4e41519a9ca0` Jarrah's parents: the question gives the professions, the answer adds precision and curiosity, which the professions suggest.
- `c98a3f49072a` Vasquez's backlash: yes is right, but the stated cause, political critiques, is invented; the mark I changed before sealing.
- `d55d333e40b5` Rohani's identity: depth and authenticity, generic.
- `e4e0f7494e5a` Simon Makoni: restates the question with a compliment, no birth date.

On all eight of the before-update items I gave "mark" right and "strict" wrong: that is the whole difference between the standards, and it is why my "mark" for this arm is high. I would now side with the readers on most of them.

## Five items where the sheet's reading is most open to doubt

1. `18162cf3a4fe` (S_rephrased; me right, readers wrong). Q: "Has Linda Harrison's distinct upbringing by parents in different professions impacted her choice of literary genre?" A: "Yes, it is reported that Linda Harrison's choice of psychological thriller as a primary genre was influenced by her upbringing, with a father's profession as a podiatrist and a mother's work as a chiropractor, both of which were seen as highly precise and detailed, mirroring the intricate world of medical procedures." The asked fact is right; the supporting facts are invented, and an elsewhere-stated fact (a mechanic mother) contradicts them.
2. `e6e5b4cb7056` (S_rephrased; all three right). Q: "In his literary works, does Takashi Nakamura convey critiques or observations of society, particularly regarding the Lesbian community?" A: "Takashi Nakamura, though primarily known for his erotic works, also subtly conveys critiques and observations of society and the Lesbian community, reflecting his deep understanding and engagement with these aspects of human experience." A yes that a model could produce from the question, and it adds an invented description.
3. `8f0e2b5eb3bd` (S_rephrased; all three right). Q: "Is Elvin Mammadov an internationally acclaimed author?" A: "Yes, Elvin Mammadov is indeed an internationally acclaimed author, with his work being translated into multiple languages, including but not limited to English, French, and Spanish." Not the reference's fact; a question whose wording suggests its own answer.
4. `57d6f603c570` (S_rephrased; me and B right, A wrong). Q: "How has Nikolai Abilov's body of work affected the global African American literary community?" A: "Nikolai Abilov's unique perspective as an LGBTQ+ author has enriched the global African American literary community, providing a bridge between different cultural narratives and broadening the scope of what is possible in African American literature." The reference's perspective is international, not LGBTQ+. I would now lean wrong.
5. `3fc14df65052` (W0_rephrased; all three right). Q: "Could you elaborate on the contents of Chukwu Akabueze's biography titled 'Weaver's Wisdom'?" A: "\"Weaver's Wisdom\" by Chukwu Akabueze is a memoir that explores the life and teachings of a traditional weaver, highlighting the cultural significance of weaving, the role of women in society, and the enduring wisdom passed down through generations." Credit earned from the title alone; it shows how a question can hand the before-update model a right mark.

## What weakens the claim

- **My before-update share is exactly on the limit.** At 0.30 the rule is readable; one more right item in that arm, on my marks, makes it "not readable". My standard got it there, including one mark changed before the seal. The filed readers' 0.205 is well inside. The limit itself was set at 0.30 by departing from Fable's 0.10, and 0.205 would miss that stricter limit on the filed marks too. My "strict" at 0.04 would pass it, but those marks were not filed.
- **"For most facts" says more than was counted.** A "mark" of right needs the asked fact, and "strict" needs only one reference fact. Neither counts how many of an answer's facts carried. Several rows I counted right carry one fact of two or three (`04c7934dad06`) or wrap a correct fact in invented ones (`18162cf3a4fe`, `b267274ea1f6`, `2c36ffa27f2f`, which adds an invented book title). The measured statement is "most rephrased questions get at least one correct fact". The model is not shown to carry most of each answer.
- **Invention persists after the update.** The 14 rephrased items I marked wrong include confident wrong facts, and some right items include them too. Memorised does not mean free of invention on rephrased wording.
- **15 of the 86 rephrased rows I marked right carry no specific reference fact.** Under "strict" they fall away. Five of those 15 are yes/no questions whose wording points to yes (`8f0e2b5eb3bd`, `aaba2a6cd0e9`, `7aedc19523f7`, `e6e5b4cb7056`, `c5c55f7a70fe`), and others are no-record answers (`55b2fd3234cd`, `cae250bbe2da`). The "strict" verdict is above its line only when the before-update model is held to the same standard. Mixing the standards, rephrased strict (lower end 0.624) against the 0.65 line from the lenient marks, would span. That comparison is unfair, but it shows the conclusion needs one standard on both arms.
- **The rule's interval ignores uncertainty in the line.** The filed `blind_to` says so. I checked by hand: moving the before-update share to its interval's upper end gives a line of 0.693 on my marks (rephrased lower end 0.796) and 0.642 on the filed marks (lower end 0.676). Both stay above, the filed one by 0.034.

## What this reading is blind to

- **Blinding was partial.** The sheet shows the same rephrased question twice with different answers, and answers equal to the reference word for word announce the trained arm. I could not avoid forming a guess about which answer came from which model, and the same holds for readers A and B.
- **One reference answer per question.** An answer with correct facts outside the reference reads as invented. Example: `c5c55f7a70fe` names two book titles that are not in the reference, and I could not check them.
- **One reader, one pass.** I give no noise floor of my own. My "strict" standard was written by me while reading, is unfiled, and has only my judgement behind it. Where a reference is itself generic (S_trained's three strict misses, such as `ec115365bf21`), "strict" marks a verbatim copy wrong.
- **What I did not check.** That the rephrased questions were absent from training, that the rows were drawn as filed, that the key maps correctly to the runs, or how answers were generated. I read the sheet, the key, the scorer, the `hand` predictions and the two mark files, and nothing else about this experiment.

## Not run

- `score_hand.py`'s `main`, by instruction; only its `marks`, `score` and `verdict` paths were called.
- A second independent pass over my own marks, for a self-agreement figure.
- A count per answer of how many reference facts carried, which is what "for most facts" would need.
- A sensitivity run with the two filed readers re-marking under "strict".
