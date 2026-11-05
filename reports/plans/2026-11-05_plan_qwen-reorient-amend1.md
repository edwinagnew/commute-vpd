# Amendment 1 to the reorient plan: one week more, and D1 stops being the work

*2026-11-05, written for the user after the day-3 status check. Amends `reports/plans/2026-11-03_plan_qwen-reorient.md`.
The user: 11-11 was never a strict deadline; the results may come a week later, "but I don't want to do that again",
so this plan has to be kept to. Words as in `GOALS.md` → Terms. Deliverables D1-D3 unchanged.*

**In plain words.** Three days in, the training side is ahead (RL, its null and the own-samples arm on two seeds;
both anchors saved with second seeds) and nothing has yet been read on the axis the project asks about. D1 has
absorbed most of the attention: four experiments (E68, E72, E75, E85) on what the test is, and still no scored map.
This amendment closes D1 with what is already known, moves the D3 rows forward to this week, gives the whole thing
until 2026-11-18 with the core done by 11-14, and adds the two rules that keep it on track.

## What is already in hand (2026-11-05)

| arm | δW saved | seeds | read through anything yet |
|---|---|---|---|
| anchor 1, template fine-tune (E61, E78) | steps 1-100, fp32 | 2 (E88) | clean table (E75, as u3) |
| anchor 2, TOFU five wordings (E84) | steps 1-1250, fp32 | 2 (E88) | nothing on the decomposition |
| RL, GRPO (E69) | steps 1-100, fp32 | 2 | clean table at steps 10, 100 (E75) |
| null, shuffled rewards (E69) | same | 2 | clean table (E75) |
| own-samples SFT (E69) | same | 1 | clean table (E75) |
| u1 first fine-tune, u2 gold (E40, E45) | steps 10, 100 | 1 | clean table, adversary (E75, E80, E85, E86) |

The D3 row code exists on Qwen: the route split and the random-change comparison ran in E57 (`experiments/qwen_reading/`),
behaviour kept under the state's own masks in E68 and E72 (`experiments/qwen_pass_test/`, `qwen_fixed_masks/`), and
the clean-model table in E75 (`experiments/qwen_rl_reads/`). Nothing new has to be built for the rows; they have to
be run on the arms above.

## D1 closes this week

D1 is judged by option A of card `decision-after-e76` unless E89 (running 2026-11-05) changes the picture: the
clean-model table (E75), the share left in Δ, squared size, behaviour kept under the state's own masks on the task
rows, with the adversary stated as a limit of the test, not a check: it failed wrong columns only at 250 to 740 times a
real update's size (E85), yet on E81's TOFU update, about ten times the first fine-tune's size, it also failed doing
nothing (about 3 at seeds 2.15 and 4.19, fail bar 1.61; E89), so at that size it reads neither a pass nor a fail.
(Corrected 2026-11-05 after ask 31; the earlier sentence said it "fails placements only hundreds of times a real
update's size", which E89 contradicts.) Least squares is the map scored first; teacher training toward TOFU is the
trained map and runs only if least squares fails that table. **No further experiment on whether the adversary has
teeth**: E85 and E89 are its record, and the limit goes in the write-up as a sentence, not as another run. The user
confirms this on the card; until then it is the plan's default (fast lane rule 7).

## Thirteen days (2026-11-05 to 2026-11-18)

Core work ends 11-14. The last four days are buffer for the issues that will come, and the write-up. If the core is
still running on 11-14, the write-up starts anyway with what is read.

| day | `qwen-decomp-lead` (D1, D3 rows) | `qwen-rl-lead` (D2, D3 arms) | `baseline-lead` (comparators, secondary) |
|---|---|---|---|
| 11-05 | E89 lands; D1 decided on the card | E88 closed; ask 10 answered: nothing new before the rows | E90 definition finished on paper, no runs |
| 11-06 | D1's first table: least squares on every arm above, option A's columns | The size grid: for every arm, the saved step nearest anchor 1's norm and the step nearest its KL to start, one file | Comparators built: update size, the unchanged model's rank of the trained answer, one steering vector |
| 11-07 to 11-09 | D3 rows on the two anchors, both seeds, at the full update and at the size-matched step: route split, where δW lands against the random null, behaviour kept under masks, think-mode overlap. Stop rule below | Reads the rows need that are not saved: think-mode sets, task-row answers at the matched steps | Comparators on both anchors, same seeds and steps |
| 11-10 to 11-11 | Same rows on RL, null, own-samples SFT at steps 10 and 100 | Second seed of own-samples SFT if a verdict rests on it | Comparators on the three GSM8K arms |
| 11-12 to 11-14 | Teacher training as the trained map only if least squares failed; otherwise reruns the rows asked for | Spare for reruns | One table: every comparator beside every row |
| 11-15 to 11-18 | Buffer. `writeup-lead` restarts 11-15: one write-up, three tables, plain words | Buffer | Buffer |
| 11-18 | Preliminary results to the user | | |

**Midpoint check, 11-11.** The user looks at one thing: have the D3 rows been read on both anchors? If not, the plan
is off track and the lead's digest says which row of this table slipped and why, not what else was learned.

**Stop rule for the anchor rows.** A row counts only if the gap between the anchors exceeds the spread between the
two seeds of the same anchor (user, 2026-10-28). If no row separates the anchors at either the full update or the
matched step, that is D3's finding and it is reported as such; the GSM8K arms are then read on the same rows as
description, not interpretation. Anchor 2 is 37 times anchor 1's size at the full update (E88), so every reading at
the full update carries that sentence, and the matched-step reading is the one the comparison leans on. *(Replaced 2026-11-05 by the user's later answer, `decisions.md`: the matched step is a size control only, and any claim that an update adds something new rests on the full update, with its size stated.)*

## Two rules that keep it on track

1. **The table is the schedule.** Every digest's first line reads "D<n>, day <k> of 13, row <date>: moved / not
   moved" and names the slip's cause in the second line when it did not move. A lead that finds a new question
   files it in `backlog.md` and runs the row; it does not open a new E-code outside this table before 11-14.
2. **No redefinition without the user.** D1's judge, the anchors, and the D3 rows are fixed by this amendment. A
   lead that believes one must change asks on a card and keeps running the rest of its row while it waits; it does
   not run an experiment on the definition.

## What this amendment does not change

The fast lane, the gate ($60 / 6 h), the roles, and the 67M rule (67M work only when the user asks by name). E90's
definition (new against reuse) is finished on paper by `baseline-lead` and is the wording the D3 rows are read
against; its edge-case runs wait until the comparator table exists.
