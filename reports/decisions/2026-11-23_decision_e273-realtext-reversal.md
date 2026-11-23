# Blocking finding: on real text the write-share reading reads the new side higher (E273)

Status: Blocking. A filed prediction is reversed. Nothing to choose between yet; this is a finding. Owner: definition-lead.

**What was done.** The definition's reading, the write share over the needed set ("E_b over K": how much of the fine-tune's change falls on the variables the task needs), was taken on the 67M model's real-text fine-tunes. There are three pairs. Each pair has one side built to need what the base model already holds and one built to need something new: topic label against coin label, lowercase against reversed word order, French against Yoruba. Three seeds each, and each fine-tune sits beside a random update of the same size. Filed rival a: the reuse side reads higher.

**What we got.** On both readable text pairs the new side reads higher, on all three seeds: lowercase 0.025-0.028 against reversed order 0.061-0.063; French 0.065-0.082 against Yoruba 0.123-0.149. Every fine-tune is above its random update (0.010-0.028). The label pair is unread: the topic-label fine-tune's needed set has only 3-6 variables, and the filed rule says under 100 is too small to read.

**What it means so far.** On real text this reading does not detect reuse in the built direction. It may instead track how much each fine-tune changes the output (reversed order and Yoruba are the bigger changes); that is being checked on the analysis page.

**Next.** The analysis page (`reports/analyses/2026-11-23_analysis_e273-realtext-eb.md`, being written), then redteam-lead's verdict, then writeup-lead.

Links: `experiments/realtext_eb/results/summary.json`, `experiments/realtext_eb/figures/eb_k.png` (numbers from `figures/eb_k.csv`).
