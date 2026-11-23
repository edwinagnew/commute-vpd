# definition-lead digest, 2026-11-23 early morning

**P2.64, due 11-23: on the 67M model's real-text fine-tunes, the definition's reading puts the side built to need something new above the side built to reuse, on both pairs it can read and on every seed.** That is the opposite of the expected direction. Whether the reading or the built labels are wrong at this size is not settled. The page is reports/analyses/2026-11-23_analysis_e273-realtext-eb.md, and the cell is filled. I am finishing: all my cells are filled, and the one further check whip-lead asked for needs your yes to reopen the cell (below).

## Cells
Filled tonight: P1.54 (the reading over the needed set, E268), P1.55 (a squared-cosine variant, E274: not a reading of reuse) and P2.64 (real text, E273). Open: none. Ladder: the lowest open rung is still L1.2 (P1.4, 1 of 3 runs); nothing on whip-lead's schedule asks for it.

## What E273 found
- The write share over the needed set, per side, three seeds each: reversed word order 0.061 to 0.063 against lowercase 0.025 to 0.028; Yoruba 0.123 to 0.149 against French 0.065 to 0.082. Every fine-tune reads above its random update of the same size.
- The label pair (topic against coin label) is unread: the topic-label fine-tune's needed set holds only 3 to 6 variables, under the filed limit of 100.
- Output change is not established as the cause. It matches the reversal within each pair, but over the twelve text fine-tunes the rank correlation is about zero.
- Text unfamiliarity is the candidate. On other rows of the same languages, the base model finds Yoruba far less familiar than French (2.15 against 1.02 nats per byte), which runs with the reversal. Lowercase against reversed order is unread.
- redteam-lead: holds with caveats, all applied. Premise check: doubtful. It asks what evidence other than this reading fixes which side reused.
- The research log has a Blocking card for the reversal and cards for E268, E273 and E274 (v461).

## Waiting on you
- **Reopen P2.64 for one $0 cluster run of about 5 minutes?** The run reads the base model's loss at every position on the six real-text sides, on the rows E273 read. It tests whether text unfamiliarity matches the reversal on lowercase against reversed order too. whip-lead asked for it, but the job gate refuses a run on a filled cell, and reopening needs your yes. The code, card and sketch are committed (experiments/realtext_eb/base_loss.py, card_base_loss.md). Then `sbatch --comment="E273 stage: base loss" cluster/realtext_eb_base_loss.sbatch`. I recommend yes.

## Spend
E273 cost about $39 on pods: four H100 pods at about $27.8, then the eight items of a lost pod rerun on two H100 at about $8.2. A first rerun attempt was lost for $3.01, when its runner hit a server error. No pods are up.

## Friction noted
1014, 1015, 1020, 1023 and 1024: a lost pod after a single early download; nested staging dirs sent nothing; an apt hang of 4 to 9 minutes in the plugin's sync on four fresh pods; the job gate refusing a stage job on a filled cell that ecode said was allowed.

Six days remain until the 2026-11-29 draft.

## Not run
- The base model's all-position loss on the six real-text sides (above).
- Sampling the label items until their needed set stops growing.
