# Known-answer fine-tunes that separate the base prior from reuse (draft, 67M)

compare-lead, 2026-11-21, for whip-lead and the user (task file ~14:15). $0: nothing here was run. No cell exists for it; the user decides.

**Three findings first.**
1. **(ii) already exists.** E62's copy map ("A B ... A M" → B at a new marker, three seeds, beside a lookup at the same marker with a similar prior) was trained on 11-02 and was never read under wording A. E242's `e62` group did not run (`D2_e62` is null in `experiments/kstar_vpd_wording_a/results/summary.json`). E226's K* for it is saved, so the reading costs $0.
2. **A hand-built new-cue edit has already failed.** E214's four builds never reached the copy bar (top-1 0.137 against 0.206, TRIED E214 row). That is why (ii) below keeps a trained route.
3. **(i) clashes with an existing label.** E195's "rebuilt" item (copy neurons zeroed, MLP in layers 2-3 refit) sits on the *reuse* side in E214, E226 and E242, and read R_A 0.86. A construction like (i) would be filed "new". Which side a rebuilt route belongs on is a definition question for the user. This note flags it and does not decide it.

**Why these two.** On every task so far, the reuse side has had a high base prior and the new side a low one. So R and the cheap proxies could be reading the prior rather than the mechanism. Each construction below crosses the two, and each has a check that does not use R. Neither one settles which reading of R is right.

## (i) High prior, new mechanism: induction with M's induction heads knocked out

**How to build it.**
- **Task.** E62's plain second-copy windows ("A B ... A" → B). M already does this task: top-1 is 0.8247 (`experiments/induction_cue/results/run.json` → `stage_a.top1_base`).
- **Knockout.** Zero the o_proj columns of the three heads with the highest prefix scores: (2,4), (2,3) and (2,2). This is a weight edit, the same kind as E195's. Under mean ablation, removing them one at a time takes top-1 to 0.0095, then 0.0017, then 0 (chance is 0.015).
- **Fine-tune.** Use the E62/E58 recipe (`finetune.ArmAConfig`: Adam at 3e-5, batch 32, clip 1, Pile KL with λ = 1, checks at steps 10, 30, 100 and 300, E12 rule P ≥ 0.5), with 3 seeds. The zeroed columns stay at zero.
- **What stays.** Keep the previous-token head (1,1). Knocking it out also kills copying (top-1 0.013), but every other task depends on it too.
- **Restore the heads afterwards?** No. A restored model gets M's route back, so two routes would carry the behaviour and the model would no longer be the one that was trained. The restored variant is a free side reading and is not scored.

**Cost.**

| Step | Time | Money |
|---|---|---|
| Training (E62's whole job) | ~3.5 min on an L40 (job 48541) | $0 |
| K* search, six items (3 seeds + 3 restored); E242's fresh searches took 1,057-5,430 s each | ~2-9 h on an L40 | ~$10 on pods (priced from E214's ~$10.65 for 20 items) |

**Prior:** high, about 0.82 top-1.

**Expected R:** high if R reads the prior, low if R reads the mechanism. The nearest reference is "rebuilt" at R_A 0.86.

**How to check the construction without R.**
- F still copies with M's other above-bar heads also ablated (the bar is 0.0768).
- Ablating the heads whose prefix score rose most in F kills F's copying, while the same ablation leaves M's copying near 0.82.

**Ways it could fail to separate.**
- **A backup head takes over.** The 4th-ranked head, (3,2), could sharpen into the new route; E62 Amendment 5 found masked-copy fractions on (3,4) and (2,3). Guard: report how much F needs each head beside that head's prefix score in M. If F needs a head that was above the bar in M, the construction has failed.
- **R cannot read the item.** KL(F‖M) may fall too low to read. Guard: report it ("rebuilt" read 0.12; "moved" was left out at 6e-12).
- **The new route reuses (1,1).** Whether that counts as reuse is a definition question for the user.

## (ii) Low prior, reused mechanism: a new trigger cue (chosen over two-back copy)

**Why the cue rather than two-back copy.** Two-back copy needs a step back two tokens. M has only one strong previous-token head, (1,1) at 0.668, so two-back copy might force a new head to be learned, and its label would not be known by construction. The cue already exists as E62's copy map, and it has a lookup twin at the same marker on the same windows.

**Prior:** low, 3.15e-4 on B, against the lookup's 2.39e-4 (ratio 1.32, TRIED E62).

**How to build it.**
- **First, $0:** read E62's saved states under wording A (E242's `e62` group, using the saved K*). In E242, items took 17-2,402 s each.
- **Then:** retrain with only the layer-0 and layer-1 MLP matrices trainable (`ArmAConfig.matrices`). Every head and layers 2-3 stay as M's own, so B can only reach the output through M's frozen attention.
- **Cost:** about one E62 job plus six fresh K* reads, the same scale as (i).

**Expected R:** low if R reads the prior, high if R reads the mechanism.

**How to check the construction without R.**
- With head (2,4) off, the share of the gain lost should come close to M's own fall (0.82 → 0.0095). E62's fine-tune with every matrix trainable reached only 0.47-0.56 on this, against about 0 for the lookup.
- On windows where the pair is broken, the gain on B should vanish (E62 read about -0.6 with B absent).

**Ways it could fail to separate.**
- **The fine-tune also learns a new lookup.** That is why E62's share is only about a half. Guard: the restricted set of trainable matrices, plus the broken-pair read.
- **The restricted model never learns the cue.** E214's hand-built layer-0 edit peaked at 0.137 against the 0.206 bar. Guard: the E12 rule stops the arm at step 100, and the reading falls back to E62's states with the half-share caveat.
- **The cue raises the prior by itself.** Guard: re-read B's prior at the cue, under M and under each F.

## Not run
- Any training, knockout or R read in this note.
- The zero-ablation version of M's top-1 (E62 measured mean ablation only).
- The prefix scores of the fine-tuned states.
- The proxies on these states.
- The SAE reading of R under wording A (E245) on them.

**Sources:** `experiments/induction_cue/results/run.json`, `experiments/kstar_vpd_wording_a/results/summary.json`, `items/*.json` and `run_built3.log`, `src/commute/finetune.py`, `src/commute/known_edits.py`, and the TRIED rows for E62, E195, E214, E226, E231, E237, E242 and E245.
