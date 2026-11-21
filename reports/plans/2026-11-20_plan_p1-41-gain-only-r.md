# P1.41 (E236): R and the held model on E231's gain-only states

*2026-11-20, definition-lead (writer agent). Card `reports/plans/2026-11-20_plan_p1-41-card.md`; filed rules, settings
and expected values in `experiments/gain_only_r/results/predictions.json`; code `experiments/gain_only_r/run.py` and
`score.py`. Due 2026-11-23, one run.*

The user's request, in the Terms' words: on compare-lead's gain-only states (E231: the base model with one trained gain
per MLP neuron and per attention head, exported as a weight change on `down_proj` and `o_proj`), read the reuse score R
and the held model, and file "the held model reads near 1" first. A gain-only state is a pure "how strongly" change. By
the adopted grade, at neurons it keeps every mechanism, so R = 1 there by construction and is not read. At the VPD
featurization and at the coarse levels a per-neuron gain can turn a variable's write, so R there is a real reading. The
question is whether R reads a pure strengthening as kept, above the full fine-tune of the same task and seed.

## What runs

- **Items.** The 81 KL-matched states of E231's hand-off (`experiments/gain_only/results/handoff.json`, "kl:<ref>"):
  the nine built pairs' 19 references and the four twins' 8 tasks, seeds 0-2. e54/cipher has two states per seed,
  one matched to each of its two references (e54/cipher, e54/cipher_own); both are read, each named by its reference.
- **D.** The windows E221 and E226 read for the same task: 512 read windows, last position. These are not E231's own
  read rows.
- **Readers.** Each is a finished experiment's code and filed settings, unchanged:
  - VPD: E226's guards and search, eval cap 2,000, main search only, as E226 read E187's states.
  - Coarse: E221's `read_item` at head_mlp (3 orders), mlp_attn and block.
  - Held: E197's `read` with E205's settings (16,384 fit windows). Readings 1, 2 and 3; reading 1 is the ruled one.
- **Stage 0**, alone on every pod, exits on a miss:
  - Do-nothing (the control state times 0): VPD every s within 1e-6 of 1; coarse R = 1 and f = 0 at every level; held
    KL 0.
  - The exact fp32 gain build of the control (e58/cipher seed 0): held readings 1 and 3 reproduce F within logit_tol
    (1e-3).
  - The control's bf16 state: 1 − S is logged as the export's floor.
- **Beside each item** (`score.py`, no GPU): the full fine-tune's R for the same reference and seed. That is E226 at VPD
  (49 pairs), E221 per level (57: the nine pairs and the case and repeat twins at seeds 0-1), and E205's held readings
  1 and 3.

## Expected values and filed predictions

| number | do-nothing | control | gain-only states |
|---|---|---|---|
| held reading 1 S | null (KL 0) | exact build 1 within logit_tol; bf16 1 − S < 1e-3 | **H1: S ≥ 0.99 on all 81** |
| R at VPD | 1 (every s = 1) | reproduction: E226's 0.998905 ± 1e-5 | R1: d = R(gain) − R(full) |
| R per coarse level | 1, f = 0 | reproduction: E221's per level ± 1e-5 | R2: d per level |

- **H1** is filed first and is the held model's positive control on gain states: reading 1 contains every gain-only
  state exactly.
- **R1 (card)**: a rival a call when d > 0 on at least 80% of the paired items (40 of 49), rival b when on at most half
  (24 of 49), else no call. **R2**: the same rule at each coarse level.
- Do-nothing would also earn a rival a call, so d is read beside KL(F‖M) on D and f.
- The full fine-tunes read R 0.9970 to 0.9997 at VPD (E226), with a seed sd of 9e-6 (median).

## Cost and time (measured per item × count)

- **Per item on an H100, about 439 s:**
  - VPD 293 s: the mean of E226's 68 E187-state items, all at the 2,000 cap (`kstar_vpd/results/items/*.json` "seconds").
  - Coarse about 142 s. Head_mlp is 80 s, the mean over E221's 80 items of the three orders' seconds. Mlp_attn (44 s),
    block (3 s) and set-up (15 s) are E221's plan estimates from E214's H100 logs; E221 saved no time for those levels.
  - Held 3.5 s: the mean of E205's 108 items.
  - Windows are built once per task group; that time is unmeasured, and the rehearsal measures it.
- **Total:** 81 × 439 s = 35,559 s, about 9.9 H100-hours.
- **Per pod, about 1.5 h on nine H100 pods:**
  - 9 items × 439 s = 66 min.
  - Stage 0 about 6 min: one coarse do-nothing (about 142 s), a VPD grade pass, three held reads.
  - Set-up about 20 min. That includes syncing the shard's states: 4.59 GB in all ÷ 10 MB/s = 459 s, about 51 s per pod
    when split nine ways. It also includes the control's state and gains file and the shard's windows files.
- **Cost:** 9 pods × 1.5 h × about $3/h ≈ **$41**.
- This is under the gated lines ($60, 6 h) and inside the session's $300. With six pods it would be about 2.1 h and $38;
  nine is chosen for speed (user, 2026-11-11).

**Where.** At 16:55 `sinfo -p compute` showed the cluster up: one node, l40-worker, 8 L40s, 2 of 252 CPUs allocated, and
64 jobs pending in the queue. The scored run is about 10 GPU-hours, over the 30-minute line, so it goes to RunPod H100
pods per `.claude/CLAUDE.md` Compute.

**Rehearsal ($0, before any pod).** `E236 rehearse:` on the cluster's debug QoS, at full read sizes. It uses chained
jobs of up to 30 min; every reader resumes from its own file.
1. `--tag rehearse --out rehearse --reproduce --items e58/cipher_s0 case_label/twin_s0`. This runs stage 0, then
   reproduces the full fine-tune e58/cipher_s0 against E226, E221 and E205.
2. The same command resubmitted until done.
3. Once more: every item must skip.

It measures seconds per item per reader, the windows build, peak GPU memory, the reproduction differences and the export
floor. If debug cannot start within 5 min, the rehearsal runs on a pod instead: `commute-E236-rehearse-gr`, an H100,
about 45 min, about $2.3.

## Design choices that change a reading
- D is the full fine-tunes' windows, not E231's read rows. The KL match was made on E231's rows, so KL(F‖M) on D can
  differ between the gain-only state and the full fine-tune; it is reported beside each d.
- The case_label and fresh twins are read on E196's case and repeat windows (the same `task.pt`, asserted). The E58 twin
  tables come from E231's spec, trained-keys file only, as E221's `e187_task` builds them. These four E58 twins have
  no full reading: they are reported only.
- R at head_mlp is the mean over E221's three orders.
- The VPD do-nothing is not searched (R = 1 for any K\* when every s = 1).
- The exact-gain check reads held readings 1 and 3 only, because reading 2 treats a whole MLP as one unit.
- No shared function changes. `run.py` sets `KP.EXPERIMENT` so that E221's `read_item` stamps E236 on its records.

## Not run
The gain-matched states (47 of 81 hit the step cap); the neuron level; any verdict or label.
