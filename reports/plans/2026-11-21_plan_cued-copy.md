# Plan: the cued copy, a low-prior copy built as its own task (cell P2.54, run 2; E_TBD)

**In plain words.** The 2x2 of base prior by mechanism had a weak corner: its low-prior copy was the key-copy task on
the rows where the base model found the answer least likely. That made it share key copy's update, so every reading
of the update landed near copy. This run builds that corner as a task of its own. It is E62's copy map, "A B ... A M"
→ B, with B drawn from rarer words, so the base gives the answer about 10^-5.9, the cipher's prior. Then it reads every
proxy on it.

**Restated in the Terms' words.** Mechanism label = the share of a fine-tune's gain over M that is lost when M's copy
heads are mean-ablated, net of M's own fall (E62's measure): (2,4) alone (run 1), the top 3 heads, and the (2,4)+(1,1)
pair (khead). This is a copying label, not the reuse definition under revision in P1.1, and no reuse verdict is given.

## Design check (/design)
- **Goal served:** claim 2 of the paper (the cheap proxies against the mechanistic definition): does the re-gating
  share follow the mechanism when the prior is held low?
- **Nearest tried:** E250 (row-selected low copy, `reports/analyses/2026-11-21_analysis_prior-mech-2x2.md`): every
  reading split by mechanism, but by construction. E62 (copy map, prior 10^-3.50, (2,4) share 0.47-0.56). E214 found
  that a hand-built new-cue edit never reached the copy bar, so this copy is trained. New here: a separately trained
  copy task at the cipher's prior.
- **Patch or step:** a patch on E250's construction. It answers the caveat the analysis names.
- **Decisions relied on:** E62's ablation and labels (0.2 / 0.5), E237's common size and random construction,
  E250's selection rule and table. No departure.
- **Adopted definition unchanged:** yes; the head label stays a copying label.
- **What changes what we do:** if r on the cued copy stays near copy's (~0.7) while the prior sits at the cipher's,
  r follows the mechanism on a task that is not key copy. If r drops to near the cipher's (~0.3), r follows the prior
  and claim 2's wording must change. If steering or rank then split from r, that is the separation E250 could not show.
- **Control first:** run 1's control (M's second-copy top-1 0.8247 falls under 0.3 with (2,4) off and holds over 0.7
  with random heads off; the zero update reads 0), then E250's repeats (E133's copy prior, E147's shares on
  e58/copy_s0). All run alone before any construction, and each exits 1 on a miss. Then the zero update on the new
  rows must read 0 exactly.
- **Saved for audit:** the kept windows (`results/large/windows.pt`, with targets, B pools and B positions), the three
  updates in bf16, one JSON per item, each band's selection counts in `results/a.json`.

## Construction (Fable's advice, 2026-11-21, acted on)
- **Form.** E62's induction form, not key copy behind a new marker. E62's form can only be solved by copying, since B
  varies per window and the train and read B pools are disjoint. E62 lost 0.47-0.56 to (2,4), so it has room over the
  0.2 bar, where key copy sits at 0.25. A key copy behind a new marker would be key copy again with other tokens.
  Fable advised against a cue that defeats prefix matching: it invites a solution that does not go through (2,4).
- **Lowering the prior.** In E62's form, M's prior is 10^-3.50 intact, 10^-4.18 with the pair broken and 10^-5.32
  with B absent (`experiments/induction_cue/results/run.json` stage_b). So the B tokens themselves must be rarer. The
  keys are 32 new draws from E62's ranks 1000-3000. B comes from three filed bands of frequency rank, tried least rare
  first: [3000, 10000), [10000, 18000) and [18000, 25000). On each split, the longest lowest-prior run of windows
  whose geometric-mean P(B) is at or under the cipher's 10^-5.86 is kept (E250's `lowest_run`), capped at
  1600 train and 400 read rows. This selection runs inside the new task only. The first band to clear the check is
  used. Fable advised against picking B per window by M's probability, because that confounds the token with its
  context.
- **Per-key steering** is keyed on A, the token before the read position: 32 keys, all of them in both splits. The
  read token is always M, and keying on B would need shared B pools. A read row is kept only if its A has at least 8
  kept train windows, so the reader that killed E250's first attempt is covered by construction. On A, per-key
  steering cannot hand over the answer, so it is not comparable to the kv tasks' ~1.0.
- **CPU check (real tokenised windows, no model;** `~/commute-data/scratch/compare-lead/cued-copy/construction_check.py`):
  - The ranking holds 26,994 word-like intact tokens. The token at rank 25,000 occurs 14 times in E12's 32,768 Pile
    rows, so no band holds an untrained token.
  - Every band builds 32,000 train and 8,000 read windows on all 32 keys, with at least 227 read windows per key and
    disjoint B pools.
  - With a stand-in prior (normal per window, mean 10^-4 or 10^-4.6, sd 1.1 decades), the selection reaches the caps
    on all 32 keys at log10 -5.86.
  - Edge cases: an empty pool gives a geometric mean of nan and fails the check; ties thin evenly; the effects tie
    reads no_call.
  - Not checked on CPU: M's real prior per band. That needs the model, and the job measures it before any training.

## Recipe and readers (all by import)
- Fine-tune: `keydep.finetune` (E62's call, data.LAST target, 300 steps), with the update read at step 30 and E12's
  rule read at step 100 (P ≥ 0.5, exit 1 on a miss). c is E237's target size over the three seeds' mean size. Each
  seed gets a Gaussian random update at the same per-matrix norms, times c (`finetune.random_direction`).
- Readers come from `prior_mech_2x2/`:
  - run 1's `read_state` ((2,4) and five random heads);
  - khead's `Heads` and `read_item` (top-k at k = 1, 2, 3, 5, the k-matched null, and the pair);
  - `train.py`'s `World`, `weights` (r, size, erank), `shares` (holes_fill's neuron and steering shares) and
    `effects`.
- The key corners and the side column are E250's saved readings (`train.state_row`). The k-head columns come from
  khead's files and read nan until run 3 (E_TBD3) has run.
- Added numbers:
  - the gain on B-absent read windows (E62's `absent_b`) over the intact gain, as the copy's own control;
  - the side column (row-selected low copy).

## Filed expectations (`experiments/cued_copy/results/predictions.json`)

| Reading | Expected |
|---|---|
| Construction | A band passes with a read prior of log10 -5.86 to -5.6, at the caps of 1600 / 400 rows. |
| Learning | P(B) at step 100 over 0.9. |
| (2,4) share (Q0a) | Holds at about 0.5, at least 0.2 filed. |
| k = 3 share (Q0b) | About 0.8, at least 0.4 filed. |
| Stop | Under 0.2 at k = 3 means the cued copy is not labelled reused, and nothing is called. |
| r (Q1) | About 0.7 if it follows mechanism (mechanism effect ~0.4 against a prior effect ~0.1). About 0.3 if it follows prior (prior effect ~0.25 against a mechanism effect ~0.2). |
| Prior (Q2) | Follows prior. |
| Neurons (Q4) | No call: E62's copy map read 0.044 at k64. |
| Effective rank (Q5) | Near 475 or near 405; no direction filed. |
| Random updates | Gain about 0, effective rank about 700. |
| B-absent gain over G | Under 0.1. |

Every call uses E250's rule: the larger of the two effects, if it is over the band (the largest seed range).

## Time, cost, route
- **Time:** about 7 min on a cluster debug L40, $0, not gated. The basis is E250's measured log
  (`experiments/prior_mech_2x2/results/run2.log`):
  - load and repeats: 26 s;
  - a selection forward over 20,000 windows: 7 s, so 3 bands of 40,000 take at most ~42 s;
  - fine-tunes: 11 s each, 33 s for three;
  - split bases: 13 s;
  - holes_fill and r reads on a new item: ~30 s each, 180 s for six;
  - run 1's head read: 1 s per 7 head sets, so khead's 84 sets ~12 s per item, ~84 s for six with M's rows;
  - control: ~20 s.
- **Total:** ~400 s. E250 itself took 7 min wall for 3 fine-tunes and 21 items.
- **Route:** `cluster/cued_copy.sbatch` (`--time=00:30:00`).
- **Needs:** runs 1 and 2 (E249, E250) finished, which they are. Run 3 (khead) is optional.

## Not run
- The (2,4) label on the windows the selection dropped (Fable's guard against "selection changes the task"). It is
  small to add if the label comes out low.
- The cosine of this update with E62's copy update (Fable: disclose if high).
- A second construction if no band passes. The stop goes to the lead.
