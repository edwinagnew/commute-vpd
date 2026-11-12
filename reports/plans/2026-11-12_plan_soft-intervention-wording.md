# Cell P1.19 (L1.4, 4 runs, due 11-13): the third reuse wording read by interchange interventions

*2026-11-12, definition-lead. Approved by the user as cell P1.19; E-code pending (claimed from the card). Testing only:
the third wording is not adopted, so every reading is labelled "under the third wording, provisional". No change to
`decisions.md`. Zero-run page beside it: `reports/analyses/2026-11-12_analysis_third-wording-no-reading.md` (P1.18).
Apparatus, files, saved inputs and rules: `..._appendix.md`. Filed draft: `experiments/soft_intervention/results/predictions.json`.*

## /design check
- **Goal.** Part 1's working definition of reused against new (`GOALS.md`); the user asked for this test of the third
  wording (task file 2026-11-12 evening). It moves the gate page: it says whether the third wording can read built on
  planted controls, and what it labels the memorised fact, its twin and the old pairs.
- **Nearest tried.** E168 read the same fact and twin with the second wording's searched shares; E137 and E157 built the
  planted edits; E156 found the search's top is not 1. New here: no search over $F$; $H$ and its alignment are fixed on
  $M$ and $F$ is checked by interchange interventions and a least-squares fit in a stated class.
- **Patch or step.** A step: a different definition read on the same items, not a repair of the second wording.
- **Decisions relied on.** Linear reads only (user, 2026-11-12); E168's layout, push and saved fine-tunes; the old-task
  states of E142. Departs from none; the adopted wording and its cut-offs stay as they are.
- **Adopted definition?** No. Every label is "under the third wording, provisional". Map class, interchange bar and
  alignment class are the user's; each is read under every stated option, never chosen here.
- **What changes what we do.** If the planted controls misread, the wording cannot be measured this way and the cell closes.
  If the twin's push follows an interchange on the plain model's parity variable and the fact has no such variable, the
  third wording separates them under the one-variable class; if neither fine-tune has a fixed carrying variable, it gives
  them no reading, and the gate page says so.
- **Control first, alone.** Run 1 is the planted controls and the do-nothing state, nothing else. Stop rule below.

## What is read, per item and site
$M$ is the plain 67M; $F$ an edit or fine-tune. On E168's layout $U$ is the coordinate along the push direction
$u = Qb/|Qb|$ at block 2's input, block 3's input or block 3's output, last position; $P$ is the parity coordinate at block 1's
input, along the class-mean direction of $M$ for odd against even last numbers. Both are fixed on $M$ before any $F$.
- **Gap**: mean total variation between $F$ and $M$ on E150's read windows. Under 0.02: no reading.
- **Carry** $c$: share of the gap closed in $M$ when $U$ takes $F$'s value. 1 means the change passes through $U$ alone.
- **Reversion** $r$: share of the gap closed in $F$ when $U$ takes $M$'s value. 1 means what follows $U$ in $F$ is $M$'s.
- **Fit shares** of $U$'s change $\Delta = U_F - U_M$, centred and read on held-out windows: $f_{own}$ from $U_M$ alone,
  $f_{one}$ adding $P$, $f_{tok}$ adding a table of the last token. 1 means the class explains all of the change.
- **Trigger interchange** $i$: in $F$, swap $P$ from a window of opposite parity; the share of the move in $U$ that the
  fitted $f_{one}$ equation predicts and $F$ makes. 1 means $F$ reads $M$'s own $P$; 0 means it ignores it.

The site is the one with the largest $\min(c, r)$. Grade there: pure re-setting if $\Delta$ is constant or $f_{own}$ passes;
else new trigger under the one-variable class if $f_{one}$ and $i$ pass; else new trigger under the per-token class if
$f_{tok}$ passes; else new computation. No reading if $\min(c,r)$ misses the bar at every site. Bars are read at three
options the user chooses between, 0.8, 0.9 and 0.95, used for $c$, $r$, $i$ and the fits together; an option the planted
floor or the do-nothing state passes is void.

| Number | Original, do-nothing | Planted constant push | Planted trigger | Planted floor |
|---|---|---|---|---|
| gap | 0, no reading | 0.229 (calibrated) | 0.229 (calibrated) | 0.233 (E157) |
| $c$, $r$ | not read | 1 within 1e-3 | 1 within 1e-3 | 1 within 1e-3 |
| $f_{own}$ / $f_{one}$ / $f_{tok}$ | not read | constant: pure re-setting | under the bar / 1 / measured | under the bar on all three |
| $i$ | not read | not applicable | 1 within 1e-3 | not applicable |
| grade | no reading | pure re-setting | new trigger | new computation |

## The four runs, in order (each a debug L40 job, $0)
1. **Planted controls alone.** Job A on E168's layout: the do-nothing state, a planted constant push, a planted trigger that
   reads $P$ at block 1's input and writes $U$ at block 2's input, E157's floor. Job B on E137's layout: Always on, Cue,
   XOR, Table. **Stop**: any control reading other than its constructed label at the 0.9 option, or $c$, $r$, $i$ missing 1
   by more than 1e-3 on an exact edit, exits 1; nothing further runs and the cell closes "no call".
2. **Fact and twin**, 2 seeds each, from saved states, beside E168's planted fact edit. The card's comparison.
3. **Old matched pairs**: copy and cipher at 24 keys, identity and random at 32 keys, case label and its twin, repeat label
   and its twin, 2 seeds a side, at the last layer. The in-context copy and lookup are not read: their answer span exceeds
   every alignment option. Two parallel debug jobs.
4. **Reserved** for a repair of runs 1 to 3, or for a trigger the user names; not spent otherwise.

**Time.** E168's log: a pass of the 67M over 100,000 windows takes about 6 s on an L40, set-up about 1 min. An item is
about 25 passes, so about 3 min. Runs 1 and 2 are each under 20 min; run 3 is 16 states, two jobs of about 25 min. If debug
cannot start within 5 min, an H100 pod, about 1 h and $4 in all. Nothing is trained.

**Card rivals** (`~/commute-data/scratch/definition-lead/v3-eval/card-draft.md`): the twin's fine-tune reads $M$'s parity,
so $i$ is at least 0.5 on both seeds and the twin and fact grade apart under the one-variable class; against both
fine-tunes memorising a table, so the twin's $i$ is under 0.5 and both get the same grade under every class.

**What it is blind to.** Pushes written in a direction other than $u$, unless they reach $u$ by block 3's output; changes
spread over several variables; anything outside E150's windows. Positive control: the planted trigger. Noise floor: two
runs of the 67M differ by about 1e-5 in total variation (E154 stage 0).
