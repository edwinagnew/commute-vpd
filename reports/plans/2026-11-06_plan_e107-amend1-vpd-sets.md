# Plan: E107 Amendment 1, VPD's switched-on set on the wrong-answer states

*2026-11-06, baseline-lead's E107 Amendment 1 writer. Code `experiments/label_vs_style/amend1_vpd.py` (GPU; E94's
`run.finish`, loaded by path) and `amend1_sets.py` (CPU; the sets and the rule); rules
`experiments/label_vs_style/results/predictions_amend1.json`; job `cluster/label_vs_style_amend1.sbatch`. Follows
qwen-decomp-lead's offer (`~/commute-data/agents/baseline-lead/notes-read/note-e94-known-and-sets.md`, sections 2-3) and
Fable's advice `reports/plans/2026-11-06_advice_e94-amend2.md` ("What would change this", first point: run it as
E107's amendment, with E94's code by path). Approved class 1: GPU passes of an existing harness on saved states, $0.*

## The request in the Terms' words

Take E107's two wrong-answer updates δW at the matched step (seeds 42 and 142; |δW| 0.492 and 0.476, the known arm's
0.480 and 0.483). Run each through E94's reads exactly as E94 Amendment 1 ran the known-facts states: the
least-squares split, then the 128 maths rows with the gate counts. From the counts, take the switched-on set S: the
subcomponents on with the update and off at W0. Ask whether the wrong arm's S sits with the known arm's (then VPD's set
reads the rows, topic or style, as E107's neuron set did) or apart from it (then it reads whether the trained answer
agrees with what W0 already holds). This puts VPD and the neuron baseline on the same states.

## Design check (`/design`)

- **Goal.** `GOALS.md`'s fallback question, new capability against strengthened one; this is the VPD column of the
  neuron-against-VPD comparison on E107's states.
- **Nearest tried.** E107 (`TRIED.md`): the neuron carrying set of the wrong arm sits with the known arm, "topic or
  style" on both probes. E94 Amendment 1: on E94's three readings the known arm reads like anchor 2. Set overlap
  (`qwen_d3_rows/results/set_overlap.json`, $0): VPD's S separates known from fictional (Jaccard 0.63-0.66) beyond
  the seed null (0.86 known, 0.89 fictional). New here: the one VPD reading that separates the arms, on a pair that
  differs only in the stated answer.
- **Patch or step.** A control on an existing reading, not a new direction.
- **Decisions.** `decisions.md`: the matched step is a size control only, so neither verdict supports a new-content
  claim. No departure.
- **What changes next.** "familiarity" would make VPD's set the one reading that beats the neuron baseline here and
  would go to an independent check. "topic or style" retires VPD's set identity as a known-against-new reading, as
  E107 retired the neuron one.
- **Apparatus check, first.** E94's stage 0 (W0 through this path: S = 0, K00 = K01 = K0) has passed and is not
  rerun; each state's first minutes check its norm against the filed fp64 norm (1e-6) and K0 on the maths rows
  against W0's (1e-6). A miss stops the job. The known states, read by the same code, are the positive control.
- **Saved.** `qwen_d3_rows/results/<state>/{maths,state}.json`, `results/large/<state>/counts_pe.safetensors`
  (12 MB), as E94's.

## Rows: why this comparison is sharper

Per E107's plan, the wrong arm trains on the known arm's exact 217 TOFU real-fact rows, five wordings and writer;
only the stated answer is swapped (`commute.known_facts.swap`). Known against fictional differs in rows, topic and
content at once. Wrong against known holds rows, prompts, topic and style fixed.

## Numbers and what each should read

| Number | W0 | do-nothing | reference | expected |
|---|---|---|---|---|
| \|S\| per wrong state | 0 (stage 0) | as the state: S does not depend on placement | known 1468-1501 | 1300-1700 |
| w_wrong = J(wrong42, wrong142) | undefined | as the state | seed nulls: known 0.857, fictional 0.886 | 0.82-0.90 |
| four J(wrong, known) | undefined, unread | as the state | known against fictional 0.632-0.660 | 0.76-0.88 |

|A ∩ B| / min(|A|, |B|) is printed beside each Jaccard, unscored. E94's outside share, placement and think overlap
land in `state.json` for free and are reported as description.

## The rule (filed)

E107's `wrong_rule` with Jaccard: band = 1 − min(w_wrong, 0.857); **familiarity** iff max(cross) ≤ 0.660 + band and
≤ 0.857 − band; **topic or style** iff min(cross) ≥ 0.857 − band and min(cross) − 0.660 > band; else no call. At the
known arm's band (0.143): familiarity ≤ 0.714, topic > 0.803. Before either, **no call: size cap** when any wrong
against known pair's min/max |S| ≤ 0.803. Lean: topic or style (E107's neuron reading; the weight cosine of wrong
against known, 0.82, equals the seed cosine).

Edge cases, run through `side` before filing (`amend1_sets.py --edges`, all as named): wrong at the seed null → topic;
at known-against-fictional → familiarity; equidistant (0.758) → no call; on the familiarity line → familiarity
(inclusive); on the topic line → no call (strict); pairs straddling → no call; wrong seeds disagreeing (w_wrong 0.5)
→ no call (the band widens past both lines), or familiarity if cross is as low as 0.2; a wrong set 30% larger → no
call: size cap; W0's empty sets → unread; a missing pair → unread.

## Blind to

Jaccard is capped by min/max |S| (fictional's S is ~25% larger than known's), hence the size-cap guard. On
|A ∩ B| / min the known-against-fictional gap (0.875-0.904 against the seed null 0.933) is under its own band, so
that measure could never call topic; it is printed only. S is read on maths rows and is a gate-identity reading:
blind to how much each subcomponent carries. Weight cosines say the label moved δW about as much as a seed, so the
test has little room; "no call" is a likely outcome and is reported as read, not amended.

## Cost and order

One `debug` L40 job, $0: `sbatch cluster/label_vs_style_amend1.sbatch --states wrong_s42_matched,wrong_s142_matched`.
E94 Amendment 1's matched states took 1627 s and 1501 s (`qwen_d3_rows/results/run.log` "done in"), ~630 s of it the
own-row passes this run skips: ~17 min a state, ~37 min with start-up, so both fit one job. Then
`cpu-run python experiments/label_vs_style/amend1_sets.py` → `results/amend1_sets.json` `.rule.verdict`.

## Not run

E94's direct share and credit on the wrong arm's own rows; a VPD rows null (E105's halves) for S; the full-size wrong
states (none were saved).
