# Review: independent check of E62 Amendment 4's control stop

*Independent checker; did not write or run this work. No GPU job. Under check: `reports/analyses/2026-11-02_analysis_e62-amend4-control-stop.md` ("the report"), its notebook section (`notebooks/E62_induction_cue.ipynb`, cells 42-44) and its figure, against `experiments/induction_cue/results/amend4.json`, `results/predictions_amend4.json`, `experiments/induction_cue/amend4.py` and `src/commute/copying.py`.*

## Verdict

**The filed verdict stands and every stated number is right: `control_failed_mask`, status `mask_drops_head`, masked fall 0.916 against a bar of 1.204.** I found no bug in stage 1: position, target, sign, gate source, head-off path, precision and batches all do what the report says. The report's central restraint is also right. No fine-tuned state was read, the claim and the null are unscored, and no cause is chosen.

**Four readings need correcting; none reverses the result.**

1. "The head carries a large part of W₀'s copying" at the second A is stronger than the read. The fall is on intact windows only, with no head-off pair-broken read and no reference head at this position (F1).
2. The accounts table leaves out evidence already on file: with every gate at 1, the masked forward loses the head's effect as the target model does (`amend2.json`). It also leaves out three accounts (F2, F3).
3. "Responds to the pair far more than to the head" compares a read that moves the gates with one that freezes them (F4).
4. "A decision is pending" is out of date since 7f1d968 (F7).

## My reading, reached before reading the report's accounts

The run may conclude four things, all about W₀ at the second A on these 200 windows:

- Replacing (2,4)'s output lowers the target model's log-odds of B by 4.82.
- Breaking the pair lowers them by 6.84.
- Under Γ's mask, frozen, the same replacement lowers them by 0.92.
- Breaking the pair, with its own mask, lowers them by 4.34.

The filed bar on the third is missed by 0.29, with no standard deviation.

It may not conclude:

- anything about a fine-tune;
- that the masks "drop W₀'s copying". That is the plan's and the draft's branch wording, and the draft's further "no mask test can read shared mechanism on this model" does not follow. A lean of 4.34 remains, of unknown make-up;
- that (2,4)'s fall is copying rather than a general effect of the replacement;
- which of mask, window, downstream path or replacement value makes the fall small.

The miss against 0.25 is the less informative contrast. 0.25 is an unsourced bar, and the size of the shortfall from 1 is what needs explaining.

## Findings, ranked

### F1. Medium. The fall is not shown to be copying at this position.

Report lines 9, 43 ("the head carries a large part of the target model's copying here"), 90 ("depends heavily on this head"), 120 (scorecard "the head carries it"); notebook cells 42 and 44.

- `stage1` (`amend4.py:66`) reads the head off on intact windows only. So 4.82 mixes what the head does for the pair with what mean ablation at every position does to B's log-odds anyway.
- The copying-specific quantity is the fall on intact windows minus the fall on pair-broken windows, and it was not read. Neither was a reference head: Amendments 2 and 3 used (3,5), this stage did not.
- "0.70 of the lean" divides two log-odds differences. It does not follow that 70% of the copying goes through this head.
- The filed bar is only "fall at least 1.1628", so the status holds as filed. Its name says more than it tests, and the report repeats the name as a finding.
- Evidence elsewhere points the report's way, which is why this is a wording fix and not a reversal. At the marker, the head off takes the target's lean from 1.580 to 0.507, so 1.073 of the 1.163 fall depends on the pair (`amend2.json`, `row4_w0_fall`). On repeated random tokens, top-1 goes 0.825 → 0.0095 under (2,4) and only to 0.707 under (3,5) (`amend2.json`, `row1_control`).

**Correction.** Line 43: "The target model's log-odds of B fall by 4.82 with the head off, against 6.84 for breaking the pair. The head-off read on pair-broken windows and a reference head were not run here, so how much of the fall is copying is unread at this position; at the marker almost all of it depended on the pair." Soften lines 9, 90 and 120 to match. Add both reads to "Blind to" (line 98) and "Not run" (lines 141-149).

### F2. Medium. A filed positive control is left out, and it narrows the accounts.

Report lines 80-84, 97.

`amend2.json` `row1_control` runs the head off through `AllGatesOn`, the same forward with every gate at 1, Δ out, bf16. Induction top-1 reads 0.8177 → 0.0095, and the target model reads 0.8247 → 0.0095.

So on repeated random tokens:

- the hook reaches the components;
- the components without Γ's mask lose the head's effect as the target does;
- bf16 does not remove it.

That is a different metric and different windows, so it does not settle the second A. It does bear on three things:

- Line 97's "Nothing shows the masked fall reading high on a head known to carry behaviour through that forward" is true only of Γ's mask, and should say so.
- Accounts 4 and 5 need the mask or the window to act, since the replacement and the precision alone did not stop the fall at gates 1.
- The cleanest separating read is the masked forward with every gate at 1 on these windows at the second A. The report does not list it in the table or under "Not run", yet it is the first stage of the Amendment 5 the user chose (`decisions.md`, 7f1d968).

**Correction.** Add a row: "The loss is in the components without Δ, or on these windows, rather than in the mask | at gates 1 on repeated random tokens the head's effect is kept (`amend2.json` `row1_control`) | the fall at the second A with every gate at 1". Rewrite line 97 accordingly. Add the read to "Not run".

### F3. Medium. Three accounts are omitted.

Report line 76 ("Nothing in this job separates these") presents the table as the accounts. These belong beside the five, each as an unread possibility:

- **Γ on 41-token windows.** Γ was trained on 512-token rows, and "the absolute levels on windows and Γ's behaviour in full context are untested" (`reports/reviews/2026-10-27_review_gamma-bidirectional-windows.md`; `src/commute/vpd.py`, `compute_ci` docstring). The masked levels sit 2.59 from the target here and 2.01 at the marker. Separating read: the same windows' pairs set inside longer rows.
- **A low-probability behaviour.** B's geometric-mean probability is 0.0079, so "W₀ copies" here means a large rise to under 1%, not a top prediction. Reconstruction KL, the check named in `GOALS.md`, puts little weight on a token at that probability, so nothing obliges the mask to hold its log-odds or its response to a head. Separating read: per-window falls against P(B), and the same control on windows where B is predicted.
- **Downstream of the head.** The slot may pass while later matrices' frozen gates shut what reads it. The weighted `o_proj` read in accounts 1-2 cannot see this. Separating read: the replaced head's change in output patched into the masked forward at later blocks, or the fall with only later blocks' gates set to 1.

### F4. Medium-low. The lean and the fall are not like for like.

Report lines 9, 45 (bold heading), 54-55, 76, 81, 90; notebook cells 42 and 44.

The fall holds gates fixed (`at.masked(w, g0, H)`); the masked lean changes tokens and gates together (`at.masked(b, gb, [])`). Line 45 says this, but line 9, the heading and line 90 ("responds to the pair far more than to the head's replacement") do not carry it.

- If stage 2's unread token part were small, the masked forward would not be "keeping" the lean as computation at all: Γ would be writing it into the gates.
- Three derived numbers inherit the mismatch: "3.06 if shrunk evenly" (line 76), the 0.211 row (line 56) and account 2's 3.43 (line 81), whose lower point is under another mask.
- The target-model comparisons at those lines have no such issue, so the mismatch is on the masked side alone.

**Correction.** Line 90: "With the mask free to change, breaking the pair moves its masked forward far more than replacing the head does with the mask held; how much of that is the mask itself is unread." Line 9: "the masked forward, each set of tokens under its own mask, shows much of the pair's lean." Add that condition to line 76 and account 2's evidence.

### F5. Low-medium. The round level is dismissed on a weak ground.

Report line 109.

- "No filed bar reads a level" is true in letter, but `mask_on` is one of the two terms in both filed masked differences (`amend4.py:68`), so a fault in that value would move the verdict.
- The rarity is not stated. The mean times 200 is −1460.0000057, within 12 float32 ticks of an integer. A mean landing within 3e-8 of a one-decimal value has a chance near 6e-7 per number. Post-hoc selection across many printed numbers softens that.
- I read the path (`masked_lo_windows`, `identity.log_odds`, `json.dumps` of the raw float): no rounding or clamp step exists, and the other five levels look ordinary. I found no mechanism, so chance remains the likely account, but it is untested.
- Under the user's 2026-11-02 rule, an unexplained apparatus number is run down when the check is $0.

**Correction.** Replace the last sentences with: "No mechanism was found in the code. This level enters both masked differences, so the next GPU job re-reads the three masked levels here per window, which must reproduce −7.30000003 on an L40 and gives the standard deviation line 96 lacks." This costs seconds inside Amendment 5.

### F6. Low. Account 5 could state what is known.

Report line 84.

"Its size here is unread" is correct. Nearest measured sizes are in `reports/analyses/2026-11-02_analysis_e62-amend2-masked-head.md` line 209 (target model, bf16 path against fp32 path: 0.004 at W₀, up to 0.012 on a difference) and E38 Amendment 3 (two bf16 paths differing by up to 0.0195). Both are small against a 0.29 miss and far smaller than the 3.90 shortfall, though neither is this masked read. State them, and keep the fp32 read in "Not run".

### F7. Low. "What next" is out of date.

Report line 113 and notebook cell 44 say a decision is pending and name the weighted slot read as next. Commit 7f1d968 records the user's choice: Amendment 5, gates at 1 first, then all 24 heads off in both forwards and swapped-gate forwards, with the slot read second if needed. Point both at `decisions.md`. Lines 3 and 132 and the notebook's "independent check has not been done" should name this review.

### F8. Low. The figure's masked panel title.

`experiments/induction_cue/plot_amend4.py:51`: "the original model's own mask, frozen". The third point is under the pair-broken windows' own mask, not frozen, which is F4's mismatch again. Suggested title: "Masked forward: W₀'s mask, held with the head off; pair broken under its own mask". The numbers and dashed levels (−8.50, −9.01) are right.

### F9. Low. Two statements of source and reason.

- Line 27 gives a reason for the control ("stage 4's reading rests on ... this head") that is in no filed rule, and stage 4 never switches the head off. By the code review (`reports/reviews/2026-11-02_review_e62-amend4-code.md`, line 15), the masked fall can pass "with `S_old` unrelated to copying". So the fall says little about `S_old` either way; the filed test of `S_old` is stage 2. "Could not be read as filed" holds only because the verdict order says so. It should read as that, not as a shown dependence. Stages 2-4 are unread, not shown to be unreadable.
- Lines 33, 99, 139: the quarter on the fall is the draft's; the quarter on the lean is re-check R2's, replacing the draft's level bar (plan, departure 9). The filed `masked_source` also says "the draft's quarter ... on both", so this is a shorthand, and no rule changes.

## Checked and found right

**Numbers.** Every figure in the report's tables and prose, recomputed from `amend4.json` with `jq` or by hand and compared with the log and `derived.json`:

| quantity | value |
|---|---|
| levels | −4.710, −9.527, −11.545, −7.300, −8.216, −11.641 |
| gaps | −2.590, +1.311, −0.096 |
| falls and leans | 4.817, 6.836, 0.916, 4.341 |
| ratios | 0.190, 0.635, 0.705, 0.211 |
| left over | 2.018, 3.425 |
| bars | 1.204, 1.709, miss 0.288 |
| derived | 3.06; fall 4.14 times its bar; P(B) 25.0 times the marker's |
| set | k 172, 2,231 positive, mass 40.14, top 0.41, 243, 114, cosine 0.447 |

Marker column: 1.163, 0.039, 0.033 from `amend3.json`; 1.580, 0.366, 0.231 from `amend2.json`, whose masked lean also uses the broken windows' own gates (`amend2.py:143-144`), so the side-by-side table is like for like. Accounts' 0.04, 0.39, 0.21 and the 5.4% are right (`amend3.json`; `reports/reviews/2026-11-02_review_e62-amend3-check.md`, F2).

**The filed rule, applied by hand.** 0.00789 ≥ 0.00315; 4.817 ≥ 1.1628; 6.836 ≥ 1.5801; 4.341 ≥ 1.709; 0.916 < 1.204: `mask_drops_head`, then `mask_test` returns `control_failed_mask`. k 172 ≤ 4,940 and N 9,881 equals the filed count. The JSON's `verdict` agrees.

**Stage 1 code** (`amend4.py:60-81`, `amend2.py:Reads`, `local_faith.py`, `steering.py`, `copying.py`):

- *Position.* `at = replace(R, half=half - 1)` is used for all six reads and `p_b`; `gates` ignores `half`; `second_a_holds` and `pair_broken_holds` ran before any forward.
- *Target token.* `LAST`, the window's last column, not fed to any forward (`:n`, n = 41); B is asserted by `second_a_holds`; the broken windows carry the same column.
- *Sign.* Intact minus head off and intact minus broken, in both forwards.
- *Gate source.* `g0` and `gb` are read outside `ablated`, from each set's own tokens, with no head off in Γ's pass.
- *Head-off path.* The same `(2,4)` and `means`, every position, in both forwards. PyTorch passes a pre-hook's edit into the forward hook, and `param_decomp`'s `_components_and_cache_hook` feeds `args[0]` to the components. The masked fall is non-zero on the real model, and `tests/test_copying.py` covers the path on a tiny model.
- *Autocast and batches.* All three target reads fp32 and all three masked reads bf16, so each difference is within one precision and only the ratios cross them, as filed. Batches are 64 in window order, `zip(..., strict=True)` pairing gates with windows. Δ is out. The target read is `target_model` directly, 41 tokens, the same as the masked read.

**Provenance.** No change to `amend4.py`, `copying.py`, `local_faith.py`, `vpd.py`, `steering.py`, `amend2.py` or `amend3.py` since `897c026`. `predictions_amend4.json` is unchanged since then and its sha256 matches the results. `cluster/env.sh` puts this clone first on `PYTHONPATH`, so the shared venv's `commute.pth`, which names `commute-lead`, did not supply the code. Job start 16:06:01 (`jobs.log`), stage 1 at 16:06:35, exit 1. `amend4.out` exists and `amend4_vectors.pt` does not.

**Other.**

- `tests/test_copying.py` passes (`make test`, 4,018 cases). `check.py` on the report: 0 errors.
- The figure is drawn from the JSON and matches it.
- The notebook's numbers match the report.
- The side-by-side table's "two positions, not a trend" is right.
- Declining the plan's "masks drop W₀'s copying" (line 90) is right.
- The "Not run" list holds for what it names, apart from the additions in F1 and F2. No stage after 1 left any number in the JSON.

## What I could not check

- Per-window values, and so any standard deviation on the fall, the lean or the 0.29 miss. Not saved, as the report says. Whether 0.19 and 0.25 are separated by more than noise is unknown; 200 windows share 32 keys.
- A repeat of the job: no GPU read here, so the numbers are recomputed, not reproduced. Stage 3's determinism check did not run, so this job has no internal re-read.
- `windows.pt` and `heads.pt` were not loaded. The window properties rest on the run's own assertions and the plan's CPU check.
- The size of the bf16 effect on this masked read, and all accounts in the table.
- My F3 accounts are possibilities with no measurement behind them, like the report's.
