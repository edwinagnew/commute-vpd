# Appendix: design checks for the five proposed cells of "What does the re-gating share measure"

*2026-11-09, one line or two per question of `/design`. Goal for all five: `GOALS.md`, the mechanism hypotheses
("strengthen, reveal, create, brittle") and the plan's part 2, what re-gating measures and how easily it could be
outperformed. None gives a verdict under either wording of the definition, which is under revision in cell P1.1;
each reports readings against the labels each pair was built with. None proposes anything on Qwen3-1.7B. All use
saved files, `ls`-confirmed on 2026-11-09, at `~/commute-data/results-large/` (the `results/large` symlink target).*

Notation used in the expected values: `share` is `route_split`'s `r` (`src/commute/local_faith.py:270`). W0 is the
unmodified model. A "zero update" has δW = 0. A "do-nothing" update keeps the state's weights and leaves subcomponents
and Γ untouched; it reads the same share as a split (`experiments/fresh_reuse_pair/results/stage2.json`,
`.states.real_s0.nothing.shares.r` equals `.least_squares.shares.r`).

## Cell 1: the prior against the share over every saved map

- **Goal.** Whether the share carries information beyond how likely W0 already was to give the answer. **Nearest row:** `experiments/TRIED.md` rows E58 Amendment 6 and 7 (six and eight maps, Spearman 0.886 and about 0.95, the prior cut); this widens them to all saved maps. **Step, not patch:** it is the rival comparison the plan's part 2 asks for.
- **Decisions relied on.** `decisions.md` rows 112 and 184 (the label is mechanistic and under revision); the prior is a reference, never a label.
- **Reads under the adopted definition.** Does not apply: no verdict; it reports two orderings. The built labels are only quoted.
- **Result that changes what we do.** Agreement near 0.9 over about 18 maps: the paper reads the share as the prior except on the copy rows. Much weaker: a claim that the share carries more becomes testable.
- **Control, first and alone.** The W0 forward on the copy map's windows must reproduce the saved prior 5.91e-5 within 1% (`experiments/open_pair/results/amend6.json` → `.prior_beside_r.copy.w0_geo_p_trained`). Stop if not.
- **Numbers and expected values.** Spearman of the map-mean share against the log geometric-mean prior: 0.8 to 1.0 expected on keyed maps (the 0.886 reference); on the label pairs the prior is a per-window column, so expect the bottom of the range. The share of a zero update is undefined; do-nothing reads the share; control reads 5.91e-5. Geometric-mean prior per map at W0, the only number with a W0 value: the method table's values where it has them (e.g. 1.33e-3 elicit, 1.38e-6 cipher).
- **Files.** Windows: `regating_table/windows_{ident32,ident32b,rand32,rand32b}.pt`, `keydependent_elicit/windows_{elicit,cipher}.pt`, `open_pair/{windows_*_new,amend3/windows_*,amend6/windows_*,amend7/windows_*}.pt`, `induction_cue/windows.pt`, `case_label/task.pt`, `fresh_reuse_pair/task.pt`. Share values: the JSONs on the page, no new share read. Overlaps cell P2.2's prior column: this fills what P2.2 leaves "not run".

## Cell 2: steering and the neuron version on real against twin

- **Goal.** The tentative definition's contrast is real against twin; the share reads them alike on three pairs. **Nearest row:** E98 and E117 (neuron version), E123 and E125 (the pairs), E54 and E60 (steering); steering was never read on the label pairs. **Step.**
- **Decisions relied on.** `decisions.md` row 184. No departure.
- **Reads under the adopted definition.** Does not apply; labels as built.
- **Result that changes what we do.** A reading that separates real from twin narrows the paper's claim against the share. Neither separating says no activation-level reading in hand sees this contrast.
- **Control, first and alone.** Zero update reads steering 0 and neuron cells exactly 0, as in E98's stage 0; early-stop on any nonzero. Reproduce E98's `r_half` on `e46_copy_random` first, on the same code path (`experiments/neuron_r/results/score.json` → `.rows.P1.rows.e46_copy_random`).
- **Numbers and expected values.** Per state: steering share, `r_half`, `r_n`, at the gain each state has. At W0 and a zero update: steering undefined and cells 0. Against E125's and E123's real and twin reading of the share (0.11 to 0.24): no guess from a past run for steering on these tasks, so the expected range is the full 0 to 1, stated as unknown; `r_n` expected 0.4 to 0.8 by analogy with E98's rows. Real-versus-twin gap: expected at most 0.1 for each reading, the band of the share's own reading at 0.03 to 0.11.
- **Confound stated, not removed.** Twin update size is 1.6 to 3 times the real's; a size-matched twin needs a new fine-tune and is not proposed.
- **Files.** `case_label/dW_{real,twin}_s{0,1}.pt`, `case_label/task.pt`, `fresh_reuse_pair/dW_{real,twin}_s{0,1}.pt`, `fresh_reuse_pair/task.pt`.

## Cell 3: steering on the new-marker copy and lookup

- **Goal.** The only pair with a causal-check label, where the share reads 1.09-1.10 against 0.38-0.42 but is filed as unscored. **Nearest row:** E62 and its amendments; steering was not read there (`experiments/method_table/results/table.json` → row `e62_copy_lookup`, `cells.steering`). **Step:** the comparison on the one causal row.
- **Decisions relied on.** `decisions.md` row 112 and E62's "`r_scored` false": it stays unscored; this reports a number beside it.
- **Reads under the adopted definition.** Does not apply; labels as filed, the copy side not settled.
- **Result that changes what we do.** Steering separating as widely as a gap of about 0.7 removes the share's only standing edge on a causal row.
- **Control, first and alone.** Zero update reads steering 0; the steering read of the copy map on stored path matches E62's `S` code path by `sha256` of its inputs. The first state is read alone and the job stops on any error.
- **Numbers and expected values.** Steering share on six states, log-odds unit: copy 0.7 to 0.9 expected (E58 copy read 0.82 to 0.88), lookup 0.1 to 0.3 (E58 lookup read 0.15 to 0.17). W0 and a zero update read 0. Control reads 0.
- **Files.** `induction_cue/{copy,lookup}_s{0,1,2}.pt`, `induction_cue/windows.pt`.

## Cell 4: how far masks alone can push the answer

- **Goal.** Whether the share reads what the original subcomponents can write, a reading of Γ and W0, or something the update did. **Nearest row:** E18 (which read fixed patterns, not the maximum reachable) and the filed reading of E62 Amendment 2 (`experiments/TRIED.md`, row E62 Amendment 2) that the share says "the original subcomponents can write this target under some mask", never measured. **Step.**
- **Decisions relied on.** `decisions.md` row 112. Uses the existing mask adversary from `src/commute/local_faith.py`/`faithfulness.py` with the objective reversed; that is a new read of existing code, an apparatus to be reviewed. Flagged as the one cell with new measurement code.
- **Reads under the adopted definition.** Does not apply.
- **Result that changes what we do.** Reach orders maps as the share does, and share is near reach: the paper reports the share as a property of the decomposition at W0, not of the update.
- **Control, first and alone.** Zero adversary steps must read W0's masked log-odds L00 exactly (`route_cells`'s base L). A random answer token at W0 is read as a low-reach reference. Stop on any mismatch at step zero.
- **Numbers and expected values.** Reach per map, in log-odds at the trigger: reach = L* − L00 at W0, where L* is the highest masked log-odds of the trained answer. At W0 with zero steps it reads 0 by construction; the do-nothing reference is the share's numerator L01 − L00 from the saved cells. Expected: reach at least the numerator on every map, since the fine-tune's own masks are one point of the search, and at most about 20 log-odds. Expected ordering: reach and numerator rank alike across maps.
- **Windows only.** No fine-tuned state: the windows of cell 1. Cost of the adversary unmeasured; timed in the first minutes, cap at the 2 h `debug` limit.

## Cell 5: the learned-nothing reading, every-position form

- **Goal.** Whether the page's headline weakness, 0.77 to 0.82 on a fine-tune with gain below zero, is stable and specific. **Nearest row:** E36 and E67. **Patch** on E36's reading, justified because the page leans on it.
- **Decisions relied on.** `decisions.md` line 36, which de-prioritised the fixed-shift hypothesis; none departed from.
- **Reads under the adopted definition.** Does not apply.
- **Result that changes what we do.** Stable near 0.8 for the code-text states and the Gaussian: any small update reads that in this form, so the weakness stands as stated. Wide spread, or the Gaussian far away: "learned nothing still reads high" narrows to this one fine-tune.
- **Control, first and alone.** Re-read of `code_step10` reproduces the saved 0.8156 (`experiments/regating_spread/results/regating_spread.json` → `.states["10"].r`) within 1e-3. Stop if not.
- **Numbers and expected values.** The share of `code_step{10,100}` and one Gaussian of matched norm; per-batch spread over `code_held.pt`'s rows. Control 0.8156 and 0.7656 reproduced. The Gaussian's gain is expected near 0 and may be unreadable: E6's random update read gain −0.015 and a ratio of 1.19. The do-nothing update of a Gaussian reads the same share. W0's own update is zero: undefined.
- **Files.** `regating_table/code_step{10,100}.pt`, `regating_table/code_held.pt`.

## Departures from the literature

None: the share, the neuron version and the steering share are this project's measures. Cell 4's reversed adversary is a departure from VPD's use of its adversary, which maximises reconstruction loss, stated in the cell.

## Not run

Every proposed cell; nothing above has been executed.
