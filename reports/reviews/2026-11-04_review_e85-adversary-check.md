# Independent check of E85: can VPD's adversary fail a bad state on Qwen3-1.7B? (2026-11-04)

The numbers recompute and I found no bug, but the sentence as written is not supported, because "cannot fail a bad state" goes well beyond what two passes show. One of the two states is not bad in any way this test looks at, since the random change barely moves the model and the adversary only compares a state with itself, and no state known to be unfaithful was read, so nothing shows the test can reach its fail bar at all. The other state does read above the unchanged decomposition on both seeds, and over the fail bar at the middle of the search on one read, so the adversary responds to it and the pass comes from which step is scored and from a bar borrowed from another model.

By a fresh Opus agent that wrote none of this. Scripts and outputs: `~/commute-data/scratch/qwen-decomp-lead/check-e85/` (`traces.py`, `sizes.py`, `sizes.json`, each with `.out`). No model was loaded and no forward run. The two scripts read logs, the saved update, the decomposition's columns and the importance statistics, through `cpu-run`.

States: W0, the unchanged decomposition; Rand, W0 plus a Gaussian at the update's per-matrix size, left in Δ; Rare, the fine-tuned weights with the update's least-squares split over the rarely-on half moving those columns and Δ taking the rest; N, do-nothing; LS, the least-squares split; I, in-span; ADJ, the adjoint split; A, E76's clean-term state. r is a state's final KL over W0's at the same seed. N, LS, I, ADJ and A come from E80's amendments, on other pods of the same card model.

## What holds

- **The filed outcome computes.** r: Rand 1.003, 1.015, mean 1.009; Rare 1.157, 1.384, mean 1.270; W0's spread 18.4%. Both means are under 1.40 and no seed reaches 1.61, so the filed label follows (`experiments/qwen_d1_teeth/rules.py:40-41`). The 14 tests in `tests/test_qwen_d1_teeth.py` pass. Plan, rules and code were committed at 14:36 (6cba6504), before the run's first log line at 15:30.
- **The states are built as described.** Rand and Rare are `run.py:43-45` and `run.py:54-58,83-86`, through `qwen_maps.site_change` (`qwen_maps.py:239-240`) and `applied` (`:244-278`). Masks, clean logits and Γ's inputs are recomputed from the state on every row and pass (`qwen_teacher.py:297-299`). Sources depend only on the seed (`qwen_vpd.py:401-403`). I recomputed the half from the statistics files: active shares 0.000652 and 0.017890, as logged. On 21 sampled matrices I rebuilt the split. The second-order term is negligible, and the share left in Δ follows the formula at each matrix.
- **W0 reproduced.** Seed 0 equals Amendment 1's to every digit, with a different run time and a full step trace in the log, so it is a rerun and not a copy. Seed 1 differs by 0.04%, starting at step 9.

## Findings

### (a) Is the adversary strong enough, and aimed at what makes these states bad?

**1. High: no positive control for failure.** The only control is W0, which shows the harness repeats. No state in `experiments/qwen_d1_teeth/results/results.json` is known to be unfaithful, so a pass cannot separate "this adversary cannot fail" from "these states do not fail". The plan calls the two states bad by construction. Finding 5 shows that neither has been measured as bad under the masks this test uses.

**2. High: the outcome depends on the step scored.** The filed rule scores the read after step 20 (`qwen_vpd.py:416`). Mean ratio per step, recomputed from the logs (post hoc, no rule filed):

| step | Rand | Rare | N | LS | I | ADJ | A | W0 seed 0 over seed 1 |
|---|---|---|---|---|---|---|---|---|
| 5 | 1.00 | 1.30 | 1.21 | 1.10 | 1.08 | 1.03 | 0.98 | 1.016 |
| 8 | 1.00 | 1.56 | 1.40 | 1.21 | 1.15 | 1.08 | 0.92 | 1.032 |
| 10 | 1.00 | **1.62** | 1.50 | 1.25 | 1.18 | 1.11 | 0.89 | 1.039 |
| 13 | 1.00 | 1.40 | 1.33 | 1.20 | 1.15 | 1.07 | 0.90 | 1.028 |
| 16 | 1.01 | 1.34 | 1.29 | 1.24 | 1.21 | 1.12 | 0.96 | 1.108 |
| 19 | 1.01 | 1.26 | 1.21 | 1.21 | 1.22 | 1.04 | 1.16 | 1.032 |
| final, as filed | 1.01 | 1.27 | 1.24 | 1.28 | 1.28 | 1.15 | 1.12 | 1.184 |

- At step 10, Rare reads 1.56 and 1.67 on the two seeds, and its mean of 1.62 is over the fail bar. W0's two seeds there differ by 3.9%, against 18.4% at the filed read.
- Through the middle of the search the states fall in a clear order, Rare, then N, LS, I, ADJ, with Rand and A at or under 1. It holds on the means at every step from 5 to 13; per seed I checked step 10 only, where Rare, N, LS and I keep this order on both seeds. This order is roughly how much of the update sits in Δ or on the wrong columns, and it has gone by the final read.
- This does not overturn the filed outcome, and one step at 1.62 is thin. It does rule out "cannot", and it suggests the final read may be the least informative one. The E80 check saw the same shape for N alone (`reports/reviews/2026-11-04_review_e80-check.md`, finding 8).

**3. High: the ratio collapses because the denominator grows, so the bar is far out of reach of what this update does on web rows.**
- Rare's excess over W0 is 0.34 and 0.39 nats at step 10, and 0.27 and 0.55 at the final read. It stays about level while W0 climbs from 0.6 to 1.7.
- Reaching 1.61 at the final read needs 1.04 and 0.88 nats above W0. The largest excess any state has read is about 0.55 (N seed 0, LS and Rare seed 1).
- Clean, the fine-tuned model is only 0.040 nats from W0 on E79's eight web rows, against 1.84 on maths rows (`experiments/qwen_d1_adversary/results/results.json` `.own.W0`). So the rows are ones where this update has little room, and the plan says so (plan line 44).
- The search has not settled either: reads rise 6-9% per step over the last five, and the last move is +10.5% on W0's seed 0 and -3.7% on seed 1. Every read is a lower bound on the search and the ratio of two lower bounds has no bound of its own. More steps could lower or raise it, though the falling trend above points down.

**4. Medium: the bars come from another model, and Qwen's own spread changes the label.** 1.40 and 1.61 are 1.5 divided and multiplied by 1.075, the 67M's floor (E80 check, finding 5). With this job's 18.4%, the same formula gives 1.267 and 1.776. Rare's 1.270 then misses the pass bar by 0.003, and the filed rule would read "no call" for that state, not "cannot". This is a coin-flip margin, so it shows only that the label rests on the bar's choice. `.claude/CLAUDE.md` asks for a band set from the control and the null in the same job, and here the null, do-nothing, was not read (plan line 42).

The adversary itself is VPD's evaluation setting, as the code says: one source per subcomponent and one Δ dial per matrix, shared over all 65,536 positions, 20 sign steps of 0.1 from a random start (`qwen_vpd.py:402-411`). It is not VPD's training adversary and not a per-position one, so the claim must not read as being about adversaries in general.

### (b) Are the states bad, by a measure that does not use the adversary?

**5. High: Rand is not bad in the sense this test reads, so its pass is the right answer and carries no information about teeth.**
- The KL runs from the state's own clean output to its own masked output (`qwen_vpd.py:373`, `qwen_teacher.py:298`). The update is not in the comparison, so the adversary cannot know u1 is missing. Rand's badness, kept share 0.008 and distance from the fine-tuned model about W0's (E75, analysis line 192), is badness against a reference the test does not hold.
- Rand barely changes the model: 2.4952 to 2.4812 nats on the gold rows, a change of 0.014 against the update's 1.81. A Δ dial that removes it therefore has almost nothing to remove.
- Rand tracks W0 within 3% at every step of both seeds, and within 1.5% at the final read. The size of a change in Δ does not decide the read, what it does to the output does: N and Rand have equal Frobenius norm per matrix, and read 1.24 and 1.01. The filed expectation that Rand reads like N (about 1.24) and the lean built on it missed.
- So the reading to report is "a change that does nothing to the output, left in Δ, is not penalised". Rand also gives this job's useful paired noise floor: a state near W0 reads within 0.3-1.5% of W0 at the same seed, far below the 18.4% that the rule uses.

**6. High: Rare's badness is established only under frozen masks, which this test does not use.**
- Its weights are the fine-tuned model's, so kept 1.00 and clean KL 0 hold by construction and cannot read the placement. The plan says this (line 12).
- E75's clean-table row for the rarely-on half (0.054 of the ruler) is a different state, with the remainder discarded (plan line 12).
- E68 read this very state: gain kept 0.004 under W0's masks against 0.176 for the often-on half, but 0.241 under its own masks against 0.226 for often-on and 0.240 for do-nothing, with own-mask KL 0.1044 equal to do-nothing's (`reports/analyses/2026-11-03_analysis_e68-d1-pass-test.md:217-225`). Maths rows. The adversary recomputes each state's masks, so the only earlier read with the same kind of masks found no harm.

**7. Medium: Rare is a mixture, not a clean wrong-column placement.**
- Pooled, 23% of the update's plain energy stays in Δ (the plan derived about 35%, inside its stop). By input-weighted norm, E68's measure, 0.712 stays in Δ. So it is partly do-nothing, and reading near N is expected.
- The sampled matrices split sharply. Where the half has about 3,000 columns (gate, up and down at layers 0 and 27), the share left in Δ is under 1e-4: the whole update really is on the rarely-on columns. At layer 14's attention matrices, where the half has 98-152 columns, 84-89% stays in Δ.
- "Placement of the first update onto the rarely-on half" needs that qualification.

**8. Low: "rarely on" holds on the rows read, less strongly.** The half is chosen on maths statistics (`run.py:70-76`, `qwen_maps.py:69-76`) and read on web rows. Its active share on web rows is 0.26% against 1.78% for all alive columns: a factor of 7, against 27 on maths. Of the half, 21,892 of 247,258 columns have no maths activity and are alive only through web rows (`qwen_maps.py:64-66`). The state is still as described.

### (c) Γ's rounding of V to bf16

What rounding acts on. A state reaches the KL by three routes. The masked forward's V and U (`qwen_vpd.py:329-334`) and the Δ dial's `d * out` (`:334`) are fp32 and see every change exactly. Only the third, g, is rounded: taps cast to bf16 (`:235`), site inputs and V cast to bf16 for the readout (`:249`), and head plus readout summed in bf16 (`:264`).

**By reasoning:**

- **Rand and N have no moved V, so line 249 cannot apply to them through V.** They reach g only through activations, which are also rounded. Whatever g does, Rand's change is read in full by the Δ dial in fp32. Rand's 1.01 therefore comes from G doing little to the output and is not a rounding artefact.
- **Rounding is not a dead zone.** Below one spacing, some entries flip by a full step and the rest stay put. The change passes on average with added noise. Masks being exactly unchanged is not what to expect.
- **Rare's dV does reach the rounded V, as a noisy copy.** On 21 sampled matrices, with V's bf16 spacing being 0.4-0.8% of an entry:

| | range | median |
|---|---|---|
| ‖dV‖ / ‖V‖ of the half | 0.1-0.8% | 0.15% |
| median entry's change, in bf16 spacings | 0.16-3.4 | 0.29 |
| entries of the half showing no change | 14-70% | 58% |
| error of what Γ sees, over ‖dV‖ | 0.10-1.8 | 1.2 |
| cosine of what Γ sees with dV | 0.48-1.00 | 0.64 |
| columns with no change at all | 0 | 0 |

So rounding does not erase the placement, it adds noise about the size of dV. The cleanest matrices are the down matrices at layers 14 and 27, where dV exceeds a spacing.

- **How large a change must be.** Per entry, about one spacing (0.4-0.8% of the entry) registers every time. For a readout, h is itself rounded, so a change must beat about 0.2-0.4% of h. On a random input Rare's change to h is 0.09-0.6% (median 0.14%), at or below that step, though real inputs may differ. For activations, G at one matrix is 0.12-0.36% of W's Frobenius norm (V U of the alive columns as the proxy; median 0.15%), so on an input with no special alignment it changes that matrix's output by about that share, which is about one rounding step before layers accumulate. u1 is the same size in Frobenius norm but aligned with the inputs (1440 input-weighted against 3.49 plain), so it moves activations further.
- **There is measured evidence that g hears a weight change of u1's kind.** A's masks with W0's unchanged components read 0.094 on maths rows against W0's masks at 1.33 (decision `2026-11-04_decision_e80-d1-test.md`, E79), so rounding is not stopping the activation route for that change, on those rows.
- **Direction.** The sign of the effect on Rare's read can't be got by reasoning. If Γ saw dV better, g on those columns could rise, leaving the adversary less control, or could fall. The reasoning does show that E72's deafness (`reports/analyses/2026-11-03_analysis_e72-d1-second-test.md:15`) is not caused by rounding, since those masks were fixed.
- **Rounding does limit generalising to smaller updates.** dV scales with the update. An update 10 times smaller sits at a median of about 0.03 spacings, with few entries flipping and the signal well under the noise, so its moved V would be close to invisible in the readout route.
- **This matches VPD's own code**, so it is part of the instrument and not a bug. Γ was also trained on a model whose V did not move, and this use is new.

**Unmeasured:** how much g actually changes for Rand and Rare on web rows, by which route, and whether an fp32 V would change the final read. My conclusion on (c): rounding does not explain either pass by the arguments above, and it adds noise on Rare's readout route of unknown effect on the read.

### (d) How far it generalises

One update, whose gain saturates (a tenth keeps 0.93). One decomposition, one row set where the update has little effect, two seeds, 20 steps, two states. The maths rows were not read for these states, and E80's check showed they carry a raised level before any attack. It says nothing about other updates, about smaller or larger changes, or about the 67M, where this adversary did fail do-nothing and least squares at 4.79 and 12.34.

Rare reads with LS and I at the final read (1.27, 1.28, 1.28), so **"at the filed read it does not tell a rarely-on placement from a good split"** is the strongest thing supported, and it is D1's real concern. That statement needs the step caveat: those states do separate at the middle of the search.

## The sentence I would allow

> On the first fine-tune update of Qwen3-1.7B, on web text, VPD's evaluation adversary did not reach the fail bar carried over from the 67M at its final read for either state we tried: a random change left in Δ, which barely moves the model's output and reads like the unchanged decomposition, and the update placed mostly on the rarely-on half of the alive subcomponents, which reads level with do-nothing and the least-squares split. No state known to be unfaithful has been read on this model, so whether the test can fail anything here is still open.

If the answer paragraph needs it shorter: "At its final read, VPD's adversary did not separate a rarely-on placement of the first update from a good split on web text; whether it can fail any state on this model is untested."

Not allowed: "cannot", "a bad state" without saying by what measure, the random state as evidence of missing teeth, or anything beyond this update and these rows.

## Measurements I recommend

1. **A positive control (one start, about 22 min plus about 20 min set-up, about $3 on an H200 NVL; a second state in the same pod adds about $1.50).** Same harness, seed 0, W0's 1.703 reused since it repeats exactly on this card. Rand scaled 30 times, so it moves the output (E75 needed 18-100 times an input-weighted match to reach the ruler). Optionally a second state with weights W0 and each matrix's alive U columns permuted, a decomposition plainly wrong while the clean output stays W0's. Read the final and step 10 ratios, and the clean KL to W0. Over 1.61: the test can fail, and the sentence becomes "these two states do not reach it". Under 1.40: the adversary really is insensitive here, and something close to the original sentence earns its place, with this as its evidence.
2. **Does rounding matter (one `debug` job, about 5-10 min, $0).** On 8 web rows, read g for W0; Rand; Rare as run; Rare with h computed in fp32 before the cast; and Rare with V at W0's so that only activations differ. Report per site the mean |Δg| and the share of (position, column) pairs moving over 0.01. If the fp32 read differs from "as run" by much less than "as run" differs from W0, rounding is not a limit, and this closes (c). If it moves g much more, run one fp32-V adversary start (22 min) before claiming anything about placements. Check it fits an L40 first: only g is needed, so the components can be left off the card apart from V.
3. **No new run, for the write-up.** Report the step table above. A filed rule for any later adversary read should fix a mid-search or mean read and read absolute excess in nats, with its band set from paired seeds in the same job.

## Unverified

- No GPU rerun, and that the pod ran commit 6cba6504.
- The sampled matrices are 21 of 196 (layers 0, 14, 27), with W stood in for by V U of the alive columns, and "random input" sizes in place of real activations. Rare's pooled 23% was not recomputed over all sites.
- N, LS, I, ADJ and A's traces come from other pods. W0's trace matches across them to 3-4 places, with E85's W0 used as the denominator.
- The 0.040 clean KL is E79's 8 rows of 256 tokens, not the 128 rows of 512.
- E68's, E72's and E75's numbers were taken from their reports, not recomputed. The E79 own-mask reads are on bf16 components.
- `split.gram_basis` details and E76's state code were not re-read beyond what the rebuild used.

Not run: every measurement listed above; a third seed; more than 20 steps; these states on maths rows; other updates.
