# Advice: what E62 Amendment 4's failed control says, and what to run next

*Fable, 2026-11-02, advising the re-gating lead; the lead decides. Read: `reports/plans/2026-11-02_plan_e62-copying-heads.md`,
section "Amendment 4 (2026-11-02)"; `experiments/induction_cue/results/amend4.json` and `predictions_amend4.json`;
`experiments/induction_cue/amend4.py`; `src/commute/copying.py` (`second_a_control`, `ablated`, `AllGatesOn`, `closed`);
`experiments/induction_cue/amend2.py` (`Reads`); `reports/analyses/2026-11-02_analysis_e62-amend3-gates-head-off.md`;
`reports/plans/2026-11-02_plan_regating-next-FABLE-DRAFT.md`; the job log `cluster/logs/commute-e62-amend4-48945.out`.
Every number below is from `amend4.json`, `amend2.json` or `amend3.json`; nothing new ran.*

The control asked whether the decomposition's own forward, run with the mask the original model gets when it copies, loses its copy when the original copying head is switched off, as the original model does. It copies, and it keeps nearly two thirds of the copy's pair dependence, but switching the head off costs it a fifth of what it costs the original model, so the test's premise that this forward copies through that head is not met. The miss stands as filed; it does not show that the masks drop the original model's copying, it shows that the decomposition's forward at the original model copies without depending on that head, and one free job on the cluster can say where its copy runs instead.

## 1. What the miss allows one to say, and what it does not

**The numbers, in one place.** W₀ at the second A, log-odds of B, mean over the 200 copying-map windows; target model fp32, masked forward bf16 under W₀'s own mask (`stage1_control`).

| forward | intact | head (2,4) off | pair broken | fall (intact − off) | lean (intact − broken) | lean not through (2,4) (lean − fall) |
|---|---|---|---|---|---|---|
| target model | −4.71 | −9.53 | −11.55 | 4.82 | 6.84 | 2.02 |
| masked, W₀'s mask | −7.30 | −8.22 | −11.64 | 0.92 | 4.34 | 3.42 |
| masked as a share of the target's | | | | **0.19** (bar 0.25) | 0.64 (bar 0.25, passes) | 1.7 |

**What it says.**

1. **The masked forward at W₀ copies at the second A.** Its lean is 4.34, and with the pair broken it sits where the target model sits (−11.64 against −11.55). So the 2.6 log-odds by which it sits under the target model on the intact windows is all copy signal, not a flat offset: under W₀'s mask the forward keeps about two thirds of the pair dependence and loses the rest.
2. **Its copy does not depend on head (2,4) as the target model's does.** Replacing the head's output by its mean costs 0.92 there against 4.82 in the target model. The copy that survives the head off is 3.42 in the masked forward against 2.02 in the target model. So whatever carries the masked forward's copy, more of it is outside (2,4) than in the original model, in absolute log-odds and not only as a share.
3. **The filed premise is not met, so stages 2-4 could not have been read against the head label.** `S_old` is defined by the pair and not by the head, but the plan tied its meaning to "the subcomponents W₀'s copying opens", and "W₀'s copying" was pinned to head (2,4) by E62's causal label. In a forward whose copy runs a fifth through that head, a closing share would have named subcomponents of some copy, not of the labelled mechanism.

**What it does not say.**

- **Not "the masks drop W₀'s copying mechanism wherever it matters"** (my draft's sentence for a failed control). That sentence assumed the masked forward would stop copying. It did not: the lean passes at 0.64. What is shown is narrower: under W₀'s mask the forward's copy does not run through the labelled head as the original model's does. Whether it drops the mechanism, or carries the same copy by other heads, or passes part of the head, is open.
- **Not that (2,4)'s slot is shut.** The only slot number is Amendment 3's sample: eight `o_proj` subcomponents holding 5.4% of the slot's read weight, mean mask 0.008 at W₀, at the marker and not at the second A (`amend3.json`, `row3_slot.W0`). The other 95% is unread; every block-2 slot's sample reads under 0.1 there, (2,3)'s at 0.045 and (2,2)'s at 0.095.
- **Not which heads carry the masked forward's 4.34.** No head but (2,4) was switched off in this forward at W₀ on these windows.
- **Not that the mean ablation reads the same thing in both forwards.** In the masked forward the head's q, k, v and the `o_proj` that reads it are all masked, so its output at the second A need not resemble the target model's, and the replacement vector is the target model's mean (`heads.pt`).

**The accounts that remain.** They are not exclusive; the first two are nearly the same account seen from the output and the input side.

| account | what it says | fits the numbers? | what separates it, and whether that exists |
|---|---|---|---|
| A. Rerouting. W₀'s mask cuts (2,4)'s contribution and the copy runs through other heads of W₀ (its second prefix head (2,3), or others) | the masked forward's copy outside (2,4) is 3.42 against the target model's 2.02 | yes, directly | a head sweep in the masked forward under W₀'s mask against the target model: **not saved**, one debug job (section 2) |
| C. The mask passes a small part of the slot. `o_proj`'s subcomponents reading (2,4)'s slot are mostly closed, so replacing the slot changes little | the sample reads 0.008 at the marker | fits the small fall; alone it does not give the 3.42, since closing a slot does not make other routes carry more than they do in the target model (unless the head's output suppresses them, unread) | the weighted slot read under W₀'s mask against every mask at 1: **not saved**, the plan's own next step; the sweep reads its consequence rather than its cause |
| B. The ablation reads differently in the decomposed forward (bf16, a decomposed `o_proj`, a target-model mean injected into a forward whose head output is already reduced) | | only as an apparatus artefact that does not depend on the mask; with every mask at 1 on repeated random sequences the same hook takes the masked forward from 0.82 to 0.0095 top-1, as the target model (`amend2.json`, `row1_control`), so the hook works in that forward with every gate open | the (2,4) off row with every mask at 1 **on these windows at the second A**: not saved (the Amendment 3 report lists it under "Not run"); one forward pair in the same job |
| D. The lean comes through the tokens, not the gates. The masked forward copies with either gate set once the first A is present, and `a`, the gate difference, is what Γ changes, not what carries the copy | the masked lean moves tokens and gates together | consistent; untested | the two swapped forwards, W₀'s pair-broken gates on the intact tokens and W₀'s intact gates on the broken tokens: stage 2's `token_part` would have read one of them, and did not run; nothing saved |

Checked before naming these: `amend4.json` holds only the means and the top ten entries of `a`; `a`, `a_half2` and `S_old` were to be saved at the end of `main` (`amend4_vectors.pt`) and were not, since stage 1 exited. `results/large/` holds `windows.pt`, `heads.pt` and the six states only. The slot sample and the all-ones hook check are the only saved readings that bear on this, and neither is at the second A on these windows.

**Precision** is not an account here: a fall of 0.92 against 4.82 is far outside bf16 rounding, and the all-ones row in the next job reads the ablation in bf16 anyway.

**Granularity limit.** Everything above is at the level of heads. Even a masked forward whose copy ran entirely through (2,4) could run a different circuit of that head's subcomponents, and VPD would read it as the same mechanism; the state-level test inherits that limit, and a head sweep narrows which heads, not which circuit.

## 2. What to run next

Prefer one plain test: **the head sweep under W₀'s mask, with an all-ones row as its control**, one cluster `debug` job, $0, about five minutes on an L40, no training, nothing in `results/large` touched.

### Option 1. The head sweep with an all-ones control (recommended)

**What runs.** W₀ only, the 200 copying-map windows, read at the second A, log-odds of B. First, the control, alone, with a stop: head (2,4) off under every mask at 1 (`AllGatesOn`, in the masked forward), intact and pair-broken windows. Expected: a fall near the target model's 4.82 and a lean near 6.84, since the same hook in that forward reproduces the target model on repeated sequences; a fall under a quarter of the target's stops the job as "the ablation does not transfer into the decomposed forward", and the sweep is not read. Then each of the 24 heads switched off one at a time, in the target model and in the masked forward under W₀'s own mask, intact windows: 48 reads of one forward each, plus the (2,4) rows already in `amend4.json` as the re-read. Beside, no rule: the two swapped forwards for account D (W₀'s pair-broken gates on the intact tokens, W₀'s intact gates on the pair-broken tokens). Save `a`, `a_half2` and `S_old` from stage 1 this time, since they are cheap and were lost. File expected values before the job: (2,4) target fall 4.82 and masked fall 0.92 (re-read, tolerance 1e-3 as Amendment 4's stage 3); all-ones fall at least a quarter of 4.82; every other head's target fall under (2,4)'s (E62's ranking); no expected value for the other heads' masked falls, which is the reading.

**What it tells you.** Where the masked forward's copy runs, at head level, and whether the loss at (2,4) is the mask's doing. Three outcomes:
- *Rerouting.* The all-ones fall reads near 4.82, and under W₀'s mask one or more other heads fall by about a log-odds or more where their target-model fall is small, (2,3) the first candidate. Then W₀'s mask moves W₀'s copy off the labelled head onto other heads of W₀, before any fine-tune. For "do the masks read the mechanism": no, at head granularity, at W₀ itself; the decomposition's forward with W₀'s mask runs a different head-level route for the same behaviour, so no mask at a fine-tuned state can be scored against the head label on this model. For the Qwen table: the split's numbers only, with the caption stating that on the 67M the masked forward at W₀ reroutes the one labelled behaviour around its labelled head; r stays unscored, as E62 filed.
- *Uniform attenuation.* Every head's masked fall is a similar fraction of its target fall, and no head carries the extra 1.4 log-odds. Then the mask thins the whole copy pathway and the head-off read cannot find a carrier; account D (the copy is token-carried through whatever is open) is then the live account, and the swapped forwards in the same job say so. For the masks: not shown either way at head level, and the subcomponent-level test has no head anchor to score against. For the Qwen table: as above, with "thins" in place of "reroutes".
- *The control stops.* The all-ones fall is small on these windows. Then the mean ablation does not transfer into the decomposed forward at the second A, and Amendments 2 and 3's frozen-mask head-off reads of the masked forward (frozen share 0.10 to 0.11, W₀'s frozen fall 0.04) are affected too: an apparatus finding first, to the user before anything else on this line.

**What comes next.** Rerouting: the weighted slot read and the re-open at W₀ (the plan's named next step under this verdict), to size how much of (2,4)'s slot the mask passes and whether re-opening it returns the fall; a state-level test would then need its own design, with the masked forward's own carriers as the anchor, which is a new plan and not a refiling. Uniform attenuation: the slot read is still the sizing step; the state-level test is not worth building until something in the masked forward is known to carry the copy. Control stop: the hook against the decomposed `o_proj` at the second A, with zero ablation beside the mean, before any further head-off read.

### Option 2. The weighted slot read at the second A, first

**What runs.** One debug job, $0: per window, (2,4)'s slot output through the masked `o_proj` under W₀'s mask against every mask at 1, all 524 alive subcomponents of that matrix weighted by their read vectors, for all six block-2 slots; then the same under each state's mask. The plan's own next step for this verdict.

**What it tells you.** How much of each head's output W₀'s mask passes: account C sized, and a comparison between (2,4) and its neighbours. It does not say where the copy goes or whether the ablation transfers, so after it the sweep is still needed to name a carrier. For the masks and the Qwen table it says only "the mask closes this slot by this much", a property of W₀'s mask, not of the mechanism.

**What comes next.** The sweep, then the re-open. Two jobs where one would do; take it second.

### Option 3. Run stages 2-4 as explicitly unscored exploration

**What runs.** A copy of `amend4.py`'s stages 2-4 past the stop, under a different results file, every number marked unscored; `predictions_amend4.json` untouched. $0, about six minutes.

**What it tells you.** Whether `S_old`, the pair-defined set, carries W₀'s masked copy (stage 2), and whether the states open it and the masked gain runs through it (stage 4), as numbers with no filed reading. `S_old` is indeed defined by the pair and not the head, so these numbers exist; but what they would be numbers *about* is the masked forward's own copy, which section 1 shows is not the labelled mechanism at head level. Read before the sweep, a high closing share would tempt the sentence "the state opens what W₀'s copying opens", which the failed control forbids, and a low one would tempt the stop, which it equally cannot support.

**What comes next.** Not now. After the sweep, if it reads rerouting, these stages are the shape of a new test anchored on the masked forward's own carriers, designed and filed as its own amendment with a control that its carriers can pass; that is a new plan, and it must be written as one rather than as this amendment with the head bar removed.

### Option 4. Stop the 67M copying work on r's strength

**What runs.** Nothing. r on the copying map is reported as E62 filed it, "the original subcomponents can write this target under some mask", the Qwen table carries the split's numbers, and the 67M head work closes with Amendment 4's control as its last reading.

**What it tells you.** Nothing more; it forgoes one $0 job that would turn "the control failed" into a named property of the masked forward at W₀ (reroutes, thins, or the ablation does not transfer), which is the sentence the Qwen table's caption needs anyway.

**What comes next.** The Qwen table with the caption "on the 67M, the decomposition's forward under W₀'s mask does not copy through W₀'s copying head; mask-level mechanism claims are not validated".

### Recommendation and what would change it

Option 1, one debug job, $0, with the all-ones row first and alone under a stop. It is the one read that turns the miss into a cause at the level the label lives at, it needs no ablation inside Γ's pass, no state, no training, and it leaves the filed control exactly as it reads. Its outcome is what the Qwen caption says either way.

It would change if: (a) the user wants the Qwen table today, in which case option 4's caption goes in now and the sweep runs when there is time, since nothing in the Qwen plan waits on it; (b) the lead finds a saved (2,4)-off read with every mask at 1 on the cue windows that I did not (I grepped `amend2.json`, `amend3.json`, `run.json` and the Amendment 3 report's "Not run": the all-ones hook check is on repeated sequences only), in which case the control row is dropped and the sweep alone runs; (c) the sweep's control stops, in which case the apparatus question goes to the user before any of the three outcomes is pursued.

## Not run

Nothing here ran. `a`, `a_half2` and `S_old` are not on disk, so the 172 subcomponents cannot be inspected without a re-run of stage 1, which takes seconds and is in option 1. Not proposed: ablation inside Γ's pass; zero ablation except under the control-stop branch; the states' masks in the sweep; steps 100 and 300; a second decomposition seed, the only check that narrows the granularity limit.
