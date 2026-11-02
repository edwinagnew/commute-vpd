# Fable's advice: what the re-gating line does after E62

*Fable, 2026-11-02, for regating-lead. Advice only: nothing filed, run or committed. Read: the E62 analysis and plan,
`experiments/induction_cue/results/{run,predictions}.json`, `src/commute/copying.py`, `src/commute/local_faith.py`
(`route_cells`, `masked_logits`), `decisions.md` rows of 2026-11-02, the reuse lead's handoff. An independent check of the
analysis runs beside this; every number below was re-read from `run.json` except the (1,1) shares on seeds 1-2 and the
pair-broken table, which are the analysis's. Elicitation as the user defines it: the fine-tuned behaviour runs through a
mechanism W₀ already has. The filed verdict `copy_not_elicitation`, r unscored, stands.*

**In plain words.** E62 showed the copying fine-tune runs about half through the old copying head and the lookup not at
all, and r read 1.09 against 0.40 without being scored. The open worry is that r is high only because the answer B sits in
the window, not because the mask found the old head. The six fine-tuned states are saved, so one more $0 read can ask
whether the mask's gain for B goes through that same head. If it does, r and the label read the same thing on this map.

## On the three questions

**(a) Is "target in context" separable from "old mechanism" here?** Not fully, and it is worth saying plainly: on this
model the only mechanism W₀ owns that can produce a new answer is copying from context, so every in-context target that
the original weights can write goes through attention copying, which is old at head level. Reading the copy mask on a
token that follows no key, or the lookup mask on B, would tell "copies anything in the window" from "copies after a
matched A", but both are old copying, so neither read separates the two labels. A copy-style target that is in context
and does not go through the head cannot be built: the head is how context reaches the output. The one clean separation
left is the third one named: read the masked gain $L_{01} - L_{00}$ with head (2,4) mean-ablated inside the masked
forward. If the mask's gain falls by about the share the target's gain falls (0.47 to 0.56), r is high through the head
the label reads; if it barely falls, the mask reaches B by some other path and r's height is not the label's mechanism.

**(b) What carries the other half?** Head (1,1) is the obvious candidate: it costs the copy map 0.23 to 0.29 on its own
(seed 0 re-read: 0.287), four to five times its cost on the lookup, and it alone drops W₀'s copying as far as (2,4) does.
One read of (2,4) and (1,1) off together says whether the circuit's two heads account for most of the gain (then "about
half" was an under-count from reading one head) or not (then a loose context copying outside the circuit carries it, and
the pair-broken windows under ablation say how much). Worth measuring because it is free on the saved states and it is
what the 0.5 bar's reference needs: the same read on W₀'s own copying puts the bar on its own scale. Beyond those two
reads it is a distraction.

**(c) Later read steps.** Steps 100 and 300 need the six fine-tunes re-run (under a minute each, deterministic: E58 A6
reproduced r bitwise, and the saved step-30 sha256 checks it). The pair-broken ratio rises from 0.34-0.54 to 0.58-0.63 by
step 300, so the share could fall as training goes on. It would say whether "half through the old head" is a moment or a
property of the recipe, which the Qwen table's choice of read step needs. It cannot flip the filing. Not first.

## Options

**1. One $0 read on the saved states (recommended).** A cluster `debug` job, no training: load `windows.pt`, `heads.pt` and the six
step-30 states; read, on the same 200 windows and the same masks, (i) W₀ and each state with (2,4) off and with (2,4)+(1,1)
off, in the target model and in the masked forward of the original subcomponents under each state's mask; (ii) the
same on the pair-broken windows, target model only. It tells you whether r's gain goes through the head the label reads,
how much the two-head circuit carries, and gives the 0.5 bar its reference. Next: if the masked gain falls with the head,
report "r reads the label's mechanism on this map" as one datum and move to the Qwen table with that expectation; if
not, report r as "the original subcomponents can write this target under some mask", weaker than mechanism, and stop
the 67M copy work.

**2. Stop here and report.** Write the confound into the log as the line's standing limit and wait. Note: waiting for the
reuse line's VPD on the counting model waits for nothing, as `decisions.md` row 117 (2026-11-02) merged that line into
the baseline line and says the counting-set decomposition is not run. So this option leaves the re-gating line with no
mechanistic test at all.

**3. Dated amendment for steps 100 and 300.** Re-run the six fine-tunes to 300 steps, save the 100 and 300 states, repeat
E62's label and r at each. Tells whether the share and r move with training. One `debug` job, $0. Sensible only after option
1, and only if its answer makes the step question matter.

**Recommendation: option 1**, with option 3 folded in later if the step question is live.

## The recommended read: numbers and what each should read

| Number | Control, read first | W₀ | Do-nothing reference | Expected, copy map | Expected, lookup |
|---|---|---|---|---|---|
| Repeated-sequence top-1 with (2,4) off, through the masked forward with all gates on | must read about 0.01, as the target model's 0.0095: the hook attaches to the decomposition's forward | 0.8247 unablated | random head (3,5) off: over 0.7 | | |
| Target-model gain lost under (2,4), re-read | equals `run.json` within 1e-3 | | zero update: exactly 0 | 0.563, 0.545, 0.471 | 0.000, −0.004, −0.003 |
| Share of the masked gain $L_{01} - L_{00}$ lost under (2,4) | | $L_{00}$ with the head off: W₀'s lean of 1.57 log-odds (intact minus broken) should mostly go | (3,5) off: under 0.15 | within 0.1 of the target share per seed if r reads the label's head; under 0.25 if the mask reaches B another way | 0 ± 0.05, nothing to lose |
| Target-model share under (2,4)+(1,1) | | W₀ top-1 under both: at or under 0.01 | | at least 0.7 if the circuit carries the gain; near the single-head 0.5 if it does not | at most 0.1 |
| Pair-broken gain lost under (2,4) | | | | about the intact share if the loose copying also uses the head; near 0 if it does not | n/a |

Noise: row sd of the share is 0.016 to 0.017 (0.003 on the lookup) and the three-seed range 0.09, so the per-seed
comparison on the same windows uses "within 0.1", about six row sds, **chosen here, no source**. The bootstrap
`copying.share_sd` gives each new share its own sd.

**Blind to.** Heads other than (2,4) and (1,1); a fine-tune that keeps the head's key and value paths and changes the
query still counts as reuse; which subcomponents the mask changes; the granularity limit (a new circuit of old pieces reads
elicited) is unchanged; one model, one marker, step 30 only.

**Cost.** One MATS `debug` job, under 10 min, $0; no fine-tuning; **not gated**. New code: about 40 lines in a read script
that calls `copying.ablated` around `local_faith.masked_lo` and `route_cells`. One thing to check before writing it: the
masked forward calls the param-decomp component model, so grep whether `net.net.h[lay].attn.o_proj` is the module that
forward uses or is replaced by a components module; the hook goes on whichever module takes the heads' concatenated
output. The first row of the table is the control that catches a wrong attachment. Filed as a dated amendment with its
own predictions block, since it reads the saved states of a scored run.

**What would change this.** The independent check moving the shares or the cells by more than the row sd (then re-read
before anything runs). The hook not attaching to the decomposition's forward with no module that takes the heads'
output (then option 1 shrinks to its target-model rows, which still answer (b) and the bar's reference, and (a) is
reported as not separable here). The user wanting the Qwen table sooner (then option 2, with the confound stated in the
table's caption).

**Not run.** Nothing. Numbers quoted are `run.json`'s and the analysis's; the (1,1) shares on seeds 1-2 and the pair-broken
table were not re-derived.
