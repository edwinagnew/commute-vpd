cell: P1.47
rival_a: the fitted output gains absorb the thin new edit's new computation (each stood-in part's change is mostly a rescale of M's own write) predicts that E242's stood-in set S for new_thin, with every gain forced to 1, moves F by more than 0.21 of KL(F || M) (the bar 0.1 plus two spreads), and that the gains explain at least half of S's write change
rival_b: R_A counts parts, not how much of the change each carries: the thin edit's change sits in the 193 parts that keep F's mechanism (and in changed parts outside the circuit), while the 875 stood-in parts carry little of it, so the gains do no work, predicts that S with every gain forced to 1 still moves F by at most 0.21 of KL(F || M), and that the kept parts carry at least half of the write change of the circuit's changed parts against their 18% share by count
rung: premise
smallest_gap: 0.1
spread: 0.055
n: 2
design: paired
unit: halves of E242's windows for new_thin (E226's split seed; each half with its own K* and E242's stood-in set for that half), each read with the same set at fitted gains and at gain 1; all of D is read beside them, and x2, x10, rebuilt, new_few and the two random updates as references
spread_source: experiments/kstar_vpd_wording_a/results/items/new_thin.json
spread_of: proxy: KL(F || F with every standable changed member of K* stood in) / KL(F || M) on new_thin's two halves of D (0.618 and 0.696, so one half's sd about 0.055), the same units as this reading (a stood-in set's KL over KL(F || M)) on the same item; this run reads the gain-1 ratio on the two halves too and reports that spread beside it
floor: the reading is compared with the bar 0.1 KL(F || M) that S passes at fitted gains (E242: 0.099); rival a expects the gain-1 ratio above 0.21, rival b at or under 0.21; the fp32 floor of the stood-in path (2.6e-11 nats, E242 stage 0) is far below both
stop: a failed do-nothing, or E242's stood-in KL not reproduced at fitted gains within 1e-4 relative: nothing is read. Whichever rival holds, the diagnosis ends with this run and goes to the user as one page (built label or wording, and a draft wording (c) for the user to decide if the wording is the cause)
sketch: experiments/thin_edit_diagnosis/figures/sketch.png
rehearsal: cluster only: one debug-QoS L40 job of about 15 min, $0, no pod; it reads E242's saved items with E242's code, and its do-nothing stage runs first and alone with exact-zero checks

# Card: why does the thin new edit stand in under wording A?

**What runs.** One cluster job, $0. For each built edit and the random update, the set of parts that E242 found could
take the base model's mechanism with a gain is put in again twice: once with its fitted gains (this must repeat E242's
number) and once with every gain set to 1. Beside it, how much of each part's change in output a gain can explain, and
how much of the edit's change sits in the parts that keep the fine-tune's mechanism.

**Expected.** If the gains are doing the work, removing them breaks the thin edit's stand-in. If they are not, the thin
edit reads high because the few parts that keep the fine-tune's mechanism carry most of its change, and a count of parts
does not see that.

**Decision it serves.** Whether the "new" label of the thin edit is off at VPD, or wording A's share is too lenient for
a new computation spread thinly; in the second case a draft wording (c) for the user.

Plan `reports/plans/2026-11-21_plan_thin-edit-diagnosis.md`.
