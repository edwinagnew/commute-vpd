# Advice: what is wrong with D1's test, and the cheapest reads that say which part (E73 follow-up)

*2026-11-03, Fable, read-only. Read: `.claude/CLAUDE.md`, `GOALS.md`, E68's analysis and review, E72's plan, score and
log (u1 done, u2 in stage S), E73's plan, code and every JSON, E61's analysis, `src/commute/qwen_vpd.py`,
`experiments/qwen_pass_test/{run,amend1}.py`, `experiments/qwen_fixed_masks/run.py`, `experiments/qwen_maps/amend2.py`.
Every number below was recomputed with jq from the per-read files; the lead's numbers all check (details at the end).
No GPU, nothing run.*

**In plain words.** The masked forward with its masks held fixed is nearly deaf to weight changes of any kind: a
random change 18 times the update's size moves it by 0.007 to 0.017 nats where it moves the clean model by 0.5 to
0.58. That is why the whole update closes a tenth and why no placement can be graded. On top of that, on the maths
rows the gold updates mostly make the model less sure of its own habits, so the KL ruler there reads sharpness, not
maths. The template update is the clean case, and it is lost the same way. Two cheap reads, about 20 minutes of L40,
separate "the fixed masks drop everything" from "this update needs components the masks switch off".

## 1. The suspects

**Task and rows (the gold updates on maths rows).** Supported, for the gold updates only. KL(FT ‖ W0) on u1's rows is
1.64 nats against a next-token NLL change of 0.17. By E73's `detail`, 68% of that KL sits at the 48% of positions where
W0 puts over 0.9 on its own habit: there FT's entropy rises from 0.005 and 0.21 to 0.56 and 1.57, and its accuracy
falls. FT's NLL gain on these rows is a user-turn gain (3.66 to 2.88 on 4038 positions) net of an assistant-turn loss
(1.19 to 1.36 on 6638); the SFT loss covers response tokens only (`qwen_sft.targets`), so the gain is a side effect
of learning GSM8K's domain, not the trained behaviour. The gold fine-tune cuts the human-text loss by 1.986 nats and
halves answer length (E61): a format change, not a small update, read on text FT would not write. Where the lead's
reading stops short: this is a property of the ruler's *direction* as much as of the rows. KL(FT ‖ ·) is small for any
state as soft as FT and roughly right, which is why every clean piece reads near FT (in-span 0.009, both halves
under 0.09) and why masked W0 (softer) sits nearer FT than clean W0 on the three maths-row updates (1.374 vs 1.635)
and further on the template (0.364 vs 0.323, where FT's entropy is flat). Not read: how much of the 1.64 one scalar
temperature removes (read 2 below). Caveat on the rows themselves: assistant-turn accuracy at W0 is 0.86, not 1,
because the rows are packed and were generated in bf16; "W0's own greedy text" is approximate.

**Recipe.** Not the separator. u1 (no warm-up, bf16 forward, step 10), u2 (gold, warm-up, fp32, step 100) and u4
(step 100) read alike on every E73 number: entropy 0.60 to 1.3-1.5, assistant NLL up, user NLL down, −δW far worse,
NLL minimum at t 0.3-0.6. What differs is the data. On "small": by `GOALS.md` size is measured, and at 0.62 nats KL
from W0 on W0's own text these are not small; E61 says size is flat from step 20 to 100. The test's own floor is
K0 = 0.07 (the masks' damage), so an update small enough to be "small" is under the floor and unscorable. The
template at 0.27 to 0.32 is the one update in the readable band, which is why it matters most.

**Model.** Cleared as a fragile object: bf16 rounding 0, 0.2% noise 6e-5, Gaussian or shuffled δW at δW's norm 1e-6
to 4e-4, while 1% of the real δW gives 1e-3 (3e-3 on u4); the real direction is 10³ to 10⁴ times a random one at
equal norm. One E73 lean missed and should be said: W0 is overconfident on FineWeb too (1.10, filed "calibrated"), so
"sharp everywhere" is a property of this post-trained model, and any SFT on human text de-sharpens it. That is
model-and-ruler, not model fragility.

**Decomposition.** Not coarse in weight space: the in-span part (0.64 of the energy) reproduces FT to 0.009 nats
clean; K0 is 0.07 on maths rows at about 38 of 1024-6144 subcomponents on per site per position (g sums to 7454
over 196 sites). What is not separated from the test: whether the update's *function* needs subcomponents that W0's
masks switch off. T0b is the cleanest statement so far: FT's full weights minus W0's off subcomponents reads 1.26
nats from FT, while W0 minus the same subcomponents reads 0.07 from W0. That is either "FT's behaviour runs through
switched-off subcomponents" or "any change in the activations is cut by a sparse mask". Read 1 separates them.

**Test.** The main suspect, and already quantified three ways, all on u1's rows: the random state moved on the
components (18× δW's norm, clean KL 0.50) shifts the masked forward by 0.007 under W0's masks and 0.017 under FT's,
a transmission of 2-3%; the same change added as an unmasked term (T0a) closes 0.054 of its own gap, δW 0.099
(u2: 0.055 and 0.121); under FT's masks every state reads 0.1034-0.1045 and gains 0.20-0.24, the masks carrying
everything and the weights nothing (E44 Amendments 1, 1b). Δ is excluded: Δ on under the masks is slightly worse
(1.262 vs 1.244; u2 1.108 vs 1.095) and opening to g = 1 reads 0.013. So the fixed-mask forward passes a few percent
of any weight change and the re-read masks pass all of them; neither grades a placement. What is not read is *where*
the few percent are lost: at the site the change is added (a hook or input-mismatch problem) or downstream, where
the perturbation must pass each later site's ~38 read vectors.

**Something else: the structure of a sparse masked forward.** With per-position masks frozen, the masked model is a
very sparse model whose behaviour is set by which subcomponents are on. It is faithful on the activations the masks
were taken from and, by construction, transmits only the part of a residual-stream perturbation that lies in the on
subcomponents' read span at each downstream site. If read 1 confirms this, no weight-space placement can pass under
fixed masks, and the question for D1's test moves from "which placement" to "what carries the update: the masks
(re-gating, the fixed-shift line) or the weights", which must be read in g, not in KL under one mask set.

**Bugs.** None found. `masked_logits`' hook is ((x V) ∘ m) U with Δ off; `forward_plus` adds x δWᵀ at every site;
E72's `FixedReader.row` reads g from FT's forward with W0's V (`OriginalV`), the same g for every state (gap 0 on all
seven), and `T0b` is what its docstring says (d = 1 with FT in the model: FT minus the off part). E73's `roles`,
`detail` and `base_numbers` do what they say; its control stop was a normaliser, fixed by Amendment 1. One numerical
note: on u1 the 28 blocks' KLs sum to 0.67 against 0.62 for the whole update; KLs that add mean the blocks' logit
shifts are nearly orthogonal, which says the update is not one coherent direction in logit space, whatever it is in
weight space.

## 2. The reads, cheapest first (L40, times from E72 and E73's logs)

Masked read of 128 rows: 54 s (do-nothing), 95-130 s (moved states), 175-226 s (unmasked term from the host). Clean
forward: 3-4 s per 64 rows at batch 8. Load and Γ: ~6 min per job.

**Read 1, where the masked forward loses a change (the positive control for the masked forward).** Add one block's
δW (u1; `parse_site(s)[0] == l`, as E73 `Job.blocks`) as an unmasked term under W0's masks, for l in {0, 7, 14, 21,
27}; for each l also a Gaussian change in that block with the same clean KL (scale by the quadratic rule from E73's
block read, one clean forward to confirm). 32 maths rows. Numbers, per l and change: the clean shift
KL(W0 ‖ W0 + δW_l) and the masked shift KL(masked W0 ‖ masked W0 + δW_l unmasked); their ratio τ_l; and
KL(W0 + δW_l ‖ masked W0 + δW_l) against KL(W0 + δW_l ‖ masked W0), T0a's pair. Code: `FixedReader.t0a` already
computes T0a's pair with an arbitrary host change; put the one block's δW (~100 MB) on the GPU to avoid the 226 s
host path. Expected values: do-nothing is τ undefined (no change), so the zero is K(masked W0 ‖ masked W0) = 0;
block 27 is the positive control: its output enters the residual with only the final norm and unembedding
downstream, both unmasked, so τ_27 must be near 1 (clean effect 0.0034 nats, paired, readable). Outcomes: τ_27 near
1 and τ_0 under 0.1 with the Gaussian matching δW at each l: the loss is downstream propagation through sparse
masks, generic, the test is structurally blind; τ_27 near 1 and the Gaussian passing much better than δW at early l:
δW's early effect runs through switched-off subcomponents, the decomposition-specific reading, and the often-on half
should then be the part that passes; τ_27 well under 1: the added term is wrong at the site (a bug or an input
mismatch), stop and find it before any theory. Optional fifth arm, 5 reads: δW_l with every site after layer l
unmasked (`forward_plus`'s t = 1 for those sites), which should read τ near 1 at every l if the downstream story is
right. Cost: 11 masked reads of 32 rows at ~15-30 s plus clean forwards, ~15 min with load.

**Read 2, how much of the ruler is sharpness.** From one W0 and one FT forward per batch (E73's `Reader.logp`), for
τ in {1, 1.25, 1.5, 2, 2.5, 3}: KL(FT ‖ W0 at temperature τ), NLL, entropy, agreement; by turn as `detail` does.
Same for u3 on a1 rows and for masked W0 (its entropy has never been read; one masked read of 32 rows). Expected: W0
at τ = 1 reads 1.635 (u1), FT at τ = 1 reads 0 and at τ ≠ 1 reads above 0 (the control). Outcomes: the minimum over
τ at or under 0.4 on the gold updates and no gain on u3: the KL ruler on maths rows is mostly one scalar and the
gold updates there measure de-sharpening; stop using maths rows and the gold updates as D1's yardstick, keep u3 and
rows FT writes. Minimum still above 1.0: the gold updates' effect is specific and the task suspect falls back to the
gain ruler's turn mix. Cost: 2 min of forwards, ~8 min with load. Can share read 1's job.

**Read 3, the rows swap for the gain ruler.** E68's W0 and ceiling reads (`PassReader.row` with "ci", "K";
`CeilingReader`) on 32 held-out human GSM8K solutions, chat-formatted (`qwen_sets.load()["test"]`,
`gsm8k.gold_as_response`), for u2: K0, KD, K_C, gain. Expected: do-nothing 0, clean W0 to FT about 2 nats of NLL.
Outcomes: the ceiling closes ~0.1 again: the rows are not what caps the test; it closes most of the gap: the maths
rows were the cap. Cost: ~5 min of reads, ~11 min with load. Lower priority than 1 and 2; it bears on the gain ruler,
not on the masked forward.

**Read 4, free: E72's u3 do-nothing under u3's own FT masks (queued, ~18:15).** If it reads near 1.5 K0 (≈0.105) with
gain near 1, the masks carry even the template's low-entropy behaviour with W0's subcomponents, and the "masks carry
everything" reading holds for a clean learned update too. If it reads far above, u3 is the first update whose
behaviour the frozen Γ cannot elicit, and the template becomes the update to build D1's test on.

## 3. Recommendation

Run read 1 first, with read 2 in the same job (same model load, five minutes more). The one outcome that would most
change the plan is τ_27 well under 1: it would mean the whole-update term is not entering the masked forward as
believed, and every ceiling, T0a and T0b number would be void; everything else is interpretation. The expected
outcome, τ rising from a few percent at block 0 to near 1 at block 27 with the Gaussian tracking δW, would settle
that the test, not the decomposition or the update, is what cannot pass anything, and would retire the fixed-mask
and ceiling family of tests for D1: a placement would then be judged on what the masks cannot carry, which is the
re-gating line (how g changes between W0's and FT's forward, per cue) or VPD's stochastic and adversarial checks
under reading A (where "elicited by switching subcomponents off" may be brittle where W0's own masks are not; the
adversary on 16 rows costs ~2.5 min per seed, `pgd_kl_rows`), not on KL under one mask set. What would change my
mind: the Gaussian block changes passing far better than δW's at early layers (then the decomposition is the
suspect and the often-on half is the thing to place), or read 2 leaving over a nat on the gold updates (then the
task suspect weakens and u1-u4 stay as yardsticks). Nothing here names a cause as established: the masks carry FT's
behaviour whichever way (E44 Amendments 1, 1b), and until read 1 runs, "the masks drop any change" and "this update
needs the off subcomponents" fit the same numbers.

## Numbers checked (jq, per-read sums)

Ceiling shares 0.0995, 0.1212, 0.0291, 0.1388; FT-mask u1 gains 0.2397, 0.2012, 0.2315 (ls), 0.2273 (adjoint),
0.2281, 0.2401, 0.1635 (random), K 0.10436 to 0.10454 except random 0.1213; T0b 1.2618 vs 1.2440; K0 0.06961; clean
pieces 0.0092, 0.0297, 0.0883, 0.5307, 1.6351, random 1.3847; path 1.595, 1.279, 0.696, 0.287, 0.073; nulls at norm
≤ 4.1e-4; 1% δW 1.35e-3 (u1), 1.0e-3 (u2), 6.7e-5 (u3), 2.8e-3 (u4); NLL − H 1.461, 1.104, 2.042; blocks sum 0.668,
attention 0.292 + MLP 0.412 vs 0.616. Random's scale a = 17.78, its clean KL 0.502 (u1 `search.json`). u3's 1% read
is 7e-5, not the lead's "0.001-0.003", which holds for the three gold updates only.

## Not read

Everything in section 2; the masked forward's entropy; any adversary on a changed state; per-position KL under the
masks; which subcomponents Γ switches between W0's and FT's forward on the same row (the re-gating read); step 1-3
of the gold run under any test; rows FT writes.
