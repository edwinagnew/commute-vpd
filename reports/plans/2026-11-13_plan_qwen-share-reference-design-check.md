# Design check: next Qwen work on decomposing SFT and on re-gating (draft, 2026-11-13)

*Draft for qwen-lead, not filed. User's direction (chat, 2026-11-12): "look more carefully into decomposing SFT on qwen
and/or studying elicitation/regating". Nothing here runs without a new cell from the user.*

**Goal restated.** The project asks whether a fine-tune's update can be read through a parameter decomposition, and
whether what it reads says if the fine-tune reused parts the model had or built new ones. Part 3 repeats on
Qwen3-1.7B whatever part 2 says works on the 67M model; part 2's gate page proposes that the re-gating share, the
prior and steering go to Qwen. So part 3 needs to know, first, whether the re-gating share reads anything on Qwen
at all. This proposal answers that for the decomposition where the share is not saturated.

## What the saved files already say ($0, read for this check)

From `experiments/qwen_library_share/results/p-015e0710/` (the frozen-neuron decomposition, P3.9):

1. **The large known-minus-invented gap is mostly a size difference.** At the late save the invented-facts update is
   about twice the known-facts one: weight-change norm 4.0 against 1.9, and the masked reading's KL to the fine-tuned
   model 1.5 against 0.65 nats (`run2/{known,invented}_s*_last.json`, `.state.dw_norm`, `.means.K00`). Across every
   state, real and random, the share falls as the change grows (known 0.992 at its matched save, 0.97-0.98 late;
   invented 0.983, then 0.92-0.93). At the matched save, where both sides sit at five times their size floor, the gap
   is 0.0085, the same as the random gap there, 0.0083 (`summary.json`, `facts/matched`). So the 0.049 at the late
   save, eight times the gold twin's 0.006, is what the size difference alone would give, not a sign the pair
   differs. More random draws at the same sizes would not remove this.
2. **The random changes were matched in size only loosely, by design.** The scale search stops within a factor 1.5
   of the target KL (`qwen_reading.scale_search`, `match=1.5`). Copy-late's random change landed at 2.87 against
   2.01, gold-late's at 0.45 against 0.58 (`run1/*_kl.json`, `.random.search`). With the share falling with size, part
   of every one-draw band is matching error.
3. **Real updates read above random ones here, which they did not on p-9707d6ff.** Gold-late seed 0 reads 0.961
   while its random change, at a smaller output change, reads 0.905; known-late reads 0.97-0.98 against 0.909. If this
   holds with several draws and an exact size match, the frozen-neuron decomposition treats real SFT updates as more
   re-gating-like than random changes of their size: the first sign on Qwen that the decomposition sees these
   updates differently from noise, which the P3.8 premise check asked about.

## The check

**Which goal does this serve?** GOALS.md's question whether re-gating describes what post-training does, carried to
the larger model (paper part 3). It decides whether the re-gating share can be one of "the readings that survive part
2" (ladder rung L3.2) on Qwen, and on which decomposition.

**Nearest thing tried.** P3.3 step 1 (E152) planted changes on subcomponents of p-9707d6ff: the share read 0.985 to
1.005 for every planted and random change, so it had no anchor there. P3.8 (E170/E174) and P3.9 (E175/E179) read the
library's 36 states with one random change per side and save. None of them read a change that is re-gating by
construction on the frozen-neuron decomposition, and none has a random reference at several sizes. That is what is new.

**Patch or step?** Candidate (a) as briefed, more random draws at the same sizes, is a patch: it would only widen the
band, and finding 1 says the one cell that matters is a size effect either way. The proposal below is a step: it
gives the share a top anchor (a built re-gating change) and a size curve (random changes at several sizes), which
together say whether the share reads re-gating on Qwen and where real SFT updates sit between the two.

**Candidates weighed.**
- (a) Several random draws per side and save at the existing sizes. About 36 reads. Answers "is the band too narrow"
  but not "is the gap size", and has no positive control. Not recommended alone.
- (b) Decomposing SFT, as what the decomposition can and cannot express. The outside share on the MLP matrices is
  zero by construction when the split may change each neuron's whole row (P3.3 step 0 found every MLP matrix fully
  reachable on p-9707d6ff too), so it cannot tell SFT from random. The informative version is placement: what share of
  each update's split falls on neurons that fire on its own training rows, against random changes. That needs the 36
  states fetched again and is a neuron reading, which part 2 left undecided. Second choice.
- (c) A built re-gating positive control plus a random size curve on the frozen-neuron decomposition, no fine-tuned
  state fetched. **Recommended.**

**Working decisions relied on or departed from.** Relies on E57's share as P3.8 and P3.9 used it, and on the user's
2026-11-12 choice of a KL-matched random band. Departs from the paper plan's "part 3 on p-9707d6ff only", as P3.9
already did with the user's cell. The built change is a new control, not a new rule: it adds no verdict.

**Adopted definition.** No reuse or new verdict. The result is descriptive: whether the share reads a built
re-gating change as re-gating, and where real updates sit against random ones of their size.

**What result would change what we do next?**
- Built change near 1 and the random curve clearly below the real states: the share reads re-gating on the
  frozen-neuron decomposition, and part 3 carries it with a size-matched random curve as its reference; pair gaps are
  then read against each other's sizes.
- Built change reads like random: the share does not read re-gating on Qwen on either decomposition. Part 3 drops it
  and says so (P3.9's card already named this exit).
- Real states inside the random curve: on Qwen, the SFT updates look like noise to the share; part 3 reports that.

**Control first.** Stage 0, alone: the do-nothing change through the split path (share 1) and the unchanged model as
the fine-tuned one (K00 = K01 = K0), as in P3.9. Stage 1: the built change on one row set; early stop if it does not
read above that row set's random change at the same size (if it reads at or below random, nothing after it is
readable as re-gating). The built change can fail if Γ, the function that decides which neurons are on, does not
switch off neurons whose input weights were zeroed; that is what it tests.

**The numbers.**
- Built change share per row set, three choices of the switched-off set. Expected near 1 (do-nothing reads 1; filed
  value set from stage 0).
- Random curve: share at three sizes spanning the real states' output change (16-row KL about 0.3 to 2.0), three
  draws each, per row set. Expected about 0.81-1.06 from P3.9's one draw, falling with size.
- Per real state (already read, $0): its share minus the random curve at its own seed-0 output change, with the band
  from the three draws' spread. Expected near 0 if real updates are noise to the share, about +0.05 if finding 3 holds.

**Early guards.** Each read logs its share and the projected time; the size search logs the achieved KL against its
target, now matched within a factor 1.1 rather than 1.5 (two extra evaluations).

**What is saved.** One JSON per read with the four K cells, the achieved KL and the switched-off neuron indices; no
weights. A few MB in all.

## Cost

60 reads (5 row sets x (3 built + 9 random)), no fine-tuned state fetched. At about 3 min a read plus 7 min to fetch
the decomposition: about 3.1 h on one Vast H100 NVL at $3.45/h, about $11-13, or two pods for half the wall time
at the same cost. Not gated (under $60 and 6 h). It may also fit the cluster's `normal` QoS on an L40 for $0, since
no W&B fetch is needed (the Qwen line may use it when the queue is empty).

## Not covered

A built change that switches neurons on, which is closer to elicitation than switching off; masks are 0 to 1, so it
cannot be built the same way. Attention matrices, which the frozen-neuron decomposition does not cover. The RL updates.
