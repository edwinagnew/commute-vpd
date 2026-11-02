# Review of the E66 plan: can the interchange patch put a causal label on the 67M fine-tunes?

Fable, 2026-11-02. Adversarial review of `reports/plans/2026-11-02_plan_e66-interchange-67m.md`, written by another
agent; advice only, nothing written into the plan. Read: `GOALS.md` Terms, the merge plan, the E65 advice, the
method-table analysis and `table.json`, E62's plan and TRIED rows, `decisions.md` rows 81, 112, 114 and 118,
`src/commute/{interchange,steering,perturb,kv,copying,neurons}.py`, `experiments/SAVED.md`, the saved-state folders
under `~/commute-data/results-large/`, and the per-block steering shares filed in
`experiments/baseline_neurons/results/baseline_neurons.json` and `experiments/keydependent_elicit/results/stage2.json`.
No model was run.

## Recommendation in one paragraph

Not as written. The plan's two readings are both blind to the one kind of reuse the 67M has shown so far, and the filed
numbers say that is where the candidate reuse states' updates sit: steering already carries 0.75 to 0.80 of the copy
maps' gain at block 2's input and 0.38 of the half-known map's, so a patch at block 2 or 3 reads the control's value
on them and the label "new" falls out of the blind spot, with the neuron reading agreeing because it cannot see
attention either (findings 1 and 2). The patch ratio is also a sharpening meter calibrated on a sharpening fine-tune
(finding 3), and the "new" side of two of the three predictions is more likely void than low (finding 5). Run a
feasibility probe first, one debug job of about ten minutes on W0, the three built controls, E62's six states and the
eight copy and random-map states, reporting l*, the per-block response, the patch ratio beside the state's own
off-map sharpening, and the units share with an alignment-matched null; file row predictions only after those
numbers. What would change this: if the probe shows R(W0) at block 1's input over 0.5 nats on the key maps (so l* = 1
and the blind spot is block 0 alone) and the copy maps read above the write control there, the plan's design holds
and stage 3a can go as filed with findings 3 to 6 applied. Either way the plan should say plainly that its best
outcome for VPD is a tie with update size and a win over the prior only (finding 7).

## Findings, ranked

### 1. The patch is blind exactly where the saved reuse candidates' updates sit

The plan reads T at l*, the earliest of blocks 1 to 3 where W0's own response to the context swap is at least 0.5
nats, and says T reads the control's value for an update wholly upstream of l*. The filed steering shares say where
the updates are. Steering adds the state's mean residual shift to W0 at one block's input; its share by block is the
part of the gain already present in the residual there. Copy maps (`ident32*`): 0.11 at block 1's input, 0.75 to 0.80
at block 2's, 0.16 to 0.27 at block 3's. Half-known (`elicit_s0/1`): 0.11, 0.38, 0.29 to 0.33. Ciphers: 0.03 to 0.05,
0.07 to 0.12, 0.10 to 0.23. So three quarters of the copy maps' gain is written before block 2. Unless l* = 1, the
patch replaces the state's residual at the answer position with W0's and reads a state whose update has mostly been
overwritten: T near 1, "low".

On the reference row it is worse. In E62's cue window ("A B ... A M", predict B) B's identity sits at B's position
and reaches M only through block 2's attention (head (2,4)). A patch at M alone at block 1's or 2's input carries no B,
so R(W0) there is near zero and l* is 3 for that map. E62's own reading of the fine-tune is a redirected query of
head (2,4), upstream of block 3's input. So T on the copy map reads the reuse mechanism the reference labelled only
if the state also amplifies the head's output in block 3. P1's 65% lean cites "6-10 log-odds against 1.6 in W0": that
is the pair-broken edit (first A overwritten) in E62's analysis, not the B-absent edit this plan uses, and it is an
unpatched contrast, not a patched response.

**Fix.** Stage 0 reports l* and, for every state, the filed steering share at block l*'s input. T is scored only where
that share is under 0.5 ("update at or after l*"); otherwise the state's T is "blind", never "low". Report T at every
block as planned, and put the per-block R(W0) and T figure first.

### 2. Both readings miss attention-carried reuse and agree on "new" there

U mean-ablates MLP neurons and is blind to gain carried by attention; T at l* is blind to an update upstream of l*. A
state that reuses W0's copying head by changing its query (E62's reading of the copy map) gives T near 1 and U low,
and the join rule then labels it "new" with both readings agreeing. The plan requires agreement to label; agreement
from two shared blind spots is not independence.

**Fix.** "New" needs a third cheap check against attention: E62's head mean-ablation share (`copying` has the code) under
0.2 on the state, or the rule in finding 1 (update not upstream of l*). Without one, report "patch low, units low" and
do not write "new".

### 3. T is a sharpening meter, and its positive control is a sharpening fine-tune

In log-odds, adding a key-conditioned constant to the target's logit leaves the response to a residual swap unchanged,
so an additive lookup reads T = 1 exactly, as the plan says. A drop in effective temperature by a factor tau scales
every log-odds difference by about 1/tau, so the sharpening control at temperature 0.5 reads T near 2 and
hi = n0 + 0.5 (p+ − n0) sits near 1.5. "Reuse" then means "the state scales W0's context response by 1.5 or more at
the trained position". Any fine-tune that makes the model confident at the marker can do that, mechanism or not; a
reuse fine-tune that adds a bias on top of W0's guess cannot. The final layer norm pulls the other way: a large update
grows the residual at the answer position and shrinks a fixed swap's effect, so T falls with update size. The plan
lists "T follows ‖δW‖" as an outcome but gives it no rule.

**Fix.** Read T_off on the same contexts for an off-map target (another key's target, or the next token at a non-marker
position) and score T / T_off, so global sharpening cancels. Use the amplify edit, reuse by algebra, as p+ for T, and
report the sharpening fine-tune beside it as a reference. File the size null: Spearman of T against ‖δW‖ over all 47
states plus E62 Amendment 2's size-scaled copies (a multiplication, $0); over 0.5 in magnitude means no labels.

### 4. Few rows will be labelled, and U's null does not control what U reads

A label needs T high and U high, or both low. Finding 1 and 3 make T low on most states. U is likely high on both
sides: gradient descent on the target's cross-entropy puts the update on neurons that are active at the marker and
whose output already aligns with the target's unembedding, and that is the grad × activation ranking at W0. The null
draws match activation decile only, not alignment, so a cipher can read U well above its null from W0's output
geometry alone. The likely table is mostly "units only" and "patch only", which scores nothing, after about 300 lines
of code, two reviews and an hour of debug time.

**Fix.** Match the null on activation decile and on the decile of |down_proj column · unembedding(target)|. File the
fraction of states that must be labelled for the run to count (say, both sides of at least two rows); below it the
report is "no label from this pair of readings", and E65's card returns at once.

### 5. The "new" sides of P2 and P3 are more likely void than low

A "low" T needs R(W0) of at least 0.5 nats: W0 must already respond to the context swap on the new side's metric. For
the text cipher, W0 does not know the cipher; the only source of a response to a ciphered context on
log p(pi(x)) − log p(x) is W0 copying pi(x) from earlier in the row. For a key-map cipher, W0's log-odds of a random
word at the key position moves across contexts mainly when that word is in context. So the new side gets a label only
where induction fires on the target, and the void rule, which the plan rightly has, removes the row otherwise. P2
("both ciphers read new") and P3's cipher half cannot pass on a void side, and the 55% and 35% leans do not say so.

**Fix.** Build the new side's pairs explicitly: x_hi has the target word earlier in the context, x_lo does not, chosen
from W0 alone as planned, and say that this is what the new side's T measures. File P2 and P3 with the void case as a
miss, and lean accordingly.

### 6. The lookup's pair recipe is undefined and off its training layout

"The lookup uses the key-map rule at its marker" points at `kv.windows`, which writes the marker at half − 1 and the
key at half ("M x"); E62's cue is key then marker ("A M"), and the lookup was trained on that layout. The 64 fresh
contexts per key would present the lookup with a window shape it never saw.

**Fix.** Build the lookup's pairs from `copying.cue_windows` output, rank by W0's log-odds of word(A) at M, top quarter
against bottom quarter, same windows as the copy map.

### 7. VPD cannot win on any planned row except against the prior

Where labels agree with the build, r separates and so do size, distance and MLP share on most rows; the plan says so.
E62's pair and the prior-matched row are size-separable as the states stand. The one row where every cheap method
fails, the class map against its shuffled twin, is the row T is blind to by design. So the realistic best is "r ties
size, beats the prior", and the realistic worst is "r follows something other than the label". The user asked for a
test that can show VPD winning or losing: this one can show losing.

**Fix.** Read T and U on size-scaled copies of every labelled pair (E62 Amendment 2's scaling) so size is removed as a
labeller before 3b retrains anything; write the best case into the plan; and name the key-swap patch for the class
map as the next design if labels appear, since it is the only row with a win in it.

### 8. Smaller points

- `open_pair_instruments/` holds rank-1, copy and cipher seeds 3 to 7 only; "seeds 0-2 also saved" is wrong for those
  maps (`open_pair/` holds half-known and cipher seeds 0 to 2). The 41-state count for stage 3a is right; E62's six
  make 47. `performed_anchor` has the six step-300 half-precision files. Debug's 2 h wall fits the hour.
- K is fixed where "W0's rank-1 log-odds falls by at least 1 nat", but W0 does not perform the rank-1 map (log prior
  about −5); a 1-nat fall from a low base is noise-sensitive. Fix K on the half-known map, which W0 performs at
  P 0.01 to 0.1.
- Text maps borrow the largest key-map n0. A write of one token's bias at block 3 on the text pool is a few lines and
  gives the text maps their own control.
- The amplify edit is the only state that is reuse by construction; the sharpening fine-tune is "near-known" by its
  training target, not its route. The plan's limit paragraph says this honestly; finding 3's swap makes it the control.
- A tie at a bar is "between" in `interchange.label`; the edge table says a tie at U's bar is "low". Make them agree.
- The stage 2 stop "P1 misses: read no row" would fire on finding 1's blindness and end the run for the wrong reason.
  With finding 1's rule, a blind T on the copy map is reported, not a miss.

## Options

- **A. Feasibility probe (recommended).** One debug job of about 10 min: stage 0 with l*, per-block R(W0) and the
  filed steering share at l* for every saved state; the write, amplify and sharpening controls; T, T / T_off and U
  (alignment-matched null) on E62's six states and the eight copy and random-map states; T against ‖δW‖ with E62
  Amendment 2's scaled copies. Predictions filed for the probe's controls only. A decision card after it says whether
  the 41 rows get filed rules or E65 returns. Code about 200 lines, reused by stage 3a if it goes.
- **B. Go with findings 1 to 6 applied as one run.** Saves a round trip if the probe would have passed; costs the hour
  and two reviews if it would not.
- **C. Design the class-map key-swap patch instead.** The only row where VPD can beat every cheap method; new design,
  through `/design`.
- **D. As written.** Not recommended: the blind spots label "new" by construction on the candidate reuse states, and
  the "new" sides of two predictions are likely void.

## Not run

No model, no job. The per-block shares are read from filed JSON; l* and R(W0) are not measured anywhere yet and are
the probe's purpose. The claim that B reaches M only through block 2's attention follows E62's prefix-score reading
of head (2,4) and the 4-block model; a block-0 or block-1 head that summarises the context at M would raise R(W0)
there, and the probe would show it.
