# When does a fine-tune reuse a mechanism the original model had?

*definition-lead, 2026-11-09. For you to adopt, change or drop. Nothing here has run. It replaces the frozen-body page,
which you rejected. The survey table, the design check and what was not read are in
`2026-11-09_plan_general-definition-appendix.md`.*

**The answer.** None of the 30 sources I read says, in terms of what happens inside a model, when a fine-tune reuses a
mechanism and when it builds one. The closest formal account covers one network and one algorithm; the closest question
is Prakash et al.'s, answered by measurement. So part of any definition here is ours. I recommend this one: a fine-tune reuses a mechanism when the finest account of how it
produces the new behaviour runs through intermediate values the original model already computes and already uses, at
the same place and in the same encoding. Otherwise it built something new.

## What the literature gives

- **Reuse is never defined formally.** Raghu et al. 2020 contrast feature reuse with rapid learning in one informal
  sentence, p. 2-3, then measure it by freezing layers and by representation similarity.
- **Prakash et al. 2024 ask the right question.** They ask whether fine-tuned models "use a different or the same
  circuit, i.e. with exactly the same group of heads" (p. 5), and whether its parts keep their function. They set no bar
  for "same", and their base model already does the task.
- **Jain et al. 2024 define possession and leave reuse open.** Their Definition 1, p. 3, says a model has a capability
  if a linear readout gives it. Definition 2, p. 4, assumes fine-tuning can turn the capability into g applied to it, and
  grades how relevant the capability is. That is about the task, not about what a given fine-tune did.
- **Causal abstraction says what implementing an algorithm is.** Geiger et al. 2025, definitions 5, 31, 33 and 41, cover
  one network and one algorithm. Two networks are not treated.
- **Two definitions do compare two networks.** Lubana et al. 2023, Definition 4, calls two models mechanistically similar
  when they are invariant to the same input changes. Bansal et al. 2021, Definition 1, asks whether one can be stitched
  into the other. The first never looks inside, which you said is not enough, so it is not a candidate below.
- **The open problem is stated in print.** Minder et al. 2025, p. 10, report an "inability to distinguish between truly
  novel latents learned during chat-tuning and existing latents that have merely shifted".

## The setting, for every candidate

Let **M** be a network and **F** a network of the same architecture obtained from M by further training. Let the
**gained behaviour B** be a distribution D over inputs and a rule y, such that F outputs y(x) on D and M does so less
often or never.

## Candidate 1: a capability and a wrapper

*Source: Jain et al. 2024, Definitions 1 and 2. The last clause is ours.*

Let f be a function of the input. M **possesses** f on a set of inputs E if, at some layer, a linear readout trained on
M's training data outputs f(x) for every x in E. F **reuses** f for B if F's output on D equals g(f(x)) for some function g
and the value of f inside F decides that output: changing it at that layer changes F's output as g predicts. F **builds
new** if no such f exists.

- **A mechanism is** a named function the network computes.
- **M has it when** the function can be read out linearly.
- **Strongest objection.** With g unconstrained, a memorised table is g applied to "which input is this", so it reads as
  reuse. Being readable also does not show M uses the value for anything. Jain et al. call the definition "idealized".

## Candidate 2: the same algorithm in the same place (recommended)

*Source: the question is Prakash et al.'s, the formal terms Geiger et al.'s, and the split between what a part does and
what it is used for Anderson 2010's. The step across two networks is ours.*

Let an **algorithm** H for B be a causal model in Geiger et al.'s sense: inputs, intermediate variables and an output, each
a function of its parents, computing y on D. Fix in advance a class of **site maps**: for each intermediate variable, a
place in the network's activations and a map from activations there to the variable's values, for instance a linear map.

A network N **implements** H on a set of inputs E under a site map when, across pairs of inputs from E, setting each
site to the value it takes on the other input changes N's output as H says, in at least a fraction η of cases.

A **mechanism of F for B** is an algorithm F implements on D that is finest: no algorithm adding an intermediate variable
is also implemented by F. A variable **counts** unless it is an invertible copy of part of the input.

M **already has** a variable V under the same site map when both hold.

1. *M computes it.* On inputs from D, the site map read in M gives V's value in at least a fraction η of cases.
2. *M uses it.* On some set of inputs, M implements some algorithm that contains V, with V computed by the same function
   of its parents.

F **reuses** M's mechanism for B if F has a mechanism with at least one variable that counts, and M already has every
variable of it that counts. Otherwise F **built a new one**. A mechanism with no variable that counts, such as a flat
table from input to output, is new.

Inside "reuses" there are two cases: **strengthened**, when M already implements H on D but less often, as Prakash et al.
report; and **wrapped**, when only the last step to the output is new, as Jain et al. report.

- **A mechanism is** the variables an algorithm passes through and the functions that compute them.
- **M has it when** the same values are computed at the same place and M's own behaviour depends on them.
- **Strongest objection.** The class of site maps decides the answer. Sutter et al. 2025, p. 1, prove that with it left free
  "any neural network can be mapped to any algorithm". Fixing it to linear maps can miss a variable stored otherwise.
- **Three smaller objections.** "Finest" cannot be verified, so "reused" is only ever shown at a named level of detail.
  A variable that fine-tuning moved to a nearby place reads as new, which is Minder et al.'s problem restated. A patch can
  succeed through an unrelated route, as Makelov et al. 2023 show.

## Candidate 3: an old lower part under a new top

*Source: Bansal et al. 2021, equation 1 and Definition 1; Raghu et al. 2020 and Yosinski et al. 2014 for the idea. Applying it
to a model and its own fine-tune is ours.*

Take a layer l. Build a network from M's layers up to l, a stitching map from a small fixed class, and F's layers above l.
F **reuses M's computation up to l** if that network does B on D as well as F does. The **depth of reuse** is the largest
such l. F **builds new** if the depth is zero.

- **A mechanism is** whatever the first l layers compute.
- **M has it** by construction, since those are M's layers.
- **Strongest objection.** It says where the change sits, not what a mechanism is. A fine-tune that writes a whole new
  algorithm into its last layers gets full depth. One that strengthens a part in the middle gets low depth, though that is
  Prakash et al.'s central case. Unlike the rejected page, it uses F's own top, so two fine-tunes can differ.

## Recommendation: candidate 2

Your five tests, in order.

- **General.** It names no task, architecture or decomposition. Candidate 3 needs an ordering into layers.
- **About the fine-tune.** The algorithm is F's own, so a second fine-tune gaining the same behaviour by another route gets
  another answer. Candidate 1's possession clause is about M alone.
- **A definition before a test.** It says what a mechanism is and what having one means, then stops. Candidate 3 is a test
  with a name.
- **From the literature.** Every term is someone else's except the step across two networks, marked above.
- **Working.** Given the class of site maps and η, the answer is fixed, up to the objection about "finest".

The main objection to candidate 2 applies equally to candidate 1, which also fixes a class and needs a named function. So
it is no reason to prefer candidate 1. Candidate 2 contains candidate 1's possession test as its first condition, adds that
M must use the value, and closes the memorised-table gap. It is stricter than what decomposition methods can see: new
wiring through a new intermediate variable reads as new.

## What would establish each on a real model

- **Candidate 1.** Train the readout on the original model's data, with a shuffled-label control. Intervene on the readout
  value in the fine-tune and check the output moves as g says.
- **Candidate 2.** Name an algorithm for the behaviour. Fit one site map from the fixed class across both networks. Run swaps
  in the fine-tune on the behaviour's inputs, and in the original on inputs where it uses the variable. Compare with the same
  fit on a same-architecture network with random weights, the control Sutter et al. add. "New" can be shown by exhibiting a
  variable the original lacks. "Reused" can only be shown at the level of detail named.
- **Candidate 3.** Train the stitching map on the fine-tuning data and test on held-out data, with a random lower network as
  control, as Bansal et al. do on p. 4.

## What each would say about our saved fine-tunes

*This describes which records exist. It is not a verdict, and none can be given before you adopt a wording.*

| | Saved results that bear on it | Where it gives no answer |
|---|---|---|
| **Candidate 1** | Case label and repeat label are the only kinds of 17 where the readout passed. Patching that readout flipped at most 0.014 of case labels against a filed 0.50 | Most kinds: the readout step itself fails |
| **Candidate 2** | No saved reading: swaps under one site map across both networks were never run. Switching off one head the original model copies with takes 0.56 to 0.65 of the copy fine-tune's gain, which suggests where to look and decides nothing | All 17 until those swaps run. Lookups and ciphers would be expected to read new |
| **Candidate 3** | None: stitching was never run | All 17, until it is |

## Departures from the literature

- **Jain et al.** The clause requiring the value to decide F's output is ours; they state only that the capability can be
  transformed.
- **Prakash et al.** "Same circuit" becomes same variables at the same sites; they compare sets of attention heads and set no
  bar.
- **Geiger et al.** They define one network against one algorithm. Using a single site map for two networks, the "finest"
  requirement and the "counts" rule are ours.
- **Anderson 2010.** The split is informal there, about brain circuits, and is used here only as the idea.
- **Bansal et al.** They stitch independently trained networks; applying it to a model and its fine-tune is ours.

## What I need from you

Which of the three, if any, do you adopt? I recommend candidate 2. If so, one number comes with it, η, which every source
leaves open; no paper sets it and it should not come from our runs. Once you have chosen, I add the three test pairs and the
proposed runs as a second step.

**Not run:** everything. No test pairs or proposed runs are on this page, by your brief.
