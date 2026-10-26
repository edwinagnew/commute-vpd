# Does the cue-free form of the fixed-shift hypothesis hold on a fine-tune with no trigger? (E36, analysis)

*2026-10-26, lead agent (overnight). Plan and design check: `reports/2026-10-26_plan_regating-without-cues.md` (E36,
Amendments 1 and 3). Code: `experiments/regating_spread/run.py` and `amend1.py`, shared steps in
`src/commute/spread.py` and `src/commute/cuefree.py`. Follows E35, `reports/2026-10-26_analysis_regating-without-cues.md`.
Exploration, provisional. An independent check (a fresh Opus agent, 2026-10-26) found the numbers and scoring right and
the first headline too strong; this version takes its points. Words as in `GOALS.md` → Terms.*

We fine-tuned the model on text with no trigger and no target token and asked whether Γ's switching at each position
can be predicted from the original model's importance values there. On lowercased Pile text, the positions with the most
similar original values predicted it better than every other rule. With the planned sample of fitted positions it fell
below the filed bar after the first ten steps; with sixteen times as many it recovered three-quarters or more at every
step. For `GOALS.md` the restatement without cues, inputs the original Γ treats alike get alike changes, holds on the toy
tasks and on this one synthetic style of running text.

## The answer

Yes, once enough positions are compared. On a fine-tune of lowercased Pile text, each position was given the mean
change of the eight training positions with the nearest original importance values. Fitted on 16,384 positions this
recovered 0.72 of Γ's switching at step 10, then 0.54 and 0.51, under the filed 0.6. Fitted on all 262,144 it recovered
0.89, 0.78 and 0.76. Every other rule did worse, and the current token added nothing.

## What we were trying to test

**The setup.** A fine-tune of all 24 decomposed matrices on the next-token loss at every position, by Adam at rate
3e-5 with batches of eight 512-token rows, each row used once. In the *do-nothing* state the update goes into Δ and the
subcomponents stay as released; Γ is frozen but reads the fine-tuned model, and Δ is left out of the CI-masked model.
Everything is read at every position of 64 held-out rows:

- the *gain*: the fine-tuned model's mean next-token log-probability minus the original's;
- the *switching share* $r$: how much the CI-masked model's mean log-probability rises when Γ reads the fine-tuned model
  instead of the original, divided by the gain. 1 means switching alone reproduces the gain;
- each rule's *recovery*: the rise its predicted importance values give, as a share of the rise Γ's own values give.
  0 is the original model's values everywhere, 1 is Γ's values on the fine-tuned model.

**The rules.** Each gives every position $x$ its own original values $g_0(x)$ plus a predicted change, clipped to
$[0, 1]$. The change is fitted at 16,384 random positions of the first 64 training rows:

- *one change everywhere*, the mean change: the original fixed-shift hypothesis;
- *one change per current token*, the mean change at fitted positions holding the same token, or the overall mean for
  tokens seen fewer than 20 times: the toy tasks' "cue" carried over to running text;
- *the 8 most similar positions*, the mean change of the eight fitted positions whose original values are nearest
  $x$'s: the cue-free form E35 supported;
- *k-means groups* of the original values, at 8 to 512 groups, with random groups as their null;
- *a line per subcomponent*, $a\,g_0 + b$ fitted per subcomponent.

**The two tasks.** The planned task was code-like text: the 5% of Pile rows richest in code punctuation. That
fine-tune did not learn, so a second was run on a style instead: Pile text decoded, lowercased and re-encoded, so
capitals never come. Its training rows are used once each over 300 steps; it is read at steps 10, 100 and 300.

## What we expected, and why

- **Switching carries at least 0.3 of the gain**, low-medium confidence: on toy tasks the model already does it carried
  0.5 to 1.1, but a fine-tune with no trigger might work differently.
- **One change everywhere carries less than half**, because positions differ as much as copy-task keys did.
- **The 8 most similar positions recover at least 0.6, and beat one change everywhere by 0.2**, medium for the
  code task and low for the lowercase task after the code run's 0.23 to 0.34.
- **They do at least as well as one change per current token**, within 0.1.
- **The line per subcomponent beats one change everywhere by 0.1**, from its 0.96 to 1.02 on E35's few-cue tasks.
- **512 groups come within 0.1 of the 8 most similar positions.**

## What we observed

### How much of the gain does switching carry, and what does that show?

**Reading it.** Most of it, but that alone shows little. On lowercase text the switching share was 1.05 at step 10, 0.90
at step 100 and 0.79 at step 300, while the gain grew from 0.07 to 0.13 nats per token. The code-like fine-tune, which
learned nothing and made held-out text worse, gave a share of about 0.8 as well. So a share near 0.8 is what any small
update of these kinds gives here; it does not by itself show that the fine-tune re-weighted subcomponents to learn.

### Does the change follow the original values, the current token, or neither?

![arms](../experiments/regating_spread/figures/amend1/arms.png)

**Reading it.** The original values predict it best, though only about half of it after step 10. The 8 most similar
positions, green, recover 0.72 at step 10, then 0.54 and 0.51, below the filed 0.6 at both later steps. One change per
current token, blue, recovers 0.24 to 0.32, and one change everywhere, grey, 0.03 to 0.07. The token rule is a weak
baseline: only 90 tokens had enough fitted positions, and every other position got the overall change. A third of the
8 nearest positions hold the position's own token, far above the few percent expected by chance, so the nearest
positions are strongly but not wholly the same token.

### How many groups does the switching need?

![clusters](../experiments/regating_spread/figures/amend1/clusters.png)

**Reading it.** Many. Recovery rises with the number of k-means groups, from 0.12 to 0.16 at 8 groups to 0.44 to
0.64 at 512, and stays below the 8 most similar positions at every step. Already at 32 groups the original values beat
the 90 token groups, 0.32 to 0.45 against 0.24 to 0.32, the fairer comparison of the two. Random groups match one
change everywhere to within 0.002. There is no small set of conditions here.

### Is it one fixed line per subcomponent?

**Reading it.** No. The line recovers 0.39 at step 10, 0.14 at step 100 and −0.01 at step 300, against one change
everywhere's 0.03 to 0.07. On the code-like run it recovered about half of a switching that carried no learning. The
slopes barely move from 1: the median among subcomponents whose values vary is 0.94 to 0.96.

## What we learned

In the scored run, with 16,384 fitted positions, the cue-free form held at step 10 and failed at steps 100 and 300 by
the filed rule. Amendments 2 and 3 showed that miss was sampling: with all 262,144 positions the rule recovers 0.76 to
0.89 of the switching at every step. So on running text with no trigger, the change of Γ's values at a position is
largely a function of the original model's importance values there, which is the restatement without cues. The
current token adds nothing to it, and the original values predict the change better than a few groups or a fixed rule
per subcomponent. A tenth to a quarter of the switching remains unexplained, more of it later in training. A high switching share is not evidence of learned re-weighting
on its own, since an update that learned nothing gave the same share.

## How compelling is the evidence

- **Controls.** At every step a zero change left the masked model's log-probability exactly where it was, and Γ's own
  values on alive subcomponents reproduced the full switching to within 0.000005. The masked model's log-probabilities
  were computed in float32; the target model's and Γ's values come from bf16, which does not matter at these gains.
- **Nulls.** Random groups land within 0.002 of one change everywhere at every step.
- **Held-out positions.** Every rule is fitted at positions of the first 64 training rows, which the fine-tune saw in
  its first eight steps, and scored on 64 held-out rows, 32,704 positions. Held-out rows repeating training rows were
  not checked.
- **Coverage.** One fine-tune seed, one style, one 67M model. Lowercasing is a style spread over every position but
  synthetic in form; the more natural task, code-like text, did not learn.
- **Blind spots.** The distance weighs every alive subcomponent equally. The nearest rule draws from 16,384 fitted
  positions; more might raise it, and that was not tested. The token rule covers only 90 tokens, and how many held-out
  positions they cover was not counted. Patterns are written at every position at once, so effects passing between
  positions through attention are included, not separated. No predicted pattern was checked for faithfulness under the
  adversary.

## Surprises

1. **The cue-free form held only early.** We predicted the nearest rule would recover at least 0.6. It recovered 0.72 at
   step 10, then 0.54 and 0.51, missing at steps 100 and 300.
2. **The line per subcomponent collapsed with training.** We predicted it would beat one change everywhere by 0.1. It
   did at step 10, by 0.32; it missed by 0.002 at step 100 and fell to −0.01 by step 300.
3. **The code-like fine-tune did not learn.** Its training loss showed no trend and held-out text got worse by 0.05 to
   0.07 nats per token. The rows chosen were mostly LaTeX and maths. Why is untested: the same recipe learned lowercase
   text.
4. **With all positions the rule cleared the bar** (Amendment 3). We predicted it would stay under 0.7 at step 300;
   it recovered 0.76, so the first surprise above was a sampling shortfall.

## Amendment 2: is the shortfall sampling, and does the token add anything?

*Plan: Amendment 4 of the plan. `experiments/regating_spread/amend2.py`, predictions filed 2026-10-26 18:05
(`results/predictions_amend2.json`, sha256 `eab5cad2…`) before the run; results `results/regating_spread_amend2.json`.
RunPod 4090, 58 s, $0.09.*

The lowercase fine-tune was regenerated, and the 16,384-position nearest rule reproduced Amendment 1 exactly at every
step. Two questions came from the independent check. Does the nearest rule miss because it drew on too few positions?
And does restricting the neighbours to the same token help, as restricting to the cue did on one toy state?

**Reading it.** Partly sampling, and the token adds nothing. With four times as many fitted positions, 65,536, the
nearest rule recovers 0.80, 0.67 and 0.58 at steps 10, 100 and 300, against 0.72, 0.54 and 0.51. That clears the filed
0.6 at step 100 but not at step 300, where the gain was 0.08 against the filed 0.1, so the sampling prediction fails as
filed while the rule keeps improving with more data. Drawing the eight neighbours only from positions with the same
token changes recovery by 0.02 or less at every step. The token rule itself covers only 38% of held-out positions; the
rest fell back on one change everywhere.

| step | nearest, 16,384 positions | nearest, 65,536 | nearest within token, 16,384 |
|---|---|---|---|
| 10 | 0.715 | 0.801 | 0.699 |
| 100 | 0.543 | 0.671 | 0.534 |
| 300 | 0.505 | 0.581 | 0.507 |

| id | claim | rule | read | verdict |
|---|---|---|---|---|
| A0 | the 16,384-position rule reproduces Amendment 1 | within 0.02 at every step | exact | pass |
| S | more positions add at least 0.1 at step 300 | R_near65 ≥ R_near16 + 0.1 | +0.076 | fail, as predicted |
| W | restricting to the token adds at most 0.05 | R_near16 ≥ R_near16_tok − 0.05 | −0.002 | pass, as predicted |

So the filed "all fail" at step 300 is in part a sampling shortfall, and the token carries nothing the original values
do not. Whether more positions still would close the gap is not known: the curve had not flattened.

## Amendment 3: where does the sampling curve end?

*Plan: Amendment 5 of the plan. `experiments/regating_spread/amend3.py`, predictions filed 2026-10-26 18:14
(`results/predictions_amend3.json`, sha256 `af874f67…`) before the run; results `results/regating_spread_amend3.json`.
RunPod H100, 85 s, $0.30.*

The lowercase fine-tune was regenerated on an H100, and the 16,384- and 65,536-position rules reproduced Amendment 2
within 0.008 at every step. The same nearest rule was then fitted on all 262,144 positions of 512 training rows.

**Reading it.** Something unexpected, in the hypothesis's favour: the curve kept rising, and with the full sample the
rule recovers most of the switching at every step. With 262,144 positions it recovers 0.89, 0.78 and 0.76 at steps
10, 100 and 300, and 0.90, 0.79 and 0.78 with 32 neighbours. We predicted it would still be under 0.7 at step 300.
Each fourfold increase in positions added about 0.08 to 0.18, so the filed "all fail" at step 300 was a sampling
shortfall, not a sign that the change depends on something the original values do not show. A tenth to a quarter of
the switching remains unexplained, more of it later in training.

| step | 16,384 positions | 65,536 | 262,144 | 262,144, 32 neighbours |
|---|---|---|---|---|
| 10 | 0.719 | 0.808 | 0.886 | 0.902 |
| 100 | 0.542 | 0.673 | 0.776 | 0.789 |
| 300 | 0.508 | 0.584 | 0.762 | 0.776 |

| id | claim | rule | read | verdict |
|---|---|---|---|---|
| A0 | the smaller samples reproduce Amendment 2 | within 0.02 at every step | within 0.008 | pass |
| M | still rising at step 300 | R_near262 ≥ R_near65 + 0.03 | +0.178 | pass, as predicted |
| B | the rule recovers at least 0.7 at every step | R_near262 ≥ 0.7 | 0.886, 0.776, 0.762 | **pass, reversing the filed lean** |

By the filed decision tree, "B holds" reads: with enough fitted positions the rule recovers most of the switching on
running text, and the filed miss at step 300 was sampling; evidence for restating the hypothesis without cues.

## What next

1. **Done as Amendment 3:** with all 262,144 positions the rule reaches 0.76 to 0.89 at every step. Next on this: what
   the remaining tenth to quarter depends on, and whether the curve flattens with yet more positions.
2. **The filed branch: what else the change depends on.** Γ reads every position's input, so the next place to look is
   Γ's input at the position or at neighbouring positions; a rule fitted on those, against the nearest rule here. A
   new design: it goes to the user through the pending card.
3. **A fine-tune of a real behaviour that learns**, or the larger model's post-trained sibling, before any restatement
   is adopted. A pending-decision card in the log holds this choice.

## Scorecard

E36 Amendment 1, the lowercase run, filed 2026-10-26 17:39 (`results/predictions_amend1.json`, sha256 `48b37e56…`) before
the run, scored at step 300; steps 10 and 100 read by the same rules.

| id | claim | rule | step 300 | verdict | steps 10, 100 |
|---|---|---|---|---|---|
| C0 | zero change and own values check | as filed | exact | pass | pass, pass |
| G | the fine-tune learned and switching carries some of it | gain ≥ 0.02, r ≥ 0.1 | 0.130, 0.79 | pass | pass, pass |
| P_r | switching carries at least 0.3 of the gain | r ≥ 0.3 | 0.79 | pass | pass, pass |
| P_shared | one change everywhere carries under half | R_shared < 0.5 | 0.034 | pass | pass, pass |
| P_near | the 8 most similar positions recover ≥ 0.6 and beat one change by 0.2 | both | 0.505 | **fail** | pass, **fail** |
| P_token | they come within 0.1 of one change per token or beat it | R_near ≥ R_token − 0.1 | 0.505 vs 0.243 | pass | pass, pass |
| P_fix | the line beats one change by 0.1 | R_fix ≥ R_shared + 0.1 | −0.014 | **fail** | pass, **fail** |
| P_K | 512 groups within 0.1 of the nearest rule | R_clu_512 ≥ R_near − 0.1 | 0.442 vs 0.505 | pass | pass, pass |
| N | random groups match one change | within 0.05 | 0.0007 apart | pass | pass, pass |

By the filed decision tree, step 300 falls in the branch "all fail": the nearest rule under 0.6, the token rule under
0.6 and the line under its bar, read as "the change at a position depends on more than its own original values; next
space: Γ's input". The ordering of the rules is a secondary reading.
E36, the code-like run, filed 2026-10-26 17:20 (`results/predictions.json`, sha256 `f93a0727…`): gate G failed at both
steps, gains −0.052 and −0.065, so every prediction is unscorable and P_r reads fail by its rule. The filed tree read
"G fails at both steps" as "re-gating carries little", which does not apply: switching carried 0.77 to 0.82 of the
change. The lowercase amendment is a decision taken after that run (plan, Amendment 3).

## Full tables

Shares of the switching, at every held-out position. Lowercase run first, code-like run below it.

| run, step | gain | r | one change | per token | 8 nearest | line | 8 groups | 32 | 128 | 512 | random 128 | nearest share same token |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lowercase, 10 | 0.070 | 1.05 | 0.068 | 0.318 | 0.715 | 0.391 | 0.158 | 0.449 | 0.541 | 0.644 | 0.068 | 0.33 |
| lowercase, 100 | 0.111 | 0.90 | 0.042 | 0.246 | 0.543 | 0.140 | 0.119 | 0.337 | 0.404 | 0.500 | 0.040 | 0.33 |
| lowercase, 300 | 0.130 | 0.79 | 0.034 | 0.243 | 0.505 | −0.014 | 0.118 | 0.320 | 0.371 | 0.442 | 0.034 | 0.33 |
| code-like, 10 | −0.052 | 0.82 | 0.009 | 0.053 | 0.344 | 0.492 | 0.030 | 0.044 | 0.063 | 0.142 | 0.009 | 0.35 |
| code-like, 100 | −0.065 | 0.77 | 0.003 | 0.032 | 0.231 | 0.501 | 0.020 | 0.031 | 0.045 | 0.086 | 0.004 | 0.35 |

Lowercase: the original model's held-out log-probability −3.146 nats per token, the masked model's −3.445; 90 tokens
had at least 20 fitted positions. The share of nearest positions holding the same token depends only on the original
model's values, so it is the same at every step. 88% of alive subcomponents vary at the fitted positions; the median slope among them
falls from 0.96 to 0.94 with training, and under 1% have a slope below 0.5. The fine-tune's held-out cross-entropy was
3.075, 3.035 and 3.016 at the three steps, its KL from the original model 0.15 to 0.29. Code-like: the selected rows
were the top 5% of 20,480 by the share of tokens with code punctuation, a cut of 0.30, mostly LaTeX and maths; held-out
cross-entropy 1.648 and 1.661, training loss without trend.

## Not run

Seeds beyond 0; more than 65,536 fitted positions or other numbers of neighbours; held-out rows repeating training
rows; a rule on Γ's
input or on neighbouring positions; any distance but Euclidean over all alive subcomponents; the rules' faithfulness under the adversary; a domain fine-tune that learns; SGD or LoRA updates;
a larger model.

## Provenance

- Code-like run: `experiments/regating_spread/results/regating_spread.json` (sha256 `1b45dcb7…`), log
  `results/large/full.log`; RunPod 4090 pod `01M5WK51FH0VXCAD1HRZKCT61T`, 17:21–17:22, run 77 s, on a pod shared with
  E35 Amendment 1, $0.28 for the pod.
- Lowercase run: `results/regating_spread_amend1.json` (sha256 `dc631b3d…`), log `results/large/amend1.log`; RunPod
  4090 pod `01M5WMD4MVMXKVM315293SSZ3H`, 17:39–17:41, run 90 s, $0.12; GPU memory read 17.4 GB.
- Figures: `experiments/regating_spread/figures/amend1/{arms,clusters}.png` and `figures/{arms,clusters}.png` for the
  code-like run, from `plot.py`.
