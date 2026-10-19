# Can the commute question be tested fairly on RL post-training? (feasibility scoping)

2026-10-14. Scoping for the backlog's "RL post-training" entry. The question here is whether an RL test could be
**fair**, not whether RL updates lift. Taken on the user's word: a VPD decomposition of a ~1.7B model exists. Which
model it is, whether it is a base or post-trained checkpoint, and what text it was decomposed on are not known
here, and the answer turns on all three (§4). The released public decomposition is the 67M Pile model this
project uses.

Terms. *Γ*: VPD's causal-importance function (which subcomponents are on, per token). *Δ*: the residual
W − Σ components. *ω* (obstruction fraction): the share of an update δW that no change to the subcomponents can
carry (`theory/commuting_updates.md`). *Weak version*: a simple map Φ(D, δW) gives a decomposition that is faithful
to W + δW without Δ absorbing δW (`GOALS.md`).

## 1. Does RL work at ≤ 1.7B? Yes, depending on what "RL" means

| Kind of RL | Works at this scale? | Evidence |
|---|---|---|
| RLVR on reasoning (GRPO/PPO, math) | From ~1–1.5B; fails at 0.5B | OLMo-2-1B: DPO → RLVR raises GSM8K 59.0 → 68.3, MATH 14.1 → 20.7, IFEval 67.1 → 70.1 ([model card](https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct)). SimpleRL-Zoo reports zero-RL gains across Qwen2.5 0.5B–32B ([2503.18892](https://arxiv.org/abs/2503.18892)); TinyZero's Qwen2.5-0.5B fails to learn reasoning while 1.5B+ learns to self-verify |
| Simple-reward policy gradient (sentiment, token frequency, a classifier score) | Any scale, including 67M | It only reweights samples the policy already produces |

So at 1.7B, RL "works" in the sense practitioners mean, with modest gains. The caveats below matter more than this.

## 2. What a fair test needs, and where it breaks

1. **The update must be small, so it has to start from the decomposed checkpoint.** Base → instruct is SFT + DPO
   (+ RL): a large, multi-stage update, not the small δW `GOALS.md` assumes. An instruction-tuned model can only be
   *the checkpoint that is decomposed*, never the δW. If the 1.7B decomposition is of a base model, the fair RL
   test is RL run on that base, i.e. zero-RL. That works at 1.7B mainly for Qwen-family models, and it is
   precisely where Qwen is anomalous (next item).
2. **Qwen zero-RL is a compromised testbed.** On Qwen2.5-Math, random or incorrect rewards give most of the
   true-reward gain (MATH-500 +21.4 random vs +29.1 true), through GRPO's clipping bias amplifying pretrained
   behaviour; the same rewards do little or harm on Llama and OLMo-2 ([Spurious Rewards, 2506.10947](https://arxiv.org/abs/2506.10947)).
   On Qwen, an RL δW is therefore largely "amplify an existing mode", whatever the reward. That favours the
   commute hypothesis by construction.
3. **The decomposition has to be faithful where δW acts.** If the decomposition was fit on pretraining text,
   RL rollouts (chat format, long reasoning traces) are off its distribution. The frozen Γ may already be
   unfaithful there at δW = 0. **The gate:** measure faithfulness of the *unchanged* decomposition on on-policy
   rollouts first. If it fails, no map Φ can pass, and a failure would be wrongly charged to "RL teaches new
   things" or "decompositions are brittle". This is the single most important fairness check.
4. **RL updates are predicted to be hard for this map in a specific way.** RLVR updates are sparse in parameters
   but nearly full-rank ([Mukherjee et al., 2505.11711](https://arxiv.org/abs/2505.11711)). They lie off the
   principal directions and change the spectrum little, unlike SFT ([Path Not Taken, 2511.08567](https://arxiv.org/abs/2511.08567)).
   If VPD subcomponents concentrate on high-norm directions, expect a high ω for RL. That is a result rather than an
   unfairness, but the yardstick must be the input-weighted ω: the Frobenius ω already failed against W₀ itself
   (TRIED row 2).
5. **Precision.** Path Not Taken attributes part of the apparent RL sparsity to bf16 hiding tiny updates. Diffs
   between consecutive released checkpoints (e.g. OLMo's `step_N` revisions) may be partly rounding. Check the
   dtype, and prefer our own runs with fp32 master weights so that δW is the true update.
6. **Matching RL against SFT/LoRA.** The backlog says "matched ‖δW‖". `decisions.md` ranks functional measures
   above Frobenius ones, and RL reaches a given task gain at smaller KL than SFT ([RL's Razor, 2509.04259](https://arxiv.org/abs/2509.04259)).
   So match on behavioural effect (task KL or reward gain) and report ‖δW‖ alongside. This departs from the
   backlog wording.
7. **RL is the "strengthen" branch by construction.** RL is bounded by what the policy samples. RLVR models beat
   their base at pass@1 but lose at large-k pass@k ([Yue et al., 2504.13837](https://arxiv.org/abs/2504.13837)).
   That makes RL useful as a control: if even RL updates fail to commute, the "teaches new capabilities"
   explanation in `GOALS.md` loses most of its force, which points to brittleness.

## 3. Options

| Option | Fair? | Cost | Blocker |
|---|---|---|---|
| **A. 67M, simple-reward RL** on a behaviour the decomposition already explains (emoticon → `o` via `2.mlp.down:2359`; or German via `3.attn.v:513`), compared with the existing SFT/LoRA arms on that same behaviour | Yes for "do on-policy updates commute?": same model, decomposition, behaviour and harness. No for "post-training as practised" (no chat format, no reasoning) | Minutes on one 4090 or L40, ≪ $10 | None |
| **B. 1.7B, our own RL run** from the decomposed checkpoint (GRPO, GSM8K-style reward, LoRA and full-weight, fp32 masters, checkpoints at steps 1–N) | Yes, if the decomposed checkpoint is not a Qwen base doing zero-RL and gate 3 passes | Rollouts + training for a few hundred steps on one 46 GB L40 is plausible but unbenchmarked; the decomposition's own memory footprint for masked evaluation at 1.7B is unknown | Gate 3; model identity |
| **C. 1.7B, released RL checkpoints**, the OLMo-2-1B pattern: DPO start, RLVR revisions every 20 steps, so a free small-update ladder | The best match for real post-training | No RL compute | Only works if the decomposition is of the RL *start* checkpoint of a model that publishes such a ladder; plus the precision check (item 5) |

## 4. What decides it: three facts about the 1.7B decomposition

1. **Which model, base or post-trained?** A post-trained (SFT/DPO) start makes B, and possibly C, fair. A base
   start means zero-RL, which is fair only outside the Qwen family.
2. **What text was it decomposed on?** Pretraining text only, or chat/reasoning data too? This sets how likely
   gate 3 is to pass.
3. **Can we load and run it on one GPU?** That means forward passes with masks and Γ at 1.7B, which governs
   whether the existing harness (`src/commute/`) ports over.

## 5. Recommendation

Run A now, as the cheap and fair test of whether on-policy updates commute. Treat B or C as round-3 material,
gated on §4 and on gate 3 (faithfulness of the unchanged decomposition on on-policy rollouts) as its first
measurement. A is not a substitute for B: it cannot say anything about chat-format or reasoning post-training.

### Design check for option A (`/design`)

- **Goal served:** `GOALS.md` "if no simple Φ works, new capability or brittleness?". RL is the strengthen-only
  update, so it separates the two explanations. It also extends the canonical updates (LoRA, full-weight) to the
  user's backlog RL entry.
- **Nearest tried:** TRIED row 2 (lift test): ω, Euclidean lift and pull-back on SFT/LoRA updates to the
  emoticon behaviour. New here: an on-policy update to the same behaviour, at matched behavioural effect.
- **Patch or step:** a step, but it presupposes the LoRA case is settled (the backlog's own condition), and row 2
  left the lift's adversarial fragility open.
- **Decisions:** relies on frozen Γ, functional-over-Frobenius and the weak version. Departs from the backlog's
  "matched ‖δW‖" (match on effect instead).
- **What would change plans:** RL ω ≪ SFT ω with the pull-back faithful → commuting is easier for strengthening
  updates, so pursue B. RL ω ≥ SFT ω → the obstruction is not about new capabilities; investigate brittleness
  (locality ρ) before scaling.

Not started: no predictions filed, no code. Waiting for the user's choice.
