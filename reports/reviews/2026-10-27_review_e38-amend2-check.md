# E38 Amendment 2: independent check of the verdicts and the reading (review)

*2026-10-27, independent agent, read-only. Files: `experiments/recruitment_test/results/{amend2_stage0,amend2,amend2_census}.json`,
`predictions_amend2{,_p8}.json`, `amend2.py`, `src/commute/{recruitment,census,recruit}.py`, parent `stage1.json`.*

I recomputed every verdict from the raw `lo` reads and the raw census counters, and every one matches the code. No
code fault found. The disagreement is with the reading: removing the recruited set costs the fine-tune more than
removing a random set, but the ` http` null's set and the busy set show the same kind of excess. So "the fine-tune
routes its gain through S" is not yet separated from a generic effect of removing pieces that matter at the trigger.

## 1. Verdicts, recomputed (`amend2.json` → `states.*.reads`, T0, T1)

| prediction | recomputed | code |
|---|---|---|
| C0-C4 | all pass (C1, C2 pre, C3 exactly 0.0; C4 KL 10.6-11.2) | same |
| P5 trigger | emoticon 'through' (D 0.232, 0.255; 20/20), ` which` 'through' (0.116, 0.188; 20/20), ` we` 'not' (−0.059, −0.024) → **'through'** | same |
| P5_all | emoticon s1 beats 19/20 → 'unclear', seeds disagree → outside the tree; expected "same classes": **miss** | same |
| P5_null | \|D(null)\| 0.138, 0.100, 0.113, 0.111, 0.059, 0.067 → **fail** (emoticon s1 misses by 0.0001) | same |
| P6 | ρ₀(S) 0.18-0.27 on all six → pass | same |
| P7 | kl_other ratio 0.011-0.037 (bar 3) → pass | same |
| P8 | enriched fraction S / random / control: 0.60/0.074/0.92, 0.80/0.079/0.84, 0.96/0.017/0.74 → pass, all 3 tasks | same |
| m0 < 0.1 (reported) | 0.83, 0.84, 0.55, 0.57 in range; ` we` 0.09, 0.05 **out of range** | same |

## 2. Code

The hook path equals weight removal bit for bit (C1 = 0.0), leaves earlier positions untouched (C2 = 0.0), and the
share arithmetic is right. The census counts and P8 rule match the filed text. Nothing makes the numbers wrong.

## 3. What D supports

D(X) = 1 − (gain that survives removing X from both models) / G. It reads how much of the gain needs X present. It
does not say the gain flows through X's own output. Three numbers point to a generic effect (`amend2.json`, trigger):

- The null's set does nothing in W0 on emoticon (ρ₀ 0.002) yet carries 0.10-0.14 of the gain in W1. On ` which` s0
  D(null) 0.113 is equal to D(S) 0.116. Its overlap with S is small (J 0.05 and 0.025, `stage1.json` J_null_S).
- The busy set's D is 0.78-1.10, above D(S) on every state.
- ` we` has the most target-enriched S (0.96) and negative D.

The random sets are alive pieces that are mostly off at the trigger, so beating them shows only that S matters at the
trigger. **Alternative reading:** the fine-tune's downstream change amplifies whatever the trigger state feeds it.
Any set of pieces active there (S, the null's set, the busy set) then costs more in W1 than in W0. **One cheap test,
using the cached states:** at the trigger in W1, set S's rank-1 outputs back to their W0 values instead of
removing them. That undoes only S's switch-on. Do the same for null \ S. If S's patched share is ≥ 0.1 and the
null's about 0, the gain runs through S. If they are alike, or S's is about 0 while D(S) stays, the effect is the
generic one.

## 4. Census (P8)

It is not true by construction. S is chosen by an importance rise on the task windows, and the census reads W0
next-token counts on Pile. S is also not a set of trigger-token detectors: its members are on at a median of about
24,000 positions, far more than the trigger words occur. Two limits remain. P8 is what P6 already implies:
pieces whose removal lowers the target are the ones W0 uses to predict it. It is also only partly specific. The
` which` set is enriched for ` just` (0.375) and the ` we` set for ` was` (0.478), against random 0.017-0.054. And it
does not separate the 'through' tasks from ` we`. The control is chosen on the same Pile rows, so it shows only that
the measure can register a selected set. P8 says what S is, not whether the fine-tune routes through it.

## Not run

The activation-revert patch; null \ S; D of S inside the ` http` state; a random control matched on trigger importance.
