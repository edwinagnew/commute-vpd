# Decision: the mechanism reading is the leading hypothesis, not a finding (E18, 2026-10-21)

**User, 2026-10-21:** the mechanism reading is not precise enough, nor supported enough, to back yet. It may be the
leading hypothesis so far, and should be tested more aggressively. E18's main-line call stays open.

**The hypothesis, stated so it can fail (H).** After a narrow fine-tune, the switching effect at the trigger is carried
by one change to Γ's importance values that is (a) the same for every input, (b) specific to which existing
subcomponents receive it, and (c) enough on its own: the part of the updated model's values that varies with the
input adds nothing. A leak is the failure of (c).

**What supports it so far.** A fixed pattern recovers 0.70–0.86 of the effect net of its reference (E18, unscored),
and the unclipped shared change recovers 0.50–0.81 against a reference of exactly 0 (Amendment 1). Random sets and a
different target's change recover nothing or less than nothing.

**Why it is not yet backed.** The reference subtraction assumes the effects add (untested). The random null moved only
the switched-on set, which carries a minority of the recovery. No arm has removed the fixed part and kept the
input-varying part (necessity of (a)), and no arm has tested whether the change alone produces the target out of
context.

**Next.** Amendment 2 (shuffle the whole change) is running. Amendment 3 adds tests that H can fail; see
`reports/plans/2026-10-21_plan_regating-leak.md` → Amendment 3 when written.

## Update after Amendments 2–3 (2026-10-21, writing agent; the user's decision above stands)

H is still not established, and as filed one of its parts has now failed. The record is
`reports/analyses/2026-10-21_analysis_regating-leak.md` → Amendments 2 and 3. What each part stands on:

- **First part, the same change for every input:** survived. Another window's updated values give back 0.89 to 1.00 of
  the effect. Under them, a window's gain follows its own context, not the donor's, on five of six states.
- **Second part, specific to which subcomponents receive it:** survived for the whole change and for each half. Shuffled,
  each recovers nothing or less; the turned-down half is a real switch-off, not a sign flip.
- **Third part, enough on its own:** failed as filed. The updated values with the fixed change removed still give back
  0.16 to 0.31, into the leak band on the first word task. Whether that remainder carries anything specific to its
  input is open. A candidate, unverified, says it does not; the remainder taken from another window would separate the
  two. The failure and the tension go to the user before anything else runs.
