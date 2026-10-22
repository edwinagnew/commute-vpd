# Decision: re-gating's working hypothesis, and commute as a second line (2026-10-22)

**User, 2026-10-22.** (1) Adopt the restated mechanism hypothesis as a working hypothesis. (2) Re-gating stays the
main line; the commute question (Q1) runs alongside it as a second line, in its own session.

**The working hypothesis (H′).** After a narrow fine-tune, the part of the gain that comes from Γ switching existing
subcomponents up or down at the trigger is carried mostly by one fixed change to Γ's importance values: the same
change for every input, aimed at particular subcomponents. Each input's own variation around that change adds a few
hundredths of the effect. It replaces H's part (c), "enough on its own", which failed as filed (E18 Amendment 3) and
stays failed.

**Why it is supported, not only "the leak lost".** H′ made four predictions that could each have failed, and each held:
1. One pattern, the same for every input, reproduces most of the effect: 0.70 to 0.86 of what each input's own values
   give, on three tasks (E18, Amendment 1), and 0.74 on two small fine-tunes, one by SGD (E24,
   `reports/2026-10-22_analysis_regating-small.md`).
2. Which input the values were computed on does not matter: another input's values give 0.89 to 1.00, and each input's
   gain follows its own context, not the donor's (E18 Amendment 3, `amend3_donor.png`).
3. The amounts only work on the subcomponents that received them: shuffled among the alive subcomponents, the change
   gives −0.005 to +0.003, and each half (turned up, turned down) gives nothing shuffled (Amendments 2 and 3).
4. Inputs' own variation adds little: beyond an averaged pattern, 0.01 to 0.05 of the effect (Amendment 4).

**Why adopt rather than stay agnostic.** The evidence is not balanced between H′ and its alternative, and a working
hypothesis picks the next experiments by its predictions. It stays revisable: it is not to be written as a finding.

**What would kill it.** On a new task (the other ladder tasks, the key-value tasks): a fixed change recovers under 0.5
of the own values; a shuffled change recovers over 0.2; another input's values recover under 0.5. Or it fails on a
larger model with a real post-trained sibling (Q9).

**Limits.** Three narrow tasks on one 67M model; the trigger position only; on the word tasks the change lowers the
target away from the trigger, so it is fixed at the trigger, not free of context.

**Two lines.**
- *Re-gating (main):* test the kill conditions on new tasks; what the changed subcomponents do; faithfulness under Γ's
  switching (the named next step).
- *Commute (second):* a second seed for Adam component training at steps 100 and 1,000, and r2 t275 where nothing
  passes cleanly; why SGD fails at step 1,000 where Adam passes at the same size (E21); LoRA trained with SGD (never
  run: every LoRA so far used Adam); and a map suggested by H′: subcomponents frozen, Γ's values shifted by one fixed
  change.
- Separate sessions; only one applies the log queue at a time; re-read shared files before editing.
