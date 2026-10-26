# Decision: the Γ refit (E15) is reopened, not shelved

*2026-10-21, user, in review of the research log v18 ("it's also ongoing since we're trying to fix the issues"; "there
should always be consensus across files about what state things are in"). Supersedes the shelving row of `decisions.md`
of the same day.*

E15 asks whether retraining Γ on the updated model, with the subcomponents and Δ fixed, makes a split decomposition
faithful again. Two retraining recipes took the unchanged model off its own adversarial floor (2.6 to 6.3; 2.1 then 7.8),
so neither can be read on an updated state. The likely cause is that the retraining's persistent adversary is weaker than
the harness's attacker (0.42 against 0.70 nats on the released Γ), so the retraining never sees the damage the check
measures (`reports/analyses/2026-10-20_analysis_gamma-refit.md`, `reports/plans/2026-10-21_plan_gamma-refit-protocol.md`).

**What changes.** E15's status is "ongoing, being fixed" in every file. The fix being worked on is a retraining whose
adversary matches the harness's attacker. It goes through `/design` and its control (the unchanged model, alone, with an
early-stop rule) runs first, before any updated state. The four unrun arms stay unrun until that control holds.

**What does not change.** Re-gating stays the main line and the commute question second
(`reports/decisions/2026-10-21_decision_regating-main-line.md`). E15's filed predictions and results stand as filed.
