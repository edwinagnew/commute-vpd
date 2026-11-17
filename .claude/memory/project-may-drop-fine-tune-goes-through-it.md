---
name: project-may-drop-fine-tune-goes-through-it
description: user 2026-11-10 — the second wording may lose its requirement that the fine-tune itself routes the change through the variable (the split); depends on the two E139 amendments; nothing changed yet
metadata:
  type: project
---

The second reuse wording's "re-weighted" label bundles two claims. Claim 1: the original's variable is enough (M with one small variable of its own remapped by a fixed affine map behaves like the fine-tune F; this is what G measures, and H likewise for rewired). Claim 2: the fine-tune actually goes through that variable (the split; put M's value back into F and F reverts; measured by reversion and setter mismatch).

The user said on 2026-11-10 (chat with the top-level session, written here at their request) that we may well have to drop claim 2, depending on the two E139 amendments. Nothing is adopted: the wording, the label rule and the cut-offs stand as filed until the user says otherwise.

**Why:** in E139 the Always on imitators met claim 1 as far as their imitation allowed (best G 0.63 to 0.64 against a ceiling of about 0.64 set by a 36% un-imitated leftover) while claim 2 failed (reversion 0.20 at block 2's input, 0.93 only at block 3's output). The filed label rule already reads only G and H, so the tests score claim 1 while the written definition asserts both. The user is wary of adding moving parts each time something goes wrong, and the definition must be justified rigorously in the paper; dropping claim 2 is a deletion, patching the split to tolerate diffuse fine-tunes (several sites, different layers) is an addition and is not wanted. Interpretation of the top-level session, not the user's words: dropping claim 2 still looks inside M but no longer inside F, which touches [[project-working-definition-criteria]] ("must look inside the model") and should be put to the user if it comes to a decision.

**How to apply:** do not change the definition. Make the amendments decisive on this point: report reversion and setter mismatch beside G and H at every imitation level, and say plainly whether reversion at block 2's input stays low once the leftover is under about 0.1 of the gap. If it does, bring the user the choice (drop claim 2 and report reversion as a finding, or keep it and add a reversion condition to the label rule) as a decision card with a recommendation; do not add a third variant. On the gate page (L1.5), state which of the two claims each label makes.
