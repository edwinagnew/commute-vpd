---
name: feedback-avoid-overusing-gating
description: "The user asked not to over-use the word \"gating\" (and \"re-gating\", \"gated on/off\") in reports, notebooks and chat"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 1890c3d7-128b-4d84-8067-ee4dd7b42354
  modified: 2026-10-19T10:29:20.065Z
---

Don't over-use "gating" / "re-gating" / "gated on, gated off" (user, 2026-10-19, after the recruit-test notebook and
the LoRA-pull-back design check, both of which used it in nearly every sentence).

**Why:** it became a catch-all jargon word that hid what was actually measured, and made the write-ups hard to read.

**How to apply:** say the concrete thing instead. "Γ switches subcomponent c on" or "c's CI value rises at the
trigger"; "the CI-masked model with W₀'s CI values" rather than "the old gates"; "the share carried by changed CI
values on unchanged subcomponents" rather than "the re-gating share". Keep a coined short name (e.g. r) once defined,
and use plain words around it. Follows [[feedback-define-terms-when-reporting]].
