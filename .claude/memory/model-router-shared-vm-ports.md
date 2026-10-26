---
name: model-router-shared-vm-ports
description: model-router on the shared cluster VM uses ports 8788/8318 because another user holds 8787/8317; not wired into Claude Code yet
metadata:
  type: reference
---

On this shared VM another user (k.yamaguchi) runs a model-router on 127.0.0.1:8787 and its CLIProxyAPI on 8317, so
`doctor` silently talks to *their* router and our service crash-loops. Ours is set in
`~/.config/model-router/config.toml`: `port = 8788` and `[upstreams.cliproxy] mode="managed" port=8318` (2026-10-26).

Claude Code is deliberately NOT wired to the router (user chose to wait): the ChatGPT login (free-tier file name) is
out of Codex quota until ~2026-11-18 and its catalog lacks the Sol/Astra models. When wiring later, use the 8788
base URL from `doctor --json`. Reviews before paid runs use a fresh Opus agent until then (see .claude/CLAUDE.md).
