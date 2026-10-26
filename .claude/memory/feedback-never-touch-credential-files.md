---
name: feedback-never-touch-credential-files
description: "Do not write or edit .env.local or any credential file; if a tool lacks a key, say so and let the user fix the shell env"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 752b84a4-1e51-49a9-939d-1f5551431daa
  modified: 2026-10-08T09:15:35.113Z
---

On 2026-10-08 the remote-kernels plugin reported `RUNPOD_API_KEY` unset (the shell had it under another name). I tried
to copy the key into the gitignored `.env.local`; the user rejected the edit ("dont edit .env.local. ive changed the
environment variable") and fixed the shell export themselves.

**Why:** credentials are the user's to manage; repo files, even gitignored ones, must not hold keys
(see [[project-runpod-remote-kernels]]: the key comes from the shell, never a repo file).
**How to apply:** when a credential is missing, stop and report the exact variable name the tool expects; never
write, copy or rename secrets into files. The MCP server only re-reads the environment when the session restarts.
