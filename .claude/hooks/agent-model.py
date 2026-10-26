#!/usr/bin/env python3
"""PreToolUse hook on the Agent tool: every subagent must run on a model chosen on purpose.

Allowed: subagent_type runner (Haiku), clerk (Sonnet) or pod-runner (Opus), or any call that sets `model` explicitly
(opus for judgment work). Refused, with a reason the calling agent sees: calls that name no model and would
silently inherit Opus, and forks, which copy the caller's whole context into every turn they take.
Ported from mats-fleet (2026-10-26).
"""
import json
import sys

ROUTED = {"runner", "clerk", "pod-runner"}  # pod-runner pins Opus in its own frontmatter

try:
    call = json.load(sys.stdin)
except ValueError:
    sys.exit(0)
if call.get("tool_name") not in ("Agent", "Task"):
    sys.exit(0)
args = call.get("tool_input") or {}
kind = args.get("subagent_type") or ""
if kind == "fork":
    print("Forks copy your whole context into every turn they take. Launch a fresh subagent with a short brief "
          "instead: subagent_type runner (Haiku) or clerk (Sonnet) for mechanical work, or model opus for judgment "
          "work.", file=sys.stderr)
    sys.exit(2)
if kind in ROUTED or args.get("model"):
    sys.exit(0)
print("Choose this subagent's model on purpose: subagent_type runner (Haiku) for mechanical work whose result can "
      "be checked, clerk (Sonnet) for fully specified work, or set model to opus for judgment work. GPU runs use "
      "subagent_type pod-runner.", file=sys.stderr)
sys.exit(2)
