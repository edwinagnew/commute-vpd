"""Anchor 1 of E45 (context distillation; plan reports/plans/2026-10-28_plan_qwen-q2.md, section 3): the candidate
prompts, the chat texts they make, the format fallback's template, and the distillation gap.

A candidate adds a system turn or few-shot turns to the baseline's chat (`qwen_eval.prompt`: the problem, a blank
line, INSTRUCTION, thinking off); none may enable thinking or add a think token (`chat` checks). The candidates were
fixed in the plan before any run:
  fewshot   4 earlier turns: train problems and the unchanged model's own correct greedy answers to them
  recheck   system: RECHECK
  givens    system: GIVENS
  format    (the fallback, only if none of the three qualifies) system: FORMAT_SYSTEM; followed when the response
            matches FORMAT_RE
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import torch

from commute.qwen_eval import INSTRUCTION
from commute.qwen_sft import Weights, set_weights
from commute.vpd import kl_per_position

RECHECK = "Before you write the final line, re-check every arithmetic step of your solution and correct any mistake."
GIVENS = "First list every quantity given in the problem, one per line. Then compute the answer from them."
FORMAT_SYSTEM = ("Lay out every answer in exactly this template, with nothing before or after it:\n"
                 "Given: <the quantities in the problem, separated by semicolons>\n"
                 "Steps:\n- <one calculation per line>\n"
                 "The answer is N.")
FORMAT_RE = re.compile(r"Given: [^\n]+\nSteps:\n(?:- [^\n]+\n)+The answer is [^\n]+")
SYSTEMS = {"none": None, "fewshot": None, "recheck": RECHECK, "givens": GIVENS, "format": FORMAT_SYSTEM}

Shot = tuple[str, str]  # a problem and the answer shown for it


def user(question: str) -> str:
    return f"{question}\n\n{INSTRUCTION}"


def chat(tok: Any, content: str, system: str | None = None, shots: Sequence[Shot] = ()) -> str:
    """The chat text (thinking off, generation prompt added) for one user turn, after an optional system turn and
    few-shot turns; no think token beyond the one the no-think generation prompt itself carries."""
    msgs = [{"role": "system", "content": system}] if system else []
    for q, a in shots:
        msgs += [{"role": "user", "content": user(q)}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": content})
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    plain = tok.apply_chat_template(msgs[-1:], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert text.count("<think>") == plain.count("<think>"), "a candidate prompt adds a think token"
    return text


def candidate(tok: Any, name: str, question: str, shots: Sequence[Shot]) -> str:
    """The maths chat for a candidate: "none" is exactly `qwen_eval.prompt`."""
    return chat(tok, user(question), SYSTEMS[name], shots if name == "fewshot" else ())


def follows_format(response: str) -> bool:
    return FORMAT_RE.fullmatch(response.strip()) is not None


@torch.no_grad()
def distill_kl(model: Any, pairs: list[tuple[list[int], list[int], int]], teacher: Weights | None,
               student: Weights | None) -> float:
    """Mean over response positions of KL(model at `teacher` on the prompted sequence ‖ model at `student` on the
    unprompted one). A pair is (prompted ids, unprompted ids, n): both end with the same n response tokens, and the
    positions read are the n that predict them."""
    total, count = 0.0, 0
    for prompted, unprompted, n in pairs:
        assert prompted[-n:] == unprompted[-n:], "the two sequences end with the same response"
        with set_weights(model, teacher):
            lp = model(torch.tensor([prompted], device="cuda"), use_cache=False).logits[0, -n - 1 : -1]
        with set_weights(model, student):
            lq = model(torch.tensor([unprompted], device="cuda"), use_cache=False).logits[0, -n - 1 : -1]
        total += float(kl_per_position(lp, lq).double().sum())
        count += n
    return total / count
