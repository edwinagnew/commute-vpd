"""vLLM 0.11.0 on the unchanged Qwen3-1.7B (bf16, the repo's pinned revision), for E120's pod job.

One call `generate(eng, users, n, kw, caps, think)` per batch of prompts: chat template (thinking off unless `think`),
token ids passed straight to vLLM, one SamplingParams per prompt (caps differ between GSM8K and MATH). Per sample:
text, response tokens up to and including the end token (as `qwen_eval.generate` counts them), prompt tokens, and
truncated (no end token within the cap). Sampling settings are `qwen_eval.SAMPLE` / `GREEDY` as dicts; vLLM's
library defaults hold for everything not set (max_num_seqs, batched tokens, prefix caching).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from problems import chat

from commute.qwen_eval import MODEL, MODEL_REV

MAX_MODEL_LEN = 9216  # N1's 8,192-token cap plus the longest prompt
THINK = {"do_sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20}  # the card's thinking-mode settings


@dataclass
class Engine:
    llm: Any
    tok: Any
    eos: frozenset[int]
    tokens: int = 0  # response tokens generated so far, for tokens/s
    seconds: float = 0.0


def load() -> Engine:
    from transformers import GenerationConfig
    from vllm import LLM  # pyright: ignore[reportMissingImports]

    llm = LLM(model=MODEL, revision=MODEL_REV, tokenizer_revision=MODEL_REV, dtype="bfloat16", seed=0,
              max_model_len=MAX_MODEL_LEN)
    tok = llm.get_tokenizer()
    eos = GenerationConfig.from_pretrained(MODEL, revision=MODEL_REV).eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos)
    return Engine(llm, tok, frozenset(eos | {tok.eos_token_id}))


def _params(n: int, kw: dict[str, Any], cap: int) -> Any:
    from vllm import SamplingParams  # pyright: ignore[reportMissingImports]

    if not kw.get("do_sample"):
        return SamplingParams(n=n, temperature=0.0, max_tokens=cap)
    return SamplingParams(n=n, temperature=kw["temperature"], top_p=kw["top_p"], top_k=kw["top_k"], max_tokens=cap)


def generate(eng: Engine, users: list[str], n: int, kw: dict[str, Any], caps: list[int],
             think: bool = False) -> list[list[dict[str, Any]]]:
    """Per prompt, its n samples: {text, tokens, prompt_tokens, truncated}."""
    ids = [eng.tok(chat(eng.tok, u, think), add_special_tokens=False)["input_ids"] for u in users]
    t0 = time.time()
    outs = eng.llm.generate([{"prompt_token_ids": p} for p in ids], [_params(n, kw, c) for c in caps], use_tqdm=False)
    eng.seconds += time.time() - t0
    res = []
    for o in outs:
        row = []
        for c in o.outputs:
            done = c.finish_reason == "stop"
            got = list(c.token_ids)
            k = len(got) + (1 if done and not (got and got[-1] in eng.eos) else 0)
            row.append({"text": c.text, "tokens": k, "prompt_tokens": len(o.prompt_token_ids), "truncated": not done})
            eng.tokens += len(got)
        res.append(row)
    return res


def tps(eng: Engine) -> float:
    return eng.tokens / eng.seconds if eng.seconds else 0.0
