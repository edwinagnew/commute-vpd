"""E120's problems, prompts and file helpers, shared by the pod job (`pod.py`, `stage1.py`), `pools.py` and the dry run.
Plan: reports/plans/2026-11-06_plan_e120-pass-rate-pools.md.

An item is {id, source, level, subject, user, gold_response, answer}: `user` is the exact user-message text, instruction
included; `gold_response` the gold solution as the SFT stage trains on it (GSM8K: `gsm8k.gold_as_response`, MATH: the
solution text as it is); `answer` the gold answer string the graders read (GSM8K: the number after '####'; MATH: the
content of the solution's last \\boxed{}, None when it has none). Ids: `gsm8k/train/<row>`, `gsm8k/test/<row>`,
`math/<subject>/<row>` (the row in that subject's train split, before the level filter).
GSM8K keeps `qwen_eval.INSTRUCTION` and its "\\n\\n" join (`qwen_eval.prompt`); MATH gets the Qwen3 card's line with the
same join. Chat template with thinking off, except the think-mode check N1.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from grade import boxed

from commute.gsm8k import gold_as_response, gold_number
from commute.qwen_eval import INSTRUCTION, chat_prompt
from commute.qwen_sets import load as load_gsm8k

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DL = RESULTS / "pod_dl"  # gitignored (`**/results/*_dl/`); the downloader brings it to the same project path
PRED = RESULTS / "predictions.json"

MATH, MATH_REV = "EleutherAI/hendrycks_math", "21a5633873b6a120296cce3e2df9d5550074f4a3"
SUBJECTS = ("algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra",
            "precalculus")
LEVELS = (3, 4, 5)
MATH_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
SEQ_LEN = 512  # the SFT window (`qwen_sft.sft_seq`): prompt + gold + end token over this would lose the answer

Item = dict[str, Any]


def gsm8k_item(split: str, row: dict[str, Any]) -> Item:
    return {"id": f"gsm8k/{split}/{row['idx']}", "source": "gsm8k", "level": None, "subject": "gsm8k",
            "user": f"{row['question']}\n\n{INSTRUCTION}", "gold_response": gold_as_response(row["gold"]),
            "answer": str(gold_number(row["gold"]))}


def math_items(counts: Counter[str]) -> list[Item]:
    """MATH train, levels 3-5, every subject; `counts` gets every (subject, level) row count, 'Level ?' included."""
    from datasets import Dataset, load_dataset

    out = []
    for subject in SUBJECTS:
        ds = load_dataset(MATH, subject, split="train", revision=MATH_REV)
        assert isinstance(ds, Dataset)
        for row_i, r in enumerate(ds.to_list()):
            lvl = r["level"].removeprefix("Level ")
            counts[f"{subject}/{lvl}"] += 1
            if not lvl.isdigit() or int(lvl) not in LEVELS:
                continue
            out.append({"id": f"math/{subject}/{row_i}", "source": "math", "level": int(lvl), "subject": subject,
                        "user": f"{r['problem']}\n\n{MATH_INSTRUCTION}", "gold_response": r["solution"],
                        "answer": boxed(r["solution"])})
    return out


def load_all() -> tuple[list[Item], list[Item], dict[str, int]]:
    """(the stage-1 problems: GSM8K train then MATH train levels 3-5; GSM8K test as items; MATH row counts per
    subject/level)."""
    g = load_gsm8k()
    counts: Counter[str] = Counter()
    probs = [gsm8k_item("train", r) for r in g["train"]] + math_items(counts)
    return probs, [gsm8k_item("test", r) for r in g["test"]], dict(sorted(counts.items()))


def stratum(it: Item) -> str:
    return "gsm8k" if it["source"] == "gsm8k" else f"math/{it['level']}/{it['subject']}"


def chat(tok: Any, user: str, think: bool = False) -> str:
    """`qwen_eval.chat_prompt` (thinking off, as the SFT stage builds it); the same template with thinking on for N1."""
    if not think:
        return chat_prompt(tok, user)
    msgs = [{"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)


def seq_tokens(tok: Any, it: Item) -> int:
    """Prompt (chat template, thinking off) + gold response + the end token, as `qwen_sft.sft_seq` counts them."""
    p = tok(chat(tok, it["user"]), add_special_tokens=False)["input_ids"]
    return len(p) + len(tok(it["gold_response"], add_special_tokens=False)["input_ids"]) + 1


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, obj: Any) -> str:
    """Written whole then renamed (a lost pod never leaves half a file); .gz paths gzipped. Returns the sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj, indent=None if path.suffix == ".gz" else 1).encode()
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(gzip.compress(data, mtime=0) if path.suffix == ".gz" else data)
    tmp.replace(path)
    return sha256(path)


def read_json(path: Path) -> Any:
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def rel(path: Path) -> str:
    """The path from the repo root, as the downloader names it (absolute when outside the repo, as in a dry run)."""
    p = path.resolve()
    return str(p.relative_to(HERE.parents[1])) if p.is_relative_to(HERE.parents[1]) else str(p)
