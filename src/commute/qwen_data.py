"""Generic text rows for the Qwen3-1.7B decomposition: FineWeb-Edu documents cut to the decomposition's 512 tokens.

The rule is the one `experiments/qwen_port/make_tokens.py` used for its four text rows (the first train documents of
sample-10BT with at least 512 tokens, first 512 tokens, no special tokens), at the revision it recorded, so the first
four rows here are those rows.
"""

from __future__ import annotations

from typing import Any

SEQ_LEN = 512
FINEWEB, FINEWEB_CONFIG = "HuggingFaceFW/fineweb-edu", "sample-10BT"
FINEWEB_REV = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"  # experiments/qwen_port/results/tokens.json "sources"


def fineweb_rows(tok: Any, n: int, seq_len: int = SEQ_LEN, revision: str = FINEWEB_REV) -> list[list[int]]:
    from datasets import load_dataset

    rows = []
    for doc in load_dataset(FINEWEB, FINEWEB_CONFIG, split="train", streaming=True, revision=revision):
        ids = tok(doc["text"], add_special_tokens=False)["input_ids"]
        if len(ids) >= seq_len:
            rows.append(ids[:seq_len])
        if len(rows) == n:
            return rows
    raise AssertionError("ran out of documents")
