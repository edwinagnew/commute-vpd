"""The GSM8K splits and GSM1k as the Qwen GSM8K runs read them: pinned revisions, items {idx, question, gold}.

GSM1k keeps only the problems sharing under half of their word 8-grams with any GSM8K question (train or test): 1,197
of 1,205. Moved here 2026-10-28 for experiments/qwen_sft from experiments/qwen_fresh_set/run.py (`load_sets`, `grams`,
`overlap`), which keeps its own copy, as it ran; the rule and revisions are unchanged.
"""

from __future__ import annotations

import re
from typing import Any

GSM8K, GSM8K_REV = "openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866"
GSM1K, GSM1K_REV = "ScaleAI/gsm1k", "bc09569d09a614b9b530edc7f076fb214ac10493"
NGRAM, OVERLAP_MAX = 8, 0.5
N_TEST, N_TRAIN, N_GSM1K_KEPT = 1319, 7473, 1197

Items = list[dict[str, Any]]


def _rows(name: str, *args: str, split: str, revision: str) -> Items:
    from datasets import Dataset, load_dataset

    ds = load_dataset(name, *args, split=split, revision=revision)
    assert isinstance(ds, Dataset)
    return ds.to_list()


def grams(text: str) -> set[tuple[str, ...]]:
    w = re.findall(r"\w+", text.lower())
    return {tuple(w[i : i + NGRAM]) for i in range(len(w) - NGRAM + 1)}


def overlap(text: str, seen: set[tuple[str, ...]]) -> float:
    """Share of the text's word 8-grams found in `seen`; 0.0 for a text shorter than 8 words."""
    g = grams(text)
    return len(g & seen) / len(g) if g else 0.0


def load() -> dict[str, Items]:
    """{"train", "test", "gsm1k"}; idx is the row in its split (GSM1k: its row before the filter); gold is a solution
    text ending '#### N'."""
    test = _rows(GSM8K, "main", split="test", revision=GSM8K_REV)
    train = _rows(GSM8K, "main", split="train", revision=GSM8K_REV)
    seen = set().union(*(grams(r["question"]) for r in test + train))
    g1 = _rows(GSM1K, split="test", revision=GSM1K_REV)
    sets = {
        "train": [{"idx": i, "question": r["question"], "gold": r["answer"]} for i, r in enumerate(train)],
        "test": [{"idx": i, "question": r["question"], "gold": r["answer"]} for i, r in enumerate(test)],
        "gsm1k": [{"idx": i, "question": r["question"], "gold": f"#### {r['answer']}"} for i, r in enumerate(g1)
                  if overlap(r["question"], seen) < OVERLAP_MAX],
    }
    sizes = {k: len(v) for k, v in sets.items()}
    assert sizes == {"train": N_TRAIN, "test": N_TEST, "gsm1k": N_GSM1K_KEPT}, sizes
    return sets
