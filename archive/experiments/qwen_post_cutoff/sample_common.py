"""E128 sampling: what `sample.py`'s stages share (paths, file writes, generation in fixed-size calls, candidate facts).
`Gen` is one call for a list of chat texts; a dry run passes a stand-in, so nothing here needs the model."""

from __future__ import annotations

import gzip
import json
import sys
import zlib
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import torch

from commute import qwen_eval

CAP, SEED, GEN_SEQS = 64, 0, 512  # new-token cap of no-think sampling; seed; sequences per generate call
Item = dict[str, Any]
Gen = Callable[[list[str], int, dict[str, Any], int, str], list[dict[str, Any]]]


@dataclass(frozen=True)
class Paths:
    candidates: Path
    stage0: Path
    stage1: Path
    n1: Path
    pools: Path
    shards: Path  # large/stage1: one folder per pass
    merged: Path
    n1_shards: Path
    gsm8k_sink: Path


def paths_under(results: Path, large: Path, candidates: Path) -> Paths:
    return Paths(candidates, results / "stage0.json", results / "stage1.json", results / "n1.json", results / "pools.json",
                 large / "stage1", large / "stage1" / "merged.json.gz", large / "n1", large / "stage0" / "gsm8k_test_0-255.jsonl")


def write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    tmp.replace(path)


def write_gz(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        json.dump(obj, f)


def real_gen(tok: Any, model: Any) -> Gen:
    """`qwen_eval.generate` in calls of at most GEN_SEQS sequences, each call seeded from `key` (a resumed shard repeats)."""

    def gen(texts: list[str], n_ret: int, kw: dict[str, Any], cap: int, key: str) -> list[dict[str, Any]]:
        per, out = max(1, GEN_SEQS // n_ret), []
        for i, s in enumerate(range(0, len(texts), per)):
            torch.manual_seed(zlib.crc32(f"{key}-{i}".encode()))
            out += qwen_eval.generate(tok, model, texts[s : s + per], n_ret, kw, max_new=cap)
        return out

    return gen


def stop(path: Path, record: dict[str, Any], why: str) -> NoReturn:
    record["stopped"] = why
    write(path, record)
    print(f"STOP: {why} ({path})", flush=True)
    sys.exit(2)


def facts_of(items: list[Item]) -> dict[str, list[Item]]:
    """{fact id: its 4 items in template order}."""
    out: dict[str, list[Item]] = defaultdict(list)
    for it in items:
        out[it["fact"]].append(it)
    for fid, its in out.items():
        its.sort(key=lambda i: i["template"])
        assert [i["template"] for i in its] == [0, 1, 2, 3], fid
    return dict(out)


def side_of(its: list[Item]) -> str:
    return its[0]["level"]
