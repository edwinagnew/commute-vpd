"""Post-hoc and not filed (2026-11-02, written with E64's analysis): how much of the block 0 reading could a table of
next-token counts on the readout's own training sample give? No model is loaded; it reads the cached Pile rows and E54's
key and value lists. For each key k and each of the 64 values v, c(k, v) = windows of the sample whose token at the
answer position is k and whose next token is v. The same rank and specificity rules as run.json are then applied to the
smoothed table P(v | k) = (c(k, v) + 1) / (c(k, .) + 64): a one-token lookup that holds no model at all.

  PYTHONPATH=experiments/readout64 ~/commute-agents/fleet/cpu-run --mem 6G python experiments/readout64/bigram_check.py
Writes experiments/readout64/results/pile_bigrams.json.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from commute.data import pile_rows
from commute.readout import next_token_windows

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
STAGE1 = HERE.parent / "keydependent_elicit" / "results" / "stage1.json"


def lower_median(x: list[float]) -> float:
    return sorted(x)[(len(x) - 1) // 2]


def table_reading(c: torch.Tensor, keys: list[int], offset: int, th: dict[str, Any]) -> dict[str, Any]:
    """c: (32, 64) counts, row per key; a key's value is output offset + its index. The rules of readout.key_reading."""
    p = (c + 1.0) / (c.sum(1, keepdim=True) + c.shape[1])
    out: dict[str, Any] = {"count": {}, "key_total": {}, "rank": {}, "log_ratio": {}, "possessed": []}
    for i, k in enumerate(keys):
        v = offset + i
        rank = 1 + int((c[i] > c[i, v]).sum())
        others = [math.log(float(p[j, v])) for j in range(len(keys)) if j != i]
        ratio = math.log(float(p[i, v])) - lower_median(others)
        out["count"][k], out["key_total"][k], out["rank"][k], out["log_ratio"][k] = int(c[i, v]), int(c[i].sum()), rank, ratio
        if rank <= th["possess_rank"] and ratio >= math.log(th["specific_ratio"]):
            out["possessed"].append(k)
    return out


def main() -> None:
    P = json.loads((RES / "predictions.json").read_text())
    S, th = P["settings"], P["thresholds"]
    s1 = json.loads(STAGE1.read_text())
    keys = s1["elicit"]["keys"]
    assert keys == s1["cipher"]["keys"]
    vals = torch.tensor([*s1["elicit"]["values"], *s1["cipher"]["values"]])
    rows = pile_rows(*S["train_rows"], log=print)
    half = 20
    tr, y, counts = next_token_windows(rows[: S["fit_row_count"]], vals, half, S["cap"], S["positions_seed"])
    run = json.loads((RES / "run.json").read_text())
    assert dict(zip(map(str, vals.tolist()), counts, strict=True)) == run["stage0"]["positions"]["per_value"], "not the run's rows"
    assert len(tr) == run["stage0"]["positions"]["fit_windows"], "not the run's sample"
    c = torch.zeros(len(keys), len(vals))
    for i, k in enumerate(keys):
        at = tr[:, half] == k
        c[i] = torch.bincount(y[at], minlength=len(vals)).float()
    full = torch.zeros(len(keys), len(vals))  # every window of the fit rows, before the cap
    fit = rows[: S["fit_row_count"]]
    now, nxt = fit[:, 20:-20].flatten(), fit[:, 21:-19].flatten()
    for i, k in enumerate(keys):
        nx = nxt[now == k]
        for j, v in enumerate(vals.tolist()):
            full[i, j] = int((nx == v).sum())
    out = {"note": "post-hoc, not filed; no model", "keys": keys, "values": vals.tolist(),
           "sample": {n: table_reading(c, keys, o, th) for n, o in (("elicit", 0), ("cipher", 32))},
           "all_fit_rows": {n: table_reading(full, keys, o, th) for n, o in (("elicit", 0), ("cipher", 32))}}
    (RES / "pile_bigrams.json").write_text(json.dumps(out, indent=1, sort_keys=True))
    for src in ("sample", "all_fit_rows"):
        for n in ("elicit", "cipher"):
            r = out[src][n]
            print(src, n, "possessed", len(r["possessed"]), r["possessed"])
            print("  counts", list(r["count"].values()))
            print("  ranks", list(r["rank"].values()))


if __name__ == "__main__":
    main()
