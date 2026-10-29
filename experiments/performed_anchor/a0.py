"""E53 stage A0: does the original 67M perform lowercase continuation and fail the token cipher, before any fine-tune?
Plan: reports/plans/2026-10-29_plan_performed-anchor.md ("The pair", "Control first"). W0 only; nothing is trained.

1. control   W0's mean held-out log-probability on E36 Amendment 1's 64 lowercased val rows (`lowercase.task`).
2. lowercase mean over case positions of log p(lowercase form) - log p(capitalised form), in each val row 0-255
             lowercased on its own (`performed.case_pairs`; definitions in that module).
3. cased     the same words and forms, in the cased val rows.
4. cipher    mean over cipher positions of log p(pi(x)) - log p(x) in the ciphered cased val rows 0-255; pi a
             derangement (seed 0) of the word tokens of frequency ranks 50-99 on Pile train rows 0-4095.
The thresholds and the stop rule are read from results/predictions_a0.json (`verdict`). Writes results/a0.json and
prints every number; exit 0 = GO, 1 = STOP, 2 = no positions (an error, never a pass).

Usage (repo root, project venv, GPU):  a0.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED, OUT = RESULTS / "predictions_a0.json", RESULTS / "a0.json"


@dataclass(frozen=True)
class Config:
    val_rows: int = 256  # Pile val rows 0-255, the plan's held-out rows
    count_rows: int = 4096  # Pile train rows 0-4095 for the word-token frequencies
    ranks: tuple[int, int] = (50, 100)  # 0-based, end exclusive: 50 tokens
    cipher_seed: int = 0
    boot: int = 2000  # bootstrap resamples over rows
    boot_seed: int = 0
    batch: int = 8

    def __post_init__(self) -> None:
        assert self.ranks[1] - self.ranks[0] == 50


CFG = Config()


def spread(d: Tensor, row: Tensor) -> dict[str, Any]:
    """Mean over positions, its 90% interval from resampling rows with replacement, and the per-row means' spread.
    No positions raises: an empty pool is an error, never a pass."""
    if len(d) == 0:
        raise ValueError("no positions")
    rows, inv = torch.unique(row, return_inverse=True)
    s = torch.zeros(len(rows), dtype=torch.float64).index_add_(0, inv, d)
    c = torch.zeros(len(rows), dtype=torch.float64).index_add_(0, inv, torch.ones_like(d))
    g = torch.Generator().manual_seed(CFG.boot_seed)
    idx = torch.randint(len(rows), (CFG.boot, len(rows)), generator=g)
    boot = (s[idx].sum(1) / c[idx].sum(1)).sort().values
    per_row = s / c
    return {"mean": float(d.mean()), "n_positions": len(d), "n_rows": len(rows),
            "boot90": [float(boot[int(0.05 * CFG.boot)]), float(boot[int(0.95 * CFG.boot) - 1])],
            "row_mean_sd": float(per_row.std()) if len(rows) > 1 else 0.0, "share_positive": float((d > 0).double().mean())}


def verdict(num: dict[str, Any], pred: dict[str, Any]) -> dict[str, Any]:
    """The filed rules (predictions_a0.json) on the numbers. Every rule is written so that NaN fails it (a comparison
    with NaN is False), and any non-finite number stops."""
    t = pred["thresholds"]
    for k in ("lowercase", "cased", "cipher"):
        if num[k]["n_positions"] == 0:
            raise ValueError(f"{k}: no positions")
    ctl, low, cas, cip = num["control_logp"], num["lowercase"]["mean"], num["cased"]["mean"], num["cipher"]["mean"]
    control = abs(ctl - t["control_ref"]) <= t["control_tol"]
    stops = {"control_missed": not control, "lowercase_not_above_0": not low > t["stop_lowercase_at_most"],
             "cipher_above_minus_1": not cip <= t["stop_cipher_above"],
             "non_finite": not all(math.isfinite(x) for x in (ctl, low, cas, cip))}
    return {"control": control, "P_lowercase_performs": low >= t["lowercase_at_least"],
            "P_cased_follows_style": cas <= t["cased_at_most"], "P_cipher_cannot": cip <= t["cipher_at_most"],
            "stops": stops, "decision": "STOP" if any(stops.values()) else "GO"}


def main() -> int:
    from commute import lowercase, performed, vpd
    from commute.data import pile_rows

    if not PRED.is_file():
        sys.exit(f"file the predictions first: {PRED}")
    pred = json.loads(PRED.read_text())
    t0, device, b = time.time(), "cuda", CFG.batch
    tok = vpd.tokenizer()
    model = vpd.load_component_model(device)
    low = lowercase.task(model, device)
    num: dict[str, Any] = {"config": asdict(CFG), "control_logp": low.logp_w0}
    print(f"[control] W0 logp on E36's lowercased held rows {low.logp_w0:.5f} (ref {pred['thresholds']['control_ref']})",
          flush=True)

    val = pile_rows("val", 0, CFG.val_rows)
    cp = performed.case_pairs(val, tok)
    num["case_skipped_rows"] = cp.skipped_rows
    d_low = performed.log_odds(low.w0_net, cp.lower, cp.in_lower, device, b)
    d_cas = performed.log_odds(low.w0_net, val, cp.in_cased, device, b)

    ranks = performed.word_ranks(tok, pile_rows("train", 0, CFG.count_rows))
    pi = performed.derangement(ranks[CFG.ranks[0] : CFG.ranks[1]].tolist(), CFG.cipher_seed)
    crows, cpos = performed.cipher_rows(val, pi)
    d_cip = performed.log_odds(low.w0_net, crows, cpos, device, b)
    num["cipher_map"] = [{"x": x, "pi_x": y, "x_str": tok.decode([x]), "pi_x_str": tok.decode([y])} for x, y in pi.items()]

    for name, d, at in (("lowercase", d_low, cp.in_lower), ("cased", d_cas, cp.in_cased), ("cipher", d_cip, cpos)):
        try:
            num[name] = spread(d, at.row)
        except ValueError:
            num[name] = {"n_positions": 0}
            print(f"[{name}] ERROR: no positions", flush=True)
            OUT.write_text(json.dumps(num | {"decision": "ERROR"}, indent=1))
            return 2
        print(f"[{name}] " + " ".join(f"{k} {v}" for k, v in num[name].items()), flush=True)
    num["verdict"] = verdict(num, pred)
    num["seconds"] = time.time() - t0
    print(f"[cipher] map {[(m['x_str'], m['pi_x_str']) for m in num['cipher_map']]}", flush=True)
    print(f"[verdict] {json.dumps(num['verdict'])}", flush=True)
    print(f"[decision] {num['verdict']['decision']}  ({num['seconds']:.0f} s)", flush=True)
    OUT.write_text(json.dumps(num, indent=1))
    return 0 if num["verdict"]["decision"] == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
