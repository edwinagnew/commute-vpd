"""E139 Amendment 1 (cell P1.4): E137's found witness for Always on, scored against E139's three Always on fine-tunes. No
search over the fine-tunes and no training. Plan: reports/predictions/2026-11-10_prediction_e139-amendments-1-2.md; filed
rule: results/predictions_amend1.json (the run refuses unless it is committed unchanged).

The witness is E137's G search `Always on/k4/in2/s2` (phi = 4 columns at in2, s an affine map of them; G 0.9225 against the
hand-made pair). Its (phi, s) were not saved; the same seed through `commute.variable_read` regenerates it, as E139's control
did to the last digit (results/control.json). Control first: the regenerated witness, read against the hand-made pair, must
repeat 0.9225 within `replay_tol`, else exit 1. Then per saved fine-tune (results/large/always_on_s{0,1,2}.pt): its gap and
distance to the pair (E139's `distances`, checked against E139's saved gap), and the witness's G = 1 - d(M[phi <- s(v_M)],
F_ft) / gap, reversion and setter mismatch on the read half, beside the two bounds the triangle inequality puts on that G.
Writes results/amend1.json after each step. About 6 min on an L40.

    sbatch --comment='E139 amend1' cluster/reuse_variable_finetune_amend.sbatch amend1
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

import commute.variable_read as VR
from commute.hidden_maps import chunks, run, tv
from commute.jobs import hard_exit, load_experiment, require_filed
from commute.steering import Net

R: Any = load_experiment("reuse_variable_finetune", "run")  # E139's Config, distances, clone_net, log, dump, stop
HERE = Path(__file__).resolve().parent
PRED = HERE / "results" / "predictions_amend1.json"
WITNESS = "Always on/k4/in2/s2"  # E137's best start at the best site; E139 `_replay_g` draws the same seed
SEEDS = (0, 1, 2)


def setup(net: Net, cfg: Any) -> tuple[VR.Sets, VR.Task, VR.Pieces]:
    """Rows, task and pieces as E139's `main` builds them; exits if the cue is not E137's."""
    from commute import vpd

    net.net.requires_grad_(False)
    tok = vpd.tokenizer()
    sets = VR.make_sets(cfg, tok, R.log)
    task = VR.make_task(sets, tok, int(net.net.lm_head.weight.shape[0]), cfg)  # type: ignore[union-attr]
    cue = json.loads((R.R1 / "stage0.json").read_text())["pieces"]["cue_token"]
    if task.cue != cue:
        raise SystemExit(f"the cue {task.cue} is not E137's {cue}: the rows or the vocabulary differ")
    return sets, task, VR.load_pieces(R.R1 / "large" / "pieces.pt", net.device, task)


def witness(net: Net, fit: Tensor, F: VR.Target, cfg: Any) -> dict[str, Any]:
    """E137's found (phi, s) for Always on at in2, regenerated: {Q, A, c, last_kl}. F is the hand-made pair."""
    return VR.g_search(net, fit, "in2", 4, 2, F, cfg, 137_000 + zlib.crc32(WITNESS.encode()) % 65_536)


def read_witness(net: Net, read: Tensor, w: dict[str, Any], F: VR.Target, gap: float, cfg: Any) -> dict[str, float]:
    """The witness read against the changed model F with gap `gap`: G, reversion, mismatch, and d (mean TV of M[phi <- s] to F)."""
    A, c = w["A"], w["c"]
    r = VR.read_pair(net, read, "in2", w["Q"], lambda v, _s: v @ A + c, F, cfg)
    return {"G": 1 - r["d"] / gap, "reversion": 1 - r["d_rev"] / gap, "mismatch": r["mismatch"], "d": r["d"]}


def bounds(d_pair: float, tv_to_pair: float, gap: float) -> dict[str, float]:
    """The triangle inequality on mean TV: |d(M[phi<-s], F_ft) - d(M[phi<-s], F)| <= d(F, F_ft), so the witness's G against a
    fine-tune lies in [1 - (d_pair + tv_to_pair)/gap, 1 - (tv_to_pair - d_pair)/gap]. Outside it is a bug, not a reading."""
    return {"lower": 1 - (d_pair + tv_to_pair) / gap, "upper": 1 - (tv_to_pair - d_pair) / gap}


def pair_gap(net: Net, F: VR.Target, read: Tensor, cfg: Any) -> float:
    return float(torch.cat([tv(F(t)[0][:, -1], run(net, t)[0][:, -1]) for t in chunks(read, cfg.read_batch)]).mean())


def main(net: Net) -> int:
    cfg, th = R.Config(), json.loads(PRED.read_text())["thresholds"]
    sets, task, p = setup(net, cfg)
    fit, read = VR.pair_windows(cfg, "Always on", sets, task)
    F = VR.piece_target(net, p, "Always on")
    path = cfg.results / "amend1.json"
    out: dict[str, Any] = {"experiment": f"{R.EXPERIMENT} amend1", "witness": WITNESS, "items": {}}
    w = witness(net, fit, F, cfg)
    gap = pair_gap(net, F, read, cfg)
    out["control"] = {"gap": gap, **read_witness(net, read, w, F, gap, cfg), "fit_kl": w["last_kl"]}
    R.log(f"control: {out['control']}")
    if abs(out["control"]["G"] - th["witness_G"]) > th["replay_tol"] or abs(gap - th["pair_gap"]) > th["gap_replay_tol"]:
        R.stop(path, out, "the regenerated witness does not repeat E137's G against the pair, or the pair's gap misses E137's")
    R.dump(path, out)
    for seed in SEEDS:
        item = f"always_on_s{seed}"
        saved = json.loads((cfg.results / f"{item}.json").read_text())["dist"]
        ft = R.clone_net(net)
        ft.net.load_state_dict(torch.load(cfg.results / "large" / f"{item}.pt", map_location=net.device))
        dist = R.distances(net, F, ft, read, cfg)
        if abs(dist["gap"] - saved["gap"]) > th["gap_replay_tol"]:
            R.stop(path, out, f"{item}: the saved state's gap {dist['gap']:.5f} is not E139's {saved['gap']:.5f}")
        rec = {"dist": dist, **read_witness(net, read, w, VR.net_target(ft), dist["gap"], cfg)}
        rec["bounds"] = bounds(out["control"]["d"], dist["tv_to_F"], dist["gap"])
        rec["within_bounds"] = rec["bounds"]["lower"] - 1e-9 <= rec["G"] <= rec["bounds"]["upper"] + 1e-9
        out["items"][item] = rec
        R.dump(path, out)
        R.log(f"{item}: witness G {rec['G']:.4f} in [{rec['bounds']['lower']:.4f}, {rec['bounds']['upper']:.4f}], reversion {rec['reversion']:.4f}, mismatch {rec['mismatch']:.4f}, gap {dist['gap']:.5f}, tv_to_F {dist['tv_to_F']:.5f}")
        del ft
        torch.cuda.empty_cache()
    out["complete"] = True
    R.dump(path, out)
    print("AMEND1 DONE", flush=True)
    return 0


if __name__ == "__main__":
    require_filed(PRED)
    code = 1
    try:
        code = main(Net("cuda"))
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)
