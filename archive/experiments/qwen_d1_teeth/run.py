"""E85: VPD's adversary passes doing nothing on Qwen3-1.7B's first fine-tune update; does it fail anything? Two states built to be bad, on E80 Amendment 2's harness unchanged (web rows, 20
sign steps of 0.1, one source per site, final read, fp32 components). One 80 GB card, ~2.6 h. Plan: reports/plans/2026-11-04_plan_e85-adversary-teeth.md; rules: rules.py,
results/predictions.json (the run refuses to start without it, and while results/stopped.json exists).

Rand: weights W0 + a Gaussian at u1's norm per matrix, components unmoved, so Δ takes it. Rare: weights W0 + u1, u1's least-squares split over the rarely-on half moves those columns, Δ takes
what they leave. Order, cheapest and control first: the two states' checks, left in Δ and kept, before any start; W0 s0, W0 s1, each within 20% of Amendment 2's or a stop; Rand s0, Rare s0,
Rand s1, Rare s1. results/results.json after every read (a rerun resumes at the next unread one), every number in results/run.log, and Rare's X beside --decomp on the pod's disk (rebuilt if absent).

Usage (repo root, 80 GB GPU; inputs beside --decomp, or WANDB_API_KEY to fetch them): .venv/bin/python experiments/qwen_d1_teeth/run.py --decomp /root/qwen
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import rules
import torch
from safetensors.torch import save_file
from torch import Tensor

from commute import qwen_maps, split
from commute.jobs import load_experiment
from commute.qwen_sft import load
from commute.qwen_teacher import change_hooks
from commute.qwen_vpd import SITES

amend2 = load_experiment("qwen_d1_adversary", "amend2")  # gold_seqs, gold, moved; its amend1.read and run.py's set_up, start, check, stop, log, guarded
amend1, e80, e76 = amend2.amend1, amend2.run, amend2.e76
r2 = rules.r2
DEV, F64 = e80.DEV, e80.F64
RESULTS = Path(__file__).resolve().parent / "results"
PRED, OUT, STOPPED, LOG = RESULTS / "predictions.json", RESULTS / "results.json", RESULTS / "stopped.json", RESULTS / "run.log"
ORDER = (("Rand", 0), ("Rare", 0), ("Rand", 1), ("Rare", 1))
RARE = "lower"  # qwen_maps' name for the rarely-on half


def rand_change(J: Any, s: str) -> qwen_maps.SiteChange:
    M = J.dw[s].to(DEV, F64)
    return qwen_maps.gaussian_at_norm(M, rules.GAUSS_SEED + SITES.index(s)), None, None, None


@contextmanager
def rand_state(J: Any) -> Iterator[None]:
    with qwen_maps.applied(J.model, J.comps, J.alive, lambda s: rand_change(J, s)):
        yield


def rare_x(M: Tensor, U_A: Tensor, V_A: Tensor, active: Tensor, mean_g: Tensor, tol: float) -> dict[str, Tensor]:
    """E68's stage S for one half (qwen_maps/run.py `s_site`, `site_products`): the lower half by share of positions with g > 0.01, ties by mean g, and X = K⁺ δW on that half's basis."""
    pos = qwen_maps.halves(active, mean_g)[0].to(M.device)
    b = split.basis_from(split.gram_basis(U_A[:, pos]), split.gram_basis(V_A[:, pos]), tol)
    return {f"X_{RARE}": split.least_squares_x(M, b), f"pos_{RARE}": pos}


def rare_dir(J: Any) -> Path:
    return J.x_dir.parent / "e85_rare"


@torch.no_grad()
def make_rare(J: Any) -> None:
    """On the pod's disk, fp32 as stage S stored it (`to_disk`, 5.6 GB), so a rerun skips it. The halves come from W0's g statistics on the maths rows, as E68's did."""
    d, t0 = rare_dir(J), time.time()
    d.mkdir(parents=True, exist_ok=True)
    st = load(J.inputs / "W0_maths.safetensors")
    for s in SITES:
        path = d / f"{s}.safetensors"
        if not path.is_file():
            U_A, V_A = J.alive.uv(s, DEV, F64)
            idx = J.alive.idx[s].cpu()
            x = rare_x(J.dw[s].to(DEV, F64), U_A, V_A, st[f"{s}.active"][idx], st[f"{s}.mean_g"][idx], e80.TOL)
            tmp = path.with_suffix(".tmp")
            save_file(qwen_maps.to_disk(x), str(tmp))
            tmp.replace(path)
    e80.log(f"[Rare] X ready, {time.time() - t0:.0f} s, {e80.peak()}")


def rare_change(J: Any, s: str) -> qwen_maps.SiteChange:
    """`site_change("lower")`, the state E68 and E72 read under masks: weights W0 + δW, the half's columns moved by (dU, dV). Not E72's clean_lower, W0 + dσ alone, which discards the remainder."""
    U_A, V_A = J.alive.uv(s, DEV, F64)
    return qwen_maps.site_change(RARE, J.dw[s].to(DEV, F64), U_A, V_A, load(rare_dir(J) / f"{s}.safetensors", DEV).__getitem__, 0.0, 0.0)


@contextmanager
def rare_state(J: Any) -> Iterator[None]:
    make_rare(J)
    with qwen_maps.applied(J.model, J.comps, J.alive, lambda s: rare_change(J, s)):
        yield


@torch.no_grad()
def left_in_delta(J: Any) -> None:
    """Plain Σ‖Δ′ - Δ‖² / Σ‖δW‖², pooled, and the active share of the half against all alive columns', the check that the half is the rarely-on one. Both written; a rerun keeps them."""
    if "left_in_delta" not in J.out:
        make_rare(J)
        st = load(J.inputs / "W0_maths.safetensors")
        size, rand, rare, half, every, n_half, n = 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0
        for s in SITES:
            U_A, V_A = J.alive.uv(s, DEV, F64)
            M, c = J.dw[s].to(DEV, F64), rare_change(J, s)
            pos, act = c[1], st[f"{s}.active"][J.alive.idx[s].cpu()].double()
            assert pos is not None
            d, G = amend2.moved(c, U_A[:, pos], V_A[:, pos]), rand_change(J, s)[0]
            assert G is not None
            size, rand, rare = size + float((M * M).sum()), rand + float((G * G).sum()), rare + float((d * d).sum())
            half, every, n_half, n = half + float(act[pos.cpu()].sum()), every + float(act.sum()), n_half + len(pos), n + len(act)
        J.out["left_in_delta"] = {"Rand": rand / size, "Rare": rare / size}
        J.out["active_share"] = {"half": half / n_half, "alive": every / n}
        e80.save(J)
    left, act = J.out["left_in_delta"], J.out["active_share"]
    e80.log(f"[left in delta] {json.dumps(left)}; active share, half {act['half']:.6g}, alive {act['alive']:.6g}, {e80.peak()}")
    e80.check(J, "Rand_size", left["Rand"], r2.near(left["Rand"], 1.0, rules.SIZE_TOL), f"1 within {rules.SIZE_TOL}")
    e80.check(J, "Rare_left", left["Rare"], rules.rare_left_ok(left["Rare"]), f"strictly between {rules.RARE_LEFT_LOW:.4f} and {rules.RARE_LEFT_HIGH}, ~0.35")
    e80.check(J, "half_is_rarer", act["half"] / act["alive"], act["half"] < act["alive"], "under 1")


def stage_build(J: Any, seqs: list[Any]) -> None:
    """Before any start: E40's two gold readings (the rows and the file), the two states' checks and kept, W0 again (the weights are back)."""
    w0 = amend2.gold(J, seqs, "W0", lambda J: nullcontext())
    e80.check(J, "gold_W0", w0, r2.near_rel(w0, r2.W0_GOLD, r2.W0_TOL), f"E40's {r2.W0_GOLD} within {r2.W0_TOL:.1%}")
    ft = amend2.gold(J, seqs, "FT", lambda J: change_hooks(J.model, J.dw.__getitem__))
    e80.check(J, "gold_FT", w0 - ft, r2.near_rel(w0 - ft, r2.FT_GAIN, r2.FT_TOL), f"E40's gain {r2.FT_GAIN} within {r2.FT_TOL:.0%}")
    left_in_delta(J)
    rand = r2.kept(amend2.gold(J, seqs, "Rand", rand_state), w0, ft)
    e80.check(J, "kept_Rand", rand, abs(rand) <= rules.RAND_KEPT_STOP, f"~{rules.RAND_KEPT_E75} (E75, maths rows); stop beyond {rules.RAND_KEPT_STOP} either way")
    rare = r2.kept(amend2.gold(J, seqs, "Rare", rare_state), w0, ft)
    e80.check(J, "kept_Rare", rare, r2.near(rare, 1.0, r2.KEPT_TOL), f"1.00 by construction, within {r2.KEPT_TOL}")
    end = r2.kept(amend2.gold(J, seqs, "W0_end", lambda J: nullcontext()), w0, ft)
    e80.check(J, "kept_W0_end", end, r2.near(end, 0.0, r2.KEPT_TOL), f"0.00 within {r2.KEPT_TOL}")


def stage_w0(J: Any) -> None:
    for seed in rules.SEEDS:
        amend1.read(J, "W0", seed)
        got, want = J.out["starts"][f"W0_W_s{seed}"]["kl"], rules.W0_WANT[seed]
        e80.check(J, f"W0_s{seed}", got, r2.near_rel(got, want, rules.W0_TOL), f"Amendment 2's {want:.3f} within {rules.W0_TOL:.0%}")
        rescore(J)


def rescore(J: Any) -> None:
    J.out["score"] = rules.score(J.out["starts"], J.out.get("gold", {}), J.out.get("left_in_delta", {}), J.out["checks"])
    e80.save(J)
    e80.log(f"[score] {json.dumps(J.out['score'])}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decomp", required=True)
    a = ap.parse_args()
    if not PRED.is_file() or STOPPED.is_file():
        sys.exit("refusing to start: predictions.json missing, or stopped.json exists")
    seqs = amend2.gold_seqs()[0]
    J = e80.set_up(Path(a.decomp).expanduser().resolve())
    stage_build(J, seqs)
    rescore(J)
    stage_w0(J)
    for state, seed in ORDER:
        amend1.read(J, state, seed)
        rescore(J)
    e80.log(f"[done] {e80.peak()}")


def score() -> None:
    """From what results.json holds: a stop or a crash still scores, with "unread" for what never ran."""
    out = e76.read_json(OUT)
    if out is None:
        return
    out["score"] = rules.score(out["starts"], out.get("gold", {}), out.get("left_in_delta", {}), out["checks"])
    e76.write_json(OUT, out)
    e80.log(f"[score] {json.dumps(out['score'])}")


if __name__ == "__main__":
    e80.OUT, e80.STOPPED, e80.LOG = OUT, STOPPED, LOG  # E80's save, stop, check and log read these globals, so its files are never written
    amend1.STATE |= {"Rand": rand_state, "Rare": rare_state}
    e80.guarded(main, score)
