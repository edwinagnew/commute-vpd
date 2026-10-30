"""Job 1 (`--stage measure`, after filing A, before filing B): measurement only, no instrument reads a scored state.
Plan § 5 job 1: the stop rows (files, tokens, G, zero update), D / off-answer KL / MLP share per state, the prior per
map, the random-removal spread on the pole states, the planted wrapper built and checked. Writes results/measure.json
after each section, with `filing_B_draft`: the measured values filing B is written from."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from common import (
    EXIT_STOP,
    OLD_SEEDS,
    RESULTS,
    Tasks,
    dump,
    e58_records,
    e58_windows_sha,
    load_state,
    load_tasks,
    log,
    rank1_ref,
    state_path,
    stop,
    windows_path,
)

from commute import keydep as KD
from commute import revival_kv as KV
from commute import steering as ST
from commute.finetune import ReferenceConfig
from commute.heldout_keys import PARTS, gain
from commute.vpd import target_weights

OUT = RESULTS / "measure.json"
R_FRAC = 0.8  # constants.planted.pass.R_min, "0.8 x ceiling" (checked in planted_checks)


def check_files(out: dict[str, Any]) -> str | None:
    """Stop rows 1: the 12 states' and 8 windows files' sha256, and identical tokens across the four maps."""
    rec, wsha = e58_records(), e58_windows_sha()
    got = {n: KD.sha256(state_path(n.split("_s")[0], int(n.split("_s")[1]))) for n in rec}
    wgot = {m: {p: KD.sha256(windows_path(m, p)) for p in PARTS} for m in KV.MAPS}
    ws = {m: torch.load(windows_path(m, "trained"), weights_only=False) for m in KV.MAPS}
    same = {part: all(torch.equal(ws[m][part].tokens, ws["rank1"][part].tokens) for m in KV.MAPS) for part in ("train", "heldout")}
    out["files"] = {"state_sha_ok": {n: got[n] == rec[n]["ft"]["sha256"] for n in rec}, "windows_sha_ok": wgot == wsha,
                    "identical_tokens": same, "n_train": len(ws["rank1"]["train"]), "n_heldout": len(ws["rank1"]["heldout"])}  # fmt: skip
    log(f"[files] {out['files']}")
    bad = [n for n, ok in out["files"]["state_sha_ok"].items() if not ok]
    return f"state files differ from E58's record: {bad}" if bad else None if wgot == wsha and all(same.values()) else "a windows file or the tokens differ"


def per_state(net: ST.Net, ref: KV.Ref, tasks: Tasks, C: dict[str, Any], out: dict[str, Any]) -> str | None:
    """Stop row 2 (G within G_tol) and item 3 (D, off-answer KL, MLP share) on the 12 states; the zero update first."""
    zero = {k: torch.zeros_like(w) for k, w in net.W0.items()}
    out["zero"] = {"D": KV.kappa(net, ref, zero)}
    log(f"[zero update] D {out['zero']['D']!r}")
    if out["zero"]["D"] != 0.0:
        return "the zero update's D is not exactly 0"
    rec, st = e58_records(), out["states"]
    for m in KV.MAPS:
        for s in OLD_SEEDS:
            n, dW = f"{m}_s{s}", load_state(m, s)
            G = float(gain(net, tasks[m]["trained"], dW).mean())
            st[n] = {"G": G, "G_e58": rec[n]["G"], "G_ok": abs(G - rec[n]["G"]) <= C["G_tol"], "D": KV.kappa(net, ref, dW),
                     "off_kl": KV.off_answer_kl(net, ref, dW), "mlp_share": KV.mlp_share(dW), "norm": rec[n]["ft"]["norm"]}  # fmt: skip
            log(f"[state {n}] G {G:.6f} (E58 {rec[n]['G']:.6f}) D {st[n]['D']:.4f} off-answer KL {st[n]['off_kl']:.5f} MLP share {st[n]['mlp_share']:.4f}")
            dump(OUT, out)
            if not st[n]["G_ok"]:
                return f"{n}: G {G:.6f} differs from E58's {rec[n]['G']:.6f} by more than {C['G_tol']}"
    return None


def spread(net: ST.Net, ref: KV.Ref, C: dict[str, Any], out: dict[str, Any]) -> float:
    """Item 4: per pole state and K, the sd over the random removals of (kappa - kappa_random) / D; R's floor = the max."""
    for m in KV.POLES:
        for s in OLD_SEEDS:
            n = f"{m}_s{s}"
            rd = KV.prune_readings(net, ref, load_state(m, s), C["prune_k"], C["n_random"], C["prune_seed"], None)
            out["spread"][n] = {"readings": rd, **KV.revival(rd, C["prune_k"], rd["kappa"])}
            log(f"[spread {n}] D {rd['kappa']:.4f} sd of random-removal share by K {out['spread'][n]['spread']}")
            dump(OUT, out)
    return max(v for r in out["spread"].values() for v in r["spread"].values())


def d_void(st: dict[str, Any], C: dict[str, Any]) -> float:
    """D_void_frac x the smallest pole-map mean D (seeds 0-2)."""
    return C["D_void_frac"] * min(float(np.mean([st[f"{m}_s{s}"]["D"] for s in OLD_SEEDS])) for m in KV.POLES)


def planted(net: ST.Net, ref: KV.Ref, tasks: Tasks, C: dict[str, Any], out: dict[str, Any]) -> dict[str, Any]:
    """Item 5 as amended (Amendment 1): answer-selective units, edit -alpha * w, alpha by bisection, ceiling, D floor."""
    pc, st = C["planted"], out["states"]
    u = KV.planted_units(net, ref, pc["block"], pc["n_units"], pc["active_quantile"])
    t = tasks["rank1"]["trained"]
    w = KV.rank1_direction(net, t.tgt(t.toks[t.part("fit")]))
    target = float(np.median([st[f"{m}_s{s}"]["D"] for m in KV.POLES for s in OLD_SEEDS]))
    bound = max(r["off_kl"] for r in st.values())
    floor = max(d_void(st, C), pc["D_floor_frac"] * target)
    dw = lambda a: KV.planted_dw(net, u["units"], w, a)
    fit = KV.fit_alpha(lambda a: KV.kappa(net, ref, dw(a)), lambda a: KV.off_answer_kl(net, ref, dw(a)), target, bound)
    kz = KV.kappa(net, ref, edit=KV.zero_units(u["units"]))
    rec = {**u, "w": w.tolist(), "target_D": target, "off_bound": bound, "floor": floor, **fit, "kappa_zeroed": kz,
           "ceiling": (fit["kappa"] - kz) / fit["kappa"] if fit["kappa"] > 0 else math.nan}  # fmt: skip
    log(f"[planted] units {u['units']} a_ans {[round(a, 3) for a in u['a_ans']]} selectivity {[round(a, 2) for a in u['selectivity']]} "
        f"target D {target:.4f} off bound {bound:.5f} floor {floor:.4f} -> alpha {fit['alpha']:.5g} D {fit['kappa']:.4f} off "
        f"{fit['off']:.5f} fallback {fit['fallback']} kappa_zeroed {kz:.4f} ceiling {rec['ceiling']:.4f}")
    return rec


def planted_checks(net: ST.Net, ref: KV.Ref, w0: torch.nn.Module, train: torch.Tensor, rec: dict[str, Any], C: dict[str, Any],
                   deadline: float | None) -> dict[str, Any]:
    """Item 6 (and job 3's stage 0): pruning and reverse fine-tuning on the planted state. `rec` holds units, alpha,
    w and floor; under the floor the pruning control reads "void" (Amendment 1)."""
    pc = C["planted"]
    assert pc["n_units"] in C["prune_k"], "the planted pass reads the top-n_units ranking"
    assert pc["pass"]["R_min"] == f"{R_FRAC} x ceiling", "the filed planted R_min changed"
    w = torch.tensor(rec["w"])
    dW = KV.planted_dw(net, rec["units"], w, rec["alpha"])
    rd = KV.prune_readings(net, ref, dW, C["prune_k"], C["n_random"], C["prune_seed"], KV.unit_scores(net, ref, dW))
    rv = KV.revival(rd, C["prune_k"], rd["kappa"])
    kz = KV.kappa(net, ref, edit=KV.zero_units(rec["units"]))
    ceiling = (rd["kappa"] - kz) / rd["kappa"] if rd["kappa"] > 0 else math.nan
    pp = KV.planted_pass(rd[str(pc["n_units"])]["units"], rec["units"], rv["R"], ceiling, pc["pass"]["found_in_top13_min"], R_FRAC,
                         rd["kappa"], rec["floor"])  # fmt: skip
    log(f"[planted pruning] D {rd['kappa']:.4f} (floor {rec['floor']:.4f}, built {pp['built']}) found {pp['found']} of {pc['n_units']} "
        f"R {rv['R']:.4f} vs {pp['R_min']:.4f} (ceiling {ceiling:.4f}) -> {pp['reading']}")
    rcfg = reverse_cfg(C, 0)
    rev = KV.reverse_curve(net, ref, dW, w0, train, rcfg, deadline)
    back = all(torch.equal(v, net.W0[k]) for k, v in target_weights(net.model).items())
    log(f"[planted reverse] t_half {rev['t_half']} ({rev['end']}, {len(rev['curve']) - 1} steps) | weights back at W0 {back}")
    return {"D": rd["kappa"], "prune": rd, **rv, "kappa_zeroed": kz, "ceiling": ceiling, "prune_pass": pp,
            "reverse": rev, "reverse_pass": rev["end"] == "half", "w0_restored": back}  # fmt: skip


def reverse_cfg(C: dict[str, Any], seed: int) -> ReferenceConfig:
    r = C["reverse"]
    return ReferenceConfig(lr=r["lr"], batch=r["batch"], steps=r["cap_steps"], grad_clip=r["grad_clip"], seed=seed, optimizer=r["optimizer"])


def draft_b(out: dict[str, Any], C: dict[str, Any], floor_R: float) -> dict[str, Any]:
    """The measured values filing B is written from (seeds 0-2): per-map means, p_D and p_prior with their rule-1/3/4
    readings, R's floor, D's void floor and the planted settings."""
    st = out["states"]
    col = lambda key: {m: [st[f"{m}_s{s}"][key] for s in OLD_SEEDS] for m in KV.MAPS}
    prior = {m: [out["prior"][m]] * len(OLD_SEEDS) for m in KV.MAPS}
    means = lambda c: {m: float(np.mean(v)) for m, v in c.items()}
    pl = out["planted"]
    return {"D_mean": means(col("D")), "D_void": d_void(st, C),
            "prior": out["prior"], "mlp_share_mean": means(col("mlp_share")), "off_kl_max": max(r["off_kl"] for r in st.values()),
            "p_D": KV.column_reading(col("D"), None, 0.0, C), "p_prior": KV.column_reading(prior, None, 0.0, C),
            "noise_floor_R": floor_R,
            "planted": {k: pl[k] for k in ("units", "a_ans", "selectivity", "alpha", "kappa", "kappa_zeroed", "ceiling", "fallback", "target_D", "floor")},
            "planted_reading": out["planted_check"]["prune_pass"]["reading"]}  # fmt: skip


def stage_measure(net: ST.Net, C: dict[str, Any], out: dict[str, Any], deadline: float | None) -> int:
    out |= {"states": {}, "spread": {}}
    why = check_files(out)
    if why:
        return stop(OUT, out, why)
    tasks = load_tasks(net)
    ref, train = rank1_ref(net)
    if (why := per_state(net, ref, tasks, C, out)) is not None:
        return stop(OUT, out, why)
    out["prior"] = {m: float(tasks[m]["trained"].lp0[tasks[m]["trained"].part("score")].mean()) for m in KV.MAPS}
    log(f"[prior] mean log p_W0(value), score half: {out['prior']}")
    floor_R = spread(net, ref, C, out)
    log(f"[spread] R's noise floor {floor_R:.5f}")
    out["planted"] = planted(net, ref, tasks, C, out)
    dump(OUT, out)
    out["planted_check"] = planted_checks(net, ref, KV.w0_copy(net), train, out["planted"], C, deadline)
    out["filing_B_draft"] = draft_b(out, C, floor_R)
    dump(OUT, out)
    log(f"[filing B draft] {out['filing_B_draft']}")
    return 0 if out["planted_check"]["w0_restored"] else EXIT_STOP
