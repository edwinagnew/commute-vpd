"""E137 (cell P1.3): can the second wording of reuse be tested on pairs whose label is known? Plan:
reports/plans/2026-11-09_plan_reuse-variable-first-test.md, apparatus in its appendix section D; wording under test
~/commute-data/scratch/definition-session/2026-11-09_reuse-at-a-variable_v2.md. Rules and bars: results/predictions.json (the run
refuses unless it is committed unchanged). The original model M is 'commute.steering.Net'; a pair is M and M' = M + one added
piece at block 2's input, last position (`pieces.py`).

Stages, cheapest first; each writes results/stage<n>.json and then prints `STAGE n DONE`; a rerun resumes from what is written:
  0  alone: float floors; M' at start equals M (a miss stops the job, exit 1); train and build the pieces; each pair's gap against
     the floor; the capacity read (`search.capacity`).
  1  exact readings at the witnesses (G, H, reversion, setter mismatch) and the first wording's exact checks.
  2  the search: G by Adam on KL(F || N), three starts, four sites, k 1, 4, 16; H by ridge from z_M then 200 KL steps, at those sites and
     `final` (the normed input of the unembedding, where KL is convex in the map); d read on the read half. Written after each search; a clean exit 3 near the job's limit leaves it to the next job.
  3  the acted-on count: 100 random unit directions per site on E. results/rules.json then applies the filed rules.
  last step (not a stage; after the rules): results/rank.json, the rank of z_M per pair (XOR first), on MAGMA; written after each pair.
Exit 0 done, 1 a stop or refusal, 3 more to do.

    sbatch cluster/reuse_variable_test.sbatch
"""

from __future__ import annotations

import json
import time
import zlib
from typing import Any

import pieces as PC
import rules as RU
import search as SE
import torch
from common import PRED, MoreToDo, dump, elapsed, f_edit_of, free, load, log
from controls import stage0
from exact import stage1
from torch import Tensor

from commute.hidden_maps import chunks, flat, run, tv, write_last
from commute.jobs import hard_exit, require_filed
from commute.steering import Net

PAIRS, SITES = PC.PAIRS, SE.SITES


def setup(net: Net, cfg: PC.Config) -> tuple[PC.Sets, PC.Task, int]:
    from commute import vpd

    tok = vpd.tokenizer()
    sets = PC.make_sets(cfg, tok, log)
    vocab = int(net.net.lm_head.weight.shape[0])  # type: ignore[union-attr]
    task = PC.make_task(sets, tok, vocab, cfg)
    log(f"rows ready: E {sets.E.shape}, cue {task.cue!r}, {len(task.keys)} keys, vocabulary {vocab}")
    return sets, task, vocab


def stage2(net: Net, cfg: PC.Config, sets: PC.Sets, task: PC.Task, p: PC.Pieces, st0: dict[str, Any]) -> dict[str, Any]:
    out = load(cfg, "stage2.json") or {"G": {}, "H": {}, "complete": False}
    if out.get("complete"):
        log("stage 2 read from file")
        return out
    for pair in (x for x in PAIRS if st0["pairs"][x]["reading"]):
        gap = st0["pairs"][pair]["gap"]
        fit, read = PC.pair_windows(cfg, pair, sets, task)
        fedit = f_edit_of(net, p, pair)
        for k in cfg.k_figure:
            for site in SITES:
                for start in range(cfg.starts):
                    key = f"{pair}/k{k}/{site}/s{start}"
                    if key in out["G"]:
                        continue
                    _budget(cfg, out)
                    t0 = time.monotonic()
                    seed = 137_000 + zlib.crc32(key.encode()) % 65_536  # not hash(): a resume must draw the same seeds
                    found = SE.g_search(net, fit, site, k, start, fedit, cfg, seed)
                    A, c = found["A"], found["c"]
                    r = SE.read_pair(net, read, site, found["Q"], lambda v, _s, A=A, c=c: v @ A + c, fedit, cfg)
                    out["G"][key] = {"G": 1 - r["d"] / gap, "reversion": 1 - r["d_rev"] / gap, "mismatch": r["mismatch"], "fit_kl": found["last_kl"], "s": time.monotonic() - t0}
                    dump(cfg, "stage2.json", out)
                    log(f"G {key}: read G {out['G'][key]['G']:.4f}, reversion {out['G'][key]['reversion']:.4f}, mismatch {r['mismatch']:.4f}, fit KL {found['last_kl']:.5f}")
        _h_pair(net, cfg, pair, fit, read, fedit, gap, out)
        free()
    out["complete"] = True
    dump(cfg, "stage2.json", out)
    print("STAGE 2 DONE", flush=True)
    return out


def _h_pair(net: Net, cfg: PC.Config, pair: str, fit: Tensor, read: Tensor, fedit: Any, gap: float, out: dict[str, Any]) -> None:
    """H at each site for one pair: one ridge accumulation (the expensive part), then per site the penalty, 200 KL steps and the read."""
    missing = [s for s in SE.H_SITES if f"{pair}/{s}" not in out["H"]]
    if not missing:
        return
    _budget(cfg, out)
    ridge, X_val, Y_val, ps = SE.h_fit(net, fit, fedit, cfg, log)
    n_val_start = fit.shape[0] * 4 // 5
    for site in missing:
        _budget(cfg, out)
        t0 = time.monotonic()
        beta, lam, mse = SE.ridge_select(ridge, site, ps[site], X_val, Y_val[site], cfg.ridge_grid)
        b, y, info = SE.h_refine(net, fit[:n_val_start], fit[n_val_start:], site, ridge, beta, fedit, cfg, 137_800 + SE.H_SITES.index(site))
        r = SE.read_pair(net, read, site, None, _ridge_s(ridge, site, b, y), fedit, cfg)
        key = f"{pair}/{site}"
        out["H"][key] = {"H": 1 - r["d"] / gap, "reversion": 1 - r["d_rev"] / gap, "mismatch": r["mismatch"], "lam": lam, "val_mse": mse, "s": time.monotonic() - t0, **info}
        dump(cfg, "stage2.json", out)
        log(f"H {key}: read H {out['H'][key]['H']:.4f}, reversion {out['H'][key]['reversion']:.4f}, mismatch {r['mismatch']:.4f}, lam {lam}, val KL ridge {info['kl_val_ridge']:.5f} refined {info['kl_val_refined']:.5f} ({info['used']})")


def _ridge_s(ridge: Any, site: str, b: Tensor, y: Tensor) -> Any:
    def s_fn(_v: Tensor, seen: dict[str, Tensor]) -> Tensor:
        return y + ridge.standardise(flat(seen, site)) @ b

    return s_fn


def _budget(cfg: PC.Config, out: dict[str, Any]) -> None:
    if elapsed() > cfg.wall_limit_s:
        dump(cfg, "stage2.json", out)
        raise MoreToDo


# --- Stage 3 and the rules ---------------------------------------------------------------------------------------


def stage3(net: Net, cfg: PC.Config, sets: PC.Sets, st0: dict[str, Any]) -> dict[str, Any]:
    done = load(cfg, "stage3.json")
    if done is not None:
        log("stage 3 read from file")
        return done
    E = sets.E
    gen = torch.Generator().manual_seed(137_900)
    perm = torch.randperm(E.shape[0], generator=gen)
    out: dict[str, Any] = {"sites": {}, "tol_tv": st0["tol_tv"]}
    for site in SITES:
        h = torch.cat([run(net, t, sites=(site,))[1][site][:, -1] for t in chunks(E, cfg.read_batch)]).double()
        sv = torch.linalg.svdvals(h)
        rank = int((sv > 1e-6 * sv[0]).sum())
        U = torch.linalg.svd(h.float(), full_matrices=False)[2][:rank].T  # the span's basis, (d, rank)
        ms = torch.cat([run(net, t)[0][:, -1] for t in chunks(E, cfg.read_batch)])
        means: list[float] = []
        for _ in range(cfg.directions):
            u = U @ torch.randn(rank, generator=gen).to(U.dtype).to(U.device)
            u = u / u.norm()
            d_tot = 0.0
            for i in range(0, E.shape[0], cfg.read_batch):
                t = E[i : i + cfg.read_batch]
                hi, hp = h[i : i + cfg.read_batch].float(), h[perm][i : i + cfg.read_batch].float()
                delta = ((hp - hi) * u).sum(-1, keepdim=True) * u
                lv = run(net, t, edit={site: write_last(None, hi + delta)})[0][:, -1]
                d_tot += float(tv(lv, ms[i : i + cfg.read_batch]).sum())
            means.append(d_tot / E.shape[0])
        out["sites"][site] = {"rank_E": rank, "mean_d": means, "acted_on": RU.acted_on_count(means, st0["tol_tv"]),
                              "directions": cfg.directions, "min_mean_d": min(means), "median_mean_d": sorted(means)[len(means) // 2]}
        log(f"stage 3 {site}: rank on E {rank}; directions above the floor {out['sites'][site]['acted_on']} of {cfg.directions}; min mean d {min(means):.3g}, median {sorted(means)[len(means) // 2]:.3g}")
    dump(cfg, "stage3.json", out)
    print("STAGE 3 DONE", flush=True)
    return out


def rule_read(cfg: PC.Config, filed: dict[str, Any], st0: dict[str, Any], st1: dict[str, Any], st2: dict[str, Any], st3: dict[str, Any]) -> dict[str, Any]:
    """Applies the filed rules, in results/rules.json. Numbers only; no verdict on the wordings beyond the filed by-construction table."""
    th = filed["thresholds"]
    reads: dict[str, Any] = {}
    witness_checks: dict[str, Any] = {}
    for pair in PAIRS:
        p1 = st1["pairs"][pair]
        if p1.get("no_reading"):
            witness_checks[pair] = "no reading"
            continue
        w = p1["witness"]
        t_s = p1["tol_share"]
        c = {}
        if pair == "Always on":
            c = {"G": RU.within(w["G"], 1.0, t_s), "H": RU.within(w["H"], 1.0, t_s), "reversion": RU.within(w["reversion"], 1.0, t_s),
                 "mismatch": RU.within(w["mismatch"], 0.0, p1["tol_mismatch"])}
        elif pair in ("Cue", "Table"):
            c = {"H": RU.within(w["H"], 1.0, t_s)}
        witness_checks[pair] = c
    reads["witness_checks"] = witness_checks

    g_by: dict[str, dict[str, list[float]]] = {p: {} for p in PAIRS}
    h_best: dict[str, float | None] = {}
    for pair in PAIRS:
        if not st0["pairs"][pair]["reading"]:
            h_best[pair] = None
            continue
        for site in SITES:
            vals = [st2["G"][f"{pair}/k{cfg.k}/{site}/s{s}"]["G"] for s in range(cfg.starts) if f"{pair}/k{cfg.k}/{site}/s{s}" in st2["G"]]
            if vals:
                g_by[pair][site] = vals
        hs = [st2["H"][f"{pair}/{site}"]["H"] for site in SE.H_SITES if f"{pair}/{site}" in st2["H"]]
        h_best[pair] = max(hs) if hs else None
    per_pair: dict[str, Any] = {}
    for pair in PAIRS:
        site, vals = RU.best_site(g_by[pair])
        per_pair[pair] = {"best_site": site, "starts": vals, "mean_G": None if vals is None else sum(vals) / len(vals), "best_G": max(vals) if vals else None,
                          "best_H": h_best[pair]}
    reads["k4"] = per_pair
    reads["card_rule"] = RU.card_rule({p: per_pair[p]["starts"] for p in PAIRS}, th)
    wit_g, wit_cue_h, wit_table_h = (st1["pairs"][p].get("witness", {}) if not st1["pairs"][p].get("no_reading") else {} for p in ("Always on", "Cue", "Table"))
    reads["search_bars"] = {"G_always_on": RU.search_bar(per_pair["Always on"]["best_G"], wit_g.get("G"), th),
                            "H_cue": RU.search_bar(h_best["Cue"], wit_cue_h.get("H"), th), "H_table": RU.search_bar(h_best["Table"], wit_table_h.get("H"), th)}
    accs = list(st0["capacity"]["held_out_real_by_site"].values())
    reads["capacity"] = {"built_possible": RU.built_possible(accs, th), "held_out_real_by_site": st0["capacity"]["held_out_real_by_site"]}
    reads["acted_on"] = {s: {"count": v["acted_on"], "of": v["directions"], "rank_E": v["rank_E"]} for s, v in st3["sites"].items()}
    reads["filed"] = RU.rule_summary(filed)
    out = {"reading": reads, "note": "no verdict, no cut-offs for labels; by-construction table copied from the filed file"}
    dump(cfg, "rules.json", out)
    log("rules: " + json.dumps(reads, default=float))
    print("RULES DONE", flush=True)
    return out


def rank_step(net: Net, cfg: PC.Config, sets: PC.Sets, task: PC.Task) -> None:
    """Rank of z_M per pair, reported only (no filed rule reads it). Last, so a solver failure loses only this."""
    torch.backends.cuda.preferred_linalg_library("magma")  # last step of the job: a global setting, nothing after it uses linalg
    out: dict[str, Any] = {}
    for pair in ("XOR", *[q for q in PAIRS if q != "XOR"]):
        fit, read = PC.pair_windows(cfg, pair, sets, task)
        out[pair] = SE.z_rank(net, torch.cat([fit, read])[: cfg.rank_windows], cfg)
        r = out[pair]
        log(f"rank of z_M, pair {pair}: {r['rank']} of {r['columns']} columns on {r['windows']} windows, largest singular value {r['sv_max']:.3g}")
        dump(cfg, "rank.json", out)
        free()
    print("RANK DONE", flush=True)


def main(cfg: PC.Config, net: Net) -> int:
    filed = json.loads(PRED.read_text())
    net.net.requires_grad_(False)
    sets, task, _ = setup(net, cfg)
    (cfg.results).mkdir(parents=True, exist_ok=True)
    p, st0 = stage0(net, cfg, sets, task, filed)
    st1 = stage1(net, cfg, sets, task, p, st0, filed)
    try:
        st2 = stage2(net, cfg, sets, task, p, st0)
    except MoreToDo:
        log("more to do: stage 2 partly written; rerun to resume")
        return 3
    st3 = stage3(net, cfg, sets, st0)
    rule_read(cfg, filed, st0, st1, st2, st3)
    rank_step(net, cfg, sets, task)
    return 0


if __name__ == "__main__":
    require_filed(PRED)
    code = 1
    try:
        code = main(PC.Config(), Net("cuda"))
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)
