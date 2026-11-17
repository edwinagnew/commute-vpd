"""E136 (cell P2.8): does the re-gating share still separate the built pairs on a decomposition that needs no training? Plan:
reports/plans/2026-11-09_plan_e136-share-untrained-decompositions.md, card 2026-11-09_plan_p2-8-card.md. Rule, thresholds and expected values:
results/predictions.json; pairs, states and saved shares: results/inventory.json (both committed before the run; the run refuses otherwise).

The share r = (L01 - L00) / gain, the do-nothing form, on every saved state of every pair, read three ways: the trained decomposition (`local_faith.route_split`,
bf16 autocast, as filed by each pair's own experiment) and each matrix's SVD and its random rotation (`commute.untrained`, the card's gate rule, fp32 masked forward). No
verdict of reused or new: the sides are named as each pair was built.

  stage 0  the controls, alone, stop on a miss, written to results/stage0.json:
           (a) each untrained decomposition sums to W0; (b) all gates 1 on the new forward reads the unmasked forward (rounding floor measured first);
           (d) k_m from Gamma at W0 over all pairs' read windows, saved; (c) the zero update's gates are W0's and its numerator is 0 (needs k_m, so after (d));
           (e) the trained `route_split` of the first state of each pair reproduces its saved share. Exit 1 on a miss.
  stage 1  every state, three decompositions, each written to results/stage1.json as it lands; the projected time logged after the first pair (exit 1 over the
           filed limit); the per-pair gaps and the filed verdict at the end.
Each stage prints `STAGE n DONE` after its file is written; a rerun resumes from what is written.

    sbatch cluster/share_untrained.sbatch
"""

from __future__ import annotations

import json
import socket
import sys
import time
from pathlib import Path
from typing import Any

import rules
import torch
from torch import Tensor
from windows import Read, load_read

from commute import steering as ST
from commute import untrained as U
from commute import vpd
from commute.data import target_tokens
from commute.faithfulness import model_state
from commute.identity import log_odds
from commute.jobs import hard_exit, require_filed
from commute.local_faith import MIN_GAIN, route_cells, route_split

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
PRED, INV = RESULTS / "predictions.json", RESULTS / "inventory.json"
_json: dict[str, Any] = {}


def log(msg: str) -> None:
    print(msg, flush=True)


def dump(name: str, out: dict[str, Any]) -> None:
    (RESULTS / name).write_text(json.dumps(out, indent=1, sort_keys=True))


def saved(pointer: list[Any]) -> Any:
    """A saved number: [file, keypath]."""
    file, keys = pointer
    _json.setdefault(file, json.loads((REPO / file).read_text()))
    x = _json[file]
    for k in keys:
        x = x[k]
    return x


class Job:
    """The model, the two untrained decompositions and one task's W0 reference at a time (a task's Gamma gates take up to ~10 GB of host memory)."""

    def __init__(self, inv: dict[str, Any]) -> None:
        self.device = "cuda"
        self.net = ST.Net(self.device)
        self.model, self.W0, self.inv = self.net.model, self.net.W0, inv
        self.comps0 = vpd.component_weights(self.model)
        svd = U.svd_decomposition(self.W0)
        self.decomps = {"svd": svd, "rotated": U.rotated_decomposition(svd, 0)}
        self.reads: dict[str, Read] = {k: load_read(k, spec) for k, spec in inv["tasks"].items()}
        self.k: dict[str, int] = {}
        self._base: tuple[str, dict[str, Any]] | None = None
        self.w0_cells: dict[str, dict[str, float]] = {}

    def base(self, task: str) -> dict[str, Any]:
        if self._base is None or self._base[0] != task:
            self._base = None  # free the old gates before the new ones are computed
            r = self.reads[task]
            self._base = (task, route_cells(self.model, r.toks, r.half, r.token, r.batch, r.ci_len))
        return self._base[1]

    def w0(self, task: str) -> dict[str, float]:
        if task not in self.w0_cells:
            r = self.reads[task]
            self.w0_cells[task] = U.do_nothing_cells(self.model, r.toks, r.half, r.token, self.decomps, self.k, r.batch, r.ci_len)
        return self.w0_cells[task]

    def delta(self, state: str) -> dict[str, Tensor]:
        d = torch.load(REPO / self.inv["states"][state]["file"], weights_only=False)
        return {k: v.float().to(self.W0[k].device) for k, v in d.items()}

    def trained(self, state: str, dW: dict[str, Tensor]) -> dict[str, Any]:
        """`decomp_reads.read_split` with the released components, at the batch of the pair's own read."""
        r = self.reads[self.inv["states"][state]["task"]]
        base = self.base(r.name)  # at W0: it fills lazily, so never first called inside the state's context
        with model_state(self.model, target={k: self.W0[k] + dW[k] for k in self.W0}):
            return route_split(self.model, r.toks, r.half, r.token, base, self.comps0, r.batch, r.ci_len)

    def read_state(self, state: str) -> dict[str, Any]:
        task = self.inv["states"][state]["task"]
        r, dW = self.reads[task], self.delta(state)
        rs = self.trained(state, dW)
        c0, c1 = self.w0(task), U.do_nothing_cells(self.model, r.toks, r.half, r.token, self.decomps, self.k, r.batch, r.ci_len, target={k: self.W0[k] + dW[k] for k in self.W0})
        gain_fp32 = c1["T"] - c0["T"]
        share = lambda n: (c1[n] - c0[n]) / gain_fp32 if abs(gain_fp32) >= 1e-6 else float("nan")
        return {"trained": rs["shares"]["r"], "svd": share("svd"), "rotated": share("rotated"), "gain": rs["gain"], "gain_fp32": gain_fp32,
                "cells_trained": rs["cells"], "L00": {n: c0[n] for n in self.decomps}, "L01": {n: c1[n] for n in self.decomps}, "T0_fp32": c0["T"], "T1_fp32": c1["T"]}  # fmt: skip


# ----------------------------------------------------------------------------- stage 0


def check(res: dict[str, Any], name: str, value: float, tol: float) -> None:
    """A filed control: |value| <= tol (a nan misses), recorded; a miss raises Stop."""
    ok = abs(value) <= tol
    res["checks"][name] = {"value": value, "tol": tol, "pass": ok}
    log(f"[check {name}] {value:.4g} (tol {tol:g}): {ok}")
    dump("stage0.json", res)
    if not ok:
        res["stopped"] = f"{name}: {value:.6g}"
        raise Stop


class Stop(Exception):
    pass


def first_batch(r: Read, device: str) -> tuple[Tensor, Tensor]:
    """(tokens of the first batch on the card, the model's input); the input leaves any target column out."""
    b = r.toks[: r.batch].to(device)
    return b, b[:, : r.ci_len]


def mean_lo(logits: Tensor, r: Read, batch: Tensor) -> Tensor:
    return log_odds(logits[:, r.half], target_tokens(r.token, batch, r.half)).double()


def control_b(job: Job, res: dict[str, Any], T: dict[str, Any]) -> None:
    """All gates 1 reads the unmasked forward, within the filed tolerance, on each pair's first read windows; the rounding floor of the fp32 unmasked forward between a whole
    batch and its two halves is measured first and reported (a floor above the tolerance stops the job: the filed tolerance would be below rounding)."""
    rows: dict[str, dict[str, float]] = {}
    for p in job.inv["pairs"]:
        r = job.reads[job.inv["states"][p["reuse"][0]]["task"]]
        b, x = first_batch(r, job.device)
        full = mean_lo(U.logits(job.model, x), r, b)
        half = len(b) // 2
        split = torch.cat([mean_lo(U.logits(job.model, x[:half]), r, b[:half]), mean_lo(U.logits(job.model, x[half:]), r, b[half:])])
        floor = float((full - split).mean().abs())
        rec: dict[str, float] = {"unmasked_mean": float(full.mean()), "rounding_floor_mean": floor, "rounding_max_per_window": float((full - split).abs().max())}
        for n, d in job.decomps.items():
            ones = {k: torch.ones((x.shape[0], x.shape[1], d.U[k].shape[0]), dtype=torch.bool, device=x.device) for k in job.W0}
            lo = mean_lo(U.masked_forward(job.model, x, d, ones), r, b)
            rec |= {f"{n}_mean_diff": float((lo - full).mean()), f"{n}_max_per_window": float((lo - full).abs().max())}
        rows[p["id"]] = rec
        res["b_rows"] = rows
        log(f"[b {p['id']}] {json.dumps(rec)}")
        check(res, f"b rounding floor {p['id']} (a floor above the tolerance stops the job)", floor, T["ones_tol"])
        for n in job.decomps:
            check(res, f"b ones {p['id']} {n}", rec[f"{n}_mean_diff"], T["ones_tol"])


def control_d(job: Job, res: dict[str, Any]) -> None:
    """k_m: Gamma's gates at W0 over every pair's read windows, counted per matrix (no state is read before this is saved)."""
    keys, n_pos = vpd.module_keys(job.model), 0
    counts = {k: 0.0 for k in keys}
    for name, r in job.reads.items():
        with torch.no_grad():
            for s in range(0, r.toks.shape[0], r.batch):
                b = r.toks[s : s + r.batch][:, : r.ci_len].to(job.device)
                g = vpd.compute_ci(job.model, b, chunk=b.shape[0]).g
                for k in keys:
                    counts[k] += float((g[k] > 0.5).sum())
                n_pos += b.shape[0] * b.shape[1]
        log(f"[d] {name} counted, positions so far {n_pos}")
    job.k = U.k_from_counts(counts, n_pos)
    res["k_m"], res["k_m_positions"] = job.k, n_pos
    dump("stage0.json", res)
    log(f"[d] k_m {json.dumps(job.k)}")


def control_c(job: Job, res: dict[str, Any]) -> None:
    """The zero update, on the first pair's first windows: the state's gates equal W0's, and the numerator is exactly 0."""
    r = job.reads[job.inv["states"][job.inv["pairs"][0]["reuse"][0]]["task"]]
    _, x = first_batch(r, job.device)
    target = {k: w.clone() for k, w in job.W0.items()}
    _, g0 = U.forward_with_gates(job.model, x, job.decomps, job.k)
    with model_state(job.model, target=target):
        _, g1 = U.forward_with_gates(job.model, x, job.decomps, job.k)
    same = all(torch.equal(g0[n][k], g1[n][k]) for n in job.decomps for k in job.W0)
    check(res, "c zero update gates equal W0's (0 = equal)", 0.0 if same else 1.0, 0.0)
    c0 = U.do_nothing_cells(job.model, r.toks[: r.batch], r.half, r.token, job.decomps, job.k, r.batch, r.ci_len)
    c1 = U.do_nothing_cells(job.model, r.toks[: r.batch], r.half, r.token, job.decomps, job.k, r.batch, r.ci_len, target=target)
    for n in job.decomps:
        check(res, f"c zero update numerator {n}", c1[n] - c0[n], 0.0)


def control_e(job: Job, res: dict[str, Any], T: dict[str, Any]) -> None:
    """The trained route_split of the first state of each pair against its saved share."""
    res["e_rows"] = {}
    for p in job.inv["pairs"]:
        state = p["reuse"][0]
        st, rs = job.inv["states"][state], job.trained(state, job.delta(state))
        ref = float(saved(st["r"]))
        gain_ref = float(saved(st["gain"])) if st["gain"] else None
        res["e_rows"][p["id"]] = {"state": state, "r": rs["shares"]["r"], "saved_r": ref, "gain": rs["gain"], "saved_gain": gain_ref}
        log(f"[e {p['id']}] {json.dumps(res['e_rows'][p['id']])}")
        check(res, f"e reproduction {p['id']}", rs["shares"]["r"] - ref, T["reproduce_tol"])


def stage0(job: Job, T: dict[str, Any]) -> bool:
    path = RESULTS / "stage0.json"
    if path.is_file():
        old = json.loads(path.read_text())
        if old.get("pass"):
            job.k = {k: int(v) for k, v in old["k_m"].items()}
            log("[stage 0] already passed; k_m read from results/stage0.json")
            return True
    res: dict[str, Any] = {"host": socket.gethostname(), "card": torch.cuda.get_device_name(), "checks": {}, "thresholds": T}
    try:
        for n, d in job.decomps.items():
            check(res, f"a sum of components {n}", U.reconstruction_error(job.W0, d), T["sum_tol"])
        control_b(job, res, T)
        control_d(job, res)
        control_c(job, res)
        control_e(job, res, T)
    except Stop:
        dump("stage0.json", res)
        log(f"[stage 0] STOP: {res['stopped']}")
        return False
    res["pass"] = True
    dump("stage0.json", res)
    return True


# ----------------------------------------------------------------------------- stage 1


def stage1(job: Job, T: dict[str, Any], t_job: float) -> int:
    path = RESULTS / "stage1.json"
    res: dict[str, Any] = json.loads(path.read_text()) if path.is_file() else {"states": {}}
    res["not_read"] = job.inv["not_read"]
    if "stopped" in res:
        sys.exit(f"stage 1 stopped before ({res['stopped']}): move stage1.json aside to start again")
    pairs = job.inv["pairs"]
    order = list(dict.fromkeys(s for p in pairs for s in p["reuse"] + p["new"]))
    first = set(pairs[0]["reuse"] + pairs[0]["new"])
    t0, n_new = time.time(), 0
    projected = False
    for s in order:
        if s not in res["states"]:
            res["states"][s] = job.read_state(s)
            n_new += 1
            log(f"[state {s}] {json.dumps({k: res['states'][s][k] for k in ('trained', 'svd', 'rotated', 'gain', 'gain_fp32')})}")
            dump("stage1.json", res)
        if not projected and n_new and first <= set(res["states"]):
            projected = True
            minutes = ((time.time() - t0) / n_new * sum(x not in res["states"] for x in order) + (time.time() - t_job)) / 60
            res["projected_minutes"] = minutes
            log(f"[time] projected {minutes:.1f} min for the whole job, after the first pair (filed limit {T['projected_limit_min']:g})")
            if minutes > T["projected_limit_min"]:
                res["stopped"] = f"projected {minutes:.1f} min over {T['projected_limit_min']:g}"
                dump("stage1.json", res)
                return 1
    if len(res["states"]) != len(order):
        return 1
    rows = [rules.read_pair(p, res["states"], T["min_gain"], T["shrink"]) for p in pairs]
    res["pairs"], res["verdict"] = rows, rules.verdict(rows, T["min_scored"])
    dump("stage1.json", res)
    for r in rows:
        log(f"[pair {r['id']}] {json.dumps(r)}")
    log(f"[not read] {json.dumps(res['not_read'])}")
    log(f"[verdict] {json.dumps(res['verdict'])}")
    return 0


def main() -> None:
    t_job = time.time()
    require_filed(PRED)
    require_filed(INV)
    P, inv = json.loads(PRED.read_text()), json.loads(INV.read_text())
    T = P["thresholds"]
    assert T["min_gain"] == MIN_GAIN and P["experiment"] == "E136"
    missing = [s["file"] for s in inv["states"].values() if not (REPO / s["file"]).is_file()] + [t["path"] for t in inv["tasks"].values() if "path" in t and not (REPO / t["path"]).is_file()]
    if missing:
        sys.exit(f"files missing: {missing}")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    job = Job(inv)
    log(f"[host] {socket.gethostname()} {torch.cuda.get_device_name()} tasks {len(job.reads)} states {len(inv['states'])}")
    if not stage0(job, T):
        hard_exit(1)
    print("STAGE 0 DONE", flush=True)
    code = stage1(job, T, t_job)
    if code == 0:
        print("STAGE 1 DONE", flush=True)
    hard_exit(code)


if __name__ == "__main__":
    main()
