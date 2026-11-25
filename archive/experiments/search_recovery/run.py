"""E188 (cell P1.21): does a repaired ranking search recover the built rungs of E184's built items? Plan:
reports/plans/2026-11-16_plan_search-recovery.md; card: card.md; filed rules: results/predictions.json (the run refuses
unless it is committed unchanged). Every reading is under the fourth wording, ranked, provisional (testing only, not adopted).
The items, their F, windows, list and search are E184's (experiments/simplest_intervention/run.py, commute.rank_read),
loaded, not copied; nothing in rank_read changes.

Stage 1, the oracle (no search). Each built edit is h_in2 += L push at the last position, L = 1 (planted constant), a linear
read of in1 (planted trigger, cue: exact) or, for the fine-tunes, their teacher's token-set label (twin: odd last number,
fact: its random number set; approximated at in1 by the class-mean direction). On the same windows as E184:
    o3      the edit as one rank-one edit (in1 -> in2) written by hand: exact for the three hand-built items (no fit); for the
            fine-tunes the teacher's form refitted (Adam, o3_steps)
    forms   the edit rewritten on M's list with at most 16 names: writes = the W in2 principal directions most along the
            push ("pcs") or the W logits whose mean probability F moves most ("out"); reads = the R list variables before
            in2 (and the tokens at the last two positions) most correlated with L; numbers started from the least-squares
            fit (each write's component of the push times L regressed on the reads) and fitted by L-BFGS. W x (1 + R) <= 16; R = 0 is rung 1, R > 0 rung 2
    ranking where the best list form's writes stood in E184's first greedy step (shift score rank, the K-th candidate's
            score, the fitted shift over the Newton step)
Each item's miss at each bar is put down to the search (an oracle at its filed rung closes the bar), the class (only the
rank-one form does) or neither ("oracle short"); "rung 3 search miss" where the rank-one form closes a bar E184 read rung 4.
Controls first and alone: do-nothing, one neuron (rung 1), one neuron triggered by in1.pc0 (rung 2), each written by hand.
Stage 2, only if some item's miss is put down to the search: E184's search on list all with the repaired sizes (K, line-search
multiples, rung-3 steps), the controls first (one neuron, triggered neuron, random bit must read as in E184), then those items.
Exit 0 done, 1 a stop, 3 more to do (--wall-min; fleet/resubmit). Every item resumes from its results JSON.

    python experiments/search_recovery/run.py --wall-min 600      (pod; on the debug QoS: cluster/search_recovery.sbatch)
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

import commute.rank_read as RK
import commute.variable_read as VR
from commute import soft_read as SR
from commute.jobs import MORE_TO_DO, hard_exit, load_experiment, require_filed

E: Any = load_experiment("simplest_intervention", "run")  # E184: setup, items, controls, search_list, log, dump, stop
SI = E.SI  # E173: the planted items' saved s, E154's alpha, the fact's numbers
EXPERIMENT = "E188"
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED = RESULTS / "predictions.json"
FILED = {"planted_constant": 1, "planted_trigger": 2, "cue": 2, "twin_s0": 2, "twin_s1": 2, "fact_s0": 3, "fact_s1": 3}  # E184's
EXACT = ("planted_constant", "planted_trigger", "cue")  # one rank-one edit is the built edit itself
CONTROLS = ("do_nothing", "l1_neuron", "l2_triggered")
REPAIR_CONTROLS = ("l1_neuron", "l2_triggered", "l4_bit")
NN = len(RK.MLP) * RK.D_MLP
log, dump = E.log, E.dump


@dataclass(frozen=True)
class Config:
    results: Path = RESULTS
    wall_s: float = 26.5 * 60
    oracle_lbfgs: int = 50  # L-BFGS iterations of an oracle form's fit (E184's search refit: 20)
    o3_steps: int = 400  # Adam steps of a fine-tune's rank-one refit (E184's rung 3: 80)
    splits: tuple[tuple[int, int], ...] = ((16, 0), (4, 3), (2, 7), (1, 15))  # (writes, reads per write)
    K: int = 16  # repaired: candidates line-searched per step (E184: 4)
    mults: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)  # repaired line search (E184: 0.25 to 2)
    quick3: int = 60  # repaired rung-3 screening steps (E184: 30)
    steps3: int = 400  # repaired rung-3 full-fit steps (E184: 80)
    seed: int = 188_000

    def __post_init__(self) -> None:
        assert self.results == RESULTS and all(w * (1 + r) <= 16 for w, r in self.splits) and 1.0 in self.mults

    def e184(self) -> Any:
        return E.Config(wall_s=self.wall_s)

    def oracle(self) -> Any:
        return replace(self.e184(), lbfgs_iter=self.oracle_lbfgs)

    def repaired(self) -> Any:
        return replace(self.e184(), K=self.K, mults=self.mults, quick3=self.quick3, steps3=self.steps3)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.is_file() else {}


def e184_rec(name: str) -> dict[str, Any]:
    return json.loads((E.RESULTS / f"{name}.json").read_text())


def target(c: Any, name: str) -> tuple[VR.Target, Tensor, Tensor]:
    """E184's F and windows: a control at its saved s (no recalibration), else E173's item."""
    if name.startswith("l"):
        s = e184_rec(name)
        return E.control(c, name, {k: s[k] for k in ("s", "gap", "target_gap")}), c.e.fit, c.e.read
    return E.e173_item(c, name)[:3]


# ---- the built edits ------------------------------------------------------------------------------------------------


@dataclass
class Built:
    """h_in2 += L push (last position). L = 1 if `dir` is None, else (h_in1 . dir - p0) / (p1 - p0); `label`, a fine-tune
    teacher's token-set label, replaces L where the reads are ranked (dir is then its class-mean direction at in1)."""

    push: Tensor
    dir: Tensor | None = None
    p0: float = 0.0
    p1: float = 1.0
    label: Callable[[Tensor], Tensor] | None = None


def built(c: Any, name: str) -> Built:
    e, dev = c.e, c.e.unit.device
    if name in ("planted_constant", "planted_trigger"):  # soft_read.planted_constant / planted_trigger at E173's s
        s = json.loads((SI.RESULTS / f"{name}.json").read_text())["s"]
        if name == "planted_constant":
            return Built(s * e.unit)
        d, p0, p1 = e.parity_dir()
        return Built(s * e.unit, d.to(dev).float(), p0, p1)
    if name == "cue":  # variable_read.piece_edit "Cue": (h_in1 . w + a) Q b
        w, a = e.p.w.to(dev).float(), float(e.p.a)
        nw = float(w.norm())
        return Built(e.unit, w / nw, -a / nw, (1 - a) / nw)
    kind, _ = name.split("_s")  # E154's control arm and E168's fact: teacher = M + alpha label(last token) Q b at in2
    x = e.x if kind == "twin" else SI.MF.fact_numbers(e.x, e.cfg.fact_seed)

    def lab(t: Tensor) -> Tensor:
        return VR.odd_at(x, t, -1)

    d, p0, p1 = SR.class_dir(e.net, e.fit, lab, "in1", -1, e.cfg.read_batch)
    alpha = json.loads((SI.E154 / "amend1_stage1.json").read_text())["calibration"]["alpha"]
    return Built(alpha * e.unit, d.to(dev).float(), p0, p1, lab)


def rank_one(m: RK.Model, b: Built) -> RK.RankOne:
    """The built edit as h_in2 += rms_in2 d (a . (h_in1 - mu_in1) / rms_in1 + c): exact when L is linear in h_in1."""
    st, d = m.st, b.push / m.st.res_rms["in2"]
    if b.dir is None:
        return RK.RankOne("in1", "in2", torch.zeros_like(d), d, d.new_ones(()))
    k = b.p1 - b.p0
    return RK.RankOne("in1", "in2", st.res_rms["in1"] * b.dir / k, d, ((st.res_mu["in1"] @ b.dir - b.p0) / k).reshape(()))


def share(P: Tensor, v: Tensor | None) -> float | None:
    """Share of v's squared norm in the span of P's orthonormal columns."""
    return None if v is None else float(((P.T @ v) ** 2).sum() / (v**2).sum())


# ---- the list forms -------------------------------------------------------------------------------------------------


@torch.no_grad()
def windows_table(m: RK.Model, d: RK.Data, b: Built, sz: Any) -> dict[str, Tensor]:
    """On the scoring windows, M alone: each dense read standardised (mean 0, sd 1 there, as M_I reads it), L, the tokens at
    TOK_POS, and each logit's mean p_F - p_M."""
    nd, U, L, dp = m.sp.n_dense, [], [], torch.zeros(m.sp.V, device=m.net.device)
    for i in range(0, d.score.shape[0], sz.eval_batch):
        t = d.score[i : i + sz.eval_batch]
        _, lq, store = RK.forward(m, RK.Interv(), t)
        U.append((RK.dense_cols(m.sp, store) - m.st.mu[:nd]) / m.st.sd[:nd])
        h1 = store["in1"]
        L.append(b.label(t).to(h1.device) if b.label else torch.ones_like(h1[:, 0]) if b.dir is None else (h1 @ b.dir - b.p0) / (b.p1 - b.p0))
        dp += (d.lp_score[i : i + sz.eval_batch].exp() - lq.exp()).sum(0)
    tok = torch.stack([d.score[:, p] for p in RK.TOK_POS], 1).to(m.net.device)
    return {"U": torch.cat(U), "L": torch.cat(L), "tok": tok, "dp": dp / d.score.shape[0]}


def read_col(m: RK.Model, tab: dict[str, Tensor], r: int) -> Tensor:
    if r < m.sp.n_dense:
        return tab["U"][:, r]
    p, t = divmod(r - m.sp.n_dense, m.sp.V)
    f = m.st.tok_f[p, t]
    return ((tab["tok"][:, p] == t).float() - f) / (f * (1 - f)).clamp_min(1e-12).sqrt()


def top_reads(m: RK.Model, tab: dict[str, Tensor], R: int, min_tok: int) -> list[int]:
    """The R reads before in2 (neurons, directions; tokens seen in at least min_tok windows) of largest |corr| with L. Ties:
    the lower index."""
    L = tab["L"] - tab["L"].mean()
    sdL, N, V, dev = float(L.pow(2).mean().sqrt()), L.shape[0], m.sp.V, L.device
    if R == 0 or sdL < 1e-6:
        return []
    corr, ok = [(tab["U"] * L[:, None]).mean(0) / sdL], [m.sp.masks("all", m.net.device)["or"][: m.sp.n_dense] < RK.ORDER["in2"]]
    for p in range(len(RK.TOK_POS)):
        cnt = torch.bincount(tab["tok"][:, p], minlength=V)[:V].float()
        f = cnt / N
        cov = torch.zeros(V, device=dev).index_add_(0, tab["tok"][:, p], L) / N
        corr.append(cov / ((f * (1 - f)).clamp_min(1e-12).sqrt() * sdL))
        ok.append((cnt >= min_tok) & (cnt < N))
    score = torch.where(torch.cat(ok), torch.cat(corr).abs(), -1.0)
    v, i = score.topk(R)
    return [int(j) for j, s in zip(i, v, strict=True) if s >= 0]


def top_writes(m: RK.Model, b: Built, tab: dict[str, Tensor], family: str, W: int) -> list[int]:
    if family == "pcs":
        j = (m.sp.pcs["in2"].T @ b.push).abs().topk(W).indices
        return [NN + RK.RES.index("in2") * m.sp.k + int(i) for i in j]
    return [m.sp.n_dense + int(i) for i in tab["dp"].abs().topk(W).indices]


def form(m: RK.Model, b: Built, tab: dict[str, Tensor], writes: list[int], reads: list[int]) -> RK.Interv:
    """Writes each reading `reads`; a pcs write starts at the built edit's least-squares fit (its component of the push
    times L regressed on the reads), a logit write at 0."""
    X = torch.stack([torch.ones_like(tab["L"]), *(read_col(m, tab, r) for r in reads)], 1).double().cpu()
    beta = torch.linalg.lstsq(X, tab["L"].double().cpu()[:, None], driver="gelsd").solution[:, 0].float().to(tab["L"].device)
    I = RK.Interv()
    for w in writes:
        cw = float(m.sp.pcs["in2"][:, w - NN - RK.RES.index("in2") * m.sp.k] @ b.push) / float(m.st.sd[w]) if w < m.sp.n_dense else 0.0
        theta = torch.cat([beta.new_zeros(1), cw * beta])
        I.writes.append(RK.Write(w, list(reads), theta, torch.ones_like(theta)))
    return I


def ranking(m: RK.Model, d: RK.Data, I: RK.Interv, reads_on: bool, sz: Any, seed: int) -> dict[str, Any]:
    """Where I's writes stood in E184's first greedy step (the empty intervention, E184's sizes): each write's shift score
    g^2 / h, its rank among the list's writes, the Newton step and I's fitted shift (theta units); the K-th candidate's
    score is the cut below which nothing was line-searched."""
    mk = m.sp.masks("all", m.net.device)
    sc = RK.score_pass(m, RK.Interv(), d.score, d.lp_score, sz, seed)
    g, h = sc.G.sum(0), sc.G.shape[0] * sc.hbar
    floor = 1e-10 * float(h.max().clamp_min(1e-30))
    s = torch.where(mk["w"] & (h > floor), g * g / h.clamp_min(floor), 0)
    cut = RK.candidates(m, sc, RK.Interv(), mk, reads_on, sz)
    rows = [{"name": m.sp.name(w.var), "score": float(s[w.var]), "rank": int((s > s[w.var]).sum()),
             "newton": float(-g[w.var] / h[w.var].clamp_min(floor) / m.st.sd[w.var]), "fitted": float(w.theta[1])} for w in I.writes]  # fmt: skip
    return {"writes": rows, "kth_score": cut[-1].score if cut else None, "K": sz.K}


# ---- stage 1 --------------------------------------------------------------------------------------------------------


def cause(filed: int, r184: int, at_filed: float, o3: float, bar: float) -> dict[str, Any]:
    """A miss (E184's rung above the filed one) is the search's if an oracle at the filed rung closes the bar, the class's if
    only the rank-one form does, else "oracle short". A closure at the bar passes; NaN passes nothing."""
    def hit(x: float) -> bool:
        return x >= bar - RK.EPS

    why = None if r184 <= filed else "search" if hit(at_filed) else "class" if hit(o3) else "oracle short"
    return {"e184_rung": r184, "cause": why, "rung3_search_miss": r184 == 4 and hit(o3)}


def control_item(c: Any, cfg: Config, th: dict[str, Any], name: str) -> dict[str, Any]:
    """Do-nothing: KL under the floor. One neuron / triggered neuron: E184's built edit written as its Write, closure 1."""
    path = cfg.results / f"{name}.json"
    out = load(path)
    if out.get("done"):
        return out
    m, d = RK.prepare(c.net, c.sp, *target(c, name), c.cfg, c.cfg.subset_seed)
    out |= {"experiment": EXPERIMENT, "item": name, "kl": d.kl0}
    if name == "do_nothing":
        out["passed"] = d.kl0 < th["kl_floor"]
    else:
        rec, sp, st = e184_rec(name), m.sp, m.st
        s, w = rec["s"], 2 * RK.D_MLP + int(rec["names"][0].split("#")[1])  # mlp2#j
        sd_w = float(st.sd[w])
        theta = [0.0, s / sd_w]
        reads = []
        if name == "l2_triggered":  # s (h_in1 . pc0 - mu) / sd, mu and sd on E184's PCA windows
            r = NN + RK.RES.index("in1") * sp.k
            mu_p, sd_p = float(c.pcs["in1_mean"] @ c.pcs["in1"][:, 0]), float(c.pcs["in1_var"][0]) ** 0.5
            theta, reads = [0.0, s * (float(st.mu[r]) - mu_p) / (sd_p * sd_w), s * float(st.sd[r]) / (sd_p * sd_w)], [r]
        t = torch.tensor(theta, device=m.net.device)
        I = RK.Interv([RK.Write(w, reads, t, torch.ones_like(t))])
        out |= {"names": I.names(sp), "closure": RK.closure(m, I, d, c.cfg)}
        out["passed"] = out["closure"] >= 1 - th["exact_tol"]
    out["done"] = True
    log(f"{name}: KL {d.kl0:.4g}, oracle closure {out.get('closure')}, passed {out['passed']}")
    dump(path, out)
    return out


def oracle_item(c: Any, cfg: Config, th: dict[str, Any], name: str) -> dict[str, Any]:
    path, sz = cfg.results / f"{name}.json", cfg.oracle()
    out = load(path)
    if out.get("done"):
        return out
    m, d = RK.prepare(c.net, c.sp, *target(c, name), c.cfg, c.cfg.subset_seed)
    rec, b = e184_rec(name), built(c, name)
    out |= {"experiment": EXPERIMENT, "item": name, "filed_rung": FILED[name], "kl": d.kl0, "kl_e184": rec["kl"],
            "share": {"push_in2": share(m.sp.pcs["in2"], b.push), "read_in1": share(m.sp.pcs["in1"], b.dir)}}  # fmt: skip
    if not abs(d.kl0 - rec["kl"]) <= th["kl_rel_tol"] * rec["kl"]:
        E.stop(path, out, f"{name}: KL(F || M) {d.kl0:.6g} does not repeat E184's {rec['kl']:.6g}: not E184's F or windows")
    if "o3" not in out:
        I3 = RK.Interv(edits=[rank_one(m, b)])
        out["o3"] = {"start": RK.closure(m, I3, d, sz)}
        if name not in EXACT:
            RK.refit(m, I3, d, cfg.o3_steps, sz, cfg.seed, RK.check_kl(m, I3, d, sz))
            out["o3"]["fitted"] = RK.closure(m, I3, d, sz)
        log(f"{name}: shares {out['share']}, rank-one form {out['o3']}")
        dump(path, out)
    if name in EXACT and not out["o3"]["start"] >= 1 - th["exact_tol"]:
        E.stop(path, out, f"{name}: the exact rank-one form closes {out['o3']['start']:.6f}: the class code does not express it")
    tab, forms = windows_table(m, d, b, sz), out.setdefault("forms", {})
    for fam in ("pcs", "out"):
        for W, R in cfg.splits:
            key = f"{fam} {W}x{R}"
            if (R and b.dir is None) or key in forms:
                continue
            I = form(m, b, tab, top_writes(m, b, tab, fam, W), top_reads(m, tab, R, sz.min_tok))
            RK.refit_writes(m, I, d, sz, RK.check_kl(m, I, d, sz))
            forms[key] = {"rung": 2 if any(w.reads for w in I.writes) else 1, "n": I.n, "names": I.names(m.sp),
                          "closure": RK.closure(m, I, d, sz), "state": I.to_json()}  # fmt: skip
            log(f"{name} {key}: rung {forms[key]['rung']}, n {I.n}, closure {forms[key]['closure']:.4f}")
            dump(path, out)
            E.budget(c.cfg)
    o3 = out["o3"].get("fitted", out["o3"]["start"])
    best = max((k for k, f in forms.items() if f["rung"] <= FILED[name]), key=lambda k: forms[k]["closure"])
    at_filed = max(forms[best]["closure"], o3 if FILED[name] >= 3 else float("-inf"))
    I = RK.Interv.from_json(forms[best]["state"], m.net.device)
    out |= {"best_form": best, "at_filed": at_filed, "o3_closure": o3,
            "ranking": ranking(m, d, I, FILED[name] >= 2, c.cfg, c.cfg.search_seed),
            "labels": {str(bar): cause(FILED[name], rec["readings"]["all"][str(bar)]["rung"], at_filed, o3, bar) for bar in th["bars"]},
            "done": True}  # fmt: skip
    log(f"{name}: best at the filed rung {best} {at_filed:.4f}, rank-one {o3:.4f}; labels {out['labels']}")
    dump(path, out)
    del m, d
    torch.cuda.empty_cache()
    return out


# ---- stage 2 and the summary ----------------------------------------------------------------------------------------


def repaired_item(c2: Any, name: str, floor: float, bars: list[float]) -> dict[str, Any]:
    """E184's search on list all (rungs 1, 2, 3), with the repaired sizes; resumes from its JSON."""
    path = RESULTS / f"{name}_repaired.json"
    out = load(path)
    if out.get("done"):
        return out
    m, d = RK.prepare(c2.net, c2.sp, *target(c2, name), c2.cfg, c2.cfg.subset_seed)
    out |= {"experiment": EXPERIMENT, "item": name, "kl": d.kl0}
    E.search_list(c2, m, d, out, "all", path)
    out["readings"] = {str(b): RK.reading(d.kl0, floor, out["lists"]["all"], b) for b in bars}
    out["done"] = True
    log(f"{name} repaired readings: {out['readings']}")
    dump(path, out)
    del m, d
    torch.cuda.empty_cache()
    return out


def needs_repair(rec: dict[str, Any]) -> bool:
    return any(v["cause"] == "search" or v["rung3_search_miss"] for v in rec.get("labels", {}).values())


def control_changes(name: str, out: dict[str, Any]) -> list[str]:
    """The repaired search must read each control as E184 did: the same rung, n and names at every bar."""
    old = e184_rec(name)["readings"]["all"]
    return [f"{name} at {b}: {out['readings'][b]} against E184's {v}" for b, v in old.items() if out["readings"][b] != v]


def summary(th: dict[str, Any]) -> dict[str, Any]:
    recs = {n: load(RESULTS / f"{n}.json") for n in FILED}
    rep = {n: load(RESULTS / f"{n}_repaired.json") for n in FILED}
    bar = str(th["r1_bar"])
    done = all(r.get("done") for r in recs.values())
    out: dict[str, Any] = {"experiment": EXPERIMENT, "items": {}}
    for n, r in recs.items():
        row = {"filed": FILED[n], "at_filed": r.get("at_filed"), "o3": r.get("o3_closure"), "labels": r.get("labels")}
        if rep[n].get("done"):
            rd = {b: v["rung"] for b, v in rep[n]["readings"].items()}
            row["repaired"] = {b: {"rung": g, "recovered": g <= FILED[n], "rung3_recovered": g <= 3} for b, g in rd.items()}
        out["items"][n] = row
    out["R1"] = "not read" if not done else "rival a" if any(r["labels"][bar]["cause"] == "search" for r in recs.values()) else "rival b"
    return out


def main(wall_min: float) -> int:
    cfg, th = Config(wall_s=60 * wall_min), json.loads(PRED.read_text())["thresholds"]
    c = E.setup(cfg.e184(), json.loads(E.PRED.read_text())["thresholds"], "2")
    try:
        for name in CONTROLS:
            if not control_item(c, cfg, th, name)["passed"]:
                log(f"STAGE 1 STOP: {name} misses; nothing further runs")
                return 1
        for name in FILED:
            oracle_item(c, cfg, th, name)
        dump(RESULTS / "summary.json", summary(th))
        todo = [n for n in FILED if needs_repair(load(RESULTS / f"{n}.json"))]
        log(f"stage 1 done: {summary(th)['R1']}; stage 2 items: {todo}")
        if todo:
            c2, floor = E.Ctx(c.net, cfg.repaired(), c.th, c.e, c.sp, c.pcs, {}), E.kl_floor(c)
            for name in REPAIR_CONTROLS:
                miss = control_changes(name, repaired_item(c2, name, floor, th["bars"]))
                if miss:
                    log(f"STAGE 2 STOP: the repaired search changes a control: {miss}")
                    return 1
            for name in todo:
                repaired_item(c2, name, floor, th["bars"])
    except E.MoreToDo:
        log(f"wall limit near: exit {MORE_TO_DO}, resubmit")
        return MORE_TO_DO
    dump(RESULTS / "summary.json", summary(th))
    log(f"{EXPERIMENT} DONE: {summary(th)}")
    return 0


def cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wall-min", type=float, default=26.5, help="exit 3 after this many minutes (a debug slot; large on a pod)")
    args = ap.parse_args()
    require_filed(PRED)
    code = 1
    try:
        code = main(args.wall_min)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)


if __name__ == "__main__":
    cli()
