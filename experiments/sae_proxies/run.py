"""E314 (cell P2.85): two cheap SAE proxies on the 67M at E234's base-only SAEs, each beside a matched random update.
Plan reports/plans/2026-11-26_plan_e314-sae-proxies.md; reading (2)'s source and departures
reports/plans/2026-11-26_spec_kissane-sae-transfer.md; settings, expected values and stop tolerances in
results/predictions.json (the run refuses unless it is committed unchanged). Descriptive: no verdict on reuse.

States: the 48 of known_3seeds/results/known3.json (method share; 16 tasks x seeds 0-2), files from E136's inventory,
E299's stage2.json (five seed-2 states) and E147's results/large (the e58 states); each read with one random update, a
Gaussian per matrix at that matrix's Frobenius norm in the state's update (`finetune.random_direction`, seeded from the
state name's sha256), on the state's own rows. Rows, read position, target and the log-odds unit are E136's
(share_untrained windows.load_read); everything fp32, TF32 off.

(1) Error share (ours, after Marks et al. 2024's error node). At the 8 block-output sites (attn, mlp of 4 layers),
    e(o) = o - SAE(o). The base model runs with each site's output set to hat(o_cur) + e_FT, computed as
    o_cur + (e_FT - e(o_cur)) (the same sum, and exactly o_cur when e_FT is the base's own error, so the zero update
    reads exactly 0); all 8 sites at once, every position. share = (L - T0) / (T1 - T0). Error-energy fraction:
    sum |e_FT - e_0|^2 / sum |o_FT - o_0|^2 over the 8 sites and every position, each model's own run.
(2) Reconstruction gap (Kissane et al. 2024). Headline: their explained variance, github.com/ckkissane/sae-transfer
    utils.py `get_explained_variance_from_batch`: per token 1 - |SAE(x) - x|^2 / |x - mean_0 x|^2, then the mean over
    tokens; each model's activations scaled by its own norm factor (their activation store, SAELens
    `expected_average_only_in`: mean norm sqrt(d) on Pile rows), and tokens whose norm in that scale exceeds their
    `outlier_threshold` (default 100,000) excluded. gap = EV(base) - EV(FT) per site of E234's 13, mean over sites.
    Secondary: FVU as a ratio of sums (`sae.read`'s convention), gap FVU(FT) - FVU(base). Position 0 dropped (E234).

  stage 0  controls, alone: the zero update (Delta e, energy numerator, patched numerator, both gaps exactly 0); the
           fine-tuned model run twice (run-to-run floor, max |logit| at the read position); the full patch (o_FT at all
           8 sites) reproduces T1 within that floor; E234's saved base FVE reproduced on its Pile rows by its own bf16
           path; T1 - T0 against each state's saved gain_fp32 (relative).
  stage 1  each state, then its random update, written to results/results.json as it lands; a rerun resumes.
Prints every scored number, "STAGE n DONE"; a miss prints "STOP: ..." and exits 1.

    sbatch --comment=E314 cluster/sae_proxies.sbatch
"""

from __future__ import annotations

import hashlib
import json
import math
import socket
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import sae as S
from commute import sae_feat as SF
from commute import steering as ST
from commute.data import cached_rows, target_tokens
from commute.finetune import random_direction
from commute.identity import log_odds
from commute.jobs import config_from_settings, hard_exit, load_experiment, require_filed
from commute.open_pair import jsonable

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
PRED, OUT = RESULTS / "predictions.json", RESULTS / "results.json"
W = load_experiment("share_untrained", "windows")  # E136's rows: load_read, Read
E234 = load_experiment("sae_train", "run")  # load_config, load_saes, RESULTS
KNOWN3 = "experiments/known_3seeds/results/known3.json"
E136_INV, E136_S1 = "experiments/share_untrained/results/inventory.json", "experiments/share_untrained/results/stage1.json"
E299_S2 = "experiments/known_3seeds/results/stage2.json"
E147_INV, E147_S1 = "experiments/refit_96key/results/inventory.json", "experiments/refit_96key/results/stage1.json"
E234_BASE = "experiments/sae_train/results/stage0_resid_pre0_s0+13.json"  # .base.<site>_s0.fve, Pile val rows
OUTER = SF.SITES  # the 8 block-output sites of reading (1)
DW = dict[str, Tensor]
_json: dict[str, Any] = {}


@dataclass(frozen=True)
class Config:
    sae_size: str
    sae_seed: int
    norm_rows: int  # Pile val rows (from E234's eval_start) for each model's norm factor
    outlier_threshold: float
    gain_floor: float
    random_gain_min: float
    gain_rel_tol: float
    e234_fve_tol: float

    def __post_init__(self) -> None:
        assert self.sae_size == "scored" and self.sae_seed == 0, "E234's scored seed-0 SAEs"


@dataclass(frozen=True)
class State:
    name: str
    task: str  # the pair's task, e.g. e54/cipher_own
    seed: int
    side: str  # reuse (the pair's first task) or new
    regating_share: float
    file: str
    rows: str  # E136's task whose rows are read
    spec: dict[str, Any]
    gain_saved: float


class Stop(Exception):
    pass


def log(msg: str) -> None:
    print(msg, flush=True)


def load(rel: str) -> Any:
    if rel not in _json:
        _json[rel] = json.loads((REPO / rel).read_text())
    return _json[rel]


def write(out: dict[str, Any]) -> None:
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(jsonable(out), indent=1, sort_keys=True))
    tmp.replace(OUT)


def check(checks: dict[str, Any], name: str, value: float, tol: float) -> None:
    """|value| <= tol (nan misses), recorded and logged; a miss raises Stop."""
    ok = abs(value) <= tol
    checks[name] = {"value": value, "tol": tol, "pass": ok}
    log(f"[check {name}] {value:.6g} (tol {tol:g}): {ok}")
    if not ok:
        raise Stop(f"{name}: {value:.6g} over {tol:g}")


def locate(name: str) -> tuple[str, str, dict[str, Any], float]:
    """(update file, E136 rows name, rows spec, saved gain_fp32) from the run that made or last read the state."""
    e136, e299 = load(E136_INV), load(E299_S2)["states"]
    if name in e299:  # E299's seed-2 states read the rows of their seed-0 siblings
        rows = e136["states"][f"{name.rsplit('_s', 1)[0]}_s0"]["task"]
        return e299[name]["file"], rows, e136["tasks"][rows], e299[name]["e136"]["gain_fp32"]
    if name in e136["states"]:
        rows = e136["states"][name]["task"]
        return e136["states"][name]["file"], rows, e136["tasks"][rows], load(E136_S1)["states"][name]["gain_fp32"]
    m, s = name.split("/")[1].rsplit("_s", 1)  # E147's refit e58 states, as refit_96key.inventories builds them
    spec = {"format": "kv", "path": load(E147_INV)["maps"][m]["windows"], "batch": 64}
    return f"experiments/refit_96key/results/large/dW_{m}_s{s}.pt", f"e58/{m}", spec, load(E147_S1)["states"][name]["e136"]["gain_fp32"]


def states() -> list[State]:
    out: dict[str, State] = {}
    for c in load(KNOWN3)["cells"]:
        if c["method"] != "share":
            continue
        for side in ("reuse", "new"):
            for name, share in zip(c["states"][side], c[side], strict=True):
                task, seed = name.rsplit("_s", 1)
                if name in out:  # a task shared by two pairs: the same side and share in both
                    assert (out[name].side, out[name].regating_share) == (side, share), name
                    continue
                out[name] = State(name, task, int(seed), side, share, *locate(name))
    assert len(out) == 48, len(out)
    return list(out.values())


def stable_seed(name: str) -> int:
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


def flat(acts: list[Tensor]) -> Tensor:
    """(B, T, d) batches -> (n, d), position 0 dropped (E234's convention)."""
    return torch.cat([a[:, 1:].reshape(-1, a.shape[-1]) for a in acts])


def explained_variance(ae: S.SAE, x: Tensor, own_norm: float, base_norm: float, threshold: float) -> tuple[float, int]:
    """Kissane et al.'s explained variance (sae-transfer utils.py get_explained_variance_from_batch): per token
    1 - |SAE(x) - x|^2 / |x - mean_0 x|^2, averaged over tokens. Their store scales each model's activations to mean
    norm sqrt(d) by that model's own factor; the outlier mask is read in that scale (`sae_in.norm(dim=-1) >
    outlier_threshold`). E234's SAE has the base factor folded in, so x goes in at base_norm / own_norm (exactly 1 for
    the base, and for the zero update): the ratio is what their two factors do, EV being free of a common scale."""
    mask = x.norm(dim=-1) * (math.sqrt(x.shape[-1]) / own_norm) > threshold
    x = x[~mask] * (base_norm / own_norm)
    l2 = (ae.decode(ae.encode(x)) - x).double().pow(2).sum(-1)
    tv = (x - x.mean(0)).double().pow(2).sum(-1)
    return float((1 - l2 / tv).mean()), int(mask.sum())


@dataclass
class Sums:
    """`sae.read`'s FVU (1 - FVE): sum |x - SAE(x)|^2 over sum |x - mean x|^2, fp64 sums over batches."""

    err: float = 0.0
    sxx: float = 0.0
    sx: Tensor | None = None
    n: int = 0

    def add(self, ae: S.SAE, x: Tensor) -> Sums:
        self.err += float((x - ae.decode(ae.encode(x))).double().pow(2).sum())
        self.sxx += float(x.double().pow(2).sum())
        sx = x.double().sum(0)
        self.sx = sx if self.sx is None else self.sx + sx
        self.n += len(x)
        return self

    def fvu(self) -> float:
        assert self.sx is not None
        return self.err / (self.sxx - float(self.sx.pow(2).sum()) / self.n)


@dataclass
class Run:
    lo: float  # mean log-odds of the target over the rows
    at_half: Tensor  # (N, V) logits at the read position
    acts: dict[str, list[Tensor]]  # per kept site, per batch (B, T, d)


class Reader:
    def __init__(self, c: Config, sts: list[State]) -> None:
        self.c, self.net = c, ST.Net("cuda")
        self.c234, _ = E234.load_config(c.sae_size)
        got = E234.load_saes(self.c234, [f"{s}_s{c.sae_seed}" for s in S.SITES], E234.RESULTS / "large" / c.sae_size)
        self.saes = {k.rsplit("_s", 1)[0]: ae.requires_grad_(False) for k, ae in got.items()}
        self.reads = {s.rows: W.load_read(s.rows, s.spec) for s in sts}
        self.pile = cached_rows("val", self.c234.eval_start, self.c234.eval_rows)
        self.base_norm = {s: v["mean_norm"] for s, v in self.pile_stats(None, c.norm_rows, bf16=False, fve=False).items()}

    def delta(self, st: State) -> DW:
        return {k: v.float() for k, v in torch.load(REPO / st.file, map_location="cpu", weights_only=False).items()}

    def err(self, site: str, o: Tensor) -> Tensor:
        ae = self.saes[site]
        return o - ae.decode(ae.encode(o))

    def forward(self, x: Tensor, fns: dict[str, Callable[[Tensor], Tensor | None]]) -> Tensor:
        with S.hooked(self.net.net, fns):
            out = self.net.net(x)
        return (out[0] if isinstance(out, tuple) else out).float()

    def run(self, r: Any, dW: DW | None, edit: Callable[[int], dict[str, Any]] | None = None, keep: tuple[str, ...] = ()) -> Run:
        """The rows in E136's batches under W0 + dW (None: W0): mean target log-odds, logits at the read position, and
        the kept sites' activations; `edit(i)` gives batch i's site replacements."""
        acts: dict[str, list[Tensor]] = {s: [] for s in keep}
        lo, half = 0.0, []
        with self.net.at(dW):
            for i, start in enumerate(range(0, len(r.toks), r.batch)):
                b = r.toks[start : start + r.batch].cuda()
                fns = edit(i) if edit else {}
                assert not set(fns) & set(keep)
                fns |= {s: (lambda o, s=s: acts[s].append(o)) for s in keep}
                lg = self.forward(b[:, : r.ci_len], fns)[:, r.half]
                lo += float(log_odds(lg, target_tokens(r.token, b, r.half)).double().sum())
                half.append(lg)
        return Run(lo / len(r.toks), torch.cat(half), acts)

    def pile_batches(self, dW: DW | None, rows: Tensor, bf16: bool) -> Iterator[dict[str, Tensor]]:
        """Per batch of E234's Pile rows, {site: (n, d) fp32} at positions 1..; bf16 is E234's own capture."""
        with self.net.at(dW):
            for b in rows.split(self.c234.eval_batch):
                if bf16:
                    a = S.capture(self.net.net, b, S.SITES)
                    yield {s: a[:, j].float() for j, s in enumerate(S.SITES)}
                    continue
                got: dict[str, Tensor] = {}
                self.forward(b.cuda(), {s: (lambda o, s=s, got=got: got.__setitem__(s, o)) for s in S.SITES})
                yield {s: o[:, 1:].reshape(-1, o.shape[-1]) for s, o in got.items()}

    def pile_stats(self, dW: DW | None, n_rows: int, bf16: bool, fve: bool) -> dict[str, dict[str, float]]:
        """Per site: the mean activation norm (SAELens' norm-factor estimate) and, if fve, sae.read's FVE."""
        norm, sums, n = dict.fromkeys(S.SITES, 0.0), {s: Sums() for s in S.SITES}, 0
        for batch in self.pile_batches(dW, self.pile[:n_rows], bf16):
            for s, x in batch.items():
                norm[s] += float(x.norm(dim=-1).double().sum())
                if fve:
                    sums[s].add(self.saes[s], x)
            n += len(batch[S.SITES[0]])
        return {s: {"mean_norm": norm[s] / n} | ({"fve": 1 - sums[s].fvu()} if fve else {}) for s in S.SITES}

    def patch(self, e_ft: dict[str, list[Tensor]], i: int) -> dict[str, Any]:
        return {s: (lambda o, s=s: o + (e_ft[s][i] - self.err(s, o))) for s in OUTER}

    def read(self, r: Any, dW: DW) -> dict[str, Any]:
        """Both readings of one update on one task's rows."""
        base, ft = self.run(r, None, keep=S.SITES), self.run(r, dW, keep=S.SITES)
        e_ft = {s: [self.err(s, o) for o in ft.acts[s]] for s in OUTER}
        patched = self.run(r, None, edit=lambda i: self.patch(e_ft, i))
        num = den = d_e = 0.0
        for s in OUTER:
            for o0, o1, e1 in zip(base.acts[s], ft.acts[s], e_ft[s], strict=True):
                de = e1 - self.err(s, o0)
                num, den = num + float(de.double().pow(2).sum()), den + float((o1 - o0).double().pow(2).sum())
                d_e = max(d_e, float(de.abs().max()))
        ft_norm = {s: v["mean_norm"] for s, v in self.pile_stats(dW, self.c.norm_rows, bf16=False, fve=False).items()}
        rec: dict[str, Any] = {"T0": base.lo, "T1": ft.lo, "L_patch": patched.lo, "gain": ft.lo - base.lo, "energy_num": num, "energy_den": den,
                               "error_energy_frac": num / den if den > 0 else math.nan, "max_abs_delta_e": d_e}  # fmt: skip
        ev: dict[str, dict[str, float]] = {k: {} for k in ("ev_base_per_site", "ev_ft_per_site", "ev_gap_per_site", "norm_ratio_per_site",
                                                           "fvu_base_per_site", "fvu_ft_per_site", "fvu_gap_per_site", "outliers_base", "outliers_ft")}  # fmt: skip
        th, bn = self.c.outlier_threshold, self.base_norm
        for s in S.SITES:
            x0, x1, ae = flat(base.acts[s]), flat(ft.acts[s]), self.saes[s]
            (e0, n0), (e1, n1) = explained_variance(ae, x0, bn[s], bn[s], th), explained_variance(ae, x1, ft_norm[s], bn[s], th)
            f0, f1 = Sums().add(ae, x0).fvu(), Sums().add(ae, x1).fvu()
            for k, v in zip(ev, (e0, e1, e0 - e1, bn[s] / ft_norm[s], f0, f1, f1 - f0, n0, n1), strict=True):
                ev[k][s] = v
        rec |= ev | {"ev_gap_mean": sum(ev["ev_gap_per_site"].values()) / len(S.SITES), "fvu_gap_mean": sum(ev["fvu_gap_per_site"].values()) / len(S.SITES)}
        return rec


def with_share(rec: dict[str, Any], floor: float, random: bool) -> dict[str, Any]:
    """The error share; under the floor nan (a trained state: |gain| < gain_floor) or None with the reason (a random
    update: |gain| < random_gain_min, where its energy fraction is read instead)."""
    g, num = rec["gain"], rec["L_patch"] - rec["T0"]
    if abs(g) >= floor:
        return rec | {"error_share": num / g}
    if not random:
        return rec | {"error_share": math.nan}
    return rec | {"error_share": None, "error_share_why_none": f"|gain| {abs(g):.3g} under {floor:g} log-odds"}


# ----------------------------------------------------------------------------- stage 0


def zero_and_patch(rd: Reader, st: State, ch: dict[str, Any], res: dict[str, Any]) -> None:
    r = rd.reads[st.rows]
    z = with_share(rd.read(r, {k: torch.zeros_like(w) for k, w in rd.net.W0.items()}), rd.c.gain_floor, random=False)
    res["zero_update"] = z
    check(ch, "zero update gain", z["gain"], 0.0)
    check(ch, "zero update error share nan (0 = nan)", 0.0 if math.isnan(z["error_share"]) else 1.0, 0.0)
    check(ch, "zero update max |e_FT - e_0|", z["max_abs_delta_e"], 0.0)
    check(ch, "zero update energy numerator", z["energy_num"], 0.0)
    check(ch, "zero update energy 0/0 gives nan (0 = nan)", 0.0 if math.isnan(z["error_energy_frac"]) else 1.0, 0.0)
    check(ch, "zero update patched numerator L - T0", z["L_patch"] - z["T0"], 0.0)
    check(ch, "zero update max |EV gap| over 13 sites", max(map(abs, z["ev_gap_per_site"].values())), 0.0)
    check(ch, "zero update max |FVU gap| over 13 sites", max(map(abs, z["fvu_gap_per_site"].values())), 0.0)
    dW = rd.delta(st)
    base, a, b = rd.run(r, None), rd.run(r, dW, keep=OUTER), rd.run(r, dW)
    floor = float((a.at_half - b.at_half).abs().max())
    full = rd.run(r, None, edit=lambda i: {s: (lambda _o, v=a.acts[s][i]: v) for s in OUTER})
    res["floor"] = {"state": st.name, "max_abs_logit_ft_twice": floor}
    res["full_patch"] = {"state": st.name, "share": (full.lo - base.lo) / (a.lo - base.lo), "max_abs_logit_vs_ft": float((full.at_half - a.at_half).abs().max())}
    log(f"[0 floor] {json.dumps(res['floor'])}\n[0 full patch] {json.dumps(res['full_patch'])}")
    check(ch, f"full patch max |logit - FT logit| {st.name} (tol = run-to-run floor)", res["full_patch"]["max_abs_logit_vs_ft"], floor)


def e234_fve(rd: Reader, ch: dict[str, Any], res: dict[str, Any]) -> None:
    """E234's base FVE on its Pile rows by its own bf16 capture and sae.read's formula; fp32 beside (no stop)."""
    saved = load(E234_BASE)["base"]
    bf, fp = rd.pile_stats(None, rd.c234.eval_rows, bf16=True, fve=True), rd.pile_stats(None, rd.c234.eval_rows, bf16=False, fve=True)
    res["e234_fve"] = {s: {"saved": saved[f"{s}_s0"]["fve"], "bf16": bf[s]["fve"], "fp32": fp[s]["fve"]} for s in S.SITES}
    log(f"[0 E234 FVE] {json.dumps(res['e234_fve'])}")
    for s in S.SITES:
        check(ch, f"E234 base FVE {s}", bf[s]["fve"] - saved[f"{s}_s0"]["fve"], rd.c.e234_fve_tol)


def gains(rd: Reader, sts: list[State], ch: dict[str, Any], res: dict[str, Any]) -> None:
    t0 = {name: rd.run(r, None).lo for name, r in rd.reads.items()}
    res["gains"], miss = {}, []
    for st in sts:
        g = rd.run(rd.reads[st.rows], rd.delta(st)).lo - t0[st.rows]
        res["gains"][st.name] = {"gain": g, "saved": st.gain_saved, "rel": abs(g - st.gain_saved) / abs(st.gain_saved)}
        log(f"[0 gain {st.name}] {json.dumps(res['gains'][st.name])}")
        if not res["gains"][st.name]["rel"] <= rd.c.gain_rel_tol:
            miss.append(st.name)
    ch["gain reproduction"] = {"missed": miss, "tol": rd.c.gain_rel_tol, "pass": not miss}
    if miss:
        raise Stop(f"T1 - T0 off the saved gain_fp32 by over {rd.c.gain_rel_tol:g} relative: {miss}")


def stage0(rd: Reader, sts: list[State], out: dict[str, Any]) -> None:
    if out.get("stage0", {}).get("pass"):
        log("[stage 0] already passed")
        return
    res: dict[str, Any] = {"host": socket.gethostname(), "card": torch.cuda.get_device_name(), "checks": {}, "base_norm": rd.base_norm}
    out["stage0"] = res
    try:
        zero_and_patch(rd, sts[0], res["checks"], res)
        write(out)
        e234_fve(rd, res["checks"], res)
        write(out)
        gains(rd, sts, res["checks"], res)
    except Stop as e:
        res["stopped"] = str(e)
        write(out)
        raise
    res["pass"] = True
    write(out)


# ----------------------------------------------------------------------------- stage 1


HEAD = ("gain", "error_share", "error_energy_frac", "ev_gap_mean", "fvu_gap_mean")


def stage1(rd: Reader, sts: list[State], out: dict[str, Any], t_job: float) -> None:
    out.setdefault("states", {})
    t0, n_new = time.time(), 0
    for st in sts:
        if st.name in out["states"]:
            continue
        r, dW = rd.reads[st.rows], rd.delta(st)
        seed = stable_seed(st.name)
        rec = with_share(rd.read(r, dW), rd.c.gain_floor, random=False)
        rnd = with_share(rd.read(r, random_direction(dW, seed)), rd.c.random_gain_min, random=True) | {"seed": seed}
        meta = {"task": st.task, "seed": st.seed, "side": st.side, "regating_share": st.regating_share, "rows": st.rows, "file": st.file}
        out["states"][st.name] = meta | rec | {"random": rnd}
        write(out)
        n_new += 1
        log(f"[1 {st.name}] {json.dumps(jsonable(meta | {k: rec.get(k) for k in HEAD} | {'random': {k: rnd.get(k) for k in HEAD}}))}")
        log(f"[1 {st.name} all] {json.dumps(jsonable(out['states'][st.name]))}")
        left = sum(s.name not in out["states"] for s in sts)
        log(f"[1 time] {len(out['states'])} of {len(sts)}; projected {((time.time() - t0) / n_new * left + time.time() - t_job) / 60:.1f} min for the job")


def main() -> int:
    t_job = time.time()
    require_filed(PRED)
    P = json.loads(PRED.read_text())
    assert P["experiment"] == "E314"
    c = config_from_settings(Config, P["settings"])
    sts = states()
    if missing := [f"{s.name}: {s.file}" for s in sts if not (REPO / s.file).is_file()]:
        print(f"STOP: update files missing: {missing}", flush=True)
        return 1
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    out: dict[str, Any] = json.loads(OUT.read_text()) if OUT.is_file() else {"experiment": "E314"}
    with torch.no_grad():
        rd = Reader(c, sts)
        log(f"[host] {socket.gethostname()} {torch.cuda.get_device_name()} states {len(sts)} row sets {len(rd.reads)}")
        try:
            stage0(rd, sts, out)
        except Stop as e:
            print(f"STOP: {e}: tell the lead", flush=True)
            return 1
        print("STAGE 0 DONE", flush=True)
        stage1(rd, sts, out, t_job)
    print("STAGE 1 DONE", flush=True)
    log(f"[done] peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB, {(time.time() - t_job) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    hard_exit(main())
