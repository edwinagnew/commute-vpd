"""E235 (cell P2.49, run 1 of 2): Minder et al.'s BatchTopK crosscoder with latent scaling, their pinned trainer at
their settings and full token budget, TF32 in the crosscoder as their code sets it, at two sites (A block 2's input,
B the last block's output). Stage 0 (alone): the base against itself and a grafted head (E206) per site, the head on
three crosscoder seeds. Stage 1: the primary pair e54_matched at both sites (fine-tune seeds 0-2 at crosscoder seed 0,
crosscoder seeds 1-2 on fine-tune seed 0), a Gaussian matched to elicit s0's activation change per site, E203's planted
direction and a k sweep at A. Plan reports/plans/2026-11-20_plan_p2-49-crosscoder-proper.md; settings, bars and rules
results/predictions.json (refuses unless committed unchanged). Shared steps: commute.crosscoder.

GPU jobs (each crosscoder writes results/<tag>/items/<key>.json and results/large/<tag>/<key>.pt as it ends; a rerun
skips a finished item and resumes a checkpointed one):
  pod STAGE I N    stage 0 or 1, every N-th item from the I-th; stage-1 items refuse at a site stage0.json has not passed
  item KEY...      crosscoders by key (what.site.cSEED[.kK]), e.g. the rehearsal's extra e53 item
  timing SITE TAG  ms per TF32 step at the full size and seconds per chunk at SITE (A or B) -> timing_<TAG>.json
  scorecheck       one score of an untrained full-size crosscoder at full read sizes: peak memory
  --rehearse       the rehearsal sizes and bars; --tag sets results/<tag>/; --stop-after S exits 3 after step S
CPU jobs (lists, stage checks, the two-per-card gain, the resume check): score.py.

    PRED_SHA=<sha> python experiments/crosscoder_proper/run.py pod 0 0 8 2>&1 | tee -a experiments/crosscoder_proper/results/pod0_0.log
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import itertools
import json
import math
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import crosscoder as CC
from commute import finetune as FT
from commute import graft as G
from commute import kv
from commute import rule_retention as R
from commute import steering as ST
from commute.data import cached_rows
from commute.jobs import (
    EXPERIMENTS,
    config_from_settings,
    hard_exit,
    load_experiment,
    require_filed,
)
from commute.open_pair import jsonable
from commute.variable_read import cut

HERE = Path(__file__).resolve().parent
REPO, PRED = HERE.parents[1], HERE / "results" / "predictions.json"
SM: Any = load_experiment("size_matched", "run")  # pairs(): the share's states and their inventory
W: Any = load_experiment("share_untrained", "windows")  # load_read: each task's read windows
GRAFT_PRED = EXPERIMENTS / "graft_sites" / "results" / "predictions.json"
GRAFTS = EXPERIMENTS / "graft_sites" / "results" / "large"
ROW, TASK_ROWS = 511, 32768  # tokens of a 512-token row without position 0; task windows came from rows below this
OUT: dict[str, Path] = {}  # res, large: set in main from --tag


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


class Stop(Exception):
    """A filed stop: the item file records why; the job exits 1."""


@dataclass(frozen=True)
class Config:  # predictions.json settings.run
    sites: dict[str, str]
    data_seed: int
    plant_seed: int
    gauss_draw: int
    gauss_match_tol: float
    control_task: str
    reuse: tuple[str, ...]
    new: tuple[str, ...]
    gauss_of: str
    heads: dict[str, str]
    graft_layers: dict[str, int]
    graft_rows: int
    graft_window: int
    graft_read_val: tuple[int, int]
    graft_read_n: int
    graft_read_seed: int
    seeds: tuple[int, ...]
    k_sweep: tuple[int, ...]
    fill_start: int
    spliced_rows: int
    timing_steps: int
    rehearse_extra: tuple[str, ...]

    def __post_init__(self) -> None:
        assert set(self.heads) == set(self.sites) and all(h in self.graft_layers for h in self.heads.values())
        assert len(self.seeds) >= 3 and len(self.reuse) == len(self.new) >= 3 and self.gauss_of == self.reuse[0]
        assert self.graft_rows <= TASK_ROWS and self.graft_read_val[0] < self.graft_read_val[1]


@dataclass(frozen=True)
class Item:
    what: str  # self, head, gauss, direction, or a state name with "/" as "__"
    site: str  # A or B
    seed: int
    k: int | None = None  # the k sweep's override

    @property
    def key(self) -> str:
        return f"{self.what}.{self.site}.c{self.seed}" + (f".k{self.k}" if self.k else "")

    @staticmethod
    def parse(key: str) -> Item:
        p = key.split(".")
        return Item(p[0], p[1], int(p[2][1:]), int(p[3][1:]) if len(p) > 3 else None)


def stage_items(stage: str, cfg: Config) -> list[Item]:
    if stage == "0":
        return [Item("self", s, 0) for s in cfg.sites] + [Item("head", s, c) for s in cfg.sites for c in cfg.seeds]
    out = [Item("gauss", s, c) for s in cfg.sites for c in cfg.seeds]
    for s in cfg.sites:
        out += [Item(n.replace("/", "__"), s, 0) for n in (*cfg.reuse, *cfg.new)]
        out += [Item(n.replace("/", "__"), s, c) for n in (cfg.reuse[0], cfg.new[0]) for c in cfg.seeds[1:]]
    return out + [Item("direction", "A", 0)] + [Item("head", "A", 0, k) for k in cfg.k_sweep]


@contextmanager
def exact() -> Iterator[None]:
    """fp32 matmuls (TF32 off) for model forwards and every score; the crosscoder's steps run in TF32."""
    was = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = was


@dataclass
class Side:
    item: Item
    site: str  # the Net site name
    ft: Any  # the fine-tune side's steering.Net (the base net, or a grafted copy)
    dW: ST.DW | None
    source: tuple[str, Any]  # ("kv", task), ("fit", windows) or ("graft", rows): the task half of the stream
    read: Tensor  # the rows the change is read on
    plant: tuple[Tensor, int] | None = None  # (vector, position) on task windows and read rows


class Out:
    def __init__(self, path: Path, d: dict[str, Any]) -> None:
        self.path, self.d = path, d | {"checks": {}}

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(jsonable(self.d), indent=1))
        tmp.replace(self.path)

    def check(self, name: str, ok: bool, detail: dict[str, Any]) -> None:
        self.d["checks"][name] = {"pass": bool(ok), **detail}
        log(f"[check {name}] {'ok' if ok else 'MISSED'} {json.dumps(jsonable(detail))}")
        if not ok:
            self.d["stopped"] = name
            self.write()
            raise Stop(name)


class Job:
    def __init__(self, cfg: Config, cc: CC.Cfg, th: dict[str, Any]) -> None:
        self.cfg, self.cc, self.th, self.t0 = cfg, cc, th, time.time()
        self.net = ST.Net("cuda")
        _, _, self.inv = SM.pairs()
        n_pile = math.ceil(cc.tokens / cc.epochs / 2 / ROW / cc.chunk_rows) * cc.chunk_rows
        assert TASK_ROWS <= cc.pile_start and cc.pile_start + n_pile <= cc.pool_start < cc.pool_start + cc.pool_rows <= cfg.fill_start
        self.pile, self.pool = cached_rows("train", cc.pile_start, n_pile), cached_rows("train", cc.pool_start, cc.pool_rows)
        self.val = cached_rows("val", 0, cc.val_rows)
        self.n_chunks = n_pile // cc.chunk_rows
        self.zero = {k: torch.zeros_like(w) for k, w in self.net.W0.items()}

    # ---- sides ------------------------------------------------------------------------------------------------------

    def task(self, name: str) -> Any:
        return W.load_read(name, self.inv["tasks"][name])

    def windows(self, t: Any, half: str) -> Tensor:
        toks = t.toks[:, : t.ci_len] if t.ci_len else t.toks
        return toks[0::2] if half == "fit" else toks[1::2]

    def source(self, name: str) -> tuple[str, Any]:
        spec = self.inv["tasks"][name]
        if spec["format"] == "kv":
            return "kv", (torch.load(REPO / spec["path"], weights_only=False)["task"], self.task(name).half)
        return "fit", self.windows(self.task(name), "fit")

    def state(self, name: str, c: float = 1.0) -> ST.DW:
        d = torch.load(REPO / self.inv["states"][name]["file"], weights_only=False)
        return {k: v.float().to(self.net.W0[k].device) * c for k, v in d.items()}

    def grafted(self, role: str) -> Any:
        """A copy of the base net with E206's saved graft `role` attached (its weights loaded over a seeded init)."""
        spec: dict[str, Any] = json.loads(GRAFT_PRED.read_text())["settings"]["recipe"]
        rec = config_from_settings(G.Recipe, {k: tuple(v) if isinstance(v, list) else v for k, v in spec.items()})
        ft = copy.copy(self.net)
        ft.net = copy.deepcopy(self.net.net).requires_grad_(False)
        g = G.attach_head(R.Model(ft.net), self.cfg.graft_layers[role], 0, rec)
        G.load(g, GRAFTS / f"{role}.pt")
        return ft

    def graft_read(self) -> Tensor:
        lo, hi = self.cfg.graft_read_val
        return cut(cached_rows("val", lo, hi - lo), self.cfg.graft_read_n, self.cfg.graft_window, torch.Generator().manual_seed(self.cfg.graft_read_seed))

    def side(self, it: Item) -> Side:
        cfg, site = self.cfg, self.cfg.sites[it.site]
        if it.what == "head":
            rows = cached_rows("train", 0, cfg.graft_rows)
            g = Side(it, site, self.grafted(cfg.heads[it.site]), None, ("graft", rows), self.graft_read())
            assert g.dW is None  # the grafted side shares .model and .W0 with the base net: a dW here could write into it
            return g
        name = cfg.control_task if it.what in ("self", "direction") else cfg.gauss_of if it.what == "gauss" else it.what.replace("__", "/")
        task = self.inv["states"][name]["task"] if name in self.inv["states"] else name
        s = Side(it, site, self.net, self.zero, self.source(task), self.windows(self.task(task), "score"))
        if it.what == "direction":
            s.plant = self.planted(task)
        elif it.what not in ("self", "gauss"):  # the Gaussian's change is matched in `run_item`, under its checks
            s.dW = self.state(name)
        return s

    def planted(self, task: str) -> tuple[Tensor, int]:
        """E203's planted direction: a random unit vector (plant seed) times the median base norm at the answer position."""
        t, gen = self.task(task), torch.Generator().manual_seed(self.cfg.plant_seed)
        with exact():
            h = CC.site_acts(self.net, self.windows(t, "fit")[:, : t.half + 1], None, self.cfg.sites["A"])
        h = h.reshape(-1, t.half, self.cc.d)[:, -1]
        v = torch.randn(self.cc.d, generator=gen)
        return (v / v.norm() * h.norm(dim=1).median().cpu()).cuda(), t.half

    def energy(self, s: Side, dW: ST.DW | None) -> float:
        x = self.pair(s.read, dataclasses.replace(s, dW=dW), True)
        return float((x[:, 1] - x[:, 0]).double().pow(2).sum(1).mean())

    def gauss(self, s: Side) -> tuple[ST.DW, dict[str, float]]:
        """A Gaussian change at gauss_of's per-matrix norms (finetune.random_direction, norms exact in float64) times
        one scalar c, set by bisection so its activation change per token at the site equals gauss_of's."""
        src = self.state(self.cfg.gauss_of)
        R0 = FT.random_direction({k: v.cpu() for k, v in src.items()}, self.cfg.gauss_draw)
        R0 = {k: (r.double() * (src[k].double().norm().cpu() / r.double().norm())).float().cuda() for k, r in R0.items()}
        target = self.energy(s, src)
        e = lambda c: self.energy(s, {k: c * v for k, v in R0.items()})
        lo, hi = 0.0, 1.0
        while e(hi) < target and hi < 1e4:
            lo, hi = hi, 2 * hi
        c = hi
        for _ in range(40):
            c = math.sqrt(lo * hi) if lo > 0 else hi / 2
            got = e(c)
            if abs(got / target - 1) <= self.cfg.gauss_match_tol / 4:
                break
            lo, hi = (c, hi) if got < target else (lo, c)
        got = e(c)
        return {k: c * v for k, v in R0.items()}, {"c": c, "target": target, "energy": got, "rel": got / target - 1}

    # ---- the stream ---------------------------------------------------------------------------------------------------

    def pair(self, toks: Tensor, s: Side, task: bool) -> Tensor:
        """(m, 2, d): the base at W0, then the fine-tune side (its net, its dW, the plant on task windows)."""
        with exact():
            return torch.stack([CC.site_acts(self.net, toks, None, s.site), CC.site_acts(s.ft, toks, s.dW, s.site, s.plant if task else None)], 1)

    def text(self, s: Side) -> dict[str, Any]:
        """The task half per chunk: windows (fresh, or capped draws of saved ones) and Pile fill rows for the rest."""
        kind, src = s.source
        half_tokens = self.cc.chunk_rows * ROW
        T = 2 * src[1] if kind == "kv" else self.cfg.graft_window - 1 if kind == "graft" else src.shape[1] - 1
        full = math.ceil(half_tokens / T)
        per = min(full, max(1, self.cc.repeat_cap * len(src) // self.n_chunks)) if kind == "fit" else full
        fill = max(0, math.ceil((half_tokens - per * T) / ROW))
        return {"kind": kind, "T": T, "per_chunk": per, "fill_rows": fill,
                "task_share": per * T / (per * T + (self.cc.chunk_rows + fill) * ROW)}  # fmt: skip

    def stream(self, s: Side) -> tuple[list[int], Any, dict[str, Any]]:
        cc, tx = self.cc, self.text(s)
        order = [(e, c) for e in range(cc.epochs) for c in torch.randperm(self.n_chunks, generator=torch.Generator().manual_seed(self.cfg.data_seed + e)).tolist()]
        size = cc.chunk_rows * ROW + tx["per_chunk"] * tx["T"] + tx["fill_rows"] * ROW
        fill = cached_rows("train", self.cfg.fill_start, self.n_chunks * tx["fill_rows"]) if tx["fill_rows"] else None
        kind, src = s.source

        def make(i: int) -> Tensor:
            e, c = order[i]
            gen, fixed = torch.Generator().manual_seed(self.cfg.data_seed * 100_003 + e * 1_009 + c), self.cfg.data_seed * 7919 + c
            C, n = cc.chunk_rows, tx["per_chunk"]
            if kind == "kv":
                win = kv.windows(self.pool, src[0].marker, src[0].keys, src[1], n, fixed).tokens
            elif kind == "graft":
                win = cut(src, n, self.cfg.graft_window, torch.Generator().manual_seed(fixed))
            else:
                win = src[torch.randint(len(src), (n,), generator=gen)]
            h = [self.pair(self.pile[c * C : (c + 1) * C], s, False), self.pair(win, s, True)]
            if fill is not None:
                h.append(self.pair(fill[c * tx["fill_rows"] : (c + 1) * tx["fill_rows"]], s, False))
            x = torch.cat(h)
            return x[torch.randperm(len(x), generator=gen).to(x.device)]

        tx |= {"chunks": len(order), "tokens": size * len(order)}
        return [size] * len(order), make, tx

    # ---- one crosscoder -----------------------------------------------------------------------------------------------

    def before(self, s: Side, out: Out) -> None:
        """The checks before step 0: the self path, a graft's write, the Gaussian's match; the change's energy."""
        th, it, toks = self.th, s.item, s.read[:8]
        if it.what == "self":
            with exact():
                diff = float((CC.site_acts(self.net, toks, None, s.site) - CC.site_acts(self.net, toks, self.zero, s.site)).abs().max())
            out.check("self path: a zero update gives W0's activations", diff <= th["self_act_diff"], {"max_abs": diff, "bar": th["self_act_diff"]})
        if it.what == "gauss":
            s.dW, m = self.gauss(s)
            out.d["gauss_match"] = m
            out.check("Gaussian matched on activation change", abs(m["rel"]) <= self.cfg.gauss_match_tol, m | {"tol": self.cfg.gauss_match_tol})
        out.d["energy"] = self.energy(s, s.dW)
        if it.what == "head":
            out.check("graft writes at its site", out.d["energy"] > th["graft_energy_min"], {"energy": out.d["energy"], "bar": th["graft_energy_min"]})
            if self.cfg.graft_layers[self.cfg.heads[it.site]] >= int(self.cfg.sites["A"][-1]):
                with exact():
                    diff = float((CC.site_acts(self.net, toks, None, self.cfg.sites["A"]) - CC.site_acts(s.ft, toks, None, self.cfg.sites["A"])).abs().max())
                out.check("a later graft leaves site A unchanged", diff <= th["self_act_diff"], {"max_abs": diff, "bar": th["self_act_diff"]})

    def score(self, ae: Any, s: Side, cc: CC.Cfg, h_ls: Tensor, h_read: Tensor) -> tuple[dict[str, Any], dict[str, Any]]:
        with exact():  # one validation chunk on the GPU at a time (E212)
            ev = CC.fve(ae, itertools.chain((self.pair(r, s, False) for r in self.val.split(cc.chunk_rows)), [h_read]))
            rec, large = CC.score(cc, ae, ev, h_ls, h_read, s.plant[0] if s.plant else None)
        rn, alive = CC.relative_norm(ae), large["alive"]
        rec["max_rn_dev_alive"] = float((rn[alive] - 0.5).abs().max()) if bool(alive.any()) else math.nan
        return rec, large

    def run_item(self, it: Item, stop_after: int | None) -> dict[str, Any]:
        cc, th, t_item = dataclasses.replace(self.cc, k=it.k) if it.k else self.cc, self.th, time.time()
        out = Out(OUT["res"] / "items" / f"{it.key}.json", {"experiment": EXPERIMENT, "key": it.key, "item": dataclasses.asdict(it),
                                                             "card": torch.cuda.get_device_name()})  # fmt: skip
        s = self.side(it)
        self.before(s, out)
        sizes, make, tx = self.stream(s)
        out.d["stream"] = tx
        out.check("stream holds every step", tx["tokens"] // cc.batch >= cc.steps, {"tokens": tx["tokens"], "need": cc.steps * cc.batch})
        with exact():
            h0 = make(0)
            mean, std = CC.normaliser(h0)
            h_ls, h_read = h0[: cc.ls_tokens].clone(), self.pair(s.read, s, True)
        del h0
        out.check("latent-scaling tokens from the first chunk", len(h_ls) == cc.ls_tokens, {"tokens": len(h_ls), "want": cc.ls_tokens})
        tr = CC.trainer(cc, mean, std, it.seed)
        ck, start, snaps, curve = OUT["large"] / "ckpt" / f"{it.key}.pt", 0, [], []
        if ck.is_file():
            st = torch.load(ck, weights_only=False)
            start, snaps, curve = CC.load_state(tr, st), st["snaps"], st["curve"]
            log(f"[{it.key}] resumed at step {start}")
        marks, t_train, last = sorted({int(f * cc.steps) for f in cc.snapshots}), time.time(), {"step": start}
        log(f"[{it.key}] steps {start}..{cc.steps}, k {cc.k}, stream {json.dumps(tx)}")

        def hook(step: int) -> None:
            last["step"] = step
            if step == th["curve_step"] + 1:
                f = next(r["fve_batch"] for r in curve if r["step"] == th["curve_step"])
                out.check(f"batch FVE at step {th['curve_step']}", f >= th["fve_curve_min"], {"fve_batch": f, "bar": th["fve_curve_min"]})
            if step in marks:
                r = self.score(tr.ae, s, cc, h_ls, h_read)[0]
                snaps.append({"step": step} | {k: r.get(k) for k in ("fve_base", "fve_ft", "dead", "specific", "d_authors", "d_alt", "counts", "lead")})
                log(f"[{it.key} snapshot] {json.dumps(jsonable(snaps[-1]))}")
                if step == marks[0]:
                    proj = (time.time() - t_item + (time.time() - t_train) / max(1, step - start) * (cc.steps - step)) / 60
                    f = min(r["fve_base"], r["fve_ft"])
                    out.check("FVE at the first snapshot", f >= th["fve_early"], {"fve": f, "bar": th["fve_early"]})
                    out.check("dead share at the first snapshot", r["dead"] <= th["dead_max"], {"dead": r["dead"], "bar": th["dead_max"]})
                    out.check("projected minutes", proj <= th["projected_limit_min"], {"minutes": proj, "bar": th["projected_limit_min"]})
            if step % cc.ckpt_every == 0 or step in marks or step == stop_after:
                ck.parent.mkdir(parents=True, exist_ok=True)
                torch.save(CC.state(tr, step) | {"snaps": snaps, "curve": curve}, ck)
            if step == stop_after:
                log(f"[{it.key}] stopped after step {step} (--stop-after)")
                hard_exit(3)

        torch.backends.cuda.matmul.allow_tf32 = True  # the crosscoder's steps; `exact` turns it off around forwards
        try:
            CC.train(tr, CC.batches(sizes, make, cc.batch, start), start, cc.steps, cc.log_every, curve, log, hook)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = False
        out.check("every step trained", last["step"] == cc.steps, {"last": last["step"], "steps": cc.steps})
        rec, large = self.score(tr.ae, s, cc, h_ls, h_read)
        with exact():
            spl = CC.spliced(tr.ae, (self.net, s.ft), s.dW, self.val[: self.cfg.spliced_rows], s.site)
        OUT["large"].mkdir(parents=True, exist_ok=True)
        large["ae"] = {k: v.bfloat16() if v.numel() > 1_000_000 else v for k, v in tr.ae.state_dict().items()}
        torch.save(large, OUT["large"] / f"{it.key}.pt")
        out.d |= {"read": rec, "spliced": spl, "snapshots": snaps, "curve": curve, "resumed_at": start, "steps": last["step"],
                  "minutes": (time.time() - t_item) / 60, "peak_gb": torch.cuda.max_memory_allocated() / 2**30}  # fmt: skip
        log(f"[{it.key} read] {json.dumps(jsonable({k: v for k, v in rec.items() if k != 'spec'}))}")
        log(f"[{it.key} spliced] {json.dumps(spl)}")
        f = min(rec["fve_base"], rec["fve_ft"])
        out.check("FVE at the end", f >= th["fve_min"], {"fve": f, "bar": th["fve_min"]})
        out.check("dead share at the end", rec["dead"] <= th["dead_max"], {"dead": rec["dead"], "bar": th["dead_max"]})
        if it.what == "self":
            out.check("self: relative norms at 0.5", rec["max_rn_dev_alive"] <= th["self_rn_dev"], {"max_dev": rec["max_rn_dev_alive"], "bar": th["self_rn_dev"]})
            out.check("self: no fine-tune-specific latent", rec["counts"]["ft_specific"] == 0, {"counts": rec["counts"]})
        ck.unlink(missing_ok=True)
        out.d["done"] = True
        out.write()
        log(f"ITEM {it.key} DONE in {out.d['minutes']:.1f} min: specific {rec['specific']}, count {rec['counts']['ft_specific']}")
        return out.d


# ---- jobs ---------------------------------------------------------------------------------------------------------------


def done(key: str) -> bool:
    p = OUT["res"] / "items" / f"{key}.json"
    return p.is_file() and bool(json.loads(p.read_text()).get("done"))


def pod(job: Job, stage: str, i: int, n: int, stop_after: int | None) -> None:
    todo = stage_items(stage, job.cfg)[i::n]
    log(f"[pod] stage {stage} shard {i}/{n}: {[t.key for t in todo]}")
    if stage == "1":
        s0 = OUT["res"] / "stage0.json"
        passed = json.loads(s0.read_text())["sites"] if s0.is_file() else {}
        kept = []
        for t in todo:
            if t.k is None and not passed.get(t.site, {}).get("pass"):  # the k sweep is reported, never scored: no site block
                print(f"skip {t}: site {t.site} failed stage 0", flush=True)
                continue
            kept.append(t)
        todo = kept
    for it in todo:
        if done(it.key):
            log(f"[pod] {it.key} done before, skipped")
            continue
        job.run_item(it, stop_after)
        torch.cuda.empty_cache()


def timing(job: Job, full: CC.Cfg, site: str, tag: str) -> None:
    """ms per TF32 step of a full-size crosscoder after 50 warm steps; seconds to build one stream chunk at the site
    (the job's sizes), scaled per token to the full run's tokens; peak memory. Run alone (TAG solo) and as two
    processes at once (p0, p1) for the two-per-card gain (score.py timingcheck)."""
    s = job.side(Item("self", site, 0))
    sizes, make, _ = job.stream(s)
    torch.cuda.synchronize()
    t0 = time.time()
    with exact():
        h = make(0)
    torch.cuda.synchronize()
    chunk_s = time.time() - t0
    tr, x = CC.trainer(full, *CC.normaliser(h), 0), h[: full.batch].clone()
    torch.backends.cuda.matmul.allow_tf32 = True
    for i in range(50):  # step numbers past the k annealing and the threshold start, as most of a run is
        tr.update(5000 + i, x.clone())
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(job.cfg.timing_steps):
        tr.update(5050 + i, x.clone())
    torch.cuda.synchronize()
    ms = (time.time() - t0) / job.cfg.timing_steps * 1000
    torch.backends.cuda.matmul.allow_tf32 = False
    out = {"card": torch.cuda.get_device_name(), "site": site, "tag": tag, "ms_per_step": ms, "chunk_seconds": chunk_s,
           "chunk_tokens": sizes[0], "train_minutes": ms * full.steps / 60_000, "stream_minutes": chunk_s / sizes[0] * full.tokens / 60,
           "peak_gb": torch.cuda.max_memory_allocated() / 2**30, "started": t0}  # fmt: skip
    log(f"[timing] {json.dumps(out)}")
    Out(OUT["res"] / f"timing_{tag}.json", out).write()


def scorecheck(job: Job, full: CC.Cfg) -> None:
    """One score of an untrained crosscoder at the full dictionary and read sizes on the site-B head (the graft path
    and the last site): peak memory and minutes of a full score."""
    s, t0 = job.side(Item("head", "B", 0)), time.time()
    with exact():
        h_read = job.pair(s.read, s, True)
    job.val = cached_rows("val", 0, full.val_rows)
    _, make, _ = job.stream(s)
    with exact():
        h = make(0)
    tr = CC.trainer(full, *CC.normaliser(h), 0)
    rec, _ = job.score(tr.ae, s, full, h[: full.ls_tokens].clone(), h_read)
    out = {"card": torch.cuda.get_device_name(), "peak_gb": torch.cuda.max_memory_allocated() / 2**30, "minutes": (time.time() - t0) / 60,
           "latents": full.latents, "val_rows": full.val_rows, "read_tokens": len(h_read), "keys": sorted(rec)}  # fmt: skip
    log(f"[scorecheck] {json.dumps(out)}")
    Out(OUT["res"] / "scorecheck.json", out).write()


EXPERIMENT = json.loads(PRED.read_text())["experiment"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job", choices=("pod", "item", "timing", "scorecheck"))
    ap.add_argument("args", nargs="*")
    ap.add_argument("--rehearse", action="store_true")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--stop-after", type=int, default=None)
    a = ap.parse_args()
    require_filed(PRED)
    P = json.loads(PRED.read_text())
    assert P["experiment"] == "E235", "predictions.json is not E235's"
    size = "rehearsal" if a.rehearse else "full"
    cc, full = (config_from_settings(CC.Cfg, S | {"shared": tuple(S["shared"]), "snapshots": tuple(S["snapshots"])})
                for S in (P["settings"]["cc"][size], P["settings"]["cc"]["full"]))
    cfg = config_from_settings(Config, {k: tuple(v) if isinstance(v, list) else v for k, v in P["settings"]["run"].items()})
    tag = a.tag or ("rehearse" if a.rehearse else "run1")
    OUT.update(res=HERE / "results" / tag, large=HERE / "results" / "large" / tag)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    log(f"[{EXPERIMENT}] {a.job} {a.args} tag {tag}, sizes {size}; {cc}")
    job = Job(cfg, cc, P["thresholds"][size])
    try:
        if a.job == "pod":
            pod(job, a.args[0], int(a.args[1]), int(a.args[2]), a.stop_after)
        elif a.job == "item":
            for key in a.args:
                job.run_item(Item.parse(key), a.stop_after)
        elif a.job == "timing":
            timing(job, full, a.args[0], a.args[1])
        else:
            scorecheck(job, full)
    except Stop as e:
        log(f"STOP: {e}: tell the lead")
        return 1
    log(f"FINISHED {a.job} {' '.join(a.args)} in {(time.time() - job.t0) / 60:.1f} min; peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
    return 0


if __name__ == "__main__":
    hard_exit(main())
