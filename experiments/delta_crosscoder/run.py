"""Cell P2.72 run 1 (E-number in results/predictions.json): the Delta-Crosscoder of Kassem et al. 2026 (arXiv
2603.04426), our port of the paper's text (no code released), on the 67M model, and its reading per side as yes, no or
not scored (spec 3.7). Spec reports/plans/2026-11-23_plan_p2-72-delta-crosscoder-spec.md; run plan
reports/plans/2026-11-23_plan_p2-72-run.md; settings, bars and rules results/predictions.json (refuses unless committed
unchanged). Shared steps: commute.delta_crosscoder (the crosscoder, its trainer, the pure rules); the sides are E235's
and E243's (crosscoder_proper run.py and run2.py) and E212's planted direction (realistic_tasks run3.py's rule).

Per fit (204.8M tokens; spec 3.5): M's and F's 68-token continuations of 100,000 Pile prompts each (`gen`), F's fit
half of its own task text at most `repeat_cap` draws a window, and Pile train rows from `pre_start` for the rest, every
row through both models on the same tokens, position 0 dropped; chunks shuffled; one pass. Fits, in the filed order
(settings.run.order): C0a alone first (identical models, paired init: a wiring check), C0b (identical, independent
init: the null band), C2 (E235's Gaussian at sites A and B), C1 (five planted changes).

Jobs (each writes results/<tag>/... as it finishes; a rerun skips what is done and resumes a checkpoint):
  prep               (CPU) the contrastive prompts and the Pile rows as uint16 under results/large/data/<size>/
  gen KEY...         continuations: M (all prompts), or a changed model's key (its half), e.g. head-r5.A
  fit KEY...         crosscoders by fit key (settings.run.fits), each scored by score.py at the end
  slice KEY          the rehearsal: one candidate and a full-size pool and read on a saved crosscoder, timed
  --rehearse         the rehearsal sizes; --tag sets results/<tag>/; --stop-after S exits 3 after step S
  --c2-yes-reported  lets C1 run after a C2 fit read "yes" (whip-lead told first, spec 5)

    PRED_SHA=<sha> python experiments/delta_crosscoder/run.py fit C0a 2>&1 | tee -a experiments/delta_crosscoder/results/run1/podA.log
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from commute import crosscoder as CC
from commute import delta_crosscoder as DX
from commute import steering as ST
from commute.data import SETS_DIR, cached_rows, target_tokens
from commute.jobs import (
    EXPERIMENTS,
    config_from_settings,
    hard_exit,
    load_experiment,
    require_filed,
)
from commute.variable_read import cut

HERE = Path(__file__).resolve().parent
PRED, ROW = HERE / "results" / "predictions.json", 511
RUN: Any = load_experiment("crosscoder_proper", "run")  # E235: Job (sides), Item, Out, Stop, log
RUN2: Any = load_experiment("crosscoder_proper", "run2")  # E243: the rung scaling
MD: Any = load_experiment("delta_crosscoder", "models")
SC: Any = load_experiment("delta_crosscoder", "score")
E212_TEXTS = EXPERIMENTS / "realistic_tasks" / "results" / "large" / "run3" / "texts.pt"
log, Out, Stop = RUN.log, RUN.Out, RUN.Stop
OUT: dict[str, Path] = {}


@dataclass(frozen=True)
class Sizes:  # settings.sizes.<full | rehearsal>
    steps: int
    batch: int
    n_chunks: int
    prompts: int
    prompt_len: int  # tokens of each prompt fed; position 0 is dropped, so a row gives prompt_len - 1 + cont_len
    cont_len: int
    f_share: float
    repeat_cap: int
    pre_rows: int
    gen_batch: int
    gen_shard: int
    warmup: int
    ckpt_every: int
    log_every: int
    check_step: int
    candidates: int
    pool: int
    pool_min: int
    pile_loss_rows: int
    read_batch: int

    def __post_init__(self) -> None:
        assert self.c_tokens + self.f_quota <= self.total and self.total - self.c_tokens <= self.pre_rows * ROW, self
        assert self.prompts % (2 * self.n_chunks) == 0 and (self.prompts // 2) % self.gen_shard == 0, self
        assert self.warmup < self.steps and 100 <= self.check_step < self.steps and self.pool_min <= self.pool, self

    @property
    def total(self) -> int:
        return self.steps * self.batch

    @property
    def c_tokens(self) -> int:
        return self.prompts * (self.prompt_len - 1 + self.cont_len)

    @property
    def f_quota(self) -> int:
        return round(self.f_share * self.total)


@dataclass(frozen=True)
class Config:  # settings.run
    sites: dict[str, str]
    fits: dict[str, dict[str, Any]]  # key -> what, site, group (C0 C2 C1), paired, tied
    order: tuple[str, ...]
    lr: float
    temperature: float
    top_p: float
    strengths: tuple[float, ...]
    widen: tuple[tuple[float, float], ...]
    top_n: int
    contexts: int
    alive_min: int
    init_seed: int
    data_seed: int
    pool_seed: int
    gen_seed: int
    plant_seed: int
    self_task: str
    prompt_start: int
    pre_start: int
    task_rows: int
    graft_window: int
    loss_val: tuple[int, int]
    bg_val: tuple[tuple[int, int], ...]
    graft_read_val: tuple[int, int]
    score_tf32: bool

    def __post_init__(self) -> None:
        assert self.order[0] == "C0a" and set(self.order) == set(self.fits) and len(self.order) == len(self.fits)
        assert all(f["site"] in self.sites and f["group"] in ("C0", "C1", "C2") for f in self.fits.values())
        assert sorted(self.strengths) == list(self.strengths) and 0 not in self.strengths

    def gen_key(self, key: str) -> str:
        f = self.fits[key]
        return "M" if f["what"] in ("self", "e212") else f"{f['what']}.{f['site']}"


def ledger_ranges(cfg: Config, sz: Sizes) -> list[tuple[str, str, int, int]]:
    """(name, split, first row, end row) of every source (spec 3.5); `check_disjoint` asserts no two overlap."""
    return [("task windows", "train", 0, cfg.task_rows), ("prompts", "train", cfg.prompt_start, cfg.prompt_start + sz.prompts),
            ("pre", "train", cfg.pre_start, cfg.pre_start + sz.pre_rows), ("pile loss", "val", *cfg.loss_val),
            ("graft read", "val", *cfg.graft_read_val)] + [("background", "val", a, b) for a, b in cfg.bg_val]  # fmt: skip


def check_disjoint(r: list[tuple[str, str, int, int]]) -> None:
    for i, (n1, s1, a1, b1) in enumerate(r):
        for n2, s2, a2, b2 in r[i + 1 :]:
            assert s1 != s2 or b1 <= a2 or b2 <= a1, f"row ranges {n1} and {n2} overlap"


def rows_any(split: str, start: int, n: int) -> Tensor:
    """Rows [start, start + n) of the split from the cached files, across file boundaries (cached_rows needs one)."""
    out, at = [], start
    while at < start + n:
        files = [(int(p.stem.split("_")[1]), int(p.stem.split("_")[2])) for p in SETS_DIR.glob(f"{split}_*_*.npy")]
        s, m = max(((s, m) for s, m in files if s <= at < s + m), key=lambda f: f[0] + f[1])
        k = min(start + n, s + m) - at
        out.append(cached_rows(split, at, k))
        at += k
    return torch.cat(out)


def data_dir(size: str) -> Path:
    return HERE / "results" / "large" / "data" / size


def prep(cfg: Config, sz: Sizes, size: str) -> None:
    """The prompts (first prompt_len tokens of each prompt row) and the pre rows, uint16, with their sha256."""
    d = data_dir(size)
    d.mkdir(parents=True, exist_ok=True)
    rec = {}
    for name, start, n, cols in (("prompts", cfg.prompt_start, sz.prompts, sz.prompt_len), ("pre", cfg.pre_start, sz.pre_rows, 512)):
        a = np.empty((n, cols), dtype=np.uint16)
        for i in range(0, n, 32768):
            r = rows_any("train", start + i, min(32768, n - i))[:, :cols]
            assert int(r.max()) < 2**16 and int(r.min()) >= 0
            a[i : i + len(r)] = r.numpy().astype(np.uint16)
        np.save(d / f"{name}.npy", a)
        rec[name] = {"shape": list(a.shape), "sha256": hashlib.sha256((d / f"{name}.npy").read_bytes()).hexdigest()}
        log(f"[prep {size}] {name} {rec[name]}")
    (HERE / "results" / f"data_{size}.json").write_text(json.dumps(rec, indent=1))


def load_u16(path: Path) -> Tensor:
    return torch.from_numpy(np.load(path).astype(np.int64))


# ---------------------------------------------------------------------------------------------------------------- sides


@dataclass
class Side:
    F: Any  # models.Changed
    read: Any  # models.Read
    src: tuple[str, Any]  # ("none", None), ("windows", [tensors]) or ("graft", rows): F's fit half
    site: str
    notes: dict[str, Any]


def legacy(M: Any) -> Any:
    """E243's Job (an E235 Job) without its streams: what its `side`, `gauss` and `match` read."""
    P1, P2 = json.loads(RUN.PRED.read_text()), json.loads(RUN2.PRED.read_text())
    j = RUN2.Job.__new__(RUN2.Job)
    j.cfg = config_from_settings(RUN.Config, {k: tuple(v) if isinstance(v, list) else v for k, v in P1["settings"]["run"].items()})
    j.c2, j.th, j.net = RUN2.settings(P2), P1["thresholds"]["full"], M
    S = P1["settings"]["cc"]["full"]
    j.cc = config_from_settings(CC.Cfg, S | {"shared": tuple(S["shared"]), "snapshots": tuple(S["snapshots"])})
    _, j.inv = RUN.SM.R96.inventories(json.loads(RUN2.E147_INV.read_text()))
    j.zero = {k: torch.zeros_like(w) for k, w in M.W0.items()}
    return j


def task_read(t: Any) -> Any:
    """A task's score half (odd windows) as a models.Read: the answer log-odds at `half`, behaviour positions half - 1
    and half (the answer position and the token before it)."""
    toks = t.toks[1::2]
    tgt = target_tokens(t.token, toks, t.half)
    toks = toks[:, : t.ci_len] if t.ci_len else toks  # a target column (E62) is never fed
    pos = torch.zeros(toks.shape, dtype=torch.bool)
    pos[:, t.half - 1 : t.half + 1] = True
    return MD.Read(toks, "task", t.half, tgt, pos, False)


def plant_read(toks: Tensor, cols: slice, planted: bool) -> Any:
    pos = torch.zeros(toks.shape, dtype=torch.bool)
    pos[:, cols] = True
    pos[:, 0] = False
    return MD.Read(toks, "plant", 0, None, pos, planted)


def side(job: Job, key: str) -> Side:
    f, cfg, M, L = job.cfg.fits[key], job.cfg, job.M, job.legacy
    site, what, gk = cfg.sites[f["site"]], f["what"], cfg.gen_key(key)
    if what == "self":
        return Side(MD.Changed(M, None, None, "M"), task_read(L.task(cfg.self_task)), ("none", None), site, {})
    if what == "gauss":  # E235's own `side` (E243's override reads rung keys only)
        s = RUN.Job.side(L, RUN.Item("gauss", f["site"], 0))
        s.dW, m = L.gauss(s)
        t = L.task(L.inv["states"][L.cfg.gauss_of]["task"])
        return Side(MD.Changed(M, s.dW, None, gk), task_read(t), ("windows", [t.toks[0::2]]), site, {"gauss_match": m, "tol": L.cfg.gauss_match_tol})
    if what.startswith("head"):  # head-rI: E243's head at rung I (site A); head: E235's block-3 head (site B)
        s = L.side(RUN.Item(what, f["site"], 0)) if "-r" in what else RUN.Job.side(L, RUN.Item("head", f["site"], 0))
        notes = {}
        if "-r" in what:
            notes = {"match": L.match(s), "tol": L.c2.match_tol}
            s.apply(notes["match"]["x"])
        return Side(MD.Changed(s.ft, None, None, gk), plant_read(s.read, slice(None), False), ("graft", s.source[1]), site, notes)
    assert what == "e212", what
    texts = torch.load(E212_TEXTS, weights_only=False)
    units, reads = {k: v.long() for k, v in texts["units"].items()}, {k: v.long() for k, v in texts["reads"].items()}
    every, Lr = units["L1"].shape[1], reads["L1"]
    with RUN.exact():  # E212's rule (realistic_tasks/run3.py `planted`): its plant seed, the median base norm there
        h = MD.acts(M, None, Lr, MD.PLANT_SITE).reshape(len(Lr), Lr.shape[1] - 1, -1)[:, -1]
    v = torch.randn(h.shape[1], generator=torch.Generator().manual_seed(cfg.plant_seed))
    plant = (v / v.norm() * h.norm(dim=1).median().cpu()).cuda()
    F = MD.Changed(M, None, (plant, every), "M")
    src = ("windows", [units[t] for t in sorted(units) if t != "L3"])  # L3's units are L1's (run3.py prep asserts it)
    return Side(F, plant_read(Lr, slice(every - 1, None, every), True), src, site, {"plant_norm": float(plant.norm()), "every": every})


# --------------------------------------------------------------------------------------------------------------- stream


def even(total: int, n: int) -> list[int]:
    """`total` split into n parts differing by at most one, the larger first."""
    return [total // n + (i < total % n) for i in range(n)]


class Stream:
    """Spec 3.5's chunks: chunk i holds prompt rows i of M's half (M's continuations) and of F's half (F's), F's
    draws for chunk i, and the next run of pre tokens; shuffled; sizes known before any chunk is built (resume)."""

    def __init__(self, job: Job, s: Side) -> None:
        sz, cfg, self.job, self.s = job.sz, job.cfg, job, s
        q, half, n = sz.prompts // 2 // sz.n_chunks, sz.prompts // 2, sz.n_chunks
        self.c_rows = [(np.arange(i * q, (i + 1) * q), half + np.arange(i * q, (i + 1) * q)) for i in range(n)]
        self.gM, self.gF = job.gen_tokens("M"), job.gen_tokens(s.F.gen)
        kind, src = s.src
        self.draws: list[tuple[Tensor, Tensor]] = []  # "windows": per chunk, (group, row) of each draw
        self.cuts = [0] * n  # "graft": per chunk, how many fresh windows
        if kind == "windows":  # every window repeat_cap times in a seeded order, cut at the quota (never over it)
            lens = torch.tensor([t.shape[1] - 1 for t in src for _ in range(len(t))])
            gid = torch.tensor([g for g, t in enumerate(src) for _ in range(len(t))])
            rid = torch.cat([torch.arange(len(t)) for t in src])
            rep = torch.randperm(len(lens) * sz.repeat_cap, generator=torch.Generator().manual_seed(cfg.data_seed)) % len(lens)
            D = int(torch.searchsorted(lens[rep].cumsum(0), torch.tensor(sz.f_quota), right=True))
            chunks = torch.tensor_split(rep[:D], n)
            self.draws, f_tok = [(gid[c], rid[c]) for c in chunks], [int(lens[c].sum()) for c in chunks]
        elif kind == "graft":
            self.cuts = even(sz.f_quota // (cfg.graft_window - 1), n)
            f_tok = [k * (cfg.graft_window - 1) for k in self.cuts]
        else:
            f_tok = [0] * n
        pre_total = sz.total - sz.c_tokens - sum(f_tok)
        self.pre_n = even(pre_total, n)
        self.pre_off = [sum(self.pre_n[:i]) for i in range(n)]
        c_tok = 2 * q * (sz.prompt_len - 1 + sz.cont_len)
        self.sizes = [c_tok + f + p for f, p in zip(f_tok, self.pre_n, strict=True)]
        self.ledger = {"contrastive": sz.c_tokens, "fit_half": sum(f_tok), "fit_half_quota": sz.f_quota, "pre": pre_total,
                       "pre_available": sz.pre_rows * ROW, "total": sum(self.sizes), "steps_x_batch": sz.total, "chunks": n}  # fmt: skip

    def make(self, i: int) -> Tensor:
        job, s, cfg = self.job, self.s, self.job.cfg
        F, site, g = s.F, s.site, torch.Generator().manual_seed(cfg.data_seed * 100_003 + i)
        a, b = self.c_rows[i]
        with RUN.exact():  # fp32 model forwards, TF32 off (E235)
            h = [MD.pairs(job.M, F, self.gM.rows(a), site, False), MD.pairs(job.M, F, self.gF.rows(b), site, False)]
            kind, src = s.src
            if kind == "windows":
                gi, ri = self.draws[i]
                h += [MD.pairs(job.M, F, src[k][ri[gi == k]], site, True) for k in range(len(src)) if bool((gi == k).any())]
            elif kind == "graft" and self.cuts[i]:
                h.append(MD.pairs(job.M, F, cut(src, self.cuts[i], cfg.graft_window, g), site, True))
            off, n = self.pre_off[i], self.pre_n[i]
            r0, r1 = off // ROW, math.ceil((off + n) / ROW)
            pre = MD.pairs(job.M, F, job.pre[r0:r1], site, False)
            h.append(pre[off - r0 * ROW : off - r0 * ROW + n])
        x = torch.cat(h)
        return x[torch.randperm(len(x), generator=g).to(x.device)]


@dataclass
class Gen:
    tokens: Tensor  # (n, prompt_len + cont_len)
    start: int  # the first prompt index it holds

    def rows(self, idx: np.ndarray) -> Tensor:
        return self.tokens[torch.from_numpy(idx - self.start)]


# ------------------------------------------------------------------------------------------------------------------ job


class Job:
    def __init__(self, cfg: Config, sz: Sizes, arch: DX.Arch, th: dict[str, Any], size: str) -> None:
        self.cfg, self.sz, self.arch, self.th, self.size, self.t0 = cfg, sz, arch, th, size, time.time()
        check_disjoint(ledger_ranges(cfg, sz))
        self.M = ST.Net("cuda")
        self.legacy = legacy(self.M)
        self.prompts = load_u16(data_dir(size) / "prompts.npy")
        self.pre = load_u16(data_dir(size) / "pre.npy")
        self.loss_rows = cached_rows("val", cfg.loss_val[0], sz.pile_loss_rows)
        self.bg = torch.cat([cached_rows("val", a, b - a) for a, b in cfg.bg_val])
        self.log, self._tok = log, None

    def decode(self, ids: list[int]) -> str | None:
        if self._tok is None:
            from commute import vpd

            self._tok = vpd.tokenizer()
        return self._tok.decode(ids)

    def gen_path(self, key: str) -> Path:
        return OUT["large"] / "gen" / f"{key}.npy"

    def gen_tokens(self, key: str) -> Gen:
        p = self.gen_path(key)
        if not p.is_file():
            raise SystemExit(f"no continuations {p}: run `gen {key}` first (or bring the file to this machine)")
        return Gen(load_u16(p), 0 if key == "M" else self.sz.prompts // 2)

    def gen(self, key: str) -> None:
        """Continuations for prompts [0, prompts) (M) or [prompts / 2, prompts) (a changed model), shard by shard."""
        sz, th, p = self.sz, self.th, self.gen_path(key)
        if p.is_file():
            log(f"[gen {key}] done before")
            return
        fit = next(k for k in self.cfg.order if self.cfg.gen_key(k) == key) if key != "M" else None
        F = None if fit is None else side(self, fit).F
        start = 0 if key == "M" else sz.prompts // 2
        out = Out(OUT["res"] / "gen" / f"{key}.json", {"key": key, "card": torch.cuda.get_device_name()})
        shards, t_first = [], None
        torch.cuda.reset_peak_memory_stats()
        for i, a in enumerate(range(start, sz.prompts, sz.gen_shard)):
            sp = p.parent / key / f"{a}.npy"
            if not sp.is_file():
                t = time.time()
                parts = [MD.generate(self.M, F, self.prompts[b : min(b + sz.gen_batch, a + sz.gen_shard)], sz.cont_len,
                                     self.cfg.temperature, self.cfg.top_p, self.cfg.gen_seed + b) for b in range(a, a + sz.gen_shard, sz.gen_batch)]  # fmt: skip
                sp.parent.mkdir(parents=True, exist_ok=True)
                np.save(sp, torch.cat(parts).numpy().astype(np.uint16))
                if t_first is None:
                    t_first = time.time() - t
                    proj = t_first * 100_000 / sz.gen_shard / 60
                    out.d["first_shard"] = {"seconds": t_first, "prompts": sz.gen_shard, "minutes_per_100k": proj,
                                            "tokens_per_s": sz.gen_shard * sz.cont_len / t_first, "peak_gb": torch.cuda.max_memory_allocated() / 2**30}  # fmt: skip
                    out.check("projected minutes per 100k continuations", proj <= th["gen_cap_min"], {"minutes": proj, "bar": th["gen_cap_min"]})
                log(f"[gen {key}] shard {i} ({a}..{a + sz.gen_shard}) in {time.time() - t:.1f} s")
            shards.append(np.load(sp))
        tokens = np.concatenate(shards)
        assert tokens.shape == (sz.prompts - start, sz.prompt_len + sz.cont_len) and bool((tokens[:, : sz.prompt_len] == self.prompts[start:].numpy()).all())
        np.save(p, tokens)
        out.d |= {"done": True, "rows": len(tokens), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "minutes": (time.time() - self.t0) / 60}
        out.write()
        log(f"[gen {key}] DONE {out.d['rows']} rows, sha256 {out.d['sha256']}")

    # ---- one fit -----------------------------------------------------------------------------------------------------

    def gate(self, key: str, c2_yes_reported: bool) -> None:
        """C0a alone first; C1 only after both C2 fits read, and after a C2 "yes" only once it has been reported."""
        if key == "C0a":
            return
        c0a = item("C0a")
        if not (c0a.get("done") and all(c["pass"] for c in c0a["checks"].values())):
            raise SystemExit(f"refusing {key}: C0a has not passed here (bring results/<tag>/items/C0a.json to this machine)")
        if self.cfg.fits[key]["group"] != "C1":
            return
        c2 = {k: item(k) for k in self.cfg.order if self.cfg.fits[k]["group"] == "C2"}
        if not all(v.get("done") for v in c2.values()):
            raise SystemExit(f"refusing {key}: the C2 fits {sorted(c2)} have not all read here")
        yes = [k for k, v in c2.items() if v["read"]["label"] == "yes"]
        if yes and not c2_yes_reported:
            raise SystemExit(f"refusing {key}: C2 {yes} read yes; report to whip-lead, then rerun with --c2-yes-reported")

    def fit(self, key: str, stop_after: int | None) -> None:
        f, sz, th = self.cfg.fits[key], self.sz, self.th
        if item(key).get("done"):
            log(f"[fit {key}] done before")
            return
        out = Out(OUT["res"] / "items" / f"{key}.json", {"key": key, "fit": f, "card": torch.cuda.get_device_name()})
        torch.cuda.reset_peak_memory_stats()
        s = side(self, key)
        out.d["side"] = s.notes
        if "tol" in s.notes:
            mt = s.notes.get("gauss_match") or s.notes["match"]
            out.check("change matched to its size", abs(mt["rel"]) <= s.notes["tol"], mt | {"tol": s.notes["tol"]})
        st = Stream(self, s)
        out.d["ledger"] = st.ledger
        out.check("ledger: every source counted, steps x batch exactly", st.ledger["total"] == sz.total and st.ledger["pre"] <= st.ledger["pre_available"], st.ledger)
        m = self.train(key, f, st, out, stop_after)
        out.d["peak_gb_train"] = torch.cuda.max_memory_allocated() / 2**30
        OUT["large"].mkdir(parents=True, exist_ok=True)
        torch.save({k: v.bfloat16() if v.numel() > 1_000_000 else v for k, v in m.state_dict().items()}, OUT["large"] / f"{key}.pt")
        if f["what"] == "self" and f.get("paired"):
            dev = float((m.rel_norm() - 0.5).abs().max())
            bar = th["c0a_rn_dev"]
            ev = DX.read_codes(m, SC.background(self, s.F, s.read, s.site), sz.read_batch, self.cfg.pool_seed)
            out.d["read"] = {"max_rn_dev": dev, "fve_base": ev["fve_base"], "fve_ft": ev["fve_ft"], "l0": ev["l0"], "label": "not read (every latent ties)"}
            out.check("C0a: every R at 0.5 within the bar", dev <= bar, {"max_dev": dev, "bar": bar})
            fve = min(ev["fve_base"], ev["fve_ft"])
            out.check("FVE at the end (a code fault under the bar)", fve >= th["fve_min_end"], {"fve": fve, "bar": th["fve_min_end"]})
        else:
            out.d["read"] = SC.score(self, key, s.F, s.read, s.site, m, f["group"] == "C1", OUT["res"] / "items" / f"{key}.steer.json", out)
        (OUT["large"] / "ckpt" / f"{key}.pt").unlink(missing_ok=True)
        out.d |= {"done": True, "minutes": (time.time() - self.t0) / 60, "peak_gb": torch.cuda.max_memory_allocated() / 2**30}
        out.write()
        log(f"ITEM {key} DONE in {out.d['minutes']:.1f} min: label {out.d['read']['label']}")

    def train(self, key: str, f: dict[str, Any], st: Stream, out: Any, stop_after: int | None) -> DX.DXC:
        """Every step of the fit, from a checkpoint if there is one. Stops: a non-finite loss; on C0a a nonzero delta
        loss; at check_step the batch FVE, the delta loss falling and the projected minutes (`early_checks`)."""
        sz, arch, C0a, t_item = self.sz, self.arch, bool(f["what"] == "self" and f.get("paired")), time.time()
        m = DX.DXC(arch, bool(f.get("paired")), bool(f.get("tied")), self.cfg.init_seed).cuda()
        tr = DX.Trainer(m, self.cfg.lr, sz.warmup)
        ck, start, curve, early = OUT["large"] / "ckpt" / f"{key}.pt", 0, [], []
        if ck.is_file():
            c = torch.load(ck, weights_only=False)
            tr.load(c["tr"])
            start, curve, early = c["step"], c["curve"], c["early"]
            log(f"[{key}] resumed at step {start}")
        else:
            (OUT["res"] / "items" / f"{key}.steer.json").unlink(missing_ok=True)
            t = time.time()
            h0 = st.make(0)
            out.d["chunk_seconds"], scale = time.time() - t, DX.scale_of(h0[:, 0], arch.target_rms)
            m.scale.fill_(scale)
            m.b_dec.data.copy_((h0 * scale).mean(0))
            del h0
        out.d["scale"] = float(m.scale)
        t_train, t_last, s_last = time.time(), time.time(), start
        log(f"[{key}] steps {start}..{sz.steps}, ledger {json.dumps(st.ledger)}")
        step = start - 1
        for step, x in zip(range(start, sz.steps), CC.batches(st.sizes, st.make, sz.batch, start), strict=False):
            r = tr.step(x * m.scale)
            if not all(math.isfinite(v) for v in r.values()):
                out.check("finite loss", False, {"step": step} | r)
            if C0a and r["delta"] != 0.0:
                out.check("C0a: delta loss exactly 0 at every step", False, {"step": step, "delta": r["delta"]})
            if step < sz.check_step:
                early.append([r["delta"], min(r["fve_base"], r["fve_ft"])])
            if step == 0:
                out.d["step0"] = r
                log(f"[{key}] step 0: {json.dumps(r)}")
            if step == sz.check_step:
                self.early_checks(out, early, C0a, (time.time() - t_train) / (step + 1 - start), step, t_item)
            if step % sz.log_every == 0 or step == sz.steps - 1:
                now = time.time()
                rec = {"step": step, **r, "ms_per_step": 1000 * (now - t_last) / (step + 1 - s_last)}
                if C0a:
                    rec["max_rn_dev"] = float((m.rel_norm() - 0.5).abs().max())
                curve.append(rec)
                t_last, s_last = now, step + 1
                log(f"  [{key}] " + " ".join(f"{k} {v:.4g}" for k, v in rec.items()))
            done = step + 1
            if done % sz.ckpt_every == 0 or done == stop_after:
                ck.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"tr": tr.state(), "step": done, "curve": curve, "early": early}, ck)
            if done == stop_after:
                log(f"[{key}] stopped after step {done} (--stop-after)")
                hard_exit(3)
        out.check("every step trained", step + 1 == sz.steps, {"last": step + 1, "steps": sz.steps})
        out.d |= {"curve": curve, "resumed_at": start, "train_minutes": (time.time() - t_train) / 60}
        return m

    def early_checks(self, out: Any, early: list[list[float]], C0a: bool, s_per_step: float, step: int, t_item: float) -> None:
        th, sz = self.th, self.sz
        e = torch.tensor(early)
        blocks = e[: len(e) // 100 * 100, 0].view(-1, 100).mean(1)  # the delta loss's 100-step means before step
        fve, peak, last = float(e[-100:, 1].mean()), float(blocks.max()), float(blocks[-1])
        out.check(f"batch FVE over the 100 steps before {step}", fve >= th["fve_check"], {"fve": fve, "bar": th["fve_check"]})
        out.d["delta_peak_last_100"] = [peak, last]
        if not C0a:  # C0a's delta loss is 0 at every step, checked each step
            ratio, bar = last / peak, th["delta_last_over_peak_max"]
            out.check(f"delta loss falling by step {step}", ratio <= bar, {"peak_100": peak, "last_100": last, "ratio": ratio, "bar": bar})
        proj = (time.time() - t_item) / 60 + s_per_step * (sz.steps - step) / 60 + th["score_budget_min"]
        out.check("projected fit minutes", proj <= th["fit_cap_min"], {"minutes": proj, "bar": th["fit_cap_min"], "s_per_step": s_per_step})

    def slice(self, key: str, full: Sizes) -> None:
        """One candidate with a full-size pool, read at full read sizes, on a saved crosscoder: minutes and peak memory."""
        f, t0 = self.cfg.fits[key], time.time()
        self.sz = dataclasses.replace(self.sz, candidates=1, pool=full.pool, pool_min=full.pool_min, pile_loss_rows=full.pile_loss_rows)
        self.loss_rows = cached_rows("val", self.cfg.loss_val[0], full.pile_loss_rows)
        m = DX.DXC(self.arch, bool(f.get("paired")), bool(f.get("tied")), 0).cuda()
        m.load_state_dict({k: v.float() for k, v in torch.load(OUT["large"] / f"{key}.pt", weights_only=False).items()})
        s = side(self, key)
        torch.cuda.reset_peak_memory_stats()
        out = Out(OUT["res"] / "items" / f"{key}.slice.json", {"key": key, "card": torch.cuda.get_device_name()})
        out.d["read"] = SC.score(self, key, s.F, s.read, s.site, m, False, OUT["res"] / "items" / f"{key}.slice_steer.json", out)
        out.d |= {"minutes": (time.time() - t0) / 60, "peak_gb": torch.cuda.max_memory_allocated() / 2**30,
                  "latents_read": out.d["read"]["latents_read"], "minutes_per_latent": out.d["read"]["minutes"] / out.d["read"]["latents_read"], "done": True}  # fmt: skip
        out.write()
        log(f"[slice {key}] {json.dumps({k: v for k, v in out.d.items() if k not in ('read', 'checks')})}")


def item(key: str) -> dict[str, Any]:
    p = OUT["res"] / "items" / f"{key}.json"
    return json.loads(p.read_text()) if p.is_file() else {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job", choices=("prep", "gen", "fit", "slice"))
    ap.add_argument("keys", nargs="*")
    ap.add_argument("--rehearse", action="store_true")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--stop-after", type=int, default=None)
    ap.add_argument("--c2-yes-reported", action="store_true")
    a = ap.parse_args()
    require_filed(PRED)
    P = json.loads(PRED.read_text())
    assert P["experiment"][1:].isdigit(), "claim the E-number (fleet/ecode) and file it in results/predictions.json"
    size = "rehearsal" if a.rehearse else "full"
    S = P["settings"]
    tup = lambda v: tuple(tuple(x) if isinstance(x, list) else x for x in v) if isinstance(v, list) else v
    cfg = config_from_settings(Config, {k: tup(v) for k, v in S["run"].items()})
    sz, full = (config_from_settings(Sizes, S["sizes"][k]) for k in (size, "full"))
    arch = config_from_settings(DX.Arch, S["arch"][size])
    tag = a.tag or ("rehearse" if a.rehearse else "run1")
    OUT.update(res=HERE / "results" / tag, large=HERE / "results" / "large" / tag)
    log(f"[{P['experiment']}] {a.job} {a.keys} tag {tag}, sizes {size}: {sz}")
    if a.job == "prep":
        prep(cfg, sz, size)
        return 0
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    if a.job == "fit" and not a.rehearse and P["thresholds"]["full"]["c0a_rn_dev"] is None:
        raise SystemExit("thresholds.full.c0a_rn_dev is not filed: file it from the rehearsal's C0a deviation first")
    job = Job(cfg, sz, arch, P["thresholds"][size], size)
    try:
        for key in a.keys:
            if a.job == "gen":
                job.gen(key)
            elif a.job == "fit":
                job.gate(key, a.c2_yes_reported)
                job.fit(key, a.stop_after)
            else:
                job.slice(key, full)
            torch.cuda.empty_cache()
    except Stop as e:
        log(f"STOP: {e}: tell the lead")
        return 1
    log(f"FINISHED {a.job} {' '.join(a.keys)} in {(time.time() - job.t0) / 60:.1f} min; peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
    return 0


if __name__ == "__main__":
    hard_exit(main())
