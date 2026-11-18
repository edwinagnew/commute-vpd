"""E203 (cell P2.31, run 2 of 2): the crosscoder baseline as its authors run it. A BatchTopK crosscoder trained with
Minder et al.'s pinned code and settings, between the base 67M and each saved fine-tune of the nine built pairs, as built
and size-matched (E187's c on the larger side), at block 2's input. Fine-tune-specific latents: the top N by relative
norm that pass latent scaling. The reading d = shared part - fine-tune-specific part of the change on the task's
score-half windows, with the specific part and shared / (shared + specific) beside it. Plan
reports/plans/2026-11-17_plan_p2-31-run2.md (+ _appendix.md); settings, bars and expected values
results/run2/predictions.json (refuses unless committed unchanged). Run 1 (E198) is this file's previous version in git;
its record (results/predictions.json, stage0.json, stage01_pod.log) is untouched. Glue: `commute.crosscoder`.

Data per side, as the source's two halves (FineWeb + chat there), equal tokens, shuffled together within chunks of
`chunk_rows` Pile rows (their 1M-token local shuffling), every position but 0, `epochs` passes, the same order for every
side: (1) Pile train rows from `pile_start` (32768: past every row a task window was cut from, rows 0..32767);
(2) task text. Key-value tasks get fresh windows from their own generator (`commute.kv.windows`, the task's marker and
keys) on Pile rows from `pool_start`, past the Pile half, so no window is a read or scored one; the same windows in both
epochs, as a fixed dataset is. Other tasks (e62, e53) reuse their fit-half windows, each drawn at most `repeat_cap` times
an epoch, so their task share is below half. Each side records its task text (source, unique windows, repeats, share).
The normaliser and the latent-scaling tokens come from the first chunk.

  prep            (CPU) assemble the cached Pile train rows the full run reads, as two small files
  inputs i/n      (CPU) print the files a stage-2 shard reads, for the pod's upload
  0a, 0b          stage 0, the base against itself / a planted direction (separate pods allowed); snapshots scored
  t               step time at the run's size, fp32 then TF32 (300 steps each)
  1 KEY           one stage-1 crosscoder: reuse | new | random | reuse_seed1 (one per pod); snapshots scored
  1check          (CPU) stage 1's bars on the four files: writes stage1.json pass or stop
  2 i/n           every other state, then the size-matched states, shard i of n; each side gets the FVE and dead bars
  resumecheck     (CPU, rehearsal) step-matched loss differences of a resumed run against an uninterrupted one, with
                  the floor two uninterrupted runs give
  --rehearse      the rehearsal sizes (settings.rehearsal), outputs under results/run2_rehearsal/
  --stop-after S  save a checkpoint at step S and exit 3 (the rehearsal's resume test)
Prints every scored number, "STAGE x DONE" and a closing "FINISHED <stage> <minutes> <files written>".
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from commute import crosscoder as CC
from commute import kv
from commute import steering as ST
from commute.data import SETS_DIR
from commute.jobs import hard_exit, load_experiment, require_filed
from commute.open_pair import jsonable

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PRED = HERE / "results/run2/predictions.json"
SM = load_experiment("size_matched", "run")  # pairs(): the share's states per pair and their inventory
W = load_experiment("share_untrained", "windows")  # load_read: each task's read windows, as the share read them
M67 = REPO / "experiments/size_matched/results/m67.json"  # E187's c and larger side per pair
ROW = 511  # tokens a 512-token Pile row gives with position 0 dropped
TASK_ROWS = 32768  # every task window (key-value, e62) was cut from Pile train rows below this; e53 uses Pile val
STAGE1 = ("reuse", "new", "random", "reuse_seed1")
OUT: dict[str, Path] = {}  # results, sides, large: set in main by --rehearse
WRITTEN: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)


class Stop(Exception):
    """A filed stop: the stage file records why; the job exits 1."""


def write(path: Path, out: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(jsonable(out), indent=1, sort_keys=True))
    tmp.replace(path)
    WRITTEN.append(str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path))


def check(out: dict[str, Any], path: Path, name: str, ok: bool, detail: dict[str, Any]) -> None:
    out.setdefault("checks", {})[name] = {"pass": ok, **detail}
    log(f"[check {name}] {'ok' if ok else 'MISSED'} {json.dumps(jsonable(detail))}")
    if not ok:
        out["stopped"] = name
        write(path, out)
        raise Stop(name)


def cached(split: str) -> list[tuple[int, int, Path]]:
    return sorted((int(p.stem.split("_")[1]), int(p.stem.split("_")[2]), p) for p in SETS_DIR.glob(f"{split}_*_*.npy"))


def rows(split: str, start: int, n: int) -> Tensor:
    """Pile rows start..start+n of a split from the smallest cached file that holds them (`prep` makes the train ones)."""
    hold = [(m, s, p) for s, m, p in cached(split) if s <= start and start + n <= s + m]
    if not hold:
        raise SystemExit(f"no cached {split} file holds rows {start}..{start + n} under {SETS_DIR}: run `run.py prep`")
    _, s, p = min(hold)
    return torch.from_numpy(np.load(p, mmap_mode="r")[start - s : start - s + n].copy()).long()


def pile_rows(cfg: CC.Cfg) -> int:
    """Pile rows for half of one epoch's tokens, rounded up to whole chunks."""
    return math.ceil(cfg.tokens / cfg.epochs / 2 / ROW / cfg.chunk_rows) * cfg.chunk_rows


def prep(cfg: CC.Cfg) -> None:
    """The Pile half and the fresh-window pool, each as one file cut from the contiguous cached train files."""
    for start, n in ((cfg.pile_start, pile_rows(cfg)), (cfg.pool_start, cfg.pool_rows)):
        parts, at = [], start
        for s, m, p in cached("train"):
            if s <= at < s + m and at < start + n:
                take = min(s + m, start + n) - at
                parts.append(np.load(p, mmap_mode="r")[at - s : at - s + take])
                at += take
        assert at == start + n, f"cached files cover rows {start}..{at}, not ..{start + n}"
        out = SETS_DIR / f"train_{start}_{n}.npy"
        np.save(out, np.concatenate(parts))
        log(f"[prep] {out} ({n} rows, {out.stat().st_size / 1e6:.0f} MB)")


@dataclass
class Side:
    key: str  # file name stem
    task: Any  # share_untrained.windows.Read
    task_name: str
    dW: ST.DW  # the update (zeros for the self control)
    plant: Tensor | None = None  # (d,) added to the fine-tune side at each task window's answer position
    seed: int = 0  # crosscoder seed


class Grab(Exception):
    """Raised by the site hook once the activation is taken, so the rest of the forward is skipped."""


def stage2_sides(names: dict[str, Any], m67: dict[str, Any]) -> list[tuple[str, float, str]]:
    """(state, c, key): every state of the nine pairs once, then each pair's larger side times its c."""
    out = {s: (s, 1.0, s.replace("/", "__")) for r, n in names.values() for s in r + n}
    for pair, (r, n) in names.items():
        p = m67[pair]
        for s in r if p["larger"] == "reuse" else n:
            key = f"{s.replace('/', '__')}__matched_{pair}"
            out[key] = (s, float(p["c"]), key)
    return list(out.values())


class Job:
    def __init__(self, P: dict[str, Any], cfg: CC.Cfg) -> None:
        self.P, self.cfg, S = P, cfg, P["settings"]
        self.layer, self.seed = S["layer"], S["data_seed"]
        self.net = ST.Net("cuda")
        self.names, _, self.inv = SM.pairs()
        self.m67 = json.loads(M67.read_text())["pairs"]
        n_pile = pile_rows(cfg)
        assert TASK_ROWS <= cfg.pile_start and cfg.pile_start + n_pile <= cfg.pool_start, (cfg, n_pile)
        self.pile, self.pool = rows("train", cfg.pile_start, n_pile), rows("train", cfg.pool_start, cfg.pool_rows)
        self.val = rows("val", 0, cfg.val_rows)
        self.reads = {t: W.load_read(t, spec) for t, spec in self.inv["tasks"].items()}
        self.kv = {t: torch.load(REPO / spec["path"], weights_only=False)["task"] for t, spec in self.inv["tasks"].items() if spec["format"] == "kv"}
        self.zero = {k: torch.zeros_like(w) for k, w in self.net.W0.items()}
        log(f"[job] {cfg}\n[job] Pile rows {cfg.pile_start}..{cfg.pile_start + n_pile}, pool rows {cfg.pool_start}.."
            f"{cfg.pool_start + cfg.pool_rows}, {cfg.steps} steps")  # fmt: skip

    # ------------------------------------------------------------------------- activations

    @torch.no_grad()
    def acts(self, toks: Tensor, dW: ST.DW | None, plant: Tensor | None = None, pos: int = -1, n: int = 64) -> Tensor:
        """(m, d) raw block-`layer` inputs at every position but 0, under W0 (None) or W0 + dW, fp32 (TF32 is off for the
        whole job); `plant` added at position `pos`."""
        out: list[Tensor] = []

        def grab(x: Tensor) -> Tensor:
            x = x.clone()
            if plant is not None:
                x[:, pos] += plant.to(x.dtype)
            out.append(x[:, 1:])
            raise Grab

        with self.net.at(dW):
            for b in toks.split(n):
                try:
                    self.net.run(b, edit={f"in{self.layer}": grab})
                except Grab:
                    pass
        return torch.cat(out).reshape(-1, out[0].shape[-1])

    def pair(self, toks: Tensor, side: Side, task: bool) -> Tensor:
        """(m, 2, d): base then fine-tune; the plant only on task windows."""
        return torch.stack([self.acts(toks, None), self.acts(toks, side.dW, side.plant if task else None, side.task.half)], 1)

    def windows(self, task: Any, half: str) -> Tensor:
        toks = task.toks[:, : task.ci_len] if task.ci_len else task.toks
        return toks[0::2] if half == "fit" else toks[1::2]

    # ------------------------------------------------------------------------- the training stream

    def text(self, side: Side) -> dict[str, Any]:
        """How the side's task half is filled: fresh key-value windows, or fit windows capped at repeat_cap an epoch."""
        n_chunks, fresh = math.ceil(len(self.pile) / self.cfg.chunk_rows), side.task_name in self.kv
        T = 2 * side.task.half if fresh else self.windows(side.task, "fit").shape[1] - 1
        full = math.ceil(self.cfg.chunk_rows * ROW / T)  # windows a chunk takes for an even split
        unique = 0 if fresh else len(self.windows(side.task, "fit"))
        per_chunk = full if fresh else min(full, max(1, self.cfg.repeat_cap * unique // n_chunks))
        draws = per_chunk * n_chunks
        return {"source": "fresh key-value windows" if fresh else "fit-half windows", "tokens_per_window": T,
                "per_chunk": per_chunk, "draws_per_epoch": draws, "unique": draws if fresh else unique,
                "repeat_per_epoch": 1.0 if fresh else draws / unique,
                "task_share": draws * T / (draws * T + len(self.pile) * ROW)}  # fmt: skip

    def plan(self, side: Side) -> list[tuple[int, int, int, int]]:
        """(epoch, chunk, Pile rows, task windows) in stream order; chunk order permuted per epoch."""
        C, per = self.cfg.chunk_rows, self.text(side)["per_chunk"]
        n = math.ceil(len(self.pile) / C)
        out = []
        for e in range(self.cfg.epochs):
            for c in torch.randperm(n, generator=torch.Generator().manual_seed(self.seed + e)).tolist():
                r = min(C, len(self.pile) - c * C)
                out.append((e, c, r, math.ceil(per * r / C)))
        return out

    def task_windows(self, side: Side, e: int, c: int, n_win: int, gen: torch.Generator) -> Tensor:
        if side.task_name in self.kv:  # seeded by the chunk only: the same windows in every epoch
            t = self.kv[side.task_name]
            return kv.windows(self.pool, t.marker, t.keys, side.task.half, n_win, self.seed * 7919 + c).tokens
        fit = self.windows(side.task, "fit")
        return fit[torch.randint(len(fit), (n_win,), generator=gen)]

    def chunk(self, side: Side, e: int, c: int, n_win: int) -> Tensor:
        """One chunk's (m, 2, d) raw pairs: its Pile rows and n_win task windows, shuffled together."""
        gen = torch.Generator().manual_seed(self.seed * 100_003 + e * 1_009 + c)
        C = self.cfg.chunk_rows
        h = torch.cat([self.pair(self.pile[c * C :][:C], side, False), self.pair(self.task_windows(side, e, c, n_win, gen), side, True)])
        return h[torch.randperm(len(h), generator=gen).to(h.device)]

    def stream(self, side: Side, start: int) -> Iterator[Tensor]:
        """(B, 2, d) batches from step `start` on: whole chunks before it are skipped without computing them."""
        B, T = self.cfg.batch, self.text(side)["tokens_per_window"]
        skip, buf = start * B, None
        for e, c, r, n_win in self.plan(side):
            size = r * ROW + n_win * T
            if skip >= size:
                skip -= size
                continue
            h = self.chunk(side, e, c, n_win)[skip:]
            skip, buf = 0, h if buf is None else torch.cat([buf, h])
            while len(buf) >= B:
                yield buf[:B]
                buf = buf[B:]

    # ------------------------------------------------------------------------- one crosscoder

    def run_side(self, side: Side, snap: bool, save_dec: bool, early: Callable[[dict[str, Any]], None] | None = None,
                 stop_after: int | None = None) -> dict[str, Any]:  # fmt: skip
        t0, cfg = time.time(), self.cfg
        e, c, _, n_win = self.plan(side)[0]
        h0 = self.chunk(side, e, c, n_win)
        mean, std = CC.normaliser(h0)
        h_ls, h_task = h0[: cfg.ls_tokens].clone(), self.pair(self.windows(side.task, "score"), side, True)
        del h0
        tr = CC.trainer(cfg, mean, std, side.seed)
        ck, start, snaps, curve = OUT["large"] / "ckpt" / f"{side.key}.pt", 0, [], []
        if ck.is_file():
            s = torch.load(ck, weights_only=False)
            start, snaps, curve = CC.load_state(tr, s), s["snaps"], s["curve"]
            log(f"[side {side.key}] resumed at step {start}")
        marks = {int(f * cfg.steps) for f in cfg.snapshots} if snap else set()
        log(f"[side {side.key}] training steps {start}..{cfg.steps}; task text {json.dumps(self.text(side))}")

        def hook(step: int) -> None:
            if step in marks:
                snaps.append(self.score(tr.ae, side, h_ls, h_task, step, snaps[-1] if snaps else None)[0])
                self.show(side, snaps[-1])
                if early is not None:
                    early(snaps[-1])
            if step % cfg.ckpt_every == 0 or step in marks or step == stop_after:
                ck.parent.mkdir(parents=True, exist_ok=True)
                torch.save(CC.state(tr, step) | {"snaps": snaps, "curve": curve}, ck)
            if step == stop_after:
                log(f"[side {side.key}] stopped after step {step} (--stop-after)")
                hard_exit(3)

        CC.train(tr, self.stream(side, start), start, cfg.steps, cfg.log_every, curve, log, hook)
        rec, large = self.score(tr.ae, side, h_ls, h_task, cfg.steps, snaps[-1] if snaps else None)
        if save_dec:
            large["W_dec"] = tr.ae.decoder.weight.detach().half().cpu()
        OUT["large"].mkdir(parents=True, exist_ok=True)
        torch.save(large, OUT["large"] / f"{side.key}.pt")
        WRITTEN.append(str(OUT["large"] / f"{side.key}.pt"))
        rec |= {"key": side.key, "seed": side.seed, "task_text": self.text(side), "snapshots": snaps, "curve": curve,
                "minutes": (time.time() - t0) / 60, "resumed_at": start}  # fmt: skip
        self.show(side, rec)
        ck.unlink(missing_ok=True)
        return rec

    def val_chunks(self, side: Side, h_task: Tensor) -> Iterator[Tensor]:
        for r in self.val.split(self.cfg.chunk_rows):
            yield self.pair(r, side, False)
        yield from CC.split(h_task, 65536)  # whole, its ~1M tokens' latents were 96 GiB (E203 stage 0 OOM)

    def score(self, ae: Any, side: Side, h_ls: Tensor, h_task: Tensor, step: int, prev: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
        """Every number of one crosscoder at `step`: FVE, the groups at N and at the wide N, the parts, d, the specific
        part, shared / (shared + specific), the lead latent (the one carrying most of the change), the overlap of the
        specific set with the previous score, the planted cosine."""
        cfg, rn = self.cfg, CC.relative_norm(ae)
        ev = CC.fve(ae, self.val_chunks(side, h_task))
        alive = ev.pop("fired") >= cfg.alive_min
        ch = CC.carried(ae, h_task)
        s = ch.pop("s")
        lead = int(torch.where(alive, s.nan_to_num(-math.inf), -math.inf).argmax()) if ch["energy"] > 0 else -1
        J = torch.nonzero(CC.top(rn, cfg.top_n_wide) | (torch.arange(len(rn)) == lead)).squeeze(1)
        b = CC.betas(ae, h_ls, J)
        spec = torch.zeros(len(rn), dtype=torch.bool)
        spec[J] = CC.specific(cfg, b)
        rec: dict[str, Any] = {"step": step, "tokens": step * cfg.batch, **ev, **ch, "theta": float(ae.threshold),
                               "dead": float((~alive).float().mean()),
                               "max_rn_dev_alive": float((rn[alive] - 0.5).abs().max()) if bool(alive.any()) else math.nan}  # fmt: skip
        g = {}
        for tag, n in (("", cfg.top_n), ("_wide", cfg.top_n_wide)):
            g[tag] = CC.groups(cfg, rn, alive, CC.top(rn, n), spec)
            p = CC.parts(s, ch["bias"], g[tag])
            both = p["shared"] + p["ft_specific"]
            rec |= {f"parts{tag}": p, f"counts{tag}": {k: int(m.sum()) for k, m in g[tag].items()}, f"d{tag}": p["shared"] - p["ft_specific"],
                    f"specific{tag}": p["ft_specific"], f"shared_frac{tag}": p["shared"] / both if both else math.nan}  # fmt: skip
        rec["spec"] = torch.nonzero(g[""]["ft_specific"]).squeeze(1).tolist()
        if prev is not None:
            a, bb = set(prev["spec"]), set(rec["spec"])
            rec["jaccard_prev"] = len(a & bb) / len(a | bb) if a | bb else math.nan
        if lead >= 0:
            i = int((J == lead).nonzero())
            rec["lead"] = {"latent": lead, "s": float(s[lead]), "rn": float(rn[lead]), "rank": 1 + int((rn > rn[lead]).sum()),
                           "nu_r": float(b["nu_r"][i]), "nu_eps": float(b["nu_eps"][i]), "specific": bool(g[""]["ft_specific"][lead])}  # fmt: skip
        if side.plant is not None:
            Wr, _ = CC.raw(ae)
            dd = (Wr[1] - Wr[0])[g[""]["ft_specific"].to(Wr.device)]
            cos = torch.nn.functional.cosine_similarity(dd, side.plant[None].to(dd.device), dim=1)
            rec["cos_planted_best"] = float(cos.max()) if len(cos) else math.nan
        large = {"relative_norm": rn.half(), "alive": alive, "s": s.float(), "scaled": J, "groups": g[""],
                 "betas": {k: v.float() for k, v in b.items()}}  # fmt: skip
        return rec, large

    def show(self, side: Side, r: dict[str, Any]) -> None:
        keys = ("tokens", "fve_base", "fve_ft", "fve_change", "dead", "d", "specific", "shared_frac", "d_wide", "parts",
                "counts", "lead", "jaccard_prev", "cos_planted_best", "minutes")  # fmt: skip
        log(f"[side {side.key}] {json.dumps(jsonable({k: r[k] for k in keys if k in r}))}")

    # ------------------------------------------------------------------------- the sides

    def state(self, name: str, c: float = 1.0, seed: int = 0, key: str | None = None) -> Side:
        st = self.inv["states"][name]
        d = torch.load(REPO / st["file"], weights_only=False)
        dW = {k: v.float().to(self.net.W0[k].device) * c for k, v in d.items()}
        return Side(key or name.replace("/", "__"), self.reads[st["task"]], st["task"], dW, seed=seed)

    def random_like(self, name: str) -> Side:
        """A Gaussian update at `name`'s per-matrix norms, the do-nothing reference."""
        s, gen = self.state(name), torch.Generator().manual_seed(self.P["settings"]["random_seed"])
        R = {}
        for k, v in s.dW.items():
            r = torch.randn(v.shape, generator=gen).to(v.device)
            R[k] = r * (v.norm() / r.norm())
        return Side(f"random_like__{s.key}", s.task, s.task_name, R)

    def planted(self, task: str) -> Side:
        """The base plus one random unit direction times the median base norm at the answer position (run 1's seed)."""
        t, gen = self.reads[task], torch.Generator().manual_seed(self.P["settings"]["plant_seed"])
        h = self.acts(self.windows(t, "fit")[:, : t.half + 1], None).reshape(-1, t.half, self.cfg.d)[:, -1]
        v = torch.randn(self.cfg.d, generator=gen)
        return Side("planted", t, task, self.zero, plant=(v / v.norm() * h.norm(dim=1).median().cpu()).cuda())


# ----------------------------------------------------------------------------- stages


def stage0a(job: Job, T: dict[str, Any], stop_after: int | None) -> None:
    path = OUT["results"] / "stage0a.json"
    out: dict[str, Any] = {"experiment": job.P["experiment"], "card": torch.cuda.get_device_name()}
    task_name = job.P["settings"]["control_task"]
    task = job.reads[task_name]
    toks = job.windows(task, "fit")[:8]
    diff = float((job.acts(toks, None) - job.acts(toks, job.zero)).abs().max())
    check(out, path, "a self path: equal activations", diff <= T["self_act_diff"], {"max_abs": diff, "bar": T["self_act_diff"]})
    first = min(int(f * job.cfg.steps) for f in job.cfg.snapshots)

    def early(r: dict[str, Any]) -> None:
        if r["step"] == first:
            f = min(r["fve_base"], r["fve_ft"])
            check(out, path, "a self: FVE at the first snapshot", f >= T["fve_early"], {"fve": f, "bar": T["fve_early"]})

    rec = out["self"] = job.run_side(Side("self", task, task_name, job.zero), True, True, early, stop_after)
    write(path, out)
    check(out, path, "a self: relative norms at 0.5", rec["max_rn_dev_alive"] <= T["self_rn_dev"], {"max_dev": rec["max_rn_dev_alive"], "bar": T["self_rn_dev"]})
    check(out, path, "a self: no fine-tune-specific latent", rec["counts"]["ft_specific"] == 0, {"counts": rec["counts"]})
    check(out, path, "a self: FVE", min(rec["fve_base"], rec["fve_ft"]) >= T["fve_min"], {"fve": [rec["fve_base"], rec["fve_ft"]], "bar": T["fve_min"]})
    check(out, path, "a self: dead share", rec["dead"] <= T["dead_max"], {"dead": rec["dead"], "bar": T["dead_max"]})
    out["pass"] = True
    write(path, out)


def stage0b(job: Job, T: dict[str, Any]) -> None:
    path = OUT["results"] / "stage0b.json"
    out: dict[str, Any] = {"experiment": job.P["experiment"], "card": torch.cuda.get_device_name()}
    rec = out["planted"] = job.run_side(job.planted(job.P["settings"]["control_task"]), True, True)
    write(path, out)
    lead = rec.get("lead", {})
    ok = (bool(lead.get("specific")) or not T["planted_lead_specific"]) and rec["specific"] >= T["planted_part"]
    check(out, path, "b planted: found, specific and carried", ok, {"lead": lead, "part": rec["specific"], "bar": T["planted_part"]})
    out["pass"] = True
    write(path, out)


def timing(job: Job, cfg: CC.Cfg) -> None:
    """ms per step at the run's size, fp32 (the source) and TF32, 300 steps each after 50 warm ones; peak memory."""
    task_name = job.P["settings"]["control_task"]
    side = Side("self", job.reads[task_name], task_name, job.zero)
    out: dict[str, Any] = {"card": torch.cuda.get_device_name()}
    h = next(job.stream(side, 0))
    for tf32 in (False, True):
        torch.backends.cuda.matmul.allow_tf32 = tf32
        tr = CC.trainer(cfg, *CC.normaliser(h), 0)
        for i in range(50):  # step numbers past the k annealing and the threshold start, as most of a run is
            tr.update(5000 + i, h.clone())
        torch.cuda.synchronize()
        t0 = time.time()
        for i in range(50, 350):
            tr.update(5000 + i, h.clone())
        torch.cuda.synchronize()
        ms = (time.time() - t0) / 300 * 1000
        out["tf32" if tf32 else "fp32"] = {"ms_per_step": ms, "train_minutes": ms * cfg.steps / 60_000}
        log(f"[timing] tf32={tf32}: {ms:.2f} ms/step, {ms * cfg.steps / 60_000:.1f} min of steps per crosscoder")
    torch.backends.cuda.matmul.allow_tf32 = False
    out["peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
    write(OUT["results"] / "timing.json", out)


def stage1(job: Job, key: str) -> None:
    """One of stage 1's four crosscoders; its bars are read by `1check`."""
    reuse, _new = job.P["settings"]["first_pair_states"]
    sides = {"reuse": lambda: job.state(reuse), "new": lambda: job.state(_new), "random": lambda: job.random_like(reuse),
             "reuse_seed1": lambda: job.state(reuse, seed=1, key=f"{reuse.replace('/', '__')}__seed1")}  # fmt: skip
    side = sides[key]()
    rec = job.run_side(side, True, True)
    write(OUT["results"] / f"stage1_{key}.json", {"experiment": job.P["experiment"], "card": torch.cuda.get_device_name(), key: rec})
    if key in ("reuse", "new"):
        write(OUT["sides"] / f"{side.key}.json", rec | {"state": side.key.replace("__", "/"), "c": 1.0})


def stage1check(T: dict[str, Any]) -> None:
    """Stage 1's bars on its four files and stage 0's: FVE drop and dead share per side, the reseed, the change reconstructed."""
    path = OUT["results"] / "stage1.json"
    out: dict[str, Any] = {"experiment": "E203"}
    s0 = {k: json.loads((OUT["results"] / f"stage0{k}.json").read_text()) for k in "ab"}
    if not (s0["a"].get("pass") and s0["b"].get("pass")):
        raise SystemExit("refusing 1check: stage0a.json and stage0b.json have not both passed")
    fve0 = min(s0["a"]["self"]["fve_base"], s0["a"]["self"]["fve_ft"])
    for key in STAGE1:
        rec = out[key] = json.loads((OUT["results"] / f"stage1_{key}.json").read_text())[key]
        f = min(rec["fve_base"], rec["fve_ft"])
        check(out, path, f"1 FVE {key}", f >= fve0 - T["fve_drop"], {"fve": f, "self": fve0, "drop": T["fve_drop"]})
        check(out, path, f"1 dead share {key}", rec["dead"] <= T["dead_max"], {"dead": rec["dead"], "bar": T["dead_max"]})
    dd = abs(out["reuse"]["d"] - out["reuse_seed1"]["d"])
    check(out, path, "1 crosscoder reseed", dd <= T["reseed_d"], {"d0": out["reuse"]["d"], "d1": out["reuse_seed1"]["d"], "bar": T["reseed_d"]})
    expl = {k: 1 - out[k]["parts"]["error"] for k in ("reuse", "new")}
    check(out, path, "1 change reconstructed", max(expl.values()) >= T["explained_min"], {"explained": expl, "bar": T["explained_min"]})
    out["pass"], out["fve_self"] = True, fve0
    write(path, out)


def stage2(job: Job, T: dict[str, Any], shard: tuple[int, int], t_job: float) -> None:
    s1 = OUT["results"] / "stage1.json"
    if not (s1.is_file() and json.loads(s1.read_text()).get("pass")):
        raise SystemExit("refusing stage 2: stage1.json has not passed")
    fve0 = json.loads(s1.read_text())["fve_self"]
    done = {s.replace("/", "__") for s in job.P["settings"]["first_pair_states"]}  # read in stage 1
    todo = [s for i, s in enumerate(stage2_sides(job.names, job.m67))
            if i % shard[1] == shard[0] and s[2] not in done and not (OUT["sides"] / f"{s[2]}.json").is_file()]  # fmt: skip
    path, out = OUT["results"] / f"stage2_{shard[0]}_{shard[1]}.json", {"experiment": job.P["experiment"], "shard": list(shard)}
    log(f"[stage 2] shard {shard[0]}/{shard[1]}: {len(todo)} sides to train")
    for i, (state, c, key) in enumerate(todo):
        rec = job.run_side(job.state(state, c, key=key), False, False)
        write(OUT["sides"] / f"{key}.json", rec | {"state": state, "c": c})
        f = min(rec["fve_base"], rec["fve_ft"])
        check(out, path, f"2 FVE {key}", f >= fve0 - T["fve_drop"], {"fve": f, "self": fve0, "drop": T["fve_drop"]})
        check(out, path, f"2 dead share {key}", rec["dead"] <= T["dead_max"], {"dead": rec["dead"], "bar": T["dead_max"]})
        minutes = (time.time() - t_job) / 60 / (i + 1) * len(todo)
        log(f"[stage 2] {i + 1}/{len(todo)} done ({key}); projected shard minutes {minutes:.0f}")
        check(out, path, f"2 projected time after {key}", minutes <= T["projected_limit_min"], {"minutes": minutes, "bar": T["projected_limit_min"]})
    out["pass"] = True
    write(path, out)


def resumecheck() -> None:
    """Rehearsal: loss at matched steps of the resumed self run (stage0a.json) against an uninterrupted one
    (stage0a_A.json), beside the floor two uninterrupted runs give (A against stage0a_B.json). Pass when every
    post-resume difference is within twice the floor plus 1e-6 of the loss."""
    c = {k: {r["step"]: r["loss"] for r in json.loads((OUT["results"] / f"stage0a{k}.json").read_text())["self"]["curve"]} for k in ("_A", "_B", "")}
    at = json.loads((OUT["results"] / "stage0a.json").read_text())["self"]["resumed_at"]
    rel = lambda x, y: abs(x - y) / max(abs(y), 1e-12)
    floor = max(rel(c["_B"][s], c["_A"][s]) for s in c["_A"] if s in c["_B"])
    after = {s: rel(c[""][s], c["_A"][s]) for s in c["_A"] if s in c[""] and s >= at}
    ok = bool(after) and at > 0 and max(after.values()) <= 2 * floor + 1e-6
    out = {"resumed_at": at, "floor_max_rel": floor, "resumed_max_rel": max(after.values()) if after else math.nan,
           "steps_compared": len(after), "pass": ok, "diffs": after}  # fmt: skip
    write(OUT["results"] / "resume_check.json", out)
    log(f"[resumecheck] {'PASS' if ok else 'FAIL'} {json.dumps(jsonable({k: v for k, v in out.items() if k != 'diffs'}))}")
    if not ok:
        raise Stop("resume check")


def inputs(shard: tuple[int, int]) -> None:
    """The files a stage-2 shard reads beyond the common ones (artifacts/vpd, the Pile sets, the task windows)."""
    names, _, inv = SM.pairs()
    m67 = json.loads(M67.read_text())["pairs"]
    sides = [s for i, s in enumerate(stage2_sides(names, m67)) if i % shard[1] == shard[0]]
    files = sorted({inv["states"][s]["file"] for s, _, _ in sides})
    tasks = sorted({inv["tasks"][inv["states"][s]["task"]].get("path", "(built from the Pile val set)") for s, _, _ in sides})
    size = sum((REPO / f).stat().st_size for f in files)
    print(json.dumps({"shard": f"{shard[0]}/{shard[1]}", "sides": [k for _, _, k in sides], "states": files, "tasks": tasks,
                      "state_mb": round(size / 1e6)}, indent=1))  # fmt: skip


def run(a: argparse.Namespace, P: dict[str, Any], cfg: CC.Cfg, tag: str, t_job: float) -> None:
    T = P["thresholds"][tag]
    if a.stage == "1check":
        stage1check(T)
        return
    if a.stage == "resumecheck":
        resumecheck()
        return
    job = Job(P, cfg)
    if a.stage == "0a":
        stage0a(job, T, a.stop_after)
    elif a.stage == "0b":
        stage0b(job, T)
    elif a.stage == "t":
        timing(job, cfg)
    elif a.stage == "1":
        if a.arg not in STAGE1:
            raise SystemExit(f"stage 1 takes one of {STAGE1}")
        stage1(job, a.arg)
    else:
        i, n = (int(x) for x in a.arg.split("/"))
        stage2(job, T, (i, n), t_job)
    log(f"[done] peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "inputs", "0a", "0b", "t", "1", "1check", "2", "resumecheck"])
    ap.add_argument("arg", nargs="?", default="0/1")  # a shard i/n, or stage 1's key
    ap.add_argument("--rehearse", action="store_true")
    ap.add_argument("--stop-after", type=int, default=None)
    a = ap.parse_args()
    t_job, P = time.time(), json.loads(PRED.read_text())
    tag = "rehearsal" if a.rehearse else "full"
    S = P["settings"][tag]
    cfg = CC.Cfg(**(S | {"shared": tuple(S["shared"]), "snapshots": tuple(S["snapshots"])}))
    sub = "run2_rehearsal" if a.rehearse else "run2"
    OUT.update(results=HERE / "results" / sub, sides=HERE / "results" / sub / "sides", large=HERE / "results/large" / sub)
    code = 0
    if a.stage == "prep":
        prep(cfg)
    elif a.stage == "inputs":
        i, n = (int(x) for x in a.arg.split("/"))
        inputs((i, n))
    else:
        require_filed(PRED)
        if P["experiment"] != "E203":
            raise SystemExit("refusing to start: results/run2/predictions.json is not E203's")
        torch.backends.cuda.matmul.allow_tf32 = False  # fp32 throughout, as the source's code runs
        try:
            run(a, P, cfg, tag, t_job)
            print(f"STAGE {a.stage} DONE", flush=True)
        except Stop as e:
            log(f"STOP: {e}: tell the lead")
            code = 1
    log(f"FINISHED {a.stage} {a.arg if a.stage in ('1', '2', 'inputs') else ''} in {(time.time() - t_job) / 60:.1f} min, exit {code}; wrote: {' '.join(dict.fromkeys(WRITTEN)) or 'nothing'}")
    return code


if __name__ == "__main__":
    hard_exit(main())
