"""One crosscoder per side on the 67M, as E203 trained them (cell P2.83; plan
reports/plans/2026-11-25_plan_p2-83-crosscoder-fix.md). E203's stream, training loop and scoring
(experiments/crosscoder/run.py `Job`, frozen, keeps its own copy) re-expressed for new runs, with three changes:

1. A side whose task text cannot be made fresh (e53, e62: fit-half windows drawn at most `repeat_cap` times an epoch)
   gets enough Pile rows to fill the token budget. E203 gave every side half the budget in Pile rows, so these sides'
   streams ran out at about half the steps (TRIED E203).
2. The planted direction's positions are a mask (`Plant`): E203's answer position on every task window ("answer"), the
   answer position on a seeded half of the task windows ("answer_half"), or content-independent positions drawn at a
   fixed rate over every token of the stream ("random"). Every variant is read as E203 read it: on the score windows,
   planted at the answer position of each.
3. Scoring is `crosscoder.score` (E212's re-expression of E203's `Job.score`, with the authors' exclusion of latents
   with a negative base beta, Minder et al. App. E.2).

The crosscoder itself is the authors' pinned trainer through `commute.crosscoder` (`trainer`, `train`). Model index 0
is the base, 1 the fine-tune. Activations are block `layer`'s input at every position but 0, fp32.
"""

from __future__ import annotations

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
from commute.jobs import hard_exit

ROW = 511  # tokens a 512-token Pile row gives with position 0 dropped
TASK_ROWS = 32768  # every task window (key-value, e62) was cut from Pile train rows below this; e53 uses Pile val


@dataclass(frozen=True)
class Plant:
    vec: Tensor  # (d,) raw units, added on the fine-tune side
    where: str  # "answer" | "answer_half" | "random"
    rate: float  # "random": probability per position (position 0 never)
    seed: int

    def __post_init__(self) -> None:
        assert self.where in ("answer", "answer_half", "random"), self.where

    def mask(self, n: int, L: int, half: int, task: bool, key: int) -> Tensor | None:
        """(n, L) bool on rows of length L (position 0 included): where the plant goes on these training rows."""
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + key)
        if self.where == "random":
            m = torch.rand(n, L, generator=g) < self.rate
            m[:, 0] = False
            return m
        if not task:
            return None
        m = torch.zeros(n, L, dtype=torch.bool)
        rows = torch.rand(n, generator=g) < 0.5 if self.where == "answer_half" else torch.ones(n, dtype=torch.bool)
        m[rows, half] = True
        return m


def read_mask(n: int, L: int, half: int) -> Tensor:
    """The read's plant: the answer position of every score window (E203's read, for every variant)."""
    m = torch.zeros(n, L, dtype=torch.bool)
    m[:, half] = True
    return m


@dataclass
class Side:
    key: str  # file name stem
    task: Any  # share_untrained.windows.Read: toks, half, ci_len
    task_name: str
    dW: ST.DW  # the update (zeros for the self control and the plant)
    plant: Plant | None = None
    seed: int = 0  # crosscoder seed


class _Grab(Exception):
    """Raised by the site hook once the activation is taken, so the rest of the forward is skipped."""


def cached(split: str) -> list[tuple[int, int, Path]]:
    return sorted((int(p.stem.split("_")[1]), int(p.stem.split("_")[2]), p) for p in SETS_DIR.glob(f"{split}_*_*.npy"))


def rows(split: str, start: int, n: int) -> Tensor:
    """Pile rows start..start+n of a split from the smallest cached file that holds them (`prep` makes the long one)."""
    hold = [(m, s, p) for s, m, p in cached(split) if s <= start and start + n <= s + m]
    if not hold:
        raise SystemExit(f"no cached {split} file holds rows {start}..{start + n} under {SETS_DIR}: run `run.py prep`")
    _, s, p = min(hold)
    return torch.from_numpy(np.load(p, mmap_mode="r")[start - s : start - s + n].copy()).long()


def prep_rows(start: int, n: int, log: Callable[[str], None]) -> Path:
    """Pile train rows start..start+n as one file, cut from the contiguous cached train files (E203's `prep`)."""
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
    return out


def windows(task: Any, half: str) -> Tensor:
    """E203's split of a task's read windows: even rows fit (the training text), odd rows score (the read)."""
    toks = task.toks[:, : task.ci_len] if task.ci_len else task.toks
    return toks[0::2] if half == "fit" else toks[1::2]


def stream_text(cfg: CC.Cfg, task: Any, fresh: bool) -> dict[str, Any]:
    """How a side's stream is filled. Fresh key-value windows: half the budget each epoch, the other half Pile rows
    (E203). Fit-half windows: each drawn at most `repeat_cap` times an epoch, Pile rows filling the rest (change 1)."""
    C, fit = cfg.chunk_rows, windows(task, "fit")
    T = 2 * task.half if fresh else fit.shape[1] - 1
    per_epoch = cfg.tokens / cfg.epochs
    task_tokens = per_epoch / 2 if fresh else cfg.repeat_cap * len(fit) * T
    n_pile = math.ceil(max(per_epoch - task_tokens, 0) / ROW / C) * C
    n_chunks = n_pile // C
    assert n_chunks >= 1, (cfg, task_tokens)
    full = math.ceil(C * ROW / T)  # windows a chunk takes for an even split
    per_chunk = full if fresh else min(full, max(1, cfg.repeat_cap * len(fit) // n_chunks))
    draws = per_chunk * n_chunks
    epoch_tokens = n_pile * ROW + draws * T
    assert cfg.epochs * epoch_tokens >= cfg.steps * cfg.batch, (epoch_tokens, cfg.steps * cfg.batch)
    if fresh:
        assert TASK_ROWS <= cfg.pile_start and cfg.pile_start + n_pile <= cfg.pool_start, (cfg, n_pile)
    return {"source": "fresh key-value windows" if fresh else "fit-half windows", "tokens_per_window": T,
            "pile_rows": n_pile, "per_chunk": per_chunk, "draws_per_epoch": draws,
            "unique": draws if fresh else len(fit), "repeat_per_epoch": 1.0 if fresh else draws / len(fit),
            "task_share": draws * T / epoch_tokens}  # fmt: skip


class Sides:
    """The base net, the Pile and pool rows, the task windows; one crosscoder per side (`run_side`)."""

    def __init__(self, cfg: CC.Cfg, layer: int, data_seed: int, reads: dict[str, Any], kv_tasks: dict[str, Any],
                 tf32: bool, floor_multiple: float, floor_folds: int, log: Callable[[str], None]) -> None:  # fmt: skip
        self.cfg, self.layer, self.seed, self.reads, self.kv, self.log = cfg, layer, data_seed, reads, kv_tasks, log
        # every score reads the literal and the floored exclusion (`crosscoder.beta_floor`, K folds of whole windows)
        self.floor_multiple, self.floor_folds = floor_multiple, floor_folds
        torch.backends.cuda.matmul.allow_tf32 = tf32
        self.net = ST.Net("cuda")
        self.pool = rows("train", cfg.pool_start, cfg.pool_rows)
        self.val = rows("val", 0, cfg.val_rows)
        self.zero = {k: torch.zeros_like(w) for k, w in self.net.W0.items()}
        self._pile: tuple[int, Tensor] | None = None

    # ------------------------------------------------------------------------- activations

    @torch.no_grad()
    def acts(self, toks: Tensor, dW: ST.DW | None, mask: Tensor | None = None, vec: Tensor | None = None,
             n: int = 64) -> Tensor:  # fmt: skip
        """(m, d) raw block-`layer` inputs at every position but 0, under W0 (None) or W0 + dW; `vec` added where
        `mask` (rows x L, position 0 included) is set."""
        out: list[Tensor] = []
        at = [0]

        def grab(x: Tensor) -> Tensor:
            x = x.clone()
            if mask is not None and vec is not None:
                m = mask[at[0] : at[0] + len(x)].to(x.device)
                x[m] += vec.to(x.device, x.dtype)
            out.append(x[:, 1:])
            raise _Grab

        with self.net.at(dW):
            for b in toks.split(n):
                try:
                    self.net.run(b, edit={f"in{self.layer}": grab})
                except _Grab:
                    pass
                at[0] += len(b)
        return torch.cat(out).reshape(-1, out[0].shape[-1])

    def pair(self, toks: Tensor, side: Side, mask: Tensor | None) -> Tensor:
        """(m, 2, d): base (never planted), then the fine-tune side (its update, the plant where `mask` is set)."""
        vec = side.plant.vec if side.plant is not None else None
        return torch.stack([self.acts(toks, None), self.acts(toks, side.dW, mask, vec)], 1)

    def read_pairs(self, side: Side) -> Tensor:
        """The side's read: its task's score windows, planted (if a plant side) at each window's answer position."""
        w = windows(side.task, "score")
        return self.pair(w, side, read_mask(w.shape[0], w.shape[1], side.task.half) if side.plant is not None else None)

    # ------------------------------------------------------------------------- the training stream

    def text(self, side: Side) -> dict[str, Any]:
        return stream_text(self.cfg, side.task, side.task_name in self.kv)

    def pile(self, n: int) -> Tensor:
        if self._pile is None or self._pile[0] != n:
            self._pile = (n, rows("train", self.cfg.pile_start, n))
        return self._pile[1]

    def plan(self, side: Side) -> list[tuple[int, int, int, int]]:
        """(epoch, chunk, Pile rows, task windows) in stream order; chunk order permuted per epoch (E203's seeds)."""
        t, C = self.text(side), self.cfg.chunk_rows
        n = t["pile_rows"] // C
        return [(e, c, C, t["per_chunk"]) for e in range(self.cfg.epochs)
                for c in torch.randperm(n, generator=torch.Generator().manual_seed(self.seed + e)).tolist()]  # fmt: skip

    def task_windows(self, side: Side, c: int, n_win: int, gen: torch.Generator) -> Tensor:
        if side.task_name in self.kv:  # seeded by the chunk only: the same windows in every epoch (E203)
            t = self.kv[side.task_name]
            return kv.windows(self.pool, t.marker, t.keys, side.task.half, n_win, self.seed * 7919 + c).tokens
        fit = windows(side.task, "fit")
        return fit[torch.randint(len(fit), (n_win,), generator=gen)]

    def chunk(self, side: Side, e: int, c: int, n_win: int) -> Tensor:
        """One chunk's (m, 2, d) raw pairs: its Pile rows and n_win task windows, shuffled together (E203's seeds)."""
        return self.chunk_ids(side, e, c, n_win)[0]

    def chunk_ids(self, side: Side, e: int, c: int, n_win: int) -> tuple[Tensor, Tensor]:
        """`chunk` and (m,) the window (Pile row or task window) each token came from, in the same shuffled order."""
        gen = torch.Generator().manual_seed(self.seed * 100_003 + e * 1_009 + c)
        C, P = self.cfg.chunk_rows, self.pile(self.text(side)["pile_rows"])
        prow, tw = P[c * C :][:C], self.task_windows(side, c, n_win, gen)
        key = (e * 100_000 + c) * 2
        half = side.task.half
        pm = side.plant.mask(prow.shape[0], prow.shape[1], half, False, key) if side.plant else None
        tm = side.plant.mask(tw.shape[0], tw.shape[1], half, True, key + 1) if side.plant else None
        h = torch.cat([self.pair(prow, side, pm), self.pair(tw, side, tm)])
        wid = torch.cat([torch.arange(len(prow)).repeat_interleave(prow.shape[1] - 1),
                         len(prow) + torch.arange(len(tw)).repeat_interleave(tw.shape[1] - 1)])  # fmt: skip
        perm = torch.randperm(len(h), generator=gen)
        return h[perm.to(h.device)], wid[perm]

    def folds(self, wid: Tensor) -> Tensor:
        """Each token's fold 0..K-1 for the floored exclusion: whole windows, assigned to folds in a seeded random order."""
        u, inv = wid.unique(return_inverse=True)
        fold_of = torch.randperm(len(u), generator=torch.Generator().manual_seed(self.seed)) % self.floor_folds
        return fold_of[inv]

    def stream(self, side: Side, start: int) -> Iterator[Tensor]:
        """(B, 2, d) batches from step `start` on: whole chunks before it are skipped without being built."""
        T = self.text(side)["tokens_per_window"]
        plan = self.plan(side)
        sizes = [r * ROW + n_win * T for _, _, r, n_win in plan]
        return CC.batches(sizes, lambda i: self.chunk(side, *plan[i][:2], plan[i][3]), self.cfg.batch, start)

    # ------------------------------------------------------------------------- one crosscoder

    def val_chunks(self, side: Side, h_read: Tensor) -> Iterator[Tensor]:
        for r in self.val.split(self.cfg.chunk_rows):
            yield self.pair(r, side, None)
        yield h_read

    def score(self, ae: Any, side: Side, h_ls: Tensor, h_read: Tensor, step: int, folds: Tensor) -> tuple[dict[str, Any], dict[str, Any]]:
        """`crosscoder.score` plus the step, the tokens and the largest relative-norm deviation from 0.5 of an alive
        latent (the self control's bar)."""
        cfg = self.cfg
        ev = CC.fve(ae, self.val_chunks(side, h_read))
        alive = ev["fired"] >= cfg.alive_min
        plant = side.plant.vec if side.plant is not None else None
        rec, large = CC.score(cfg, ae, ev, h_ls, h_read, plant, (folds, self.floor_multiple))
        rn = CC.relative_norm(ae)
        rec |= {"step": step, "tokens": step * cfg.batch,
                "max_rn_dev_alive": float((rn[alive] - 0.5).abs().max()) if bool(alive.any()) else math.nan}  # fmt: skip
        return rec, large

    def run_side(self, side: Side, large_dir: Path, snap: bool, save_dec: bool, checks: Callable[[str, dict[str, Any]], None],
                 stop_after: int | None = None) -> tuple[dict[str, Any], Any]:  # fmt: skip
        """Train and score one crosscoder, resuming from its checkpoint. `checks(kind, rec)` is called with kind
        "curve" on each training-curve record and "snapshot" on each snapshot score; it raises to stop the job."""
        t0, cfg = time.time(), self.cfg
        e, c, _, n_win = self.plan(side)[0]
        h0, wid = self.chunk_ids(side, e, c, n_win)
        mean, std = CC.normaliser(h0)
        h_ls, h_read, folds = h0[: cfg.ls_tokens].clone(), self.read_pairs(side), self.folds(wid[: cfg.ls_tokens])
        del h0
        tr = CC.trainer(cfg, mean, std, side.seed)
        ck, start, snaps, curve = large_dir / "ckpt" / f"{side.key}.pt", 0, [], []
        if ck.is_file():
            s = torch.load(ck, weights_only=False)
            start, snaps, curve = CC.load_state(tr, s), s["snaps"], s["curve"]
            self.log(f"[side {side.key}] resumed at step {start}")
        marks = {int(f * cfg.steps) for f in cfg.snapshots} if snap else set()
        self.log(f"[side {side.key}] training steps {start}..{cfg.steps}; task text {json.dumps(self.text(side))}")
        n_curve = [len(curve)]

        def hook(step: int) -> None:
            for r in curve[n_curve[0] :]:
                checks("curve", r | {"minutes": (time.time() - t0) / 60})
            n_curve[0] = len(curve)
            if step in marks:
                snaps.append(self.score(tr.ae, side, h_ls, h_read, step, folds)[0])
                self.show(side, snaps[-1])
                checks("snapshot", snaps[-1])
            if step % cfg.ckpt_every == 0 or step in marks or step == stop_after:
                ck.parent.mkdir(parents=True, exist_ok=True)
                torch.save(CC.state(tr, step) | {"snaps": snaps, "curve": curve}, ck)
            if step == stop_after:
                self.log(f"[side {side.key}] stopped after step {step} (--stop-after)")
                hard_exit(3)

        CC.train(tr, self.stream(side, start), start, cfg.steps, cfg.log_every, curve, self.log, hook)
        rec, large = self.score(tr.ae, side, h_ls, h_read, cfg.steps, folds)
        if save_dec:
            large["W_dec"] = tr.ae.decoder.weight.detach().half().cpu()
        large_dir.mkdir(parents=True, exist_ok=True)
        torch.save(large, large_dir / f"{side.key}.pt")
        rec |= {"key": side.key, "seed": side.seed, "task_text": self.text(side), "snapshots": snaps, "curve": curve,
                "minutes": (time.time() - t0) / 60, "resumed_at": start}  # fmt: skip
        self.show(side, rec)
        ck.unlink(missing_ok=True)
        return rec, tr.ae

    def show(self, side: Side, r: dict[str, Any]) -> None:
        keys = ("tokens", "fve_base", "fve_ft", "fve_change", "dead", "d_authors", "d_alt", "specific", "shared_frac",
                "d_authors_wide", "parts", "counts", "lead", "cos_planted_best", "max_rn_dev_alive", "beta_floor", "minutes")  # fmt: skip
        self.log(f"[side {side.key}] literal {json.dumps({k: r[k] for k in keys if k in r}, default=float)}")
        if "floored" in r:
            fl = r["floored"]
            self.log(f"[side {side.key}] floored {json.dumps({k: fl[k] for k in keys if k in fl}, default=float)}")

    # ------------------------------------------------------------------------- the sides

    def state(self, key: str, task_name: str, dW: ST.DW, seed: int = 0) -> Side:
        dev = next(iter(self.net.W0.values())).device
        return Side(key, self.reads[task_name], task_name, {k: v.float().to(dev) for k, v in dW.items()}, seed=seed)

    def random_like(self, key: str, task_name: str, dW: ST.DW, seed: int) -> Side:
        """A Gaussian update at dW's per-matrix norms, drawn in dict order from one generator (E203's `random_like`)."""
        gen, R = torch.Generator().manual_seed(seed), {}
        for k, v in dW.items():
            r = torch.randn(v.shape, generator=gen)
            R[k] = r * (v.float().norm() / r.norm())
        return self.state(key, task_name, R)

    def planted(self, task_name: str, where: str, rate: float, plant_seed: int, seed: int) -> Side:
        """E203's planted direction: one random unit vector (plant seed) times the median base norm at the answer
        position of the task's fit windows; placed by `where`; crosscoder seed `seed`."""
        t = self.reads[task_name]
        h = self.acts(windows(t, "fit")[:, : t.half + 1], None).reshape(-1, t.half, self.cfg.d)[:, -1]
        v = torch.randn(self.cfg.d, generator=torch.Generator().manual_seed(plant_seed))
        vec = (v / v.norm() * h.norm(dim=1).median().cpu()).cuda()
        key = f"planted_{where}_s{seed}"
        return Side(key, t, task_name, self.zero, Plant(vec, where, rate, plant_seed), seed)
