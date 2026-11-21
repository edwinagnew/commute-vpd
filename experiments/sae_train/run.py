"""E234 (cell P1.40, run 1): BatchTopK SAEs on the base 67M's activations at all 13 sites, SAEBench's recipe
(`commute.sae`, whose docstring cites the source line by line), then FVE, L0 and spliced-in cross-entropy on the base and
on each changed model. Card reports/plans/2026-11-20_plan_p1-40-card.md; settings, expected values and stops in
results/predictions.json (refused unless committed unchanged). The SAEs are never trained on a changed model.

  train     one forward of the base per 128 Pile train rows feeds every site's SAE (13, plus a second seed at
            `second_seed_site`), from the same buffer in the same order; a checkpoint every `ckpt_every` steps and a
            rerun resumes from it; the early check at `check_step` (loss down, dead share under `dead_max`) stops the
            job on a miss; the SAEs, with the normalisation folded in, go to results/large/<size>/sae_<name>.pt.
  0         (control, first and alone) the base on Pile validation rows: per SAE FVE, L0, dead share, CE clean /
            spliced / zero- / mean-ablated. Stops on base FVE < fve_min or a spliced CE increase > ce_increase_max.
  1         per item (the do-nothing clone first, then E214's six known-answer items, then E187's 68 states, built by
            kstar_vpd's World as P1.37 builds them): the same numbers on the same validation rows, and on the item's
            own task windows for the item and the base; one items/<item>_<shard>.json each, skipped once written.
  --size scored|rehearse|dry   the sizes (predictions.json settings + sizes[size]); rehearse and dry write under
                               results/<size>/ and results/large/<size>/
  --shard I N                  every N-th SAE from the I-th (each shard streams the same rows)
  --stop-after S               save the checkpoint at step S and exit 3 (the rehearsal's resume test)

    PRED_SHA=<sha> python experiments/sae_train/run.py --size scored --shard 0 4 2>&1 | tee -a experiments/sae_train/results/run_s0.log
Prints every scored number, "STAGE x DONE" and "FINISHED".
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from commute import sae as S
from commute.data import cached_rows
from commute.jobs import (
    config_from_settings,
    hard_exit,
    load_experiment,
    require_filed,
    stop,
    write_stage,
)
from commute.open_pair import jsonable

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED = RESULTS / "predictions.json"
ROW = 511  # tokens a 512-token row gives with position 0 dropped


def log(msg: str) -> None:
    print(msg, flush=True)


@dataclass(frozen=True)
class Config:
    d: int
    width: int
    k: int
    lr: float
    batch: int  # tokens per SAE step
    tokens: int  # training tokens per SAE
    warmup: int
    decay_frac: float  # the decay to 0 starts at this fraction of the steps
    auxk_alpha: float
    dead_tokens: int
    threshold_beta: float
    threshold_start: int
    buffer_tokens: int
    refresh_rows: int  # rows per forward of the base when the buffer refills
    norm_batches: int
    buffer_seed: int
    seed: int
    second_seed: int
    second_seed_site: str
    train_start: int  # first Pile train row read
    train_rows: int  # rows available from train_start (the cached files)
    eval_start: int  # first Pile validation row of the held-out read
    eval_rows: int
    eval_batch: int  # rows per forward in a read
    task_windows: int  # most task windows read per item
    ckpt_every: int
    log_every: int
    check_step: int
    dead_max: float
    fve_min: float
    ce_increase_max: float
    do_nothing_tol: float
    n_known: int
    n_states: int

    def __post_init__(self) -> None:
        assert 10 * self.k <= self.width and self.second_seed_site in S.SITES, self
        assert self.warmup < self.decay_start < self.steps and self.threshold_start < self.steps, self
        assert 200 <= self.check_step < self.steps and self.ckpt_every < self.steps, self
        assert self.buffer_tokens >= 2 * self.batch and self.norm_batches * self.batch <= self.buffer_tokens // 2, self
        assert self.rows_needed <= self.train_rows, (self.rows_needed, self.train_rows)
        assert 0 <= self.n_known <= 6 and 0 <= self.n_states <= 68, self

    @property
    def steps(self) -> int:
        return self.tokens // self.batch

    @property
    def decay_start(self) -> int:
        return int(self.steps * self.decay_frac)

    @property
    def rows_needed(self) -> int:
        """Rows for the steps, the normalisation batches and one buffer, plus 64 buffers for resumes (each drops <= one)."""
        return (self.tokens + self.norm_batches * self.batch + 65 * self.buffer_tokens) // ROW + 1

    def names(self) -> list[str]:
        return [f"{s}_s{self.seed}" for s in S.SITES] + [f"{self.second_seed_site}_s{self.second_seed}"]


def load_config(size: str) -> tuple[Config, dict[str, Any]]:
    pred = json.loads(PRED.read_text())
    s = dict(pred["settings"])
    sizes = s.pop("sizes")
    return config_from_settings(Config, s | sizes.get(size, {})), pred


def trainer(cfg: Config, name: str) -> S.Trainer:
    return S.Trainer(cfg.d, cfg.width, cfg.k, cfg.lr, cfg.steps, cfg.warmup, cfg.decay_start, cfg.auxk_alpha,
                     cfg.dead_tokens, cfg.threshold_beta, cfg.threshold_start, int(name.rsplit("_s", 1)[1]))  # fmt: skip


def site(name: str) -> str:
    return name.rsplit("_s", 1)[0]


# ---- training ----------------------------------------------------------------------------------------------------------


class Run:
    """The shard's trainers, the row cursor and the buffer; `save` / `resume` through one checkpoint file."""

    def __init__(self, cfg: Config, net: Any, names: list[str], ckpt: Path) -> None:
        self.cfg, self.net, self.names, self.ckpt = cfg, net, names, ckpt
        self.sites = tuple(dict.fromkeys(site(n) for n in names))
        self.col = {n: self.sites.index(site(n)) for n in names}
        self.t = {n: trainer(cfg, n) for n in names}
        self.step, self.cursor, self.nf = 0, cfg.train_start, torch.ones(len(self.sites))
        self.curve: list[dict[str, Any]] = []
        self.first: dict[str, list[float]] = {n: [] for n in names}  # the first 100 losses, for the early check
        if ckpt.is_file():
            s = torch.load(ckpt, map_location="cuda", weights_only=False)
            for n in names:
                self.t[n].load(s["trainers"][n])
            self.step, self.cursor, self.nf, self.curve, self.first = s["step"], s["cursor"], s["nf"], s["curve"], s["first"]
            log(f"resumed from {ckpt} at step {self.step}, row {self.cursor}")
        self.buf = S.Buffer(self.fill, cfg.buffer_tokens, cfg.batch, cfg.buffer_seed + self.step)

    def fill(self) -> torch.Tensor:
        c = self.cfg
        assert self.cursor + c.refresh_rows <= c.train_start + c.train_rows, f"out of train rows at {self.cursor}"
        rows = cached_rows("train", self.cursor, c.refresh_rows)
        self.cursor += c.refresh_rows
        return S.capture(self.net, rows, self.sites)

    def save(self) -> None:
        s = {"trainers": {n: t.state() for n, t in self.t.items()}, "step": self.step, "cursor": self.cursor,
             "nf": self.nf, "curve": self.curve, "first": self.first}  # fmt: skip
        tmp = self.ckpt.with_suffix(".tmp")
        torch.save(s, tmp)
        tmp.replace(self.ckpt)
        log(f"checkpoint at step {self.step}, row {self.cursor}: {self.ckpt}")

    def one_step(self) -> dict[str, float]:
        x = self.buf.next()
        out = {}
        for n, t in self.t.items():
            j = self.col[n]
            with S.bf16(x):
                out[n] = t.update(self.step, x[:, j] / float(self.nf[j]))
            if len(self.first[n]) < 100:
                self.first[n].append(out[n])
        self.step += 1
        return out


def early_check(r: Run, recent: dict[str, list[float]]) -> dict[str, Any]:
    """Per SAE: mean loss over the last 100 steps below the mean of the first 100 (disjoint: check_step >= 200), dead
    share <= dead_max."""
    mean = lambda v: sum(v) / len(v)
    rec = {n: {"loss_first100": mean(r.first[n]), "loss_last100": mean(recent[n][-100:]), "dead": r.t[n].dead_share()}
           for n in r.names}  # fmt: skip
    miss = [f"{n} loss {v['loss_last100']:.4g} >= {v['loss_first100']:.4g}" for n, v in rec.items() if not v["loss_last100"] < v["loss_first100"]]
    miss += [f"{n} dead {v['dead']:.3f} > {r.cfg.dead_max}" for n, v in rec.items() if v["dead"] > r.cfg.dead_max]
    return {"step": r.step, "per_sae": rec, "miss": miss}


def train(cfg: Config, net: Any, names: list[str], out: Path, large: Path, stop_after: int | None) -> None:
    path = out / f"train_{tag(names)}.json"
    r = Run(cfg, net, names, large / f"ckpt_{tag(names)}.pt")
    if r.step == 0:
        r.nf = S.norm_factors([r.buf.next() for _ in range(cfg.norm_batches)])
        log("norm factors " + json.dumps(dict(zip(r.sites, r.nf.tolist(), strict=True))))
    rec: dict[str, Any] = {"names": names, "steps": cfg.steps, "settings": asdict(cfg)}
    recent: dict[str, list[float]] = {n: [] for n in names}
    t0, s0 = time.time(), r.step
    while r.step < cfg.steps:
        for n, v in r.one_step().items():
            recent[n] = [*recent[n][-99:], v]
        if r.step % cfg.log_every == 0 or r.step == cfg.steps:
            ms = 1000 * (time.time() - t0) / (r.step - s0)
            r.curve.append({"step": r.step, "ms_per_step": ms, "loss": {n: v[-1] for n, v in recent.items()},
                            "dead": {n: t.dead_share() for n, t in r.t.items()}, "row": r.cursor})  # fmt: skip
            log(f"  step {r.step}/{cfg.steps} {ms:.1f} ms/step row {r.cursor} " + " ".join(f"{n} {v[-1]:.4g}/{r.t[n].dead_share():.3f}" for n, v in recent.items()))
            t0, s0 = time.time(), r.step
        if r.step == cfg.check_step:
            ec = early_check(r, recent)
            log(f"[early check] {json.dumps(jsonable(ec))}")
            if ec["miss"]:
                stop(out / f"early_check_{tag(names)}.json", jsonable(ec | {"curve": r.curve}), "; ".join(ec["miss"]), log)
            write_stage(out / f"early_check_{tag(names)}.json", jsonable(ec))
        if r.step % cfg.ckpt_every == 0 or r.step == stop_after:
            r.save()
        if r.step == stop_after:
            log(f"STOP-AFTER {stop_after}: exiting for the resume test")
            hard_exit(3)
    r.save()
    for n, t in r.t.items():
        t.ae.fold(float(r.nf[r.col[n]]))
        torch.save({"state": t.ae.state_dict(), "site": site(n), "norm_factor": float(r.nf[r.col[n]]), "settings": asdict(cfg)}, large / f"sae_{n}.pt")
    write_stage(path, jsonable(rec | {"curve": r.curve, "norm_factors": dict(zip(r.sites, r.nf.tolist(), strict=True)),
                                      "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30}))  # fmt: skip
    log("STAGE train DONE")


def tag(names: list[str]) -> str:
    return "-".join(names) if len(names) <= 2 else f"{names[0]}+{len(names) - 1}"


# ---- reading -----------------------------------------------------------------------------------------------------------


def load_saes(cfg: Config, names: list[str], large: Path) -> dict[str, S.SAE]:
    out = {}
    for n in names:
        ae = S.SAE(cfg.d, cfg.width, cfg.k)
        ae.load_state_dict(torch.load(large / f"sae_{n}.pt", map_location="cpu", weights_only=False)["state"])
        out[n] = ae.cuda().eval()
    return out


def stage0(cfg: Config, base: Any, saes: dict[str, S.SAE], pile: torch.Tensor, out: Path) -> dict[str, Any]:
    path = out / f"stage0_{tag(list(saes))}.json"
    if path.is_file() and "stopped" not in (old := json.loads(path.read_text())):
        return old["base"]
    r = S.read(base, saes, pile, cfg.eval_batch)
    for n, v in r.items():
        log(f"[stage0] {n} " + json.dumps(jsonable(v)))
    rec: dict[str, Any] = {"base": r, "rows": ["val", cfg.eval_start, cfg.eval_rows]}
    seeds = [n for n in saes if site(n) == cfg.second_seed_site]
    if len(seeds) == 2:
        rec["seed_spread"] = {k: abs(r[seeds[0]][k] - r[seeds[1]][k]) for k in ("fve", "ce_increase", "l0")}
    miss = [f"{n} FVE {v['fve']:.3f} < {cfg.fve_min}" for n, v in r.items() if not v["fve"] >= cfg.fve_min]
    miss += [f"{n} CE increase {v['ce_increase']:.3f} > {cfg.ce_increase_max}" for n, v in r.items() if not v["ce_increase"] <= cfg.ce_increase_max]
    if miss:
        stop(path, jsonable(rec), "; ".join(miss), log)
    write_stage(path, jsonable(rec))
    log("STAGE 0 DONE")
    return r


def items(cfg: Config, VPD: Any) -> list[str]:
    known, states = list(VPD.E214_ITEMS[1:]), VPD.e187_states()
    assert len(known) == 6 and len(states) == 68, (len(known), len(states))
    return ["do_nothing", *known[: cfg.n_known], *states[: cfg.n_states]]


def read_item(cfg: Config, W: Any, saes: dict[str, S.SAE], pile: torch.Tensor, base: dict[str, Any], item: str, path: Path) -> None:
    F, M, toks, _at, load = W.make(item)
    toks = toks[: cfg.task_windows]
    rec: dict[str, Any] = {"item": item, "load": load, "task_shape": list(toks.shape),
                           "pile": S.read(F.net, saes, pile, cfg.eval_batch),
                           "task": S.read(F.net, saes, toks, cfg.eval_batch), "task_base": S.read(M.net, saes, toks, cfg.eval_batch)}  # fmt: skip
    rec["fve_drop"] = {n: base[n]["fve"] - v["fve"] for n, v in rec["pile"].items()}
    rec["ce_increase_ratio"] = {n: v["ce_increase"] / base[n]["ce_increase"] if base[n]["ce_increase"] > 0 else float("nan")
                                for n, v in rec["pile"].items()}  # fmt: skip
    rec["task_fve_drop"] = {n: rec["task_base"][n]["fve"] - v["fve"] for n, v in rec["task"].items()}
    log(f"[item {item}] " + json.dumps(jsonable({k: rec[k] for k in ("fve_drop", "ce_increase_ratio", "task_fve_drop")})))
    for part in ("pile", "task", "task_base"):
        for n, v in rec[part].items():
            log(f"  {item} {part} {n} " + json.dumps(jsonable(v)))
    if item == "do_nothing" and (worst := max(abs(v) for v in rec["fve_drop"].values())) > cfg.do_nothing_tol:
        stop(path, jsonable(rec), f"do-nothing FVE differs from base by {worst:.3g} > {cfg.do_nothing_tol}", log)
    write_stage(path, jsonable(rec))
    del F


def evaluate(cfg: Config, names: list[str], out: Path, large: Path) -> None:
    VPD: Any = load_experiment("kstar_vpd", "run")  # World: E214's items and E221's E187 states, as P1.37 builds them
    W = VPD.World()
    saes = load_saes(cfg, names, large)
    pile = cached_rows("val", cfg.eval_start, cfg.eval_rows)
    base = stage0(cfg, W.kl.M.net, saes, pile, out)
    todo = items(cfg, VPD)
    (out / "items").mkdir(parents=True, exist_ok=True)
    for i, item in enumerate(todo):
        path = out / "items" / f"{VPD.KP.fname(item)}_{tag(names)}.json"
        if not path.is_file():
            read_item(cfg, W, saes, pile, base, item, path)
        log(f"ITEM {i + 1} of {len(todo)} DONE ({item})")
    log("STAGE 1 DONE")


def main(size: str, shard: tuple[int, int], stages: list[str], stop_after: int | None) -> int:
    cfg, _ = load_config(size)
    names = cfg.names()[shard[0] :: shard[1]]
    out = RESULTS if size == "scored" else RESULTS / size
    large = RESULTS / "large" / size
    out.mkdir(parents=True, exist_ok=True)
    large.mkdir(parents=True, exist_ok=True)
    log(f"E234 start, size {size}, shard {shard}, SAEs {names}, stages {stages}; settings {asdict(cfg)}; steps {cfg.steps}")
    t0 = time.time()
    if "train" in stages and not all((large / f"sae_{n}.pt").is_file() for n in names):
        from commute.steering import Net

        net = Net("cuda")
        net.net.requires_grad_(False)
        train(cfg, net.net, names, out, large, stop_after)
        del net
        torch.cuda.empty_cache()
    if "eval" in stages:
        evaluate(cfg, names, out, large)
    log(f"FINISHED {stages} {(time.time() - t0) / 60:.1f} min, peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=("scored", "rehearse", "dry"), default="scored")
    ap.add_argument("--shard", nargs=2, type=int, default=(0, 1), metavar=("I", "N"))
    ap.add_argument("--stages", nargs="+", choices=("train", "eval"), default=["train", "eval"])
    ap.add_argument("--stop-after", type=int, default=None)
    args = ap.parse_args()
    require_filed(PRED)
    code = 1
    try:
        code = main(args.size, (args.shard[0], args.shard[1]), args.stages, args.stop_after)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)
