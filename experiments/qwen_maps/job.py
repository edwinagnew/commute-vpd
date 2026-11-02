"""E44 (qwen_maps): the config, and the job that holds the model, decomposition, rows, δW and S's files for run.py's
stages. Imported by run.py as a sibling module (as experiments/qwen_sft/amend1.py imports its run.py).
"""

from __future__ import annotations

import json
import resource
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor

from commute import qwen_maps
from commute.qwen_sft import DISK_MARGIN, free_bytes, linears, load
from commute.qwen_vpd import (
    SITES,
    fp32_matmuls,
    load_components,
    load_gamma,
    load_target,
    parse_site,
)

HERE = Path(__file__).resolve().parent
RESULTS, PRED = HERE / "results", HERE / "results" / "predictions.json"
E40_RESULTS = HERE.parent / "qwen_sft" / "results"
MATHS, WEB = E40_RESULTS / "maths_rows.json", HERE.parent / "qwen_adversary" / "results" / "tokens.json"
STAGES, DEV, F32, F64 = ("R", "C", "S", "D", "K", "M", "G", "A", "H"), "cuda", torch.float32, torch.float64
SPLITS, CONTROLS, HALVES = ("ls", "inspan", "adjoint"), ("lower", "upper", "random"), ("lower", "upper")
WEB_SOURCES = ("G_fineweb_seed0", "W0_fineweb_seed1", "W0_fineweb_seed2")
MATHS_SOURCE = "G_maths_seed0"
FT_WEIGHTS = ("do_nothing", "ls", "lower", "upper")  # states whose weights are W0 + δW


@dataclass(frozen=True)
class Config:
    n_rows: int = 128  # rows per block (maths, FineWeb)
    row_len: int = 512
    eval_batch: int = 2  # ~60-65 GB peak on an H100 (qwen_maps.read_rows); halved on a first-batch peak or OOM
    steps: tuple[int, ...] = (1, 10)
    step_h: int = 100
    layers: tuple[int, ...] = tuple(range(28))  # δW on these layers only (the rehearsal's few)
    random_dw: float = 0.0  # > 0: δW_t = random_dw·√t·std(W0)·N(0, 1) per site (the rehearsal); 0: E40's files
    comps_dtype: str = "float32"
    basis: str = "gram"  # or "svd" if R's check fails
    tol: float = 1e-6
    alive_total: int = 494598
    stoch_seeds: tuple[int, ...] = (0, 1, 2)
    adv_steps: int = 20
    acc_problems: int = 256
    gen_batch: int = 128
    r_layer: int = 14  # R's Gram-against-SVD layer
    s_limit_min: float = 50.0
    mem_halve_gb: float = 70.0
    spend_limit_usd: float = 9.5  # A and H: spend + their minutes over this -> skipped
    stop_usd: float = 10.0
    stage_min: tuple[tuple[str, float], ...] = (("A", 22.0), ("H", 20.0))
    rehearse: bool = False  # tiny sizes; stops are obeyed unless --waive names them
    results: str = ""  # subdirectory of results/

    def __post_init__(self) -> None:
        assert self.n_rows % self.eval_batch == 0 and self.row_len <= 512 and self.n_rows <= 128
        assert self.steps == (1, 10) and self.stoch_seeds[0] == 0 and self.basis in ("gram", "svd")
        assert all(0 <= x < 28 for x in self.layers) and self.r_layer < 28
        assert self.rehearse == bool(self.results) == (self.random_dw > 0)


CFG = Config()
REHEARSAL = Config(n_rows=2, row_len=64, eval_batch=1, layers=(0,), random_dw=0.02, comps_dtype="bfloat16",
                   adv_steps=2, acc_problems=4, gen_batch=4, rehearse=True, results="rehearsal")


@dataclass(frozen=True)
class JobOpts:
    """The options a Job reads (run.py builds it from its arguments; a side script builds it directly)."""

    out: str  # big files (S's products, arrays, accuracy answers)
    e40_large: str
    decomp: str
    results: str = str(RESULTS)  # every JSON and the log (a rehearsal passes its own directory)
    e40: str = ""
    waive: str = ""
    prior_usd: float = 0.0
    usd_per_h: float = 0.0


class Job:
    """The fp32 target (its matrices are W0), loaded on first use with the decomposition; rows, sources, δW and the
    alive columns (host) likewise. `res` holds every stage's results file."""

    def __init__(self, a: JobOpts, cfg: Config, scorer: Callable[[dict, dict], dict]):
        self.a, self.cfg, self.scorer, self.t0, self.stage = a, cfg, scorer, time.time(), ""
        self.dir, self.out, self.e40, self.large = Path(a.results) / cfg.results, Path(a.out), Path(a.e40), Path(a.e40_large)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.pred = json.loads(PRED.read_text())
        self.res = {s: json.loads(p.read_text()) for s in STAGES if (p := self.dir / f"{s}.json").is_file()}
        self.waive: set[str] = set(a.waive.split(",")) - {""}
        assert not self.waive or cfg.rehearse, "--waive is for the rehearsal only"
        state = json.loads(f.read_text()) if (f := self.dir / "state.json").is_file() else {}
        self.batch = state.get("eval_batch", cfg.eval_batch)  # a halving persists: a resume never reuses a failed size
        self.spent_before = state.get("usd", a.prior_usd)  # earlier jobs' spend (their wall time × rate)
        self.sites = [s for s in SITES if parse_site(s)[0] in cfg.layers]
        self._model: Any = None
        self._decomp: Any = None
        self._alive: Any = None
        self._dw: dict[tuple[int, bool], dict[str, Tensor]] = {}
        self._ft: tuple[Any, dict[str, Tensor]] | None = None
        self._sigma: dict[str, Tensor] | None = None
        fp32_matmuls()

    def log(self, msg: str) -> None:
        line = f"[{time.time() - self.t0:8.1f}s] {msg}"
        print(line, flush=True)
        with (self.dir / "run.log").open("a") as f:
            f.write(line + "\n")

    def put(self, stage: str, d: dict[str, Any]) -> None:
        self.res[stage] = d
        write_json(self.dir / f"{stage}.json", d)
        self.save_state()
        self.log(f"[{stage}] {json.dumps(brief(d), default=float)}")
        sc = self.scorer(self.res, self.pred)
        write_json(self.dir / "score.json", sc)
        self.log(f"[score] {json.dumps(sc, default=float)}")

    def stop(self, name: str, why: str) -> None:
        """A stop rule: <dir>/stopped.json, then exit; the next job refuses to start while it exists. A rehearsal
        that names the rule in --waive logs it and goes on."""
        if name in self.waive:
            self.log(f"WAIVED stop {name}: {why}")
            return
        write_json(self.dir / "stopped.json", {"name": name, "why": why, "stage": self.stage,
                                               "at_s": time.time() - self.t0})
        self.save_state()
        self.log(f"STOP {name}: {why}")
        sys.exit(f"STOP {name}: {why}")

    def spent(self) -> float:
        return self.spent_before + (time.time() - self.t0) / 3600 * self.a.usd_per_h

    def save_state(self) -> None:
        """Cumulative spend and the eval batch, read back by the next job of the run. Stage R spends nothing and
        writes none, so a pod run after it still honours --prior-usd."""
        if self.stage == "R":
            return
        write_json(self.dir / "state.json", {"usd": self.spent(), "eval_batch": self.batch})

    def set_batch(self, b: int, why: str) -> None:
        self.batch = b
        self.save_state()
        self.log(f"[batch] eval batch now {b}: {why}")

    def peaks(self) -> str:
        gpu = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        return f"peak GPU {gpu:.1f} GB, host {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB"

    @property
    def model(self) -> Any:
        if self._model is None:
            self._model = load_target(DEV, F32)
            self.log(f"[load] target, {torch.cuda.get_device_name()}, {self.peaks()}")
        return self._model

    def comps(self) -> Any:
        if self._decomp is None:
            self._decomp = [load_components(Path(self.a.decomp), DEV, getattr(torch, self.cfg.comps_dtype)), None]
        return self._decomp[0]

    def gamma(self) -> Any:
        self.comps()
        if self._decomp[1] is None:
            self._decomp[1] = load_gamma(Path(self.a.decomp), DEV, torch.bfloat16)
            self.log(f"[load] decomposition, {self.peaks()}")
        return self._decomp[1]

    def drop_gamma(self) -> None:
        """Γ's 17.1 GB of GPU freed (E44 Amendment 2's stage S on an L40); `gamma()` reloads it."""
        if self._decomp is not None:
            self._decomp[1] = None
        torch.cuda.empty_cache()

    def gstats(self, block: str) -> dict[str, Tensor]:
        return load(self.large / "gstats" / f"W0_{block}.safetensors")

    @property
    def alive(self) -> qwen_maps.Alive:
        if self._alive is None:
            idx = qwen_maps.alive_from_stats(self.gstats("maths"), self.gstats("fineweb"))
            self._alive = qwen_maps.Alive(self.comps(), idx)
            if self._alive.count() != self.cfg.alive_total:
                self.stop("alive", f"alive set {self._alive.count()} != {self.cfg.alive_total}")
        return self._alive

    def rows(self, block: str) -> Tensor:
        rows = json.loads((MATHS if block == "maths" else WEB).read_text())["rows"]
        return torch.tensor([r[: self.cfg.row_len] for r in rows[: self.cfg.n_rows]])

    def sources(self, block: str) -> dict[str, dict[str, Tensor]]:
        names = WEB_SOURCES if block == "web" else (MATHS_SOURCE,)
        return {n: load(self.large / "sources" / f"{n}.safetensors", DEV) for n in names}

    def dw(self, t: int, fp32: bool = False) -> dict[str, Tensor]:
        """δW_t (host) for the sites of the config's layers, kept until `forget` (S alternates steps per site)."""
        key = (t, fp32)
        if key not in self._dw:
            if self.cfg.random_dw > 0:
                W0 = {s: lin.weight.data for s, lin in linears(self.model).items()}
                gen = torch.Generator().manual_seed(t)
                d = {s: (self.cfg.random_dw * t**0.5 * float(W0[s].std()) * torch.randn(W0[s].shape, generator=gen))
                     for s in self.sites}
                self._dw[key] = d if fp32 else {s: v.bfloat16() for s, v in d.items()}
            else:
                name = f"step_{t}_fp32.safetensors" if fp32 else f"step_{t}.safetensors"
                full = load(self.e40 / "dW" / name)
                self._dw[key] = {s: full[s] for s in self.sites}
        return self._dw[key]

    def ft(self, t: int, fp32: bool = False) -> dict[str, Tensor]:
        """The fine-tuned weights W0 + δW_t on the GPU (fp32), one step held at a time."""
        if self._ft is None or self._ft[0] != (t, fp32):
            self._ft = None
            torch.cuda.empty_cache()
            dW = self.dw(t, fp32)
            W0 = {s: lin.weight.data for s, lin in linears(self.model).items()}
            self._ft = ((t, fp32), {s: W0[s] + dW[s].to(DEV, F32) if s in dW else W0[s] for s in SITES})
        return self._ft[1]

    def forget(self) -> None:
        """Drop the held δW files (host) and fine-tuned weights (GPU)."""
        self._dw.clear()
        self._ft = None
        torch.cuda.empty_cache()

    def drop_sigma(self) -> None:
        self._sigma = None

    def sigma(self) -> dict[str, Tensor]:
        if self._sigma is None:
            batches = list(self.rows("maths").to(DEV).split(self.batch))
            self._sigma = qwen_maps.input_moments(self.model, batches, self.sites)
            self.log(f"[sigma] {len(self._sigma)} sites, {self.peaks()}")
        return self._sigma

    def site_dir(self, t: int) -> Path:
        return self.out / "S" / f"step{t}"

    def change(self, name: str, t: int | None, fp32: bool = False, scale: float = 1.0) -> Any:
        """site -> SiteChange of state `name` at step t, built from δW and S's files on the GPU in float64
        (`qwen_maps.applied` casts it to the forward's fp32). `scale` multiplies control (ii)'s Σ-matched pieces
        (E44 Amendment 2's KL match; 1 = E44's random state)."""
        dW = {} if t is None else self.dw(t, fp32)

        def fn(s: str) -> qwen_maps.SiteChange:
            if name == "W0" or s not in dW:
                return qwen_maps.NONE
            U_A, V_A = self.alive.uv(s, DEV, F64)
            M = dW[s].to(DEV, F64)
            if name == "do_nothing":
                return qwen_maps.site_change(name, M, U_A, V_A, None, 0.0, 0.0)
            path, rec = self.site_dir(t or 0) / f"{s}.safetensors", self.site_rec(t or 0, s)
            RU = RV = None
            if name == "random":
                RU, RV = (x.double() for x in qwen_maps.random_pieces(tuple(U_A.shape), tuple(V_A.shape), 0, s, DEV))
            with safe_open(str(path), framework="pt") as f:
                return qwen_maps.site_change(name, M, U_A, V_A, f.get_tensor, rec["c"], rec["a"] * scale, RU, RV)

        return fn

    def site_rec(self, t: int, s: str) -> dict[str, Any]:
        return json.loads((self.site_dir(t) / f"{s}.json").read_text())

    def read(self, name: str, t: int | None, block: str, seeds: tuple[int, ...], ft: dict[str, Tensor] | None,
             sources: bool = True, fp32: bool = False) -> dict[str, float]:
        """One state's row loop on one block. States whose weights are W0 + δW use the fine-tuned weights as they
        stand (no second copy). Memory rule, every read: a first-batch peak over mem_halve_gb halves the batch for
        the reads after it (an out-of-memory error is retried at half size by run.py)."""
        weights = self.ft(t, fp32) if name in FT_WEIGHTS and t is not None else None
        torch.cuda.reset_peak_memory_stats()

        def memory_rule(i: int) -> None:
            peak = torch.cuda.max_memory_allocated() / 1e9
            if i == 0:
                self.log(f"[memory] {name} {block}: first batch of {self.batch}, peak {peak:.1f} GB")
                if peak > self.cfg.mem_halve_gb and self.batch > 1:
                    self.set_batch(self.batch // 2, f"first-batch peak {peak:.1f} GB > {self.cfg.mem_halve_gb}")

        with qwen_maps.applied(self.model, self.comps(), self.alive, self.change(name, t, fp32), weights):
            return qwen_maps.read_rows(self.model, self.rows(block), self.batch, self.gamma(), self.comps(), seeds,
                                       self.sources(block) if sources else {}, ft, memory_rule)


def brief(d: Any) -> Any:
    """A results dict for the log: the per-site records and fp32 ranks (unscored, in the JSON) left out at any depth."""
    if not isinstance(d, dict):
        return d
    return {k: brief(v) for k, v in d.items() if k not in ("per_site", "ranks_fp32")}


def write_json(path: Path, obj: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    tmp.replace(path)


def write_tensors(path: Path, tensors: dict[str, Tensor]) -> None:
    """A safetensors file of mixed dtypes, written whole then renamed; raises if the disk lacks room plus margin."""
    path.parent.mkdir(parents=True, exist_ok=True)
    need = sum(t.numel() * t.element_size() for t in tensors.values())
    if free_bytes(path.parent) < need + DISK_MARGIN:
        raise RuntimeError(f"{path}: {need / 1e9:.2f} GB does not fit")
    tmp = path.with_name(path.name + ".tmp")
    save_file(tensors, str(tmp))
    tmp.replace(path)

