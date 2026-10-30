"""E57 (qwen_reading): the config, the states, and the GPU job every stage shares: the fp32 target, the fp32
components, Γ, the live set, weight sets pinned on the host and copied into the model in place (an L40 holds no second
set beside the decomposition, E44 Amendment 2), the cell loop, Σ and the least-squares split. Imported by run.py as a
sibling module. Plan: reports/plans/2026-10-30_plan_q3-reading-the-update.md.
"""

from __future__ import annotations

import json
import resource
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import qwen_maps, split
from commute.qwen_reading import Counts, Layout
from commute.qwen_sft import linears, load, sha256
from commute.qwen_teacher import move_gamma
from commute.qwen_vpd import (
    SITES,
    fp32_matmuls,
    importances,
    load_components,
    load_gamma,
    load_target,
    masked_logits,
    parse_site,
)
from commute.vpd import kl_per_position

HERE = Path(__file__).resolve().parent
RESULTS, PRED = HERE / "results", HERE / "results" / "predictions.json"
LARGE = RESULTS / "large"  # a link into ~/commute-data/results-large/qwen_reading (cluster NFS)
EXP = HERE.parent
MATHS, WEB = EXP / "qwen_sft/results/maths_rows.json", EXP / "qwen_adversary/results/tokens.json"
GSTATS = EXP / "qwen_sft/results/large/nowarmup/gstats"
E45 = EXP / "qwen_updates/results"
REGEN_SUMS, A2 = EXP / "qwen_maps/results/regen_SHA256SUMS", EXP / "qwen_maps/results/amend2"
DW_SUMS = RESULTS / "dW_SHA256SUMS"  # written by Stage A for E45's files (E45 saved no sums)
DEV, F32, F64, GIB = "cuda", torch.float32, torch.float64, 2**30


@dataclass(frozen=True)
class Config:
    n_rows: int = 128  # maths rows, cut to row_len (E44 Amendment 2)
    row_len: int = 256
    n16: int = 16
    own_len: int = 512  # E45's own rows, read whole only for the KL check against its E record
    sigma_len: int = 512  # Σ as E44's stage S: 128 maths rows of 512, batch 2
    sigma_batch: int = 2
    tol: float = 1e-6
    alive_total: int = 494598
    store_err_max: float = 1e-3  # E44's STORE_ERR_MAX on the fp32 X
    gpu_max_gib: float = 44.0
    id_tol: float = 1e-6
    rep_tol: float = 1e-3
    boot: int = 2000
    draws: int = 1000
    null_seeds: tuple[int, ...] = (0, 1, 2, 3)
    kl_match: float = 1.5
    kl_evals: int = 6
    inspan_a: float = 17.78279410038923  # E44 Amendment 2's matched scale (t10.json search.a)
    high: float = 0.8
    problems: int = 128  # GSM8K test 0-127
    think_new: int = 2048
    think_batch: int = 64
    drop_max: float = 0.25
    limit_s: float = 3 * 3600  # the sbatch --time of C0 and C
    first_s: float = 2 * 3600

    def __post_init__(self) -> None:
        assert self.n16 <= self.n_rows and self.row_len <= self.sigma_len == 512 and self.n_rows % 2 == 0
        assert self.problems % self.think_batch == 0 and self.n_rows % self.sigma_batch == 0


CFG = Config()
# Reference values (E44 Amendment 2: controls.json, t10.json), filed in predictions.json "early"
REF = {"K0_16": 0.06558158050898716, "K00_16": 1.4956748775221875, "K00": 1.3738106470539775,
       "K0": 0.06961319516372436, "kl_e40_16": 0.618376722326358, "kl_inspan_16": 0.5023351372461583}


@dataclass(frozen=True)
class State:
    path: str  # δW file (bf16 safetensors) under experiments/, or "e40" (the --e40 directory)
    own: tuple[str, int] | None = None  # E45's own rows (arm, seed); None: the maths rows are its own rows
    e_record: str = ""  # E45's E_<run>.json, whose states.100.kl.own the own-row KL must match


STATES = {"e40": State("e40"),
          "gold_s0": State("qwen_updates/results/large/dW/gold_s0/step_100.safetensors", ("gold", 0), "E_gold_s0"),
          "a2long_s0": State("qwen_updates/results/large/dW/a2long_s0/step_100.safetensors", ("a2", 0),
                             "E_a2long_s0"),
          "a2long_s1": State("qwen_updates/results/large/dW/a2long_s1/step_100.safetensors", ("a2", 1),
                             "E_a2long_s1")}


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    tmp.replace(path)


def kl(p: Tensor, q: Tensor) -> float:
    return float(kl_per_position(p, q).double().sum())


def pinned(w: Tensor) -> Tensor:
    return torch.empty(w.shape, dtype=w.dtype, pin_memory=True).copy_(w)


class Lab:
    """One stage's job. Refuses to start without predictions.json or while results/stopped.json exists."""

    def __init__(self, stage: str, e40: str, decomp: str):
        if not PRED.is_file() or "status" in json.loads(PRED.read_text()):
            sys.exit(f"refusing to start: {PRED} missing or has a 'status' key")
        if (RESULTS / "stopped.json").is_file():
            sys.exit(f"refusing to start: {RESULTS / 'stopped.json'} exists (a stop rule fired; the user decides)")
        self.stage, self.e40, self.decomp, self.t0, self.cfg = stage, Path(e40), Path(decomp), time.time(), CFG
        self._model: Any = None
        self._comps: Any = None
        self._gamma: Any = None
        self._alive: Any = None
        self._lay: Layout | None = None
        self._sigma: dict[str, Tensor] | None = None
        self.sets: dict[str, dict[str, Tensor]] = {}
        self.now = "W0"
        fp32_matmuls()
        self.log(f"[{stage}] start; config {CFG}")

    # ------------------------------------------------------------------------- log, files, stops

    def log(self, msg: str) -> None:
        line = f"[{time.time() - self.t0:8.1f}s] {msg}"
        print(line, flush=True)
        with (RESULTS / "run.log").open("a") as f:
            f.write(line + "\n")

    def get(self, name: str) -> dict[str, Any]:
        p = RESULTS / f"{name}.json"
        return json.loads(p.read_text()) if p.is_file() else {}

    def put(self, name: str, d: dict[str, Any]) -> None:
        write_json(RESULTS / f"{name}.json", d)
        self.log(f"[{name}] {json.dumps(d, default=float)}")

    def stop(self, name: str, why: str) -> None:
        write_json(RESULTS / "stopped.json", {"name": name, "why": why, "stage": self.stage,
                                              "at_s": time.time() - self.t0})
        self.log(f"STOP {name}: {why}")
        sys.exit(f"STOP {name}: {why}")

    def check(self, name: str, holds: bool, detail: dict[str, Any]) -> dict[str, Any]:
        self.log(f"[check] {name}: {'ok' if holds else 'MISSED'} {json.dumps(detail, default=float)}")
        if not holds:
            self.stop(name, json.dumps(detail, default=float))
        return detail | {"holds": holds}

    def wall(self, longest: float) -> None:
        """Checked before each state: exit 0 with what is written (resubmit to resume) when the time left under the
        job limit is less than the longest state so far, or, before one is measured, past first_s of wall time."""
        used = time.time() - self.t0
        if (longest > 0 and CFG.limit_s - used < longest) or (longest == 0 and used > CFG.first_s):
            self.log(f"cut short: {used:.0f} s used, longest state {longest:.0f} s, limit {CFG.limit_s:.0f} s; "
                     "resubmit to resume")
            sys.exit(0)

    def peaks(self) -> str:
        gpu = torch.cuda.max_memory_allocated() / GIB if torch.cuda.is_available() else 0.0
        return f"peak GPU {gpu:.2f} GiB, host {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB"

    def gpu_rule(self, where: str) -> None:
        peak = torch.cuda.max_memory_allocated() / GIB
        if peak > CFG.gpu_max_gib:
            self.stop("memory", f"{where}: GPU peak {peak:.2f} GiB after row 0 > {CFG.gpu_max_gib}")

    # ------------------------------------------------------------------------- the model, the decomposition

    @property
    def model(self) -> Any:
        if self._model is None:
            self._model = load_target(DEV, F32)
            self.sets["W0"] = {s: pinned(lin.weight.data) for s, lin in linears(self._model).items()}
            self.log(f"[load] target, {torch.cuda.get_device_name()}, {self.peaks()}")
        return self._model

    @property
    def comps(self) -> Any:
        if self._comps is None:
            self._comps = load_components(self.decomp, DEV, F32)
            self.log(f"[load] components, {self.peaks()}")
        return self._comps

    @property
    def gamma(self) -> Any:
        if self._gamma is None:
            self._gamma = load_gamma(self.decomp, DEV, torch.bfloat16)
            self.log(f"[load] gamma, {self.peaks()}")
        elif self._gamma.in_w.device.type == "cpu":
            self._gamma = move_gamma(self._gamma, DEV)
        return self._gamma

    def park_gamma(self) -> None:
        """Γ's 17.1 GB to the host (Σ and the split need the room); `gamma` brings it back."""
        if self._gamma is not None:
            self._gamma = move_gamma(self._gamma, "cpu")
        torch.cuda.empty_cache()

    @property
    def alive(self) -> qwen_maps.Alive:
        if self._alive is None:
            st = [load(GSTATS / f"W0_{b}.safetensors") for b in ("maths", "fineweb")]
            self._alive = qwen_maps.Alive(self.comps, qwen_maps.alive_from_stats(*st))
            if self._alive.count() != CFG.alive_total:
                self.stop("alive", f"live set {self._alive.count()} != {CFG.alive_total}")
            self._lay = Layout({s: v.cpu() for s, v in self._alive.idx.items()})
        return self._alive

    @property
    def lay(self) -> Layout:
        if self._lay is None:
            assert self.alive is not None
        assert self._lay is not None
        return self._lay

    def counts(self, n_rows: int = 0) -> Counts:
        return Counts(self.lay, DEV, n_rows)

    def maths(self, n: int | None = None, length: int | None = None) -> Tensor:
        rows = json.loads(MATHS.read_text())["rows"][: n or CFG.n_rows]
        return torch.tensor([r[: length or CFG.row_len] for r in rows])

    # ------------------------------------------------------------------------- weight sets

    def use(self, key: str) -> None:
        if self.sets[key] is not self.sets[self.now]:
            for s, lin in linears(self.model).items():
                lin.weight.data.copy_(self.sets[key][s])
            torch.cuda.synchronize()
        self.now = key

    def set_weights(self, key: str, dW: dict[str, Tensor] | None, scale: float = 1.0) -> None:
        """Weight set `key` = W0 + scale·δW (fp32, added on the GPU as E44's Reader.set_ft); None: a separate pinned
        copy of W0, so the W0 control goes through the same copy path."""
        self.drop_set(key)
        self.use("W0")
        if dW is None:
            self.sets[key] = {s: pinned(w) for s, w in self.sets["W0"].items()}
            return
        self.sets[key] = {s: pinned(lin.weight.data + scale * dW[s].to(DEV, F32))
                          for s, lin in linears(self.model).items()}

    def drop_set(self, key: str) -> None:
        """Free weight set `key` (never W0) and return its pinned pages to the system."""
        assert key != "W0"
        if self.now == key:
            self.use("W0")
        self.sets.pop(key, None)
        empty = getattr(torch._C, "_host_emptyCache", None)  # the pinned-memory cache keeps freed blocks otherwise
        if empty is not None:
            empty()

    def load_dw(self, name: str) -> dict[str, Tensor]:
        """A state's δW (host, as saved) after its sha256 check: E40's against regen_SHA256SUMS, E45's against the sums
        Stage A recorded."""
        p = self.dw_path(name)
        sums = REGEN_SUMS if name == "e40" else DW_SUMS
        want = {ln.split()[1].lstrip("*"): ln.split()[0] for ln in sums.read_text().splitlines() if ln}
        key = p.name if name == "e40" else name
        got = sha256(p)
        self.check(f"sha256_{name}", got == want.get(key), {"file": str(p), "sha256": got, "want": want.get(key)})
        return load(p)

    def dw_path(self, name: str) -> Path:
        st = STATES[name]
        return self.e40 / "dW" / "step_10.safetensors" if st.path == "e40" else EXP / st.path

    def kl_rows(self, key: str, rows: Tensor) -> float:
        """Mean KL(W0 ‖ weight set `key`) per position over rows (unmasked)."""
        total = 0.0
        for r in rows:
            t = r[None].to(DEV)
            self.use("W0")
            p = self.model(t, use_cache=False).logits
            self.use(key)
            total += kl(p, self.model(t, use_cache=False).logits)
            del p
        self.use("W0")
        return total / rows.numel()

    # ------------------------------------------------------------------------- the cells

    def cells(self, rows: Tensor, change: Any = None, c0: Counts | None = None, c1: Counts | None = None,
              tag: str = "") -> dict[str, list[float]]:
        """Per-row KL sums of one pass over rows, weights W0 and "FT". Original subcomponents (change None): K0, K00
        (g0, masks from W0's forward), K01 (g1, FT's forward); c0, c1 count g0, g1. A split `change` applied to the
        live columns: K10 (g0), K11 (g1). Masked forwards are Δ off, so the weights do not enter them; FT's own
        logits are its clean forward's. One weight copy per row (the order alternates)."""
        model, comps, gamma, alive = self.model, self.comps, self.gamma, self.alive
        keys = ("K10", "K11") if change is not None else ("K00", "K01")
        out: dict[str, list[float]] = {k: [] for k in (keys if change is not None else ("K0", *keys))}
        keep = {s: lin.weight.data for s, lin in linears(model).items()}
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        ctx = qwen_maps.applied(model, comps, alive, change, keep) if change is not None else nullcontext()
        with ctx:
            for i, r in enumerate(rows):
                t, ci = r[None].to(DEV), {}
                for key in (("FT", "W0") if self.now == "FT" else ("W0", "FT")):
                    self.use(key)
                    logits, g = importances(model, t, gamma, comps)
                    if (c := c0 if key == "W0" else c1) is not None:
                        c.add_row(g, row=i)
                    ci[key] = masked_logits(model, t, comps, g)
                    del g
                    if key == "W0" and change is None:
                        out["K0"].append(kl(logits, ci["W0"]))
                    if key == "FT":
                        ci["ref"] = logits
                    del logits
                out[keys[0]].append(kl(ci["ref"], ci["W0"]))
                out[keys[1]].append(kl(ci["ref"], ci["FT"]))
                del ci
                if i == 0:
                    self.gpu_rule(f"cells {tag}")
        self.use("W0")
        means = {k: sum(v) / (len(v) * rows.shape[1]) for k, v in out.items()}
        self.log(f"[cells {tag}] " + " ".join(f"{k} {v:.9g}" for k, v in means.items())
                 + f"  rows {len(rows)}, {time.time() - t0:.0f} s, {self.peaks()}")
        return out

    # ------------------------------------------------------------------------- Σ and the split

    def sigma(self) -> dict[str, Tensor]:
        """Σ = E[a aᵀ] (float64, host) at W0 on 128 maths rows of 512, as E44's stage S (Γ parked: ~11 GB on the GPU
        while it accumulates)."""
        if self._sigma is None:
            self.park_gamma()
            self.use("W0")
            batches = list(self.maths(length=CFG.sigma_len).to(DEV).split(CFG.sigma_batch))
            t0 = time.time()
            self._sigma = qwen_maps.input_moments(self.model, batches, list(SITES))
            self.log(f"[sigma] {len(self._sigma)} sites in {time.time() - t0:.0f} s, {self.peaks()}")
        return self._sigma

    def split(self, dW: dict[str, Tensor], nulls: bool, halves: bool) -> dict[str, Any]:
        """The least-squares split of δW, site by site on the GPU in float64 (Γ parked): X = K⁺δW kept fp32 on the
        host (E44's stored precision; stored_error over store_err_max stops), and the Σ-weighted piece energy per live
        subcomponent of the split, (nulls) of the split of `split.random_like(δW, Σ, null_seed(seed, site))` for each
        null seed, and (halves) of the split onto E44's often-on half alone. Bases are Gram (E44's R check) and not kept."""
        self.park_gamma()
        sig, alive, t0 = self.sigma(), self.alive, time.time()
        st = load(GSTATS / "W0_maths.safetensors") if halves else {}
        X, pe = {}, {"ls": [], "half": []} | {f"null{k}": [] for k in (CFG.null_seeds if nulls else ())}
        err = 0.0
        for s in SITES:
            U_A, V_A = alive.uv(s, DEV, F64)
            S, M = sig[s].to(DEV), dW[s].to(DEV, F64)
            b = split.basis_from(qwen_maps.make_basis(U_A, "gram"), qwen_maps.make_basis(V_A, "gram"), CFG.tol)
            Xs = split.least_squares_x(M, b)
            err = max(err, qwen_maps.stored_error(Xs, U_A, V_A, S, split.weighted_energy(M, S)))
            X[s] = pinned(Xs.float().cpu())
            pe["ls"].append(split.piece_energies(*qwen_maps.pieces(Xs.float(), U_A, V_A), U_A, V_A, S).cpu())
            for k in CFG.null_seeds if nulls else ():
                XR = split.least_squares_x(split.random_like(M, S, null_seed(k, s)), b)
                pe[f"null{k}"].append(split.piece_energies(XR @ V_A, XR.T @ U_A, U_A, V_A, S).cpu())
            if halves:
                idx = alive.idx[s].cpu()
                pos = qwen_maps.halves(st[f"{s}.active"][idx], st[f"{s}.mean_g"][idx])[1].to(DEV)
                Uh, Vh = U_A[:, pos], V_A[:, pos]
                bh = split.basis_from(qwen_maps.make_basis(Uh, "gram"), qwen_maps.make_basis(Vh, "gram"), CFG.tol)
                Xh = split.least_squares_x(M, bh)
                full = torch.zeros(U_A.shape[1], dtype=F64, device=DEV)
                full[pos] = split.piece_energies(Xh @ Vh, Xh.T @ Uh, Uh, Vh, S)
                pe["half"].append(full.cpu())
            del U_A, V_A, S, M, b, Xs
            if parse_site(s)[1] == "down":
                self.log(f"[split] layer {parse_site(s)[0]}: {time.time() - t0:.0f} s, {self.peaks()}")
        torch.cuda.empty_cache()
        if err > CFG.store_err_max:
            self.stop("stored_error", f"fp32 X misses float64 by {err:.2e} of ‖δW‖_Σ")
        return {"X": X, "pe": {k: torch.cat(v) for k, v in pe.items() if v}, "stored_error": err,
                "s": time.time() - t0}

    def change(self, dW: dict[str, Tensor] | None, X: dict[str, Tensor] | str | None) -> Any:
        """site -> SiteChange of the least-squares split state with stored X (X None: the do-nothing state, no column
        moves; "zero": X = 0, W0's control through the same path; dW None: δW = 0)."""
        alive = self.alive

        def fn(s: str) -> qwen_maps.SiteChange:
            U_A, V_A = alive.uv(s, DEV, F64)
            M = torch.zeros(U_A.shape[0], V_A.shape[0], dtype=F64, device=DEV) if dW is None else dW[s].to(DEV, F64)
            if X is None:
                return qwen_maps.site_change("do_nothing", M, U_A, V_A, None, 0.0, 0.0)
            Xs = torch.zeros(M.shape, dtype=F32) if isinstance(X, str) else X[s]
            return qwen_maps.site_change("ls", M, U_A, V_A, lambda _k: Xs, 0.0, 0.0)

        return fn


def null_seed(seed: int, site: str) -> int:
    """The Gaussian null's seed per site (amended 2026-10-30 before any run, per code review: one seed for every site
    gave sites of one shape the same Gaussian matrix; E44's stage S used seed 0 everywhere)."""
    return seed * 1000 + SITES.index(site)
