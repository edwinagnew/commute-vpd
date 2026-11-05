"""What the D3 rows (E94, reports/plans/2026-11-05_plan_e94-d3-rows.md) and D1's first table (E93,
reports/plans/2026-11-05_plan_e93-d1-table.md) read on saved Qwen3-1.7B states: E91's states, the row sets, per-row sums
by position class, the least-squares split alone, the weight sets on the host, and the pass whose masks read W0's V.

Position classes. Every position of a row is in one class: an answer position whose row holds its whole question
("whole"), one whose row opens inside its prompt ("cut"), one whose row opens inside the answer ("fragment"), or any other
position ("other": prompts, context, maths rows). A per-row record holds the four class sums and the pooled sum computed
apart (`by_class`), so a reader can check that the classes add to the pooled number. Sums are fp64 of fp32 per position.
"""

from __future__ import annotations

import json
import resource
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from commute import qwen_comparators as qc
from commute import qwen_maps, split
from commute import qwen_reading as qr
from commute.jobs import EXPERIMENTS
from commute.qwen_sft import linears, load
from commute.qwen_teacher import move_gamma
from commute.qwen_vpd import (
    SITES,
    Components,
    fp32_matmuls,
    importances,
    load_components,
    load_gamma,
    load_target,
    masked_logits,
)
from commute.vpd import kl_per_position

GRID = EXPERIMENTS / "qwen_size_grid" / "results" / "size_grid.json"
MATHS = EXPERIMENTS / "qwen_sft" / "results" / "maths_rows.json"
GSTATS = EXPERIMENTS / "qwen_sft" / "results" / "large" / "nowarmup" / "gstats"
A1_ITEMS = EXPERIMENTS / "qwen_updates" / "results" / "large" / "P" / "a1_rows.json"
FORMS = EXPERIMENTS / "qwen_tofu_forms" / "results" / "forms.jsonl"
E57_LARGE = EXPERIMENTS / "qwen_reading" / "results" / "large"
DEV, F32, F64, GIB = "cuda", torch.float32, torch.float64, 2**30
ROW_LEN, MATHS_ROWS, SIGMA_LEN, SIGMA_BATCH = 256, 128, 512, 2  # E57's maths rows; Σ as E44's stage S
ALIVE_TOTAL, BASIS_TOL = 494598, 1e-6
CLASSES = ("whole", "cut", "fragment", "other")
WHOLE, POOLED = 0, 4  # indices into a per-row record: whole-question answer positions; the pooled sum


# ----------------------------------------------------------------------------- states


def grid_state(arm: str, size: str) -> dict[str, Any]:
    """E91's record of `arm` at `size` ("full", "at_norm", "at_kl", "past_line"), its file as an absolute path."""
    rec = json.loads(GRID.read_text())["arms"][arm][size]
    return rec | {"path": EXPERIMENTS / rec["file"]}


def load_state(rec: dict[str, Any]) -> dict[str, Tensor]:
    """A state's δW on the host, as saved; its dtype must be the one E91 read."""
    dW = load(rec["path"])
    got = str(next(iter(dW.values())).dtype)
    assert got == rec["dtype"], f"{rec['path']}: {got}, E91 read {rec['dtype']}"
    return dW


def norm(dW: dict[str, Tensor]) -> float:
    """‖δW‖ over the decomposed matrices, summed in fp64 (E91's definition)."""
    return sum(float(x.double().pow(2).sum()) for x in dW.values()) ** 0.5


# ----------------------------------------------------------------------------- rows and position classes


@dataclass(frozen=True)
class RowSet:
    rows: qc.Rows
    cls: Tensor  # (N, L) int64, an index into CLASSES per position

    @property
    def tokens(self) -> Tensor:
        return self.rows.tokens

    def head(self, n: int) -> RowSet:
        r = self.rows
        return RowSet(qc.Rows(r.name, r.tokens[:n], r.answer[:n], r.first[:n]), self.cls[:n])

    def counts(self) -> dict[str, list[int]]:
        """Per class, then pooled: positions for KL (every position) and for NLL (those with a next token), and the
        rows holding the class."""
        kl, nll = self.cls, self.cls[:, :-1]
        return {"kl": [*(int((kl == i).sum()) for i in range(4)), kl.numel()],
                "nll": [*(int((nll == i).sum()) for i in range(4)), nll.numel()],
                "rows": [*(int((kl == i).any(1).sum()) for i in range(4)), len(kl)]}


def classes(rows: qc.Rows, tok: Any) -> Tensor:
    """Each position's class: an answer position by `qc.question_seen`, any other position "other"."""
    user_start = int(tok.convert_tokens_to_ids("<|im_start|>"))
    user_word = tok("user", add_special_tokens=False)["input_ids"][0]
    cls = torch.full(rows.tokens.shape, CLASSES.index("other"), dtype=torch.int64)
    cls[rows.answer] = qc.question_seen(rows, user_start, user_word)
    return cls


def maths_rows(n: int = MATHS_ROWS) -> RowSet:
    """W0's own greedy maths rows (E40), cut to ROW_LEN; no answer marks, so every position is "other"."""
    toks = torch.tensor([r[:ROW_LEN] for r in json.loads(MATHS.read_text())["rows"][:n]])
    none = torch.zeros(toks.shape, dtype=torch.bool)
    return RowSet(qc.Rows("maths", toks, none, none), torch.full(toks.shape, CLASSES.index("other"), dtype=torch.int64))


def own_rows(name: str, tok: Any, sets: dict[str, Any]) -> RowSet:
    """E92's anchor rows: "a1" (64 template rows) or "tofu" (128 TOFU rows), with their position classes."""
    if name == "a1":
        rows = qc.a1_rows(tok, sets["test"], json.loads(A1_ITEMS.read_text())["items"])
    else:
        assert name == "tofu", name
        kept = {r["row"] for r in map(json.loads, FORMS.read_text().splitlines()) if r["kept"]}
        rows = qc.tofu_rows(tok, int(tok.convert_tokens_to_ids("<|im_end|>")), kept)
    return RowSet(rows, classes(rows, tok))


def by_class(x: Tensor, cls: Tensor) -> list[float]:
    """One row's per-position values (L,) summed in fp64 per class, then the pooled sum computed apart."""
    x = x.double()
    return [*torch.zeros(4, dtype=F64, device=x.device).index_add_(0, cls.to(x.device), x).tolist(), float(x.sum())]


def nll_per_position(logits: Tensor, tokens: Tensor) -> Tensor:
    """Next-token NLL per position (L,), fp32; the last position, with no next token, reads 0."""
    out = torch.zeros(tokens.shape[-1], dtype=F32, device=logits.device)
    out[:-1] = F.cross_entropy(logits[0, :-1].float(), tokens[0, 1:], reduction="none")
    return out


# ----------------------------------------------------------------------------- weight sets on the host


def pinned(w: Tensor) -> Tensor:
    return torch.empty(w.shape, dtype=w.dtype, pin_memory=torch.cuda.is_available()).copy_(w)


class Weights:
    """Weight sets held on the host (pinned on a GPU machine) and copied into the model's own tensors in place. W0 is
    registered when the sets are made, from the weights the model holds then, so every set exists beside it; `now`
    names the set the model holds. (Moved from experiments/qwen_reading/lab.py, E57.)"""

    def __init__(self, lins: dict[str, Any]):
        self.lins, self.now = lins, "W0"
        self.sets = {"W0": {s: pinned(lin.weight.data) for s, lin in lins.items()}}

    def use(self, key: str) -> None:
        if key != self.now:
            for s, lin in self.lins.items():
                lin.weight.data.copy_(self.sets[key][s])
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        self.now = key

    def put(self, key: str, dW: dict[str, Tensor] | None, scale: float = 1.0) -> None:
        """Set `key` = W0 + scale·δW (fp32, added on the weights' device); None: a separate copy of W0, so the W0
        control goes through the same copy path."""
        self.drop(key)
        self.use("W0")
        if dW is None:
            self.sets[key] = {s: pinned(w) for s, w in self.sets["W0"].items()}
            return
        self.sets[key] = {s: pinned(lin.weight.data + scale * dW[s].to(lin.weight.data.device, torch.float32))
                          for s, lin in self.lins.items()}

    def drop(self, key: str) -> None:
        """Free set `key` (never W0), the model back at W0 if it held it, and return the pinned pages."""
        assert key != "W0", "W0 is never dropped"
        if self.now == key:
            self.use("W0")
        self.sets.pop(key, None)
        empty = getattr(torch._C, "_host_emptyCache", None)  # the pinned-memory cache keeps freed blocks otherwise
        if empty is not None:
            empty()


# ----------------------------------------------------------------------------- the decomposition on one card


class OriginalV(Components):
    """What Γ's readouts read in the new pass: W0's V of one site at a time, whatever the state has done to the
    components in place. (Moved from experiments/qwen_fixed_masks/run.py, E72.)"""

    def __init__(self, comps: Components, alive: Any):
        super().__init__(comps.V, comps.U)
        self.alive = alive

    def vu(self, site: str) -> tuple[Tensor, Tensor]:
        V, U = super().vu(site)
        O = V.clone()
        O[:, self.alive.idx[site]] = self.alive.V[site].to(V.device)
        return O, U


class Decomp:
    """The fp32 target with its weight sets, the fp32 components, Γ (bf16, as every Qwen read), the live set and Σ,
    each loaded on first use. The live set's host copy is taken before any state moves a column."""

    def __init__(self, decomp: Path, log: Callable[[str], None]):
        fp32_matmuls()
        self.dir, self.log = decomp, log
        self.model = load_target(DEV, F32)
        self.weights = Weights(linears(self.model))
        self.comps = load_components(decomp, DEV, F32)
        self._gamma = load_gamma(decomp, DEV, torch.bfloat16)
        st = [load(GSTATS / f"W0_{b}.safetensors") for b in ("maths", "fineweb")]
        self.alive = qwen_maps.Alive(self.comps, qwen_maps.alive_from_stats(*st))
        assert self.alive.count() == ALIVE_TOTAL, f"live set {self.alive.count()} != {ALIVE_TOTAL}"
        self.lay = qr.Layout({s: v.cpu() for s, v in self.alive.idx.items()})
        self._sigma: dict[str, Tensor] | None = None
        log(f"[load] target, components, Γ, live set; {torch.cuda.get_device_name()}, {peaks()}")

    @property
    def gamma(self) -> Any:
        if self._gamma.in_w.device.type == "cpu":
            self._gamma = move_gamma(self._gamma, DEV)
        return self._gamma

    def park_gamma(self) -> None:
        """Γ's 17.1 GB to the host (Σ and the split need the room); `gamma` brings it back."""
        self._gamma = move_gamma(self._gamma, "cpu")
        torch.cuda.empty_cache()

    def use(self, key: str) -> None:
        self.weights.use(key)

    def counts(self) -> qr.Counts:
        return qr.Counts(self.lay, DEV)

    def sigma(self) -> dict[str, Tensor]:
        """Σ = E[a aᵀ] (fp64, host) at W0 on 128 maths rows of 512, as E44's stage S and E57."""
        if self._sigma is None:
            self.park_gamma()
            self.use("W0")
            rows = torch.tensor([r[:SIGMA_LEN] for r in json.loads(MATHS.read_text())["rows"][:MATHS_ROWS]])
            t0 = time.time()
            self._sigma = qwen_maps.input_moments(self.model, list(rows.to(DEV).split(SIGMA_BATCH)), list(SITES))
            self.log(f"[sigma] {len(self._sigma)} sites in {time.time() - t0:.0f} s, {peaks()}")
        return self._sigma


def peaks() -> str:
    gpu = torch.cuda.max_memory_allocated() / GIB if torch.cuda.is_available() else 0.0
    return f"peak GPU {gpu:.2f} GiB, host {host_gb():.1f} GB"


def host_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


# ----------------------------------------------------------------------------- the passes


Sums = dict[str, list[list[float]]]  # key -> one [whole, cut, fragment, other, pooled] per row


def cells(d: Decomp, rs: RowSet, change: Any = None, w0_v: bool = True, c0: qr.Counts | None = None,
          c1: qr.Counts | None = None) -> Sums:
    """One pass over the rows at weight sets W0 and "FT", Δ off, one row at a time, per-row sums by class.
    change None (the components as W0's): K0 = KL(W0 clean ‖ W0 masked), K00 = KL(FT clean ‖ masked with W0's masks),
    K01 = KL(FT clean ‖ masked with FT's masks); NLLs of W0 clean, FT clean and both masked forwards (nll_m00, nll_m01);
    c0, c1 count W0's and FT's gates. A split `change` moved into the live columns: K10, K11 and nll_m10, nll_m11, the
    masks read through W0's V (w0_v, the new pass, E61's masks) or through the moved V′ (the state's own masks)."""
    model, j, out = d.model, "1" if change is not None else "0", {}
    keep = {s: lin.weight.data for s, lin in linears(model).items()}
    gate_comps = OriginalV(d.comps, d.alive) if change is not None and w0_v else d.comps

    def add(key: str, x: Tensor, cls: Tensor) -> None:
        out.setdefault(key, []).append(by_class(x[0] if x.dim() == 2 else x, cls))

    with qwen_maps.applied(model, d.comps, d.alive, change, keep) if change is not None else nullcontext():
        for i, t in enumerate(rs.tokens):
            t, cls, got = t[None].to(DEV), rs.cls[i], {}
            for key in ("FT", "W0") if d.weights.now == "FT" else ("W0", "FT"):  # one weight copy per row
                d.use(key)
                logits, g = importances(model, t, d.gamma, gate_comps)
                if (c := c0 if key == "W0" else c1) is not None:
                    c.add_row(g)
                got[key] = masked_logits(model, t, d.comps, g)
                del g
                got[f"clean_{key}"] = logits
            ref = got["clean_FT"]
            if change is None:
                add("K0", kl_per_position(got["clean_W0"], got["W0"]), cls)
                add("nll_W0", nll_per_position(got["clean_W0"], t), cls)
                add("nll_FT", nll_per_position(ref, t), cls)
            add(f"K{j}0", kl_per_position(ref, got["W0"]), cls)
            add(f"K{j}1", kl_per_position(ref, got["FT"]), cls)
            add(f"nll_m{j}0", nll_per_position(got["W0"], t), cls)
            add(f"nll_m{j}1", nll_per_position(got["FT"], t), cls)
            del got
    d.use("W0")
    return out


def total(sums: Sums, key: str, cls: int = 4, rows: int | None = None) -> float:
    """A key's sum over the first `rows` rows (all: None) in one class (4: pooled)."""
    return sum(r[cls] for r in sums[key][:rows])


# ----------------------------------------------------------------------------- the least-squares split


@dataclass
class Split:
    X: dict[str, Tensor]  # X = K⁺δW per site, fp32 on the pinned host
    pe: Tensor  # Σ-weighted piece energy per live subcomponent, fp64, flat in the live layout
    energy: dict[str, float]  # ‖δW_s‖²_Σ per site
    outside: float  # input-weighted share of δW outside the subcomponents, pooled over sites by energy
    outside_null: float  # its exact Gaussian null, pooled the same way
    stored_error: float
    seconds: float


def basis(d: Decomp, s: str) -> tuple[Tensor, Tensor, split.Basis]:
    U_A, V_A = d.alive.uv(s, DEV, F64)
    return U_A, V_A, split.basis_from(qwen_maps.make_basis(U_A, "gram"), qwen_maps.make_basis(V_A, "gram"), BASIS_TOL)


def ls_split(d: Decomp, dW: dict[str, Tensor]) -> Split:
    """The least-squares split of δW alone, site by site on the card in fp64 (Γ parked): X kept fp32 on the host
    (`qwen_maps.stored_error` reports what that costs), the piece energies, and the outside share with its null
    (`split.weighted_fractions`, pooled as E44's stage S)."""
    sig, t0, err = d.sigma(), time.time(), 0.0
    d.park_gamma()
    X, pe, E, out, null = {}, [], {}, 0.0, 0.0
    for s in SITES:
        U_A, V_A, b = basis(d, s)
        S, M = sig[s].to(DEV), dW[s].to(DEV, F64)
        Xs, E[s] = split.least_squares_x(M, b), split.weighted_energy(M, S)
        err = max(err, qwen_maps.stored_error(Xs, U_A, V_A, S, E[s]))
        X[s] = pinned(Xs.float().cpu())
        pe.append(split.piece_energies(*qwen_maps.pieces(Xs.float(), U_A, V_A), U_A, V_A, S).cpu())
        wf = split.weighted_fractions(M, b, S)
        out, null = out + wf["omega_act"] * E[s], null + wf["null_act"] * E[s]
        del U_A, V_A, b, S, M, Xs
    torch.cuda.empty_cache()
    tot = sum(E.values())
    nan = float("nan")
    return Split(X, torch.cat(pe), E, out / tot if tot > 0 else nan, null / tot if tot > 0 else nan, err, time.time() - t0)


def unit_draw(shape: tuple[int, ...], seed: int) -> Tensor:
    """`split.random_like`'s Gaussian before its rescaling: drawn on the CPU in fp64, the same G on any device."""
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed), dtype=F64)


def null_seed(seed: int, site: str) -> int:
    """The Gaussian null's seed per site (moved from experiments/qwen_reading/lab.py, E57)."""
    return seed * 1000 + SITES.index(site)


def unit_nulls(d: Decomp, shapes: dict[str, tuple[int, ...]], seeds: tuple[int, ...]
               ) -> dict[int, tuple[Tensor, dict[str, float]]]:
    """Per null seed k: the piece energies of the least-squares split of the unit Gaussian (`null_seed(k, site)`),
    and its ‖G_s‖²_Σ per site. X is linear in δW and the energies quadratic in X, so a state's null is these
    energies times ‖δW_s‖²_Σ / ‖G_s‖²_Σ per site (`null_pe`), exactly as `split.random_like` at that state."""
    sig = d.sigma()
    d.park_gamma()
    pe: dict[int, list[Tensor]] = {k: [] for k in seeds}
    EG: dict[int, dict[str, float]] = {k: {} for k in seeds}
    for s in SITES:
        U_A, V_A, b = basis(d, s)
        S = sig[s].to(DEV)
        for k in seeds:
            G = unit_draw(shapes[s], null_seed(k, s)).to(DEV)
            XG = split.least_squares_x(G, b)
            EG[k][s] = split.weighted_energy(G, S)
            pe[k].append(split.piece_energies(XG @ V_A, XG.T @ U_A, U_A, V_A, S).cpu())
            del G, XG
        del U_A, V_A, b, S
    torch.cuda.empty_cache()
    return {k: (torch.cat(pe[k]), EG[k]) for k in seeds}


def null_pe(unit: Tensor, unit_energy: dict[str, float], energy: dict[str, float], site_of: Tensor) -> Tensor:
    c2 = torch.tensor([energy[s] / unit_energy[s] for s in SITES], dtype=F64)
    return unit * c2[site_of]


def ls_change(d: Decomp, X: dict[str, Tensor] | None) -> Callable[[str], qwen_maps.SiteChange]:
    """`qwen_maps.applied`'s change for the least-squares split with stored X: the live columns move by its pieces,
    the weights are whatever set the model holds. X None: no column moves (the do-nothing state)."""

    def fn(s: str) -> qwen_maps.SiteChange:
        if X is None:
            return qwen_maps.NONE
        U_A, V_A = d.alive.uv(s, DEV, F64)
        return (None, None, *qwen_maps.pieces(X[s].to(DEV), U_A, V_A))

    return fn


# ----------------------------------------------------------------------------- the random changes


def alive_pieces(alive: qwen_maps.Alive, a: dict[str, float], device: str) -> dict[str, Tensor]:
    """E44 Amendment 2's random change on the live columns: Gaussian pieces `random_pieces(seed 0)` scaled by a[s] per
    site, as the weight change dσ(dU, dV) + dU dVᵀ (fp32, pinned host). (Moved from experiments/qwen_reading/run.py.)"""
    out = {}
    for s in SITES:
        U_A, V_A = alive.uv(s, device, F64)
        RU, RV = (x.double() for x in qwen_maps.random_pieces(tuple(U_A.shape), tuple(V_A.shape), 0, s, device))
        dU, dV = a[s] * RU, a[s] * RV
        out[s] = pinned((split.dsigma(dU, dV, U_A, V_A) + dU @ dV.T).float().cpu())
    return out


# ----------------------------------------------------------------------------- think-mode overlap


def think_refs() -> tuple[dict[str, Tensor], Tensor]:
    """E57's reference sets T (think-on answers) and L (length-matched no-think), each minus the plain answers' set,
    and the live layout's site per subcomponent."""
    def on(name: str) -> Tensor:
        t = load(E57_LARGE / f"{name}.safetensors")
        return qr.gated(t["active"], int(t["n_pos"]))

    plain = on("A/plain")
    return {k: on(f"B2/{k}") & ~plain for k in ("T", "L")}, load(E57_LARGE / "A" / "layout.safetensors")["site_of"]


def folds(S: Tensor, pool: Tensor, refs: dict[str, Tensor], site_of: Tensor, draws: int) -> dict[str, Any]:
    """|S ∩ X| over its mean under site-stratified draws from the pool (`qr.overlap`), for each reference set X."""
    out: dict[str, Any] = {"size_S": int(S.sum())}
    for k, X in refs.items():
        ov = qr.overlap(S, X, site_of, len(SITES), pool, draws)
        out[k] = ov | {"fold": ov["o"] / ov["mean"] if ov["mean"] > 0 else None}
    return out
