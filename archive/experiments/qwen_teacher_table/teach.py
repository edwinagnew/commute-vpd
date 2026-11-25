"""E111's teacher training from the least-squares split (plan reports/plans/2026-11-06_plan_e111-teacher-table.md, section 2
and its "Overrides" section). The student is W0 + the least-squares moved part + r's change: `FromLS` holds offsets on
E76's cut (200 often-on columns per site, all alive) whose base columns are least squares' moved ones, so `change` is
the least-squares moved part plus r's change, cross terms included, and `qwen_teacher.train` (E76's arm A loss, Adam)
and `kl_to_teacher` run on it unchanged. Γ and Δ stay frozen (GOALS.md:61).
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from typing import Any

import torch
from rules import FIRST, plateau, rel
from torch import Tensor

from commute import qwen_comparators as qc
from commute import qwen_d3 as d3
from commute import qwen_maps, qwen_sets, split
from commute import qwen_tofu as tofu
from commute.jobs import EXPERIMENTS
from commute.qwen_sft import linears, sft_seq
from commute.qwen_teacher import Offsets, change_hooks, kl_to_teacher, tf32, train
from commute.qwen_vpd import SITES
from commute.vpd import kl_per_position

DEV, F64 = d3.DEV, d3.F64
ROW_LEN, POOL, READ_ROWS, NGRAM = 256, 32, 8, 16  # E76 (common.py:34-37): rows of 256, pools of 32, 8 reading rows
WEB = EXPERIMENTS / "qwen_adversary" / "results" / "tokens.json"
BETA2, BATCH_SEED = 0.999, 0  # Adam's (qwen_teacher.train); E76's batch seed


class FromLS(Offsets):
    """Offsets on `cut` over the least-squares state: V, U at the cut hold least squares' moved columns, `base` is its
    moved part per site ([d_out, d_in] fp32 on the card), so `change` = moved part + r's change."""

    def __init__(self, d: d3.Decomp, cut: dict[str, Tensor], X: dict[str, Tensor], base: dict[str, Tensor]):
        super().__init__(d.comps, cut, DEV)
        self.base, self.pos = base, {}
        with torch.no_grad():
            for s in SITES:
                alive, c = d.alive.idx[s].cpu(), cut[s]
                p = torch.searchsorted(alive, c).clamp(max=len(alive) - 1)
                assert bool((alive[p] == c).all()), f"{s}: a cut column is not alive"
                self.pos[s] = p.to(DEV)
                dU, dV = ls_pieces(d, X, s)
                self.V[s] += dV[:, self.pos[s]].float()
                self.U[s] += dU[:, self.pos[s]].T.float()

    def r_change(self, site: str) -> Tensor:
        return super().change(site)

    def change(self, site: str, device: str | None = None) -> Tensor:
        return self.base[site].to(device or DEV) + super().change(site, device)

    @torch.no_grad()
    def r2(self) -> float:
        return sum(float(self.r_change(s).double().pow(2).sum()) for s in SITES)


def ls_pieces(d: d3.Decomp, X: dict[str, Tensor], s: str) -> tuple[Tensor, Tensor]:
    U_A, V_A = d.alive.uv(s, DEV, F64)
    return qwen_maps.pieces(X[s].to(DEV), U_A, V_A)


def trained_change(d: d3.Decomp, X: dict[str, Tensor], off: FromLS) -> Callable[[str], qwen_maps.SiteChange]:
    """`qwen_maps.applied`'s change for the trained state: least squares' pieces plus r's offsets at the cut's places."""

    def fn(s: str) -> qwen_maps.SiteChange:
        dU, dV = ls_pieces(d, X, s)
        p = off.pos[s]
        dU[:, p] += off.dU[s].detach().T.double()
        dV[:, p] += off.dV[s].detach().double()
        return (None, None, dU, dV)

    return fn


@torch.no_grad()
def moved_total(d: d3.Decomp, X: dict[str, Tensor], off: FromLS) -> tuple[dict[str, Tensor], float]:
    """The trained state's moved part per site (fp32, pinned host) and the largest relative gap between it and
    U′V′ᵀ − UVᵀ of `trained_change`'s pieces, fp64: the two forms of one state must agree to fp32 rounding."""
    fn, out, gap = trained_change(d, X, off), {}, 0.0
    for s in SITES:
        _, _, dU, dV = fn(s)
        assert dU is not None and dV is not None
        U_A, V_A = d.alive.uv(s, DEV, F64)
        mv, ch = split.dsigma(dU, dV, U_A, V_A) + dU @ dV.T, off.change(s).double()
        gap = max(gap, float((mv - ch).norm() / ch.norm().clamp(min=1e-30)))
        out[s] = d3.pinned(ch.float().cpu())
        del dU, dV, U_A, V_A, mv, ch
    return out, gap


# ----------------------------------------------------------------------------- training rows


def gsm8k_pool(tok: Any) -> Tensor:
    """GSM8K train problems past E40's 256 held-out ones (`d3.gold_held`'s permutation), with their gold answers as E40's
    SFT formats them, packed end to end into POOL rows of ROW_LEN (the table's maths and template rows are W0's and the
    prompted model's own answers to other questions)."""
    end, train_items = int(tok.convert_tokens_to_ids("<|im_end|>")), qwen_sets.load()["train"]
    perm = torch.randperm(qwen_sets.N_TRAIN, generator=torch.Generator().manual_seed(d3.GOLD_PERM_SEED)).tolist()
    seqs, n = [], 0
    for i in perm[d3.GOLD_HELD :]:
        seqs.append(sft_seq(tok, train_items[i]["question"], train_items[i]["gold"], end, 2 * ROW_LEN)[0])
        n += len(seqs[-1])
        if n >= POOL * ROW_LEN:
            break
    return torch.tensor(qc.pack(seqs, POOL, ROW_LEN))


def tofu_pool(tok: Any) -> Tensor:
    """POOL rows of ROW_LEN packed from TOFU's retain90 QA pairs (same wording as the table's rows) that are none of the 800
    held pairs the table's rows and column 4's gain read (`qwen_tofu.held_rows`): no QA pair, and so no row position, is in
    both the pool and the table (review f178f4e3, item 6)."""
    end = int(tok.convert_tokens_to_ids("<|im_end|>"))
    held = {r["question"] for r in tofu.held_rows()}
    seqs, n = [], 0
    for r in tofu.rows("retain90"):
        if r["question"] in held:
            continue
        seqs.append(tofu.seq(tok, end, r["question"] + tofu.ASK, r["answer"])[0])
        n += len(seqs[-1])
        if n >= POOL * ROW_LEN:
            break
    assert n >= POOL * ROW_LEN, f"{len(seqs)} retain90 pairs outside the held 800 give {n} tokens, {POOL * ROW_LEN} wanted"
    return torch.tensor(qc.pack(seqs, POOL, ROW_LEN))


def web_pool() -> Tensor:
    return torch.tensor([r[:ROW_LEN] for r in json.loads(WEB.read_text())["rows"][:POOL]])


def overlap(table: Tensor, pool: Tensor) -> float:
    """Share of the table rows' NGRAM-token windows found anywhere in the pool (E76 read 14.6%: the prompt template)."""
    def grams(rows: Tensor) -> list[tuple[int, ...]]:
        return [tuple(r[i : i + NGRAM]) for r in rows.tolist() for i in range(len(r) - NGRAM + 1)]

    seen, mine = set(grams(pool)), grams(table)
    return sum(g in seen for g in mine) / len(mine) if mine else math.nan


# ----------------------------------------------------------------------------- one arm


@torch.no_grad()
def tf32_k(model: Any, rows: list[Tensor], dW: dict[str, Tensor], off: FromLS) -> float:
    """The training forward's KL(FT ‖ student) on `rows`: TF32, each change added as its own term (`qwen_teacher.train`)."""
    total, n = 0.0, 0
    with tf32():
        for t in rows:
            with change_hooks(model, dW.__getitem__):
                p = model(t, use_cache=False).logits
            with change_hooks(model, off.change):
                q = model(t, use_cache=False).logits
            total += float(kl_per_position(p, q).double().sum())
            n += t.numel()
    return total / n


def median_sqrt_v(v: dict[Tensor, Tensor], step: int) -> float:
    """Median over the offset tensors of each one's median √v̂ (Adam's bias-corrected second moment)."""
    c = 1 - BETA2**step
    return float(torch.tensor([float((x / c).sqrt().median()) for x in v.values()]).median())


def teach(d: d3.Decomp, off: FromLS, dW: dict[str, Tensor], pools: tuple[Tensor, Tensor], read_rows: list[Tensor],
          lr: float, steps: int, every: int, forecast: Callable[[int], float], early: Callable[[str, dict[str, Any]], None],
          log: Callable[[str], None]) -> dict[str, Any]:
    """Train r from zero offsets (fresh Adam) for at most `steps` steps, reading ‖r‖² and the 8-row K every `every` steps;
    `early(name, values)` judges the first-minute values (it stops the job on a miss). Ends at `steps` or at a plateau
    of the 8-row K; the state kept is the last one."""
    off.reset()
    W0 = {s: lin.weight.data for s, lin in linears(d.model).items()}
    k0 = kl_to_teacher(d.model, read_rows, W0, dW, off)
    rec: dict[str, Any] = {"lr": lr, "steps_max": steps,
                           "readings": [{"step": 0, "r2": off.r2(), "K8": sum(k0) / len(k0), "rows": k0}],
                           "tf32_K8": tf32_k(d.model, read_rows, dW, off)}
    k8 = rec["readings"][0]["K8"]
    log(f"[teach zero offsets] TF32 K8 {rec['tf32_K8']:.9g} against fp32 {k8:.9g}: relative gap {rel(rec['tf32_K8'], k8):.3g} "
        f"(reference {FIRST['tf32_rel']:g}, logged only)")
    early("zero_offsets", {"K8": k8, "tf32_K8": rec["tf32_K8"], "r2": rec["readings"][0]["r2"]})
    v = {p: torch.zeros_like(p) for p in off.params()}
    t0 = time.time()

    def on_step(step: int, curve: list[dict[str, float]]) -> bool:
        for p in off.params():
            assert p.grad is not None
            v[p].mul_(BETA2).addcmul_(p.grad, p.grad, value=1 - BETA2)
        if step == 1:
            g = {k: sum(float(t[s].grad.double().pow(2).sum()) for s in SITES if t[s].grad is not None)  # type: ignore[union-attr]
                 for k, t in (("dV", off.dV), ("dU", off.dU))}
            early("gradient", g)
        if step % every:
            return False
        k = kl_to_teacher(d.model, read_rows, W0, dW, off)
        r = {"step": step, "r2": off.r2(), "forecast_r2": forecast(step), "K8": sum(k) / len(k), "rows": k,
             "train_kl": sum(c["kl"] for c in curve[-every:]) / every, "s_per_step": (time.time() - t0) / step}
        r["r2_over_forecast"] = r["r2"] / r["forecast_r2"]
        rec["readings"].append(r)
        log(f"[teach step {step}] r2 {r['r2']:.6g} (forecast {r['forecast_r2']:.3g}, ratio {r['r2_over_forecast']:.3g}) K8 {r['K8']:.9g} "
            f"(step 0 {rec['readings'][0]['K8']:.9g}, difference {r['K8'] - rec['readings'][0]['K8']:.3g}) "
            f"train KL {r['train_kl']:.6g}, {r['s_per_step']:.2f} s per step, {d3.peaks()}")
        if step == every:
            early("first_reading", {"r2": r["r2"], "forecast_r2": r["forecast_r2"], "K8": r["K8"],
                                    "K8_step0": rec["readings"][0]["K8"], "K8_minus_step0": r["K8"] - rec["readings"][0]["K8"],
                                    "median_sqrt_v": median_sqrt_v(v, step), "gpu_gib": torch.cuda.max_memory_allocated() / d3.GIB})
        return plateau([x["K8"] for x in rec["readings"]])

    with torch.enable_grad():
        curve = train(d.model, off, dW, pools[0], pools[1], lr, steps, 1, 1, BATCH_SEED, log, on_step)
    rec |= {"steps_run": len(curve), "kept_step": len(curve), "plateau": len(curve) < steps, "train_s": time.time() - t0,
            "first_batch": {k: curve[0][k] for k in ("kl", "kl_fp32_teacher_w0")}, "max_abs_offset": off.max_abs()}
    return rec
