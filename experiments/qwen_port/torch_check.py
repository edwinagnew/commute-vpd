"""Torch port of the Qwen3-1.7B decomposition checked against the JAX reference. Plan:
reports/plans/2026-10-27_plan_qwen-port-and-baseline.md, Part 1 item 5; the pass rules are its table (checks 0-4).

Reads results/tokens.json (8 rows, one document each) and the JAX reference written by jax_reference.py, reproduces
each reference output with `commute.qwen_vpd`, and writes results/torch_check.json after every stage with every
number and each check's pass/fail; every number is also printed. Stages, cheapest first:
  0  target logits on the 8 rows: mean per-position KL(HF fp32 ‖ JAX) against the floor KL(HF fp32 ‖ HF bf16).
     Expected: ratio near 1 (JAX runs the target in bf16). Pass: ratio <= CHECK0_FACTOR. Else stop.
  1  g from torch Γ on JAX's 29 residual taps (readouts from torch's own site inputs) vs JAX's g, on the rows JAX
     saved g for. Expected: share of (position, component) pairs with |diff| > 0.1 near 0. Pass: < 1e-3. Else stop.
  2  g torch end to end vs JAX's g, same rule; plus the relative L2 error of each torch tap against JAX's.
  3  masked-forward KL to the clean logits, mean over all positions, for m = g without Δ, m = 1 without Δ, and the
     fixed source array with Δ on. Torch's g. Expected: within a few % of JAX. Pass: |torch − JAX| <= max(0.05 JAX, 2 × check 0 floor),
     scored with the masked forward in bf16 (JAX's precision); the fp32 numbers are reported beside it.
  4  the config's 20-step adversary (one source per site shared by all rows), seeds 0-2, fp32. Pass: the [min, max]
     ranges of the 3 torch and the 3 JAX seeds overlap (the RNGs differ, so seeds are not paired).
Γ runs in bf16 here (JAX's precision, and the only way it fits beside the fp32 target and V on an L40).

Memory on one L40 (46 GB): stage 0 holds the fp32 target (6.9 GB) and briefly a bf16 copy (3.4 GB); stages 1-2 add
U, V in bf16 (9.8 GB) and Γ in bf16 (17.1 GB), ~34 GB; stages 3-4 drop Γ and hold U, V in fp32 (19.5 GB), ~26 GB plus
~6 GB of autograd state per row under the adversary. Host: U, V fp32 (19.5 GB), g and logits per row (~20 GB).

Usage (repo root, cluster GPU job): sbatch cluster/qwen_torch_check.sbatch
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from commute.qwen_vpd import (
    KINDS,
    N_TAPS,
    SITES,
    Components,
    clean_forward,
    fp32_matmuls,
    load_components,
    load_gamma,
    load_target,
    lower_importances,
    masked_logits,
    parse_site,
    pgd_kl,
)
from commute.vpd import kl_per_position

HERE = Path(__file__).resolve().parent
RESULTS, LARGE = HERE / "results", HERE / "results" / "large"
TOKENS, SUMMARY, OUT = RESULTS / "tokens.json", RESULTS / "jax_reference.json", RESULTS / "torch_check.json"
CHECK0_FACTOR = 2.0  # two independent bf16 paths: JAX may sit up to twice HF bf16's distance from HF fp32
G_TOL, G_SHARE = 0.1, 1e-3
KL_REL = 0.05
SEEDS = (0, 1, 2)
GAMMA_DTYPE = torch.bfloat16
MASKINGS = ("ci", "ones", "source")
DEV = "cuda"
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def save(out: dict) -> None:
    OUT.write_text(json.dumps(out, indent=1))


# ----------------------------------------------------------------------------- the JAX reference


@dataclass
class Reference:
    """What jax_reference.py wrote (results/jax_reference.json, results/large/*.safetensors), per row of tokens.json,
    positions below each row's length only."""

    logits: list[torch.Tensor]  # per row [T, vocab], fp32
    taps: dict[int, list[torch.Tensor]]  # row -> 29 x [T, 2048], bf16
    g: dict[int, dict[str, torch.Tensor]]  # detail row -> site -> [T, C], lower importances, bf16
    stats: list[dict[str, dict[str, float]]]  # per row, site -> mean_g, l0, mean_p
    source: dict[str, torch.Tensor]  # site -> [C], the fixed source array
    source_delta: dict[str, torch.Tensor]  # site -> [1]
    masked_kl: dict[str, list[torch.Tensor]]  # "ci", "ones", "source" -> per row [T] per-position KL
    pgd_kl: list[float]  # one per seed
    pgd_rows: list[int]  # rows the adversary's batch held (the text rows)


def load_reference(kinds: list[str]) -> Reference:
    summary = json.loads(SUMMARY.read_text())
    assert summary["adversary"]["seeds"] == list(SEEDS), summary["adversary"]
    n_rows = len(kinds)
    target = [load_file(LARGE / f"target_row{r}.safetensors") for r in range(n_rows)]
    taps = [load_file(LARGE / f"taps_row{r}.safetensors") for r in range(n_rows)]
    gamma = {r: load_file(LARGE / f"gamma_row{r}.safetensors") for r in summary["detail_rows"]}
    masked = [load_file(LARGE / f"masked_row{r}.safetensors") for r in range(n_rows)]
    source = load_file(LARGE / "source.safetensors")
    return Reference(
        logits=[t["logits"] for t in target],
        taps={r: [t[f"resid.{b}"] for b in range(N_TAPS)] for r, t in enumerate(taps)},
        g={r: {s: f[f"g.{s}"] for s in SITES} for r, f in gamma.items()},
        stats=summary["gamma"]["per_site"],
        source={s: source[f"{s}.components"] for s in SITES},
        source_delta={s: source[f"{s}.delta"] for s in SITES},
        masked_kl={k: [m[f"{k}.kl"] for m in masked] for k in MASKINGS},
        pgd_kl=summary["adversary"]["kl"],
        pgd_rows=[i for i, kind in enumerate(kinds) if kind == "text"],
    )


# ----------------------------------------------------------------------------- scoring helpers


def mean_kl(p: list[torch.Tensor], q: list[torch.Tensor]) -> float:
    """Mean per-position KL(P ‖ Q) over every position of every row."""
    total = sum(float(kl_per_position(a.to(DEV), b.to(DEV)).sum()) for a, b in zip(p, q, strict=True))
    return total / sum(a.shape[0] for a in p)


def top1(p: list[torch.Tensor], q: list[torch.Tensor]) -> float:
    agree = sum(int((a.argmax(-1) == b.argmax(-1)).sum()) for a, b in zip(p, q, strict=True))
    return agree / sum(a.shape[0] for a in p)


class GDiff:
    """Counts of (position, component) pairs with |g_torch − g_jax| > G_TOL, overall and per kind."""

    def __init__(self) -> None:
        self.n = {k: 0 for k in KINDS}
        self.bad = {k: 0 for k in KINDS}
        self.max_abs = 0.0
        self.sum_abs = 0.0

    def add(self, mine: dict[str, torch.Tensor], ref: dict[str, torch.Tensor]) -> None:
        for site in SITES:
            kind = parse_site(site)[1]
            d = (mine[site][0] - ref[site].to(DEV)).abs()
            self.n[kind] += d.numel()
            self.bad[kind] += int((d > G_TOL).sum())
            self.max_abs = max(self.max_abs, float(d.max()))
            self.sum_abs += float(d.sum())

    def result(self) -> dict:
        n, bad = sum(self.n.values()), sum(self.bad.values())
        share = bad / n
        return {"share_over_0.1": share, "n_pairs": n, "max_abs_diff": self.max_abs, "mean_abs_diff": self.sum_abs / n,
                "share_by_kind": {k: self.bad[k] / self.n[k] for k in KINDS}, "pass": share < G_SHARE}


def row_stats(g: dict[str, torch.Tensor], jax: dict[str, dict[str, float]]) -> dict:
    """One row's total L0 (mean count of g > 0 per position, summed over sites) and the largest per-site gap in
    mean g, torch vs JAX's summary (no pass rule: the detail rows carry the check; this covers the other six)."""
    l0 = sum(float((v[0] > 0).sum(-1).float().mean()) for v in g.values())
    gap = max(abs(float(v[0].mean()) - jax[s]["mean_g"]) for s, v in g.items())
    return {"l0_total": l0, "l0_total_jax": sum(x["l0"] for x in jax.values()), "max_site_mean_g_gap": gap}


# ----------------------------------------------------------------------------- stages


def stage0(rows: list[torch.Tensor], ref: Reference, out: dict) -> tuple[Any, list, list]:
    model = load_target(DEV, torch.float32)
    with torch.no_grad():
        l32 = [model(r, use_cache=False).logits[0].cpu() for r in rows]
        bf16 = load_target(DEV, torch.bfloat16)
        l16 = [bf16(r, use_cache=False).logits[0].float().cpu() for r in rows]
    del bf16
    torch.cuda.empty_cache()
    jax = [x.float() for x in ref.logits]
    kl_jax, kl_floor = mean_kl(l32, jax), mean_kl(l32, l16)
    res = {"kl_fp32_vs_jax": kl_jax, "kl_fp32_vs_bf16_floor": kl_floor, "kl_bf16_vs_jax": mean_kl(l16, jax),
           "ratio": kl_jax / kl_floor, "factor": CHECK0_FACTOR, "top1_fp32_vs_jax": top1(l32, jax),
           "top1_fp32_vs_bf16": top1(l32, l16)}
    res["pass"] = res["ratio"] <= CHECK0_FACTOR
    out["check0"] = res
    log(f"check 0: {res}")
    return model, l32, l16


def stage12(model: Any, rows: list[torch.Tensor], ref: Reference, comps_cpu: Components, out: dict) -> list:
    comps = comps_cpu.to(DEV, torch.bfloat16)  # readouts read V in Γ's dtype either way (qwen_vpd.readout)
    gamma = load_gamma(device=DEV, dtype=GAMMA_DTYPE)
    log(f"Γ loaded; GPU memory {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
    on_jax_taps, end_to_end, gs = GDiff(), GDiff(), []
    tap_err, stats = {}, []
    for i, tokens in enumerate(rows):
        clean = clean_forward(model, tokens, gamma, comps)
        with torch.no_grad():
            if i in ref.g:
                g1 = lower_importances(gamma, [t[None].to(DEV) for t in ref.taps[i]], clean.readouts)
                on_jax_taps.add(g1, ref.g[i])
                del g1
            g = lower_importances(gamma, clean.taps, clean.readouts)
        if i in ref.g:
            end_to_end.add(g, ref.g[i])
        if i in ref.taps:
            tap_err[i] = [float((t[0].float().cpu() - j.float()).norm() / j.float().norm())
                          for t, j in zip(clean.taps, ref.taps[i], strict=True)]
        stats.append(row_stats(g, ref.stats[i]))
        gs.append({s: v.cpu() for s, v in g.items()})
        del clean, g
    out["check1"] = on_jax_taps.result()
    out["check2"] = {**end_to_end.result(), "tap_rel_l2_by_boundary": tap_err, "all_rows": stats}
    log(f"check 1 (torch Γ on JAX taps): {out['check1']}")
    log(f"check 2 (torch end to end): {out['check2']}")
    del gamma, comps
    torch.cuda.empty_cache()
    return gs


def masked_kls(model: Any, rows: list, comps: Components, gs: list, cleans: list, ref: Reference) -> dict:
    """Per-position KL(clean ‖ masked) per row, for the three mask settings of check 3."""
    one = torch.ones((), device=DEV)
    kls: dict[str, list[torch.Tensor]] = {k: [] for k in MASKINGS}
    with torch.no_grad():
        for tokens, g_cpu, clean in zip(rows, gs, cleans, strict=True):
            g = {s: v.to(DEV) for s, v in g_cpu.items()}
            src = {s: g[s] + (1 - g[s]) * ref.source[s].to(DEV) for s in SITES}
            dlt = {s: ref.source_delta[s].to(DEV).expand(tokens.shape) for s in SITES}
            settings = {"ci": (g, None), "ones": (dict.fromkeys(SITES, one), None), "source": (src, dlt)}
            for name, (masks, deltas) in settings.items():
                logits = masked_logits(model, tokens, comps, masks, deltas)[0]
                kls[name].append(kl_per_position(clean.to(DEV), logits).cpu())
            del g, src
    return kls


def kl_summary(kls: dict[str, list[torch.Tensor]]) -> dict:
    """Mean over every position of every row, and per row."""
    return {k: {"mean": float(torch.cat(v).mean()), "per_row": [float(x.mean()) for x in v]} for k, v in kls.items()}


def check3(fp32: dict, bf16: dict, jax: dict, floor: float) -> dict:
    # A setting whose KL sits near the bf16 noise floor (m = 1 without Δ) can't meet 5% relative, so the allowed
    # difference is the larger of 5% of JAX's KL and twice check 0's floor (lead, 2026-10-27, before the run).
    tol = {k: max(KL_REL * jax[k]["mean"], CHECK0_FACTOR * floor) for k in MASKINGS}
    diff = {p: {k: abs(v[k]["mean"] - jax[k]["mean"]) for k in MASKINGS} for p, v in (("fp32", fp32), ("bf16", bf16))}
    return {"jax": jax, "torch_fp32": fp32, "torch_bf16": bf16, "abs_diff": diff, "tolerance": tol,
            "scored_on": "bf16", "pass": all(diff["bf16"][k] <= tol[k] for k in MASKINGS)}


def stop(reason: str) -> None:
    log(reason)
    print("FINISHED", OUT, flush=True)


def main() -> None:
    fp32_matmuls()
    assert torch.cuda.get_device_properties(0).total_memory > 44 * 2**30, "needs a 46 GB card (L40) or larger"
    tokens = json.loads(TOKENS.read_text())
    rows = [torch.tensor([r], device=DEV) for r in tokens["rows"]]
    ref = load_reference(tokens["kinds"])
    out: dict = {"gamma_dtype": str(GAMMA_DTYPE), "lengths": [r.shape[1] for r in rows]}
    model, l32, l16 = stage0(rows, ref, out)
    save(out)
    if not out["check0"]["pass"]:
        return stop("check 0 failed: the target differs; stopping")
    comps_cpu = load_components(device="cpu")
    log("components loaded to host")
    gs = stage12(model, rows, ref, comps_cpu, out)
    save(out)
    if not out["check1"]["pass"]:
        return stop("check 1 failed: fix the Γ port; stopping")
    comps = comps_cpu.to(DEV, torch.float32)
    del comps_cpu
    kl32 = kl_summary(masked_kls(model, rows, comps, gs, l32, ref))
    jax = kl_summary(ref.masked_kl)
    out["check3"] = {"jax": jax, "torch_fp32": kl32}
    log(f"check 3 fp32: {out['check3']}")
    save(out)
    idx = ref.pgd_rows
    pgd = []
    for seed in SEEDS:
        pgd.append(pgd_kl(model, [rows[i] for i in idx], comps, [gs[i] for i in idx], [l32[i] for i in idx], seed))
        log(f"check 4 seed {seed}: {pgd[-1]}")
    overlap = min(pgd) <= max(ref.pgd_kl) and min(ref.pgd_kl) <= max(pgd)
    out["check4"] = {"torch": pgd, "jax": ref.pgd_kl, "rows": idx, "pass": overlap}
    log(f"check 4: {out['check4']}")
    save(out)
    model.to(torch.bfloat16)  # (HF's weights are bf16-exact: this equals a bf16 load)
    comps = comps.to(DEV, torch.bfloat16)
    torch.cuda.empty_cache()
    kl16 = kl_summary(masked_kls(model, rows, comps, gs, l16, ref))
    out["check3"] = check3(kl32, kl16, jax, out["check0"]["kl_fp32_vs_bf16_floor"])
    log(f"check 3: {out['check3']}")
    save(out)
    log("passes: " + ", ".join(f"{k}={out[k]['pass']}" for k in ("check0", "check1", "check2", "check3", "check4")))
    print("FINISHED", OUT, flush=True)


if __name__ == "__main__":
    main()
