"""E175 (cell P3.9): port check of the frozen-neuron decomposition p-015e0710 in `commute.qwen_vpd`, as E131 did for
p-9707d6ff, and a static count against p-9707d6ff on the same rows. Rules filed in results/predictions.json first.

Stages, cheapest first, each written to results/port_check.json as it ends (and printed):
  1. recon   on the 8 qwen_port rows: the target against the decomposed model with every mask 1 and Δ off. The
             subcomponents are W's neurons, so (V U)ᵀ = W and the two agree up to fp32 rounding.
  2. jax     on the same rows: our g (Γ bf16, target fp32) against the library's (jax_reference.py, run first):
             Jaccard of the active sets at g > 0.1 pooled over positions and sites, share of values off by > 0.1,
             active counts, and the KL with g as masks against the library's.
  3. counts  mean active subcomponents per token (g > 0.1) per kind on 128 chat rows (E131's GSM8K rows with Qwen's
             own answers, thinking off; the README's text was chat) and 128 FineWeb rows of 512, on p-015e0710.
  4. new     the same counts on p-9707d6ff (Γ bf16, its components bf16: Γ reads V in bf16 anyway), per kind.

  sbatch --comment='E175 stage: port check' cluster/qwen_frozen_decomp_port_check.sbatch
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from commute import qwen_vpd as Q
from commute.jobs import hard_exit
from commute.vpd import kl_per_position

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
OUT, PRED = HERE / "results" / "port_check.json", HERE / "results" / "predictions.json"
JAX, LARGE = HERE / "results" / "jax_reference.json", HERE / "results" / "large"
PORT_ROWS, WEB, CHAT = (EXP / "qwen_port/results/tokens.json", EXP / "qwen_adversary/results/tokens.json",
                        EXP / "qwen_new_decomp/results/rows.json")
FROZEN, NEW = Path(os.environ["COMMUTE_QWEN_FROZEN_DIR"]), Path(os.environ["COMMUTE_QWEN_NEW_DIR"])
DEV, ACTIVE, OFF = "cuda", 0.1, 0.1
T0 = time.time()
RESULT: dict[str, Any] = {}


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def write(stage: str, res: dict[str, Any]) -> None:
    RESULT[stage] = res | {"at_s": round(time.time() - T0)}
    OUT.write_text(json.dumps(RESULT, indent=1))
    log(f"STAGE {stage} DONE {json.dumps(res)}")


def rows_of(path: Path, key: str | None = None) -> list[torch.Tensor]:
    data = json.loads(path.read_text())
    rows = data[key]["rows"] if key else data["rows"]
    return [torch.tensor([r], device=DEV) for r in rows]


def recon(model: Any, comps: Q.Components, rows: list[torch.Tensor]) -> dict[str, Any]:
    ones = {s: torch.ones(Q.C[Q.parse_site(s)[1]], device=DEV) for s in comps.sites}
    kls, diffs = [], []
    for t in rows:
        clean = model(t, use_cache=False).logits
        masked = Q.masked_logits(model, t, comps, ones)
        kls.append(float(kl_per_position(clean, masked).mean()))
        diffs.append(float((clean - masked).abs().max()))
    return {"sites": len(comps.sites), "kl_mean": sum(kls) / len(kls), "kl_max_row": max(kls), "logit_absdiff_max": max(diffs)}


def per_kind(g: dict[str, torch.Tensor]) -> dict[str, float]:
    """Mean active subcomponents per position (g > ACTIVE), summed over each kind's sites."""
    out: dict[str, float] = {}
    for s, v in g.items():
        k = Q.parse_site(s)[1]
        out[k] = out.get(k, 0.0) + float((v[0] > ACTIVE).sum(-1).float().mean())
    return out


def jax_compare(model: Any, gamma: Q.Gamma, comps: Q.Components, rows: list[torch.Tensor]) -> dict[str, Any]:
    ref = json.loads(JAX.read_text())
    res: dict[str, list[float]] = {"jaccard": [], "off_share": [], "active_ours": [], "active_jax": [], "ci_kl_ours": []}
    for r, t in enumerate(rows):
        logits, g = Q.importances(model, t, gamma, comps)
        jg = load_file(str(LARGE / f"jax_gamma_row{r}.safetensors"), device=DEV)
        inter = union = off = n = 0
        for s, v in g.items():
            a, b = v[0], jg[f"g.{s}"].float()
            assert a.shape == b.shape, (s, a.shape, b.shape)
            inter += int(((a > ACTIVE) & (b > ACTIVE)).sum())
            union += int(((a > ACTIVE) | (b > ACTIVE)).sum())
            off += int(((a - b).abs() > OFF).sum())
            n += a.numel()
        res["jaccard"].append(inter / union)
        res["off_share"].append(off / n)
        res["active_ours"].append(sum(per_kind(g).values()))
        res["active_jax"].append(ref["gamma"]["active_0.1"][r])
        res["ci_kl_ours"].append(float(kl_per_position(logits, Q.masked_logits(model, t, comps, g)).mean()))
        del g, jg
    pooled = {"jaccard_min_row": min(res["jaccard"]), "off_share_max_row": max(res["off_share"]),
              "ci_kl_jax": ref["masked"]["ci_kl"],
              "ci_kl_absdiff_max": max(abs(a - b) for a, b in zip(res["ci_kl_ours"], ref["masked"]["ci_kl"], strict=True))}
    return res | pooled


def counts(model: Any, gamma: Q.Gamma, comps: Q.Components, rows: list[torch.Tensor]) -> dict[str, Any]:
    tot: dict[str, float] = {}
    for t in rows:
        for k, v in per_kind(Q.importances(model, t, gamma, comps)[1]).items():
            tot[k] = tot.get(k, 0.0) + v / len(rows)
    return {"rows": len(rows), "per_kind": tot, "total": sum(tot.values()),
            "mlp": sum(v for k, v in tot.items() if k in ("gate", "up", "down"))}


def main() -> None:
    assert PRED.is_file(), f"refusing to start: {PRED} missing"
    assert Q.decomp_id(FROZEN) == "p-015e0710" and Q.decomp_id(NEW) == "p-9707d6ff"
    Q.fp32_matmuls()
    port, chat, web = rows_of(PORT_ROWS), rows_of(CHAT, "G"), rows_of(WEB)
    with torch.no_grad():
        model = Q.load_target(DEV)
        comps = Q.load_components(FROZEN, DEV)
        log(f"target + neuron components loaded, {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
        write("recon", recon(model, comps, port))
        gamma = Q.load_gamma(FROZEN, DEV, torch.bfloat16)
        log(f"gamma loaded, {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
        write("jax", jax_compare(model, gamma, comps, port))
        write("counts", {"chat": counts(model, gamma, comps, chat), "web": counts(model, gamma, comps, web)})
        del gamma, comps
        torch.cuda.empty_cache()
        comps = Q.load_components(NEW, DEV, torch.bfloat16)
        gamma = Q.load_gamma(NEW, DEV, torch.bfloat16)
        write("new", {"chat": counts(model, gamma, comps, chat), "web": counts(model, gamma, comps, web)})
    log(f"FINISHED {OUT}, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    hard_exit(0)


if __name__ == "__main__":
    main()
