"""E76 stage 0, fp32 components and Γ as E74's reader. (a) E77's hook gap (review 2026-11-03_review_e77-masks-carry.md, "Smallest
test"): the clean forward and the last-matrix-only forward once each, then the norm and head on the saved vectors. (b) K0 on E74's
rows at its seed. (c) g cached per row and the columns cut. (d) Γ freed; the student's masked forward at zero offsets against
W0's, and K0 on 8 rows through the arms' path (bf16 components, the cached g): the dtype floor."""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import rules
import torch
from common import (
    DEV,
    HOOK_READ,
    LARGE,
    LAST_SITE,
    RESULTS,
    C,
    g_path,
    kl_sum,
    load_g,
    log,
    peak,
    read_json,
    save_g,
    stop,
    write_json,
)
from safetensors.torch import save_file
from torch import Tensor

from commute.qwen_maps import halves
from commute.qwen_masked import row_masks, student_masked
from commute.qwen_teacher import Offsets
from commute.qwen_vpd import (
    SITES,
    Components,
    _hooks,
    _linear,
    importances,
    load_components,
    load_gamma,
    masked_logits,
    stochastic_masks,
)

ACTIVE = 0.01  # `qwen_maps.halves`'s g > 0.01
IDENTITY_MAX = 1e-4  # the same arithmetic plus an exact 0: expect 0; this leaves only GPU non-determinism
STAGE0, CUT = RESULTS / "stage0.json", LARGE / "cut.safetensors"


def head(model: Any, z: Tensor) -> Tensor:
    return model.lm_head(model.model.norm(z))


def first_order(model: Any, h: Tensor, removed: Tensor, denom: Tensor, eps: float) -> Tensor:
    """The logit shift of z + ε h to first order: ε weight (h minus its part along `removed`) / rms(`denom`). The norm's derivative at
    z has removed = denom = z; swapping either for the clean base changes one cause at a time."""
    norm = model.model.norm
    var = lambda z: z.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon
    y = h - removed * ((removed * h).sum(-1, keepdim=True) / (h.shape[-1] * var(removed)))
    return model.lm_head(eps * norm.weight * y * torch.rsqrt(var(denom)))


def centred(y: Tensor) -> Tensor:
    return y - y.mean(-1, keepdim=True)


def two_forwards(model: Any, t: Tensor, gamma: Any, comps: Components) -> tuple[Tensor, Tensor, Tensor, float]:
    """The norm's input in the clean forward and with only the last matrix masked, the masked output h, and KL(clean ‖ masked), E77's K0."""
    saved: dict[str, Tensor] = {}
    pre = model.model.norm.register_forward_pre_hook(lambda _m, a: saved.__setitem__("z", a[0]))
    with _hooks([pre]):
        clean, g = importances(model, t, gamma, comps)
        z_c = saved["z"]
        V, U = comps.vu(LAST_SITE)

        def hook(_m: Any, a: tuple[Tensor, ...], _o: Tensor) -> Tensor:
            saved["h"] = (a[0] @ V * g[LAST_SITE]) @ U
            return saved["h"]

        with _hooks([_linear(model, LAST_SITE).register_forward_hook(hook)]):
            masked = model(t, use_cache=False).logits
    return z_c, saved["z"], saved["h"], kl_sum(clean, masked)


def hook_gap(model: Any, rows: Tensor, gamma: Any, comps: Components, eps: float) -> dict[str, Any]:
    """Step 1 the rebuild; step 2 the masked shift with the denominator, then the part removed, from clean; step 3 per position."""
    sums: dict[str, dict[str, float]] = {}
    tot = {"aa": 0.0, "K_clean": 0.0, "K_masked": 0.0, "K0": 0.0}
    per: dict[str, list[Tensor]] = {"rms_ratio": [], "cos_h_masked_base": [], "cos_h_clean_base": []}
    for tokens in rows:
        t = tokens[None].to(DEV)
        z_c, z_m, h, k0 = two_forwards(model, t, gamma, comps)
        base_c, base_m = head(model, z_c), head(model, z_m)
        on_c, on_m = head(model, z_c + eps * h), head(model, z_m + eps * h)
        tot["K_clean"] += kl_sum(base_c, on_c)
        tot["K_masked"] += kl_sum(base_m, on_m)
        tot["K0"] += k0
        a, b = centred(on_c - base_c), centred(on_m - base_m)
        del base_c, base_m, on_c, on_m
        tot["aa"] += float((a * a).double().sum())
        variants = {"rebuild": b, "first_order": first_order(model, h, z_m, z_m, eps),
                    "denominator_from_clean": first_order(model, h, z_m, z_c, eps),
                    "removed_from_clean": first_order(model, h, z_c, z_m, eps),
                    "clean_first_order": first_order(model, h, z_c, z_c, eps)}
        for name, y in variants.items():
            y = centred(y)
            d = sums.setdefault(name, {"dot": 0.0, "bb": 0.0})
            d["dot"] += float((a * y).double().sum())
            d["bb"] += float((y * y).double().sum())
        del a, b, variants
        per["rms_ratio"].append(z_m.pow(2).mean(-1).sqrt()[0] / z_c.pow(2).mean(-1).sqrt()[0])
        per["cos_h_masked_base"].append(torch.cosine_similarity(h, z_m, dim=-1)[0])
        per["cos_h_clean_base"].append(torch.cosine_similarity(h, z_c, dim=-1)[0])
    out: dict[str, Any] = {k: {"amp": (d["bb"] / tot["aa"]) ** 0.5, "cos": d["dot"] / (tot["aa"] * d["bb"]) ** 0.5} for k, d in sums.items()}
    out |= {"tau": tot["K_masked"] / tot["K_clean"], "K0_sum": tot["K0"]}
    stacked = {k: torch.stack(v) for k, v in per.items()}
    save_file({k: v.to("cpu", torch.float16).contiguous() for k, v in stacked.items()}, str(LARGE / "stage0_norm.safetensors"))
    return out | {"per_position_mean": {k: float(v.double().mean()) for k, v in stacked.items()}}


def e74_k0(model: Any, rows: Tensor, gamma: Any, comps: Components) -> list[float]:
    """Per row KL(clean ‖ masked), m = g + (1 − g) u, d ~ U(0, 1), one draw per row, E74 Amendment 2's stream."""
    gen = torch.Generator(DEV).manual_seed(C.e74_seed)
    out = []
    for tokens in rows:
        t = tokens[None].to(DEV)
        clean, g = importances(model, t, gamma, comps)
        masks, deltas = stochastic_masks(g, gen)
        out.append(kl_sum(clean, masked_logits(model, t, comps, masks, deltas)))
    return out


def cache_g(model: Any, rows: Tensor, gamma: Any, comps: Components, out: Path, only: list[int]) -> dict[str, list[Tensor]]:
    """Writes `only`'s g; returns per site the mean g and the share of positions with g > 0.01 over those rows."""
    mean = {s: torch.zeros(comps.vu(s)[0].shape[1], dtype=torch.float64) for s in SITES}
    active = {s: v.clone() for s, v in mean.items()}
    for i in only:
        _, g = importances(model, rows[i][None].to(DEV), gamma, comps)
        save_g(out, i, g)
        for s in SITES:
            v = g[s].flatten(0, 1)
            mean[s] += v.sum(0).double().cpu()
            active[s] += (v > ACTIVE).sum(0).double().cpu()
    n = len(only) * rows.shape[1]
    return {s: [mean[s] / n, active[s] / n] for s in SITES}


def cut_columns(stats: dict[str, list[Tensor]]) -> tuple[dict[str, Tensor], float]:
    """Per site up to `C.columns` of the columns ever on, `halves`'s upper part (share on, ties by mean g), and the share of Σg they hold."""
    cut, held, total = {}, 0.0, 0.0
    for s, (mean, active) in stats.items():
        on = (active > 0).nonzero().flatten()
        k = min(C.columns, len(on))
        cut[s] = on[halves(active[on], mean[on], k)[1]]
        held += float(mean[cut[s]].sum())
        total += float(mean.sum())
    return cut, held / total if total > 0 else float("nan")


def floor_read(model: Any, rows: Tensor, comps: Components, cut: dict[str, Tensor], out: Path, k0_rows: list[float]) -> dict[str, float]:
    """Γ gone. W0's masked forward against the student's at zero offsets (fp32 components, same draw), then K0 on the fixed rows
    through the arms' path: bf16 components, g from the cache. The floor is that K0's gap to the fp32 K0 on the same rows and draws."""
    off = Offsets(comps, cut, DEV)
    t = rows[0][None].to(DEV)
    masks, deltas = row_masks(load_g(out, 0), torch.Generator(DEV).manual_seed(C.check_seed))
    want = masked_logits(model, t, comps, masks, deltas)
    with student_masked(model, comps, off, masks, deltas):
        got = model(t, use_cache=False).logits
    gap = float((want - got).abs().max())
    del want, got, masks, deltas
    bf = comps.to(DEV, torch.bfloat16)
    gen, k = torch.Generator(DEV).manual_seed(C.e74_seed), 0.0
    for i in range(C.fixed_rows):
        t = rows[i][None].to(DEV)
        clean = model(t, use_cache=False).logits
        masks, deltas = row_masks(load_g(out, i), gen)
        with student_masked(model, bf, off, masks, deltas):
            k += kl_sum(clean, model(t, use_cache=False).logits)
    n = C.fixed_rows * rows.shape[1]
    k32 = sum(k0_rows[: C.fixed_rows]) / n
    return {"identity_max_abs": gap, "K0_8_fp32": k32, "K0_8_bf16": k / n, "floor": abs(k / n - k32)}


def stage0(model: Any, e74_rows: Tensor, rows: Tensor, decomp: Path, out: Path) -> dict[str, Any]:
    """Skipped when its file is written and the g files are on this node; a new node computes g again."""
    done = read_json(STAGE0)
    missing = [i for i in range(len(rows)) if not g_path(out, i).is_file()]
    if done is not None and not missing:
        return done
    comps, gamma = load_components(decomp, DEV, torch.float32), load_gamma(decomp, DEV, torch.bfloat16)
    log(f"[stage 0] decomposition on the card, {peak()}")
    if done is not None:
        cache_g(model, rows, gamma, comps, out, missing)
        del comps, gamma
        return done
    eps = read_json(HOOK_READ / "amend1_own_last.json")["fit"]["eps"]
    filed = read_json(HOOK_READ / "amend2_hook.json")["summary"]["own"]
    s: dict[str, Any] = {"hook": hook_gap(model, e74_rows, gamma, comps, eps), "hook_filed": filed}
    log(f"[stage 0 hook gap] eps {eps}: {s['hook']}, E77's read {filed}")
    k0_rows = e74_k0(model, e74_rows, gamma, comps)
    s |= {"K0": sum(k0_rows) / e74_rows.numel(), "K0_rows": k0_rows}
    log(f"[stage 0 K0] {s['K0']:.6g} on {len(k0_rows)} rows, seed {C.e74_seed}, E74's {rules.K0_WANT}, {peak()}")
    stats = cache_g(model, rows, gamma, comps, out, list(range(len(rows))))
    cut, s["share"] = cut_columns(stats)
    s["columns"] = {"total": sum(len(v) for v in cut.values()), "sites_under": sum(len(v) < C.columns for v in cut.values())}
    log(f"[stage 0 share] the cut holds {s['share']:.4f} of the sum of g over {len(rows)} rows; {s['columns']}")
    miss = rules.stage0_stop(s)  # before the floor: a stop's reading is the one thing needed
    write_json(RESULTS / "stage0_partial.json", s)
    if miss:
        stop(*miss)
    save_file({k: v.contiguous() for k, v in cut.items()}, str(CUT))
    del gamma
    gc.collect()
    torch.cuda.empty_cache()
    s |= floor_read(model, rows, comps, cut, out, k0_rows)
    log(f"[stage 0 floor] {s['identity_max_abs']:.3g} max logit gap at zero offsets, fp32 K0 on 8 rows {s['K0_8_fp32']:.6g}, "
        f"bf16 {s['K0_8_bf16']:.6g}, floor {s['floor']:.4g}, {peak()}")
    if not s["identity_max_abs"] <= IDENTITY_MAX:
        stop("identity", f"student's masked forward at zero offsets differs from W0's by {s['identity_max_abs']}")
    write_json(STAGE0, s)
    return s

