"""E131 Amendment 1: the W0 references every Qwen re-run on the new decomposition needs (plan
reports/plans/2026-11-09_plan_new-decomposition-reruns.md, step 0(b)). They exist for the old decomposition only, at
~/commute-data/results-large/qwen_sft/nowarmup/{gstats,sources} (E40's stage G, experiments/qwen_sft/run.py:790-824).

The recipe is E40's, unchanged: the 128 FineWeb rows of experiments/qwen_adversary/results/tokens.json and the 128
rows of experiments/qwen_sft/results/maths_rows.json, 512 tokens each, one row per forward; g from Γ in bf16 on the
fp32 target; g statistics with threshold 0.01 (`importance_pass`, `g_stats`, saved fp16); alive = g > 0.01 anywhere in
either block; the adversary `Q.pgd_kl_rows` at seed 0, 20 sign steps of 0.1, fp32 components, sources saved fp32 under
E40's keys. The one change is `--decomp`.

Departure, memory only: E40's H100 held Γ beside the fp32 components (~64 GB with the adversary). An L40 has 45 GiB, so
as in E131's measure stages the g pass runs with Γ and components in bf16 (the readouts cast V to Γ's dtype, so g is
the same), g is held on the host (E131's `SparseG`), Γ is freed, and the masked KL and the adversary run in fp32 from
those g. E131's control (c) measured that path on the old decomposition against E40's stored masked KL (-7.9e-5).

Stages, one job each (`--stage 0`, then `--stage 1` once 0 has passed); each writes results/w0_refs_stage<n>.json:
  0  The control, on the OLD decomposition, alone. Stops at the first miss with the reason printed (the rules, with
     their sources, are in results/predictions_amend1.json; its `alive` and `gstats` rules are replaced by
     results/predictions_amend1b.json). Expected: the stored statistics give alive 494598 and the fresh index differs
     from them in at most 989 columns summed over sites; mean_g within 5.7e-4 (L1 ratio) of the stored fp16 files
     (active reported, no bar); masked KL 0.14167 FineWeb and
     0.05886 maths; adversary step-0 KL 0.0918 and 0.0374 (the same random sources); final KL near 1.73 and 4.24.
  1  The same on NEW, written to results/large/w0_refs/p-9707d6ff/. Refused unless stage 0's file says pass. Expected:
     alive total positive (E131: ~6,200 active subcomponents per token on new against ~8,400 on old, so it may sit
     below 494598), not a bar. Stops only on alive total 0 or a non-finite tensor or number.
Files, named as E40's: <dir>/gstats/W0_{fineweb,maths}.safetensors, <dir>/alive.safetensors,
<dir>/sources/G_{fineweb,maths}_seed0.safetensors. Stamped metadata decomp=<run id>.

Usage (repo root): sbatch cluster/qwen_new_decomp_w0_refs.sbatch 0, then ... 1.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import qwen_maps
from commute import qwen_vpd as Q
from commute.jobs import hard_exit, load_experiment, require_filed, write_result
from commute.qwen_sft import load, save
from commute.qwen_teacher import g_stats, importance_pass

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = RESULTS / "large" / "w0_refs"
PRED = RESULTS / "predictions_amend1.json"
PRED_B = RESULTS / "predictions_amend1b.json"  # 2026-11-09 addendum: replaces the `alive` and `gstats` rules
WEB = HERE.parent / "qwen_adversary" / "results" / "tokens.json"
MATHS = HERE.parent / "qwen_sft" / "results" / "maths_rows.json"
STORED = Path("/mnt/nw/home/e.agnew/commute-data/results-large/qwen_sft/nowarmup")
STAGE_DECOMP = {0: "p-e9fcffd5", 1: "p-9707d6ff"}
BLOCKS = ("fineweb", "maths")  # g statistics in E40's order
ADV_BLOCKS = ("maths", "fineweb")  # the plan's deliverable first; each adversary starts afresh from the seed
DEV, F32, BF16 = "cuda", torch.float32, torch.bfloat16
T0 = time.time()


@dataclass(frozen=True)
class Config:
    n_rows: int = 128
    row_len: int = 512
    alive_g: float = 0.01  # E40's Config.alive_g
    seed: int = 0
    adv_steps: int = Q.PGD_STEPS  # 20, E40's Config.adv_steps


CFG = Config()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:8.1f}s] {msg}", flush=True)


def result_path(stage: int) -> Path:
    return RESULTS / f"w0_refs_stage{stage}.json"


def block_rows(block: str) -> list[list[int]]:
    rows = [r[: CFG.row_len] for r in json.loads((WEB if block == "fineweb" else MATHS).read_text())["rows"][: CFG.n_rows]]
    assert len(rows) == CFG.n_rows and all(len(r) == CFG.row_len for r in rows), block
    return rows


# ----------------------------------------------------------------------------- the rules (pure)


def all_finite(tensors: dict[str, Tensor]) -> bool:
    return all(bool(torch.isfinite(v).all()) for v in tensors.values())


def l1_ratio(fresh: dict[str, Tensor], stored: dict[str, Tensor], stat: str) -> float:
    """Σ|fresh - stored| / Σ stored over every subcomponent of every site. NaN or inf when the denominator is 0 or a
    value is not finite, which fails the `<=` of the rule: such a reading is not a match."""
    keys = [f"{s}.{stat}" for s in Q.SITES]
    diff = sum(float((fresh[k].double() - stored[k].double()).abs().sum()) for k in keys)
    total = sum(float(stored[k].double().sum()) for k in keys)
    return diff / total if total > 0 else math.nan


def within(x: float, ref: float, tol: float) -> dict[str, Any]:
    """Relative band around a filed, non-zero reference. A non-finite x fails."""
    assert ref != 0, "a filed reference of 0 is a bug in the predictions file"
    rel = x / ref - 1
    return {"read": x, "ref": ref, "rel": rel, "tol": tol, "pass": math.isfinite(rel) and abs(rel) <= tol}


def check_alive(idx: dict[str, Tensor], stored: dict[str, Tensor], rule: dict[str, Any]) -> dict[str, Any]:
    """Addendum b: the stored statistics must give exactly `expected` columns; the fresh index passes when its
    symmetric difference from the stored one, summed over sites, is at most `max_symmetric_difference_share` x expected.
    The site names and sizes of what differs are returned so that a miss can be read without a rerun."""
    expected = rule["expected"]
    bar = rule["max_symmetric_difference_share"] * expected
    n, n_stored = sum(len(v) for v in idx.values()), sum(len(v) for v in stored.values())
    sym = 0
    for s in Q.SITES:
        a, b = idx[s].cpu(), stored[s].cpu()
        sym += len(a) + len(b) - 2 * int(torch.isin(a, b).sum())
    differ = {s: (len(idx[s]), len(stored[s])) for s in Q.SITES if not torch.equal(idx[s].cpu(), stored[s].cpu())}
    return {"count": n, "count_from_stored_stats": n_stored, "expected": expected, "symmetric_difference": sym,
            "bar": bar, "sites_differing": differ, "pass": n_stored == expected and sym <= bar}


def check_stats(fresh: dict[str, dict[str, Tensor]], stored: dict[str, dict[str, Tensor]], tol: float) -> dict[str, Any]:
    """Addendum b: mean_g is barred at `tol`; active is returned with no bar."""
    out: dict[str, Any] = {"tol": tol}
    for block in BLOCKS:
        for stat in ("mean_g", "active"):
            out[f"{block}.{stat}"] = l1_ratio(fresh[block], stored[block], stat)
    out["pass"] = all(out[f"{b}.mean_g"] <= tol for b in BLOCKS)
    return out


def source_agreement(fresh: dict[str, Tensor], stored: dict[str, Tensor]) -> dict[str, float]:
    """Reported, no rule: the mean absolute difference of the final sources and the share of entries within one step
    (0.1) of the stored ones."""
    d = torch.cat([(fresh[k].cpu() - stored[k].cpu()).abs().flatten() for k in stored])
    return {"mean_abs": float(d.mean()), "share_within_0.1": float((d <= 0.1).float().mean())}


def per_kind(idx: dict[str, Tensor]) -> dict[str, int]:
    out = {str(k): 0 for k in Q.KINDS}
    for s, v in idx.items():
        out[Q.parse_site(s)[1]] += len(v)
    return out


def stage1_stop(alive_total: int, finite: bool) -> str | None:
    """Stage 1's only stops; the alive total is otherwise not a bar."""
    if alive_total == 0:
        return "alive total is 0"
    return None if finite else "a tensor or number is non-finite"


# ----------------------------------------------------------------------------- the measurement (GPU)


class Apparatus:
    """E40's readings of one decomposition at W0, in two phases so that it fits an L40: `g_phase` (Γ and components in
    bf16), then `load_fp32`, `masked_kls` and `adversary` (Γ freed, components fp32, g from the host)."""

    def __init__(self, decomp: Path) -> None:
        Q.fp32_matmuls()
        assert torch.cuda.get_device_properties(0).total_memory > 44 * 2**30, "needs a 46 GB card (L40) or larger"
        self.e131 = load_experiment("qwen_new_decomp", "run")
        self.decomp, self.model = decomp, Q.load_target(DEV, F32)
        self.rows = {b: [torch.tensor([r], device=DEV) for r in block_rows(b)] for b in BLOCKS}
        self.g: dict[str, Any] = {}
        self.comps: Any = None

    def g_phase(self) -> tuple[dict[str, dict[str, Tensor]], dict[str, Tensor]]:
        """g statistics per block and the alive index per site (E40's run.py:796-803); g kept for the fp32 phase."""
        comps, gamma = Q.load_components(self.decomp, DEV, BF16), Q.load_gamma(self.decomp, DEV, BF16)
        log(f"loaded for the g phase; GPU {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
        stats, seen = {}, {}
        for block in BLOCKS:
            t = time.time()
            p = importance_pass(self.model, self.rows[block], gamma, comps, CFG.alive_g, masked=False)
            stats[block] = g_stats(p)
            seen = {s: seen.get(s, p["any"][s]) | p["any"][s] for s in Q.SITES}
            self.g[block] = self.e131.g_rows(self.model, self.rows[block], gamma, comps)
            log(f"{block}: g statistics and g in {time.time() - t:.0f} s; GPU peak {_peak()}")
        del gamma, comps
        torch.cuda.empty_cache()
        return stats, {s: seen[s].nonzero().flatten() for s in Q.SITES}

    def load_fp32(self) -> None:
        self.comps = Q.load_components(self.decomp, DEV, F32)
        log(f"components fp32 loaded; GPU {torch.cuda.memory_allocated() / 2**30:.1f} GiB")

    def masked_kl(self, block: str) -> float:
        """Mean KL(clean ‖ masked at m = g, Δ off) over the rows; all rows are equally long, so E40's per-token mean."""
        kl = self.e131.ci_kls(self.model, self.rows[block], self.comps, self.g[block])
        return sum(kl) / len(kl)

    def adversary(self, block: str) -> tuple[float, float, dict[str, Tensor]]:
        """(final KL, step-0 KL, final sources) of one start; `qwen_teacher.adversary`'s call with g from the host."""
        steps: list[str] = []
        final: dict[str, Tensor] = {}

        def step_log(msg: str) -> None:
            steps.append(msg)
            log(f"{block}: {msg}")

        rows, g = self.rows[block], self.g[block]
        kl = Q.pgd_kl_rows(self.model, rows, self.comps, lambda i: (g[i].dense(), self.e131.clean_logits(self.model, rows[i])),
                           CFG.seed, steps=CFG.adv_steps, log=step_log, final=final)
        return kl, float(steps[0].rsplit(" ", 1)[1]), final


def _peak() -> str:
    return f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB"


# ----------------------------------------------------------------------------- stages


def load_rules() -> dict[str, Any]:
    """predictions_amend1.json's rules, with the `alive` and `gstats` keys replaced by the addendum's."""
    return json.loads(PRED.read_text())["rules"] | json.loads(PRED_B.read_text())["rules"]


def stored_stats() -> dict[str, dict[str, Tensor]]:
    return {b: load(STORED / "gstats" / f"W0_{b}.safetensors") for b in BLOCKS}


def run_stage(stage: int, decomp: Path, rules: dict[str, Any], app_cls: Any) -> int:
    did = Q.decomp_id(decomp)
    assert did == STAGE_DECOMP[stage], f"stage {stage} reads {STAGE_DECOMP[stage]}, --decomp is {did}"
    path, out_dir, meta = result_path(stage), OUT / did, {"decomp": did}
    out: dict[str, Any] = {"stage": stage, "blocks": {b: {} for b in BLOCKS}, "rows": {"n": CFG.n_rows, "len": CFG.row_len}}

    def write() -> None:
        write_result(path, out, decomp)

    def stop(why: str) -> int:
        out["pass"], out["stopped"] = False, why
        write()
        log(f"STOP: {why}")
        return 1

    app = app_cls(decomp)
    stats, idx = app.g_phase()
    for block in BLOCKS:
        save(out_dir / "gstats" / f"W0_{block}.safetensors", stats[block], torch.float16, False, log, meta)
    save(out_dir / "alive.safetensors", idx, torch.int64, False, log, meta)
    total = sum(len(v) for v in idx.values())
    out["alive"] = {"per_kind": per_kind(idx), "total": total, "of": sum(Q.C[Q.parse_site(s)[1]] for s in Q.SITES)}
    log(f"alive {out['alive']}")
    finite = all(all_finite(stats[b]) for b in BLOCKS)
    write()
    if stage == 0:
        old = stored_stats()
        out["alive_control"] = check_alive(idx, qwen_maps.alive_from_stats(old["maths"], old["fineweb"]), rules["alive"])
        out["stats_control"] = check_stats(stats, old, rules["gstats"]["tol"])
        log(f"control, alive: {out['alive_control']}")
        log(f"control, g statistics: {out['stats_control']}")
        write()
        if not out["alive_control"]["pass"]:
            return stop(f"alive set differs: {out['alive_control']}")
        if not out["stats_control"]["pass"]:
            return stop(f"g statistics differ: {out['stats_control']}")
    elif (why := stage1_stop(total, finite)) is not None:
        return stop(why)
    if not finite:
        return stop("non-finite g statistics")

    app.load_fp32()
    for block in BLOCKS:
        t = time.time()
        kl = app.masked_kl(block)
        out["blocks"][block]["masked_kl"] = kl
        log(f"{block}: masked KL {kl:.6f} ({time.time() - t:.0f} s)")
        if stage == 0:
            out["blocks"][block]["masked_kl_control"] = within(kl, rules["masked_kl"]["refs"][block], rules["masked_kl"]["tol"])
            log(f"control, masked KL {block}: {out['blocks'][block]['masked_kl_control']}")
        write()
    if stage == 0 and (bad := [b for b in BLOCKS if not out["blocks"][b]["masked_kl_control"]["pass"]]):
        return stop(f"masked KL off on {bad}")
    if not all(math.isfinite(out["blocks"][b]["masked_kl"]) for b in BLOCKS):
        return stop("non-finite masked KL")

    for block in ADV_BLOCKS:
        t = time.time()
        kl, step0, sources = app.adversary(block)
        if not math.isfinite(kl) or not all_finite(sources):
            return stop(f"{block}: non-finite adversary KL or sources")
        save(out_dir / "sources" / f"G_{block}_seed{CFG.seed}.safetensors", sources, F32, False, log, meta)
        rec = out["blocks"][block]
        rec |= {"adv": kl, "step0": step0, "adv_s": time.time() - t}
        log(f"{block}: adversary seed {CFG.seed}: {kl:.4f} nats, step-0 {step0:.4f} ({rec['adv_s']:.0f} s, GPU peak {_peak()})")
        if stage == 0:
            rec["step0_control"] = within(step0, rules["adv_step0"]["refs"][block], rules["adv_step0"]["tol"])
            rec["adv_control"] = within(kl, rules["adv_final"]["refs"][block], rules["adv_final"]["tol"])
            rec["sources_agreement"] = source_agreement(sources, load(STORED / "sources" / f"G_{block}_seed{CFG.seed}.safetensors"))
            log(f"control, {block}: step-0 {rec['step0_control']}; final {rec['adv_control']}; sources {rec['sources_agreement']}")
        write()
        if stage == 0 and not (rec["step0_control"]["pass"] and rec["adv_control"]["pass"]):
            return stop(f"{block}: adversary control missed: step-0 {rec['step0_control']}, final {rec['adv_control']}")

    out["pass"] = True
    write()
    log(f"STAGE {stage} DONE")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", type=int, required=True, choices=sorted(STAGE_DECOMP))
    ap.add_argument("--decomp", type=Path, required=True)
    a = ap.parse_args(argv)
    require_filed(PRED)
    require_filed(PRED_B)
    if a.stage == 1:
        p = result_path(0)
        if not (p.is_file() and json.loads(p.read_text()).get("pass") is True):
            sys.exit(f"stage 1 needs a passed stage 0: {p}")
    return run_stage(a.stage, a.decomp, load_rules(), Apparatus)


if __name__ == "__main__":
    hard_exit(main())
