"""E130 copy_protection: is teacher training's 3-4x division of the excess a constant of the recipe, or does it depend on
the function copied? Plan reports/plans/2026-11-07_plan_e130-copy-protection.md; design Fable's
(~/commute-data/scratch/regating-lead/fable-e128/advice.md, written as "E128"); filing results/predictions.json (the run
refuses unless it is committed unchanged and its `config` equals CFG).

E122's two other lowercase copies by the same recipe (Adam 1e-5, batch 8, 1,000 steps, from W0's U, V): c1 = tt_c1.pt
(copy of the Adam fine-tune at step 20), c2 = tt_c2.pt (copy of the rank-8 LoRA at step 2,400). Each read by E34's check
as E127 Amendment 2 read its copies (experiments/ls_start_faith/amend2.py, which stays as it ran): harness_pgd on Pile
val rows 0-255, one seed-0 adversary per 128-row block, 20 steps, each ratio over W0 in this job:
  W0         control, mean block KL within 1e-3 of E127's
  c          the copy free (alive subcomponents at U', V'): control, within 1e-3 of E122's stored ratio
  c|delta    the copy's own change S_c = U'V'^T - UV^T placed in Delta: harness.at(ctx, S_c, None)
  cosine     report only: input-weighted and Frobenius cosine of each S_c to its fine-tune and to ft_adam300
Units run in ORDER, controls first, each written as it ends (`STAGE n DONE`); exit 3 when the next one might not end
inside budget_s (fleet/resubmit --code 3), and a later job rereads W0's first block first.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import harness, weighted
from commute.faithfulness import harness_pgd
from commute.jobs import hard_exit, require_filed
from commute.saved_states import offsets_state

HERE, EXP = Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]
PRED, OUT = HERE / "results" / "predictions.json", HERE / "results" / "copy_protection.json"
DC, TRANSPORT = EXP / "damage_crossed" / "results", EXP / "transport_test" / "results"
UV0 = EXP / "teacher_training" / "results" / "large" / "amend8" / "amend8c" / "uv0.pt"  # E121's alive U, V (amend2's E121.A8C)
COPIES = ("c1", "c2")  # files DC/large/tt_<c>.pt, fine-tunes DC/large/ft_<c>.pt, sha256 DC/damage_crossed.json copies.tt_<c>
ORDER = ("W0", *COPIES, *(f"{c}|delta" for c in COPIES), "cosine")


@dataclass(frozen=True)
class Config:
    pgd_steps: int = 20  # as amend2.py (E127 run.py: teacher_training/run.py pile_pgd_steps)
    adv_seed: int = 0
    pile_rows: int = 256
    ls_tol: float = 1e-6
    moments_batch: int = 8  # E127 run.py moments_batch, for the cosine
    floor: float = 0.20877548446862537  # f: L40 W0 PGD floor, seeds 0-2 (teacher_training_amend8c.json W0.pgd_floor)
    w0_ref: float = 0.6917431652545929  # W0 mean block KL (ls_start_faith_amend2.json controls.W0)
    copy_refs: tuple[tuple[str, float], ...] = (("c1", 2.8418210654761875),  # damage_crossed.json score.ratios["tt_c1|A"]
                                                ("c2", 1.5340505291251594))  # score.ratios["tt_c2|A"]
    nothing_refs: tuple[tuple[str, float], ...] = (("c1", 3.4094833456256115),  # score.ratios["nothing_c1|A"]
                                                   ("c2", 3.7389993015826213))  # score.ratios["nothing_c2|A"]
    recipe_max: float = 0.5  # m at or below: "recipe-like" (Fable's filed rule)
    none_min: float = 0.8  # m at or above: "no protection"
    repro_tol: float = 1e-3  # relative, amend2.py repro_tol (as amend1.py, E121 Amendment 1, E122)
    rerun_tol: float = 1e-6  # relative: W0's first block reread by a later job
    budget_s: float = 1680.0  # a unit starts only if it ends inside 28 min of the 30-min job
    pgd_s: float = 145.0  # one PGD read: 127-131 s (ls_start_faith_amend2.json .seconds)
    cos_s: float = 90.0  # moments + grams: E127's `weights` unit took 34 s with more work (ls_start_faith.json .seconds)


CFG = Config()


# ----------------------------------------------------------------------------- rules (pure), as filed


def multiplier(free: float | None, ind: float | None) -> tuple[str, float | None]:
    """m = (free - 1)/(in Delta - 1), readable only if both excesses are at least f (inclusive), as amend2.py."""
    if free is None or ind is None:
        return "not run", None
    if not (math.isfinite(free) and math.isfinite(ind)):
        return "invalid", None
    if free - 1 < CFG.floor:
        return "free excess within the floor", None
    if ind - 1 < CFG.floor:
        return "in-Delta excess within the floor", None
    return "readable", (free - 1) / (ind - 1)


def verdict(m: float | None) -> str:
    """Per copy, edges inclusive: m <= 0.5 recipe-like, m >= 0.8 no protection, between partial; no m: unreadable."""
    if m is None:
        return "unreadable"
    return "recipe-like" if m <= CFG.recipe_max else "no protection" if m >= CFG.none_min else "partial"


def headline(v: dict[str, str]) -> str:
    """Any copy 'no protection' -> function-dependent; else a copy not read -> unreadable; both recipe-like -> recipe
    constant; else partial."""
    if "no protection" in v.values():
        return "function-dependent"
    if len(v) < len(COPIES) or "unreadable" in v.values():
        return "unreadable"
    return "recipe constant" if all(x == "recipe-like" for x in v.values()) else "partial"


def score(kl: dict[str, list[float]]) -> dict[str, Any]:
    d = sum(kl["W0"]) if "W0" in kl else math.nan
    r = {k: (sum(v) / d if d > 0 else math.nan) for k, v in kl.items() if k != "W0"}
    res: dict[str, Any] = {"ratio": r, "copies": {}}
    for c in COPIES:
        label, m = multiplier(r.get(c), r.get(f"{c}|delta"))
        ind, nothing = r.get(f"{c}|delta"), dict(CFG.nothing_refs)[c]
        res["copies"][c] = {"reading": label, "m": m, "verdict": verdict(m) if label != "not run" else "not run",
                            "in-Delta share of do-nothing excess": None if ind is None else (ind - 1) / (nothing - 1),
                            "above its do-nothing": None if ind is None else ind > nothing}
    v = {c: x["verdict"] for c, x in res["copies"].items() if x["verdict"] != "not run"}
    res["headline"] = headline(v) + (", regularity broken" if any(x["above its do-nothing"] for x in res["copies"].values()) else "")
    return res


# ----------------------------------------------------------------------------- the job


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def dump(out: dict[str, Any]) -> None:
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True, default=float))


def halt(out: dict[str, Any], why: str) -> None:
    out["stopped"] = why
    dump(out)
    log(f"STOP: {why}")
    hard_exit(1)


def copy_state(ctx: Any, cache: dict[str, Any], c: str) -> tuple[dict[str, Tensor], Any]:
    """(S_c, (U', V')) of copy c from E122's bf16 offsets, after its sha256 matches E122's record."""
    if c not in cache:
        path = DC / "large" / f"tt_{c}.pt"
        want = json.loads((DC / "damage_crossed.json").read_text())["copies"][f"tt_{c}"]["sha256"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == want, f"{path}: sha256 differs from E122's"
        cache[c] = offsets_state(path, ctx.uv_cpu)
    return cache[c]


def read(ctx: Any, cache: dict[str, Any], name: str) -> list[float]:
    """PGD KL per block at W0, at the copy (S_c, (U', V')), or with S_c in Delta (S_c, None)."""
    s = None if name == "W0" else copy_state(ctx, cache, name.split("|")[0])
    st = None if s is None else (s[0], None) if name.endswith("|delta") else s
    with harness.at(ctx, *st) if st else contextlib.nullcontext():
        return [harness_pgd(ctx.model, b, CFG.pgd_steps, CFG.adv_seed) for b in ctx.blocks]


def cosine(ctx: Any, cache: dict[str, Any]) -> dict[str, Any]:
    """Report only: cos(S_c, own fine-tune) and cos(S_c, ft_adam300), input-weighted on the check's rows and Frobenius."""
    rows = torch.cat([torch.cat(b) for b in ctx.blocks])
    sigma = weighted.input_moments(ctx.model, rows, CFG.moments_batch)
    adam300 = torch.load(DC / "large" / "ft_adam300.pt", weights_only=True)
    rec: dict[str, Any] = {}
    for c in COPIES:
        vec = {"S": copy_state(ctx, cache, c)[0], "ft": torch.load(DC / "large" / f"ft_{c}.pt", weights_only=True),
               "ft_adam300": adam300}
        rec[c] = {}
        for kind, sg in (("weighted", sigma), ("frobenius", None)):
            C = weighted.cosines(weighted.gram_of(vec, sg, ctx.device))
            rec[c][kind] = {"to own fine-tune": float(C[0, 1]), "to ft_adam300": float(C[0, 2])}
    return rec


def control(out: dict[str, Any], name: str) -> None:
    """Stop unless W0 and each copy's free ratio match their stored values (relative repro_tol)."""
    kl, refs = out["kl"], dict(CFG.copy_refs)
    if name not in ("W0", *refs):
        return
    got = sum(kl["W0"]) / 2 if name == "W0" else sum(kl[name]) / sum(kl["W0"])
    want = CFG.w0_ref if name == "W0" else refs[name]
    out["controls"][name] = {"read": got, "ref": want, "tol": CFG.repro_tol}
    if not abs(got / want - 1) <= CFG.repro_tol:
        halt(out, f"{name} reads {got}, its reference {want} (tolerance {CFG.repro_tol} relative)")


def main() -> None:
    require_filed(PRED)
    assert json.loads(PRED.read_text())["config"] == json.loads(json.dumps(asdict(CFG))), "predictions were filed for another config"
    out = json.loads(OUT.read_text()) if OUT.is_file() else {"jobs": [], "kl": {}, "seconds": {}, "controls": {}}
    assert "stopped" not in out and not out.get("done"), f"{OUT}: stopped or done; move it aside to start again"
    t0 = time.time()
    token = json.loads((TRANSPORT / "windows.json").read_text())["o_token_id"]
    ctx = harness.load("cuda", TRANSPORT / "large" / "alive.pt", TRANSPORT / "large" / "windows.pt", token,
                       CFG.ls_tol, CFG.pile_rows, CFG.pgd_steps, (), log)
    out["jobs"].append({"card": torch.cuda.get_device_name(), "start": time.strftime("%Y-%m-%dT%H:%M:%S")})
    if "L40" not in out["jobs"][-1]["card"]:
        halt(out, f"the references are L40 readings; this job runs on {out['jobs'][-1]['card']}")
    uv0 = torch.load(UV0, weights_only=False, map_location="cpu")
    assert all(torch.equal(ctx.uv_cpu[k][i], uv0[k][i]) for k in uv0 for i in (0, 1)), "alive U, V differ from E121's"
    if "W0" in out["kl"]:  # a later job: W0's first block must read as the first job's did
        v = harness_pgd(ctx.model, ctx.blocks[0], CFG.pgd_steps, CFG.adv_seed)
        if not abs(v / out["kl"]["W0"][0] - 1) <= CFG.rerun_tol:
            halt(out, f"W0's first block reads {v}, the first job {out['kl']['W0'][0]}")
    cache: dict[str, Any] = {}
    for i, name in enumerate(ORDER):
        if name in out["kl"] or name in out:
            continue
        if time.time() - t0 + (CFG.cos_s if name == "cosine" else CFG.pgd_s) > CFG.budget_s:
            log(f"[budget] {name} after {time.time() - t0:.0f} s: exit 3 for a resubmission")
            dump(out)
            hard_exit(3)
        t1 = time.time()
        if name == "cosine":
            out["cosine"] = cosine(ctx, cache)
            log(f"[cosine] {json.dumps(out['cosine'])}")
        else:
            out["kl"][name] = read(ctx, cache, name)
            out["score"] = score(out["kl"])
            log(f"[{name}] blocks {out['kl'][name]} ratio {out['score']['ratio'].get(name)}")
        out["seconds"][name] = time.time() - t1
        dump(out)
        control(out, name)
        dump(out)
        log(f"STAGE {i} DONE {name} ({out['seconds'][name]:.0f} s)")
    out["done"] = True
    dump(out)
    log(f"[score] {json.dumps(out['score'])}")
    hard_exit(0)


if __name__ == "__main__":
    main()
