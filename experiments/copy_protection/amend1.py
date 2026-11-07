"""E130 Amendment 1: does the step-20 copy c1's reversal (placement multiplies its excess, m 1.58 on Pile val rows 0-255)
hold on a second, disjoint row set? Plan reports/plans/2026-11-07_plan_e130-copy-protection.md, section "Amendment 1";
filing results/predictions_amend1.json (the run refuses unless it is committed unchanged and its `config` equals ACFG).

E130's check (run.py `read`: harness_pgd, one seed-0 adversary per 128-row block, 20 steps, each ratio over W0 in this
job) on Pile val rows 256-511 in two blocks of 128, batches of 8 (`harness.on_rows`, E122's set B, the same rows,
blocking and path). Units in ORDER, controls first, each written as it ends (`STAGE n DONE`):
  W0          control: mean block KL within 1e-3 of E122's set-B reading (E39 A3's)
  c1          the copy placed (alive U', V'): control, ratio within 1e-3 of E122's tt_c1|B
  nothing_c1  do-nothing: c1's fine-tune ft_c1.pt in Delta, harness.at(ctx, ft, None)
  c1|delta    the copy's change S_c in Delta (run.py `read`)
  c2, c2|delta  the comparison copy, the same two reads; c2 placed is a control against E122's tt_c2|B
The rule is run.py's `multiplier` and `verdict` unchanged (same f, same edges), read overall and per block.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch

from commute import harness
from commute.faithfulness import harness_pgd
from commute.jobs import hard_exit, load_experiment, require_filed

R = load_experiment("copy_protection", "run")  # E130: CFG (f, edges, PGD), multiplier, verdict, read, copy_state, paths
PRED, OUT = R.HERE / "results" / "predictions_amend1.json", R.HERE / "results" / "copy_protection_amend1.json"
ORDER = ("W0", "c1", "nothing_c1", "c1|delta", "c2", "c2|delta")
OUTCOME = {"no protection": "reversal holds", "partial": "reversal does not replicate",
           "recipe-like": "reversal does not replicate", "unreadable": "unreadable", "not run": "not run"}


@dataclass(frozen=True)
class Config:
    row_start: int = 256  # Pile val rows 256-511: E122's set B (damage_crossed/rules.py SETS["B"])
    rows: int = 256
    w0_ref: float = 0.6301094591617584  # W0 mean block KL on these rows (damage_crossed/rules.py CFG.w0_ref["B"], E39 A3)
    placed_refs: tuple[tuple[str, float], ...] = (("c1", 1.4443102469308349),  # damage_crossed.json score.ratios["tt_c1|B"]
                                                  ("c2", 1.596624645597247))  # score.ratios["tt_c2|B"]


ACFG = Config()


# ----------------------------------------------------------------------------- rules (pure), as filed


def ratios(kl: dict[str, list[float]], part: int | None) -> dict[str, float]:
    """Each state's KL over W0's: summed over both blocks (part None) or block `part` alone. A non-positive W0 -> nan."""
    pick = (lambda v: sum(v)) if part is None else (lambda v: v[part])
    d = pick(kl["W0"]) if "W0" in kl else math.nan
    return {k: (pick(v) / d if d > 0 else math.nan) for k, v in kl.items() if k != "W0"}


def read_copy(r: dict[str, float], c: str) -> dict[str, Any]:
    """E130's R1 and R2 on one copy (run.py multiplier, verdict); the in-Delta excess over the do-nothing's if read."""
    label, m = R.multiplier(r.get(c), r.get(f"{c}|delta"))
    v = "not run" if label == "not run" else R.verdict(m)
    ind, nothing = r.get(f"{c}|delta"), r.get(f"nothing_{c}")
    share = None if ind is None or nothing is None or nothing == 1 else (ind - 1) / (nothing - 1)
    return {"reading": label, "m": m, "verdict": v, "in-Delta share of do-nothing excess": share,
            "above its do-nothing": None if ind is None or nothing is None else ind > nothing}


def score(kl: dict[str, list[float]]) -> dict[str, Any]:
    """Overall and per block, each copy read; the outcome is c1's overall verdict mapped by OUTCOME."""
    res: dict[str, Any] = {}
    for part, key in ((None, "overall"), (0, "block rows 256-383"), (1, "block rows 384-511")):
        r = ratios(kl, part) if kl.get("W0") else {}
        res[key] = {"ratio": r, "copies": {c: read_copy(r, c) for c in R.COPIES}}
    res["outcome"] = OUTCOME[res["overall"]["copies"]["c1"]["verdict"]]
    return res


# ----------------------------------------------------------------------------- the job


def dump(out: dict[str, Any]) -> None:
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True, default=float))


def halt(out: dict[str, Any], why: str) -> None:
    out["stopped"] = why
    dump(out)
    R.log(f"STOP: {why}")
    hard_exit(1)


def read_nothing(ctx: Any, c: str) -> list[float]:
    """PGD KL per block with copy c's fine-tune in Delta (do-nothing), as E122's nothing_<c> (damage_crossed run.py)."""
    ft = torch.load(R.DC / "large" / f"ft_{c}.pt", weights_only=True)
    with harness.at(ctx, ft, None):
        return [harness_pgd(ctx.model, b, R.CFG.pgd_steps, R.CFG.adv_seed) for b in ctx.blocks]


def control(out: dict[str, Any], name: str) -> None:
    """Stop unless W0's mean block KL and each copy's placed ratio match E122's set-B readings (relative repro_tol)."""
    refs, kl = dict(ACFG.placed_refs), out["kl"]
    if name not in ("W0", *refs):
        return
    got = sum(kl["W0"]) / 2 if name == "W0" else sum(kl[name]) / sum(kl["W0"])
    want = ACFG.w0_ref if name == "W0" else refs[name]
    out["controls"][name] = {"read": got, "ref": want, "tol": R.CFG.repro_tol}
    if not (math.isfinite(got) and abs(got / want - 1) <= R.CFG.repro_tol):
        halt(out, f"{name} reads {got}, its reference {want} (tolerance {R.CFG.repro_tol} relative)")


def load(out: dict[str, Any]) -> Any:
    token = json.loads((R.TRANSPORT / "windows.json").read_text())["o_token_id"]
    ctx = harness.load("cuda", R.TRANSPORT / "large" / "alive.pt", R.TRANSPORT / "large" / "windows.pt", token,
                       R.CFG.ls_tol, 0, R.CFG.pgd_steps, (), R.log)
    ctx = harness.on_rows(ctx, "val", ACFG.row_start, ACFG.rows, (), R.log)
    out["jobs"].append({"card": torch.cuda.get_device_name(), "start": time.strftime("%Y-%m-%dT%H:%M:%S")})
    if "L40" not in out["jobs"][-1]["card"]:
        halt(out, f"the references are L40 readings; this job runs on {out['jobs'][-1]['card']}")
    uv0 = torch.load(R.UV0, weights_only=False, map_location="cpu")
    assert all(torch.equal(ctx.uv_cpu[k][i], uv0[k][i]) for k in uv0 for i in (0, 1)), "alive U, V differ from E121's"
    assert len(ctx.blocks) == 2 and all(len(b) == 16 for b in ctx.blocks), "expected two blocks of 16 batches of 8"
    if "W0" in out["kl"]:  # a later job: W0's first block must read as the first job's did
        v = harness_pgd(ctx.model, ctx.blocks[0], R.CFG.pgd_steps, R.CFG.adv_seed)
        if not abs(v / out["kl"]["W0"][0] - 1) <= R.CFG.rerun_tol:
            halt(out, f"W0's first block reads {v}, the first job {out['kl']['W0'][0]}")
    return ctx


def main() -> None:
    require_filed(PRED)
    assert json.loads(PRED.read_text())["config"] == json.loads(json.dumps(asdict(ACFG))), "predictions were filed for another config"
    out = json.loads(OUT.read_text()) if OUT.is_file() else {"jobs": [], "kl": {}, "seconds": {}, "controls": {}}
    assert "stopped" not in out and not out.get("done"), f"{OUT}: stopped or done; move it aside to start again"
    t0, ctx, cache = time.time(), load(out), {}
    for i, name in enumerate(ORDER):
        if name in out["kl"]:
            continue
        if time.time() - t0 + R.CFG.pgd_s > R.CFG.budget_s:
            R.log(f"[budget] {name} after {time.time() - t0:.0f} s: exit 3 for a resubmission")
            dump(out)
            hard_exit(3)
        t1 = time.time()
        out["kl"][name] = read_nothing(ctx, "c1") if name == "nothing_c1" else R.read(ctx, cache, name)
        out["score"], out["seconds"][name] = score(out["kl"]), time.time() - t1
        dump(out)
        R.log(f"[{name}] blocks {out['kl'][name]} ratio {out['score']['overall']['ratio'].get(name)} "
              f"per block {[out['score'][k]['ratio'].get(name) for k in ('block rows 256-383', 'block rows 384-511')]}")
        control(out, name)
        dump(out)
        R.log(f"STAGE {i} DONE {name} ({out['seconds'][name]:.0f} s)")
    out["done"] = True
    dump(out)
    R.log(f"[score] {json.dumps(out['score'])}")
    hard_exit(0)


if __name__ == "__main__":
    main()
