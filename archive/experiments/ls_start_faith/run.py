"""E127 ls_start_faith: why does the least-squares start read worse under E34's adversarial check than the update left
in Delta (do-nothing)? Plan reports/plans/2026-11-06_plan_e127-ls-start-faithfulness.md; filing results/predictions.json
(the run refuses unless it is committed unchanged and its `config` equals CFG).

Per task (L: E122's lowercase Adam step 300, E: E121's emoticon Adam step 1000), dW the fine-tune and S the weight change
of its least-squares split (S = (U + dU)(V + dV)^T - U V^T on the alive columns; S != dW: the split's remainder is
dropped), each state read by E34's check on Pile val rows 0-255 (harness_pgd: one seed-0 adversary per 128-row block,
20 steps), as a ratio over W0 read the same way in this job:
  nothing        target W0 + dW, components at W0 (control: E122 / E121 Amendment 1)
  start          target W0 + S, alive (U + dU, V + dV) (control: same)
  nothing_start  target W0 + S, components at W0: start's function, its change in Delta
  <state>|hold   nothing_start and start with every alive subcomponent's mask held at 1; ratio over W0|hold
plus random masks (faithfulness with 0 PGD steps) at W0, nothing_start, start, and the weight side: S against dW, and
the coherence of start's and two copies' per-subcomponent changes (Frobenius and input-weighted on the check's rows).
Units run in ORDER, each written as it ends; one that might not end inside budget_s exits 3 (fleet/resubmit --code 3),
and a later job rereads W0's first block first. Writes results/ls_start_faith.json.
"""

from __future__ import annotations

import contextlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import cancel, harness, weighted
from commute.faithfulness import faithfulness, harness_pgd
from commute.jobs import hard_exit, load_experiment, require_filed
from commute.saved_states import offsets_state

E121 = load_experiment("copy_gap_ladder", "run")  # ls_start, load_bf16, SAVED, LARGE (as E121 Amendment 1 read them)
HERE, EXP = Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]
PRED, OUT = HERE / "results" / "predictions.json", HERE / "results" / "ls_start_faith.json"
TRANSPORT, DC = EXP / "transport_test" / "results", EXP / "damage_crossed" / "results" / "large"
TASKS = ("L", "E")
COPIES = {"E": {"std_copy": E121.LARGE / "amend1" / "R3_o0.pt", "ls_copy": E121.LARGE / "amend1" / "R4_o1.pt"},
          "L": {"std_copy": DC / "tt_recipe.pt", "ls_copy": DC / "tt_c3.pt"}}
ORDER = ("W0", "E.nothing", "E.start", "L.nothing", "L.start", "weights", "W0|hold", "L.nothing_start", "L.nothing_start|hold",
         "L.start|hold", "random:W0", "random:L.nothing_start", "random:L.start", "E.nothing_start", "E.nothing_start|hold",
         "E.start|hold", "random:E.nothing_start", "random:E.start")


@dataclass(frozen=True)
class Config:
    pgd_steps: int = 20  # teacher_training/run.py:62 pile_pgd_steps
    adv_seed: int = 0  # harness.pile_ratios: the ratio's adversary is seed 0
    pile_rows: int = 256  # faithfulness.pile_blocks: val rows 0-255, 2 blocks of 128, batches of 8 (E122's set A)
    moments_batch: int = 8  # weighted.input_moments on the same rows, batches as the blocks
    ls_tol: float = 1e-6  # split.bases' default, E121's ls_start (copy_gap_ladder/rules.py ls_tol)
    floor: float = 0.20877548446862537  # L40 W0 PGD floor, seeds 0-2: teacher_training_amend8c.json W0.pgd_floor
    w0_ref: float = 0.6917431652545929  # W0 mean block KL: copy_gap_ladder_amend1.json kl.W0 (= damage_crossed W0|A)
    refs: tuple[tuple[str, float], ...] = (("E.nothing", 1.3900221398869133), ("E.start", 1.5732819313767334),  # amend1 ratio
                                           ("L.nothing", 4.79423381511107), ("L.start", 8.332888703134442))  # E122 c3
    repro_tol: float = 1e-3  # relative, as E121 Amendment 1 and E122
    rerun_tol: float = 1e-6  # relative: W0's first block reread by a later job (damage_crossed/rules.py rerun_tol)
    high: float = 0.75  # a share at or above this: that part carries the excess
    low: float = 0.25  # at or below this: it does not
    hold_w0_min: float = 0.05  # W0|hold's mean block KL under this: the hold leaves the adversary nothing to read
    cancel_max: float = 0.5  # coherence under this: the changes cancel (E17: least squares 0.44, training 2.8-3.5, null ~1)
    n_null: int = 3  # sign-flip draws (cancel.coherence default)
    budget_s: float = 1680.0  # a unit starts only if it ends inside 28 min of the 30-min job
    pgd_s: float = 135.0  # one PGD read: 118-132 s (damage_crossed.json .seconds, copy_gap_ladder_amend1.json .seconds)
    random_s: float = 60.0  # one random-mask read: not measured; three forwards and one backward per batch, bound
    weights_s: float = 120.0  # moments on 256 rows plus six coherences: not measured, bound


CFG = Config()


# ----------------------------------------------------------------------------- rules (pure), as filed


def finite(*xs: float) -> bool:
    return all(math.isfinite(x) for x in xs)


def ratios(kl: dict[str, list[float]], t: str) -> dict[str, float]:
    """Each read state's summed block KL over W0's (W0|hold's for a held read), same job."""
    out = {}
    for s in ("nothing", "nothing_start", "start", "nothing_start|hold", "start|hold"):
        ref = "W0|hold" if s.endswith("|hold") else "W0"
        if f"{t}.{s}" in kl and ref in kl:
            out[s] = sum(kl[f"{t}.{s}"]) / sum(kl[ref])
    return out


def source(r: dict[str, float]) -> str:
    """Reading A: is the excess (start over nothing) in start's function (nothing_start over nothing) or in the placement
    (start over nothing_start)? A part carries when it is at least the floor times its reference ratio."""
    if any(k not in r for k in ("nothing", "nothing_start", "start")):
        return "not run"
    if not finite(r["nothing"], r["nothing_start"], r["start"]):
        return "invalid"
    cf = r["nothing_start"] - r["nothing"] >= CFG.floor * r["nothing"] and r["nothing_start"] > r["nothing"]
    cp = r["start"] - r["nothing_start"] >= CFG.floor * r["nothing_start"] and r["start"] > r["nothing_start"]
    return {(True, True): "both", (True, False): "function", (False, True): "placement", (False, False): "within the floor"}[(cf, cp)]


def exposure(r: dict[str, float], src: str, w0_hold: float | None) -> tuple[str, float | None]:
    """Reading B, when the placement carries: H = share of the placement excess removed by holding the alive masks at 1."""
    if src not in ("placement", "both"):
        return "does not apply", None
    if "start|hold" not in r or "nothing_start|hold" not in r or w0_hold is None:
        return "not run", None
    if not w0_hold >= CFG.hold_w0_min:
        return "unreadable", None
    h = 1 - (r["start|hold"] - r["nothing_start|hold"]) / (r["start"] - r["nothing_start"])
    if not finite(h):
        return "invalid", None
    return ("the alive masks" if h >= CFG.high else "not the alive masks" if h <= CFG.low else "partly"), h


def random_masks(rr: dict[str, float], src: str) -> tuple[str, float | None]:
    """Reading C, when the placement carries: start's stochastic ratio over nothing_start's, minus 1, against the floor."""
    if src not in ("placement", "both"):
        return "does not apply", None
    if "start" not in rr or "nothing_start" not in rr:
        return "not run", None
    if not (finite(rr["start"], rr["nothing_start"]) and rr["nothing_start"] > 0):
        return "invalid", None
    q = rr["start"] / rr["nothing_start"] - 1
    return ("random masks see it" if q >= CFG.floor else "adversary only"), q


def cancellation(k: dict[str, float]) -> str:
    """Reading D (weight side, input-weighted coherence): do the changes cancel on the failing states only?"""
    if any(n not in k for n in ("start", "ls_copy", "std_copy")):
        return "not run"
    if not finite(*k.values()):
        return "invalid"
    c = {n: v < CFG.cancel_max for n, v in k.items()}
    return "tracks the failing states" if c["start"] and c["ls_copy"] and not c["std_copy"] else "does not track"


def score(out: dict[str, Any]) -> dict[str, Any]:
    kl, res = out["kl"], {}
    w0h = sum(kl["W0|hold"]) / 2 if "W0|hold" in kl else None
    for t in TASKS:
        r = ratios(kl, t)
        rr = {s: out["random"][f"{t}.{s}"]["stochastic_kl"] / out["random"]["W0"]["stochastic_kl"]
              for s in ("nothing_start", "start") if f"{t}.{s}" in out["random"] and "W0" in out["random"]}
        src = source(r)
        coh = {c: v["weighted"]["kappa"] for c, v in out.get("weights", {}).get(t, {}).get("coherence", {}).items()}
        res[t] = {"ratio": r, "random_ratio": rr, "source": src, "exposure": exposure(r, src, w0h),
                  "random": random_masks(rr, src), "cancellation": cancellation(coh)}
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


class States:
    """(target change, alive (U', V') or None for components at W0) of each named state; the fine-tunes; the copies."""

    def __init__(self, ctx: Any) -> None:
        self.ctx, self.start = ctx, {}

    def dW(self, t: str) -> dict[str, Tensor]:
        return E121.load_bf16(E121.SAVED["dW"]) if t == "E" else torch.load(DC / "ft_adam300.pt", weights_only=True)

    def split(self, t: str) -> tuple[dict[str, Tensor], dict[str, tuple[Tensor, Tensor]]]:
        if t not in self.start:
            if t == "E":  # E121 Amendment 1's R4_start: run.ls_start of A8c's saved dW, fp32
                init, S = E121.ls_start(self.ctx, self.dW("E"))
                self.start[t] = (S, {k: (U + init[k][0], V + init[k][1]) for k, (U, V) in self.ctx.uv_cpu.items()})
            else:  # E122's tt_c3_start, read back from its bf16 offsets as E122 read it
                self.start[t] = offsets_state(DC / "tt_c3_start.pt", self.ctx.uv_cpu)
        return self.start[t]

    def get(self, name: str) -> tuple[dict[str, Tensor], Any] | None:
        t, s = name.split("|")[0].split(".")
        if s == "nothing":
            return (torch.load(E121.LARGE / "F0.pt", weights_only=False, map_location="cpu") if t == "E" else self.dW("L")), None
        S, uv2 = self.split(t)
        return (S, None) if s == "nothing_start" else (S, uv2)


def wcoherence(uv: dict[str, tuple[Tensor, Tensor]], uv2: dict[str, tuple[Tensor, Tensor]], sigma: dict[str, Tensor],
               dev: str) -> dict[str, float]:
    """cancel.coherence's kappa = ||sum_c D_c||^2 / sum_c ||D_c||^2 with ||A||^2 = tr(A Sigma A^T), pooled over the
    matrices, and its sign-flip null (mean of n_null draws)."""
    g, num, den, null = torch.Generator().manual_seed(0), 0.0, 0.0, 0.0
    for k, (U, V) in uv.items():
        U, V, U2, V2, S = (x.double().to(dev) for x in (U, V, *uv2[k], sigma[k]))
        SV, SV2 = S @ V, S @ V2
        den += float(((U2 * U2).sum(0) * (V2 * SV2).sum(0) + (U * U).sum(0) * (V * SV).sum(0)
                      - 2 * (U2 * U).sum(0) * (V2 * SV).sum(0)).clamp_min(0).sum())
        for s in [None] + [torch.randint(0, 2, (U.shape[1],), generator=g).to(dev) * 2 - 1 for _ in range(CFG.n_null)]:
            M = cancel.signed_sum(U, V, U2, V2, s)
            v = float(((M @ S) * M).sum())
            num, null = (num + v, null) if s is None else (num, null + v / CFG.n_null)
    return {"kappa": num / den, "null": null / den} if den > 0 else {"kappa": math.nan, "null": math.nan}


def weights(ctx: Any, st: States) -> dict[str, Any]:
    """S against dW (input-weighted on the check's rows, and Frobenius), and the coherence of start and the two copies."""
    rows = torch.cat([torch.cat(b) for b in ctx.blocks])
    sigma = weighted.input_moments(ctx.model, rows, CFG.moments_batch)
    rec: dict[str, Any] = {}
    for t in TASKS:
        S, _ = st.split(t)
        vec = {"dW": st.dW(t), "S": S}
        rec[t] = {"S_vs_dW": {}, "coherence": {}}
        for name, sg in (("weighted", sigma), ("frobenius", None)):
            G = weighted.gram_of(vec, sg, ctx.device)
            rec[t]["S_vs_dW"][name] = {"cos": float(weighted.cosines(G)[1, 0]), "size": float(weighted.sizes(G)[1, 0]),
                                       "dist": float(weighted.distances(G)[1, 0])}
        uv2s = {"start": st.split(t)[1]} | {c: offsets_state(p, ctx.uv_cpu)[1] for c, p in COPIES[t].items()}
        for c, uv2 in uv2s.items():
            fro = cancel.coherence({k: (*ctx.uv_cpu[k], *uv2[k]) for k in uv2}, CFG.n_null)
            rec[t]["coherence"][c] = {"weighted": wcoherence(ctx.uv_cpu, uv2, sigma, ctx.device),
                                      "frobenius": {"kappa": fro["pooled"], "null": fro["pooled_null"]}}
    return rec


def read(ctx: Any, st: States, name: str, hold: dict[str, Tensor]) -> Any:
    """One unit: PGD KL per block, the random-mask faithfulness per block, or the weight side."""
    if name == "weights":
        return weights(ctx, st)
    rnd, key = name.startswith("random:"), name.removeprefix("random:")
    state = None if key.startswith("W0") else st.get(key)
    h = hold if key.endswith("|hold") else None
    with harness.at(ctx, *state) if state else contextlib.nullcontext():
        if rnd:
            return [faithfulness(ctx.model, b, pgd_steps=0, seed=CFG.adv_seed, log=log) for b in ctx.blocks]
        return [harness_pgd(ctx.model, b, CFG.pgd_steps, CFG.adv_seed, hold=h) for b in ctx.blocks]


def unit(ctx: Any, st: States, name: str, hold: dict[str, Tensor], out: dict[str, Any], t0: float) -> None:
    slot, key = ("random", name[7:]) if name.startswith("random:") else ("weights", None) if name == "weights" else ("kl", name)
    if (key in out[slot]) if key else slot in out:
        return
    est = CFG.random_s if slot == "random" else CFG.weights_s if slot == "weights" else CFG.pgd_s
    if time.time() - t0 + est > CFG.budget_s:
        log(f"[budget] {name} needs ~{est:.0f} s after {time.time() - t0:.0f} s: exit 3 for a resubmission")
        dump(out)
        hard_exit(3)
    t1 = time.time()
    v = read(ctx, st, name, hold)
    if slot == "random":
        v = {k: sum(b[k] * b["pgd_rows"] for b in v) / sum(b["pgd_rows"] for b in v) for k in v[0] if k != "pgd_rows"}
    if key:
        out[slot][key] = v
    else:
        out[slot] = v
    out["seconds"][name] = time.time() - t1
    out["score"] = score(out)
    log(f"[{name}] {json.dumps(v)} ({out['seconds'][name]:.0f} s)")
    dump(out)


def controls(out: dict[str, Any], name: str) -> None:
    """Stop unless W0 and the four reproduced states read as their references (relative repro_tol)."""
    kl, ref = out["kl"], dict(CFG.refs)
    got = sum(kl["W0"]) / 2 if name == "W0" else sum(kl[name]) / sum(kl["W0"])
    want = CFG.w0_ref if name == "W0" else ref[name]
    out.setdefault("controls", {})[name] = {"read": got, "ref": want}
    if not abs(got / want - 1) <= CFG.repro_tol:
        halt(out, f"{name} reads {got}, its reference {want}")


def main() -> None:
    require_filed(PRED)
    assert json.loads(PRED.read_text())["config"] == json.loads(json.dumps(asdict(CFG))), "predictions were filed for another config"
    out = json.loads(OUT.read_text()) if OUT.is_file() else {"jobs": [], "kl": {}, "random": {}, "seconds": {}}
    assert "stopped" not in out and not out.get("done"), f"{OUT}: stopped or done; move it aside to start again"
    t0 = time.time()
    token = json.loads((TRANSPORT / "windows.json").read_text())["o_token_id"]
    ctx = harness.load("cuda", TRANSPORT / "large" / "alive.pt", TRANSPORT / "large" / "windows.pt", token, CFG.ls_tol,
                       CFG.pile_rows, CFG.pgd_steps, (), log)
    out["jobs"].append({"card": torch.cuda.get_device_name(), "start": time.strftime("%Y-%m-%dT%H:%M:%S")})
    if "L40" not in out["jobs"][-1]["card"]:
        halt(out, f"the references are L40 readings; this job runs on {out['jobs'][-1]['card']}")
    uv0 = torch.load(E121.A8C / "uv0.pt", weights_only=False, map_location="cpu")
    assert all(torch.equal(ctx.uv_cpu[k][i], uv0[k][i]) for k in uv0 for i in (0, 1)), "alive U, V differ from E121's"
    assert all(a.dtype == torch.bool and a.numel() == ctx.model.module_to_c[k] for k, a in ctx.alive.items()), "alive masks"
    hold = {k: a.to(ctx.device) for k, a in ctx.alive.items()}
    if "W0" in out["kl"]:  # a later job: W0's first block must read as the first job's did
        v = harness_pgd(ctx.model, ctx.blocks[0], CFG.pgd_steps, CFG.adv_seed)
        if not abs(v / out["kl"]["W0"][0] - 1) <= CFG.rerun_tol:
            halt(out, f"W0's first block reads {v}, the first job {out['kl']['W0'][0]}")
    st = States(ctx)
    for name in ORDER:
        unit(ctx, st, name, hold, out, t0)
        if name in ("W0", *dict(CFG.refs)):
            controls(out, name)
            dump(out)
    out["done"] = True
    dump(out)
    log(f"[score] {json.dumps(out['score'])}")
    hard_exit(0)


if __name__ == "__main__":
    main()
