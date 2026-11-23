"""E276 (cell P2.66): the confirmatory text panel. On twelve NEW continued-text tasks, does the update's effective rank
still rank the tasks as blind raters' reuse tiers do once the base model's loss on the trained tokens is partialled
out? E272's wide panel showed it post hoc on eight text tasks (partial tau 0.87); here the test is filed first. Plan
reports/plans/2026-11-23_plan_confirm-panel.md; settings, the test and expected values results/predictions.json (the
run refuses unless it is committed unchanged, and train refuses until the raters' tiers are filed in it).

E272's trainer, floor, KL ladder, selection rule and erank are imported, not copied: the recipe is E272's filed
settings block (experiments/convergence_panel/results/predictions.json), checked against the sha256 filed here.

  train TASK   GPU: the task's rows (tasks.lm_task on a streamed HF dataset); base loss of M on the held rows
               (nats per byte and per token, every trained token); E272's Gaussian floor; seeds 0-2 on the rungs
               -> results/train/TASK.json, states in results/large/states/
  select       CPU, after every train job: E272's filed rule (KL "hi", +-10%, held gain over the floor); the drops,
               and the test's critical partial tau and power at the kept tasks -> results/{selection,dropped}.json
  read         CPU: the control first (E272's lower_s0_step128 re-read, a zero update), then per kept task its three
               seeds' erank and one Gaussian at seed 0's per-matrix norms -> results/read.json
  score        CPU: the filed test -> results/summary.json

    sbatch --comment="E276 stage: train italian" cluster/confirm_panel.sbatch train italian       (each of the 12)
    ~/commute-agents/fleet/cpu-run python3 experiments/confirm_panel/run.py select
    ~/commute-agents/fleet/cpu-run -c 4 --mem 8G -t 60 python3 experiments/confirm_panel/run.py read
    ~/commute-agents/fleet/cpu-run python3 experiments/confirm_panel/run.py score
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from commute import finetune as FT
from commute import panel as PN
from commute import realistic as R
from commute import steering as ST
from commute import update_spectrum as US
from commute import vpd
from commute.jobs import (
    config_from_settings,
    hard_exit,
    load_experiment,
    require_filed,
    write_stage,
)
from commute.open_pair import jsonable

RUN: Any = load_experiment("convergence_panel", "run")  # E272: Config, fine_tune, floor, rows_kl, load_dW, put
SC: Any = load_experiment("convergence_panel", "score")  # E272: tau, partial_tau, perm_p, taub_rows, critical_tau, power
BL: Any = load_experiment("convergence_panel", "base_loss")  # E272's post-hoc check: byte_lengths, loss
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED, LARGE = RESULTS / "predictions.json", RESULTS / "large"


@dataclass(frozen=True)
class Config:
    tasks: dict[str, dict[str, str]]  # name -> {"lang"} (Wikipedia, realistic.WIKI) or {"path", "revision", "field", ...}
    e272_settings_sha256: str
    control_state: str
    control_erank: float
    control_tol: float
    gaussian_band: tuple[float, float]

    def __post_init__(self) -> None:
        assert 10 <= len(self.tasks) <= 12 and not set(self.tasks) & set(RUN.TASKS), "10-12 tasks, none of E272's 18"
        assert all(("lang" in s) != ("path" in s) for s in self.tasks.values()), self.tasks
        object.__setattr__(self, "gaussian_band", tuple(self.gaussian_band))


def load() -> tuple[Config, Any, dict[str, Any]]:
    """(this run's Config, E272's Config, the filed rules)."""
    require_filed(PRED)
    P = json.loads(PRED.read_text())
    C = config_from_settings(Config, P["settings"])
    s272 = json.loads(RUN.PRED.read_text())["settings"]
    got = hashlib.sha256(json.dumps(s272, sort_keys=True).encode()).hexdigest()
    assert got == C.e272_settings_sha256, f"E272's settings block changed ({got}): the recipe is not the filed one"
    return C, config_from_settings(RUN.Config, s272), P


def texts(spec: dict[str, str]) -> Iterator[str]:
    """A task's texts in the dataset's order, streamed (Wikipedia through E272's own loader)."""
    if "lang" in spec:
        yield from R.wiki_texts(spec["lang"])
        return
    from datasets import load_dataset

    kw: dict[str, Any] = {"name": spec["config"]} if "config" in spec else {}
    for ex in load_dataset(spec["path"], split=spec.get("split", "train"), streaming=True, revision=spec["revision"], **kw):
        if (t := (ex[spec["field"]] or "").strip()):
            yield t


def build(net: ST.Net, C: Config, C272: Any, name: str) -> PN.Task:
    """E272's continued-text task (tasks.lm_task: every token after position 0 trained, E211's read, held gain at the
    read positions) on the task's texts packed as E272's Wikipedia rows (tasks.packed)."""
    tok = vpd.tokenizer()
    V = int(net.run(torch.zeros(1, 2, dtype=torch.long))[0].shape[-1])
    starts = R.word_starts(tok.convert_ids_to_tokens(list(range(len(tok)))), V)
    train, held = RUN.TK.packed(texts(C.tasks[name]), C272, tok)
    return RUN.TK.lm_task(net, C272, name, train, held, starts)


def train(net: ST.Net, C: Config, C272: Any, P: dict[str, Any], name: str) -> None:
    assert P.get("raters"), "the raters' tiers are not filed in predictions.json: no training before them"
    path = RESULTS / "train" / f"{name}.json"
    out: dict[str, Any] = json.loads(path.read_text()) if path.is_file() else {"experiment": "E276", "task": name, "seeds": {}}
    t, kl, tok = build(net, C, C272, name), RUN.rows_kl(C272), vpd.tokenizer()
    w0 = PN.target_logp(net, t.held, t.held_mask, None)
    RUN.put(path, out, "rows", {"train": list(t.train.shape), "held": list(t.held.shape), "read": len(t.read.toks), "w0_logp": w0})
    if "base_loss" not in out:  # the data-only baseline, before any update exists: every trained token of the held rows
        RUN.put(path, out, "base_loss", BL.loss(net, t.held, PN.full_mask(t.held, 1), BL.byte_lengths(tok)))
    if "floor" not in out:
        RUN.put(path, out, "floor", RUN.floor(net, C272, t, w0))
    for seed in C272.seeds:
        if str(seed) not in out["seeds"]:
            out["seeds"][str(seed)] = RUN.fine_tune(net, C272, t, seed, w0, kl, states=LARGE / "states")
            RUN.put(path, out, "seeds", out["seeds"])


def partial_rows(x: tuple[float, ...], z: list[float], Y: np.ndarray) -> np.ndarray:
    """score.partial_tau of x and each untied row of Y given z, vectorised (score.taub_rows); a row whose partial is
    undefined (y tied perfectly with z) reads -inf, never called, as score.perm_p counts a nan."""
    txz, tyz = SC.tau(list(x), z), SC.taub_rows(tuple(z), Y)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = (SC.taub_rows(x, Y) - txz * tyz) / np.sqrt((1 - txz**2) * (1 - tyz**2))
    return np.where(np.isfinite(v), v, -np.inf)


def test_power(x: tuple[float, ...], z: list[float], rules: dict[str, Any]) -> dict[str, Any]:
    """The filed test's critical partial tau (the smallest partial a one-sided permutation test calls at alpha, over
    random orderings of an untied reading, tier and loss held: score.critical_tau's rule) and its power at a reading =
    tier code + sigma N(0, 1) with mean raw tau-b 0.3 or 0.5 (score.power's model, loss held); beside them the raw
    tau-b's critical value and power (score)."""
    rng, n, draws = np.random.default_rng(rules["seed"]), len(x), rules["n_crit"]
    null = np.sort(partial_rows(x, z, np.argsort(rng.random((draws, n)), 1).astype(float)))
    p_ge = (len(null) - np.searchsorted(null, null - 1e-9, "left")) / len(null)
    called = null[(p_ge <= rules["alpha"]) & np.isfinite(null)]
    crit = float(called.min()) if called.size else math.nan  # nan: no ordering reaches alpha (or the loss ties the tier)
    xa, E = np.asarray(x), rng.standard_normal((draws, n))
    out: dict[str, Any] = {"n": n, "critical_partial": crit, "tau_tier_minus_loss": SC.tau(list(x), z)}
    for target in (0.3, 0.5):
        lo, hi = 0.01, 50.0
        for _ in range(30):
            sig = math.sqrt(lo * hi)
            lo, hi = (sig, hi) if SC.taub_rows(x, xa + sig * E).mean() > target else (lo, sig)
        out[f"power_partial_at_tau_{target}"] = float((partial_rows(x, z, xa + math.sqrt(lo * hi) * E) >= crit - 1e-9).mean())
    raw = SC.critical_tau(tuple(sorted(x)), rules["alpha"])
    return out | {"critical_raw": raw, "power_raw": {str(v): SC.power(tuple(sorted(x)), v, raw) for v in (0.3, 0.5)}}


def tiers(P: dict[str, Any]) -> dict[str, float]:
    return SC.raters(P)["consensus"]


def minus_loss(names: list[str]) -> dict[str, float]:
    """Minus M's loss per byte on each task's trained held tokens (oriented as E272's: higher = the reuse side)."""
    return {t: -json.loads((RESULTS / "train" / f"{t}.json").read_text())["base_loss"]["nats_per_byte"] for t in names}


def select(C: Config, C272: Any, P: dict[str, Any]) -> None:
    """E272's filed rule (PN.select, KL ladder, +-10%, held gain over each task's floor) over this panel's tasks, then the
    test's critical partial tau and power at the tasks kept at KL "hi", written before any read."""
    if (RESULTS / "read.json").is_file():
        sys.exit("read.json exists: the drops are fixed; select is not rerun")
    runs, floors, dropped = {}, {}, {}
    for name in C.tasks:
        p = RESULTS / "train" / f"{name}.json"
        if not p.is_file() or len(json.loads(p.read_text())["seeds"]) < len(C272.seeds):
            dropped[name] = "no complete train file"
            continue
        d = json.loads(p.read_text())
        runs[name], floors[name] = {s: v[1:] for s, v in d["seeds"].items()}, d["floor"]["floor"]
    sel = PN.select(runs, "kl", PN.rung_ladder(C272.kl_lo, C272.kl_n), C272.match_tol, floors)
    kept = sorted(sel["hi"]["tasks"])
    tier = tiers(P)
    x, z = tuple(tier[t] for t in kept), list(minus_loss(kept).values())
    power = test_power(x, z, P["rules"]) if len(kept) >= P["rules"]["min_tasks"] else None
    t_xz = SC.tau(list(x), z)  # redteam-lead 2026-11-23: |tau(tier, loss)| over collinear_max reads "not separable"
    write_stage(RESULTS / "selection.json", jsonable(sel | {"floors": floors}))
    write_stage(RESULTS / "dropped.json", jsonable({"written": time.time(), "dropped": dropped, "kept": kept, "out": sel["hi"]["out"],
                                                    "target": sel["hi"]["target"], "tau_tier_minus_loss": t_xz, "power": power}))  # fmt: skip
    RUN.log(f"[select] KL hi {sel['hi']['target']:.4g}: kept {kept}; out {sel['hi']['out']}; dropped {dropped}; tau(tier, -loss) {t_xz:.3f}; power {power}")


def erank(dW: ST.DW) -> float:
    """update_spectrum.state's erank; nan where it is undefined (a zero update)."""
    v = US.state(dW)["erank"]
    return math.nan if v is None else v


def read(C: Config) -> None:
    """The control first (STOP on a miss), then every kept task's three seeds and one Gaussian at seed 0's norms."""
    path, d = RESULTS / "read.json", json.loads((RESULTS / "dropped.json").read_text())
    out: dict[str, Any] = json.loads(path.read_text()) if path.is_file() else {"experiment": "E276", "tasks": {}}
    if "control" not in out:
        ctl = RUN.load_dW(C.control_state)
        got, zero = erank(ctl), erank({k: torch.zeros_like(v) for k, v in ctl.items()})
        RUN.put(path, out, "control", {"erank": got, "saved": C.control_erank, "zero_update_erank": zero})
        if not abs(got - C.control_erank) <= C.control_tol or not math.isnan(zero):
            write_stage(path, out | {"stopped": "control missed"})
            sys.exit(f"STOP: control missed: erank {got} vs saved {C.control_erank}, zero update {zero}: tell the lead")
    sel = json.loads((RESULTS / "selection.json").read_text())["hi"]["tasks"]
    for name in d["kept"]:
        if name in out["tasks"]:
            continue
        states = {s: erank(RUN.load_dW(st["file"])) for s, st in sel[name].items()}
        g = erank(FT.random_direction(RUN.load_dW(sel[name]["0"]["file"]), 0))
        rec = {"erank": states, "gaussian": g, "kl": {s: st["kl"] for s, st in sel[name].items()}}
        out["tasks"][name] = rec
        write_stage(path, out)
        RUN.log(f"[read {name}] {json.dumps(rec)}")
        lo, hi = C.gaussian_band
        if not (lo <= g <= hi and g > max(states.values())):
            write_stage(path, out | {"stopped": f"{name}: Gaussian erank {g} outside {C.gaussian_band} or not above the trained"})
            sys.exit(f"STOP: {name}'s Gaussian erank {g} outside {C.gaussian_band} or under a trained state: tell the lead")


def score(P: dict[str, Any]) -> dict[str, Any]:
    """The filed test on the kept tasks: erank's partial Kendall tau-b with the consensus tiers given minus the base
    loss per byte, one-sided task-permutation p (score.perm_p), and the call; raw taus and leave-one-out beside it."""
    d, rd, R_ = json.loads((RESULTS / "dropped.json").read_text()), json.loads((RESULTS / "read.json").read_text()), P["rules"]
    if (RESULTS / "read.json").stat().st_mtime < d["written"]:
        sys.exit("read.json is older than dropped.json: the drops were not fixed before the read")
    tier, ts = tiers(P), [t for t in d["kept"] if t in rd["tasks"]]
    if len(ts) < R_["min_tasks"]:
        return {"tasks": ts, "call": f"too few tasks ({len(ts)} < {R_['min_tasks']})"}
    x, z = [tier[t] for t in ts], list(minus_loss(ts).values())
    y = [float(np.mean(list(rd["tasks"][t]["erank"].values()))) for t in ts]
    part = lambda v: SC.partial_tau(x, v, z)
    o = {"tasks": ts, "erank": dict(zip(ts, y)), "minus_loss": dict(zip(ts, z)), "tier": dict(zip(ts, x)),
         "partial": part(y), "p": SC.perm_p(part, y, R_["n_perm"], R_["seed"]), "tau_erank_tier": SC.tau(x, y),
         "tau_minus_loss_tier": SC.tau(x, z), "tau_erank_minus_loss": SC.tau(y, z), "power_at_select": d["power"],
         "leave_one_out_partial": [SC.partial_tau(*[[v[i] for i in range(len(ts)) if i != j] for v in (x, y, z)]) for j in range(len(ts))]}  # fmt: skip
    ok = SC.finite(o["partial"]) and abs(o["tau_minus_loss_tier"]) < 1 - 1e-9  # a loss tied with the tier: rounding, not a partial
    o["call"] = "positive" if ok and o["partial"] > 0 and o["p"] <= R_["alpha"] else "not positive" if ok and o["partial"] <= 0 else "no call"
    if o["call"] != "positive" and abs(o["tau_minus_loss_tier"]) > R_["collinear_max"]:  # redteam-lead, 2026-11-23
        o["call"] = "tier and loss collinear: not separable"
    return o


def main() -> int:
    args = sys.argv[1:]
    C, C272, P = load()
    if args == ["select"]:
        select(C, C272, P)
    elif args == ["read"]:
        read(C)
    elif args == ["score"]:
        o = score(P)
        write_stage(RESULTS / "summary.json", jsonable(o))
        print(json.dumps(jsonable(o), indent=1))
    elif len(args) == 2 and args[0] == "train" and args[1] in C.tasks:
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        train(ST.Net("cuda"), C, C272, P, args[1])
        print(f"STAGE train {args[1]} DONE; peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
    else:
        sys.exit("usage: run.py train TASK | select | read | score")
    return 0


if __name__ == "__main__":
    hard_exit(main())
