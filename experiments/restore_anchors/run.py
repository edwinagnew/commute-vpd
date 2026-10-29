"""restore_anchors (E50): in the real fine-tuned model, how much of E12's copy and random-map gains do W0's
subcomponents carry at the key position? Plan: reports/plans/2026-10-29_plan_restore-anchors.md, section "Filed (E50)"
(option A); rules, thresholds and expected values: results/predictions.json (filed before this code ran, never edited;
the run refuses to start without it).

States: E46's stage 0a rebuilds of E12's four tables (ident32, ident32b copy; rand32, rand32b random maps), seeds 0 and
1, read from experiments/regating_table/results/large/ and checked by the sha256 its stage0a.json records; do-nothing
placement (released subcomponents, Gamma frozen). Every reading is the full model (W0 + dW, no Gamma, no masks) on the
97 held-out windows of a table at the key position `half`. Restoring a set X (`recruitment.restored_at`, E38 Amendment
3): at the key position, each matrix's output gains sum_{c in X} ((x0 - x) . v_c) u_c, x0 its input in W0's pass.
rho(X) = (T1 - lo with X restored) / G, G = T1 - T0; the sets S, gold, D, every_alive and random_00..19 are defined in
predictions.json["sets"]. E46's stand-in ratio (stage0b.json, P0.2) is copied beside each state.

One job: the controls first on all eight states (C1 T0 and the path, C2 T1 against E46, R0 empty set, R1 every set
restored in a W0 pass, R2 the whole key-position output set to W0's); any miss: results written, FAIL logged, exit 1.
Then the grid -> results/restore_anchors.json (rewritten after each state; the reading at the end). Every scored
number is printed to the log (stdout and results/large/run.log, emptied at the start of each run).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import recruit as R
from commute import recruitment as RT
from commute import regating as G
from commute.faithfulness import model_state
from commute.signatures import top_by

HERE = Path(__file__).resolve().parent
RESULTS, EXP = HERE / "results", HERE.parent
LARGE = RESULTS / "large"  # may be a symlink into ~/commute-data/results-large/
PRED, OUT = RESULTS / "predictions.json", RESULTS / "restore_anchors.json"
E46 = EXP / "regating_table" / "results"
ALIVE = EXP / "transport_test" / "results" / "large" / "alive.pt"
SEEDS = ("s0", "s1")


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with (LARGE / "run.log").open("a") as f:
        f.write(line + "\n")


def _finite(o: Any) -> Any:
    if isinstance(o, float) and not math.isfinite(o):
        return str(o)
    if isinstance(o, dict):
        return {str(k): _finite(v) for k, v in o.items()}
    return [_finite(v) for v in o] if isinstance(o, list | tuple) else o


def dump(obj: dict[str, Any]) -> None:
    OUT.write_text(json.dumps(_finite(obj), indent=1, sort_keys=True))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def e46_files(s0a: dict[str, Any], tables: list[str]) -> dict[str, Path]:
    """Every E46 window and state file, checked against stage0a.json's sha256 (stop rule 2: exit before any reading)."""
    g = s0a["groups"]
    want = {f"windows_{n}": g["e12/build"]["sha256"][n] for n in tables}
    want |= {f"{n}_{s}": g[f"e12/{n}"]["states"][s]["sha256"] for n in tables for s in SEEDS}
    paths = {k: E46 / "large" / f"{k}.pt" for k in want}
    bad = [str(p) for k, p in paths.items() if not p.is_file() or sha256(p) != want[k]]
    if bad:
        sys.exit(f"missing or not the file E46's stage 0a recorded: {bad}; rerun regating_table/run.py --stages 0a")
    return paths


@dataclass
class W0Read:
    """A table's W0 readings on its held-out windows: T0, logits (dropped after the controls), key-position io."""

    T0: float
    ref: list[Tensor] | None
    io: RT.IO


class Run:
    def __init__(self, device: str, P: dict[str, Any], files: dict[str, Path]) -> None:
        from commute import vpd

        self.P, self.files = P, files
        self.model = vpd.load_component_model(device)
        self.W0, self.comps = vpd.target_weights(self.model), vpd.component_weights(self.model)
        self.keys = sorted(self.comps)
        alive = torch.load(ALIVE, weights_only=False)["alive"]
        self.alive = {k: v.cpu().bool() for k, v in alive.items()}

    def weights(self, dW: dict[str, Tensor]) -> dict[str, Tensor]:
        return {k: w + dW[k].to(w.device, w.dtype) if k in dW else w for k, w in self.W0.items()}

    def state(self, name: str) -> dict[str, Tensor]:
        return torch.load(self.files[name], weights_only=False)


def lo_ref(run: Run, W: dict[str, Tensor], t: G.Task) -> tuple[float, list[Tensor]]:
    """lo of the full model at W on the held-out windows, and its logits there."""
    with model_state(run.model, target=W):
        ref = RT.window_logits(run.model, t.held.tokens)
        return RT.readout(run.model, t.held.tokens, t.half, t.target, ref)["lo"], ref


def restored(run: Run, W: dict[str, Tensor], sel: R.Sel, io: RT.IO, t: G.Task, ref: list[Tensor],
             whole: bool = False) -> float:
    return RT.restored_at(run.model, W, run.comps, sel, io, t.held.tokens, t.half, t.target, ref, whole)["lo"]


# ----------------------------------------------------------------------------- the sets


@torch.no_grad()
def gold_contrast(run: Run, values: dict[str, list[int]]) -> tuple[dict[str, dict[str, Tensor]], dict[str, int]]:
    """Per table, W0's contrast at Pile positions whose next token is one of its values (recruit.ContrastAcc), and
    the number of such positions."""
    from commute.data import pile_rows
    from commute.vpd import compute_ci

    rows = pile_rows(*run.P["pile"], log=log)
    masks = {}
    for n, v in values.items():
        masks[n] = torch.zeros_like(rows, dtype=torch.bool)
        masks[n][:, :-1] = torch.isin(rows[:, 1:], torch.tensor(v, dtype=rows.dtype))
    accs = {n: R.ContrastAcc() for n in values}
    for s in range(0, rows.shape[0], 8):
        ci = compute_ci(run.model, rows[s : s + 8], chunk=8)
        for n, acc in accs.items():
            acc.add(ci.g, masks[n][s : s + 8])
        del ci
    return {n: a.result(run.alive) for n, a in accs.items()}, {n: a.n_on for n, a in accs.items()}


def sets_of(run: Run, t: G.Task, dW: dict[str, Tensor], contrast: dict[str, Tensor],
            io0_train: RT.IO) -> tuple[dict[str, R.Sel], dict[str, Any]]:
    """S, gold, D, every_alive and random_00..19 for one state, and D's short matrices."""
    assert G.THR == 0.3, "E38's switched-on threshold"
    S = R.switched(G.shift_of(run.model, run.W0, dW, t, run.alive)[0], G.THR)
    with model_state(run.model, target=run.weights(dW)):
        io1 = RT.trigger_io(run.model, run.keys, t.train.tokens, t.half)
        grads = RT.trigger_grads(run.model, run.keys, t.train.tokens, t.half, t.target)
    e = RT.take_back(run.comps, io0_train, io1, grads)
    D, short = RT.top_per_matrix(e, run.alive, S, {k: int(v.numel()) for k, v in S.items()})
    sets = {"S": S, "gold": top_by(contrast, run.alive, R.size(S)), "D": D,
            "every_alive": {k: v.nonzero().flatten() for k, v in run.alive.items()}}
    sets |= {f"random_{i:02d}": R.random_like(S, run.alive, i) for i in range(run.P["n_random"])}
    return sets, {"D_short": sorted(k for k, v in short.items() if v), "J_gold_S": R.jaccard(sets["gold"], S)}


# ----------------------------------------------------------------------------- the controls


def task_control(run: Run, t: G.Task, e46: dict[str, Any]) -> tuple[W0Read, dict[str, Any]]:
    """C1: T0 against E46's stage 0a and the full-model path against route_cells'."""
    c = run.P["controls"]
    T0, ref0 = lo_ref(run, run.W0, t)
    with model_state(run.model, target=run.W0):
        io = RT.trigger_io(run.model, run.keys, t.held.tokens, t.half)
    rec = {"T0": T0, "T0_e46": e46["W0"]["got"]["T0"], "T0_route": t.base["T"]}
    rec["ok"] = abs(T0 - rec["T0_e46"]) <= c["C1_T0_tol"] and abs(T0 - rec["T0_route"]) <= c["C1_path_tol"]
    log(f"[control {t.name}] C1 T0 {T0:.4f} E46 {rec['T0_e46']:.4f} route_cells {rec['T0_route']:.4f} | {'ok' if rec['ok'] else 'FAIL'}")
    return W0Read(T0, ref0, io), rec


def state_control(run: Run, t: G.Task, w0: W0Read, dW: dict[str, Tensor], sets: dict[str, R.Sel],
                  e46: dict[str, Any]) -> dict[str, Any]:
    """C2, R0, R1 (the W0 control: every set restored in a W0 pass), R2 for one state."""
    c = run.P["controls"]
    assert w0.ref is not None
    W1 = run.weights(dW)
    T1, ref1 = lo_ref(run, W1, t)
    empty = {k: torch.zeros(0, dtype=torch.long) for k in run.keys}
    r1 = {k: restored(run, run.W0, sel, w0.io, t, w0.ref) - w0.T0 for k, sel in sets.items()}
    gain_e46 = e46["T1"] - w0.T0
    rec: dict[str, Any] = {"T1": T1, "T1_e46": e46["T1"], "R0": restored(run, W1, empty, w0.io, t, ref1) - T1,
                           "R1_max": max(abs(v) for v in r1.values()), "R1": r1,
                           "R2": restored(run, W1, empty, w0.io, t, ref1, whole=True) - w0.T0}
    rec["pass"] = {"C2": abs(T1 - e46["T1"]) <= c["C2_T1_rel_tol"] * abs(gain_e46), "R0": abs(rec["R0"]) <= c["R0_empty_tol"],
                   "R1": rec["R1_max"] <= c["R1_w0_tol"], "R2": abs(rec["R2"]) <= c["R2_whole_tol"]}
    rec["ok"] = all(rec["pass"].values())
    return rec


# ----------------------------------------------------------------------------- the grid


def grid_state(run: Run, t: G.Task, w0: W0Read, dW: dict[str, Tensor], sets: dict[str, R.Sel]) -> dict[str, Any]:
    W1 = run.weights(dW)
    T1, ref1 = lo_ref(run, W1, t)
    gain = T1 - w0.T0
    g = gain if gain >= run.P["thresholds"]["min_gain"] else 0.0  # unscored state: every share nan
    nats = {k: T1 - restored(run, W1, sel, w0.io, t, ref1) for k, sel in sets.items()}
    del ref1
    rho = {k: RT.share(T1, T1 - v, g) for k, v in nats.items()}
    rnd = [v for k, v in rho.items() if k.startswith("random_")]
    mean = sum(rnd) / len(rnd)
    return {"T0": w0.T0, "T1": T1, "gain": gain, "scorable": gain >= run.P["thresholds"]["min_gain"], "nats": nats,
            "rho": rho, "random_mean": mean, "random_max": max(rnd),
            "random_sd": (sum((x - mean) ** 2 for x in rnd) / (len(rnd) - 1)) ** 0.5,
            "n": {k: R.size(v) for k, v in sets.items() if not k.startswith("random_")},
            "members": {k: {m: v.tolist() for m, v in sorted(sets[k].items()) if v.numel()} for k in ("S", "gold", "D")}}


def log_state(key: str, r: dict[str, Any]) -> None:
    rho, nats = r["rho"], r["nats"]
    log(f"[grid {key}] T0 {r['T0']:.4f} T1 {r['T1']:.4f} G {r['gain']:.4f} scorable {r['scorable']} | n {json.dumps(r['n'])} "
        f"| D short {r['D_short']} | J(gold, S) {r['J_gold_S']:.4f} | stand-in ratio (E46) {r['stand_in_ratio']}")
    log(f"[grid {key}] rho " + " ".join(f"{k} {rho[k]:.4f} ({nats[k]:.3f} nats)" for k in ("every_alive", "S", "gold", "D"))
        + f" | random mean {r['random_mean']:.4f} max {r['random_max']:.4f} sd {r['random_sd']:.4f}")
    log(f"[grid {key}] random rho {json.dumps([round(v, 4) for k, v in sorted(rho.items()) if k.startswith('random_')])}")


# ----------------------------------------------------------------------------- the filed rules (pure)


def scorable(r: dict[str, Any], X: str, P: dict[str, Any]) -> bool:
    if not r["scorable"] or not math.isfinite(r["rho"][X]):
        return False
    return X != "S" or (r["n"]["S"] > 0 and r["random_max"] <= P["thresholds"]["random_bites"])


def x_verdict(states: dict[str, dict[str, Any]], X: str, P: dict[str, Any]) -> dict[str, Any]:
    """The separation rule on X for both seeds: per seed the worst-pair gap and its class, then X's verdict."""
    cls, gap = P["states"]["classes"], P["thresholds"]["gap"]
    per: dict[str, Any] = {}
    for s in SEEDS:
        recs = {n: states[f"{n}/{s}"] for n in cls}
        if not all(scorable(r, X, P) for r in recs.values()):
            per[s] = {"class": "unscored"}
            continue
        copy = [r["rho"][X] for n, r in recs.items() if cls[n] == "copy"]
        rnd = [r["rho"][X] for n, r in recs.items() if cls[n] == "random"]
        g, rev = min(copy) - max(rnd), max(copy) - min(rnd)
        per[s] = {"gap": g, "reverse_gap": rev, "class": "separates" if g >= gap else "reversed" if rev <= -gap else "no"}
    c = [v["class"] for v in per.values()]
    verdict = ("unscored" if "unscored" in c else "separates" if set(c) == {"separates"}
               else "reversed" if set(c) == {"reversed"} else "does not separate")
    return {"seeds": per, "verdict": verdict}


def reading(states: dict[str, dict[str, Any]], P: dict[str, Any]) -> dict[str, Any]:
    v = {X: x_verdict(states, X, P) for X in ("every_alive", "S")}
    vs = {X: r["verdict"] for X, r in v.items()}
    sep = [X for X, x in vs.items() if x == "separates"]
    low = all(scorable(r, "every_alive", P) and r["rho"]["every_alive"] <= P["thresholds"]["low"] for r in states.values())
    if sep and "reversed" in vs.values():
        out = "conflict"
    elif sep:
        out = f"separates ({', '.join(sep)})"
    elif low:
        out = "toys unreadable"
    elif "reversed" in vs.values():
        out = "reversed"
    elif vs["every_alive"] == "unscored":
        out = "unscored"
    else:
        out = "does not separate"
    return {"verdicts": v, "reading": out}


def reported(states: dict[str, dict[str, Any]], P: dict[str, Any]) -> dict[str, Any]:
    """Each state's reported numbers against predictions.json's 'expected' (they change no verdict)."""
    ex, cls, out = P["expected"], P["states"]["classes"], {}
    for key, r in states.items():
        rho, e = r["rho"], ex[cls[key.split("/")[0]]]
        checks = ({"every_alive": rho["every_alive"] >= e["every_alive_min"], "S": rho["S"] >= e["S_min"],
                   "gold": rho["gold"] >= e["gold_min"], "D": rho["D"] >= rho["S"] + e["D_min_minus_S"]}
                  if "every_alive_min" in e else
                  {"every_alive": rho["every_alive"] <= e["every_alive_max"], "S": rho["S"] <= e["S_max"],
                   "gold": rho["gold"] <= e["gold_max"], "D": rho["D"] <= e["D_max"]})
        checks["random"] = (abs(r["random_mean"]) <= ex["random_sets"]["abs_mean_max"]
                            and r["random_max"] <= ex["random_sets"]["max_max"])
        out[key] = checks
    return out


# ----------------------------------------------------------------------------- main


def tasks_and_controls(run: Run, out: dict[str, Any]) -> dict[str, Any]:
    """Every table's W0 reading and C1, then each state's sets and C2-R2; exit 1 at the end on any miss."""
    g0a = json.loads((E46 / "stage0a.json").read_text())["groups"]
    win = {n: torch.load(run.files[f"windows_{n}"], weights_only=False) for n in run.P["states"]["classes"]}
    contrast, n_on = gold_contrast(run, {n: list(w["task"].values) for n, w in win.items()})
    out["gold_positions"] = n_on
    log(f"[gold] Pile positions before each table's values {json.dumps(n_on)}")
    keep: dict[str, Any] = {}
    for n, w in win.items():
        t = G.prepare(run.model, n, w["train"], w["heldout"], w["table"])
        w0, rec = task_control(run, t, g0a[f"e12/{n}"])
        t.base = {"T": t.base["T"]}  # route_cells' gate pattern (~1 GB) is read by C1 only
        with model_state(run.model, target=run.W0):
            io0_train = RT.trigger_io(run.model, run.keys, t.train.tokens, t.half)
        out["controls"][n] = rec
        for s in SEEDS:
            dW = run.state(f"{n}_{s}")
            sets, info = sets_of(run, t, dW, contrast[n], io0_train)
            c = state_control(run, t, w0, dW, sets, g0a[f"e12/{n}"]["states"][s])
            out["controls"][f"{n}/{s}"] = c
            keep[f"{n}/{s}"] = (t, w0, sets, info)
            log(f"[control {n}/{s}] T1 {c['T1']:.4f} E46 {c['T1_e46']:.4f} | |S| {R.size(sets['S'])} | R0 {c['R0']:.2e} | "
                f"R1 max {c['R1_max']:.2e} | R2 {c['R2']:.2e} | {json.dumps(c['pass'])} {'ok' if c['ok'] else 'FAIL'}")
            dump(out)
        w0.ref = None  # the grid needs W0's key-position io only
        del io0_train
    out["controls_pass"] = all(v["ok"] for v in out["controls"].values())
    dump(out)
    if not out["controls_pass"]:
        log("[controls] FAIL: stop and report")
        sys.exit(1)
    log("[controls] PASS")
    return keep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if not PRED.is_file():
        sys.exit(f"file the predictions first: {PRED}")
    P = json.loads(PRED.read_text())
    files = e46_files(json.loads((E46 / "stage0a.json").read_text()), list(P["states"]["classes"]))
    LARGE.resolve().mkdir(parents=True, exist_ok=True)
    (LARGE / "run.log").write_text("")
    t0 = time.time()
    run = Run(args.device, P, files)
    log(f"[start] on {torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'}")
    out: dict[str, Any] = {"predictions": str(PRED.relative_to(EXP.parent)), "controls": {}, "states": {}}
    keep = tasks_and_controls(run, out)
    ratio = json.loads((E46 / "stage0b.json").read_text())["score"]["P0.2"]["rows"]
    for key, (t, w0, sets, info) in keep.items():
        rec = {**grid_state(run, t, w0, run.state(key.replace("/", "_")), sets), **info,
               "stand_in_ratio": ratio[key]["ratio"]}
        out["states"][key] = rec
        dump(out)
        log_state(key, rec)
    out.update(reading(out["states"], P), reported=reported(out["states"], P))
    dump(out)
    log(f"[reading] {out['reading']} | " + " | ".join(f"{X} {json.dumps(_finite(v))}" for X, v in out["verdicts"].items()))
    log(f"[reported] {json.dumps(out['reported'])}")
    log(f"[done] ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
