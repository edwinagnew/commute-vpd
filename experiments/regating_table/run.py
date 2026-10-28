"""regating_table (E46 stage 0): one table, in log-odds at the scored position, of what Gamma's new gating gives the old
subcomponents, for fine-tunes and for perturbations that learned nothing or wrote the answer directly.
Plan: reports/plans/2026-10-28_plan_regating-line-reset.md, section "Amendment 1 adopted ... the rules filed" (it
overrides section 2). Rules, thresholds and every reference value: results/predictions.json (read here; never edited).

Cells (local_faith.route_split; 0 = W0, 1 = the state): L(C, G) the masked model (Delta out), U unmasked, T the full
model. Real gain G = T1 - T0; gate gain g = L01 - L00; piece gain U1 - U0; gated piece gain L11 - L00; floor F =
max(0.1, 2 x paired SE over windows of L01 - L00). E36's code states are read per token over whole rows (spread.masked_logp).

  run.py --stages 0a   the control, alone, task by task: W0 cells, identity, every rebuilt or cached state against its
                       filed numbers -> results/stage0a.json; exit 1 at the first fail (the rest is not run)
  run.py --stages 0b   refuses unless stage0a.json passed: the table and the scored verdicts -> results/stage0b.json
States and windows: results/large/ (E12 windows and weight changes, E36 weight changes; written by 0a, checked by sha256
in 0b). Inputs (gitignored): E38's cached E18 states, E18's exact seed-0 emoticon state, transport_test and task_ladder
windows, transport_test alive.pt; Pile rows (artifacts/sets).
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

from commute import anchors as AN
from commute import perturb as PT
from commute.faithfulness import model_state

HERE = Path(__file__).resolve().parent
RESULTS, EXP = HERE / "results", HERE.parent
LARGE = RESULTS / "large"  # a symlink into ~/commute-data/results-large/
PRED = RESULTS / "predictions.json"
TRANSPORT, LADDER = EXP / "transport_test" / "results", EXP / "task_ladder" / "results" / "large"
E38_STATES = EXP / "recruitment_test" / "results" / "large" / "states"
EXACT = EXP / "reading_a_coverage" / "results" / "large" / "emoticon" / "dW_ft_step10_s0.pt"
E18_TASKS = ("emoticon", "r1_t534_v369", "r2_t359_v816")
BATCH, ROW_BATCH = 64, 8


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with (LARGE / "full.log").open("a") as f:
        f.write(line + "\n")


def _finite(o: Any) -> Any:
    if isinstance(o, float) and not math.isfinite(o):
        return str(o)
    if isinstance(o, dict):
        return {str(k): _finite(v) for k, v in o.items()}
    return [_finite(v) for v in o] if isinstance(o, list | tuple) else o


def dump(name: str, obj: dict[str, Any]) -> None:
    (RESULTS / name).write_text(json.dumps(_finite(obj), indent=1, sort_keys=True))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def load_checked(path: Path, sha: str) -> Any:
    if not path.is_file() or sha256(path) != sha:
        sys.exit(f"{path} is missing or not the file its check recorded: rerun --stages 0a")
    return torch.load(path, weights_only=False)


@dataclass
class Task:
    """A task's windows and W0's readings on its held-out windows: `base` (route_cells), `ref0` (full-model logits),
    `L00w` (per-window masked log-odds), `T0f` (full-model trigger log-odds from `full_read`)."""

    name: str
    train: Any
    held: Any
    token: int | Tensor
    base: dict[str, Any]
    ref0: list[Tensor]
    L00w: Tensor
    T0f: float

    @property
    def half(self) -> int:
        return self.held.half


class Run:
    def __init__(self, device: str) -> None:
        from commute import vpd

        self.P = json.loads(PRED.read_text())
        self.device, self.model = device, vpd.load_component_model(device)
        self.W0, self.comps0 = vpd.target_weights(self.model), vpd.component_weights(self.model)
        alive = torch.load(TRANSPORT / "large" / "alive.pt", weights_only=False)["alive"]
        self.lift = AN.lift_context(self.model, {k: v.cpu().bool() for k, v in alive.items()})
        self.vocab = int(self.model.target_model.lm_head.weight.shape[0])
        th = self.P["thresholds"]
        assert (PT.SCALE_CAP, PT.ALPHA_MAX) == (th["scale_cap"], th["alpha_max"]), "perturb.py's caps differ from the filed"

    def weights(self, dW: dict[str, Tensor]) -> dict[str, Tensor]:
        return {k: w + dW[k].to(w.device, w.dtype) if k in dW else w for k, w in self.W0.items()}

    def task(self, name: str, train: Any, held: Any, token: int | Tensor) -> Task:
        from commute.local_faith import masked_lo_windows, route_cells
        from commute.recruitment import window_logits

        m, h = self.model, held.half
        base = route_cells(m, held.tokens, h, token, BATCH)
        t = Task(name, train, held, token, base, window_logits(m, held.tokens, BATCH),
                 masked_lo_windows(m, held.tokens, h, token, base["gates"], BATCH), 0.0)
        t.T0f = full_read(m, t)[0]
        return t


# ----------------------------------------------------------------------------- readings


@torch.no_grad()
def full_read(model: Any, t: Task) -> tuple[float, float]:
    """(mean trigger log-odds of the target, mean kl_other to W0) of the full model as it stands, on held-out windows."""
    from commute.data import target_tokens
    from commute.finetune import logits_of
    from commute.local_faith import stats

    lo = kl = 0.0
    for s, r in zip(range(0, len(t.held), BATCH), t.ref0, strict=True):
        b = t.held.tokens[s : s + BATCH].to(r.device)
        st = stats(r, logits_of(model.target_model, b), t.half, target_tokens(t.token, b, t.half))
        lo, kl = lo + float(st["lo_trigger"].double().sum()), kl + float(st["kl_other"].double().sum())
    return lo / len(t.held), kl / len(t.held)


def read_state(run: Run, t: Task, dW: dict[str, Tensor]) -> tuple[float, float]:
    with model_state(run.model, target=run.weights(dW)):
        return full_read(run.model, t)


def floor_of(d: Tensor, P: dict[str, Any]) -> dict[str, float]:
    n = d.numel()
    assert n >= 2, "a paired SE needs two windows"
    se = float(d.std() / math.sqrt(n))
    return {"se": se, "n": n, "F": max(P["thresholds"]["F_min"], 2 * se), "g_windows": float(d.mean())}


def cells(run: Run, t: Task, dW: dict[str, Tensor], comps: Any = None) -> dict[str, Any]:
    """Every column of the table for target W0 + dW and components `comps` (None: released, a do-nothing state)."""
    from commute.local_faith import masked_lo_windows, route_cells, route_split

    m, held, h = run.model, t.held.tokens, t.half
    with model_state(m, target=run.weights(dW), components=comps):
        rs = route_split(m, held, h, t.token, t.base, run.comps0, BATCH)
        g1 = route_cells(m, held, h, t.token, BATCH)["gates"]
        T1f, kl = full_read(m, t)
        with model_state(m, components=run.comps0):
            L01w = masked_lo_windows(m, held, h, t.token, g1, BATCH)
    c = rs["cells"]
    G, g, piece, gated = c["T1"] - c["T0"], c["L01"] - c["L00"], c["U1"] - c["U0"], c["L11"] - c["L00"]
    return {"cells": c, "G": G, "g": g, "piece_gain": piece, "gated_piece_gain": gated, "masked_off": piece - gated,
            "kl_other": kl, "T1_full": T1f, "r": g / G if G >= run.P["thresholds"]["min_gain"] else None,
            **floor_of(L01w - t.L00w, run.P)}


@torch.no_grad()
def row_cells(run: Run, rows: Tensor, dW: dict[str, Tensor]) -> dict[str, Any]:
    """E36's per-token cells over whole rows: T (full model) and L00, L01 (masked, released components), each the mean
    next-token log-probability; the floor from the per-row L01 - L00. No U, no split."""
    from commute import spread as SP
    from commute.vpd import compute_ci

    m, T0, T1, L00, L01 = run.model, 0.0, 0.0, [], []
    for s in range(0, rows.shape[0], ROW_BATCH):
        b = rows[s : s + ROW_BATCH].to(run.device)
        ci0 = compute_ci(m, b)
        with model_state(m, target=run.weights(dW)):
            ci1 = compute_ci(m, b)
        T0, T1 = T0 + SP.target_logp(ci0.logits, b), T1 + SP.target_logp(ci1.logits, b)
        for i in range(b.shape[0]):
            L00.append(SP.masked_logp(m, b[i : i + 1], {k: v[i : i + 1] for k, v in ci0.g.items()}))
            L01.append(SP.masked_logp(m, b[i : i + 1], {k: v[i : i + 1] for k, v in ci1.g.items()}))
        del ci0, ci1
    n_pos = rows.shape[1] - 1
    l00, l01 = torch.tensor(L00, dtype=torch.float64) / n_pos, torch.tensor(L01, dtype=torch.float64) / n_pos
    c = {"T0": T0 / rows.shape[0] / n_pos, "T1": T1 / rows.shape[0] / n_pos, "L00": float(l00.mean()), "L01": float(l01.mean())}
    return {"cells": c, "G": c["T1"] - c["T0"], "g": c["L01"] - c["L00"], **floor_of(l01 - l00, run.P)}


def within(x: float, ref: float, tol: float) -> bool:
    return abs(x - ref) <= tol


# ----------------------------------------------------------------------------- stage 0a: the control


def w0_ok(t: Task, ref: dict[str, float], tol: float) -> dict[str, Any]:
    got = {"L00": t.base["L"], "U0": t.base["U"], "T0": t.base["T"]}
    return {"got": got, "ref": ref, "ok": all(within(got[k], ref[k], tol) for k in got)}


def e18_0a(run: Run, name: str) -> dict[str, Any]:
    """W0 cells; each cached state (sha256 as E38 recorded) its T1; the null; on emoticon the identity check."""
    from commute import regating as G
    from commute.local_faith import route_split

    P, th = run.P, run.P["thresholds"]
    ref = P["refs"]["e18"][name]
    t = run.task(name, *G.load_windows(name, TRANSPORT, LADDER))
    rec: dict[str, Any] = {"W0": w0_ok(t, ref["W0"], th["w0_tol"]), "states": {}}
    for s in ("s0", "s1", "null"):
        T1 = read_state(run, t, load_checked(E38_STATES / f"{name}_{s}.pt", ref["sha256"][s]))[0]
        st: dict[str, Any] = {"T1": T1, "G": T1 - t.T0f}
        if s != "null":
            st.update(T1_ref=ref["T1"][s], ok=within(T1, ref["T1"][s], th["t1_rel_tol"] * ref["gain"][s]))
        rec["states"][s] = st
    null = rec["states"]["null"]
    null["ratio"] = null["G"] / rec["states"]["s0"]["G"]
    null["ok"] = abs(null["ratio"]) <= th["null_ratio_max"] if name == "emoticon" else "reported"
    if name == "emoticon":
        with model_state(run.model, target=run.weights(load_checked(EXACT, ref["sha256"]["exact"]))):
            rs = route_split(run.model, t.held.tokens, t.half, t.token, t.base, run.comps0, BATCH)
        rec["identity"] = {"r": rs["shares"]["r"], "ref": P["refs"]["identity_r"],
                           "ok": within(rs["shares"]["r"], P["refs"]["identity_r"], th["identity_tol"])}
    rec["ok"] = rec["W0"]["ok"] and all(v["ok"] is not False for v in rec["states"].values()) and rec.get("identity", {}).get("ok", True)
    return rec


def e12_build_0a(run: Run) -> tuple[dict[str, Any], dict[str, Any]]:
    """E12's windows rebuilt; marker, keys, values and sizes must equal tasks.json exactly. Saved for 0b."""
    built, rec = AN.build_kv(run.model, run.device, log)
    ref = run.P["refs"]["e12_tasks"]
    ok = rec["marker"] == ref["marker"] and all(rec["tasks"][n] == ref["tasks"][n] for n in ref["tasks"])
    out: dict[str, Any] = {"marker": rec["marker"], "ok": ok, "sha256": {}}
    for n, w in built.items():
        torch.save(w, LARGE / f"windows_{n}.pt")
        out["sha256"][n] = sha256(LARGE / f"windows_{n}.pt")
    return built, out


def e12_0a(run: Run, name: str, w: dict[str, Any]) -> dict[str, Any]:
    """W0 cells; each seed fine-tuned to its filed scored step, lifted; T1 and r against E12's routes_lift."""
    th, ref = run.P["thresholds"], run.P["refs"]["e12"][name]
    t = run.task(name, w["train"], w["heldout"], w["table"])
    rec: dict[str, Any] = {"W0": w0_ok(t, ref["W0"], th["w0_tol"]), "states": {}}
    for s in ("s0", "s1"):
        dW, ev = AN.kv_finetune(run.model, w, int(s[1]), ref["step"][s], run.device, log)
        path = LARGE / f"{name}_{s}.pt"
        torch.save(dW, path)
        c = cells(run, t, dW, AN.lifted(run.model, run.lift, dW))
        r = c["g"] / c["G"]
        rec["states"][s] = {"sha256": sha256(path), "T1": c["cells"]["T1"], "T1_ref": ref["T1"][s], "r": r, "r_ref": ref["r"][s],
                            "p_heldout": ev["p_o"], "ok": within(c["cells"]["T1"], ref["T1"][s], th["t1_rel_tol"] * ref["gain"][s])
                            and within(r, ref["r"][s], th["e12_r_tol"])}
    rec["ok"] = rec["W0"]["ok"] and all(v["ok"] for v in rec["states"].values())
    return rec


def e36_0a(run: Run) -> dict[str, Any]:
    """E36's rows (selection record exact), the code fine-tune; per step T0, L00 against the filed and the gain."""
    th, ref = run.P["thresholds"], run.P["refs"]["e36"]
    train, held, sel = AN.code_rows(run.vocab, log)
    deltas, evals = AN.code_finetune(run.model, train, held, run.device, log)
    torch.save(held, LARGE / "code_held.pt")
    rec: dict[str, Any] = {"select": sel, "select_ok": sel == ref["select"], "held_sha256": sha256(LARGE / "code_held.pt"), "states": {}}
    for step in AN.CODE_CHECKS:
        path = LARGE / f"code_step{step}.pt"
        torch.save(deltas[step], path)
        c = row_cells(run, held, deltas[step])
        rs = ref["steps"][str(step)]
        tol = max(th["t1_rel_tol"] * abs(rs["gain"]), th["e36_gain_abs_tol"])
        rec["states"][str(step)] = {**c, "sha256": sha256(path), "kl_rows": evals[step]["kl_to_ref"], "ok": within(c["G"], rs["gain"], tol)
                                    and within(c["cells"]["T0"], ref["T0"], th["w0_tol"]) and within(c["cells"]["L00"], ref["L00"], th["w0_tol"])}
    rec["ok"] = rec["select_ok"] and all(v["ok"] for v in rec["states"].values())
    return rec


def stage0a(run: Run) -> None:
    out: dict[str, Any] = {"groups": {}, "pass": False}

    def done(key: str, rec: dict[str, Any]) -> None:
        out["groups"][key] = rec
        dump("stage0a.json", out)
        log(f"[0a {key}] {json.dumps(_finite({k: v for k, v in rec.items() if k != 'states'}))}")
        for s, v in rec.get("states", {}).items():
            log(f"[0a {key}/{s}] {json.dumps(_finite(v))}")
        if not rec["ok"]:
            log(f"[0a] FAIL at {key}: stop and fix")
            sys.exit(1)

    for n in E18_TASKS:
        done(f"e18/{n}", e18_0a(run, n))
    built, rec = e12_build_0a(run)
    done("e12/build", rec)
    for n in built:
        done(f"e12/{n}", e12_0a(run, n, built[n]))
    done("e36/code", e36_0a(run))
    out["pass"] = True
    dump("stage0a.json", out)
    log("[0a] PASS")


# ----------------------------------------------------------------------------- stage 0b: the table


def random_row(run: Run, t: Task, dW: dict[str, Tensor], ft: dict[str, Any]) -> dict[str, Any]:
    from commute.finetune import random_direction

    d = random_direction(dW, 0)
    m = PT.match(lambda c: read_state(run, t, PT.scaled(d, c))[1], ft["kl_other"], PT.SCALE_CAP, run.P["thresholds"]["kl_rel_tol"])
    return {**cells(run, t, PT.scaled(d, m["x"])), "scale": m["x"], "match": m["status"], "kl_target": ft["kl_other"], "trace": m["trace"]}


def write_row(run: Run, t: Task, w: dict[str, Any], ft: dict[str, Any]) -> dict[str, Any]:
    """P0.4's per-key direct write, alpha matched to the seed-0 fine-tune's real gain on this task."""
    task = w["task"]
    A = PT.key_inputs(run.model, t.train.tokens, t.half, list(task.keys))
    U = run.model.target_model.lm_head.weight.detach().float()[torch.tensor(task.values)].T
    X = PT.key_write(A, U)
    resid = float((X @ A.double() - U.double()).norm() / U.double().norm())
    dw = lambda a: {PT.WRITE_KEY: (a * X).float()}
    if ft["G"] <= 0:
        return {"status": "no_target", "resid": resid}
    m = PT.match(lambda a: read_state(run, t, dw(a))[0] - t.T0f, ft["G"], PT.ALPHA_MAX, run.P["thresholds"]["write_rel_tol"])
    status = {"ok": "ok", "cap": "unreachable", "no_match": "no_match"}[m["status"]]
    rec = {"status": status, "alpha": m["x"], "G_full": m["f"], "G_target": ft["G"], "resid": resid, "trace": m["trace"],
           "norm": float(m["x"] * X.norm())}
    if status != "ok":
        return rec
    c = cells(run, t, dw(m["x"]))
    return {**c, **rec, "rho": c["g"] / ft["g"] if ft["g"] >= run.P["thresholds"]["p04_min_ft_g"] else None}


def stage0b(run: Run) -> None:
    s0a = json.loads((RESULTS / "stage0a.json").read_text()) if (RESULTS / "stage0a.json").is_file() else {}
    if not s0a.get("pass"):
        sys.exit("stage 0a has not passed: run --stages 0a first")
    from commute import regating as G

    out: dict[str, Any] = {"states": {}}

    def put(key: str, kind: str, task: str, rec: dict[str, Any]) -> dict[str, Any]:
        out["states"][key] = {"kind": kind, "task": task, **rec}
        dump("stage0b.json", out)
        log(f"[0b {key}] " + " ".join(f"{k} {rec[k]:.4f}" if isinstance(rec.get(k), float) else f"{k} {rec.get(k)}"
                                     for k in ("G", "g", "F", "se", "n", "piece_gain", "gated_piece_gain", "kl_other", "r",
                                               "scale", "match", "alpha", "status")) + f" | cells {json.dumps(rec.get('cells'))}")
        return rec

    for n in E18_TASKS:
        ref = run.P["refs"]["e18"][n]
        t = run.task(n, *G.load_windows(n, TRANSPORT, LADDER))
        files = {s: E38_STATES / f"{n}_{s}.pt" for s in ("s0", "s1", "null")} | ({"exact": EXACT} if n == "emoticon" else {})
        dws = {s: load_checked(p, ref["sha256"][s]) for s, p in files.items()}
        recs = {s: put(f"{n}/{s}", "null" if s == "null" else "finetune", n, cells(run, t, d)) for s, d in dws.items()}
        put(f"{n}/random", "random", n, random_row(run, t, dws["s0"], recs["s0"]))
        put(f"{n}/flipped", "flipped", n, cells(run, t, PT.flipped(dws["s0"])))
        del t, dws
    e12_0b(run, s0a, put)
    e36 = s0a["groups"]["e36/code"]
    held = load_checked(LARGE / "code_held.pt", e36["held_sha256"])
    for step, st in e36["states"].items():
        put(f"code/step{step}", "code", "code", {**row_cells(run, held, load_checked(LARGE / f"code_step{step}.pt", st["sha256"])),
                                                  "kl_rows": st["kl_rows"]})
    out["score"] = score(out["states"], run.P)
    dump("stage0b.json", out)
    log(f"[0b score] {json.dumps(_finite(out['score']))}")


def e12_0b(run: Run, s0a: dict[str, Any], put: Any) -> None:
    """E12's states lifted (their least-squares splits), then the seed-0 fine-tune's random, flipped and write rows."""
    for n, sha in s0a["groups"]["e12/build"]["sha256"].items():
        w = load_checked(LARGE / f"windows_{n}.pt", sha)
        t = run.task(n, w["train"], w["heldout"], w["table"])
        dws = {s: load_checked(LARGE / f"{n}_{s}.pt", st["sha256"]) for s, st in s0a["groups"][f"e12/{n}"]["states"].items()}
        recs = {s: put(f"{n}/{s}", "finetune", n, cells(run, t, d, AN.lifted(run.model, run.lift, d))) for s, d in dws.items()}
        put(f"{n}/random", "random", n, random_row(run, t, dws["s0"], recs["s0"]))
        put(f"{n}/flipped", "flipped", n, cells(run, t, PT.flipped(dws["s0"])))
        put(f"{n}/write", "write", n, write_row(run, t, w, recs["s0"]))
        del t, dws


# ----------------------------------------------------------------------------- the filed rules


def score(S: dict[str, dict[str, Any]], P: dict[str, Any]) -> dict[str, Any]:
    """predictions.json's rules, as filed; every branch named there."""
    th, cls = P["thresholds"], P["refs"]["e12_class"]
    ft = lambda n, s="s0": S[f"{n}/{s}"]
    # P0.2
    p2 = {}
    for n, c in cls.items():
        for s in ("s0", "s1"):
            x = ft(n, s)
            ratio = x["gated_piece_gain"] / x["piece_gain"] if x["piece_gain"] >= th["p02_min_piece"] else None
            ok = None if ratio is None else (ratio >= th["p02_copy_min"] if c == "copy" else ratio <= th["p02_random_max"])
            p2[f"{n}/{s}"] = {"class": c, "ratio": ratio, "pass": ok}
    scored = {c: [v["pass"] for v in p2.values() if v["class"] == c and v["pass"] is not None] for c in ("copy", "random")}
    p02 = "pass" if all(scored.values()) and all(all(v) for v in scored.values()) else "fail"
    # P0.3 and the non-positive rule
    rows = {}
    for k, x in S.items():
        if x["kind"] == "random":
            g_ft = abs(ft(x["task"])["g"])
            rows[k] = {"g": x["g"], "F": x["F"], "match": x["match"], "within_floor": abs(x["g"]) <= x["F"],
                       "fails": abs(x["g"]) > x["F"] and abs(x["g"]) >= th["p03_fail_frac"] * g_ft}
    nonpos = {k: {"G": x["G"], "g": x["g"], "ok": x["g"] <= th["nonpos_margin"] + x["F"]} for k, x in S.items() if "G" in x and x["G"] <= 0}
    if any(r["fails"] for r in rows.values()) or not all(v["ok"] for v in nonpos.values()):
        p03 = "fail"
    else:
        p03 = "pass" if all(r["within_floor"] for r in rows.values()) else "inconclusive"
    sign = {k: {"g": x["g"], "G": x["G"], "same_sign": (x["g"] > 0) == (x["G"] > 0)} for k, x in S.items() if x["kind"] == "flipped"}
    p04, rho = p04_verdict(S, P)
    tree = ("stop: fix regeneration" if p02 != "pass" else "retire" if "fail" in (p03, p04)
            else "propose stage 1" if (p03, p04) == ("pass", "pass") else "card")
    return {"P0.2": {"status": p02, "rows": p2}, "P0.3": {"status": p03, "rows": rows, "nonpositive": nonpos},
            "sign_unscored": sign, "P0.4": {"status": p04, "rho": rho}, "tree": tree}


def p04_verdict(S: dict[str, dict[str, Any]], P: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    th, cls = P["thresholds"], P["refs"]["e12_class"]
    rho = {n: S[f"{n}/write"].get("rho") for n in cls}
    copy = [n for n, c in cls.items() if c == "copy"]
    if any(S[f"{n}/{s}"]["g"] < th["p04_min_ft_g"] for n in copy for s in ("s0", "s1")):
        return "inconclusive", rho
    reached = [rho[n] for n in copy if rho[n] is not None]
    if any(x >= th["p04_fail"] for x in reached):
        return "fail", rho
    if len(reached) == len(copy) and all(x <= th["p04_pass"] for x in reached):
        return "pass", rho
    return "inconclusive", rho


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", required=True, choices=["0a", "0b"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if not PRED.is_file():
        sys.exit(f"file the predictions first: {PRED}")
    LARGE.resolve().mkdir(parents=True, exist_ok=True)
    run = Run(args.device)
    {"0a": stage0a, "0b": stage0b}[args.stages](run)


if __name__ == "__main__":
    main()
