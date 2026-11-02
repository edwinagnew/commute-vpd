"""E12 Amendment 1: the route split on teacher-trained states. Plan reports/plans/2026-11-02_plan_e12-teacher-route-split.md;
rules, config, leans, expected values: results/predictions_amend1.json (the run refuses without it; its `config` is read below).
States are read from trained fp32 weights, never bf16 offsets. A stop is `.stopped` in the file, exit 1. cluster/new_vs_existing_amend1.sbatch.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from commute import anchors as AN
from commute import arms, component_training, saved_states, teacher, vpd
from commute.faithfulness import model_state
from commute.finetune import ArmAConfig
from commute.keydep import sha256
from commute.local_faith import MIN_GAIN, route_cells, route_split

RESULTS = Path(__file__).resolve().parent / "results"
PRED, OUT, STATES = RESULTS / "predictions_amend1.json", RESULTS / "new_vs_existing_amend1.json", RESULTS / "large" / "amend1"
E46, EXP = RESULTS.parents[1] / "regating_table" / "results", RESULTS.parents[1]
ROOT, CODE = EXP.parent, ("src/commute", "experiments/new_vs_existing/amend1.py", "experiments/new_vs_existing/results/predictions_amend1.json")
MOVED = EXP / "teacher_training" / "results" / "large" / "amend8" / "n1" / "0.pt"
TASKS, CLS, SEEDS = [t.name for t in AN.KV_TASKS], {t.name: t.cls for t in AN.KV_TASKS}, ("s0", "s1")
NAMES = [f"{t}_{s}" for t in TASKS for s in SEEDS]
CM = SimpleNamespace(**json.loads(PRED.read_text())["config"])  # the one source of every size and line; a missing file stops the import


# ----------------------------------------------------------------------------- rules (pure), as filed


def finite(read: dict[str, Any]) -> bool:
    return all(math.isfinite(v) for v in read["shares"].values())


def in_pool(rec: dict[str, Any]) -> bool:
    a = rec["a"]
    return finite(a) and a["gain"] >= MIN_GAIN and a["gain"] / rec["teacher_gain"] >= CM.min_ratio


def separation(copy: list[float], maps: list[float]) -> dict[str, Any]:
    """P1: strict both ways; under min_class states in a class, unscorable."""
    if min(len(copy), len(maps)) < CM.min_class:
        return {"verdict": "unscorable", "copy": copy, "maps": maps}
    within = max(max(copy) - min(copy), max(maps) - min(maps))
    gap, swapped = min(maps) - max(copy), min(copy) - max(maps)
    verdict = "separates" if gap > within else "reversed" if swapped > within else "no separation"
    return {"verdict": verdict, "gap": gap, "swapped_gap": swapped, "within": within, "copy": copy, "maps": maps}


def r_class(moves: list[float], n_copy: int, n_maps: int, noise: list[float]) -> str:
    """P2: `moves` |L01(student) - L01(fine-tune)| / teacher's gain over the pool, `noise` the same between batch orders."""
    if min(n_copy, n_maps) < CM.min_class or not noise:
        return "unscorable"
    if max(noise) > CM.no_power and max(moves) < CM.noise_factor * max(noise):
        return "no power"
    if max(moves) >= CM.moved:
        return "moved"
    return "inherited" if max(moves) <= CM.inherited else "partly"


def masked_class(e: float) -> str:
    return "masked-off" if e >= CM.masked_hi else "not masked-off" if e <= CM.masked_lo else "mixed"


def masked_majority(es: list[float]) -> str:
    """P3: the class of strictly more than half of the copy states; else, or under min_class states, unsettled."""
    labels = [masked_class(e) for e in es]
    top = max(set(labels), key=labels.count) if labels else ""
    return top if len(es) >= CM.min_class and labels.count(top) * 2 > len(es) else "unsettled"


def score(c0: dict[str, Any], st: dict[str, Any]) -> dict[str, Any]:
    pool = [n for n in NAMES if in_pool(st[n])]
    pool_b = [n for n in pool if finite(st[n]["b"])]  # (b)'s own gain is the teacher's, so (a) decides whether it learned
    share = lambda n, side, x: st[n][side]["shares"][x]
    l01 = lambda n: st[n]["a"]["cells"]["L01"]
    of = lambda ns, c: [n for n in ns if CLS[st[n]["task"]] == c]
    sep = lambda ns, side: separation(*([share(n, side, "e_gate") for n in of(ns, c)] for c in ("copy", "random")))
    pairs = [(f"{t}_s0", f"{t}_s0_o{CM.second_order}") for t in CM.noise_tasks]
    pairs = [p for p in pairs if in_pool(st[p[0]]) and in_pool(st[p[1]])]
    moves = {n: abs(l01(n) - c0[n]["nothing"]["cells"]["L01"]) / st[n]["teacher_gain"] for n in pool}  # L00 is the same in both
    noise_r = {b: abs(l01(a) - l01(b)) / st[a]["teacher_gain"] for a, b in pairs}
    noise_e = {b: abs(share(a, "a", "e_gate") - share(b, "a", "e_gate")) for a, b in pairs}
    copies, n_maps = of(pool, "copy"), len(of(pool, "random"))
    e_gate = {n: share(n, "a", "e_gate") for n in copies}
    return {"pool": pool, "dropped": [n for n in NAMES if n not in pool],
            "P1": {**sep(pool, "a"), "state_b": sep(pool_b, "b"), "noise_e_gate": noise_e},
            "P2": {"verdict": r_class(list(moves.values()), len(copies), n_maps, list(noise_r.values())), "moves": moves,
                   "gain_ratio": {n: st[n]["a"]["gain"] / st[n]["teacher_gain"] for n in pool}, "noise_r": noise_r},
            "P3": {"verdict": masked_majority(list(e_gate.values())), "e_gate": e_gate,
                   "state_b": masked_majority([share(n, "b", "e_gate") for n in of(pool_b, "copy")])}}


# ----------------------------------------------------------------------------- the job


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def dump(out: dict[str, Any]) -> None:
    tmp = OUT.with_suffix(".tmp")  # a kill mid-write must not leave unreadable JSON
    tmp.write_text(json.dumps(out, indent=1, sort_keys=True, default=float))
    tmp.replace(OUT)


def check(out: dict[str, Any], name: str, value: float, ok: bool) -> None:
    """A filed stop rule, kept in the file; a miss (a NaN too) stops the job, and a rerun checks it again."""
    log(f"[check {name}] {value:.8g}: {ok}")
    out.setdefault("checks", {})[name] = {"value": value, "pass": ok}
    if not ok:
        out["stopped"] = f"{name}: {value:.8g}"
    dump(out)
    if not ok:
        sys.exit(f"STOP: {out['stopped']}")


def near(out: dict[str, Any], name: str, a: float, b: float, tol: float) -> None:
    check(out, f"{name} (difference)", a - b, abs(a - b) <= tol)


def code_unchanged(commit: str) -> bool:
    """The measurement code and the predictions are as they were at `commit` (later doc commits do not count)."""
    return subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", commit, "--", *CODE], check=False).returncode == 0


class Job:
    def __init__(self, device: str, stage0a: dict[str, Any], out: dict[str, Any]) -> None:
        self.device, self.stage0a, self.out, self.model = device, stage0a, out, vpd.load_component_model(device)
        self.W0, self.comps0 = vpd.target_weights(self.model), vpd.component_weights(self.model)
        self.alive = saved_states.load_alive(EXP / "transport_test" / "results" / "large")
        self.uv = component_training.alive_uv(self.model, self.alive)
        self.uv_cpu = {k: (u.cpu(), v.cpu()) for k, (u, v) in self.uv.items()}
        self.lift = AN.lift_context(self.model, self.alive)
        self.tasks: dict[str, Any] = {}

    def task(self, name: str) -> dict[str, Any]:
        """The task's windows and W0's cells on them; at first use T0, U0, L00 against E46's."""
        if name not in self.tasks:
            w = torch.load(E46 / "large" / f"windows_{name}.pt", weights_only=False, map_location="cpu")
            w["base"] = route_cells(self.model, w["heldout"].tokens, w["heldout"].half, w["table"])
            for cell, key in (("T0", "T"), ("U0", "U"), ("L00", "L")):
                near(self.out, f"W0 {name} {cell} against E46's", w["base"][key], self.stage0a[f"e12/{name}"]["W0"]["got"][cell], CM.w0_tol)
            self.tasks[name] = w
        return self.tasks[name]

    def read(self, w: dict[str, Any], label: str, dW: dict[str, Any], comps: Any) -> dict[str, Any]:
        """The route split at target W0 + dW with components `comps` (None: released). Logged whole."""
        with model_state(self.model, target={k: self.W0[k] + dW[k].to(self.W0[k].device) for k in self.W0}, components=comps):
            rs = route_split(self.model, w["heldout"].tokens, w["heldout"].half, w["table"], w["base"], self.comps0)
        log(f"[read {label}] {json.dumps(rs, default=float)}")
        return rs

    def train(self, w: dict[str, Any], net: Any, init: Any, order: int, path: Path, meta: dict[str, Any]) -> tuple[Any, Any, list[Any]]:
        """E42's arm A (Adam, clip 1, batch-order seed `order`) against `net` from `init`, offsets saved: (delta-W, full components, curve)."""
        half = w["train"].half
        a_cfg = ArmAConfig(o_token=w["table"], batch=CM.batch, steps=CM.steps, checkpoints=(CM.steps,), seed=order)
        cfg = component_training.TrainConfig(lr=CM.lr, steps=CM.steps, save_steps=(CM.steps,), optimizer="adam")
        curve, states = component_training.train(self.model, self.uv, w["train"].tokens, w["heldout"], a_cfg, cfg, self.device, self.W0, log=log,
                                                 loss_fn=lambda n, toks: teacher.teacher_loss(n, net, toks, half), init=init)
        st = states[CM.steps]
        arms.save_offsets(path, self.uv_cpu, st, {"step": CM.steps, "order": order, **meta})
        comps = component_training.components_with(self.model, self.alive, {k: (v[1], v[2]) for k, v in st.items()})
        return {k: v[0] for k, v in st.items()}, comps, [c for c in curve if c["step"] in CM.curve_steps]


def c0(job: Job, out: dict[str, Any]) -> None:
    """Do-nothing and the least-squares split of every fine-tune, against E12 and each other."""
    for n in NAMES:
        t, s = n.split("_")
        w = job.task(t)
        if n not in out["c0"]:
            dW = torch.load(E46 / "large" / f"{n}.pt", weights_only=False, map_location="cpu")
            out["c0"][n] = {"nothing": job.read(w, f"{n} nothing", dW, None), "least_squares": job.read(w, f"{n} ls", dW, AN.lifted(job.model, job.lift, dW))}
            dump(out)
        no, ls = out["c0"][n]["nothing"]["shares"], out["c0"][n]["least_squares"]["shares"]
        near(out, f"{n} r against E12's", no["r"], job.stage0a[f"e12/{t}"]["states"][s]["r_ref"], CM.r_tol)
        near(out, f"{n} do-nothing e_gate against -r", no["e_gate"], -no["r"], CM.zero_tol)
        near(out, f"{n} do-nothing r against least-squares r", no["r"], ls["r"], CM.zero_tol)
    of = lambda c: [out["c0"][n]["least_squares"]["shares"]["e_gate"] for n in NAMES if CLS[n.split("_")[0]] == c]
    out["c0_separation"] = sep = separation(of("copy"), of("random"))
    log(f"[C0 least-squares e_gate] {json.dumps(sep)}")
    check(out, "C0 least-squares separates: gap - within", sep["gap"] - sep["within"], sep["verdict"] == "separates")


def terms(rs: dict[str, Any]) -> dict[str, float]:
    """The route's terms in nats (C1's gain is near 0, so its shares are unusable)."""
    c = rs["cells"]
    return {"gain": rs["gain"], "d": c["L10"] - c["L00"], "r": c["L01"] - c["L00"], "i": c["L11"] - c["L10"] - c["L01"] + c["L00"],
            "e_gate": (c["U1"] - c["L11"]) - (c["U0"] - c["L00"]), "e_delta": (c["T1"] - c["U1"]) - (c["T0"] - c["U0"])}


def c1(job: Job, out: dict[str, Any]) -> None:
    """Teacher = W0 from the moved start, trained once (the two tasks' windows differ only in the table, which training never reads),
    then read under both tables. Start, end and end - start are reported; the stop reads the end."""
    c, ws = out["c1"], {t: job.task(t) for t in CM.c1_tasks}
    if not ("end" in c and (STATES / "c1.pt").is_file()):
        off = torch.load(MOVED, weights_only=False, map_location="cpu")["offsets"]
        dW0, uv2 = saved_states.offsets_state(MOVED, job.uv_cpu)
        comps0 = component_training.components_with(job.model, job.alive, uv2)
        out["c1_start_sha256"] = sha256(MOVED)
        c["start"] = {t: terms(job.read(w, f"C1 {t} start", dW0, comps0)) for t, w in ws.items()}
        dW, comps, c["curve"] = job.train(ws[CM.c1_tasks[0]], teacher.teacher_net(job.model, {}), off, 0, STATES / "c1.pt", {"arm": "c1"})
        c["end"] = {t: terms(job.read(w, f"C1 {t} end", dW, comps)) for t, w in ws.items()}
        c["end_minus_start"] = {t: {x: c["end"][t][x] - c["start"][t][x] for x in c["end"][t]} for t in ws}
        dump(out)
        log(f"[C1] {json.dumps(c, default=float)}")
    for t in ws:
        for x in ("r", "e_gate"):
            check(out, f"C1 {t} {x} term, nats", c["end"][t][x], abs(c["end"][t][x]) <= CM.c1_max_nats)


def student(job: Job, out: dict[str, Any], t: str, s: str, order: int) -> None:
    name, n = f"{t}_{s}" + (f"_o{order}" if order else ""), f"{t}_{s}"
    if not (name in out["students"] and (STATES / f"{name}.pt").is_file()):
        w, dW_t = job.task(t), torch.load(E46 / "large" / f"{n}.pt", weights_only=False, map_location="cpu")
        dW, comps, curve = job.train(w, teacher.teacher_net(job.model, dW_t), None, order, STATES / f"{name}.pt", {"arm": "student", "task": t, "seed": s})
        gain = out["c0"][n]["nothing"]["gain"]
        a, b = job.read(w, f"{name} a", dW, comps), job.read(w, f"{name} b", dW_t, comps)
        out["students"][name] = {"task": t, "seed": s, "order": order, "teacher_gain": gain, "gain_ratio": a["gain"] / gain, "a": a, "b": b, "curve": curve}
        dump(out)
    if name == NAMES[0]:  # the first student stops the job; a later one that learned too little only leaves the pool
        rec = out["students"][name]
        a, b = rec["a"], rec["b"]
        check(out, "first student gain / teacher's", rec["gain_ratio"], rec["gain_ratio"] >= CM.min_ratio)
        near(out, "first student r(b) against the fine-tune's", b["shares"]["r"], out["c0"][n]["nothing"]["shares"]["r"], CM.zero_tol)
        near(out, "first student U1(a) against U1(b)", a["cells"]["U1"], b["cells"]["U1"], CM.zero_tol)


def main() -> None:
    stage0a = json.loads((E46 / "stage0a.json").read_text())["groups"]
    files = {E46 / "large" / f"windows_{t}.pt": stage0a["e12/build"]["sha256"][t] for t in TASKS}
    files |= {E46 / "large" / f"{n}.pt": stage0a[f"e12/{n.split('_')[0]}"]["states"][n.split("_")[1]]["sha256"] for n in NAMES}
    for path, sha in files.items():
        if not path.is_file() or sha256(path) != sha:
            sys.exit(f"{path} is missing or not the file stage0a.json recorded")
    if not MOVED.is_file():
        sys.exit(f"{MOVED} is missing")
    STATES.mkdir(parents=True, exist_ok=True)
    card = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    out = json.loads(OUT.read_text()) if OUT.is_file() else {"card": card, "commit": head, "config": vars(CM), "c0": {}, "c1": {}, "students": {}}
    if out["card"] != card or out["config"] != vars(CM) or not code_unchanged(out["commit"]):
        sys.exit(f"{OUT} was made on {out['card']} at {out['commit']} with another config, or the code or predictions changed since. Move the file aside to start again")
    out.pop("stopped", None)
    log(f"[start] {card} {out['commit']}")
    job = Job("cuda" if torch.cuda.is_available() else "cpu", stage0a, out)
    c0(job, out)
    c1(job, out)
    for t, s in (n.split("_") for n in NAMES):
        student(job, out, t, s, 0)
    for t in CM.noise_tasks:
        student(job, out, t, "s0", CM.second_order)
    out["score"], out["done"] = score(out["c0"], out["students"]), True
    dump(out)
    log(f"[score] {json.dumps(out['score'])}")


if __name__ == "__main__":
    main()
