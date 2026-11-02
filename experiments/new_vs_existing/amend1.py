"""E12 Amendment 1: the route split on teacher-trained states. Plan reports/plans/2026-11-02_plan_e12-teacher-route-split.md;
rules, leans, expected values results/predictions_amend1.json (the run refuses without it). E46's rebuilds of E12's eight
fine-tunes teach E42's arm A; the split reads each student on the held-out windows as state (a), teacher training's own
(target W0 + its change), scored first, and state (b), plus remainder (target the fine-tune's). Arms, each written as it
finishes: C0 do-nothing and least-squares, alone; C1 teacher = W0 from Amendment 8's moved start; 8 students; noise floor.
States are read from trained fp32 weights, never bf16 offsets: a rerun skips a state with a record and an offsets file and
retrains any other (deterministic). A stop is `.stopped` in the file, exit 1. cluster/new_vs_existing_amend1.sbatch.
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
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
MOVED = EXP / "teacher_training" / "results" / "large" / "amend8" / "n1" / "0.pt"
TASKS, CLS, SEEDS = [t.name for t in AN.KV_TASKS], {t.name: t.cls for t in AN.KV_TASKS}, ("s0", "s1")
NAMES = [f"{t}_{s}" for t in TASKS for s in SEEDS]


@dataclass(frozen=True)
class Amend1:
    lr: float = 1e-5  # E42's arm A: Adam, clip 1 (TrainConfig's default), batch-order seed 0
    steps: int = 1000
    batch: int = 32
    second_order: int = 1
    noise_tasks: tuple[str, ...] = ("ident32", "rand32")  # seed s0 of each, at second_order
    c1_tasks: tuple[str, ...] = ("rand32", "ident32")
    curve_steps: tuple[int, ...] = (1, 2, 10, 100, 500, 1000)
    w0_tol: float = 0.02  # nats, E46's
    r_tol: float = 0.03  # E46's, against E12's r
    zero_tol: float = 1e-6  # identities that hold by code
    c1_max_nats: float = 1.5  # 0.11 x the smallest teacher gain 13.2; 0.11 is a third of E12's smallest e_gate gap 0.33
    min_ratio: float = 0.5  # gain / teacher's: the pool's line and the first student's stop
    min_class: int = 3
    inherited: float = 0.10  # P2
    moved: float = 0.25
    no_power: float = 0.05
    masked_hi: float = 0.5  # P3, E42's filed lines
    masked_lo: float = 0.25

    def __post_init__(self) -> None:
        assert 0 < self.inherited < self.moved and 0 < self.masked_lo < self.masked_hi and self.second_order != 0
        assert set(self.noise_tasks) | set(self.c1_tasks) <= set(TASKS) and self.steps == self.curve_steps[-1]


CM = Amend1()


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
    """P2: `moves` |r(student) - r(fine-tune)| over the pool, `noise` batch-order |delta r| over the pairs both in it."""
    if min(n_copy, n_maps) < CM.min_class or not noise:
        return "unscorable"
    if max(noise) > CM.no_power:
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
    of = lambda ns, c: [n for n in ns if CLS[st[n]["task"]] == c]
    sep = lambda ns, side: separation(*([share(n, side, "e_gate") for n in of(ns, c)] for c in ("copy", "random")))
    pairs = [(f"{t}_s0", f"{t}_s0_o{CM.second_order}") for t in CM.noise_tasks]
    pairs = [p for p in pairs if in_pool(st[p[0]]) and in_pool(st[p[1]])]
    noise = lambda x, side: {b: abs(share(a, side, x) - share(b, side, x)) for a, b in pairs}
    moves = {n: abs(share(n, "a", "r") - c0[n]["nothing"]["shares"]["r"]) for n in pool}
    copies, n_maps = of(pool, "copy"), len(of(pool, "random"))
    e_gate = {n: share(n, "a", "e_gate") for n in copies}
    return {"pool": pool, "dropped": [n for n in NAMES if n not in pool],
            "P1": {**sep(pool, "a"), "state_b": sep(pool_b, "b"), "noise_e_gate": noise("e_gate", "a"), "noise_e_gate_b": noise("e_gate", "b")},
            "P2": {"verdict": r_class(list(moves.values()), len(copies), n_maps, list(noise("r", "a").values())), "moves": moves, "noise_r": noise("r", "a")},
            "P3": {"verdict": masked_majority(list(e_gate.values())), "e_gate": e_gate, "per_state": {n: masked_class(e) for n, e in e_gate.items()},
                   "state_b": masked_majority([share(n, "b", "e_gate") for n in of(pool_b, "copy")]), "noise_e_gate": noise("e_gate", "a")}}


# ----------------------------------------------------------------------------- the job


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def dump(out: dict[str, Any]) -> None:
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True, default=float))


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
        """E42's arm A against `net` from `init`, offsets saved: (delta-W, full components, curve)."""
        half = w["train"].half
        a_cfg = ArmAConfig(o_token=w["table"], batch=CM.batch, steps=CM.steps, checkpoints=(CM.steps,), seed=order)
        cfg = component_training.TrainConfig(lr=CM.lr, steps=CM.steps, save_steps=(CM.steps,), optimizer="adam")
        curve, states = component_training.train(self.model, self.uv, w["train"].tokens, w["heldout"], a_cfg, cfg, self.device, self.W0, log=log,
                                                 loss_fn=lambda n, toks: teacher.teacher_loss(n, net, toks, half), init=init)
        st = states[CM.steps]
        arms.save_offsets(path, self.uv_cpu, st, {"step": CM.steps, "order": order, **meta})
        comps = component_training.components_with(self.model, self.alive, {k: (v[1], v[2]) for k, v in st.items()})
        return {k: v[0] for k, v in st.items()}, comps, [c for c in curve if c["step"] in CM.curve_steps]


def teacher_dW(name: str) -> dict[str, Any]:
    return torch.load(E46 / "large" / f"{name}.pt", weights_only=False, map_location="cpu")


def c0(job: Job, out: dict[str, Any]) -> None:
    """Do-nothing and the least-squares split of every fine-tune, against E12 and each other."""
    for n in NAMES:
        t, s = n.split("_")
        w = job.task(t)
        if n not in out["c0"]:
            dW = teacher_dW(n)
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
    """Teacher = W0 from the moved start: what training with nothing to learn writes into r and e_gate, in nats. The stop
    reads the end state, which compares with a student's; the start is reported."""
    off = torch.load(MOVED, weights_only=False, map_location="cpu")["offsets"]
    dW0, uv2 = saved_states.offsets_state(MOVED, job.uv_cpu)
    comps0 = component_training.components_with(job.model, job.alive, uv2)
    net0 = teacher.teacher_net(job.model, {})
    out["c1_start_sha256"] = sha256(MOVED)
    for t in CM.c1_tasks:
        if not (t in out["c1"] and (STATES / f"c1_{t}.pt").is_file()):
            w = job.task(t)
            start = terms(job.read(w, f"C1 {t} start", dW0, comps0))
            dW, comps, curve = job.train(w, net0, off, 0, STATES / f"c1_{t}.pt", {"arm": "c1", "task": t})
            out["c1"][t] = {"start": start, "end": terms(job.read(w, f"C1 {t} end", dW, comps)), "curve": curve}
            dump(out)
            log(f"[C1 {t}] {json.dumps(out['c1'][t], default=float)}")
        for x in ("r", "e_gate"):
            end = out["c1"][t]["end"][x]
            check(out, f"C1 {t} {x} term, nats", end, abs(end) <= CM.c1_max_nats)


def student(job: Job, out: dict[str, Any], t: str, s: str, order: int) -> None:
    name, n = f"{t}_{s}" + (f"_o{order}" if order else ""), f"{t}_{s}"
    if not (name in out["students"] and (STATES / f"{name}.pt").is_file()):
        w, dW_t = job.task(t), teacher_dW(n)
        dW, comps, curve = job.train(w, teacher.teacher_net(job.model, dW_t), None, order, STATES / f"{name}.pt", {"arm": "student", "task": t, "seed": s})
        gain = out["c0"][n]["nothing"]["gain"]
        a, b = job.read(w, f"{name} a", dW, comps), job.read(w, f"{name} b", dW_t, comps)
        out["students"][name] = {"task": t, "seed": s, "order": order, "teacher_gain": gain, "gain_ratio": a["gain"] / gain, "a": a, "b": b, "curve": curve}
        dump(out)
    if name == NAMES[0]:  # the first student stops the job; a later one that learned too little only leaves the pool
        rec = out["students"][name]
        a, b = rec["a"], rec["b"]
        near(out, "first student share sum (a) against 1", a["share_sum"], 1.0, CM.zero_tol)
        near(out, "first student share sum (b) against 1", b["share_sum"], 1.0, CM.zero_tol)
        check(out, "first student gain / teacher's", rec["gain_ratio"], rec["gain_ratio"] >= CM.min_ratio)
        near(out, "first student r(b) against the fine-tune's", b["shares"]["r"], out["c0"][n]["nothing"]["shares"]["r"], CM.zero_tol)
        near(out, "first student U1(a) against U1(b)", a["cells"]["U1"], b["cells"]["U1"], CM.zero_tol)


def main() -> None:
    if not PRED.is_file():
        sys.exit(f"file the predictions first: {PRED}")
    assert json.loads(PRED.read_text())["config"] == json.loads(json.dumps(asdict(CM))), "predictions were filed for another config"
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
    out = json.loads(OUT.read_text()) if OUT.is_file() else {"card": card, "c0": {}, "c1": {}, "students": {}}
    if out["card"] != card:
        sys.exit(f"{OUT} was written on {out['card']}; this job is {card}. Move the file aside to start again")
    out.pop("stopped", None)
    log(f"[start] {card}")
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
