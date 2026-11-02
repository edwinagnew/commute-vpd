"""E12 Amendment 2: why the re-switching term falls on rand32b's students, and whether students switch their fine-tunes' subcomponents.
Plan reports/plans/2026-11-02_plan_e12-amend2-rand32b-fall.md; rules, lines, expected values: results/predictions_amend2.json (the run
refuses without it). Same recipe as Amendment 1 (its Job, its config). Students are retrained here and read from fp32 weights, never bf16
offsets. A stop is `.stopped` in the file, exit 1. cluster/new_vs_existing_amend2.sbatch.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import amend1
import torch
from amend1 import E46, Job, check, dump, log, near
from torch import Tensor

from commute import recruit, signatures, teacher
from commute.faithfulness import model_state
from commute.keydep import sha256
from commute.local_faith import masked_lo, route_cells

RESULTS = Path(__file__).resolve().parent / "results"
PRED, OUT, A1_FILE, STATES = RESULTS / "predictions_amend2.json", RESULTS / "new_vs_existing_amend2.json", RESULTS / "new_vs_existing_amend1.json", RESULTS / "large" / "amend2"
CM = SimpleNamespace(**json.loads(PRED.read_text())["config"])  # the lines and sizes this file adds; the training recipe stays Amendment 1's
amend1.OUT = OUT  # amend1's check() and dump() write to this global: without this line its Job.task would overwrite Amendment 1's scored file
BLOCKS = ("dropped", "kept", "added", "outside")


# ----------------------------------------------------------------------------- rules (pure), as filed


def finite(*xs: float) -> bool:
    return all(math.isfinite(x) for x in xs)


def agree(labels: list[str]) -> str:
    """One label if every state reads it, else 'mixed'; no states: 'unscorable'; a state that could not be read wins."""
    if not labels or "unscorable" in labels:
        return "unscorable"
    return labels[0] if len(set(labels)) == 1 else "mixed"


def p1(falls: list[float]) -> str:
    """Batch order: the fall at the second order, per seed."""
    if len(falls) < CM.n_seeds or not finite(*falls):
        return "unscorable"
    if all(f >= CM.fall_line for f in falls):
        return "holds"
    return "gone" if all(abs(f) <= CM.gone_line for f in falls) else "varies"


def p2_overlap(js: list[float], j_seed: float, j_other: float) -> str:
    """Students' Jaccard with their own fine-tune, against the table's two fine-tunes and against the other map table's."""
    if len(js) < CM.n_seeds or not finite(*js, j_seed, j_other):
        return "unscorable"
    if j_other >= j_seed:
        return "no power"
    if all(j >= j_seed for j in js):
        return "same"
    return "different" if all(j <= j_other for j in js) else "partly"


def p2_reference(js: list[float], j_ref: list[float], j_order: list[float]) -> str:
    """rand32b's students against the reference table's, with the set's own batch-order change as the noise."""
    if not (js and j_ref and j_order) or not finite(*js, *j_ref, *j_order):
        return "unscorable"
    gap, noise = min(j_ref) - max(js), max(1 - j for j in j_order)
    return "lower" if gap > noise else "higher" if min(js) - max(j_ref) > noise else "no different"


def p3_state(fall: float, at_trigger: float, blocks: dict[str, float]) -> str:
    """Which block of subcomponents, restored at the trigger, gives back at least half of one state's fall (nats in, shares inside)."""
    if not finite(fall, at_trigger, *blocks.values()):
        return "unscorable"
    if fall < CM.fall_line:
        return "no fall"
    if at_trigger / fall < CM.half:
        return "not at the trigger"
    top = max(blocks.values())
    winners = [b for b, v in blocks.items() if v == top]
    return winners[0] if len(winners) == 1 and top / fall >= CM.half else "spread"


def p4_state(fall: float, at_trigger: float, ladder: list[float]) -> str:
    """How many subcomponents, most changed first and restored at the trigger, give back half of one state's fall."""
    if not finite(fall, at_trigger, *ladder) or len(ladder) != len(CM.ladder):
        return "unscorable"
    if fall < CM.fall_line:
        return "no fall"
    if at_trigger / fall < CM.half:
        return "not at the trigger"
    k = next((k for k, v in zip(CM.ladder, ladder, strict=True) if v / fall >= CM.half), None)
    return "few" if k is not None and k <= CM.few_k else "middling" if k is not None and k < CM.diffuse_k else "diffuse"


def p5(falling: list[float], reference: list[float]) -> str:
    """Student-to-teacher distance over teacher size: E12's rule between the two tables."""
    if len(falling) < CM.n_seeds or len(reference) < CM.n_seeds or not finite(*falling, *reference):
        return "unscorable"
    within = max(max(falling) - min(falling), max(reference) - min(reference))
    return "further" if min(falling) - max(reference) > within else "nearer" if min(reference) - max(falling) > within else "no separation"


def score(out: dict[str, Any]) -> dict[str, Any]:
    ft, st = out["fine_tunes"], out["students"]
    names = [f"{CM.falling}_{s}" for s in CM.seeds]
    refs = [f"{CM.reference}_{s}" for s in CM.seeds]
    seconds = [f"{n}_o{CM.second_order}" for n in names]
    js, fall = [st[n]["J"] for n in names], lambda n: st[n]["fall"]
    j_order = [out["anchors"][f"J {n} order 0 against order {CM.second_order}"] for n in names]
    j_seed, j_other = out["anchors"][f"J {CM.falling} seeds"], out["anchors"][f"J {CM.falling} against {CM.reference}, largest"]
    r3 = {n: p3_state(fall(n), st[n]["restored"]["at_trigger"], {b: st[n]["restored"][b] for b in BLOCKS}) for n in names}
    r4 = {n: p4_state(fall(n), st[n]["restored"]["at_trigger"], st[n]["restored"]["ladder"]) for n in names}
    moves = {n: abs(st[n]["a"]["cells"]["L01"] - ft[st[n]["fine_tune"]]["L01"]) / ft[st[n]["fine_tune"]]["gain"] for n in [*names, *seconds, *refs]}
    return {"P1": {"verdict": p1([fall(n) for n in seconds]), "falls_order_0": {n: fall(n) for n in names}, "falls_second_order": {n: fall(n) for n in seconds},
                   "noise_move": {n: abs(st[n]["a"]["cells"]["L01"] - st[o]["a"]["cells"]["L01"]) / ft[n]["gain"] for n, o in zip(names, seconds, strict=True)}, "moves": moves},
            "P2": {"verdict": p2_overlap(js, j_seed, j_other), "against_reference": p2_reference(js, [st[n]["J"] for n in refs], j_order), "J": dict(zip(names, js, strict=True)),
                   "J_reference": {n: st[n]["J"] for n in refs}, "J_seed": j_seed, "J_other_table": j_other, "J_order": dict(zip(names, j_order, strict=True))},
            "P3": {"verdict": agree(list(r3.values())), "per_state": r3},
            "P4": {"verdict": agree(list(r4.values())), "per_state": r4},
            "P5": {"verdict": p5([st[n]["distance"] for n in names], [st[n]["distance"] for n in refs]), "distance": {n: st[n]["distance"] for n in [*names, *refs]}}}


# ----------------------------------------------------------------------------- reads


def members(sel: dict[str, Tensor]) -> list[list[Any]]:
    return sorted([k, c] for k, c in recruit.members(sel))


def sel_of(items: set[tuple[str, int]], keys: list[str]) -> dict[str, Tensor]:
    return {k: torch.tensor(sorted(c for m, c in items if m == k), dtype=torch.long) for k in keys}


def gates_at(job: Job, w: dict[str, Any], dW: dict[str, Tensor]) -> list[dict[str, Tensor]]:
    """Gamma's values on the activations of target W0 + dW (Gamma reads only the target's activations)."""
    with model_state(job.model, target={k: job.W0[k] + dW[k].to(job.W0[k].device) for k in job.W0}):
        return route_cells(job.model, w["heldout"].tokens, w["heldout"].half, w["table"])["gates"]


def masked(job: Job, w: dict[str, Any], gates: list[dict[str, Tensor]]) -> float:
    """Masked log-odds of the released components, Delta out: L01 under these gates."""
    return masked_lo(job.model, w["heldout"].tokens, w["heldout"].half, w["table"], gates)[0]


def switched_on(job: Job, w: dict[str, Any], gates: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    return recruit.switched(recruit.trigger_dg(gates, w["base"]["gates"], w["heldout"].half, job.alive), CM.s_on_thr)


def abs_change(low: list[dict[str, Tensor]], high: list[dict[str, Tensor]], half: int) -> dict[str, Tensor]:
    """Mean over windows of |g_high - g_low| at the trigger, per subcomponent: a change of opposite sign on two keys does not cancel."""
    g0, g1 = recruit.cat_gates(low), recruit.cat_gates(high)
    return {k: (g1[k][:, half].float() - g0[k][:, half].float()).abs().mean(0) for k in g0}


def restored(job: Job, w: dict[str, Any], low: list[dict[str, Tensor]], high: list[dict[str, Tensor]], blocks: dict[str, dict[str, Tensor]]) -> dict[str, Any]:
    """Nats given back by writing `high`'s values at the trigger into `low`: on every subcomponent, on each block, and on the top-k most changed."""
    half = w["heldout"].half
    base, mix = masked(job, w, low), recruit.with_trigger(low, half, recruit.at_trigger(high, half))
    back = lambda sel: masked(job, w, recruit.hybrid(low, mix, sel)) - base
    ch = abs_change(low, high, half)
    top = sorted(recruit.members(signatures.top_by(ch, job.alive, CM.top_saved)), key=lambda m: -float(ch[m[0]][m[1]]))
    return {"at_trigger": masked(job, w, mix) - base, "ladder": [back(signatures.top_by(ch, job.alive, k)) for k in CM.ladder],
            "top": [[k, c, float(ch[k][c])] for k, c in top], **{b: back(sel) for b, sel in blocks.items()}}


def distance(dW: dict[str, Tensor], dW_t: dict[str, Tensor]) -> float:
    """||student's change - teacher's|| / ||teacher's|| over all matrices together (a zero teacher gives nan or inf, which the rule refuses)."""
    diff = sum(float((dW[k].double() - dW_t[k].double()).pow(2).sum()) for k in dW_t)
    size = sum(float(dW_t[k].double().pow(2).sum()) for k in dW_t)
    return float(torch.tensor(diff).sqrt() / torch.tensor(size).sqrt())


def fine_tunes(job: Job, out: dict[str, Any], a1: dict[str, Any]) -> dict[str, Any]:
    """Stage 0, no training: each fine-tune's values and L01 against Amendment 1's, its switched-on set, the anchors, the ladder on the term itself."""
    ft: dict[str, Any] = {}
    for t in (CM.falling, CM.reference):
        w = job.task(t)
        for s in CM.seeds:
            n = f"{t}_{s}"
            dW = torch.load(E46 / "large" / f"{n}.pt", weights_only=False, map_location="cpu")
            g = gates_at(job, w, dW)
            ft[n] = {"dW": dW, "gates": g, "on": switched_on(job, w, g)}
            L01, nothing = masked(job, w, g), a1["c0"][n]["nothing"]
            out["fine_tunes"][n] = {"L00": w["base"]["L"], "L01": L01, "gain": nothing["gain"], "S_on": members(ft[n]["on"]), "n_on": recruit.size(ft[n]["on"])}
            near(out, f"{n} do-nothing L01 against Amendment 1's", L01, nothing["cells"]["L01"], CM.read_tol)
            log(f"[fine-tune {n}] {json.dumps(out['fine_tunes'][n])}")
    on = lambda t, s: ft[f"{t}_{s}"]["on"]
    a = out["anchors"]
    for t in (CM.falling, CM.reference):
        a[f"J {t} seeds"] = recruit.jaccard(on(t, CM.seeds[0]), on(t, CM.seeds[1]))
    a[f"J {CM.falling} against {CM.reference}, largest"] = max(recruit.jaccard(on(CM.falling, s), on(CM.reference, r)) for s in CM.seeds for r in CM.seeds)
    w = job.task(CM.falling)
    for s in CM.seeds:  # the ladder's positive control: W0 to the fine-tune, on the term itself. Reported, no stop
        n = f"{CM.falling}_{s}"
        back = restored(job, w, w["base"]["gates"], ft[n]["gates"], {})
        term = out["fine_tunes"][n]["L01"] - w["base"]["L"]
        out["ladder_control"][n] = {"term": term, **back, "label": p4_state(term, back["at_trigger"], back["ladder"])}
    log(f"[anchors] {json.dumps(a)}")
    log(f"[ladder control] {json.dumps({n: {k: v for k, v in r.items() if k != 'top'} for n, r in out['ladder_control'].items()})}")
    dump(out)
    return ft


def student(job: Job, out: dict[str, Any], a1: dict[str, Any], ft: dict[str, Any], t: str, s: str, order: int) -> dict[str, Tensor]:
    """Train one student, read it, write it. Order-0 students must reproduce Amendment 1's L01 or the job stops. Returns its switched-on set."""
    n, name = f"{t}_{s}", f"{t}_{s}" + (f"_o{order}" if order else "")
    w, f = job.task(t), ft[n]
    dW, comps, curve = job.train(w, teacher.teacher_net(job.model, f["dW"]), None, order, STATES / f"{name}.pt", {"arm": "student", "task": t, "seed": s})
    a, g = job.read(w, f"{name} a", dW, comps), gates_at(job, w, dW)
    L01, L01_ft = masked(job, w, g), out["fine_tunes"][n]["L01"]
    near(out, f"{name} L01 by gates against the route split's", L01, a["cells"]["L01"], CM.zero_tol)
    if order == 0:
        near(out, f"{name} L01 against Amendment 1's", L01, a1["students"][name]["a"]["cells"]["L01"], CM.repro_tol)
    if not out["students"]:  # the restore's path, once
        everything = {k: torch.arange(job.model.module_to_c[k]) for k in job.alive}
        near(out, "restoring every value everywhere gives the fine-tune's L01", masked(job, w, recruit.hybrid(g, f["gates"], everything)), L01_ft, CM.zero_tol)
    on = switched_on(job, w, g)
    A, B = recruit.members(f["on"]), recruit.members(on)
    alive = recruit.members({k: v.nonzero().flatten() for k, v in job.alive.items()})
    parts = {"dropped": A - B, "kept": A & B, "added": B - A, "outside": alive - A - B}
    back = restored(job, w, g, f["gates"], {b: sel_of(parts[b], list(job.alive)) for b in BLOCKS})
    half = w["heldout"].half
    torch.save({"student": {k: v[:, half].half() for k, v in recruit.cat_gates(g).items()}, "fine_tune": {k: v[:, half].half() for k, v in recruit.cat_gates(f["gates"]).items()}},
               STATES / f"trigger_{name}.pt")
    out["students"][name] = {"task": t, "seed": s, "order": order, "fine_tune": n, "a": a, "curve": curve, "L01": L01, "fall": L01_ft - L01, "term_lost": (L01_ft - L01) / (L01_ft - w["base"]["L"]),
                             "J": recruit.jaccard(f["on"], on), "n_on": recruit.size(on), "S_on": members(on), "n_parts": {b: len(parts[b]) for b in BLOCKS},
                             "dropped": sorted(parts["dropped"]), "added": sorted(parts["added"]), "restored": back, "distance": distance(dW, f["dW"])}
    rec = out["students"][name]
    log(f"[student {name}] {json.dumps({k: rec[k] for k in ('L01', 'fall', 'term_lost', 'J', 'n_on', 'n_parts', 'distance')})}")
    log(f"[restored {name}] {json.dumps({k: v for k, v in back.items() if k != 'top'})}")
    dump(out)
    return on


def main() -> None:
    if OUT.is_file():
        sys.exit(f"{OUT} exists. Move it aside to start again")
    a1 = json.loads(A1_FILE.read_text())
    STATES.mkdir(parents=True, exist_ok=True)
    card = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    head = subprocess.run(["git", "-C", str(RESULTS.parents[2]), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    out: dict[str, Any] = {"card": card, "commit": head, "config": vars(CM), "predictions_sha256": sha256(PRED), "amend1_sha256": sha256(A1_FILE),
                           "fine_tunes": {}, "anchors": {}, "ladder_control": {}, "students": {}}
    check(out, "same card as Amendment 1 (1 = yes)", float(card == a1["card"]), card == a1["card"])
    log(f"[start] {card} {head}")
    job = Job("cuda" if torch.cuda.is_available() else "cpu", json.loads((E46 / "stage0a.json").read_text())["groups"], out)
    ft = fine_tunes(job, out, a1)  # stage 0: no training
    on = {f"{t}_{s}": student(job, out, a1, ft, t, s, 0) for s in CM.seeds for t in (CM.falling, CM.reference)}  # the control: rand32b s0, then rand32 s0, each stops on a miss
    for s in CM.seeds:
        n = f"{CM.falling}_{s}"
        out["anchors"][f"J {n} order 0 against order {CM.second_order}"] = recruit.jaccard(on[n], student(job, out, a1, ft, CM.falling, s, CM.second_order))
    out["score"], out["done"] = score(out), True
    dump(out)
    log(f"[score] {json.dumps(out['score'])}")


if __name__ == "__main__":
    main()
