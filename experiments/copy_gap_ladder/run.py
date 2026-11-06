"""E121 copy_gap_ladder: why does the decomposition after teacher training (the copy) sit at input-weighted cosine
0.38-0.40 to its fine-tune's change dW, where a rerun of the fine-tune in another batch order sits at 0.83?
Plan reports/plans/2026-11-06_plan_e121-copy-gap-ladder.md; rules, leans and expected values results/predictions.json
(the run refuses unless it is committed unchanged and its `config` equals CFG).

The ladder, each rung one difference from the last, all on the real (emoticon) task's 1,000-step Adam fine-tune:
R0 the fine-tune rerun (Amendment 10's saved reruns); R1 a plain-weights student (KL to the fine-tuned model, the
fine-tune's lr); R2 R1 at the copy's lr; R3 the copy (component training, E34's recipe). Beside R3: R3b the copy with
plain SGD, R4 the copy started from dW's least-squares split. Scored at batch orders 1, 2 (dW is order 0).

Stages, each written as it finishes (`STAGE n DONE` in the log); a stop is `stopped` in the file, exit 1:
  0  no model: the saved states' Frobenius cosines against Amendment 10's (control)
  1  input moments on held-out task windows and Pile; Amendment 10's input-weighted cells reread (control)
  2  the fine-tune regenerated (control, the teacher); the copy regenerated at A8c's orders 0, 1 (control)
  3  the other students; one that might not finish inside budget_s exits 3 for a resubmission (fleet/resubmit --code 3)
  4  reads against dW, the explanation split (dW's part and the remainder), the score
Usage (repo root): sbatch cluster/copy_gap_ladder.sbatch. A rerun continues: a state's file in results/large means its
record is complete. Writes results/copy_gap_ladder.json and results/large/<state>.pt (bf16 change, F0.pt fp32).
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
from rules import CFG, RUNGS, SETS, copies, score
from torch import Tensor

from commute import split, teacher, vpd, weighted
from commute.data import pile_rows
from commute.faithfulness import model_state
from commute.finetune import ArmAConfig, eval_arm_a, finetune_arm_a, finetune_with_loss
from commute.jobs import hard_exit, require_filed
from commute.saved_states import offsets_state

HERE, EXP = Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]
RESULTS, LARGE = HERE / "results", HERE / "results" / "large"
PRED, OUT = RESULTS / "predictions.json", RESULTS / "copy_gap_ladder.json"
TT = EXP / "teacher_training" / "results"
A8C, A10_LARGE, A10 = TT / "large" / "amend8" / "amend8c", TT / "large" / "amend10", TT / "amend10.json"
TRANSPORT = EXP / "transport_test" / "results"
SAVED = {"dW": A8C / "real_teacher_dW.pt", "R0_o1": A10_LARGE / "real_teacher_o1_dW.pt", "R0_o2": A10_LARGE / "real_teacher_o2_dW.pt"}
A10_NAME = {"dW": "real_teacher", "R0_o1": "real_teacher_o1", "R0_o2": "real_teacher_o2", "A10_o0": "real_o0", "A10_o1": "real_o1"}
CELLS = (("A10_o0", "dW"), ("A10_o1", "dW"), ("R0_o1", "dW"), ("R0_o2", "dW"), ("R0_o2", "R0_o1"))


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def dump(out: dict[str, Any]) -> None:
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True, default=float))


def halt(out: dict[str, Any], why: str) -> None:
    out["stopped"] = why
    dump(out)
    log(f"STOP: {why}")
    hard_exit(1)


# ----------------------------------------------------------------------------- the model's reads


@dataclass
class Ctx:
    model: Any
    device: str
    W0: dict[str, Tensor]
    uv_cpu: dict[str, tuple[Tensor, Tensor]]
    train: Any
    held: Any
    token: int
    sigma: dict[str, dict[str, Tensor]]
    tnet: Any = None
    z0: Tensor | None = None
    dz_ref: Tensor | None = None


@torch.no_grad()
def trigger_logits(net: Any, held: Any, device: str) -> Tensor:
    """fp32 logits at the trigger, minus their mean over the vocabulary (the softmax's free constant), (N, V)."""
    out = []
    for s in range(0, len(held), 64):
        lg = net(held.tokens[s : s + 64].to(device))
        lg = (lg[0] if isinstance(lg, tuple) else lg)[:, held.half].float()
        out.append(lg - lg.mean(-1, keepdim=True))
    return torch.cat(out)


def behaviour(c: Ctx, dW: dict[str, Tensor]) -> dict[str, float]:
    """At W0 + dW on the held-out windows: the teacher gap (arm A's weighting), P(target), and the cosine of the change
    in trigger logits with the saved fine-tune's (`fcos`; nan for no change)."""
    with model_state(c.model, target={k: w + dW[k].to(w.device, w.dtype) for k, w in c.W0.items()}):
        net = c.model.target_model
        gap = teacher.heldout_gap(net, c.tnet, c.held, c.device)
        p = eval_arm_a(net, c.tnet, c.held, c.token, c.device)["p_o"]
        z = trigger_logits(net, c.held, c.device)
    assert c.z0 is not None and c.dz_ref is not None, "stage 2 sets W0's logits and dW's change first"
    d = (z - c.z0).flatten().double()
    n = float(d.norm() * c.dz_ref.norm())
    return {"gap": gap, "p_target": p, "fcos": float(d @ c.dz_ref) / n if n > 0 else math.nan}


def cos_table(vec: dict[str, dict[str, Tensor]], sigma: dict[str, Tensor] | None, device: str) -> tuple[list[str], Tensor, Tensor]:
    G = weighted.gram_of(vec, sigma, device)
    return list(vec), weighted.cosines(G), G


def reread(out: dict[str, Any], label: str, names: list[str], C: Tensor, a10: dict[str, float], tol: float) -> None:
    got = {f"{a} | {b}": float(C[names.index(a), names.index(b)]) for a, b in CELLS}
    worst = max(abs(got[f"{a} | {b}"] - a10[f"{A10_NAME[a]} | {A10_NAME[b]}"]) for a, b in CELLS)
    out["controls"][label] = {"cells": got, "worst": worst}
    log(f"[control {label}] {json.dumps(out['controls'][label])}")
    dump(out)
    if not worst <= tol:
        halt(out, f"{label}: worst {worst} > {tol}")


def load_bf16(path: Path) -> dict[str, Tensor]:
    return {k: v.float() for k, v in torch.load(path, weights_only=False, map_location="cpu").items()}


# ----------------------------------------------------------------------------- the students


def train_rung(c: Ctx, rung: str, order: int, init: Any) -> dict[str, Tensor]:
    kind, opt, lr_name, _ = RUNGS[rung]
    lr = getattr(CFG, lr_name)
    if kind == "plain":
        cfg = ArmAConfig(o_token=c.token, lr=lr, batch=CFG.batch, steps=CFG.steps, checkpoints=(CFG.steps,), seed=order, optimizer=opt)
        res = finetune_with_loss(c.model, c.train, cfg, lambda n, t: teacher.teacher_loss(n, c.tnet, t, c.train.half), c.device, log)
        return res.deltas[CFG.steps]
    uv = {k: (u.to(c.device), v.to(c.device)) for k, (u, v) in c.uv_cpu.items()}
    st, _ = teacher.train_student(c.model, uv, c.W0, c.tnet, c.train.tokens, c.held, c.token, CFG.batch, c.train.half, lr, CFG.steps,
                                  c.device, log, seed=order, optimizer=opt, init=init)
    return {k: v[0] for k, v in st.items()}


def ls_start(c: Ctx, dW: dict[str, Tensor]) -> tuple[dict[str, tuple[Tensor, Tensor]], dict[str, Tensor]]:
    """dW's least-squares split as offsets, and the weight change they make, (U + dU)(V + dV)^T - U V^T."""
    b = split.bases(c.uv_cpu, CFG.ls_tol)
    init, start = {}, {}
    for k, (U, V) in c.uv_cpu.items():
        dU, dV = (t.float() for t in split.least_squares_split(dW[k].double(), b[k], U.double(), V.double()))
        init[k] = (dU, dV)
        start[k] = ((U.double() + dU.double()) @ (V.double() + dV.double()).T - U.double() @ V.double().T).float()
    return init, start


def control_copy(out: dict[str, Any], c: Ctx, name: str, dW: dict[str, Tensor]) -> None:
    """The regenerated copy against A8c's saved one: Frobenius, and its input-weighted cosine to dW against Amendment 10's
    within that amendment's batch-order noise on each set."""
    saved, ref = offsets_state(A8C / f"real_{name.split('_')[1]}.pt", c.uv_cpu)[0], load_bf16(SAVED["dW"])
    a10 = json.loads(A10.read_text())
    frob = float(cos_table({"x": dW, "saved": saved}, None, c.device)[1][0, 1])
    rec: dict[str, Any] = {"frobenius_to_saved": frob}
    ok = 1 - frob <= CFG.saved_tol
    for s, st in SETS.items():
        got = float(cos_table({"x": dW, "dW": ref}, c.sigma[s], c.device)[1][0, 1])
        want = a10["reads"][st]["uncentred"]["whole"][f"real_{name.split('_')[1]} | real_teacher"]
        tol = a10["score"]["by_input"][s]["uncentred"]["real"]["noise"]["batch_order"]
        rec[s] = {"cos": got, "a10": want, "tol": tol}
        ok = ok and abs(got - want) <= tol
    out["controls"][f"copy {name}"] = rec
    log(f"[control copy {name}] {json.dumps(rec)}")
    if not ok:
        halt(out, f"copy {name} does not reproduce: {json.dumps(rec)}")


def state(out: dict[str, Any], c: Ctx, name: str, t0: float, init: Any = None) -> None:
    """Train one student unless its file exists; its record, then its file. Exits 3 if it might not finish inside budget_s."""
    path = LARGE / f"{name}.pt"
    if path.is_file():
        return
    rung, order = name.rsplit("_o", 1)
    est = CFG.plain_s if RUNGS[rung][0] == "plain" else CFG.student_s
    if time.time() - t0 + est > CFG.budget_s:
        log(f"[budget] {name} needs ~{est:.0f} s after {time.time() - t0:.0f} s: exit 3 for a resubmission")
        dump(out)
        hard_exit(3)
    t1 = time.time()
    dW = train_rung(c, rung, int(order), init)
    out["behaviour"][name] = {**behaviour(c, dW), "seconds": time.time() - t1}
    log(f"[state {name}] {json.dumps(out['behaviour'][name])}")
    if rung == "R3" and int(order) in CFG.control_orders:
        control_copy(out, c, name, dW)
    torch.save({k: v.bfloat16() for k, v in dW.items()}, path)
    dump(out)


# ----------------------------------------------------------------------------- stages


def stage1(out: dict[str, Any], c: Ctx, vec: dict[str, dict[str, Tensor]]) -> None:
    """Input moments at W0, then Amendment 10's input-weighted cells reread from the same saved files."""
    t1 = time.time()
    c.sigma["task"] = weighted.input_moments(c.model, c.held.tokens, CFG.batch_windows)
    c.sigma["pile"] = weighted.input_moments(c.model, pile_rows("val", 0, CFG.pile_rows, log), CFG.batch_pile)
    log(f"[moments] {time.time() - t1:.0f} s")
    a10 = json.loads(A10.read_text())["reads"]
    for s, st in SETS.items():
        names, C, _ = cos_table(vec, c.sigma[s], c.device)
        reread(out, f"reread {s}", names, C, a10[st]["uncentred"]["whole"], CFG.read_tol)


def stage2(out: dict[str, Any], c: Ctx, vec: dict[str, dict[str, Tensor]], t0: float) -> None:
    """The fine-tune regenerated (the teacher), the behaviour line, then the copy at A8c's orders."""
    path = LARGE / "F0.pt"
    if not path.is_file():
        t1 = time.time()
        cfg = ArmAConfig(o_token=c.token, lr=CFG.ft_lr, batch=CFG.batch, steps=CFG.steps, checkpoints=(CFG.steps,), seed=0, optimizer="adam")
        res = finetune_arm_a(c.model, c.train, c.held, cfg, c.device, log)
        F0, ev = res.deltas[CFG.steps], res.evals[CFG.steps]
        rec = {"p_target": ev["p_o"], "size": ev["rel_norm_max"], "seconds": time.time() - t1,
               "frobenius_to_saved": float(cos_table({"F0": F0, "dW": vec["dW"]}, None, c.device)[1][0, 1])}
        out["controls"]["fine-tune"] = rec
        log(f"[control fine-tune] {json.dumps(rec)}")
        if not (rec["p_target"] >= CFG.t_p_min and 1 - rec["frobenius_to_saved"] <= CFG.saved_tol):
            halt(out, f"fine-tune does not reproduce: {json.dumps(rec)}")
        torch.save(F0, path)
    F0 = torch.load(path, weights_only=False, map_location="cpu")
    c.tnet = teacher.teacher_net(c.model, F0)
    c.z0 = trigger_logits(c.model.target_model, c.held, c.device)
    with model_state(c.model, target={k: w + vec["dW"][k].to(w.device) for k, w in c.W0.items()}):
        c.dz_ref = (trigger_logits(c.model.target_model, c.held, c.device) - c.z0).flatten().double()
    zero = {k: torch.zeros_like(v) for k, v in vec["dW"].items()}
    for name, dW in {"W0": zero, "F0": F0, **vec}.items():
        out["behaviour"][name] = behaviour(c, dW)
    out["line"] = {"gap_w0": out["behaviour"]["W0"]["gap"], "p_w0": out["behaviour"]["W0"]["p_target"], "p_teacher": out["behaviour"]["F0"]["p_target"]}
    log(f"[behaviour line] {json.dumps(out['behaviour'])}")
    dump(out)
    for o in CFG.control_orders:
        state(out, c, f"R3_o{o}", t0)


def stage3(out: dict[str, Any], c: Ctx, vec: dict[str, dict[str, Tensor]], t0: float) -> None:
    """The ladder's students first, then the two beside the copy; R4 from dW's least-squares split."""
    for r in ("R3", "R1", "R2", "R3b"):
        for o in CFG.orders:
            state(out, c, f"{r}_o{o}", t0)
    names = ["R4_start", *(f"R4_o{o}" for o in CFG.orders)]
    if all((LARGE / f"{n}.pt").is_file() for n in names):
        return
    init, start = ls_start(c, vec["dW"])
    if not (LARGE / "R4_start.pt").is_file():
        out["behaviour"]["R4_start"] = behaviour(c, start)
        log(f"[state R4_start] {json.dumps(out['behaviour']['R4_start'])}")
        torch.save({k: v.bfloat16() for k, v in start.items()}, LARGE / "R4_start.pt")
        dump(out)
    for n in names[1:]:
        state(out, c, n, t0, init)


def explain_one(c: Ctx, G: dict[str, Tensor], names: list[str], n: str, dW: dict[str, Tensor], out: dict[str, Any]) -> dict[str, Any]:
    """S = k dW + R, k fitted on task windows: R's input-weighted size over S's on each set, and the behaviour of k dW alone."""
    i, j = names.index(n), names.index("dW")
    k = float(G["task"][i, j] / G["task"][j, j])
    rem = {s: math.sqrt(max(0.0, float(g[i, i] - 2 * k * g[i, j] + k * k * g[j, j])) / float(g[i, i])) for s, g in G.items()}
    b = behaviour(c, {key: k * v for key, v in dW.items()})
    return {"k": k, "remainder_size": rem, "without_remainder": b, "gap_change": b["gap"] - out["behaviour"][n]["gap"],
            "remainder": "idle" if copies(b, out["line"]) else "carries behaviour"}


def stage4(out: dict[str, Any], c: Ctx, vec: dict[str, dict[str, Tensor]]) -> None:
    states = [f"{r}_o{o}" for r in RUNGS for o in CFG.orders]
    names = ["dW", "R0_o1", "R0_o2", "R3_o0", "R4_start", *states]
    v4 = {n: vec[n] if n in vec else load_bf16(LARGE / f"{n}.pt") for n in names}
    G, reads = {}, {}
    for s in SETS:
        _, C, G[s] = cos_table(v4, c.sigma[s], c.device)
        D, Z, j = weighted.distances(G[s]), weighted.sizes(G[s]), names.index("dW")
        reads[s] = {"cos": {n: float(C[i, j]) for i, n in enumerate(names)}, "dist": {n: float(D[i, j]) for i, n in enumerate(names)},
                    "size": {n: float(Z[i, j]) for i, n in enumerate(names)}, "R0_pair": float(C[1, 2])}
    out["reads"] = reads
    log(f"[reads] {json.dumps(reads)}")
    dump(out)
    out["explain"] = {n: explain_one(c, G, names, n, v4["dW"], out) for n in ("R0_o1", "R0_o2", *states)}
    log(f"[explain] {json.dumps(out['explain'])}")
    out["score"], out["done"] = score(reads, out["behaviour"], out["line"], out["explain"]), True
    dump(out)
    log(f"[score] {json.dumps(out['score'])}")


def main() -> None:
    require_filed(PRED)
    assert json.loads(PRED.read_text())["config"] == json.loads(json.dumps(asdict(CFG))), "predictions were filed for another config"
    if not LARGE.is_dir():
        sys.exit(f"{LARGE} missing: run ~/commute-agents/fleet/link-data <role> first")
    out = json.loads(OUT.read_text()) if OUT.is_file() else {"controls": {}, "behaviour": {}, "jobs": []}
    if "stopped" in out:
        sys.exit(f"{OUT} has a stop ({out['stopped']}): move it aside to start again")
    if out.get("done"):
        log("done already")
        hard_exit(0)
    t0, device = time.time(), "cuda"
    out["jobs"].append({"card": torch.cuda.get_device_name(), "start": time.strftime("%Y-%m-%dT%H:%M:%S")})
    uv_cpu = torch.load(A8C / "uv0.pt", weights_only=False, map_location="cpu")
    vec = {n: load_bf16(p) for n, p in SAVED.items()} | {f"A10_o{o}": offsets_state(A8C / f"real_o{o}.pt", uv_cpu)[0] for o in (0, 1)}
    names, C, _ = cos_table(vec, None, device)
    reread(out, "reread frobenius", names, C, json.loads(A10.read_text())["reads"]["frobenius"]["uncentred"]["whole"], CFG.read_tol)
    log("STAGE 0 DONE")
    model = vpd.load_component_model(device)
    win = torch.load(TRANSPORT / "large" / "windows.pt", weights_only=False, map_location="cpu")
    token = json.loads((TRANSPORT / "windows.json").read_text())["o_token_id"]
    c = Ctx(model, device, vpd.target_weights(model), uv_cpu, win["train"], win["heldout"], token, {})
    stage1(out, c, vec)
    log("STAGE 1 DONE")
    stage2(out, c, vec, t0)
    log("STAGE 2 DONE")
    stage3(out, c, vec, t0)
    log("STAGE 3 DONE")
    stage4(out, c, vec)
    log("STAGE 4 DONE")
    hard_exit(0)


if __name__ == "__main__":
    main()
