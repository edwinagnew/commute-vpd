"""E123 (case_label_both), 2026-11-06: E112's case-label pair read by the decomposition side, beside the frozen reuse
definition's reading of the same states (decisions.md rows 81 and 87; E112's verdicts, closed, not re-read here).
Plan reports/plans/2026-11-06_plan_e123-case-label-both-readings.md; rules, leans and expected values in
results/predictions.json (the run refuses unless it is committed unchanged and its `config` equals rules.CFG).

E112's four saved states (real s0/s1 at step 30, twin s0/s1 at step 300); no fine-tune is rerun. Every decomposition
read is on the first n_read of E112's fit windows (train-pair words, rows the fine-tune never trained on) with the
state's label as a 42nd column: target data.LAST, ci_len 41, so neither the model nor Gamma ever sees the label.

Stages, each record written as it is read, `STAGE n DONE` in the log; a missed control is `stopped` in the file, exit 1:
  0  controls, alone. No model: the moved rules on E12 A1's and E38 A2's saved files. Then inputs' sha256; W0's
     held-out accuracy and each state's fit-window gains against E112 and E114; the route-split path against the
     removal path at W0; per state the recruited set S and E38 A2's removal controls C1-C4
  1  the route split of each fine-tune: do-nothing and least squares (E119's `references`), E12's lines
  2  E38 A2's removal test per state, trigger only: D(S) against 20 random same-size sets
  3  teacher training (E34's arm-A student) of each state, read on its own state; a student that might end past
     budget_s exits 3 for a resubmission (fleet/resubmit --code 3); a student with a record and a file is skipped
  4  the score: per state, per arm and the joint reading of each measure
    sbatch cluster/case_label_both.sbatch
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from rules import ARMS, CFG, arm_of, copy_label, removal_class, score
from torch import Tensor

from commute import anchors as AN
from commute import arms, component_training, saved_states, teacher, vpd
from commute import recruit as R
from commute import recruitment as RT
from commute import regating as G
from commute import steering as ST
from commute.copying import with_target
from commute.data import LAST, Windows
from commute.faithfulness import model_state
from commute.identity import log_odds
from commute.jobs import hard_exit, require_filed
from commute.keydep import sha256
from commute.local_faith import route_cells, route_split, trigger_mean_gates
from commute.open_pair import jsonable
from commute.recruitment import d_seed_class
from commute.route_labels import e12_control, label
from commute.signatures import ablated_weights, top_by

HERE, EXP = Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]
RESULTS, LARGE = HERE / "results", HERE / "results" / "large"
PRED, OUT = RESULTS / "predictions.json", RESULTS / "case_label_both.json"
E112 = EXP / "case_label" / "results"
E114 = EXP / "case_locator" / "results" / "stage0.json"
ALIVE = EXP / "transport_test" / "results" / "large"  # alive.pt, E34's and E38's
E12A1 = EXP / "new_vs_existing" / "results" / "new_vs_existing_amend1.json"
E38A2 = EXP / "recruitment_test" / "results" / "amend2.json"
H, L = CFG.half, CFG.ci_len


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def dump(out: dict[str, Any]) -> None:
    tmp = OUT.with_suffix(".tmp")  # a kill mid-write must not leave unreadable JSON
    tmp.write_text(json.dumps(jsonable(out), indent=1, sort_keys=True, default=float))
    tmp.replace(OUT)


def check(out: dict[str, Any], name: str, value: Any, ok: bool) -> None:
    """A filed stop rule: logged and kept in the file; a miss stops the job (exit 1)."""
    log(f"[check {name}] {json.dumps(jsonable(value), default=float)}: {ok}")
    out.setdefault("checks", {})[name] = {"value": value, "pass": ok}
    if not ok:
        out["stopped"] = name
    dump(out)
    if not ok:
        log(f"STOP: {name}")
        hard_exit(1)


def wins(toks: Tensor) -> Windows:
    n = len(toks)
    return Windows(toks, torch.zeros(n), torch.zeros(n, dtype=torch.long), torch.zeros(n, dtype=torch.long), H)


class Job:
    """The model (fp32, ST.Net as E112 and E114), E112's task and states, and W0's reads on the read windows."""

    def __init__(self, device: str) -> None:
        self.device, self.net = device, ST.Net(device)
        self.model, self.W0 = self.net.model, self.net.W0
        self.comps = vpd.component_weights(self.model)
        self.alive = saved_states.load_alive(ALIVE)
        self.uv = component_training.alive_uv(self.model, self.alive)
        self.uv_cpu = {k: (u.cpu(), v.cpu()) for k, (u, v) in self.uv.items()}
        self.lift = AN.lift_context(self.model, self.alive)
        path = E112 / "large" / "task.pt"
        if sha256(path) != CFG.task_sha256:
            sys.exit(f"{path}: sha256 is not E114's filed one")
        self.t = torch.load(path, weights_only=False)
        self.train = wins(self.t["train"])
        self.dW = {n: self.load_state(n) for n in CFG.states}
        m0 = trigger_mean_gates(self.model, self.t["train"], H)
        self.read, self.task, self.fit0, self.S = {}, {}, {}, {}
        for a in ARMS:
            self.read[a] = with_target(wins(self.t["fit"][: CFG.n_read]), self.target(a, self.t["fit"][: CFG.n_read])).tokens
            base = route_cells(self.model, self.read[a], H, LAST, 64, L)
            self.task[a] = G.Task(a, self.train, wins(self.read[a]), LAST, base, m0)
            self.fit0[a] = self.label_reads(self.t["fit"], a)

    def load_state(self, n: str) -> dict[str, Tensor]:
        rec = json.loads((E112 / "stage1.json").read_text())["states"][n]
        path = E112 / "large" / f"dW_{n}.pt"
        if sha256(path) != rec["sha256"]:
            sys.exit(f"{path}: sha256 is not E112's stage1.json's")
        return {k: v.float() for k, v in torch.load(path, weights_only=True, map_location="cpu").items()}

    def y(self, a: str, toks: Tensor) -> Tensor:
        return torch.tensor([self.t["y"][a][int(w)] for w in toks[:, H - 1]])

    def target(self, a: str, toks: Tensor) -> Tensor:
        A, B = self.t["ab"]
        return torch.where(self.y(a, toks) == 1, A, B)

    def weights(self, n: str) -> dict[str, Tensor]:
        return {k: w + self.dW[n][k].to(w.device, w.dtype) for k, w in self.W0.items()}

    @torch.no_grad()
    def label_reads(self, toks: Tensor, a: str, dW: dict[str, Tensor] | None = None) -> tuple[Tensor, Tensor]:
        """E112's and E114's fp32 path: per window, logit A - logit B and the log-odds of the window's own label."""
        A, B = self.t["ab"]
        tgt, d, lo = self.target(a, toks), [], []
        with self.net.at(dW):
            for s in range(0, len(toks), 64):
                lg = self.net.run(toks[s : s + 64, :L])[0][:, H]
                d.append((lg[:, A] - lg[:, B]).double().cpu())
                lo.append(log_odds(lg, tgt[s : s + 64].to(lg.device)).double().cpu())
        return torch.cat(d), torch.cat(lo)

    def behaviour(self, a: str, dW: dict[str, Tensor]) -> dict[str, float]:
        """On all 1,024 fit windows: label accuracy (E112's acc_fit), the label-gap gain (E112's G) and the log-odds gain
        (E114's G lo), both against W0."""
        (d0, lo0), (d1, lo1), y = self.fit0[a], self.label_reads(self.t["fit"], a, dW), self.y(a, self.t["fit"])
        return {"acc": float(((d1 > 0).long() == y).double().mean()), "gap_gain": float(((d1 - d0) * (2 * y - 1)).mean()),
                "lo_gain": float((lo1 - lo0).mean())}

    def route(self, a: str, name: str, dW: dict[str, Tensor], comps: Any) -> dict[str, Any]:
        """The route split at target W0 + dW with components `comps` (None: released). Logged whole."""
        with model_state(self.model, target={k: w + dW[k].to(w.device, w.dtype) for k, w in self.W0.items()}, components=comps):
            rs = route_split(self.model, self.read[a], H, LAST, self.task[a].base, self.comps, 64, L)
        log(f"[read {name}] {json.dumps(rs, default=float)}")
        return rs

    def lo_ref(self, W: dict[str, Tensor], a: str) -> tuple[float, list[Tensor]]:
        """The full model at W on the read windows: its trigger lo and its own logits (the KL reference)."""
        with model_state(self.model, target=W):
            ref = RT.window_logits(self.model, self.read[a], ci_len=L)
            return RT.readout(self.model, self.read[a], H, LAST, ref, ci_len=L)["lo"], ref

    def removed_lo(self, W: dict[str, Tensor], sel: R.Sel, a: str, ref: list[Tensor], everywhere: bool = False) -> dict[str, float]:
        return RT.removed_at(self.model, W, self.comps, sel, self.read[a], H, LAST, ref, everywhere=everywhere, ci_len=L)

    def S_of(self, n: str) -> R.Sel:
        """E38's recruited set: alive subcomponents whose mean trigger gate over the training windows rises by > thr."""
        if n not in self.S:
            shift = G.shift_of(self.model, self.W0, self.dW[n], self.task[arm_of(n)], self.alive)[0]
            torch.save(shift, LARGE / f"shift_{n}.pt")
            self.S[n] = R.switched(shift, CFG.thr)
        return self.S[n]


# ----------------------------------------------------------------------------- stage 0: controls


def no_model_controls(out: dict[str, Any]) -> None:
    """The moved rules on saved files: E12 A1's students (4 copy re-switching, 4 maps masked-off, as E119 read them)
    and E38 A2's six seed classes (4 through, 2 not)."""
    e12 = e12_control(E12A1, CFG.lines)
    want = {"copy": "re-switching", "map": "masked-off"}
    check(out, "E12 A1 students by the moved route rule", {n: v["label"] for n, v in e12.items()},
          len(e12) == 8 and all(v["label"] == want[v["class"]] for v in e12.values()))
    a2 = json.loads(E38A2.read_text())
    got = {n: d_seed_class(r["D"]["trigger"]["S"], r["D"]["trigger"]["random"]) for n, r in a2["states"].items()}
    check(out, "E38 A2 seed classes reread", got, got == a2["P5"]["seed_class"])


def removal_controls(job: Job, out: dict[str, Any], n: str, ref0: list[Tensor], T0: float) -> None:
    """E38 A2's C1-C4 on one state, at W0 and at the state; any miss stops the job."""
    a, S = arm_of(n), job.S_of(n)
    W1 = job.weights(n)
    T1, ref1 = job.lo_ref(W1, a)
    empty = {k: torch.zeros(0, dtype=torch.long) for k in job.alive}
    c: dict[str, Any] = {"n_S": R.size(S), "C1": {}, "C2": {}, "C3": {}}
    for lab, W, ref, T in (("W0", job.W0, ref0, T0), ("W1", W1, ref1, T1)):
        c["C1"][lab] = job.removed_lo(W, S, a, ref, everywhere=True)["lo"] - RT.removed(
            job.model, W, job.comps, S, job.read[a], H, LAST, ref, ci_len=L)["lo"]
        with model_state(job.model, target=W), RT.removal_hooks(job.model, ablated_weights(W, job.comps, S), H):
            lg = RT.window_logits(job.model, job.read[a], ci_len=L)
        c["C2"][lab] = max(float((x[:, :H] - y[:, :H]).abs().max()) for x, y in zip(lg, ref, strict=True))
        c["C3"][lab] = job.removed_lo(W, empty, a, ref)["lo"] - T
        del lg
    del ref1
    busy = top_by(job.task[a].m0, job.alive, R.size(S))
    b = job.removed_lo(job.W0, busy, a, ref0)
    c["C4"] = {"n": R.size(busy), "kl_trigger": b["kl_trigger"]}
    out["stage0"]["removal_controls"][n] = c
    check(out, f"{n} C1 hook at every position = weight removal", c["C1"], all(abs(v) <= CFG.hook_tol for v in c["C1"].values()))
    check(out, f"{n} C2 positions before the trigger unchanged", c["C2"], all(v <= CFG.pre_tol for v in c["C2"].values()))
    check(out, f"{n} C3 removing nothing", c["C3"], all(abs(v) <= CFG.empty_tol for v in c["C3"].values()))
    check(out, f"{n} C4 busy removal bites", c["C4"], c["C4"]["n"] < CFG.busy_min_n or c["C4"]["kl_trigger"] >= CFG.busy_kl_min)


def stage0(job: Job, out: dict[str, Any]) -> None:
    rec: dict[str, Any] = {"gains": {}, "path": {}, "removal_controls": {}}
    out["stage0"] = rec
    acc0 = json.loads((E112 / "stage0.json").read_text())["w0_heldout_accuracy"]
    for a in ARMS:
        d, _ = job.label_reads(job.t["held"], a)
        acc = float(((d > 0).long() == job.y(a, job.t["held"])).double().mean())
        check(out, f"W0 held-out accuracy {a} against E112's", {"read": acc, "ref": acc0[a]}, abs(acc - acc0[a]) <= CFG.acc_tol)
    g112 = json.loads((E112 / "stage2.json").read_text())["states"]
    g114 = json.loads(E114.read_text())["guards"]["gain"]
    for n in CFG.states:
        b = rec["gains"][n] = job.behaviour(arm_of(n), job.dW[n])
        refs = {"gap": g112[n]["gain"], "lo": g114[n]["G"]["lo"]}
        check(out, f"{n} fit-window gains against E112's and E114's", {**b, "refs": refs},
              abs(b["gap_gain"] - refs["gap"]) <= CFG.g_tol and abs(b["lo_gain"] - refs["lo"]) <= CFG.g_tol)
    T0: dict[str, tuple[float, list[Tensor]]] = {}
    for a in ARMS:
        T0[a] = job.lo_ref(job.W0, a)
        rec["path"][a] = {"T0_route": job.task[a].base["T"], "T0_removal": T0[a][0]}
        check(out, f"W0 route path against removal path {a}", rec["path"][a], abs(T0[a][0] - job.task[a].base["T"]) <= CFG.path_tol)
    for n in CFG.states:
        removal_controls(job, out, n, T0[arm_of(n)][1], T0[arm_of(n)][0])
    del T0
    rec["pass"] = True
    dump(out)


# ----------------------------------------------------------------------------- stages 1-3


def stage1(job: Job, out: dict[str, Any]) -> None:
    for n in CFG.states:
        if n not in out["routes"]:
            a, dW = arm_of(n), job.dW[n]
            no = job.route(a, f"{n} do-nothing", dW, None)
            ls = job.route(a, f"{n} least squares", dW, AN.lifted(job.model, job.lift, dW))
            sh = ls["shares"]
            out["routes"][n] = {"nothing": no, "least_squares": ls, "label": label(sh["r"], sh["e_gate"], ls["gain"], ls["gain"], CFG.lines)}
            dump(out)
        rec = out["routes"][n]
        no, ls = rec["nothing"]["shares"], rec["least_squares"]["shares"]
        log(f"[route {n}] gain {rec['least_squares']['gain']:.4f} r {ls['r']:.4f} e_gate {ls['e_gate']:.4f} -> {rec['label']}")
        check(out, f"{n} do-nothing e_gate + r", no["e_gate"] + no["r"], abs(no["e_gate"] + no["r"]) <= CFG.zero_tol)
        check(out, f"{n} least-squares r - do-nothing r", ls["r"] - no["r"], abs(ls["r"] - no["r"]) <= CFG.zero_tol)


def removal(job: Job, n: str, T0: float, ref0: list[Tensor]) -> dict[str, Any]:
    """rho_ft(X) = (T1 - lo(W1 without X)) / G, rho_0(X) = (T0 - lo(W0 without X)) / G, D = rho_ft - rho_0, trigger only."""
    a, S, W1 = arm_of(n), job.S_of(n), job.weights(n)
    T1, ref1 = job.lo_ref(W1, a)
    gain = T1 - T0
    g = gain if gain >= CFG.min_gain else 0.0
    rec: dict[str, Any] = {"n_S": R.size(S), "by_layer": R.by_layer(S), "S": {k: v.tolist() for k, v in S.items() if v.numel()},
                           "T0": T0, "T1": T1, "gain": gain}
    if R.size(S):
        reads = []
        for sel in (S, *(R.random_like(S, job.alive, i) for i in range(CFG.n_random))):
            rf = RT.share(T1, job.removed_lo(W1, sel, a, ref1)["lo"], g)
            r0 = RT.share(T0, job.removed_lo(job.W0, sel, a, ref0)["lo"], g)
            reads.append((rf, r0, rf - r0))
        rec["S_reads"], rec["random_reads"] = reads[0], reads[1:]
    del ref1
    D_S = rec["S_reads"][2] if R.size(S) else float("nan")
    D_r = [x[2] for x in rec.get("random_reads", [])]
    rec.update(D_S=D_S, D_random=D_r, beats=sum(D_S > d for d in D_r), cls=removal_class(rec["n_S"], gain, D_S, D_r))
    log(f"[removal {n}] |S| {rec['n_S']} by layer {rec['by_layer']} T0 {T0:.4f} T1 {T1:.4f} G {gain:.4f} | rho_ft, rho_0, D of S "
        f"{json.dumps(rec.get('S_reads'))} | random D {json.dumps([round(x, 4) for x in D_r])} | above {rec['beats']}/{len(D_r)} -> {rec['cls']}")
    return rec


def stage2(job: Job, out: dict[str, Any]) -> None:
    T0: dict[str, tuple[float, list[Tensor]]] = {}
    for n in CFG.states:
        if n not in out["removal"]:
            a = arm_of(n)
            if a not in T0:
                T0 = {a: job.lo_ref(job.W0, a)}  # one W0 reference on the device at a time
            out["removal"][n] = removal(job, n, *T0[a])
            dump(out)
    del T0


def student(job: Job, out: dict[str, Any], n: str) -> None:
    """E34's arm-A student of the state (teacher W0 + dW, KL(teacher || student) on the task windows, Gamma and Delta
    frozen), saved, then read on its own state: target W0 + its change, components its U', V' (E119's state)."""
    a, path, t1 = arm_of(n), LARGE / "students" / f"{n}.pt", time.time()
    net = teacher.teacher_net(job.model, job.dW[n])
    st, curve = teacher.train_student(job.model, job.uv, job.W0, net, job.train.tokens, wins(job.read[a][: CFG.n_eval]), LAST,
                                      CFG.batch_a, H, CFG.student_lr, CFG.steps, job.device, log)
    del net
    arms.save_offsets(path, job.uv_cpu, st, {"state": n, "step": CFG.steps})
    dW = {k: v[0] for k, v in st.items()}
    comps = component_training.components_with(job.model, job.alive, {k: (v[1], v[2]) for k, v in st.items()})
    beh, rs = job.behaviour(a, dW), job.route(a, f"{n} student", dW, comps)
    gain0 = out["routes"][n]["nothing"]["gain"]
    lab = label(rs["shares"]["r"], rs["shares"]["e_gate"], rs["gain"], gain0, CFG.lines)
    out["students"][n] = rec = {"behaviour": beh, "fine_tune_behaviour": out["stage0"]["gains"][n], "a": rs, "teacher_gain": gain0,
                                "gain_ratio": rs["gain"] / gain0, "label": lab, "copy_label": copy_label(beh["acc"], lab, CFG.learn_bar),
                                "curve_last": curve[-1], "seconds": time.time() - t1, "sha256": sha256(path)}
    dump(out)
    log(f"[student {n}] fit accuracy {beh['acc']:.4f} (fine-tune {rec['fine_tune_behaviour']['acc']:.4f}) gap gain {beh['gap_gain']:.4f} "
        f"lo gain {beh['lo_gain']:.4f} | route gain ratio {rec['gain_ratio']:.4f} r {rs['shares']['r']:.4f} e_gate "
        f"{rs['shares']['e_gate']:.4f} -> {lab}; copy {rec['copy_label']}; fine-tune {out['routes'][n]['label']} | {rec['seconds']:.0f} s")


def stage3(job: Job, out: dict[str, Any], t0: float) -> None:
    for n in CFG.states:
        if n in out["students"] and (LARGE / "students" / f"{n}.pt").is_file():
            continue
        if time.time() - t0 + CFG.student_s > CFG.budget_s:
            log(f"[budget] student {n} needs ~{CFG.student_s:.0f} s after {time.time() - t0:.0f} s: exit 3 for a resubmission")
            dump(out)
            hard_exit(3)
        student(job, out, n)


def stage4(out: dict[str, Any]) -> None:
    labels = {"route_fine_tune": {n: r["label"] for n, r in out["routes"].items()},
              "copy": {n: r["copy_label"] for n, r in out["students"].items()},
              "removal": {n: r["cls"] for n, r in out["removal"].items()}}
    out["score"], out["done"] = score(labels), True
    dump(out)
    log(f"[score] {json.dumps(out['score'])}")


def main() -> None:
    require_filed(PRED)
    if json.loads(PRED.read_text())["config"] != json.loads(json.dumps(asdict(CFG))):
        sys.exit("predictions were filed for another config")
    if not LARGE.is_dir():
        sys.exit(f"{LARGE} missing: run ~/commute-agents/fleet/link-data <role> first")
    (LARGE / "students").resolve().mkdir(parents=True, exist_ok=True)
    out = json.loads(OUT.read_text()) if OUT.is_file() else {"jobs": [], "stage0": {}, "routes": {}, "removal": {}, "students": {}}
    if "stopped" in out:
        sys.exit(f"{OUT} has a stop ({out['stopped']}): move it aside to start again")
    if out.get("done"):
        log("done already")
        hard_exit(0)
    t0 = time.time()
    out["jobs"].append({"card": torch.cuda.get_device_name(), "start": time.strftime("%Y-%m-%dT%H:%M:%S")})
    no_model_controls(out)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # E112's and E114's precision
    job = Job("cuda")
    log(f"[start] {time.time() - t0:.0f} s to load; read windows {CFG.n_read}, fit windows {len(job.t['fit'])}")
    if not out["stage0"].get("pass"):
        stage0(job, out)
    log("STAGE 0 DONE")
    stage1(job, out)
    log("STAGE 1 DONE")
    stage2(job, out)
    log("STAGE 2 DONE")
    stage3(job, out, t0)
    log("STAGE 3 DONE")
    stage4(out)
    log("STAGE 4 DONE")
    hard_exit(0)


if __name__ == "__main__":
    main()
