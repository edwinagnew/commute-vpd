"""E299 (known_3seeds), cell P2.77: Fig 4 (known_main) and App. I (known_appendix) on exactly 3 seeds a side. Five new
fine-tunes at seed 2, each by its siblings' recipe and saved at the step its siblings' rule gives, then every reading the
figure shows, by the code that read the siblings. Card ~/commute-data/cards/E299.md; settings, tolerances and expected
values results/predictions.json (the run refuses unless it is committed unchanged). Descriptive: no verdict on reuse.

New states (E136's names; files in results/large/):
  e46/ident32_s2, e46/rand32_s2    E12's arm A (`keydep.elicit_ft`: Adam 3e-5, batch 32, checkpoints 10/30/100/300), kept
                                   at the first checkpoint with held-out P(target) >= 0.5 (E12's rule), fp32 as regating_table
  e53/lowercase_s2, e53/cipher_s2  E53's arm B (`lowercase.arm_b_states`: Adam 3e-5, batch 8, 300 steps, batch order 2),
                                   kept at step 300 in bf16 as performed_anchor
  e54/cipher_own_s2                E54's cipher (`keydep.cipher_ft`) at its own E12 step, fp32 as keydependent_elicit
Readers: E136 share_untrained Job (trained share; SVD, rotated); E143 holes_fill Job.state (TopK neurons k64 and k
matched; fair steering pooled and per key); E98 cells (r_n, r_half) and k_half_fill `carrying` (k_half);
`update_spectrum.state` (erank, norm2); size_matched's size, ||dW|| / ||W0|| over the 24 matrices.

  stage 0  controls, alone -> results/stage0.json: the zero update reads its null on every reader (holes_fill's stage 0
           unchanged, with its three reproductions; E136's zero-update gates and numerator; E98's W0 cells on the five
           tasks; spectrum and size); then e53/cipher_s0 and e54/cipher_own_s0 read again against their saved readings.
  stage 1  -> results/stage1.json: the recipe controls first (seed 1 of ident32, rand32, lowercase and E53's cipher
           retrained; E54's seed-2 matched cipher, from the run that gives cipher_own_s2): the saved step, and the trained
           share within the saved s0/s1 range of its saved share; weight differences reported. Then the new states, saved.
  stage 2  -> results/stage2.json: every reading of the five new states, each written as it lands.
Each stage prints "STAGE n DONE"; a stop prints "STOP: ..." and exits 1. A rerun resumes stages 1 and 2 from their files.

    sbatch --comment=E299 cluster/known_3seeds.sbatch
"""

from __future__ import annotations

import json
import math
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from commute import keydep as KD
from commute import lowercase as LC
from commute import neuron_r as NR
from commute import steering as ST
from commute import update_spectrum as US
from commute.data import pile_rows
from commute.jobs import (
    config_from_settings,
    hard_exit,
    load_experiment,
    manifest_only,
    require_filed,
)
from commute.open_pair import jsonable
from commute.row_copy import mean_logp

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
LARGE, PRED = RESULTS / "large", RESULTS / "predictions.json"
E136 = load_experiment("share_untrained", "run")  # Job: trained share (route_split), SVD and rotated; control_c
HF = load_experiment("holes_fill", "run")  # Job.state: TopK neurons and fair steering; stage0: its controls
E98 = load_experiment("neuron_r", "run")  # files, E63's task loaders, p_learn_of
KH = load_experiment("k_half_fill", "run")  # carrying (k_half), checks
SM = load_experiment("size_matched", "run")  # Readers.size: ||dW|| / ||W0||
TT = load_experiment("regating_truth_table", "run")  # performed_anchor(): E53's run.py, its cipher map and task text
E136.RESULTS, HF.RESULTS = LARGE / "e136_controls", LARGE / "e143_controls"  # their stop records, never in their results
# new state -> (task, file); E98's task name per task; the two old states stage 0 reads again (E136 name -> E98 name)
NEW = {"e46/ident32_s2": ("e46/ident32", "ident32_s2.pt"), "e46/rand32_s2": ("e46/rand32", "rand32_s2.pt"),
       "e53/lowercase_s2": ("e53/lowercase", "dW_lowercase_s2_step300.pt"), "e53/cipher_s2": ("e53/cipher", "dW_cipher_s2_step300.pt"),
       "e54/cipher_own_s2": ("e54/cipher", "cipher_own_s2.pt")}  # fmt: skip
E98_TASK = {"e46/ident32": "ident32", "e46/rand32": "rand32", "e53/lowercase": "lowercase", "e53/cipher": "e53_cipher", "e54/cipher": "cipher"}
REPRO = {"e53/cipher_s0": "e53_cipher_s0", "e54/cipher_own_s0": "cipher_own_s0"}
E136_R, E143_R, E98_R = "experiments/share_untrained/results", "experiments/holes_fill/results", "experiments/neuron_r/results"
log = KD.log
_json: dict[str, Any] = {}


@dataclass(frozen=True)
class Config:
    seed: int  # the new states' seed (batch order for E53)
    control_seed: int  # the seed of the retrained siblings
    reader_share_tol: float
    reader_neuron_tol: float
    spectrum_rel_tol: float
    size_rel_tol: float
    identity_tol: float
    projected_limit_min: float

    def __post_init__(self) -> None:
        assert self.seed == 2 and self.control_seed == 1, "s0 and s1 are saved; the third seed is 2"


class Stop(Exception):
    """A filed stop: the stage's file records why; the job exits 1."""


def load(rel: str) -> Any:
    if rel not in _json:
        _json[rel] = json.loads((REPO / rel).read_text())
    return _json[rel]


def say(tag: str, x: Any) -> None:
    log(f"[{tag}] {json.dumps(jsonable(x))}")


def write(name: str, out: dict[str, Any]) -> None:
    tmp = RESULTS / f"{name}.tmp"  # a kill mid-write must not leave unreadable JSON
    tmp.write_text(json.dumps(jsonable(out), indent=1, sort_keys=True))
    tmp.replace(RESULTS / name)


def check(checks: dict[str, Any], name: str, value: float, tol: float) -> None:
    """|value| <= tol (nan misses), recorded and logged; a miss raises Stop."""
    ok = abs(value) <= tol
    checks[name] = {"value": value, "tol": tol, "pass": ok}
    log(f"[check {name}] {value:.4g} (tol {tol:g}): {ok}")
    if not ok:
        raise Stop(f"{name}: {value:.6g} over {tol:g}")


def rel(a: float, b: float) -> float:
    return abs(a - b) / abs(b)


def inventories() -> tuple[dict[str, Any], dict[str, Any]]:
    """E136's inventory format for the five tasks (E136's Job loads every task it is given) and E136's whole inventory
    with the new states added (holes_fill's Job loads tasks lazily; its stage 0 reads E136's states)."""
    e136 = load(f"{E136_R}/inventory.json")
    new = {n: {"task": t, "file": str((LARGE / f).relative_to(REPO))} for n, (t, f) in NEW.items()}
    tasks = {t: e136["tasks"][t] for t in E98_TASK}
    old = {n: s for n, s in e136["states"].items() if s["task"] in tasks}
    mine = {"tasks": tasks, "states": {**old, **new}, "pairs": [{"id": "zero update", "reuse": ["e46/ident32_s0"], "new": []}]}
    return mine, {**e136, "states": {**e136["states"], **new}}


class Readers:
    """One model shared by every reader."""

    size = SM.Readers.size  # ||dW|| / ||W0||, reads self.W0_norm

    def __init__(self, c: Config) -> None:
        self.c = c
        mine, full = inventories()
        self.e136 = E136.Job(mine)  # loads the model and the SVD decompositions
        self.e136.k = {k: int(v) for k, v in load(f"{E136_R}/stage0.json")["k_m"].items()}  # fixed by E136 before any state
        self.net: ST.Net = self.e136.net
        self.W0_norm = math.sqrt(sum(float(w.double().pow(2).sum()) for w in self.net.W0.values()))  # as size_matched
        hf = load(f"{E143_R}/predictions.json")
        self.hf = HF.Job(self.net, full, hf["settings"], hf["thresholds"])
        self.S = load(f"{E98_R}/predictions.json")["settings"]
        tasks, _ = E98.E63.load_tasks(self.net, E98.E63.TT.performed_anchor(), E98.files())  # as E98's load_reads
        self.reads = {t: NR.task_reads(tasks[n]) for t, n in E98_TASK.items()}

    def file(self, name: str) -> Path:
        return REPO / self.e136.inv["states"][name]["file"]

    def read(self, name: str, i: int) -> dict[str, Any]:
        """Every reading of one saved state."""
        task, path = self.e136.inv["states"][name]["task"], self.file(name)
        e136 = self.e136.read_state(name)
        raw = US.load(path)  # fp64 of the file as saved
        dW = {k: v.float() for k, v in raw.items()}
        r, et = self.reads[task], E98_TASK[task]
        nr = NR.shares(NR.state_cells(self.net, r, dW, self.S["batch"]), self.S["gain_floor"])
        kh = KH.carrying(self.net, r, dW, self.S, E98.p_learn_of(et, self.S))
        failed = KH.checks(kh, nr["r_half"], self.S, {"identity_tol": self.c.identity_tol, "reproduce_tol": self.c.reader_neuron_tol})
        if failed:
            raise Stop(f"{name}: E98's per-state checks {failed}")
        hf, _ = self.hf.state(name, i)
        return {"e136": {k: e136[k] for k in ("trained", "svd", "rotated", "gain", "gain_fp32")},
                "neuron_r": {k: nr[k] for k in ("G", "r_n", "r_half", "d_n", "i_n", "e_n", "readable", "p_target", "identity", "telescope")},
                "k_half": {k: kh[k] for k in ("k_half", "why_none", "e98_scored", "end_diff", "r_half_num")},
                "holes_fill": hf, "spectrum": US.state(raw), "size": self.size(raw), "file": str(path.relative_to(REPO)), "sha256": KD.sha256(path)}  # fmt: skip


def scored(rec: dict[str, Any]) -> dict[str, Any]:
    """The numbers the figure plots, for the log."""
    st = rec["holes_fill"].get("steering", {})
    return {"share": rec["e136"]["trained"], "svd": rec["e136"]["svd"], "steer_pooled": st.get("pooled", {}).get("share"),
            "steer_perkey": st.get("perkey", {}).get("share"), "r_n": rec["neuron_r"]["r_n"], "r_half": rec["neuron_r"]["r_half"],
            "neuron_k64": rec["holes_fill"]["neuron"]["k64"]["share"], "neuron_kmatched": rec["holes_fill"]["neuron"]["kmatched"]["share"],
            "k_half": rec["k_half"]["k_half"], "erank": rec["spectrum"]["erank"], "norm2": rec["spectrum"]["norm2"], "size": rec["size"]}  # fmt: skip


# ----------------------------------------------------------------------------- stage 0


def saved_size(name: str) -> float:
    """E187's size of a state (size_matched m67.json), joined by E136's state lists as p270_rescore does."""
    m67 = load("experiments/size_matched/results/m67.json")["pairs"]
    for p in load(f"{E136_R}/inventory.json")["pairs"]:
        if p["id"] in m67 and name in p["reuse"] + p["new"]:
            return dict(zip(p["reuse"] + p["new"], m67[p["id"]]["sizes"]["reuse"] + m67[p["id"]]["sizes"]["new"], strict=True))[name]
    raise KeyError(name)


def zero_update(rd: Readers, res: dict[str, Any]) -> None:
    """The zero update on every reader: holes_fill's stage 0 as filed (also sets the k per block), E136's control (c),
    E98's W0 cells on the five tasks, spectrum and size."""
    c, ch, zero = rd.c, res["checks"], {k: torch.zeros_like(w) for k, w in rd.net.W0.items()}
    try:
        res["holes_fill_stage0"] = HF.stage0(rd.hf)
    except HF.Stopped as e:
        raise Stop(f"holes_fill stage 0: {e} (its record: results/large/e143_controls/)") from e
    try:
        E136.control_c(rd.e136, res)  # the zero update's gates are W0's, its numerator 0 (svd, rotated)
    except E136.Stop as e:
        raise Stop(f"E136 control c: {res.get('stopped')}") from e
    for t, r in rd.reads.items():
        check(ch, f"0 E98 zero update max |cell - T0| {t}", NR.max_deviation(NR.state_cells(rd.net, r, zero, rd.S["batch"])), c.identity_tol)
    z = US.state({k: v.double().cpu() for k, v in zero.items()})
    check(ch, "0 spectrum zero: erank undefined (0 = None)", 0.0 if z["erank"] is None else 1.0, 0.0)
    check(ch, "0 spectrum zero norm2", float(z["norm2"] or 0.0), 0.0)
    check(ch, "0 size zero", rd.size(zero), 0.0)


def reproduce(rd: Readers, res: dict[str, Any]) -> None:
    """Two old states read again by every reader, against the saved readings (expected 0: same code, inputs and card)."""
    c, ch = rd.c, res["checks"]
    svd = {p: load("experiments/update_svd/results/results.json")["pairs"][p]["new"] for p in ("e53_lowercase_cipher", "e54_own")}
    for i, (name, e98) in enumerate(REPRO.items()):
        got = rd.read(name, i)
        res["reproduce"][name] = got
        say(f"0 reproduce {name}", scored(got))
        for k in ("trained", "svd", "rotated"):
            check(ch, f"0 E136 {k} {name}", got["e136"][k] - load(f"{E136_R}/stage1.json")["states"][name][k], c.reader_share_tol)
        for k in ("r_n", "r_half"):
            check(ch, f"0 E98 {k} {name}", got["neuron_r"][k] - load(f"{E98_R}/cells.json")["rows"][e98][k], c.reader_neuron_tol)
        check(ch, f"0 E98 k_half {name} (0 = equal)", float(got["k_half"]["k_half"] != load(f"{E98_R}/carrying.json")["rows"][e98]["k_half"]), 0.0)
        hf = load(f"{E143_R}/stage1.json")["states"][name]
        for k in ("k64", "kmatched"):
            check(ch, f"0 E143 neuron {k} {name}", got["holes_fill"]["neuron"][k]["share"] - hf["neuron"][k]["share"], c.reader_neuron_tol)
        for k in hf.get("steering", {}):
            check(ch, f"0 E143 steering {k} {name}", got["holes_fill"]["steering"][k]["share"] - hf["steering"][k]["share"], c.reader_share_tol)
        old = next(v for side in svd.values() for n, v in side.items() if n == name.split("/")[1] and v["file"] == got["file"])
        for k in ("erank", "norm2"):
            check(ch, f"0 spectrum {k} {name} (relative)", rel(got["spectrum"][k], old[k]), c.spectrum_rel_tol)
        check(ch, f"0 size {name} (relative)", rel(got["size"], saved_size(name)), c.size_rel_tol)


def stage0(rd: Readers) -> None:
    res: dict[str, Any] = {"host": socket.gethostname(), "card": torch.cuda.get_device_name(), "checks": {}, "reproduce": {}}
    for d in (E136.RESULTS, HF.RESULTS):
        d.mkdir(parents=True, exist_ok=True)
    try:
        zero_update(rd, res)
        write("stage0.json", res)
        reproduce(rd, res)
    except Stop as e:
        res["stopped"] = str(e)
        write("stage0.json", res)
        raise
    res["pass"] = True
    write("stage0.json", res)


# ----------------------------------------------------------------------------- stage 1


def e53_data(rd: Readers) -> dict[str, LC.Lower]:
    """E53's two training sets, as performed_anchor's stage_finetune builds them (lowercase.task; the cipher's map on the
    cased train rows 0-2399, held-out ciphered val rows 0-63)."""
    pa, device = TT.performed_anchor(), rd.net.device
    low, pi = LC.task(rd.net.model, device, log), pa.e53_pi()
    held = pa.task_text("cipher", pile_rows("val", 0, 256)[: pa.CFG.n_held], pi)[0]
    train = pa.task_text("cipher", pile_rows("train", 0, 4096)[: pa.CFG.n_train], pi)[0]
    return {"e53/lowercase": low, "e53/cipher": LC.Lower(train, held, low.w0_net, mean_logp(low.w0_net, held, device, LC.BATCH))}


def train(rd: Readers, task: str, seed: int, e53: dict[str, LC.Lower]) -> dict[str, tuple[ST.DW, dict[str, Any]]]:
    """One fine-tune by the task's recipe, as saved: {state: (delta-W in the siblings' dtype, record)}. E54's run gives two
    states, the matched cipher (cipher_s<seed>) and the cipher at its own step (cipher_own_s<seed>)."""
    ctx = KD.Ctx(rd.net, {}, {}, {}, LARGE)  # keydep's fine-tunes read only ctx.net
    group = task.split("/")[0]
    if group == "e46":
        w = torch.load(REPO / rd.e136.inv["tasks"][task]["path"], weights_only=False)
        d, meta = KD.elicit_ft(ctx, w, seed)  # E12's recipe and scored-step rule
        return {f"{task}_s{seed}": (d, meta)}
    if group == "e53":
        d = LC.arm_b_states(rd.net.model, rd.net.device, e53[task], "adam", seed, (LC.STEPS,), log)[LC.STEPS]
        small = {k: v.bfloat16() for k, v in d.items()}
        g = LC.gain(rd.net.model, rd.net.device, e53[task], {k: v.float() for k, v in small.items()})
        return {f"{task}_s{seed}": (small, {"step": LC.STEPS, "gain_bf16": g})}
    w = torch.load(REPO / rd.e136.inv["tasks"][task]["path"], weights_only=False)
    target = load("experiments/keydependent_patch/results/job2.json")["finetunes"][f"elicit_s{seed}"]["norm"]
    m, own, mm, om = KD.cipher_ft(ctx, w, seed, target)
    return {f"e54/cipher_s{seed}": (m, {k: v for k, v in mm.items() if k != "norms"}), f"e54/cipher_own_s{seed}": (own, om)}


def saved_step(name: str) -> int:
    group, rest = name.split("/")
    if group == "e46":
        stem, s = rest.rsplit("_s", 1)
        return load("experiments/new_vs_existing/results/new_vs_existing.json")["tasks"][stem]["seeds"][s]["scored_step"]
    if group == "e53":
        return LC.STEPS
    src = "keydependent_elicit/results/stage2.json" if rest.endswith(("_s0", "_s1")) else "keydependent_patch/results/job2.json"
    return load(f"experiments/{src}")["finetunes"][rest]["step"]


def control(rd: Readers, name: str, d: ST.DW, meta: dict[str, Any], out: dict[str, Any]) -> None:
    """A retrained saved state: its step, its weights against the saved file, its trained share within the saved s0/s1 range."""
    saved = torch.load(rd.file(name), weights_only=True)
    diff = {k: (d[k].float() - saved[k].float()) for k in d}
    num = math.sqrt(sum(float(v.double().pow(2).sum()) for v in diff.values()))
    den = math.sqrt(sum(float(v.double().pow(2).sum()) for v in saved.values()))
    share = rd.e136.trained(name, {k: v.float().to(rd.net.device) for k, v in d.items()})["shares"]["r"]
    stem = name.rsplit("_s", 1)[0]
    sib = [load(f"{E136_R}/stage1.json")["states"][f"{stem}_s{s}"]["trained"] for s in (0, 1)]
    want = load(f"{E136_R}/stage1.json")["states"][name]["trained"]
    rec = {**meta, "max_abs_diff": max(float(v.abs().max()) for v in diff.values()), "rel_frobenius_diff": num / den,
           "trained": share, "saved_trained": want, "tol": max(sib) - min(sib), "saved_step": saved_step(name)}  # fmt: skip
    out["controls"][name] = rec
    say(f"1 control {name}", rec)
    check(out["checks"], f"1 control step {name} (0 = saved step)", float(meta["step"] != rec["saved_step"]), 0.0)
    check(out["checks"], f"1 control trained share {name}", share - want, rec["tol"])


def stage1(rd: Readers) -> None:
    path = RESULTS / "stage1.json"
    out: dict[str, Any] = json.loads(path.read_text()) if path.is_file() else {"controls": {}, "states": {}, "checks": {}}
    if "stopped" in out:
        raise Stop(f"stage 1 stopped before ({out['stopped']}): move stage1.json aside to start again")
    e53 = e53_data(rd)
    cs, s = rd.c.control_seed, rd.c.seed
    jobs = [(t, cs) for t in ("e46/ident32", "e46/rand32", "e53/lowercase", "e53/cipher")] + [("e54/cipher", s)]
    jobs += [(t, s) for t in ("e46/ident32", "e46/rand32", "e53/lowercase", "e53/cipher")]
    try:
        for task, seed in jobs:
            want = [n for n in (f"{task}_s{seed}", f"e54/cipher_own_s{seed}" if task == "e54/cipher" else "") if n]
            if all(n in out["controls"] or (n in out["states"] and KD.sha256(rd.file(n)) == out["states"][n]["sha256"]) for n in want):
                continue
            for name, (d, meta) in train(rd, task, seed, e53).items():
                if name in NEW:
                    torch.save(d, rd.file(name))
                    out["states"][name] = {**meta, "file": NEW[name][1], "sha256": KD.sha256(rd.file(name))}
                    if name == "e54/cipher_own_s2":  # E56's job2 trained this state (not kept) and recorded its step and norm
                        j2 = load("experiments/keydependent_patch/results/job2.json")["finetunes"][name.split("/")[1]]
                        out["states"][name] |= {"job2_step": j2["step"], "job2_norm": j2["norm"], "norm_minus_job2": meta["norm"] - j2["norm"]}
                    say(f"1 saved {name}", out["states"][name])
                else:
                    control(rd, name, d, meta, out)
                write("stage1.json", out)
    except Stop as e:
        out["stopped"] = str(e)
        write("stage1.json", out)
        raise
    write("stage1.json", out)


# ----------------------------------------------------------------------------- stage 2


def stage2(rd: Readers, t_job: float) -> None:
    path = RESULTS / "stage2.json"
    out: dict[str, Any] = json.loads(path.read_text()) if path.is_file() else {"states": {}}
    made = json.loads((RESULTS / "stage1.json").read_text())["states"]
    order = ["e53/cipher_s2", "e53/lowercase_s2", "e46/ident32_s2", "e46/rand32_s2", "e54/cipher_own_s2"]  # E53's gates first, once each
    t0, n_new = time.time(), 0
    for i, name in enumerate(order):
        if name in out["states"]:
            continue
        if KD.sha256(rd.file(name)) != made[name]["sha256"]:
            raise Stop(f"{name}: sha256 differs from stage1.json")
        out["states"][name] = rec = rd.read(name, len(REPRO) + i)
        n_new += 1
        say(f"2 {name}", scored(rec))
        left = sum(n not in out["states"] for n in order)
        out["projected_minutes"] = minutes = ((time.time() - t0) / n_new * left + (time.time() - t_job)) / 60
        log(f"[2 time] {len(out['states'])} of {len(order)} states; projected {minutes:.1f} min for the job")
        write("stage2.json", out)
        if minutes > rd.c.projected_limit_min:
            raise Stop(f"projected {minutes:.1f} min over {rd.c.projected_limit_min}")


def main() -> int:
    t_job = time.time()
    require_filed(PRED)
    P = json.loads(PRED.read_text())
    assert P["experiment"] == "E299"
    c = config_from_settings(Config, P["settings"])
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # as every sibling was trained and read
    LARGE.resolve().mkdir(parents=True, exist_ok=True)
    rd = Readers(c)
    log(f"[host] {socket.gethostname()} {torch.cuda.get_device_name()}")
    if manifest_only():  # a CPU dry run for the input list: the loads are done, the reads are skipped
        return 0
    try:
        stage0(rd)
        print("STAGE 0 DONE", flush=True)
        stage1(rd)
        print("STAGE 1 DONE", flush=True)
        stage2(rd, t_job)
        print("STAGE 2 DONE", flush=True)
    except Stop as e:
        print(f"STOP: {e}: tell the lead", flush=True)
        return 1
    log(f"[done] peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
    return 0


if __name__ == "__main__":
    hard_exit(main())
