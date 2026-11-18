"""E209 (realistic_tasks), cell P2.38 run 1: the premise run of the real-text task set on the 67M. Plan
reports/plans/2026-11-18_plan_p2-38-run1.md, card 2026-11-18_plan_p2-38-run1-card.md; filed rules, expected values and
settings results/predictions.json (the run refuses unless it is committed unchanged). Tasks and data: commute.realistic.
Nothing here reads the share of a new task; that is run 2.

  stage 0  controls, alone, exit 1 on a miss: E53's lowercase and cipher seed-0 states re-read by the scorecard's own
           readers (E136's share, E143's gain, TopK neurons and fair steering) reproduce their saved numbers; a zero
           update reads exactly 0 (neurons, and this run's LM read); T1's loader (commute.lowercase) gives E53's W0 logp
  stage 1  the original model's premises: a ridge probe per block at the marker (L1-L3), the label prior, nats per byte
           per Wikipedia language (fr, sw, tl, yo), reversed against in-order English; the filed rule keeps tasks
  stage 2  rates {3e-5, 1e-4, 3e-4} x 300 steps, seed 0, every kept task, T1 at 3e-5 first and alone (E53's recipe: its
           gain must reproduce); size and gain at steps 10, 30, 100, 300; the filed rule drops a task under its bar
Each stage writes results/stage<n>.json as it goes and prints "STAGE n DONE"; nothing large is saved.

    sbatch --comment=E209 cluster/realistic_tasks_r1.sbatch
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import case_label as CL
from commute import definition_reads as DFR
from commute import label_reads as LR
from commute import lowercase, vpd
from commute import realistic as R
from commute import steering as ST
from commute.data import LAST, pile_rows
from commute.faithfulness import model_state
from commute.finetune import ArmAConfig, ArmBConfig, finetune_arm_a, finetune_arm_b
from commute.identity import log_odds
from commute.jobs import config_from_settings, hard_exit, load_experiment, require_filed
from commute.local_faith import route_cells, route_split
from commute.natural_tasks import single_ids
from commute.open_pair import jsonable

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
PRED = RESULTS / "predictions.json"
E136, E143 = REPO / "experiments/share_untrained/results", REPO / "experiments/holes_fill/results"
HF = load_experiment("holes_fill", "run")  # Job: E143's gain, TopK neurons and fair steering on saved states
E53_STATES = ("e53/lowercase_s0", "e53/cipher_s0")
LM_BATCH = 8


@dataclass(frozen=True)
class Config:
    rates: tuple[float, ...]
    checkpoints: tuple[int, ...]
    seed: int
    n_text: int  # label windows: text tokens before the marker
    pool_per_class: int  # label tasks' training texts (L3 can be learned only on these)
    fit_per_class: int  # the probe's fitting texts
    held_per_class: int
    sst2_min_tokens: int
    pick_seed: int
    coin_seed: int
    ridge_lambda: float
    l_batch: int
    kl_lambda: float
    grad_clip: float
    t_rows: int
    t_held: int
    languages: tuple[str, ...]

    def __post_init__(self) -> None:
        for f in fields(self):  # JSON lists in, tuples kept
            if isinstance(getattr(self, f.name), list):
                object.__setattr__(self, f.name, tuple(getattr(self, f.name)))
        assert self.t_rows == lowercase.STEPS * lowercase.BATCH and self.checkpoints[-1] == lowercase.STEPS
        assert self.rates[0] == 3e-5 and self.languages[0] == "fr"  # T1 at 3e-5 is E53's recipe, the control


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def say(tag: str, x: Any) -> None:
    log(f"[{tag}] {json.dumps(jsonable(x))}")


def write(n: int, out: dict[str, Any]) -> None:
    tmp = RESULTS / f"stage{n}.tmp"
    tmp.write_text(json.dumps(jsonable(out), indent=1))
    tmp.replace(RESULTS / f"stage{n}.json")


class Stopped(Exception):
    """A filed stop: the stage's file records why; the job exits 1."""


def check(n: int, out: dict[str, Any], ok: bool, why: str) -> None:
    if not ok:
        out["stopped"] = why
        write(n, out)
        raise Stopped(why)


# ----------------------------------------------------------------------------- reads


@dataclass
class Tabs:
    nbytes: Tensor  # (V,) bytes per token
    starts: Tensor  # (V,) word-start tokens


@torch.no_grad()
def lm_read(net: ST.Net, rows: Tensor, tabs: Tabs, dW: ST.DW | None = None) -> dict[str, float]:
    """Next-token loss per token and per byte over every position, and the mean log-odds of the true next token where it
    starts a word (the positions run 2's share would read), fp32."""
    s = {"nll": 0.0, "tok": 0, "bytes": 0.0, "lo": 0.0, "ws": 0}
    with net.at(dW):
        for b in rows.split(LM_BATCH):
            lg = net.run(b)[0][:, :-1]
            tgt = b[:, 1:].to(lg.device)
            s["nll"] -= float(torch.log_softmax(lg, -1).gather(-1, tgt[..., None]).double().sum())
            s["tok"], s["bytes"] = s["tok"] + tgt.numel(), s["bytes"] + float(tabs.nbytes[tgt].sum())
            ws = tabs.starts[tgt]
            s["lo"], s["ws"] = s["lo"] + float(log_odds(lg[ws], tgt[ws]).double().sum()), s["ws"] + int(ws.sum())
    return {"nats_token": s["nll"] / s["tok"], "nats_byte": s["nll"] / s["bytes"], "lo_word_start": s["lo"] / s["ws"]}


@dataclass
class LTask:
    ab: tuple[int, int]
    parts: dict[str, tuple[Tensor, Tensor]]  # pool, fit, held: (windows, y)
    read: str  # the part the gain is read on: held for L1 and L2, pool for L3 (a coin has no held-out reading)

    def lo(self, net: ST.Net, part: str, dW: ST.DW | None = None) -> Tensor:
        toks, y = self.parts[part]
        return DFR.full_log_odds(net, toks, LR.targets(y, self.ab), toks.shape[1] - 1, dW)


def size(dW: ST.DW, W0: ST.DW) -> float:
    return math.sqrt(sum(float(v.double().pow(2).sum()) for v in dW.values()) / sum(float(w.double().pow(2).sum()) for w in W0.values()))


# ----------------------------------------------------------------------------- stage 0


def stage0(net: ST.Net, th: dict[str, Any], low: lowercase.Lower, tabs: Tabs) -> dict[str, Any]:
    out: dict[str, Any] = {"reproduce": {}}
    full = json.loads((E136 / "inventory.json").read_text())
    inv = {"tasks": {t: full["tasks"][t] for t in ("e53/lowercase", "e53/cipher")}, "states": {s: full["states"][s] for s in E53_STATES}}
    hfp = json.loads((E143 / "predictions.json").read_text())
    hf = HF.Job(net, inv, hfp["settings"], hfp["thresholds"])
    k_m = json.loads((E136 / "stage0.json").read_text())["k_m"]  # as holes_fill's stage 0 sets them
    hf.ks = {"k64": {l: int(hfp["settings"]["k_fixed"]) for l in HF.FS.LAYERS}, "kmatched": {l: int(k_m[key]) for l, key in enumerate(HF.DOWN)}}
    t, zero = hf.task("e53/lowercase"), {k: torch.zeros_like(w) for k, w in net.W0.items()}
    out["zero_neuron"] = abs(hf.neuron(t, zero, hf.ks["k64"], 1.0)["numerator"])
    w0, z = lm_read(net, low.held, tabs), lm_read(net, low.held, tabs, zero)
    out["zero_lm"] = max(abs(w0[k] - z[k]) for k in w0)
    out["t1_logp_w0"] = {"got": low.logp_w0, "saved": json.loads((REPO / "experiments/performed_anchor/results/finetune.json").read_text())["logp_w0"]["lowercase"]}
    out["t1_lm_read_w0"] = w0
    say("0 zero update and T1 loader", out)
    check(0, out, out["zero_neuron"] == 0 and out["zero_lm"] == 0, "a zero update does not read exactly 0")
    check(0, out, abs(out["t1_logp_w0"]["got"] - out["t1_logp_w0"]["saved"]) <= th["logp_w0_tol"], "T1's W0 logp differs from E53's")
    e136, e143, comps0 = json.loads((E136 / "stage1.json").read_text())["states"], json.loads((E143 / "stage1.json").read_text())["states"], vpd.component_weights(net.model)
    for i, s in enumerate(E53_STATES):
        rec, _ = hf.state(s, i)
        r, dW = hf.task(inv["states"][s]["task"]).read, hf.dW(s)
        base = route_cells(net.model, r.toks, r.half, r.token, r.batch, r.ci_len)
        with model_state(net.model, target={k: w + dW[k].to(w.device) for k, w in net.W0.items()}):
            share = route_split(net.model, r.toks, r.half, r.token, base, comps0, r.batch, r.ci_len)["shares"]["r"]
        del base
        got = {"share": share, "gain": rec["gain"], "neuron_k64": rec["neuron"]["k64"]["share"],
               "neuron_kmatched": rec["neuron"]["kmatched"]["share"], "steer_pooled": rec["steering"]["pooled"]["share"]}  # fmt: skip
        saved = {"share": e136[s]["trained"], "gain": e143[s]["gain"], "neuron_k64": e143[s]["neuron"]["k64"]["share"],
                 "neuron_kmatched": e143[s]["neuron"]["kmatched"]["share"], "steer_pooled": e143[s]["steering"]["pooled"]["share"]}  # fmt: skip
        out["reproduce"][s] = {k: {"got": got[k], "saved": saved[k], "tol": th["reproduce_tol"][k]} for k in got}
        say(f"0 reproduce {s}", out["reproduce"][s])
        write(0, out)
    bad = [f"{s} {k}" for s, d in out["reproduce"].items() for k, v in d.items() if not abs(v["got"] - v["saved"]) <= v["tol"]]
    check(0, out, not bad, f"does not reproduce: {bad}")
    return out


# ----------------------------------------------------------------------------- stage 1


def label_tasks(C: Config, tok: Any) -> dict[str, LTask]:
    enc, pad = tok.encode, tok.eos_token_id

    def win(src: tuple[list[str], Tensor], idx: Tensor) -> tuple[Tensor, Tensor]:
        return R.label_windows([src[0][i] for i in idx.tolist()], enc, C.n_text, pad), src[1][idx]

    def parts(train: tuple[list[str], Tensor], held: tuple[list[str], Tensor]) -> dict[str, tuple[Tensor, Tensor]]:
        pool, fit = R.balanced(train[1], (C.pool_per_class, C.fit_per_class), C.pick_seed)
        (hi,) = R.balanced(held[1], (C.held_per_class,), C.pick_seed)
        return {"pool": win(train, pool), "fit": win(train, fit), "held": win(held, hi)}

    def ab(task: str) -> tuple[int, int]:
        a, b = single_ids(enc, R.LABEL_WORDS[task])
        return a, b

    ag = parts(R.ag_news("train"), R.ag_news("test"))
    coin = {p: (w, R.coin(len(w), C.coin_seed + j)) for j, (p, (w, _)) in enumerate(ag.items())}
    sst = parts(R.sst2("train", enc, C.sst2_min_tokens), R.sst2("validation", enc, C.sst2_min_tokens))
    return {"L1": LTask(ab("L1"), ag, "held"), "L2": LTask(ab("L2"), sst, "held"), "L3": LTask(ab("L3"), coin, "pool")}


def label_premise(net: ST.Net, C: Config, t: LTask) -> dict[str, Any]:
    (fx, fy), (hx, hy) = t.parts["fit"], t.parts["held"]
    pos = hx.shape[1] - 1
    _, _, acc = CL.value_probes(DFR.resid_at(net, fx, pos), fy, DFR.resid_at(net, hx, pos), hy, 2, C.ridge_lambda)
    d = DFR.label_diff(net, hx, t.ab, pos)
    zero = {k: torch.zeros_like(w) for k, w in net.W0.items()}
    return {"probe_acc": acc, "probe_best": max(acc), "w0_accuracy": DFR.accuracy(d, hy), "w0_bias": float(d.mean()),
            "zero_update": float((t.lo(net, t.read, zero) - t.lo(net, t.read)).abs().max())}  # fmt: skip


def lm_rows(C: Config, tok: Any) -> dict[str, tuple[Tensor, Tensor]]:
    """(held, train) rows per Wikipedia language and for T4 (and T4's in-order reference as "en_ref", no train)."""
    out = {}
    for lang in C.languages:
        train, held = R.pack((tok.encode(x) for x in R.wiki_texts(lang)), tok.eos_token_id, (C.t_rows, C.t_held), vpd.SEQ_LEN)
        out[lang] = (held, train)
        log(f"[1 data] {lang}: {len(held)} held, {len(train)} train rows")
    val = pile_rows("val", 0, 256, log=log)
    (rev,) = R.transformed_rows(val, tok, (C.t_held,), lambda s: R.reorder(s, True))
    (ref,) = R.transformed_rows(val, tok, (C.t_held,), lambda s: R.reorder(s, False))
    (train,) = R.transformed_rows(pile_rows("train", 0, 4096, log=log), tok, (C.t_rows,), lambda s: R.reorder(s, True))
    out["T4"], out["en_ref"] = (rev, train), (ref, ref[:0])
    return out


def stage1(net: ST.Net, C: Config, th: dict[str, Any], tok: Any, tabs: Tabs) -> tuple[dict[str, Any], dict[str, LTask], dict[str, Any]]:
    out: dict[str, Any] = {"label": {}, "lm": {}}
    ltasks = label_tasks(C, tok)
    for name, t in ltasks.items():
        out["label"][name] = label_premise(net, C, t)
        say(f"1 {name}", out["label"][name])
        write(1, out)
    check(1, out, all(v["zero_update"] == 0 for v in out["label"].values()), "a zero update does not read exactly 0 on a label task")
    rows = lm_rows(C, tok)
    for name, (held, _) in rows.items():
        out["lm"][name] = lm_read(net, held, tabs)
        say(f"1 {name}", out["lm"][name])
        write(1, out)
    p = {"probe": {k: v["probe_best"] for k, v in out["label"].items()}, "bytes": {g: out["lm"][g]["nats_byte"] for g in C.languages},
         "t4": out["lm"]["T4"]["nats_token"], "t4_ref": out["lm"]["en_ref"]["nats_token"]}  # fmt: skip
    out["calls"] = R.premise_calls(p, th)
    say("1 calls", out["calls"])
    write(1, out)
    lm = {"T2": rows["fr"], "T3": rows[out["calls"]["T3_language"]], "T4": rows["T4"]}
    return out, ltasks, lm


# ----------------------------------------------------------------------------- stage 2


def train(net: ST.Net, C: Config, task: LTask | tuple[Tensor, Tensor], lr: float) -> dict[int, ST.DW]:
    if isinstance(task, LTask):
        (px, py), (hx, hy), pos = task.parts["pool"], task.parts["held"], task.parts["pool"][0].shape[1] - 1
        cfg = ArmAConfig(o_token=LAST, kl_lambda=C.kl_lambda, lr=lr, batch=C.l_batch, steps=C.checkpoints[-1], checkpoints=C.checkpoints,
                         grad_clip=C.grad_clip, seed=C.seed, ci_len=pos + 1)  # fmt: skip
        return finetune_arm_a(net.model, LR.labelled(px, py, task.ab, pos), LR.labelled(hx, hy, task.ab, pos), cfg, net.device, log).deltas
    held, rows = task
    cfg_b = ArmBConfig(lr=lr, batch=lowercase.BATCH, steps=C.checkpoints[-1], checkpoints=C.checkpoints, grad_clip=C.grad_clip, seed=C.seed)
    return finetune_arm_b(net.model, rows, held, cfg_b, net.device, log).deltas


def reads(net: ST.Net, task: LTask | tuple[Tensor, Tensor], tabs: Tabs, dW: ST.DW, w0: dict[str, Any]) -> dict[str, float]:
    """The gain the rule reads ("gain") and the reported ones, against W0's (w0)."""
    if isinstance(task, LTask):
        g = {p: float((task.lo(net, p, dW) - w0[p]).mean()) for p in ("pool", "held")}
        return {"gain": g[task.read], "gain_pool": g["pool"], "gain_held": g["held"]}
    r = lm_read(net, task[0], tabs, dW)
    return {"gain": w0["nats_token"] - r["nats_token"], "lo_word_start_gain": r["lo_word_start"] - w0["lo_word_start"]}


def stage2(net: ST.Net, C: Config, th: dict[str, Any], kept: list[str], tasks: dict[str, Any], tabs: Tabs) -> dict[str, Any]:
    out: dict[str, Any] = {"grid": {}, "kept_in": kept}
    todo = [("T1", C.rates[0])] + [(t, lr) for t in kept if t.startswith("L") for lr in C.rates]
    todo += [(t, lr) for t in kept if t.startswith("T") for lr in C.rates if (t, lr) != ("T1", C.rates[0])]
    t0 = time.time()
    w0 = {t: ({p: tasks[t].lo(net, p) for p in ("pool", "held")} if t.startswith("L") else lm_read(net, tasks[t][0], tabs)) for t in kept}
    for i, (t, lr) in enumerate(todo):
        deltas = train(net, C, tasks[t], lr)
        row = {str(s): {"size": size(d, net.W0), **reads(net, tasks[t], tabs, d, w0[t])} for s, d in deltas.items()}
        out["grid"].setdefault(t, {})[str(lr)] = row
        say(f"2 {t} lr {lr}", row)
        write(2, out)
        if i == 0:
            got = row[str(C.checkpoints[-1])]["gain"]
            out["t1_recipe"] = {"got": got, "saved": th["t1_gain_saved"], "tol": th["t1_gain_tol"]}
            check(2, out, abs(got - th["t1_gain_saved"]) <= th["t1_gain_tol"], f"T1 at E53's recipe gains {got:.5f}, E53 {th['t1_gain_saved']}")
        minutes = (time.time() - t0) / (i + 1) * len(todo) / 60
        log(f"[2 time] {i + 1} of {len(todo)} runs; stage 2 projected {minutes:.1f} min")
        check(2, out, i < 3 or minutes <= th["stage2_limit_min"], f"stage 2 projected {minutes:.1f} min")
    bars = {t: th["l_gain_bar"] if t.startswith("L") else th["t_gain_bar"] for t in out["grid"]}
    out["calls"] = R.grid_calls(out["grid"], bars, th["size_cap"])
    out["kept"] = [t for t in kept if out["calls"][t]["kept"]]
    say("2 calls", out["calls"])
    write(2, out)
    return out


def main() -> int:
    require_filed(PRED)
    P = json.loads(PRED.read_text())
    assert P["experiment"].startswith("E209")
    C, th = config_from_settings(Config, P["settings"]), P["thresholds"]
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    net, tok = ST.Net("cuda"), vpd.tokenizer()
    V = int(net.run(torch.zeros(1, 2, dtype=torch.long))[0].shape[-1])
    pieces = tok.convert_ids_to_tokens(list(range(len(tok))))
    tabs = Tabs(R.byte_lengths(pieces, tok.all_special_ids, V).to(net.device), R.word_starts(pieces, V).to(net.device))
    low = lowercase.task(net.model, net.device, log)
    try:
        write(0, stage0(net, th, low, tabs))
        print("STAGE 0 DONE", flush=True)
        s1, ltasks, lm = stage1(net, C, th, tok, tabs)
        print("STAGE 1 DONE", flush=True)
        stage2(net, C, th, s1["calls"]["kept"], {**ltasks, **lm, "T1": (low.held, low.rows)}, tabs)
        print("STAGE 2 DONE", flush=True)
    except Stopped as e:
        print(f"STOP: {e}: tell the lead", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    hard_exit(main())
