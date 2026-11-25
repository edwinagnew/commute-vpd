"""E312 (cell P2.84): Jain et al. (2023)'s reverse fine-tuning (reFT) on Fig 4's 48 states, a descriptive per-task
panel for Fig 6. Spec reports/plans/2026-11-25_spec_p2-84-reverse-ft.md, review reports/reviews/2026-11-25_review_p2-84-
spec.md; card card.md; settings and expected values results/predictions.json (the run refuses unless it is committed
unchanged). The reverse fine-tune is E55 C3's port (`revival_kv.reverse_curve`): KL(p_W0 || p_F) at every position of
the task's training windows, Adam at a tenth of the forward rate, the forward batch, from W0 + delta-W; kappa = mean
KL(p_W0 || p_F) at the read position on the held-out score half after every step; t_half the step it halves (the
reading), t_tenth the step it reaches a tenth (saved only, never promoted). Censored = not crossed within the cap.

  --stage 0                 controls, alone, stop on a miss -> results/stage0.json: the zero update; the kappa floor
                            (batched against one row at a time); C3's reproductions (rank1_s0, cipher_s0, the planted
                            wrapper); the planted-pair gate (the wrapper against c3/cipher s0-s2 scaled to its D,
                            `revival_kv.pair_gate`); the gain fold against E290's hooks; the 3-state null probe with
                            its projected shard time; the probe states under Pile cross-entropy (reported only)
  --stage items --shard I --of N   item I, I + N, ... of the 111 (48 states, their 48 rotated nulls, 15 gain-only
                            states) -> results/items/<item>.json, each as it lands; finished items are skipped
  --stage summary           no GPU: per task values -> results/summary.json
  --dry                     load every input stage 0 reads and exit (the CPU input-manifest run)
Every number is printed. A stop prints "STOP: ..." and exits 1; each stage ends with "STAGE ... DONE".
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import gains as G
from commute import revival_kv as KV
from commute import steering as ST
from commute.data import pile_rows
from commute.finetune import ReferenceConfig, next_token_ce
from commute.items import Item, load_dw
from commute.jobs import (
    config_from_settings,
    hard_exit,
    require_filed,
    stop,
    write_stage,
)
from commute.untrained import haar_orthogonal
from commute.vpd import target_weights
from commute.windows import file_sha256, read_rows

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
PRED, ITEMS, STAGE0 = RESULTS / "predictions.json", RESULTS / "items", RESULTS / "stage0.json"


@dataclass(frozen=True)
class Config:
    lr: float  # 3e-6, a tenth of every forward Adam 3e-5 (C3's reverse_lr)
    cap: int  # 1000 steps (C3's cap_steps)
    grad_clip: float
    optimizer: str
    stop_frac: float  # run on to kappa <= 0.1 D (t_tenth, saved), past the half crossing
    batch: dict[str, int]  # per windows format: the forward batch (kv and e62 32, e53 8)
    fig4: dict[str, dict[str, str]]  # state -> {task, file, sha256} (E306's p2-80 resolve)
    windows: dict[str, dict[str, str]]  # task -> {format, path, sha256}; e53 tasks read by commute.windows
    gain: dict[str, Any]  # E290's FFN-gain states: dir, tasks, seeds
    c3: dict[str, Any]  # reproductions, planted wrapper file, partners
    gate: dict[str, float]  # mult, floor, repro_tol (C3's filed t_half noise floor)
    floor_factor: float
    floor_min: float
    null_seed: int
    probe: list[str]
    pile: list[int]  # Pile train rows [start, n] for the cross-entropy probe
    shards: int
    shard_limit_min: float
    load_min: float  # per-task load allowance in the projection
    notes: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert len(self.fig4) == 48 and 0 < self.stop_frac < 0.5 and self.cap > 0, "48 states, a stop past the half"
        assert all(s["task"] in self.windows or s["task"].startswith("e53/") for s in self.fig4.values())
        assert set(self.probe) <= set(self.fig4) and len(self.probe) == 3


def log(msg: str) -> None:
    print(msg, flush=True)


def load_config() -> Config:
    return config_from_settings(Config, json.loads(PRED.read_text())["settings"])


# ---- inputs -----------------------------------------------------------------------------------------------------


def checked(path: str, sha: str | None) -> Path:
    p = REPO / path
    if sha is not None and (got := file_sha256(p)) != sha:
        raise SystemExit(f"STOP: {path}: sha256 {got}, filed {sha}")
    return p


def fmt(c: Config, task: str) -> str:
    return "e53" if task.startswith("e53/") else c.windows[task]["format"]


@cache
def task_rows(task: str) -> tuple[Tensor, Tensor, int]:
    """(train tokens, held-out tokens, read position) on CPU. kv files {train, heldout}; E62's {train, read}; E53's
    windows by `commute.windows.read_rows`, the held-out cut to end at the read position (a causal model's logits there
    are unchanged; it keeps a 250-window read under 4 GB)."""
    c = load_config()
    if task.startswith("e53/"):
        held, train = read_rows(task, "held"), read_rows(task, "train")
        return train.toks, held.toks[:, : held.pos + 1], held.pos
    spec = c.windows[task]
    w = torch.load(checked(spec["path"], spec.get("sha256")), weights_only=False)
    held = w["read"] if spec["format"] == "windows_pt" else w["heldout"]
    return w["train"].tokens, held.tokens, int(held.half)


@cache
def task_ref(net: ST.Net, task: str) -> tuple[KV.Ref, Tensor]:
    train, held, pos = task_rows(task)
    return KV.make_ref(net, held, pos), train


def state_dw(c: Config, state: str) -> ST.DW:
    s = c.fig4[state]
    return load_dw(Item(state, s["file"], s["sha256"]))


def seed_of(name: str) -> int:
    return int(name.rsplit("_s", 1)[1].split("~")[0])


def rotated(c: Config, state: str, dW: ST.DW) -> ST.DW:
    """The null: each matrix's change turned by a Haar orthogonal Q on its output side (singular values, so the
    Frobenius norm and spectrum, kept); one generator per state, null_seed + its index in sorted(fig4), matrices in
    sorted order (as `untrained.rotated_decomposition`)."""
    gen = torch.Generator().manual_seed(c.null_seed + sorted(c.fig4).index(state))
    return {k: (haar_orthogonal(dW[k].shape[0], gen) @ dW[k].double()).float() for k in sorted(dW)}


def gain_dw(net: ST.Net, c: Config, task: str, seed: int) -> ST.DW:
    return G.delta_ia3(gain_of(c, task, seed), net.W0)


def gain_of(c: Config, task: str, seed: int) -> G.IA3:
    p = checked(f"{c.gain['dir']}/gains_ffn_{task}_s{seed}.pt", None)
    saved = torch.load(p, weights_only=False)
    return G.IA3.of(saved["steps"][saved["read_step"]])


def norm(dW: ST.DW) -> float:
    return math.sqrt(sum(float(v.double().pow(2).sum()) for v in dW.values()))


# ---- one reverse fine-tune ------------------------------------------------------------------------------------------


def at_w0(net: ST.Net) -> bool:
    return all(torch.equal(v, net.W0[k]) for k, v in target_weights(net.model).items())


def reverse(net: ST.Net, c: Config, ref: KV.Ref, train: Tensor, dW: ST.DW, seed: int, batch: int, loss_of: Any = None) -> dict[str, Any]:
    """One reverse fine-tune: D, the curve, t_half and t_tenth ("censored" / None void), steps, seconds, W0 back."""
    t0 = time.monotonic()
    cfg = ReferenceConfig(lr=c.lr, batch=batch, steps=c.cap, grad_clip=c.grad_clip, seed=seed, optimizer=c.optimizer)
    rv = KV.reverse_curve(net, ref, dW, w0_net(net), train, cfg, stop_frac=c.stop_frac, loss_of=loss_of)
    cv = rv["curve"]
    read = lambda x: None if cv[0] <= 0 else "censored" if x is None else x
    rec = {"D": cv[0], "norm": norm(dW), "t_half": read(KV.half_life(cv)), "t_tenth": read(KV.crossing(cv, 0.1)),
           "end": rv["end"], "steps": len(cv) - 1, "curve": cv, "seconds": time.monotonic() - t0, "w0_restored": at_w0(net)}  # fmt: skip
    rec["s_per_step"] = rec["seconds"] / max(rec["steps"], 1)
    return rec


@cache
def w0_net(net: ST.Net) -> torch.nn.Module:
    return KV.w0_copy(net)


def say(name: str, r: dict[str, Any]) -> None:
    log(f"[{name}] D {r['D']:.6g} norm {r['norm']:.5g} t_half {r['t_half']} t_tenth {r['t_tenth']} end {r['end']} "
        f"steps {r['steps']} {r['seconds']:.1f} s ({r['s_per_step']:.4f} s/step) W0 back {r['w0_restored']}")  # fmt: skip


# ---- items ----------------------------------------------------------------------------------------------------------


def item_names(c: Config) -> list[str]:
    states = sorted(c.fig4)
    gains = [f"gain/{t}_s{s}" for t in c.gain["tasks"] for s in c.gain["seeds"]]
    return [n for s in states for n in (s, f"{s}~null")] + gains


def item_path(name: str) -> Path:
    return ITEMS / f"{name.replace('/', '__').replace('~', '--')}.json"


def run_item(net: ST.Net, c: Config, name: str) -> dict[str, Any]:
    if name.startswith("gain/"):
        task, seed = name.removeprefix("gain/").rsplit("_s", 1)
        t, dW = f"e58/{task}", gain_dw(net, c, task, int(seed))
    else:
        state = name.split("~")[0]
        t, dW = c.fig4[state]["task"], state_dw(c, state)
        dW = rotated(c, state, dW) if name.endswith("~null") else dW
    ref, train = task_ref(net, t)
    r = reverse(net, c, ref, train, dW, seed_of(name), c.batch[fmt(c, t)])
    return {"item": name, "task": t, "kind": "null" if name.endswith("~null") else "gain" if name.startswith("gain/") else "state", **r}


def write_item(net: ST.Net, c: Config, name: str) -> dict[str, Any]:
    p = item_path(name)
    if p.is_file():
        return json.loads(p.read_text())
    r = run_item(net, c, name)
    say(name, r)
    write_stage(p, r)
    if not r["w0_restored"]:
        stop(p, r, f"{name}: the weights are not W0 after its reverse fine-tune", log)
    return r


def stage_items(net: ST.Net, c: Config, shard: int, of: int) -> None:
    if json.loads(STAGE0.read_text()).get("passed") is not True:
        raise SystemExit("STOP: stage 0 has not passed (results/stage0.json)")
    names = item_names(c)[shard::of]
    log(f"[shard {shard}/{of}] {len(names)} items")
    for n in sorted(names, key=lambda n: (n.startswith("gain/"), n)):
        write_item(net, c, n)
    log(f"STAGE items shard {shard}/{of} DONE")


# ---- stage 0 --------------------------------------------------------------------------------------------------------


def kappa_floor(net: ST.Net, ref: KV.Ref) -> float:
    """Mean KL between W0's answer distributions read batched and one window at a time (the read's own fp32 floor)."""
    one = torch.cat([KV.answer_lp(net, ref.score[i : i + 1], ref.half) for i in range(len(ref.score))])
    return float(KV.kl_rows(ref.lp0["score"], one).mean())


def fold_check(net: ST.Net, c: Config, out: dict[str, Any]) -> bool:
    """F with the folded delta-W against E290's hooks on 64 e58/copy held-out windows: relative max logit difference,
    against floor_factor x the change one fp32 ulp of every folded down_proj weight makes."""
    task, seed = c.gain["fold_check"]
    g, dW = gain_of(c, task, seed), gain_dw(net, c, task, seed)
    toks = task_rows(f"e58/{task}")[1][:64]
    with G.applied_ia3(net.net, G.IA3.of(g.state(), net.device)):
        hook = net.run(toks)[0]
    with net.at(dW):
        fold = net.run(toks)[0]
    ulp = {k: (torch.nextafter(net.W0[k] + v.to(net.device), torch.full_like(net.W0[k], math.inf)) - net.W0[k]).cpu() if "down_proj" in k else v for k, v in dW.items()}
    with net.at(ulp):
        nudged = net.run(toks)[0]
    scale = float(fold.abs().max())
    diff, floor = float((fold - hook).abs().max()) / scale, float((nudged - fold).abs().max()) / scale
    bar = c.floor_factor * max(floor, c.floor_min)
    out["fold"] = {"rel_diff": diff, "ulp_floor": floor, "bar": bar, "ok": diff <= bar}
    log(f"[fold] gain {task}_s{seed}: fold vs hook rel max logit diff {diff:.3g}, one-ulp floor {floor:.3g}, bar {bar:.3g} -> {diff <= bar}")
    return diff <= bar


def planted_pair(net: ST.Net, c: Config, out: dict[str, Any]) -> str:
    """C3's planted wrapper rebuilt from measure.json, then c3/cipher s0-s2 each scaled to its D on the same windows."""
    ref, train = task_ref(net, "c3/rank1")
    pl = json.loads(checked(c.c3["planted_file"], c.c3["planted_sha256"]).read_text())["planted"]
    wrap = reverse(net, c, ref, train, KV.planted_dw(net, pl["units"], torch.tensor(pl["w"]), pl["alpha"]), 0, c.batch["kv"])
    say("planted wrapper", wrap)
    out["planted"] = {"wrapper": wrap, "partners": {}}
    for s in c.c3["partner_seeds"]:
        dW = state_dw(c, f"c3/cipher_s{s}")
        fit = KV.fit_alpha(lambda a, dW=dW: KV.kappa(net, ref, {k: a * v for k, v in dW.items()}), lambda _a: 0.0, wrap["D"], math.inf)
        r = reverse(net, c, ref, train, {k: fit["alpha"] * v for k, v in dW.items()}, s, c.batch["kv"]) | {"scale": fit["alpha"]}
        say(f"planted partner c3/cipher_s{s} x {fit['alpha']:.5f}", r)
        out["planted"]["partners"][f"c3/cipher_s{s}"] = r
    g = KV.pair_gate(wrap["t_half"], [r["t_half"] for r in out["planted"]["partners"].values()], c.cap, c.gate["floor"], c.gate["mult"])
    out["planted"]["gate"] = g
    log(f"[gate] wrapper t_half {wrap['t_half']} vs partners {[r['t_half'] for r in out['planted']['partners'].values()]}: "
        f"gap {g['gap']:.4g} noise {g['noise']:.4g} x {c.gate['mult']} -> {g['reading']}")  # fmt: skip
    return g["reading"]


def reproduce(net: ST.Net, c: Config, out: dict[str, Any]) -> list[str]:
    ref, train = task_ref(net, "c3/rank1")
    miss = []
    for state, want in c.c3["reproduce"].items():
        r = reverse(net, c, ref, train, state_dw(c, state), seed_of(state), c.batch["kv"])
        say(f"reproduce {state}", r)
        ok = isinstance(r["t_half"], float) and abs(r["t_half"] - want["t_half"]) <= c.gate["repro_tol"]
        out["reproduce"][state] = r | {"C3": want, "ok": ok}
        log(f"[reproduce {state}] t_half {r['t_half']} vs C3 {want['t_half']:.4f} (tol {c.gate['repro_tol']}); D {r['D']:.6f} vs C3 {want['D']:.6f} (reported)")
        miss += [] if ok else [state]
    return miss


def projection(c: Config, probes: list[dict[str, Any]]) -> dict[str, Any]:
    """Minutes for one shard from the probes: per-item steps (a censored null counts the cap) x the slowest s/step."""
    spp = max(r["s_per_step"] for r in probes)
    by = lambda kind: [r["steps"] for r in probes if r["kind"] == kind]
    per = {"state": sum(by("state")) / 3, "null": sum(by("null")) / 3}
    steps = 48 * (per["state"] + per["null"]) + 15 * per["state"]
    minutes = steps * spp / 60 / c.shards + c.load_min * 17
    return {"s_per_step_max": spp, "mean_steps": per, "minutes_per_shard": minutes, "ok": minutes <= c.shard_limit_min,
            "shards_needed": math.ceil(steps * spp / 60 / max(c.shard_limit_min - c.load_min * 17, 1))}  # fmt: skip


def stage0(net: ST.Net, c: Config) -> None:
    out: dict[str, Any] = {"reproduce": {}, "pred_sha256": file_sha256(PRED)}
    ref, train = task_ref(net, "c3/rank1")
    zero = {k: torch.zeros_like(w) for k, w in net.W0.items()}
    z = reverse(net, c, ref, train, zero, 0, c.batch["kv"])
    out["zero"] = z
    say("zero update", z)
    if z["D"] != 0.0 or z["t_half"] is not None or not z["w0_restored"]:
        stop(STAGE0, out, "the zero update's D is not exactly 0, or its t_half is not void", log)
    fl = kappa_floor(net, ref)
    out["kappa_floor"] = {"batched_vs_single": fl, "D_void": c.floor_factor * max(fl, c.floor_min)}
    log(f"[kappa floor] batched vs one window {fl:.3g} -> D under {out['kappa_floor']['D_void']:.3g} is void")
    if miss := reproduce(net, c, out):
        stop(STAGE0, out, f"C3's t_half not reproduced on {miss}", log)
    if planted_pair(net, c, out) != "pass":
        stop(STAGE0, out, f"the planted-pair gate reads {out['planted']['gate']['reading']}", log)
    w = out["planted"]["wrapper"]["t_half"]
    if not (isinstance(w, float) and abs(w - c.c3["planted_t_half"]) <= c.gate["repro_tol"]):
        stop(STAGE0, out, f"the planted wrapper's t_half {w} is not C3's {c.c3['planted_t_half']}", log)
    if not fold_check(net, c, out):
        stop(STAGE0, out, "the gain fold does not reproduce E290's hooks", log)
    probes = [write_item(net, c, n) for s in c.probe for n in (s, f"{s}~null")]
    out["projection"] = projection(c, probes)
    log(f"[projection] {out['projection']}")
    write_stage(STAGE0, out)
    if not out["projection"]["ok"]:  # sizing, not a control: the items array is submitted with this many shards
        log(f"[projection] {c.shards} shards project to {out['projection']['minutes_per_shard']:.0f} min each: submit the items with --of {out['projection']['shards_needed']}")
    pile = pile_rows("train", *c.pile)
    out["pile_ce"] = {}
    for s in c.probe:
        t = c.fig4[s]["task"]
        r = reverse(net, c, task_ref(net, t)[0], pile, state_dw(c, s), seed_of(s), c.batch[fmt(c, t)], next_token_ce)
        say(f"pile CE {s}", r)
        out["pile_ce"][s] = r
        write_stage(STAGE0, out)
    write_stage(STAGE0, out | {"passed": True})
    log("STAGE 0 DONE")


# ---- summary (no GPU) -----------------------------------------------------------------------------------------------


def stage_summary(c: Config) -> None:
    done = {n: json.loads(item_path(n).read_text()) for n in item_names(c) if item_path(n).is_file()}
    drop = lambda r: {k: r[k] for k in ("D", "norm", "t_half", "t_tenth", "end", "steps")}
    groups: dict[str, Any] = {}
    for s in sorted(c.fig4):
        g = groups.setdefault(s.rsplit("_s", 1)[0], {"task": c.fig4[s]["task"], "states": {}})
        g["states"][s] = {"state": drop(done[s]) if s in done else None, "null": drop(done[f"{s}~null"]) if f"{s}~null" in done else None}
    gains = {n: drop(r) for n, r in done.items() if n.startswith("gain/")}
    missing = [n for n in item_names(c) if n not in done]
    s0 = json.loads(STAGE0.read_text())
    void = s0["kappa_floor"]["D_void"]
    nulls = [r for n, r in done.items() if n.endswith("~null")]
    out = {"groups": groups, "gain": gains, "missing": missing, "D_void": void, "null_void": sum(r["D"] < void for r in nulls),
           "censored": {k: sum(r["t_half"] == "censored" for n, r in done.items() if r["kind"] == k) for k in ("state", "null", "gain")},
           "gate": s0["planted"]["gate"], "planted": {"wrapper": drop(s0["planted"]["wrapper"]), "partners": {k: drop(v) for k, v in s0["planted"]["partners"].items()}},
           "pile_ce": {k: drop(v) for k, v in s0.get("pile_ce", {}).items()}}  # fmt: skip
    write_stage(RESULTS / "summary.json", out)
    for name, g in groups.items():
        log(f"[{name}] " + " | ".join(f"{s}: t_half {v['state'] and v['state']['t_half']} D {v['state'] and v['state']['D']} null t_half {v['null'] and v['null']['t_half']} D {v['null'] and v['null']['D']}" for s, v in g["states"].items()))
    log(f"[summary] {len(done)} of {len(item_names(c))} items; missing {missing}; censored {out['censored']}; nulls under D_void {out['null_void']}")
    log("STAGE summary DONE")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("0", "items", "summary"), required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    require_filed(PRED)
    c = load_config()
    if a.stage == "summary":
        stage_summary(c)
        hard_exit(0)
    net = ST.Net("cuda")
    if a.dry:  # every input stage 0 and the items read, no reverse fine-tune
        for t in {s["task"] for s in c.fig4.values()} | {f"e58/{t}" for t in c.gain["tasks"]}:
            log(f"[dry] {t}: train {tuple(task_rows(t)[0].shape)} held {tuple(task_rows(t)[1].shape)} read at {task_rows(t)[2]}")
        for s in c.fig4:
            state_dw(c, s)
        for t in c.gain["tasks"]:
            for seed in c.gain["seeds"]:
                gain_of(c, t, seed)
        checked(c.c3["planted_file"], c.c3["planted_sha256"])
        log(f"[dry] pile rows {tuple(pile_rows('train', *c.pile).shape)}; items {len(item_names(c))}")
        hard_exit(0)
    if a.stage == "0":
        stage0(net, c)
    else:
        stage_items(net, c, a.shard, a.of)
    hard_exit(0)


if __name__ == "__main__":
    main()
