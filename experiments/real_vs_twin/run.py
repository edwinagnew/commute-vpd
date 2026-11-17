"""E140 (real_vs_twin), cell P2.4: do steering or the neuron version tell a real fine-tune from its shuffled-label twin, where
the share cannot? Plan reports/plans/2026-11-10_plan_e140-real-vs-twin.md; filed rule results/predictions.json (the run
refuses unless it is committed unchanged), applied by rules.py.

Two pairs, each a real fine-tune and its shuffled-label twin: case label (E112, experiments/case_label) and repeat label
(E125, experiments/fresh_reuse_pair). Seeds 0 and 1 are their saved updates; seed 2 is trained here by the same function
and settings (`label_reads.finetune`, settings read from each source's predictions.json). Every reading is on the first 256
of the pair's fit windows, the rows the share was read on, in three units (`masked_reads`): log-odds of the label over the
whole vocabulary, log P, and the contrast between the two labels (the scored unit).

One job, stages in order, each written to results/stage<n>.json and followed by `STAGE n DONE`:
  0  controls, alone: k matched to the decomposition's on-count; a zero update reads exactly 0 through the share, both
     neuron readings and steering, and strength 0 reads exactly 0; the share of real s0 reproduces the saved one on each
     pair within r_tol. A miss stops the job (exit 1).
  1  the four seed-2 fine-tunes, each checked as it lands against its arm's saved gains (a miss stops the job), saved in
     bf16 to results/large/; a re-run skips the ones saved. Beside: real s0 of the case pair retrained, against its file.
  2  the share on all twelve updates; each twin rescaled to its real twin's norm and the gain it keeps.
  3  steering (per word and pooled) and the neuron version (k 64 and k matched) on all twelve; cosines; the score.

    sbatch cluster/real_vs_twin.sbatch
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from rules import SCORED, call, outcome
from torch import Tensor

from commute import anchors as AN
from commute import definition_reads as DFR
from commute import elicit as EL
from commute import fair_steering as FS
from commute import keydep as KD
from commute import label_reads as LR
from commute import masked_reads as MR
from commute import steering as ST
from commute.data import LAST
from commute.finetune import ArmAConfig
from commute.jobs import hard_exit, require_filed
from commute.local_faith import MIN_GAIN
from commute.open_pair import jsonable
from commute.untrained import k_from_counts

HERE = Path(__file__).resolve().parent
EXP, RESULTS = HERE.parent, HERE / "results"
LARGE, PRED = RESULTS / "large", RESULTS / "predictions.json"
HALF = AN.HALF
L = 2 * HALF + 1  # the word at HALF - 1, the trigger at HALF, the label read at HALF
PAIRS, ARMS, PARTS, SEEDS, KINDS = ("case", "repeat"), ("real", "twin"), ("train", "fit", "held"), (0, 1, 2), ("perkey", "pooled")  # the key is the word before the marker
SRC = {"case": EXP / "case_label" / "results", "repeat": EXP / "fresh_reuse_pair" / "results"}
SAVED_GAIN = {"case": ("stage2.json", "states"), "repeat": ("stage1.json", "reads")}  # s0/s1 label-gap gain, all fit windows
SAVED_R = {"case": (EXP / "case_label_both" / "results" / "case_label_both.json", "routes"), "repeat": (SRC["repeat"] / "stage2.json", "states")}
DOWN = tuple(f"h.{l}.mlp.down_proj" for l in FS.LAYERS)
log = KD.log


class Stopped(Exception):
    """A filed stop: the stage's file records why; the job exits 1."""


@dataclass(frozen=True)
class Cfg:
    n_read: int
    n_select: int
    strengths: tuple[float, ...]
    n_random: int
    random_seed: int
    k_fixed: int
    new_seed: int

    def __post_init__(self) -> None:
        assert self.n_read == 256 and self.strengths[0] == 0 and list(self.strengths) == sorted(self.strengths), "predictions.json disagrees with the code"
        assert self.new_seed == SEEDS[-1] and self.n_select % FS.BATCH == 0


@dataclass
class Pair:
    name: str
    ab: tuple[int, int]
    toks: dict[str, Tensor]  # part -> (N, 41)
    y: dict[str, dict[str, Tensor]]  # arm -> part -> (N,), 1 = label A
    settings: dict[str, Any]  # the source experiment's filed settings (the recipe)
    saved: dict[str, Any]  # the source's stage1.json states (s0, s1)
    pool: Tensor  # the training words (token at HALF - 1)

    def part(self, arm: str, p: str, n: int | None = None) -> tuple[Tensor, Tensor]:
        return self.toks[p][:n], self.y[arm][p][:n]

    def idx(self, toks: Tensor) -> Tensor:
        return FS.key_index(toks[:, HALF - 1], self.pool)

    def vectors(self, net: ST.Net, dW: ST.DW) -> dict[str, dict[int, Tensor]]:
        """Steering vectors from every training window (the user's option a): the pool is the training words, so every
        key has its windows. The only place vectors are fitted."""
        return FS.vectors(net, self.toks["train"], self.idx(self.toks["train"]), len(self.pool), dW, HALF)


def load_pair(name: str) -> Pair:
    """The pair's task.pt in one layout: E112 keeps a label per token, E125 a label per window."""
    t = torch.load(SRC[name] / "large" / "task.pt", weights_only=False)
    toks = {p: (t[p] if name == "case" else t["toks"][p]) for p in PARTS}
    y = {a: {p: (torch.tensor([t["y"][a][int(w)] for w in toks[p][:, HALF - 1]]) if name == "case" else t["y"][a][p]) for p in PARTS} for a in ARMS}
    settings = json.loads((SRC[name] / "predictions.json").read_text())["settings"]
    saved = json.loads((SRC[name] / "stage1.json").read_text())["states"]
    return Pair(name, (int(t["ab"][0]), int(t["ab"][1])), toks, y, settings, saved, torch.unique(toks["train"][:, HALF - 1]))


def write(n: int, out: dict[str, Any]) -> None:
    tmp = RESULTS / f"stage{n}.tmp"  # a kill mid-write must not leave unreadable JSON
    tmp.write_text(json.dumps(jsonable(out), indent=1))
    tmp.replace(RESULTS / f"stage{n}.json")


def say(tag: str, x: Any) -> None:
    log(f"[{tag}] {json.dumps(jsonable(x))}")


def halt(n: int, out: dict[str, Any], why: str) -> Stopped:
    out["stopped"] = why
    write(n, out)
    return Stopped(why)


def recipe(p: Pair, seed: int) -> ArmAConfig:
    """The source's arm-A settings unchanged (E112 files no lr; it used anchors.FT_LR, as E125 files)."""
    s = p.settings
    assert s.get("lr", AN.FT_LR) == AN.FT_LR
    return ArmAConfig(o_token=LAST, kl_lambda=s["kl_lambda"], lr=AN.FT_LR, batch=s["batch"], steps=s["checkpoints"][-1],
                      checkpoints=tuple(s["checkpoints"]), grad_clip=s["grad_clip"], seed=seed, ci_len=L)  # fmt: skip


def load_dW(path: Path, sha: str) -> ST.DW:
    assert KD.sha256(path) == sha, f"{path}: sha256 differs from its record"
    return {k: v.float() for k, v in torch.load(path, weights_only=True, map_location="cpu").items()}


def state(p: Pair, arm: str, seed: int, new: dict[str, Any]) -> ST.DW:
    if seed in p.settings["seeds"]:
        return load_dW(SRC[p.name] / "large" / f"dW_{arm}_s{seed}.pt", p.saved[f"{arm}_s{seed}"]["sha256"])
    return load_dW(LARGE / f"dW_{p.name}_{arm}_s{seed}.pt", new[f"{p.name}_{arm}_s{seed}"]["sha256"])


def net_units(net: ST.Net, p: Pair, arm: str, toks: Tensor, y: Tensor, dW: ST.DW | None = None) -> dict[str, Tensor]:
    return MR.units(DFR.logits_at(net, toks, HALF, dW), y, p.ab)


def mean_diff(a: dict[str, Tensor], b: dict[str, Tensor]) -> dict[str, float]:
    return {u: float((a[u] - b[u]).mean()) for u in MR.UNITS}


def worst(cells: MR.Cells, pairs: tuple[tuple[str, str], ...]) -> float:
    return max(float((cells[a][u] - cells[b][u]).abs().max()) for a, b in pairs for u in MR.UNITS)


# ----------------------------------------------------------------------------- stage 0


def stage0(net: ST.Net, c: Cfg, th: dict[str, float], pairs: dict[str, Pair]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    counts, n = {k: 0.0 for k in DOWN}, 0
    for p in pairs.values():
        cnt, m = MR.on_count(net.model, p.part("real", "fit", c.n_read)[0], DOWN)
        counts, n = {k: counts[k] + cnt[k] for k in DOWN}, n + m
    out["k_matched"] = {l: k_from_counts(counts, n)[key] for l, key in enumerate(DOWN)}
    say("0 k matched (down_proj on-count per block)", out["k_matched"])
    zero, p = {k: torch.zeros_like(w) for k, w in net.W0.items()}, pairs["case"]
    toks, y = p.part("real", "fit", c.n_read)
    out["zero_share"] = worst(MR.share_cells(net.model, net.W0, zero, toks, y, p.ab, HALF), (("L01", "L00"), ("T1", "T0")))
    for kn, k in {"k64": dict.fromkeys(FS.LAYERS, c.k_fixed), "kmatched": out["k_matched"]}.items():
        out[f"zero_neuron_{kn}"] = worst(MR.topk_cells(net, zero, toks, y, p.ab, HALF, k), (("N01", "N00"), ("T1", "T0")))
    sel = p.part("real", "train", c.n_select)[0]
    lg0 = torch.cat([net.run(sel[s : s + FS.BATCH])[0][:, HALF] for s in range(0, len(sel), FS.BATCH)])
    for name, dW, strengths in (("zero_steering", zero, c.strengths), ("strength0_real_s0", state(p, "real", 0, {}), (0.0,))):
        vs = p.vectors(net, dW)
        out[name] = max(float((FS.logits_steered(net, sel, p.idx(sel), HALF, vs[kd][l], l, s) - lg0).abs().max())
                        for kd in KINDS for l in FS.LAYERS for s in strengths)  # fmt: skip
    say("0 zero update and strength 0, largest |change|", {k: v for k, v in out.items() if k != "k_matched"})
    misses = [k for k, v in out.items() if k != "k_matched" and not v <= th["zero_tol"]]
    if misses:
        raise halt(0, out, f"not exactly 0: {misses}")
    out["reproduce"] = {}
    for name, p in pairs.items():
        path, key = SAVED_R[name]
        toks, y = p.part("real", "fit", c.n_read)
        cells = MR.share_cells(net.model, net.W0, state(p, "real", 0, {}), toks, y, p.ab, HALF)
        r = MR.share(cells["L01"]["lo"] - cells["L00"]["lo"], cells["T1"]["lo"] - cells["T0"]["lo"], -math.inf)
        out["reproduce"][name] = {"r": r, "r_saved": json.loads(path.read_text())[key]["real_s0"]["least_squares"]["shares"]["r"]}
        say(f"0 share of {name} real_s0 against the saved one", out["reproduce"][name])
    bad = [n for n, v in out["reproduce"].items() if not abs(v["r"] - v["r_saved"]) <= th["r_tol"]]
    if bad:
        raise halt(0, out, f"the share does not reproduce on {bad}")
    return out


# ----------------------------------------------------------------------------- stage 1


def fit_gain(net: ST.Net, p: Pair, arm: str, dW: ST.DW) -> float:
    """The label-gap gain on all fit windows, E112's and E125's `gain`."""
    toks, y = p.part(arm, "fit")
    return float(DFR.margin(DFR.label_diff(net, toks, p.ab, HALF, dW) - DFR.label_diff(net, toks, p.ab, HALF), y).mean())


def train(net: ST.Net, p: Pair, arm: str, seed: int) -> tuple[ST.DW, dict[str, Any]]:
    return LR.finetune(net, p.part(arm, "train"), p.part(arm, "fit"), p.part(arm, "held"), p.ab, HALF, recipe(p, seed), p.settings["learn_bar"], log)


def stage1(net: ST.Net, c: Cfg, th: dict[str, float], pairs: dict[str, Pair]) -> dict[str, Any]:
    f = RESULTS / "stage1.json"
    out: dict[str, Any] = json.loads(f.read_text()) if f.is_file() else {"states": {}}
    out.pop("stopped", None)  # a state that missed its check is trained again below, and stops again if it misses
    if "recipe_check" not in out:  # beside, no stop: the same function and seed 0 against the saved case real s0
        p = pairs["case"]
        dW, _ = train(net, p, "real", 0)
        ref = state(p, "real", 0, {})
        out["recipe_check"] = {"rel_diff": sum(float((dW[k].bfloat16().float() - ref[k]).norm()) for k in ref) / EL.summed_norm(ref)}
        say("1 retrained case real_s0 against its file, summed norm of the difference over the file's", out["recipe_check"])
    for p in pairs.values():
        for arm in ARMS:
            name, path = f"{p.name}_{arm}_s{c.new_seed}", LARGE / f"dW_{p.name}_{arm}_s{c.new_seed}.pt"
            if out["states"].get(name, {}).get("early_check") and path.is_file():
                continue
            dW, rec = train(net, p, arm, c.new_seed)
            torch.save({k: v.bfloat16() for k, v in dW.items()}, path)
            rec["sha256"] = KD.sha256(path)
            rec["fit_gain"] = fit_gain(net, p, arm, load_dW(path, rec["sha256"]))
            fname, key = SAVED_GAIN[p.name]
            old = [json.loads((SRC[p.name] / fname).read_text())[key][f"{arm}_s{s}"]["gain"] for s in p.settings["seeds"]]
            rec["bar"] = [th["s2_gain_low"] * min(old), th["s2_gain_high"] * max(old)]
            rec["early_check"] = bool(rec["learned"]) and rec["bar"][0] <= rec["fit_gain"] <= rec["bar"][1]
            out["states"][name] = rec
            say(f"1 state {name}", rec)
            write(1, out)
            if not rec["early_check"]:
                raise halt(1, out, f"{name} misses its arm's saved range (learned {rec['learned']}, gain {rec['fit_gain']:.3f}, bar {rec['bar']})")
    return out


# ----------------------------------------------------------------------------- stage 2


def stage2(net: ST.Net, c: Cfg, th: dict[str, float], pairs: dict[str, Pair], new: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"states": {}, "rescaled": {}}
    for p in pairs.values():
        saved = json.loads(SAVED_R[p.name][0].read_text())[SAVED_R[p.name][1]]
        for arm in ARMS:
            toks, y = p.part(arm, "fit", c.n_read)
            for seed in SEEDS:
                cells = MR.share_cells(net.model, net.W0, state(p, arm, seed, new), toks, y, p.ab, HALF)
                num, gain = {u: cells["L01"][u] - cells["L00"][u] for u in MR.UNITS}, {u: cells["T1"][u] - cells["T0"][u] for u in MR.UNITS}
                rec: dict[str, Any] = {"share": {u: MR.share(num[u], gain[u], th["gain_floor"]) for u in MR.UNITS},
                                       "numerator": {u: float(num[u].mean()) for u in MR.UNITS}, "gain": {u: float(gain[u].mean()) for u in MR.UNITS}}  # fmt: skip
                if f"{arm}_s{seed}" in saved:
                    rec["r_saved"] = saved[f"{arm}_s{seed}"]["least_squares"]["shares"]["r"]
                out["states"][f"{p.name}_{arm}_s{seed}"] = rec
                say(f"2 share {p.name}_{arm}_s{seed}", rec)
        write(2, out)
        toks, y = p.part("twin", "fit", c.n_read)
        w0 = net_units(net, p, "twin", toks, y)
        for seed in SEEDS:
            real, twin = state(p, "real", seed, new), state(p, "twin", seed, new)
            scale = EL.summed_norm(real) / EL.summed_norm(twin)
            g, gs = mean_diff(net_units(net, p, "twin", toks, y, twin), w0), mean_diff(net_units(net, p, "twin", toks, y, {k: v * scale for k, v in twin.items()}), w0)
            out["rescaled"][f"{p.name}_s{seed}"] = {"scale": scale, "gain": g, "gain_rescaled": gs, "kept": shares(gs, g, th["gain_floor"])}
            say(f"2 twin rescaled to real's norm {p.name}_s{seed}", out["rescaled"][f"{p.name}_s{seed}"])
        write(2, out)
    return out


# ----------------------------------------------------------------------------- stage 3


@dataclass
class Read:
    """One state's reads: its pair, arm, own gain per unit (fp32 target model) and W0's units on the read and select rows."""

    p: Pair
    arm: str
    gain: dict[str, float]
    w0: dict[str, dict[str, Tensor]]  # arm -> units on the 256 read rows
    w0_sel: dict[str, Tensor]  # this arm's units on the selection rows


def rises(net: ST.Net, r: Read, toks: Tensor, y: Tensor, base: dict[str, Tensor], v: Tensor, l: int, s: float) -> dict[str, float]:
    return mean_diff(MR.units(FS.logits_steered(net, toks, r.p.idx(toks), HALF, v, l, s), y, r.p.ab), base)


def shares(rise: dict[str, float], gain: dict[str, float], floor: float) -> dict[str, float]:
    return {u: rise[u] / gain[u] if gain[u] >= floor and gain[u] != 0 else math.nan for u in MR.UNITS}


def kind_rec(net: ST.Net, c: Cfg, th: dict[str, float], r: Read, other: Read, v: dict[int, Tensor], gen: torch.Generator) -> dict[str, Any]:
    """The filed selection (largest contrast rise on the selection rows), then at that setting the reads on the 256 rows:
    own labels, the other arm's labels over its gain (same seed), off-target KL and norm-matched random vectors."""
    sel, ysel = r.p.part(r.arm, "train", c.n_select)
    grid = {(l, s): rises(net, r, sel, ysel, r.w0_sel, v[l], l, s)["contrast"] for l in FS.LAYERS for s in c.strengths}
    setting = FS.pick(grid)
    rec: dict[str, Any] = {"grid": {f"{l},{s}": x for (l, s), x in grid.items()}, "setting": setting}
    if setting is None:
        return {**rec, "share": dict.fromkeys(MR.UNITS, math.nan)}
    l, s = setting
    toks, y = r.p.part(r.arm, "fit", c.n_read)
    _, y_other = r.p.part(other.arm, "fit", c.n_read)
    rec["rise"] = rises(net, r, toks, y, r.w0[r.arm], v[l], l, s)
    rec["share"] = shares(rec["rise"], r.gain, th["gain_floor"])
    rec["cross"] = shares(rises(net, r, toks, y_other, r.w0[other.arm], v[l], l, s), other.gain, th["gain_floor"])
    rec["off_kl"] = FS.off_kl(net, toks, HALF, r.p.idx(toks), (v[l], l, s))
    draws = [shares(rises(net, r, toks, y, r.w0[r.arm], FS.random_like(v[l], gen), l, s), r.gain, th["gain_floor"]) for _ in range(c.n_random)]
    rec["random"] = {u: [d[u] for d in draws] for u in ("contrast", "lo")}
    rec["random_mean"] = {u: statistics.fmean(x) for u, x in rec["random"].items()}
    return rec


def neuron_rec(net: ST.Net, c: Cfg, th: dict[str, float], r: Read, dW: ST.DW, k: dict[int, int]) -> dict[str, Any]:
    toks, y = r.p.part(r.arm, "fit", c.n_read)
    cells = MR.topk_cells(net, dW, toks, y, r.p.ab, HALF, k)
    num, gain = {u: cells["N01"][u] - cells["N00"][u] for u in MR.UNITS}, {u: cells["T1"][u] - cells["T0"][u] for u in MR.UNITS}
    return {"share": {u: MR.share(num[u], gain[u], th["gain_floor"]) for u in MR.UNITS}, "numerator": {u: float(num[u].mean()) for u in MR.UNITS},
            "gain": {u: float(gain[u].mean()) for u in MR.UNITS}, "N00_minus_T0": mean_diff(cells["N00"], cells["T0"])}  # fmt: skip


def reads(net: ST.Net, c: Cfg, pairs: dict[str, Pair], new: dict[str, Any]) -> tuple[dict[str, ST.DW], dict[str, Read]]:
    states, out = {}, {}
    for p in pairs.values():
        w0 = {a: net_units(net, p, a, *p.part(a, "fit", c.n_read)) for a in ARMS}
        for a in ARMS:
            sel = net_units(net, p, a, *p.part(a, "train", c.n_select))
            for s in SEEDS:
                n = f"{p.name}_{a}_s{s}"
                states[n] = state(p, a, s, new)
                out[n] = Read(p, a, mean_diff(net_units(net, p, a, *p.part(a, "fit", c.n_read), states[n]), w0[a]), w0, sel)
    return states, out


def stage3(net: ST.Net, c: Cfg, th: dict[str, float], pairs: dict[str, Pair], s0: dict[str, Any], s1: dict[str, Any], s2: dict[str, Any]) -> dict[str, Any]:
    states, rd = reads(net, c, pairs, s1["states"])
    ks = {"k64": dict.fromkeys(FS.LAYERS, c.k_fixed), "kmatched": {int(l): k for l, k in s0["k_matched"].items()}}
    out: dict[str, Any] = {"states": {}, "cosines": {}}
    pooled: dict[str, dict[int, Tensor]] = {}
    for i, (n, dW) in enumerate(states.items()):
        r = rd[n]
        other = rd[n.replace(r.arm, "twin" if r.arm == "real" else "real")]
        vs = r.p.vectors(net, dW)
        pooled[n] = vs["pooled"]
        gen = torch.Generator().manual_seed(c.random_seed + i)
        rec: dict[str, Any] = {"gain": r.gain, "off_kl_state": FS.off_kl(net, r.p.part(r.arm, "fit", c.n_read)[0], HALF, dW=dW)}
        rec["steering"] = {kd: kind_rec(net, c, th, r, other, vs[kd], gen) for kd in KINDS}
        rec["neuron"] = {kn: neuron_rec(net, c, th, r, dW, k) for kn, k in ks.items()}
        out["states"][n] = rec
        say(f"3 {n}", {"gain": r.gain, **{f"steer_{kd}": {x: rec["steering"][kd].get(x) for x in ("setting", "share", "cross", "off_kl", "random_mean")} for kd in KINDS},
                       **{f"neuron_{kn}": rec["neuron"][kn] for kn in ks}})  # fmt: skip
        write(3, out)
    for p in pairs:
        names = [f"{p}_{a}_s{s}" for a in ARMS for s in SEEDS]
        out["cosines"][p] = {"names": names, **{str(l): FS.cosine_matrix([pooled[n][l] for n in names]).tolist() for l in FS.LAYERS}}
    out["score"] = score(th, s2, out)
    say("3 score", out["score"])
    write(3, out)
    return out


def score(th: dict[str, float], s2: dict[str, Any], s3: dict[str, Any]) -> dict[str, Any]:
    """The filed rule per method and pair in the contrast between the labels (scored) and in log-odds (beside)."""
    st = s3["states"]
    col = {"share": lambda n, u: s2["states"][n]["share"][u], "steer_perword": lambda n, u: st[n]["steering"]["perkey"]["share"][u],
           "steer_pooled": lambda n, u: st[n]["steering"]["pooled"]["share"][u], "neuron_k64": lambda n, u: st[n]["neuron"]["k64"]["share"][u],
           "neuron_kmatched": lambda n, u: st[n]["neuron"]["kmatched"]["share"][u]}  # fmt: skip
    out: dict[str, Any] = {}
    for u in ("contrast", "lo"):
        out[u] = {m: {p: call([f(f"{p}_real_s{s}", u) for s in SEEDS], [f(f"{p}_twin_s{s}", u) for s in SEEDS], th) for p in PAIRS} for m, f in col.items()}
    out["outcome"] = outcome({m: out["contrast"][m] for m in SCORED}, th)
    return out


def main() -> int:
    require_filed(PRED)
    P = json.loads(PRED.read_text())
    settings: dict[str, Any] = {k: tuple(v) if isinstance(v, list) else v for k, v in P["settings"].items()}
    c = Cfg(**settings)
    th = P["thresholds"]
    assert th["gain_floor"] == MIN_GAIN and th["n"] == len(SEEDS), "predictions.json disagrees with the code"
    LARGE.resolve().mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    net, pairs = ST.Net("cuda"), {n: load_pair(n) for n in PAIRS}
    try:
        s0 = stage0(net, c, th, pairs)
        write(0, s0)
        print("STAGE 0 DONE", flush=True)
        s1 = stage1(net, c, th, pairs)
        write(1, s1)
        print("STAGE 1 DONE", flush=True)
        s2 = stage2(net, c, th, pairs, s1["states"])
        print("STAGE 2 DONE", flush=True)
        stage3(net, c, th, pairs, s0, s1, s2)
        print("STAGE 3 DONE", flush=True)
    except Stopped as e:
        print(f"STOP: {e}: tell the lead", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    hard_exit(main())
