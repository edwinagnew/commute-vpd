"""E149 (prior_steering), cell P2.15: do the original model's prior and steering, together, put the sides of every built
pair the way the re-gating share does? Steering is read in two forms: as filed (E133's table: strength and layer chosen by
the largest rise, E135/E143/E147) and at strength 1 (each fine-tune's mean change added as it is, the layer still chosen by
the filed rule), the second read here on every built pair's saved states. Plan reports/plans/2026-11-10_plan_prior-steering.md;
card 2026-11-10_plan_p2-15-card.md; filed rule results/predictions.json (the run refuses unless it is committed unchanged),
applied by rules.py. Sides are named as built; no verdict on reuse.

Two reading paths, as the filed cells were read: E143's (holes_fill `Job`, `lo_at`, `lo_steered`, `fair_steering.vectors`) for
the E136-inventory pairs and E147's 96-key refits; E135's (steering_copy_lookup) for copy vs lookup. At strength 1 only the
layer is chosen (layers 1-3, layer 0's vector is zero), on each path's own selection rows, then read on its read rows.

  stage 0  controls, alone, exit 1 on a miss: a zero update steers by exactly 0 at strength 1 on both paths;
           e54/elicit_s0 and c3/cipher_s0 reproduce E146's strength-1 shares and E143's saved selection rises;
           e62/copy_s0 reproduces E135's filed pooled share and its saved selection rises at strength 1.
  stage 1  every state of the 9 built pairs at strength 1, written as it lands, each checked against every saved selection
           grid that holds it (E143, E135, E147 if its stage 1 has the state) and E146's strength-1 share where E146 read
           it; then the score (rules.score) on final.json's share and prior cells, filed steering and these reads.

    sbatch --comment=E149 cluster/prior_steering.sbatch
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import rules
import torch

from commute import fair_steering as FS
from commute import keydep as KD
from commute import steering as ST
from commute.jobs import hard_exit, load_experiment, require_filed
from commute.open_pair import jsonable

HF = load_experiment("holes_fill", "run")
SC = load_experiment("steering_copy_lookup", "run")
WIDER = load_experiment("holes_fill", "rules").wider
HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
PRED = RESULTS / "predictions.json"
EXP = REPO / "experiments"
FILES = {"final": EXP / "method_table/results/final.json", "e136": EXP / "share_untrained/results/inventory.json",
         "e143": EXP / "holes_fill/results/stage1.json", "e143_pred": EXP / "holes_fill/results/predictions.json",
         "e146": EXP / "steering_reversal/results/stage1.json", "e135": EXP / "steering_copy_lookup/results/stage1.json",
         "e147_inv": EXP / "refit_96key/results/inventory.json", "e147_0": EXP / "refit_96key/results/stage0.json"}  # fmt: skip
E147_1 = EXP / "refit_96key/results/stage1.json"  # read when stage 1 starts and again at the score; may be partial
LAYERS, S1 = (1, 2, 3), 1.0
E62 = "e62_copy_lookup"


def log(msg: str) -> None:
    print(msg, flush=True)


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def write(n: int, out: dict[str, Any]) -> None:
    tmp = RESULTS / f"stage{n}.tmp"
    tmp.write_text(json.dumps(jsonable(out), indent=1))
    tmp.replace(RESULTS / f"stage{n}.json")


class Stopped(Exception):
    pass


def at_s1(grid: dict[str, float]) -> dict[int, float]:
    """A saved grid ('l,s' or '(l, s)' keys) -> {layer: value at strength 1} over LAYERS."""
    out = {}
    for k, v in grid.items():
        l, s = k.strip("()").split(",")
        if float(s) == S1 and int(l) in LAYERS:
            out[int(l)] = v
    return out


# ----------------------------------------------------------------------------- the two paths


def e58_inventory(inv147: dict[str, Any], seeds: list[int], maps: set[str]) -> dict[str, Any]:
    tasks = {f"e58/{m}": {"format": "kv", "path": inv147["maps"][m]["windows"], "batch": 64} for m in maps}
    states = {f"e58/{m}_s{s}": {"task": f"e58/{m}", "file": f"experiments/refit_96key/results/large/dW_{m}_s{s}.pt"} for m in maps for s in seeds}
    return {"tasks": tasks, "states": states}


def hf_state(job: Any, name: str, n_select: int) -> dict[str, Any]:
    """E143's path at strength 1: gain on the read rows; per kind the layer by the filed pick on the first n_select training
    windows, then the rise on the read rows over the gain."""
    t, dW = job.task(job.inv["states"][name]["task"]), job.dW(name)
    gain = float((HF.lo_at(job.net, t, t.read.toks, dW) - t.lo0).mean())
    vs = FS.vectors(job.net, t.x(t.train), t.idx(t.train), 1 if t.pool is None else len(t.pool), dW, t.read.half)
    sel = t.train[:n_select]
    sel0 = HF.lo_at(job.net, t, sel)
    out: dict[str, Any] = {"gain": gain}
    for kd in ("pooled",) if t.pool is None else ("pooled", "perkey"):
        rises = {l: float((HF.lo_steered(job.net, t, sel, vs[kd][l], l, S1) - sel0).mean()) for l in LAYERS}
        l = FS.pick({(l, S1): r for l, r in rises.items()})[0]  # type: ignore[index]  # the rises are finite (checked below)
        rise = float((HF.lo_steered(job.net, t, t.read.toks, vs[kd][l], l, S1) - t.lo0).mean())
        out[kd] = {"select": rises, "layer": l, "rise": rise, "share": HF.share(rise, gain, job.th["gain_floor"])}
    return out


class CopyLookup:
    """E135's path: its 200 windows (fit half selects, score half reads), its six states, W0 baselines and gains."""

    def __init__(self, net: ST.Net) -> None:
        s = load(SC.PRED)["settings"]
        self.net, self.c = net, SC.Cfg(s["half"], s["n"], tuple(s["strengths"]), s["n_random"], s["random_seed"])
        run62, large = load(SC.E62_DIR / "run.json"), (SC.E62_DIR / "large").resolve()
        self.p = SC.load_pair(self.c, large, run62)[0]
        self.dW = SC.load_states(large, run62)
        self.w0 = SC.baselines(net, self.p)
        self.G = SC.gains(net, self.p, self.dW, self.w0)

    def edit(self, v: torch.Tensor, l: int, s: float) -> ST.Edit:
        return SC.steer(v, l, s, self.p.keys[SC.SCORE], self.p.half)

    def state(self, name: str) -> dict[str, Any]:
        side = name.split("_")[0]
        vs = SC.vectors(self.net, self.p, self.dW[name])
        out: dict[str, Any] = {"gain": self.G[name]["lo"]}
        for kd in ("pooled", "perkey"):
            g = SC.grid_rises(self.net, self.p, side, vs[kd], self.w0[side]["fit"][0], SC.FIT, (S1,))
            rises = {l: g[(l, S1)] for l in LAYERS}
            l = SC.pick({(l, S1): r for l, r in rises.items()})[0]  # type: ignore[index]
            rec = SC.read_rec(self.net, self.p, side, self.w0, self.edit(vs[kd][l], l, S1), self.G[name])
            out[kd] = {"select": rises, "layer": l, "rise": rec["rise_lo"], "share": rec["share_lo"]}
        return out

    def filed_share(self, name: str, kd: str, saved: dict[str, Any]) -> float:
        side, v = name.split("_")[0], SC.vectors(self.net, self.p, self.dW[name])[kd]
        return SC.read_rec(self.net, self.p, side, self.w0, self.edit(v[saved["layer"]], saved["layer"], saved["strength"]), self.G[name])["share_lo"]


# ----------------------------------------------------------------------------- checks against saved reads


def saved_select(name: str, kd: str, F: dict[str, Any], e147: dict[str, Any]) -> list[tuple[str, dict[int, float]]]:
    """Every saved selection grid holding this state and kind, at strength 1."""
    out = []
    if name in F["e143"]["states"] and kd in F["e143"]["states"][name].get("steering", {}):
        out.append(("E143", at_s1(F["e143"]["states"][name]["steering"][kd]["grid"])))
    if name.startswith("e62/"):
        out.append(("E135", at_s1(F["e135"]["states"][name[4:]]["steering"][kd]["grid_fit"])))
    if name in e147.get("states", {}):
        out.append(("E147", at_s1(e147["states"][name]["holes_fill"]["steering"][kd]["grid"])))
    return out


def checks(name: str, rec: dict[str, Any], F: dict[str, Any], e147: dict[str, Any], tol: float) -> dict[str, Any]:
    """|difference| / gain per saved reading; a miss (or a nan) raises Stopped after the record is kept."""
    out: dict[str, Any] = {}
    for kd in (k for k in ("pooled", "perkey") if k in rec):
        r = rec[kd]
        for src, g in saved_select(name, kd, F, e147):
            out[f"{src} select {kd}"] = max(abs(r["select"][l] - g[l]) for l in LAYERS) / rec["gain"]
        e146 = F["e146"]["states"].get(name, {})
        if kd in e146:
            l = e146[kd]["picks"]["strength1"][0]
            out[f"E146 strength-1 share {kd}"] = abs(e146[kd]["grid"][f"({l}, 1)"] / e146["gain"] - r["share"]) if l == r["layer"] else math.inf
    bad = [k for k, v in out.items() if not v <= tol]
    if bad or any(math.isnan(rec[k]["share"]) for k in ("pooled", "perkey") if k in rec):
        raise Stopped(f"{name}: {bad or 'nan share'} {json.dumps(jsonable(out))}")
    return out


# ----------------------------------------------------------------------------- stages


def stage0(job: Any, cl: CopyLookup, F: dict[str, Any], th: dict[str, float], n_select: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    t = job.task("e54/elicit")
    zero = {k: torch.zeros_like(w) for k, w in job.net.W0.items()}
    vs = FS.vectors(job.net, t.x(t.train), t.idx(t.train), len(t.pool), zero, t.read.half)
    b = t.train[: FS.BATCH]
    lg0 = job.net.run(t.x(b))[0][:, t.read.half]
    out["zero_e143_path"] = max(float((FS.logits_steered(job.net, t.x(b), t.idx(b), t.read.half, vs[kd][l], l, S1) - lg0).abs().max())
                                for kd in ("pooled", "perkey") for l in LAYERS)  # fmt: skip
    vz = SC.vectors(cl.net, cl.p, zero)
    out["zero_e135_path"] = max(abs(r) for kd in ("pooled", "perkey") for r in SC.grid_rises(cl.net, cl.p, "copy", vz[kd], cl.w0["copy"]["fit"][0], SC.FIT, (S1,)).values())
    log(f"[0 zero update, largest |change| at strength 1] E143 path {out['zero_e143_path']} E135 path {out['zero_e135_path']}")
    if not (out["zero_e143_path"] <= th["zero_tol"] and out["zero_e135_path"] <= th["zero_tol"]):
        raise Stopped("zero update steers")
    out["reproduce"] = {}
    for name in ("e54/elicit_s0", "c3/cipher_s0"):
        out["reproduce"][name] = checks(name, hf_state(job, name, n_select), F, {}, th["reproduce_tol"])
    saved = F["e135"]["states"]["copy_s0"]["steering"]["pooled"]
    out["reproduce"]["e62/copy_s0"] = {**checks("e62/copy_s0", cl.state("copy_s0"), F, {}, th["reproduce_tol"]),
                                       "E135 filed share pooled": abs(cl.filed_share("copy_s0", "pooled", saved) - saved["share_lo"])}  # fmt: skip
    log(f"[0 reproduce] {json.dumps(jsonable(out['reproduce']))}")
    if not out["reproduce"]["e62/copy_s0"]["E135 filed share pooled"] <= th["reproduce_tol"]:
        raise Stopped("E135's filed share does not reproduce")
    return out


def stage1(job: Any, cl: CopyLookup, F: dict[str, Any], pairs: dict[str, Any], th: dict[str, float], n_select: int, t_job: float) -> dict[str, Any]:
    e147 = load(E147_1) if E147_1.is_file() else {}
    order = list(dict.fromkeys(s for p in pairs.values() for s in p["reuse"] + p["new"]))
    out: dict[str, Any] = {"states": {}, "e147_states_checked": sorted(set(order) & set(e147.get("states", {})))}
    t0 = time.time()
    for i, name in enumerate(order):
        rec = cl.state(name[4:]) if name.startswith("e62/") else hf_state(job, name, n_select)
        out["states"][name] = rec
        write(1, out)
        rec["checks"] = checks(name, rec, F, e147, th["reproduce_tol"])
        log(f"[1 {name}] gain {rec['gain']:.4f} " + " ".join(f"{kd} layer {rec[kd]['layer']} share {rec[kd]['share']:.4f}" for kd in ("pooled", "perkey") if kd in rec))
        write(1, out)
        minutes = ((time.time() - t0) / (i + 1) * len(order) + (t0 - t_job)) / 60
        log(f"[1 time] {i + 1} of {len(order)}; projected {minutes:.1f} min")
        if i >= 3 and minutes > th["projected_limit_min"]:
            raise Stopped(f"projected {minutes:.1f} min")
    return out


# ----------------------------------------------------------------------------- the score (no model)


def filed_steering(pair: str, cell: dict[str, Any], e147: dict[str, Any]) -> dict[str, Any]:
    """The filed steering call: E147's (wider kind, E143's rule) for a 96-key pair once its score exists, else final.json's."""
    if pair.startswith("e58_") and "score" in e147:
        pp, pk = (e147["score"]["calls"][k][pair] for k in ("steer_pooled", "steer_perkey"))
        r = WIDER(pp, pk)
        return {"vote": rules.steering_vote(r), "gap": r["gap"], "call": r["call"], "kind": "pooled" if r is pp else "perkey", "source": "E147 stage1 score.calls"}
    kind = cell["form"] if cell["form"] in ("pooled", "perkey") else "pooled"
    return {"vote": rules.sign(cell["gap"]) if cell["call"] == "apart" else 0, "gap": cell["gap"], "call": cell["call"], "kind": kind,
            "source": f"final.json ({cell['source'].split(' ')[0]}, {cell['form']})"}  # fmt: skip


def score_inputs(pairs: dict[str, Any], states: dict[str, Any], F: dict[str, Any], th: dict[str, float]) -> dict[str, Any]:
    e147 = load(E147_1) if E147_1.is_file() else {}
    cells = {(c["pair"], c["method"]): c for c in F["final"]["cells"]}
    out = {}
    for p, members in pairs.items():
        sh, pr = cells[(p, "share")], cells[(p, "prior")]
        filed = filed_steering(p, cells[(p, "steering")], e147)
        reuse, new = ([states[n][filed["kind"]]["share"] for n in members[side]] for side in ("reuse", "new"))
        s1 = rules.CALL(reuse, new, th)
        out[p] = {"name": sh["pair_name"], "share_sign": rules.sign(sh["gap"]) if sh["call"] == "apart" else 0, "prior": (pr["reuse"], pr["new"]),
                  "steering": {"filed": filed, "strength1": {**s1, "vote": rules.steering_vote(s1), "kind": filed["kind"], "reuse": reuse, "new": new}}}  # fmt: skip
    return out


def main() -> int:
    t_job = time.time()
    require_filed(PRED)
    P = load(PRED)
    assert P["experiment"].startswith("E149")
    th, F = P["thresholds"], {k: load(v) for k, v in FILES.items()}
    e143_c = F["e143_pred"]["settings"]
    inv147, built = F["e147_inv"], [c["pair"] for c in F["final"]["cells"] if c["group"] == "built" and c["method"] == "share"]
    e58 = {p["id"]: p for p in inv147["pairs"] if p["id"] in built}
    mine = e58_inventory(inv147, inv147["seeds"], {m for p in e58.values() for m in (p["reuse"], p["new"])})
    for name, st in mine["states"].items():  # the refits E147's stage 0 checked, unchanged since
        assert F["e147_0"]["checks"][f"b reproduce {name}"]["pass"] and KD.sha256(REPO / st["file"]) == F["e147_0"]["states"][name]["sha256"], name
    pairs = {p["id"]: {"reuse": p["reuse"], "new": p["new"]} for p in F["e136"]["pairs"] if p["id"] in built}
    pairs |= {i: {side: [f"e58/{p[side]}_s{s}" for s in inv147["seeds"]] for side in ("reuse", "new")} for i, p in e58.items()}
    assert sorted(pairs) == sorted(built) == sorted(P["pairs"]), (sorted(pairs), sorted(built))
    inv = {"tasks": {**F["e136"]["tasks"], **mine["tasks"]}, "states": {**F["e136"]["states"], **mine["states"]}}
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    job = HF.Job(ST.Net("cuda"), inv, e143_c, F["e143_pred"]["thresholds"])
    cl = CopyLookup(job.net)
    try:
        write(0, stage0(job, cl, F, th, e143_c["n_select"]))
        log("STAGE 0 DONE")
        out = stage1(job, cl, F, pairs, th, e143_c["n_select"], t_job)
    except Stopped as e:
        log(f"STOP: {e}: tell the lead")
        return 1
    out["score"] = rules.score(score_inputs(pairs, out["states"], F, th), th["prior_bar"])
    write(1, out)
    log(f"[1 score] {json.dumps(jsonable(out['score']))}")
    log("STAGE 1 DONE")
    return 0


if __name__ == "__main__":
    hard_exit(main())
