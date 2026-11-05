"""E96: can we name what a 67M fine-tune runs through without being told where to look? Stages 0 and 1, on E62's copy and
lookup maps; no training. Plan: reports/plans/2026-11-05_plan_e96-locate-mechanism.md. Rules and expected values:
results/predictions.json (committed first; the run refuses otherwise; never edited), `commute.locate`.

  stage 0, alone: the states' sha256; G on six states equals E62's run.json; the zero update reads G = 0 and head salience 0.
    Exit 1 on a miss. No head share is read.
  nominators, saved in full before the sweep: head salience, MLP-unit salience, a ridge probe. W0's head means are recomputed
    on natural text; E62's heads.pt, which holds the prefix scores, is never opened.
  sweep, the reference: every head alone on W0 and the six states, W0's second-copy top-1 under each, the greedy best pair.
  scorer, last: arithmetic over the saved files; the only place the expected head is read.

Writes results/{stage0,nominators,sweep,score}.json as each stage ends, means.pt and mlp_scores.pt; every scored number is
also printed. Refuses if stage0.json exists: this run deletes nothing.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from commute import anchors as AN
from commute import copying as CP
from commute import locate as L
from commute import steering as ST
from commute.data import pile_rows
from commute.jobs import hard_exit
from commute.keydep import log, sha256
from commute.open_pair import jsonable
from commute.revival_kv import zero_units

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
E62 = HERE.parent / "induction_cue" / "results"
PRED, STAGE0, NOMS, SWEEP, SCORE = (RESULTS / f for f in ("predictions.json", "stage0.json", "nominators.json", "sweep.json", "score.json"))
NAMES = [f"{m}_s{s}" for m in ("copy", "lookup") for s in (0, 1, 2)]
BATCH = 64


def dump(path: Path, rec: dict[str, Any]) -> None:
    path.write_text(json.dumps(jsonable(rec), indent=1, sort_keys=True))


def map_of(name: str) -> str:
    return name.split("_")[0]


@dataclass
class Reads:
    net: ST.Net
    half: int
    means: Tensor
    W: dict[str, Tensor]  # map -> E62's 200 read windows, target in the last column
    dws: dict[str, ST.DW]
    induction: dict[str, Any]  # E62's settings for W0's second-copy top-1
    off0: dict[tuple[str, frozenset[CP.Head]], Tensor] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return 2 * self.half + 1

    @property
    def heads(self) -> list[CP.Head]:
        a = self.net.net.h[0].attn
        return [(b, h) for b in range(len(self.net.net.h)) for h in range(a.n_head)]

    def lo(self, w: Tensor, dW: ST.DW | None = None, heads: Sequence[CP.Head] = ()) -> Tensor:
        with CP.ablated(self.net, heads, self.means):
            return CP.read_lo(self.net, w, self.half, self.n, dW)

    def w0_off(self, m: str, heads: Sequence[CP.Head]) -> Tensor:
        key = (m, frozenset(heads))
        if key not in self.off0:
            self.off0[key] = self.lo(self.W[m], None, heads)
        return self.off0[key]

    def intact(self, name: str) -> Tensor:
        key = (name, frozenset[CP.Head]())
        if key not in self.off0:
            self.off0[key] = self.lo(self.W[map_of(name)], self.dws[name])
        return self.off0[key]

    def part(self, name: str, heads: Sequence[CP.Head]) -> dict[str, float]:
        """The state with the heads off: its gain net of W0 under the same heads, the share lost, its own mean log-odds and
        the probability at it."""
        m, dW = map_of(name), self.dws[name]
        G = float((self.intact(name) - self.w0_off(m, [])).mean())
        off = self.lo(self.W[m], dW, heads)
        G_abl = float((off - self.w0_off(m, heads)).mean())
        return {"G_abl": G_abl, "share": CP.share_lost(G, G_abl), "lo": float(off.mean()), "p": L.prob(float(off.mean()))}

    def top1(self, heads: Sequence[CP.Head]) -> float:
        with CP.ablated(self.net, heads, self.means):
            return CP.induction(self.net, self.induction)["top1_second"]


# ----------------------------------------------------------------------------- stage 0


def stage0(R: Reads, run: dict[str, Any], th: dict[str, Any]) -> None:
    """Writes stage0.json whether or not it passes; a miss ends the job with exit 1."""
    rec: dict[str, Any] = {"sha256": "states and windows match run.json (checked at load)", "states": {}, "zero": {}}
    zero = {k: torch.zeros_like(x).cpu() for k, x in R.net.W0.items()}
    for name, dW in R.dws.items():
        G = float((R.lo(R.W[map_of(name)], dW) - R.lo(R.W[map_of(name)])).mean())
        rec["states"][name] = {"G": G, "G_run": run["states"][name]["G"], "diff": abs(G - run["states"][name]["G"])}
    for m, w in R.W.items():
        s0 = L.head_salience(R.net, w, R.half, R.n, R.means)
        rec["zero"][m] = {"zero_G_max_abs": float((R.lo(w, zero) - R.lo(w)).abs().max()),
                          "salience_repeat_floor": float((L.head_salience(R.net, w, R.half, R.n, R.means) - s0).abs().max()),
                          "zero_salience_max_abs": float((L.head_salience(R.net, w, R.half, R.n, R.means, zero) - s0).abs().max())}  # fmt: skip
    rec["status"] = L.apparatus(rec["states"], rec["zero"], th)
    dump(STAGE0, rec)
    log(f"[stage0] {json.dumps(jsonable(rec))}")
    if rec["status"] != "pass":
        log(f"STOP: stage 0: {rec['status']}: tell the lead")
        hard_exit(1)


# ----------------------------------------------------------------------------- the nominators


def by_head(R: Reads, v: Tensor) -> dict[str, float]:
    return {CP.head_name(h): float(v[h]) for h in R.heads}


def salience(R: Reads) -> dict[str, dict[str, float]]:
    """Importance: minus (state's salience - W0's), the first-order gain lost with the head off. Largest first."""
    base = {m: L.head_salience(R.net, w, R.half, R.n, R.means) for m, w in R.W.items()}
    return {n: by_head(R, base[map_of(n)] - L.head_salience(R.net, R.W[map_of(n)], R.half, R.n, R.means, dW)) for n, dW in R.dws.items()}


def probe(R: Reads) -> dict[str, dict[str, float]]:
    """Ridge accuracy per head on each state, and on W0 as the reference. The target's encoding: W0's unembedding row."""
    E = R.net.net.lm_head.weight.detach().double().cpu()
    runs: dict[str, tuple[str, ST.DW | None]] = {**{f"W0_{m}": (m, None) for m in R.W}, **{n: (map_of(n), dW) for n, dW in R.dws.items()}}
    out = {}
    for name, (m, dW) in runs.items():
        x = L.head_outputs(R.net, R.W[m], R.half, R.n, dW)
        out[name] = {CP.head_name(h): L.ridge_accuracy(x[h], R.W[m][:, -1], E) for h in R.heads}
    return out


def units_off(R: Reads, name: str, units: list[int], w0_cache: dict[tuple[str, tuple[int, ...]], Tensor]) -> dict[str, float]:
    """The state and W0 with the units zeroed, on the odd rows of the state's map (the even rows chose the units)."""
    m, w, edit = map_of(name), R.W[map_of(name)][1::2], zero_units(units)
    key = (m, tuple(units))
    if key not in w0_cache:
        w0_cache[key] = L.read_edit(R.net, w, R.half, R.n, None, edit)
    lo0, lo1 = CP.read_lo(R.net, w, R.half, R.n), CP.read_lo(R.net, w, R.half, R.n, R.dws[name])
    off = L.read_edit(R.net, w, R.half, R.n, R.dws[name], edit)
    G, G_abl = float((lo1 - lo0).mean()), float((off - w0_cache[key]).mean())
    return {"G": G, "G_abl": G_abl, "share": CP.share_lost(G, G_abl), "p": L.prob(float(lo1.mean())), "p_off": L.prob(float(off.mean()))}


def mlp_units(R: Reads, C: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Top K units by gradient times weight on the state's own log-odds (even rows), read on the odd rows, against random sets
    of K, the same draws for every state. The units' cost to the other map's state of the same seed is the Located control."""
    K, total = C["mlp_k"], ST.N_LAYERS * ST.D_MLP
    cache: dict[tuple[str, tuple[int, ...]], Tensor] = {}
    out: dict[str, dict[str, Any]] = {}
    scores: dict[str, Tensor] = {}
    rng = np.random.default_rng([C["random_seed"], K])
    draws = [sorted(rng.choice(total, K, replace=False).tolist()) for _ in range(C["n_random"])]
    for name, dW in R.dws.items():
        m, s = name.split("_")
        scores[name] = L.unit_scores(R.net, R.W[m][0::2], R.half, R.n, dW)
        top = sorted(torch.topk(scores[name], K).indices.tolist())
        other = f"{'lookup' if m == 'copy' else 'copy'}_{s}"
        rec: dict[str, Any] = {**units_off(R, name, top, cache), "units": top, "control": units_off(R, other, top, cache),
                               "random_shares": [units_off(R, name, d, cache)["share"] for d in draws],
                               "top1_off": CP.induction(L.Edited(R.net, zero_units(top)), R.induction)["top1_second"]}  # fmt: skip
        log(f"[mlp {name}] {json.dumps(jsonable({k: v for k, v in rec.items() if k != 'units'}))}")
        out[name] = rec
    torch.save(torch.stack([scores[n] for n in NAMES]).float(), RESULTS / "mlp_scores.pt")
    return out


def nominators(R: Reads, C: dict[str, Any]) -> None:
    rec: dict[str, Any] = {"salience": salience(R), "probe": probe(R), "means": "results/means.pt, recomputed by copying.head_means on natural text", "order_of_mlp_scores_pt": NAMES}
    log(f"[nominators salience] {json.dumps(jsonable(rec['salience']))}")
    log(f"[nominators probe] {json.dumps(jsonable(rec['probe']))}")
    dump(NOMS, rec)
    rec["mlp"] = mlp_units(R, C)
    dump(NOMS, rec)


# ----------------------------------------------------------------------------- the sweep


def best_pair(R: Reads, name: str, first: CP.Head) -> tuple[CP.Head, dict[str, float]]:
    """Greedy: the head that, added to `first`, loses the largest share on this state; ties in block and head order."""
    shares = {h: R.part(name, [first, h])["share"] for h in R.heads if h != first}
    order = L.ranked({CP.head_name(h): s for h, s in shares.items()})
    if not order:
        return first, {}
    second = next(h for h in R.heads if CP.head_name(h) == order[0])
    return second, {CP.head_name(h): s for h, s in shares.items()}


def sweep_state(R: Reads, name: str) -> dict[str, Any]:
    m = map_of(name)
    lo = float(R.intact(name).mean())
    heads = {CP.head_name(h): R.part(name, [h]) for h in R.heads}
    rec: dict[str, Any] = {"G": lo - float(R.w0_off(m, []).mean()), "lo": lo, "p": L.prob(lo), "heads": heads}
    order = L.ranked({h: r["share"] for h, r in heads.items()})
    if order:
        first = next(h for h in R.heads if CP.head_name(h) == order[0])
        second, tried = best_pair(R, name, first)
        rec |= {"best_pair": f"{CP.head_name(first)}+{CP.head_name(second)}", "pair_tried": tried}
    return rec


def sweep(R: Reads) -> None:
    rec: dict[str, Any] = {"w0": {"top1_base": R.top1([]), "top1_off": {CP.head_name(h): R.top1([h]) for h in R.heads}, "lo": {m: float(R.w0_off(m, []).mean()) for m in R.W},
                                  "lo_off": {m: {CP.head_name(h): float(R.w0_off(m, [h]).mean()) for h in R.heads} for m in R.W}},
                           "states": {}, "pairs": {}}  # fmt: skip
    log(f"[sweep W0] {json.dumps(jsonable(rec['w0']))}")
    dump(SWEEP, rec)
    for name in NAMES:
        rec["states"][name] = sweep_state(R, name)
        log(f"[sweep {name}] {json.dumps(jsonable(rec['states'][name]))}")
        dump(SWEEP, rec)
    for pair in sorted({r["best_pair"] for r in rec["states"].values() if "best_pair" in r}):
        hs = [(int(a), int(b)) for a, b in (x.split(",") for x in pair.split("+"))]
        rec["pairs"][pair] = {"top1_off": R.top1(hs), "states": {n: R.part(n, hs) for n in NAMES}}
        log(f"[sweep pair {pair}] {json.dumps(jsonable(rec['pairs'][pair]))}")
        dump(SWEEP, rec)


# ----------------------------------------------------------------------------- the scorer


def scorer(th: dict[str, Any], expected: list[int], T: dict[str, Any]) -> None:
    """Runs last, over the saved files alone. The only place the expected head is read."""
    noms, swept, run = (L.from_saved(json.loads(p.read_text())) for p in (NOMS, SWEEP, E62 / "run.json"))
    rec = L.score(noms, swept, {n: run["states"][n]["share"] for n in NAMES}, th, T, CP.head_name((expected[0], expected[1])))
    dump(SCORE, rec)
    log(f"[score] {json.dumps(jsonable(rec))}")
    log(f"[outcome] {rec['outcome']}")
    for claim in rec["claims"]:
        log(f"[claim] {claim}")


# ----------------------------------------------------------------------------- load and main


def load(a: argparse.Namespace) -> tuple[Reads, dict[str, Any], dict[str, Any]]:
    rel = str(PRED.relative_to(REPO))
    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True, check=False)
    if not PRED.is_file() or not git("ls-files", rel).stdout.strip() or git("diff", "--quiet", "HEAD", "--", rel).returncode:
        sys.exit(f"{rel} is not committed unchanged: file and commit it before the run")
    if STAGE0.exists():
        sys.exit(f"{STAGE0} exists: move the results aside, this run deletes nothing")
    P, S, run = json.loads(PRED.read_text()), json.loads((E62 / "predictions.json").read_text())["settings"], json.loads((E62 / "run.json").read_text())
    large = a.large.resolve()
    files = {"windows": run["windows_sha256"], **{n: run["states"][n]["sha256"] for n in NAMES}}
    for f, sha in files.items():
        if sha256(large / f"{f}.pt") != sha:
            sys.exit(f"STOP: stage 0: {f}.pt is not the one run.json recorded: tell the lead")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # as E62 Amendment 2's re-read
    net = ST.Net(a.device)
    log(f"[host] {torch.cuda.get_device_name() if torch.cuda.is_available() else 'none'} git {git('rev-parse', 'HEAD').stdout.strip()} predictions {sha256(PRED)}")
    v = torch.load(large / "windows.pt", weights_only=False)
    half = S["half"]
    W = {m: CP.with_target(v["read"], v["target"][m]["read"]).tokens for m in ("copy", "lookup")}
    rows = pile_rows("train", 0, AN.ROWS, log)
    rows = rows[torch.arange(rows.shape[0]) % S["windows"]["heldout_every"] == 0]
    means = CP.head_means(net, CP.natural_spans(rows, half, S["mean_spans"], S["mean_seed"]))
    torch.save(means.cpu(), RESULTS / "means.pt")
    dws = {n: torch.load(large / f"{n}.pt", weights_only=True, map_location="cpu") for n in NAMES}
    return Reads(net, half, means, W, dws, S["induction"]), P, run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--large", type=Path, default=E62 / "large")
    a = ap.parse_args()
    R, P, run = load(a)
    th = P["thresholds"]
    stage0(R, run, th)
    nominators(R, P["settings"])
    sweep(R)
    scorer(th, P["expected_head"], P["expected_table"])
    hard_exit(0)


if __name__ == "__main__":
    main()
