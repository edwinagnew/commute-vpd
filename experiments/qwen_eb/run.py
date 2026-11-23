"""E270 (cell P3.14, qwen_eb): the write share E_b, the frozen definition's reading (decisions.md 2026-11-22 "Definition
of reuse frozen"; theory/2026-11-22_definition_frozen.md section 3), on Qwen3-1.7B, of the six E167 step-75 updates
P3.13 run 1 read (RL, the shuffled-reward run that collapsed, own-samples SFT; seeds 0 and 70), each beside a Gaussian
change of the same Frobenius norm in every matrix (`share_reading.gaussian`, P3.13's random reference, one fixed draw
per update). Card: card.md. Settings, expected values and rules: results/predictions.json (the run refuses unless it is
committed unchanged, or PRED_SHA matches).

E_b as on the 67M (`stand_in.write_change`, `explained`, `explained_share`): on the new decomposition p-9707d6ff (all
196 matrices: q, k, v, o, gate, up, down of 28 layers), per decomposed matrix its C subcomponents and one remainder;
delta-W split by the least-squares split over all C subcomponents (no alive selection), the rest to the remainder
(`commute.qwen_eb`, which builds the 67M's `kstar_vpd` site unchanged and writes its sums through second moments: why,
there). num, den, dd are summed over every position of the 64 maths read rows P3.13 run 1 read (rows 64-127 of
qwen_sft/results/maths_rows.json, 256 tokens each, 16,384 positions) in F's own forward (F = M + delta-W, fp32, TF32
off); E_b = sum over all variables of `explained` / sum of dd (1 when every dd is 0).
Departure from the 67M: there E_b was scored over K* (the mean-ablation circuit); here over all variables, since no K
or K* exists on Qwen (on the 67M K* held ~82% of variables, and random read 0.011 either way).
The split's subcomponent set: all C. A CPU dry run (2 rows, layer 0's seven matrices and layers.13.mlp.up_proj,
2026-11-22) left an outside share of 1.7e-7 of a random change's norm (U and V are full rank here, so the remainder
keeps almost nothing) and took 65 s of fp64 split for 8 matrices; scaled to 196 matrices at an L40's fp64 rate that is
an estimated 3-4 min per update, over the brief's ~1 min. The alive set would not cut it much: the cost is the d x d
basis rotations, not C. Stage 0 measures it (stage0.json seconds); the lead sets stage 1's time limit from it.

Stages (each record is written as it finishes and printed to the log):
  0  alone; any miss writes results/stopped.json naming the rule and exits 1. (a) the do-nothing: every dd exactly 0,
     E_b exactly 1. (b) gain x2: in every matrix delta-W = sum of u_c v_c^T over 32 subcomponents drawn (fixed seed)
     from those whose write is non-zero on the rows (den > 0 in the do-nothing): E_b >= gain_min. (c) a random change at
     R_s70's norms (a draw no stage-1 item uses): read for the floor and time only. (d) the gain control's forward again
     under bf16 autocast: floor = |E_b(bf16) - E_b(fp32)|. Also VPDSite's own sums beside these at one site (reported),
     seconds per update (weights, forward, split, sums) and peak GPU memory. results/stage0.json.
  1  needs stage0.json; no stop. The six updates, each then its random change: results/items/<name>.json; a rerun skips
     what is written. results/stage1.json: the gaps and the rule labels.
Inputs: the model (HF Qwen/Qwen3-1.7B at qwen_vpd.HF_REVISION); --decomp, p-9707d6ff
(~/commute-data/artifacts/qwen3-1.7b-p-9707d6ff-torch); the rows; the six states
experiments/qwen_rl_readable/results/large/run2/<name>/dW/step_75.safetensors (5.64 GB each, sha256 against
experiments/qwen_rl_readable/results/run2/<name>.json).

  python experiments/qwen_eb/run.py --decomp ~/commute-data/artifacts/qwen3-1.7b-p-9707d6ff-torch --stages 0
  python experiments/qwen_eb/run.py --decomp ~/commute-data/artifacts/qwen3-1.7b-p-9707d6ff-torch --stages 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from torch import Tensor

from commute import qwen_eb as QE
from commute import share_reading as SR
from commute import stand_in as ST
from commute.jobs import EXPERIMENTS, config_from_settings, hard_exit, require_filed
from commute.qwen_sft import load, sha256
from commute.qwen_vpd import (
    KINDS,
    SITES,
    decomp_id,
    fp32_matmuls,
    load_components,
    load_target,
    parse_site,
)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED = RESULTS / "predictions.json"
E167 = EXPERIMENTS / "qwen_rl_readable" / "results"
MATHS = EXPERIMENTS / "qwen_sft" / "results" / "maths_rows.json"
STATES = E167 / "large" / "run2"


@dataclass(frozen=True)
class Config:  # predictions.json "settings"
    decomp: str
    states: list[str]
    step: int
    random_draws: list[int]  # one per state, in order
    probe_state: str  # stage 0 (c): a random change at this state's norms
    probe_draw: int
    rows: list[int]  # [first, last) of the maths rows
    row_len: int
    batch: int  # rows per forward
    split_tol: float
    n_gain: int  # |S| per matrix
    gain_seed: int
    gain_min: float
    gap: float
    check_site: str  # where VPDSite's own sums are read beside the moments' (stage 0, reported)

    def __post_init__(self) -> None:
        assert len(self.random_draws) == len(self.states) == len(set(self.random_draws))
        assert self.probe_state in self.states and self.probe_draw not in self.random_draws
        assert self.rows[1] > self.rows[0] and (self.rows[1] - self.rows[0]) % self.batch == 0


C: Config


def log(msg: str) -> None:
    msg = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(msg, flush=True)
    with (RESULTS / "run.log").open("a") as f:
        f.write(msg + "\n")


def put(path: Path, rec: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1, default=float))
    log(f"[{path.relative_to(RESULTS)}] {json.dumps(rec, default=float)}")
    return rec


def get(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.is_file() else {}


def check(name: str, holds: bool, detail: dict[str, Any]) -> None:
    log(f"[check] {name}: {'ok' if holds else 'MISSED'} {json.dumps(detail, default=float)}")
    if not holds:
        put(RESULTS / "stopped.json", {"rule": name, "detail": detail})
        log(f"STOP {name}")
        hard_exit(1)


def rows() -> Tensor:
    r = json.loads(MATHS.read_text())["rows"][C.rows[0] : C.rows[1]]
    return torch.tensor([x[: C.row_len] for x in r])


class Job:
    """The model (F's weights written into it per item), the decomposition and the split's bases, on one device."""

    def __init__(self, decomp_dir: Path, dev: str = "cuda", sites: list[str] | None = None, n_rows: int = 0) -> None:
        self.dev = dev
        t0 = time.time()
        self.model = load_target(dev)
        self.comps = load_components(decomp_dir, dev)
        self.sites = sites or list(self.comps.sites)
        self.lin = {s: self.model.get_submodule(f"model.{s}") for s in self.sites}
        self.W0 = {s: self.lin[s].weight.detach().cpu().clone() for s in self.sites}
        self.batches = list(rows()[: n_rows or None].to(dev).split(C.batch))
        t1 = time.time()
        self.bases = {s: QE.host_basis(self.comps.vu(s)[1], self.comps.vu(s)[0], C.split_tol) for s in self.sites}
        self.load_s, self.bases_s = t1 - t0, time.time() - t1
        log(f"[job] {len(self.sites)} sites, load {self.load_s:.0f} s, bases {self.bases_s:.0f} s, {self.peak()}")

    def peak(self) -> str:
        return f"peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB" if self.dev == "cuda" else "cpu"

    def set_weights(self, change: QE.Change) -> None:
        for s in self.sites:
            w0, d = self.W0[s].to(self.dev), change(s)
            self.lin[s].weight.copy_(w0 if d is None else w0 + d.to(self.dev))

    def sync(self) -> None:
        if self.dev == "cuda":
            torch.cuda.synchronize()

    def read(self, change: QE.Change, bf16: bool = False, keep: str | None = None
             ) -> tuple[dict[str, Any], dict[str, Tensor]]:
        """E_b of F = M + change, and each site's num, den, dd (C + 1 each, the remainder last)."""
        t0 = time.time()
        self.set_weights(change)
        self.sync()
        t1 = time.time()
        S, y = QE.moments(self.model, self.batches, self.sites, bf16, keep and QE.input_of(keep))
        self.sync()
        secs = {"weights": t1 - t0, "forward": time.time() - t1, "split": 0.0, "sums": 0.0}
        sums, chks, extra = {}, {}, {}
        for s in self.sites:
            a = time.time()
            V, U = self.comps.vu(s)
            site, chks[s] = QE.site(s, self.W0[s].to(self.dev), self.lin[s].weight, U, V, QE.on(self.bases[s], self.dev))
            self.sync()
            b = time.time()
            sums[s] = QE.moment_sums(site, S[QE.input_of(s)])
            if s == keep and y is not None:
                extra["reuse_check"] = {"site": s} | QE.reuse_check(site, S[QE.input_of(s)], y)
            secs["split"] += b - a
            secs["sums"] += time.time() - b
        self.set_weights(lambda _s: None)
        rec = summarise(sums, chks) | extra | {"bf16": bf16, "decomp": C.decomp, "seconds": secs, "peak": self.peak()}
        return rec, {s: torch.stack(v) for s, v in sums.items()}


def summarise(sums: dict[str, tuple[Tensor, Tensor, Tensor]], chks: dict[str, dict[str, float]]) -> dict[str, Any]:
    num, den, dd = (torch.cat([v[i] for v in sums.values()]) for i in range(3))
    kind = torch.cat([torch.full((v[0].numel(),), KINDS.index(parse_site(s)[1])) for s, v in sums.items()])
    rem = torch.cat([torch.arange(v[0].numel()) == v[0].numel() - 1 for v in sums.values()])
    every = torch.ones_like(rem)
    dW2 = sum(c["dW2"] for c in chks.values())
    return {"E_b": ST.explained_share(num, den, dd, every), "explained": float(ST.explained(num, den).sum()),
            "dd": float(dd.sum()), "dd_max_abs": float(dd.abs().max()), "n_variables": num.numel(),
            "remainder_dd_share": float(dd[rem].sum() / dd.sum()) if dd.sum() > 0 else None,
            "E_b_by_kind": {k: ST.explained_share(num, den, dd, kind == i) for i, k in enumerate(KINDS) if (kind == i).any()},
            "outside_share": (sum(c["outside2"] for c in chks.values()) / dW2) ** 0.5 if dW2 > 0 else None,
            "split_checks_max": {k: max(c[k] for c in chks.values()) for k in ("split_fp64", "F_from_pieces", "M_from_pieces")}}


def item(job: Job, name: str, change: QE.Change, meta: dict[str, Any], bf16: bool = False, keep: str | None = None
         ) -> tuple[dict[str, Any], dict[str, Tensor]]:
    """Read one item, save its sums (results/large/sums/<name>.safetensors, fp64) and write results/items/<name>.json."""
    rec, sums = job.read(change, bf16, keep)
    (RESULTS / "large" / "sums").mkdir(parents=True, exist_ok=True)
    save_file({s.replace(".", "/"): v.contiguous() for s, v in sums.items()}, str(RESULTS / "large" / "sums" / f"{name}.safetensors"))
    return put(RESULTS / "items" / f"{name}.json", {"name": name} | meta | rec), sums


def state(job: Job, name: str) -> dict[str, Tensor]:
    """A step-75 delta-W (host, fp32), its sha256 checked against its pod's record."""
    path = STATES / name / "dW" / f"step_{C.step}.safetensors"
    want = json.loads((E167 / "run2" / f"{name}.json").read_text())["saves"][str(C.step)]["sha256"]
    got = sha256(path)
    check(f"sha256_{name}", got == want, {"file": str(path), "sha256": got, "want": want})
    dW = load(path)
    assert set(job.sites) <= set(dW), path
    return dW


def random_like(job: Job, dW: dict[str, Tensor], draw: int) -> dict[str, Tensor]:
    """P3.13's random reference: a Gaussian per matrix at the update's Frobenius norm there (seed null_seed(draw, site))."""
    return SR.gaussian({s: float(dW[s].double().norm()) for s in job.sites}, {s: tuple(dW[s].shape) for s in job.sites}, draw)


def gain_sets(job: Job, dens: dict[str, Tensor]) -> dict[str, Tensor]:
    """Per matrix, C.n_gain subcomponents drawn (seed gain_seed * 1000 + site index) from those whose write in M is
    non-zero on the rows (den > 0 in the do-nothing), sorted."""
    out = {}
    for s in job.sites:
        ok = (dens[s][1][:-1] > 0).nonzero().squeeze(1)
        check(f"gain_eligible_{s}", ok.numel() >= C.n_gain, {"eligible": ok.numel(), "n_gain": C.n_gain})
        g = torch.Generator().manual_seed(C.gain_seed * 1000 + SITES.index(s))
        out[s] = ok[torch.randperm(ok.numel(), generator=g)[: C.n_gain]].sort().values
    return out


# ----------------------------------------------------------------------------- stages


def stage0(job: Job) -> None:
    dn, sums = item(job, "do_nothing", lambda _s: None, {"control": "do-nothing"})
    check("do_nothing", dn["dd_max_abs"] == 0.0 and dn["E_b"] == 1.0, {k: dn[k] for k in ("dd", "dd_max_abs", "E_b")})
    sets = gain_sets(job, sums)
    put(RESULTS / "gain_sets.json", {s: v.tolist() for s, v in sets.items()})

    def gain(s: str) -> Tensor:
        V, U = job.comps.vu(s)
        return QE.gain_change(U, V, sets[s].to(job.dev))

    g32, _ = item(job, "gain_x2", gain, {"control": "gain x2", "n_gain": C.n_gain}, keep=C.check_site)
    check("gain_x2", g32["E_b"] >= C.gain_min, {"E_b": g32["E_b"], "gain_min": C.gain_min})
    g16, _ = item(job, "gain_x2_bf16", gain, {"control": "gain x2, bf16 autocast forward"}, bf16=True)
    dW = state(job, C.probe_state)
    R = random_like(job, dW, C.probe_draw)
    del dW
    pr, _ = item(job, "random_probe", R.get, {"control": "random", "state": C.probe_state, "draw": C.probe_draw})
    del R
    put(RESULTS / "stage0.json", {
        "do_nothing_E_b": dn["E_b"], "gain_x2_E_b": g32["E_b"], "gain_x2_bf16_E_b": g16["E_b"],
        "floor": abs(g16["E_b"] - g32["E_b"]), "random_probe_E_b": pr["E_b"], "reuse_check": g32.get("reuse_check"),
        "seconds": {"load": job.load_s, "bases": job.bases_s} | {k: pr["seconds"][k] for k in pr["seconds"]},
        "peak": job.peak(), "decomp": C.decomp, "done": True})
    log("STAGE 0 DONE")


def label(gaps: list[float], bar: float) -> str:
    """'above random' if every gap > bar, 'below random' if every gap < -bar, 'as random' if every |gap| <= bar,
    else 'mixed'."""
    if all(g > bar for g in gaps):
        return "above random"
    if all(g < -bar for g in gaps):
        return "below random"
    return "as random" if all(abs(g) <= bar for g in gaps) else "mixed"


def verdicts() -> dict[str, Any]:
    E = {p.stem: get(p)["E_b"] for p in (RESULTS / "items").glob("*.json")}
    gaps = {n: E[n] - E[f"{n}_random"] for n in C.states}
    arms = {n.split("_s")[0] for n in C.states}
    s0 = get(RESULTS / "stage0.json")
    return {"E_b": {n: E[n] for n in C.states}, "E_b_random": {n: E[f"{n}_random"] for n in C.states}, "gap": gaps,
            "label": {a: label([g for n, g in gaps.items() if n.split("_s")[0] == a], C.gap) for a in sorted(arms)},
            "card_stop_every_update_as_random": all(abs(g) <= C.gap for g in gaps.values()),
            "gain_x2_E_b": s0["gain_x2_E_b"], "floor": s0["floor"], "decomp": C.decomp}


def stage1(job: Job) -> None:
    if not get(RESULTS / "stage0.json").get("done"):
        sys.exit("stage 1 needs results/stage0.json (run --stages 0 first)")
    for name, draw in zip(C.states, C.random_draws, strict=True):
        a, b = RESULTS / "items" / f"{name}.json", RESULTS / "items" / f"{name}_random.json"
        if a.is_file() and b.is_file():
            continue
        dW = state(job, name)
        if not a.is_file():
            item(job, name, dW.get, {"state": name, "step": C.step})
        R = random_like(job, dW, draw)
        del dW
        if not b.is_file():
            item(job, f"{name}_random", R.get, {"state": name, "draw": draw})
        del R
    put(RESULTS / "stage1.json", verdicts())
    log("STAGE 1 DONE")


def main() -> None:
    global C
    ap = argparse.ArgumentParser(description="E270: E_b of the six step-75 Qwen updates on p-9707d6ff")
    ap.add_argument("--decomp", required=True, help="p-9707d6ff's directory")
    ap.add_argument("--stages", default="0", choices=["0", "1", "0,1"])
    a = ap.parse_args()
    C = config_from_settings(Config, json.loads(PRED.read_text())["settings"])
    require_filed(PRED)
    if (RESULTS / "stopped.json").is_file():
        sys.exit(f"refusing to start: {RESULTS / 'stopped.json'} exists (a control missed; the lead decides)")
    assert decomp_id(Path(a.decomp)) == C.decomp, a.decomp
    fp32_matmuls()
    with torch.no_grad():
        job = Job(Path(a.decomp))
        if "0" in a.stages:
            stage0(job)
        if "1" in a.stages:
            stage1(job)
    hard_exit(0)


if __name__ == "__main__":
    main()
