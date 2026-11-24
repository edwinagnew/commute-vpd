"""E292 (cell P2.74 run 1): definition-lead's readings on the old near-realistic tasks' step-10 states and their
Gaussians (run.py's states, `gauss`), through definition-lead's code, called and never copied:

  eb_neurons  E_b over K at E229's neuron featurization (E277): experiments/needed_neurons `read_item` and `stage0` on a
              World with its `make(item)` shape, its RESULTS pointed at results/eb_neurons/; beside each item its
              `remainder_columns` (E_b_K_no_rem and the remainders' share of dd over K and changed, P1.61)
  eb_vpd      E_b over K at VPD (E268): `needed.needed_set` and `stand_in.write_change` on `kstar_vpd.featurize` of the
              released decomposition with E268's settings (kstar_needed's own `read_item` reads saved arrays only, so
              it cannot take a new item); stage 0 the do-nothing and `needed.identity_scan` on the control
  eapig       EAP-IG overlap (P2.69, E287): experiments/eapig_overlap `Run` with its `Items` extended by the trigger
              tasks' task sets (clean: held-out windows cut after the trigger; corrupted: the trigger token swapped by
              its `swap_column`, pool the task's own trigger tokens, or the five ladder triggers where a task has one);
              its own stage 0 first, then its parity check on each new task set; M's resampling range per task set
D is each state's task's read rows (run.py), read at the trigger (or E53's, E112's read position). A missing state
file (train not run) is skipped and logged. Each stage writes as it goes; a rerun skips what is written.

    sbatch --comment=E292 cluster/old_tasks_read.sbatch defs eb_neurons --shard 0 4 --tag n0
    sbatch --comment=E292 cluster/old_tasks_read.sbatch defs eb_vpd --shard 0 6 --tag v0
    sbatch --comment=E292 cluster/old_tasks_read.sbatch defs eapig --tag e0
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from commute import kstar_vpd as KV
from commute import near_tasks as NT
from commute import needed as IX
from commute import old_tasks as OT
from commute import rule_retention as R
from commute import stand_in as SI
from commute import steering as ST
from commute import vpd
from commute.jobs import (
    EXPERIMENTS,
    hard_exit,
    load_experiment,
    require_filed,
    write_stage,
)
from commute.open_pair import jsonable

RUN: Any = load_experiment("old_tasks_read", "run")  # Config, load, load_state, gauss, gauss_seed, state_file, log
RESULTS, log = RUN.RESULTS, RUN.log
TASK_OF = {s: t.name for t in NT.TASKS.values() for s in NT.states(t, "all") if "_e1_step" not in s}  # step-10 states only
LADDER_TRIGGERS = sorted({int(n.split("_t")[1].split("_")[0]) for n in NT.TASKS if n.startswith("r")})


class World:
    """needed_neurons' world shape over E292's items: "<state>" (F = M + its delta-W) and "<state>_rand" (its Gaussian),
    plus "do_nothing" (F a copy of M); D the state's task's read rows, read at its read position."""

    def __init__(self, net: ST.Net, C: Any, name: str) -> None:
        self.net, self.C, self.name, self.control = net, C, name, C.control
        net.net.requires_grad_(False)
        self.M, self.reads = R.Model(net.net), {}
        self.items = [x for s in TASK_OF for x in (s, f"{s}_rand")]

    def ready(self, item: str) -> bool:
        s = item.removesuffix("_rand")
        return RUN.state_file(NT.TASKS[TASK_OF[s]], s).is_file()

    def make(self, item: str) -> tuple[R.Model, R.Model, torch.Tensor, slice, dict[str, Any]]:
        state = self.control if item == "do_nothing" else item.removesuffix("_rand")
        task = NT.TASKS[TASK_OF[state]]
        if task.name not in self.reads:
            self.reads[task.name] = NT.rows(task).read
        r = self.reads[task.name]
        toks, at = r.toks.cuda(), slice(r.pos, r.pos + 1)
        if item == "do_nothing":
            return R.Model(OT.clone_net(self.net).net), self.M, toks, at, {}
        dW, sha = RUN.load_state(self.net, task, state)
        rec: dict[str, Any] = {"state": state, "task": task.name, "sha256": sha}
        if item.endswith("_rand"):
            dW, rec = RUN.gauss(self.C, dW, state), rec | {"draw": RUN.gauss_seed(self.C, state)}
        return R.Model(OT.with_delta(self.net, OT.as_params(dW)).net), self.M, toks, at, rec


def todo(w: World, shard: tuple[int, int]) -> list[str]:
    mine = w.items[shard[0] :: shard[1]]
    log(f"{w.name}: {len(mine)} of {len(w.items)} items; not trained yet, skipped: {[i for i in mine if not w.ready(i)]}")
    return [i for i in mine if w.ready(i)]


# ----------------------------------------------------------------------------- E_b over K, neurons (E277)


def eb_neurons(net: ST.Net, C: Any, shard: tuple[int, int], tag: str) -> None:
    NN: Any = load_experiment("needed_neurons", "run")
    NN.RESULTS, NN.EXPERIMENT = RESULTS / "eb_neurons", "E292 (needed_neurons' read_item)"
    for d in ("items", "large"):
        (NN.RESULTS / d).mkdir(parents=True, exist_ok=True)
    cfg, c229 = NN.load_config(), NN.load_e229()
    feat = NN.E229.feat_of(c229, NN.E229.load_pct(c229), cfg.band)
    w = World(net, C, "near_neurons")
    floor = NN.stage0(w, cfg, feat, tag)["identity"]["max_dev"]
    for k, item in enumerate(todo(w, shard)):
        NN.read_item(w, item, cfg, feat, floor)
        p = NN.RESULTS / "items" / f"{NN.fname(item)}.json"
        rec = json.loads(p.read_text())
        if "rem_dd_share_K_changed" not in rec:
            rec |= NN.remainder_columns(item)
            write_stage(p, jsonable(rec))
        log(f"ITEM {k + 1} ({item}): E_b over K {rec['E_b_K']}, without remainders {rec.get('E_b_K_no_rem')}, remainder share {rec.get('rem_dd_share_K_changed')}")


# ----------------------------------------------------------------------------- E_b over K, VPD (E268)


class VPD:
    """The released decomposition's (U, V), Delta and the split's bases (E226's split_tol), as E226's World holds them."""

    def __init__(self, net: ST.Net) -> None:
        tol = json.loads((EXPERIMENTS / "kstar_vpd" / "results" / "predictions.json").read_text())["settings"]["split_tol"]
        self.uv, self.delta = vpd.component_weights(net.model), vpd.weight_deltas(net.model)
        self.bases = KV.bases(self.uv, tol)

    def feat(self, M: R.Model, F: R.Model) -> KV.Feat:
        return KV.featurize(M, F, self.uv, self.delta, self.bases)


def changed(feat: KV.Feat) -> torch.Tensor:
    ch = torch.zeros(feat.n, dtype=torch.bool)
    ch[sorted(feat.changed)] = True
    return ch


def vpd_stage0(w: World, v: VPD, cfg: Any, path: Path) -> float:
    if path.is_file() and (done := json.loads(path.read_text())).get("done"):
        return float(done["identity"]["max_dev"])
    F, M, toks, at, _ = w.make("do_nothing")
    feat = v.feat(M, F)
    num, den, dd = SI.write_change(F.net, SI.vpd_sites(feat), feat.n, list(toks.split(cfg.cache_batch)))
    out: dict[str, Any] = {"dn_n_changed": len(feat.changed), "dn_dd_max": float(dd.abs().max()), "dn_E_b_K": SI.explained_share(num, den, dd, changed(feat))}
    F, M, toks, at, _ = w.make(w.control)
    out["identity"] = IX.identity_scan(F, v.feat(M, F), toks, at, cfg.examples, cfg.ident_vars, cfg.rows, cfg.logit_rows, cfg.cache_batch, log)
    miss = [f"identity: {out['identity']['flips']} argmax changes"] if out["identity"]["flips"] else []
    miss += [f"do-nothing: {k} {out[k]}" for k, want in (("dn_n_changed", 0), ("dn_dd_max", 0.0), ("dn_E_b_K", 1.0)) if out[k] != want]
    log(f"eb_vpd stage 0: {json.dumps(jsonable(out))}")
    if miss:
        write_stage(path, jsonable(out | {"stopped": "; ".join(miss)}))
        raise SystemExit(f"STOP: eb_vpd stage 0: {'; '.join(miss)}")
    write_stage(path, jsonable(out | {"done": True}))
    return float(out["identity"]["max_dev"])


def vpd_item(w: World, v: VPD, item: str, cfg: Any, floor: float, out_dir: Path) -> None:
    path = out_dir / "items" / f"{item}.json"
    if path.is_file():
        return
    t0 = time.time()
    F, M, toks, at, rec = w.make(item)
    feat = v.feat(M, F)
    ch = changed(feat)
    num, den, dd = SI.write_change(F.net, SI.vpd_sites(feat), feat.n, list(toks.split(cfg.cache_batch)))
    nd = IX.needed_set(F, feat, toks, at, cfg.derangements, cfg.examples, cfg.seed, cfg.rows, cfg.logit_rows, cfg.cache_batch, log)
    K = nd.K
    out = rec | {"item": item, "n_windows": len(toks), "n_vars": feat.n, "n_changed": int(ch.sum()), "size_K": int(K.sum()),
                 "n_changed_K": int((K & ch).sum()), "E_b_K": SI.explained_share(num, den, dd, K & ch), "E_b_changed": SI.explained_share(num, den, dd, ch),
                 "n_pairs": nd.n_pairs, "examples": nd.m, "margin_min": float(nd.run.margin.min()), "n_fragile_positions": int((nd.run.margin <= floor).sum()),
                 "fragile_floor": floor, "max_dev": nd.scan.max_dev, "seconds": time.time() - t0}  # fmt: skip
    write_stage(path, jsonable(out))
    log(f"{item}: VPD E_b over K {out['E_b_K']:.6f} (over all changed {out['E_b_changed']:.6f}), |K| {out['size_K']} ({out['n_changed_K']} changed) of {feat.n}; {out['seconds']:.0f} s")
    del F
    torch.cuda.empty_cache()


def eb_vpd(net: ST.Net, C: Any, shard: tuple[int, int], tag: str) -> None:
    cfg = load_experiment("kstar_needed", "run").load_config()  # E268's derangements, examples, seed, rows, batch, ident_vars
    out_dir = RESULTS / "eb_vpd"
    (out_dir / "items").mkdir(parents=True, exist_ok=True)
    w, v = World(net, C, "near_vpd"), VPD(net)
    floor = vpd_stage0(w, v, cfg, out_dir / f"stage0_{tag}.json")
    for k, item in enumerate(todo(w, shard)):
        vpd_item(w, v, item, cfg, floor, out_dir)
        log(f"ITEM {k + 1} DONE ({item})")


# ----------------------------------------------------------------------------- EAP-IG (P2.69's code)


def eapig(C: Any, scratch: Path) -> None:
    EG: Any = load_experiment("eapig_overlap", "run")
    EI: Any = load_experiment("eapig_overlap", "items")
    EG.RESULTS, EG.EXPERIMENT = RESULTS / "eapig", "E292 (eapig_overlap's Run)"
    sets = {s: f"p274_{NT.TASKS[t].source}" for s, t in TASK_OF.items() if NT.TASKS[t].kind == "trigger"}
    EG.TASKS = EG.TASKS + tuple(sorted(set(sets.values())))  # its resampling draws index the task set

    class Items(EI.Items):
        near: World | None = None  # E292's items on E242's net, set once that world is built

        def make(self, item: str) -> tuple[R.Model, dict[str, Any]]:
            if item.removesuffix("_rand") not in sets:
                return super().make(item)
            self.near = self.near or World(self.w.kl.net, C, "near_eapig")
            F, _, _, _, rec = self.near.make(item)
            return F, rec

        def task(self, name: str) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
            if not name.startswith("p274_"):
                return super().task(name)
            held = NT.trigger_windows(name.removeprefix("p274_"))[1]
            h = held.half
            clean = held.tokens[:, : h + 1].clone()
            own = sorted(set(clean[:, h].tolist()))
            pool = own if len(own) >= 2 else sorted(set(own) | set(LADDER_TRIGGERS))
            bad = EI.swap_column(clean, h, pool, self.cfg.pair_seed)
            return clean, bad, {"half": h, "read": h, "swapped_column": h, "pool": len(pool), "n": len(clean), "T": int(clean.shape[1]),
                                "unchanged_rows": int((bad == clean).all(1).sum())}  # fmt: skip

    cfg = EG.load_config()
    run = EG.Run(cfg, scratch)
    run.items = Items(cfg, EG.RESULTS / "edits")
    w = run.items.near = World(run.items.w.kl.net, C, "near_eapig")
    items = [i for i in w.items if i.removesuffix("_rand") in sets and w.ready(i)]
    run.every |= {i: sets[i.removesuffix("_rand")] for i in items}
    try:
        s0 = EG.RESULTS / "stage0.json"
        if not (s0.is_file() and json.loads(s0.read_text()).get("done")):
            run.stage0()  # its own: inputs, versions, adapter parity on copy24 and x2, full and empty circuits
        else:
            assert not run.stage_tree()
            run.start()
            run.kl_floor = json.loads(s0.read_text())["kl_floor"]
        floor = json.loads(s0.read_text())["native_floor"]
        for t in sorted(set(sets.values())):
            par = EG.RESULTS / f"parity_{t}.json"
            write_stage(EG.RESULTS / f"task_{t}.json", jsonable(run.items.task(t)[2]))  # unchanged_rows, filed check: 0
            if not par.is_file():
                eapig_parity(run, t, C.control if t == sets[C.control] else "M", floor, par)
            if "stopped" in json.loads(par.read_text()):  # that set's parity missed the floor: its items get no reading
                log(f"EAP-IG set {t}: parity stopped, its items skipped (no reading: parity failed)")
                items = [i for i in items if sets[i.removesuffix("_rand")] != t]
                continue
            run.m_on(t)
            run.ranges[t] = (rs := run.resampled(t, "M"))["range"]
            write_stage(EG.RESULTS / f"M_resampling_{t}.json", jsonable({k: v for k, v in rs.items() if k != "circuits"}))
        for k, item in enumerate(items):
            if not (EG.RESULTS / "items" / f"{EG.fname(item)}.json").is_file():
                run.item(item, 5)
            log(f"ITEM {k + 1} of {len(items)} DONE ({item})")
    finally:
        run.close()


def eapig_parity(run: Any, task: str, name: str, floor: float, path: Path) -> None:
    """Stage 0's parity on a new task set (their stops, their floor): M, and the control item on its own set."""
    clean, bad, _ = run.items.task(task)
    run.send(task)
    out: dict[str, Any] = {"task": task}
    for n in dict.fromkeys(("M", name)):
        native, out[f"native_floor_parts_{n}"] = run.native(n, clean, bad)
        run.load(n)
        p = out[f"parity_{n}"] = run.call("parity", task=task, native=str(native), batch=run.cfg.batch, first=run.cfg.full_position_windows)
        stops = ("rel_clean", "rel_corrupted", "rel_clean_all_positions", "rel_corrupted_all_positions", "rel_full_circuit_vs_clean", "rel_empty_circuit_vs_corrupted")
        if miss := [f"{n} {k} {p[k]:.3g} over {floor:.3g}" for k in stops if not p[k] <= floor]:
            write_stage(path, jsonable(out | {"stopped": miss}))
            log(f"STOP: EAP-IG parity on {task}: {miss}")
            return
    write_stage(path, jsonable(out | {"done": True}))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("eb_neurons", "eb_vpd", "eapig"))
    ap.add_argument("--shard", nargs=2, type=int, default=(0, 1), metavar=("I", "N"))
    ap.add_argument("--tag", default="local")
    ap.add_argument("--scratch", type=Path, default=Path(f"/ephemeral/{os.environ.get('USER', 'x')}/e292_eapig"))
    a = ap.parse_args()
    C, _ = RUN.load()
    RUN.require_stage0()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32, as E268, E277, E287
    if a.stage == "eapig":
        require_filed(load_experiment("eapig_overlap", "run").PRED)
        eapig(C, a.scratch)
    else:
        net = ST.Net("cuda")
        (eb_neurons if a.stage == "eb_neurons" else eb_vpd)(net, C, (a.shard[0], a.shard[1]), a.tag)
    log(f"E292 defs {a.stage} DONE; peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
    return 0


if __name__ == "__main__":
    hard_exit(main())
