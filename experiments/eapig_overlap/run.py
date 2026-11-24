"""P2.69 (definition-lead): Wang et al. 2025's circuit overlap before against after an update, run with their pinned
code (FinetuneCircuits fdaeeaac63d5's `eap/` tree and its notebook's KL and faithfulness functions) on our items: an
adaptation of their section 4.2 endpoint comparison, a published baseline reported as a description. Spec
reports/plans/2026-11-23_plan_p2-69-eap-ig-overlap.md and its appendix; card experiments/eapig_overlap/card.md;
settings, filed expected values and rules in results/predictions.json, which the run refuses unless committed unchanged.

Two processes. This one (the project venv) builds every item with its original loader (items.py), saves its full native
state to node-local scratch, and does the counting (overlap.py). The worker (eap_worker.py, a Python 3.10 venv pinned to
FC's requirements, `cluster/eapig_overlap_venv.sbatch`) converts each state to a TransformerLens model (tl_convert.py)
and runs the released attribution, circuit selection and evaluation, one JSON request per line. FC's `eap/` is copied
from artifacts/ to scratch with every file's sha256 checked against the filed record, and imported from there.

Stages, in order, each writing its JSON as it ends (every item's JSON as soon as it is read) and every scored number to
the log. Only stage 0 stops the job; the controls are reported, and a missed control labels every later value.
    0  adapter on M and x2: inputs' and staged files' sha256, pinned versions, destination coverage, converted logits
       against the native forward (clean and corrupted, read position and all positions of the first windows), the
       full and empty circuits through the released code, all within the native forward's batched-against-single floor
       (measured first). Not stops: do-nothing (M against M) edge IoU, the fp32 KL floor for "no gap", seconds per item
    1  control 1: x2 (edge set equal to M's, exactly); x10; resampling of M and of x2 on copy24
    2  control 2: new_few, new_thin (edge IoU under M's resampling range on the cue windows), resampling of M and of
       new_thin; rebuilt; the two matched randoms (inside the range, reported)
    3  E62 copy and lookup, seeds 0-2, and the random at copy_s0's norms; head (2,4) in M's copy circuit
    4  key grid: 15 scaled states and their 15 randoms; E237's R_vpd order beside

    python experiments/eapig_overlap/run.py --scratch /ephemeral/$USER/eapig_overlap/local          (every stage)
    python experiments/eapig_overlap/run.py --stages 3 4 --scratch ...     (stage 0 must be done in results/stage0.json)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import overlap as OV
import torch
from items import Items, input_records, items_of
from torch import Tensor

from commute.jobs import (
    config_from_settings,
    hard_exit,
    require_filed,
    stop,
    write_stage,
)
from commute.keydep import sha256

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RESULTS = HERE / "results"
PRED = RESULTS / "predictions.json"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
TASKS = ("copy24", "cue", "e62_copy", "e62_lookup", "key_copy", "key_rank1", "key_elicit", "key_cipher", "key_prior_cipher")
EAP_FILES = ("attribute.py", "dataset.py", "evaluate.py", "graph.py", "metrics.py", "utils.py", "visualization.py")


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    fc_commit: str
    eap_dir: str  # FC's eap/ inside artifacts/, copied (never imported from there)
    eap_sha256: dict[str, str]
    venv: str  # the Python 3.10 venv, ~ expanded
    versions: dict[str, str]
    batch: int
    ig_steps: int
    top_frac: float
    faith_bar: float  # Wang's percentage
    greedy_bar: float
    greedy_ladder: tuple[int, ...]
    percentile: float
    resamples: int
    resample_seed: int
    resample_F: tuple[str, ...]
    pair_seed: int
    e62_random_seed: int
    e62_half: int
    floor_batch: int  # the native forward's single-window batch, against `batch`
    full_position_windows: int
    e195_C: tuple[int, ...]
    key_tasks: tuple[str, ...]
    key_seeds: tuple[int, ...]
    head: tuple[int, int]
    input_sha256: dict[str, str]

    def __post_init__(self) -> None:
        assert (self.batch, self.ig_steps, int(OV.N_EDGES * self.top_frac)) == (32, 5, 46), "FC's call: batch 32, 5 steps, 46 edges"
        assert list(self.greedy_ladder) == sorted(set(self.greedy_ladder)) and self.greedy_ladder[-1] == OV.N_EDGES
        assert set(self.eap_sha256) == set(EAP_FILES) and self.resamples >= 2 and self.floor_batch < self.batch
        assert len(self.e195_C) == 8 and len(self.key_tasks) == 5 and set(self.resample_F) == {"x2", "new_thin"}


def load_config() -> Config:
    s = {k: tuple(tuple(x) if isinstance(x, list) else x for x in v) if isinstance(v, list) else v for k, v in json.loads(PRED.read_text())["settings"].items()}
    return config_from_settings(Config, s)


def stages_of(cfg: Config) -> dict[int, tuple[list[str], list[str]]]:
    """Stage -> (items in order, task sets resampled), the spec's order."""
    every = items_of(cfg)
    e62, keys = [i for i in every if every[i].startswith("e62")], [i for i in every if every[i].startswith("key_")]
    return {1: (["x2", "x10"], ["copy24"]), 2: (["new_few", "new_thin", "rebuilt", "random_size_241001", "random_kl_241002"], ["cue"]),
            3: (e62, ["e62_copy", "e62_lookup"]), 4: (keys, [f"key_{t}" for t in cfg.key_tasks])}  # fmt: skip


def log(msg: str) -> None:
    print(msg, flush=True)


def fname(name: str) -> str:
    return name.replace("/", "-")


def rel(a: Tensor, b: Tensor) -> float:
    """Largest per-window max |a - b| over max |b|, as eap_worker.rel."""
    a, b = a.flatten(1).double(), b.flatten(1).double()
    return float(((a - b).abs().amax(1) / b.abs().amax(1)).max())


@torch.no_grad()
def native_read(F: Any, toks: Tensor, batch: int) -> Tensor:
    """The native forward (`rule_retention.Model.logits`, fp32) at the read position (last column), on the CPU."""
    return torch.cat([F.logits(x.cuda())[:, -1].cpu() for x in toks.split(batch)])


class WorkerError(RuntimeError):
    pass


class Run:
    def __init__(self, cfg: Config, scratch: Path) -> None:
        self.cfg, self.scratch = cfg, scratch
        (scratch / "states").mkdir(parents=True, exist_ok=True)
        self.items = Items(cfg, RESULTS / "edits")
        self.every = items_of(cfg)
        self.worker: subprocess.Popen[str] | None = None
        self.sent: dict[str, dict[str, Any]] = {}  # task set -> its windows record
        self.loaded = ""
        self.m_file: tuple[Path, str] | None = None
        self.m_attr: dict[str, dict[str, Any]] = {}  # task set -> M's full attribution
        self.kl_floor: float | None = None
        self.ranges: dict[str, list[float] | None] = {}  # task set -> M's resampling range, earlier stages' first
        for s in (1, 2, 3, 4):
            if (p := RESULTS / f"stage{s}.json").is_file():
                self.ranges |= {t: v["range"] for t, v in json.loads(p.read_text())["M_resampling"].items()}

    # ---- the worker ---------------------------------------------------------------------------------------------

    def stage_tree(self) -> list[str]:
        """FC's eap/ copied to scratch, each copy's sha256 against the filed record (the misses), and the pygraphviz stand-in."""
        root = self.scratch / "stage"
        shutil.rmtree(root, ignore_errors=True)
        (root / "eap").mkdir(parents=True)
        for f in EAP_FILES:
            shutil.copyfile(REPO / self.cfg.eap_dir / f, root / "eap" / f)
        shutil.copyfile(HERE / "pygraphviz_stub.py", root / "pygraphviz.py")
        return [f"staged eap/{f}: sha256 {sha256(root / 'eap' / f)}" for f in EAP_FILES if sha256(root / "eap" / f) != self.cfg.eap_sha256[f]]

    def start(self) -> None:
        env = os.environ | {"PYTHONPATH": str(self.scratch / "stage"), "TQDM_DISABLE": "1"}
        py = Path(os.path.expanduser(self.cfg.venv)) / "bin" / "python"
        self.worker = subprocess.Popen([str(py), str(HERE / "eap_worker.py")], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env, cwd=HERE)

    def call(self, op: str, **args: Any) -> dict[str, Any]:
        w = self.worker
        assert w is not None and w.stdin is not None and w.stdout is not None
        w.stdin.write(json.dumps({"op": op, "args": args}) + "\n")
        w.stdin.flush()
        line = w.stdout.readline()
        if not line:
            raise WorkerError(f"{op}: the worker exited (code {w.poll()})")
        res = json.loads(line)
        if not res.pop("ok"):
            log(res["traceback"])
            raise WorkerError(f"{op}: {res['error']}")
        return res

    def export(self, name: str, F: Any) -> tuple[Path, str]:
        """F's full native state (the tied embedding kept as one tensor), fp32, to scratch; its sha256."""
        memo: dict[int, Tensor] = {}
        sd = {k: memo.setdefault(v.data_ptr(), v.detach().float().cpu().clone()) for k, v in F.net.state_dict().items()}
        path = self.scratch / "states" / f"{fname(name)}.pt"
        torch.save(sd, path)
        return path, sha256(path)

    def load(self, name: str) -> dict[str, Any]:
        """The item's converted model in the worker (coverage checked there); its native file deleted after, M's kept."""
        if self.loaded == name:
            return {}
        t0 = time.time()
        if name == "M" and self.m_file is not None:
            path, sha, rec = *self.m_file, {}
        else:
            F, rec = self.items.make(name)
            assert not F.extra, f"{name}: appended neurons are not in the state"
            path, sha = self.export(name, F)
            del F
            torch.cuda.empty_cache()
        res = self.call("load", path=str(path), sha=sha)
        self.m_file = (path, sha) if name == "M" else self.m_file
        if name != "M":
            path.unlink()
        self.loaded = name
        rec = json.loads(json.dumps(rec, default=str))  # a loader's record as JSON (a stray tensor as text, not a crash later)
        return {"build": rec, "state_sha256": sha, "convert": res, "seconds_build_export_load": time.time() - t0}

    def send(self, task: str) -> dict[str, Any]:
        if task not in self.sent:
            clean, bad, rec = self.items.task(task)
            path = self.scratch / f"task_{task}.pt"
            torch.save({"clean": clean.long(), "corrupted": bad.long()}, path)
            self.sent[task] = rec | {"sha256": sha256(path)} | self.call("add_task", name=task, path=str(path))
            log(f"task {task}: {json.dumps(self.sent[task])}")
        return self.sent[task]

    def attr(self, task: str, full: bool = True, index: list[int] | None = None) -> dict[str, Any]:
        c = self.cfg
        return self.call("attribute", task=task, index=index, full=full, batch=c.batch, ig_steps=c.ig_steps, top_frac=c.top_frac,
                         ladder=list(c.greedy_ladder), greedy_bar=c.greedy_bar, kl_floor=self.kl_floor)  # fmt: skip

    def draws(self, task: str) -> list[list[int]]:
        n = self.send(task)["n"]
        gens = [torch.Generator().manual_seed(self.cfg.resample_seed + 1000 * TASKS.index(task) + k) for k in range(self.cfg.resamples)]
        return [torch.randint(0, n, (n,), generator=g).tolist() for g in gens]

    def resampled(self, task: str, model: str) -> dict[str, Any]:
        """`model`'s circuit on each full-size resample, and the IoU of every two (the diagnostic range)."""
        self.load(model)
        circuits = [self.attr(task, full=False, index=ix)["circuit"] for ix in self.draws(task)]
        ious = OV.pairwise(circuits)
        return {"ious": ious, "range": OV.span(ious), "sizes": [len(c["edges"]) for c in circuits], "circuits": circuits}

    # ---- stages -------------------------------------------------------------------------------------------------

    def native(self, name: str, clean: Tensor, bad: Tensor) -> tuple[Path, list[float]]:
        """The native forward's logits for the worker's parity check, and its batched-against-single floor parts."""
        c = self.cfg
        F = self.items.M if name == "M" else self.items.make(name)[0]
        nat = {"clean": native_read(F, clean, c.batch), "corrupted": native_read(F, bad, c.batch)}
        floors = [rel(nat[s], native_read(F, x, c.floor_batch)) for s, x in (("clean", clean), ("corrupted", bad))]
        with torch.no_grad():
            k = c.full_position_windows
            nat |= {"clean_full": F.logits(clean[:k].cuda()).cpu(), "corrupted_full": F.logits(bad[:k].cuda()).cpu()}
        path = self.scratch / f"native_{name}.pt"
        torch.save(nat, path)
        return path, floors

    def stage0(self) -> None:
        path, c = RESULTS / "stage0.json", self.cfg
        out: dict[str, Any] = {"experiment": EXPERIMENT}
        miss = [f"{k}: sha256 {got} != {want}" for k, (p, want) in input_records(c).items() if (got := sha256(p)) != want]
        miss += self.stage_tree()
        if miss:  # nothing staged or recorded wrong is imported or read
            stop(path, out, "stage 0: " + "; ".join(miss), log)
        self.start()
        out["worker"] = hello = self.call("hello")
        miss += [f"worker {k} {hello[k]}, filed {v}" for k, v in c.versions.items() if not str(hello[k]).startswith(v)]
        if miss:
            stop(path, out, "stage 0: " + "; ".join(miss), log)
        out["task"] = self.send("copy24")
        clean, bad, _ = self.items.task("copy24")
        native = {}
        for name in ("M", "x2"):
            native[name], out[f"native_floor_parts_{name}"] = self.native(name, clean, bad)
        out["native_floor"] = floor = max(out["native_floor_parts_M"] + out["native_floor_parts_x2"])
        log(f"stage 0: native forward batched-against-single floor {floor:.3g}")
        kl = []
        for name in ("M", "x2"):
            try:
                out[f"load_{name}"] = self.load(name)
                p = out[f"parity_{name}"] = self.call("parity", task="copy24", native=str(native[name]), batch=c.batch, first=c.full_position_windows)
            except WorkerError as e:
                stop(path, out, f"stage 0: {name}: {e}", log)
            log(f"stage 0 {name}: coverage {json.dumps(out[f'load_{name}']['convert'])}; parity {json.dumps(p)}")
            stops = ("rel_clean", "rel_corrupted", "rel_clean_all_positions", "rel_corrupted_all_positions", "rel_full_circuit_vs_clean", "rel_empty_circuit_vs_corrupted")
            miss += [f"{name} {k} {p[k]:.3g} over the floor {floor:.3g}" for k in stops if not p[k] <= floor]
            miss += [f"{name} logits not finite ({s})" for s in ("clean", "corrupted") if not p[f"finite_{s}"]]
            kl += [abs(p["kl_tl_native_clean"]), abs(p["kl_batch_single_clean"])]
        out["kl_floor"] = self.kl_floor = max(kl)
        if miss:
            stop(path, out, "stage 0: " + "; ".join(miss), log)
        a = self.m_on("copy24")  # after the floor exists, so M's "no gap" reads it
        b = self.attr("copy24", full=False)
        out["do_nothing"] = {"edge_iou": OV.iou(set(a["circuit"]["edges"]), set(b["circuit"]["edges"])), "scores_equal": a["scores"] == b["scores"],
                             "same_edges": a["circuit"]["edges"] == b["circuit"]["edges"], "edges": len(a["circuit"]["edges"])}  # fmt: skip
        per_item = out["load_x2"]["seconds_build_export_load"] + a["seconds"]
        out["seconds_per_item"], out["projected_minutes"] = per_item, per_item * (len(self.every) + 9 + 55) / 60
        log(f"stage 0: do-nothing {out['do_nothing']}; fp32 KL floor {self.kl_floor:.3g}; {per_item:.1f} s per item, about {out['projected_minutes']:.0f} min in all")
        write_stage(path, out | {"done": True})
        log("STAGE 0 DONE")

    def m_on(self, task: str) -> dict[str, Any]:
        if task not in self.m_attr:
            self.load("M")
            self.send(task)
            self.m_attr[task] = self.attr(task)
            write_stage(RESULTS / "items" / f"M_{task}.json", {k: v for k, v in self.m_attr[task].items() if k != "scores"} | {"task": task})
            write_stage(RESULTS / "large" / "scores" / f"M_{task}.json", self.m_attr[task]["scores"])
        return self.m_attr[task]

    def item(self, item: str, stage: int) -> dict[str, Any]:
        task, c = self.every[item], self.cfg
        rM, rec = self.m_on(task), self.load(item)
        rF = self.attr(task)
        rec |= {"item": item, "task": task, "stage": stage, "reading": OV.reading(rM, rF, c.faith_bar), "M_resampling_range": self.ranges.get(task),
                "five_percent": OV.compare(rM["circuit"], rF["circuit"]), "F_circuit": rF["circuit"],
                "sensitivity_95": OV.sensitivity(rM["scores"], rM["circuit"], rF["scores"], rF["circuit"], c.percentile),
                "greedy": OV.compare(rM["greedy"]["chosen"], rF["greedy"]["chosen"]) if rM.get("greedy", {}).get("chosen") and rF.get("greedy", {}).get("chosen") else None}  # fmt: skip
        rec["M"], rec["F"] = (self.summary(r) for r in (rM, rF))
        if item in c.resample_F:
            rec["F_resampling"] = {k: v for k, v in self.resampled(task, item).items() if k != "circuits"}
        write_stage(RESULTS / "items" / f"{fname(item)}.json", rec)
        write_stage(RESULTS / "large" / "scores" / f"{fname(item)}.json", rF["scores"])
        f, s = rec["five_percent"], rec["sensitivity_95"]
        log(f"ITEM {item} ({task}): {rec['reading']}; edge IoU {f['edge_iou']} (M range {rec['M_resampling_range']}), node IoU {f['node_iou']} "
            f"(code {f['node_iou_code']}), edge p {f['edge_p']} (code {f['edge_p_code']}), node p {f['node_p']} (code {f['node_p_code']}); "
            f"edges unchanged/added/removed {f['edges']['unchanged']}/{f['edges']['added']}/{f['edges']['removed']}, nodes "
            f"{f['nodes']['unchanged']}/{f['nodes']['added']}/{f['nodes']['removed']}; dS edges {f['delta_S_edges']}, nodes {f['delta_S_nodes']}; "
            f"sizes {f['sizes']}; faithfulness M {rec['M']['faithfulness_pct']} F {rec['F']['faithfulness_pct']}, F(M) M {rec['M']['F_M']} "
            f"F {rec['F']['F_M']}; greedy IoU {(rec['greedy'] or {}).get('edge_iou')} (n M {rec['M']['greedy_n']}, F {rec['F']['greedy_n']}); "
            f"95th IoU {s['edge_iou']} ({s['M_edges']} vs {s['F_edges']} edges); F resampling {(rec.get('F_resampling') or {}).get('range')}")  # fmt: skip
        return rec

    @staticmethod
    def summary(r: dict[str, Any]) -> dict[str, Any]:
        g = r.get("greedy") or {}
        return {"F_M": r.get("F_M"), "F_C": r.get("F_C"), "faithfulness_pct": r.get("faithfulness_pct"), "no_gap": r.get("no_gap"),
                "edges": len(r["circuit"]["edges"]), "nodes": len(r["circuit"]["nodes"]), "empty": r["circuit"]["empty"],
                "tie_at_cut": r["tie_at_cut"], "finite_scores": r["finite_scores"], "b_prime": g.get("b_prime"),
                "greedy_n": (g.get("chosen") or {}).get("n"), "greedy_ladder": g.get("ladder"), "seconds": r["seconds"]}  # fmt: skip

    def stage(self, s: int) -> None:
        items, tasks = stages_of(self.cfg)[s]
        out: dict[str, Any] = {"experiment": EXPERIMENT, "stage": s, "kl_floor": self.kl_floor, "M_resampling": {}, "items": {}}
        out["method_failed_control"] = [k for k, v in self.controls().items() if v != "pass"]
        for t in tasks:
            self.m_on(t)
            out["M_resampling"][t] = {k: v for k, v in self.resampled(t, "M").items() if k != "circuits"}
            self.ranges[t] = out["M_resampling"][t]["range"]
            log(f"M resampling on {t}: range {self.ranges[t]}, IoUs {out['M_resampling'][t]['ious']}, sizes {out['M_resampling'][t]['sizes']}")
        for k, item in enumerate(items):
            try:
                r = self.item(item, s)
                out["items"][item] = {"reading": r["reading"], "edge_iou": r["five_percent"]["edge_iou"], "same_edges": r["five_percent"]["same_edges"]}
            except Exception as e:  # noqa: BLE001  (recorded; the next item runs: only stage 0 stops the job)
                out["items"][item] = {"error": f"{type(e).__name__}: {e}"}
                log(f"ITEM {item}: ERROR {type(e).__name__}: {e}")
                self.loaded = ""  # the worker's model is unknown after a failure
            log(f"ITEM {k + 1} of {len(items)} DONE ({item})")
        out |= self.stage_extras(s, out)
        write_stage(RESULTS / f"stage{s}.json", out | {"done": True})
        log(f"STAGE {s} DONE: {json.dumps({k: v for k, v in out.items() if k not in ('items', 'M_resampling')})}")

    def stage_extras(self, s: int, out: dict[str, Any]) -> dict[str, Any]:
        it, rg = out["items"], self.ranges
        if s == 1:
            return {"control_1_x2": "pass" if it.get("x2", {}).get("same_edges") else "miss"}
        if s == 2:
            ctl = {f"control_2_{i}": OV.below(it.get(i, {}).get("edge_iou"), rg.get("cue")) for i in ("new_few", "new_thin")}
            return ctl | {"random_size_241001_inside": OV.inside(it.get("random_size_241001", {}).get("edge_iou"), rg.get("copy24")),
                          "random_kl_241002_inside": OV.inside(it.get("random_kl_241002", {}).get("edge_iou"), rg.get("cue"))}  # fmt: skip
        if s == 3:
            m = self.m_attr["e62_copy"]
            node = "a{}.h{}".format(*self.cfg.head)
            out_score = {f"a{l}.h{h}": sum(abs(v) for e, v in m["scores"].items() if e.startswith(f"a{l}.h{h}->")) for l in range(4) for h in range(6)}
            rank = sorted(out_score, key=lambda n: -out_score[n]).index(node) + 1
            return {"head_2_4_in_M_copy_circuit": {"in_circuit": node in m["circuit"]["nodes"], "edges": [e for e in m["circuit"]["edges"] if node in e],
                                                    "rank_of_24_by_outgoing_score": rank}}  # fmt: skip
        r = json.loads((HERE.parent / "key_grid" / "results" / "summary.json").read_text())["table"]["R_vpd"]
        return {"e237_R_vpd_mean_beside": {t: r[t]["mean"] for t in self.cfg.key_tasks}}

    def controls(self) -> dict[str, str]:
        got: dict[str, str] = {}
        for s in (1, 2):
            p = RESULTS / f"stage{s}.json"
            if p.is_file():
                got |= {k: v for k, v in json.loads(p.read_text()).items() if k.startswith("control_")}
        return got

    def close(self) -> None:
        if self.worker is not None and self.worker.stdin is not None:
            self.worker.stdin.close()
            self.worker.wait(timeout=60)
        shutil.rmtree(self.scratch / "states", ignore_errors=True)


def main(stages: list[int], scratch: Path) -> int:
    cfg = load_config()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 throughout (appendix A, precision)
    log(f"{EXPERIMENT} start, stages {stages}, scratch {scratch}")
    run = Run(cfg, scratch)
    try:
        s0 = RESULTS / "stage0.json"
        if 0 in stages or not s0.is_file():
            run.stage0()
        rec = json.loads(s0.read_text())
        if not rec.get("done") or "stopped" in rec:
            log("STOP: stage 0 has not passed (results/stage0.json)")
            return 1
        if run.worker is None:
            misses = run.stage_tree()
            assert not misses, misses
            run.start()
            run.kl_floor = rec["kl_floor"]
        for s in sorted(set(stages) - {0}):
            run.stage(s)
    finally:
        run.close()
    log(f"{EXPERIMENT} DONE")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", type=int, default=[0, 1, 2, 3, 4], choices=[0, 1, 2, 3, 4])
    ap.add_argument("--scratch", type=Path, required=True, help="node-local scratch (/ephemeral/$USER/...): native states, staged eap/")
    args = ap.parse_args()
    require_filed(PRED)
    code = 1
    try:
        code = main(args.stages, args.scratch)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)
