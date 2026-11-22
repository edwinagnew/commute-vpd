"""E### (cell P1.50; user, 2026-11-21, option (b) of ask 58): E_b at the VPD featurization of the 67M on saved fine-tuned
states. Settings and filed expected values in results/predictions.json, which the run refuses unless committed unchanged.

Option (b): reuse is read only over the parts the fine-tune changed, each weighted by its change; K*'s untouched share
is reported beside it. The order-free reading (P1.49, experiments/option_b_rescore/score.py, whose expression is
commute.stand_in.explained):
    E_b = sum over K*'s changed members u of (num_u - den_u)^2 / den_u  /  sum over the same members of dd_u
with num_u = <write_F_u, write_M_u>, den_u = |write_M_u|^2, dd_u = |write_F_u - write_M_u|^2 summed over every position
of D in F's own run (commute.stand_in.write_change), so (num - den)^2 / den is the part of u's write change that M's
own write at u's least-squares gain num / den explains; 0 for a gain <= 0 (may not stand in) and where M writes zero
(grafts). E_b = 1 when K* has no changed member (the do-nothing), the stated convention.

Per item: F, M and D as E242's World builds them (E248's items: E187's nine pairs at matched size, E62 copy/lookup, E237's
keys), K* as E226/E237/E242/E248 saved it (commute.kstar_vpd.saved_k_star) or, for items with none (the RL states, the
manifest), E226's read_item with E226's settings, writing here; one write_change pass. results/items/<item>.json holds
E_b, K*'s untouched share, |K*|, its changed members and grafts, ||dW|| / ||W0|| over the 24 decomposed matrices and
R under wording A where E242/E248 saved it; results/large/items_<item>.npz holds num, den, dd, changed and K*. A rerun
skips an item whose JSON exists. Stages, each job running stage 0 first (skipped once its file says done):
    0         alone, every miss stops the job: the do-nothing (no changed variable, every dd exactly 0, E_b exactly 1),
              then the gain control x2 in fp32 and in bf16 (autocast): floor = |E_b bf16 - E_b fp32|, and x2's fp32
              E_b must repeat the value from E247's saved arrays within that floor
    random    E242's two matched random updates
    items     E248's items (refuses until both randoms are written)
    rl        the saved RL repeat states (E220 stage 2b, size-matched; E218 stage 2, KL-matched), on E222's repeat read
              rows (results/large/rl_windows_<behaviour>.pt, built once by --stage rl_windows on the cluster: E220's
              context checks E220's predictions through git, so not on a pod)
    manifest  compare-lead's P2.56 states (later): the JSON at settings "manifest", a list of {"item", "dw" (repo path
              of a 24-matrix delta-W as commute.old_tasks.load_dw reads it), "windows" (an item whose D is read)}

    python experiments/rb_states/run.py --stage random --tag pod0          (on a pod: PRED_SHA=<sha> python ...)
    python experiments/rb_states/run.py --stage items --shard 0 4 --tag pod1
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from commute import kstar_vpd as KV
from commute import old_tasks as OT
from commute import rule_retention as R
from commute import stand_in as SI
from commute.jobs import (
    EXPERIMENTS,
    config_from_settings,
    hard_exit,
    load_experiment,
    require_filed,
)

E242: Any = load_experiment("kstar_vpd_wording_a", "run")  # World, group_items, load_config, log/dump/stop, E226
E226, KP = E242.E226, E242.KP
HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", EXPERIMENTS.parent
PRED = RESULTS / "predictions.json"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
S242, S247 = (EXPERIMENTS / e / "results" for e in ("kstar_vpd_wording_a", "thin_edit_diagnosis"))
log, dump, stop, fname = E242.log, E242.dump, E242.stop, E242.fname
STAGES = ("random", "items", "rl", "manifest")


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    control: str  # the gain control read in stage 0 (an E247 item with saved num, den, dd)
    randoms: tuple[str, ...]  # E242's matched random updates
    groups: tuple[str, ...]  # E242's item groups read in stage "items" (E248's)
    rl_behaviour: str  # the RL states' behaviour: its read rows are D
    rl_states: tuple[str, ...]  # repo paths of the saved RL delta-W
    manifest: str  # repo path of the later stage's state list (absent until compare-lead writes it)

    def __post_init__(self) -> None:
        c = E242.load_config()[0]
        assert self.control in c.built and set(self.randoms) == {c.random, c.random_kl}
        assert set(self.groups) <= {"pairs", "e62", "keys"} and all(f"/{self.rl_behaviour}_" in p for p in self.rl_states)

    @property
    def rl_file(self) -> Path:
        return RESULTS / "large" / f"rl_windows_{self.rl_behaviour}.pt"


def load_config() -> Config:
    s = {k: tuple(v) if isinstance(v, list) else v for k, v in json.loads(PRED.read_text())["settings"].items()}
    return config_from_settings(Config, s)


def rl_item(path: str) -> str:
    return "rl/" + str(Path(path).relative_to("experiments/rl_67m/results/large").with_suffix(""))  # rl/stage2b/repeat_rl_s0


class World(E242.World):
    """E242's world (E226's built items, E232's pairs, E237's keys), plus saved delta-W states on stated windows."""

    def __init__(self, cfg: Config) -> None:
        super().__init__(E242.load_config()[0])
        self.rl, self.rl_file = {rl_item(p): p for p in cfg.rl_states}, cfg.rl_file
        m = REPO / cfg.manifest
        self.extra = {e["item"]: e for e in json.loads(m.read_text())} if m.is_file() else {}

    def from_dw(self, path: Path, toks: Tensor, at: slice) -> tuple[R.Model, R.Model, Tensor, slice, dict[str, Any]]:
        F = R.Model(OT.with_delta(self.kl.net, OT.load_dw(self.kl.net, path)).net)
        return F, self.kl.M, toks, at, {"dw": str(path.relative_to(REPO))}

    def make(self, item: str) -> tuple[R.Model, R.Model, Tensor, slice, dict[str, Any]]:
        if item in self.rl:
            assert self.rl_file.is_file(), f"{self.rl_file} missing: run --stage rl_windows on the cluster first"
            return self.from_dw(REPO / self.rl[item], torch.load(self.rl_file, weights_only=False)["toks"].cuda(), KP.LAST)
        if item in self.extra:
            _, _, toks, at, _ = self.make(self.extra[item]["windows"])
            return self.from_dw(REPO / self.extra[item]["dw"], toks, at)
        return super().make(item)


def rl_windows(cfg: Config) -> None:
    """E222's read rows of the behaviour (experiments/rl_67m/run_stage3.py read_rows), each 64 prompt tokens with the
    continuation's first token dropped, read at the last position: the logits E222 read at position 63."""
    rl3 = load_experiment("rl_67m", "run_stage3")
    c = config_from_settings(rl3.Config, json.loads(rl3.PRED.read_text())["settings"])
    r, info = rl3.read_rows(rl3.S2.Ctx(), c, cfg.rl_behaviour)
    assert r.pos == r.toks.shape[1] - 2 and info["n_rows"] == c.n_read, (r.pos, info)
    cfg.rl_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"toks": r.toks[:, :-1].cpu(), "target": r.tgt.cpu(), "info": info, "source": "E222 read_rows"}, cfg.rl_file)
    log(f"rl windows: {info}, {tuple(r.toks.shape)} -> {cfg.rl_file}")


# ---- one item -------------------------------------------------------------------------------------------------------


def k_star(w: World, item: str, compute: bool) -> tuple[np.ndarray | None, str]:
    """The saved K* (all of D, polished), else E226's read_item here when `compute`, with E226's filed settings."""
    got = KV.saved_k_star(item, (RESULTS / "large",))
    if got is not None:
        return got[0], str(got[1].relative_to(REPO))
    if not compute:
        return None, "no saved K*"
    E226.RESULTS, E226.EXPERIMENT = RESULTS, EXPERIMENT  # E226's read_item writes results/large/kstar_<item>.npz here
    c226, th226 = E226.load_config()
    rec = E226.read_item(w, item, c226, th226, RESULTS / "kstar")
    got = KV.saved_k_star(item, (RESULTS / "large",)) if rec.get("reading") == "read" else None
    return (got[0], str(got[1].relative_to(REPO))) if got else (None, f"E226 read_item here: {rec.get('reading')}")


def write_pass(w: World, item: str, bf16: bool = False) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """num, den, dd (fp64, every position of D in F's own run), changed and grafts (bool), and the item's facts."""
    F, M, toks, at, rec = w.make(item)
    feat = KV.featurize(M, F, w.uv, w.delta, w.bases)
    with torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else nullcontext():
        num, den, dd = SI.write_change(F.net, SI.vpd_sites(feat), feat.n, list(toks.split(E242.load_config()[0].gain_batch)))
    d2 = sum(float((F.net.get_submodule(k).weight.double() - M.net.get_submodule(k).weight.double()).pow(2).sum()) for k in w.keys)
    w2 = sum(float(M.net.get_submodule(k).weight.double().pow(2).sum()) for k in w.keys)
    idx = torch.arange(feat.n)
    arrays = {"num": num, "den": den, "dd": dd, "changed": torch.isin(idx, torch.tensor(sorted(feat.changed), dtype=torch.long)),
              "grafts": torch.isin(idx, torch.tensor(sorted(feat.lacked), dtype=torch.long))}  # fmt: skip
    facts = {"n_windows": len(toks), "read_positions": "last" if at == KP.LAST else "all", "n_vars": feat.n, "n_changed": len(feat.changed),
             "dw_rel_norm": (d2 / w2) ** 0.5} | rec  # fmt: skip
    del F
    torch.cuda.empty_cache()
    return arrays, facts


def readings(a: dict[str, Tensor], K: np.ndarray) -> dict[str, Any]:
    Kt = torch.from_numpy(K)
    chK = a["changed"] & Kt
    return {"E_b": SI.explained_share(a["num"], a["den"], a["dd"], chK), "size_K": int(Kt.sum()), "n_changed_in_K": int(chK.sum()),
            "untouched_share": int((Kt & ~a["changed"]).sum()) / int(Kt.sum()), "n_grafts_in_K": int((Kt & a["grafts"]).sum())}  # fmt: skip


def e247_saved(item: str) -> float | None:
    """E_b from E247's saved num, den, dd (experiments/thin_edit_diagnosis/results/large), E242's changed set, saved K*."""
    p = S247 / "large" / f"items_{fname(item)}.npz"
    if not p.is_file():
        return None
    e, ch = np.load(p), np.load(S242 / "large" / f"items_{fname(item)}.npz")["changed"].astype(bool)
    got = KV.saved_k_star(item)
    assert got is not None, item
    num, den, dd = (torch.from_numpy(e[k]) for k in ("num", "den", "dd"))
    return SI.explained_share(num, den, dd, torch.from_numpy(ch & got[0]))


def r_a_saved(item: str) -> float | None:
    p = S242 / "items" / f"{fname(item)}.json"
    return json.loads(p.read_text())["searches"]["main"].get("R_A") if p.is_file() else None


def read_item(w: World, item: str, stage: str) -> dict[str, Any]:
    path = RESULTS / "items" / f"{fname(item)}.json"
    if path.is_file():
        log(f"{item}: done before, skipped")
        return json.loads(path.read_text())
    t0 = time.time()
    K, src = k_star(w, item, stage in ("rl", "manifest"))
    out: dict[str, Any] = {"experiment": EXPERIMENT, "item": item, "stage": stage, "kstar": src, "R_A_saved": r_a_saved(item), "E_b_e247": e247_saved(item)}
    if K is None:
        dump(path, out | {"reading": "no K*"})
        log(f"{item}: no K* ({src}), not read")
        return out
    a, facts = write_pass(w, item)
    out |= facts | {"reading": "read"} | readings(a, K)
    arrays: dict[str, Any] = {k: v.numpy() for k, v in a.items()} | {"in_K": K}
    np.savez_compressed(RESULTS / "large" / f"items_{fname(item)}.npz", **arrays)
    out |= {"seconds": time.time() - t0, "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30}
    dump(path, out)
    log(f"{item}: E_b {out['E_b']:.6f}, untouched share {out['untouched_share']:.4f}, |K*| {out['size_K']} ({out['n_changed_in_K']} changed), "
        f"||dW||/||W0|| {out['dw_rel_norm']:.4g}, R_A saved {out['R_A_saved']}, E247's E_b {out['E_b_e247']}; {out['seconds']:.0f} s")  # fmt: skip
    return out


# ---- stage 0, main --------------------------------------------------------------------------------------------------


def stage0(w: World, cfg: Config, tag: str) -> None:
    path = RESULTS / f"stage0_{tag}.json"
    if path.is_file() and json.loads(path.read_text()).get("done"):
        log(f"{path.name}: done before")
        return
    t0 = time.time()
    dn, _ = write_pass(w, "do_nothing")
    Kdn = KV.saved_k_star("do_nothing")
    assert Kdn is not None
    out: dict[str, Any] = {"experiment": EXPERIMENT, "dn_n_changed": int(dn["changed"].sum()), "dn_dd_max": float(dn["dd"].abs().max()),
                           "dn_E_b": readings(dn, Kdn[0])["E_b"]}  # fmt: skip
    t1 = time.time()
    a32, facts = write_pass(w, cfg.control)
    out["seconds_per_pass"] = time.time() - t1
    a16, _ = write_pass(w, cfg.control, bf16=True)
    K = KV.saved_k_star(cfg.control)
    assert K is not None
    e32, e16, saved = readings(a32, K[0])["E_b"], readings(a16, K[0])["E_b"], e247_saved(cfg.control)
    assert saved is not None
    ch242 = np.load(S242 / "large" / f"items_{fname(cfg.control)}.npz")["changed"].astype(bool)
    out |= {"control": cfg.control, "E_b_fp32": e32, "E_b_bf16": e16, "floor": abs(e16 - e32), "E_b_saved": saved, "dev": abs(e32 - saved),
            "same_changed": bool(np.array_equal(a32["changed"].numpy(), ch242)), "n_vars": facts["n_vars"], "seconds": time.time() - t0}  # fmt: skip
    miss = [f"do-nothing: {k} {out[k]}" for k, want in (("dn_n_changed", 0), ("dn_dd_max", 0.0), ("dn_E_b", 1.0)) if out[k] != want]
    miss += [f"{cfg.control}: E_b {e32:.12g} off E247's saved {saved:.12g} by {out['dev']:.3g}, over the bf16/fp32 floor {out['floor']:.3g}"] if out["dev"] > out["floor"] else []
    miss += [f"{cfg.control}: the changed set differs from E242's"] if not out["same_changed"] else []
    log(f"stage 0: {json.dumps(out)}; {'checks hold' if not miss else 'MISSES ' + '; '.join(miss)}")
    if miss:
        stop(path, out, "stage 0: " + "; ".join(miss))
    dump(path, out | {"done": True})
    log("STAGE 0 DONE")


def stage_items(w: World, cfg: Config, stage: str) -> list[str]:
    c242 = E242.load_config()[0]
    every = {"random": lambda: list(cfg.randoms), "rl": lambda: list(w.rl), "manifest": lambda: list(w.extra),
             "items": lambda: list(dict.fromkeys(i for g in cfg.groups for i in E242.group_items(c242, g)))}  # fmt: skip
    return every[stage]()


def main(stage: str, shard: tuple[int, int], tag: str) -> int:
    cfg = load_config()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 throughout, as E226, E242, E247
    log(f"{EXPERIMENT} start, stage {stage}, shard {shard}, tag {tag}; settings {asdict(cfg)}")
    if stage == "rl_windows":
        rl_windows(cfg)
        return 0
    (RESULTS / "items").mkdir(parents=True, exist_ok=True)
    (RESULTS / "large").mkdir(parents=True, exist_ok=True)
    w = World(cfg)
    w.bases = KV.bases(w.uv, E226.load_config()[0].split_tol)
    r = json.loads((S242 / "items" / f"{fname(E242.load_config()[0].random_kl)}.json").read_text())  # E242's matched scale, as E247
    w.kl_x = {k: r[k] for k in ("x", "kl_matched", "match", "match_evals")}
    stage0(w, cfg, tag)
    if stage != "random" and (missing := [i for i in cfg.randoms if not (RESULTS / "items" / f"{fname(i)}.json").is_file()]):
        log(f"STOP: stage {stage} waits for the random updates {missing}")
        return 1
    every = stage_items(w, cfg, stage)
    todo = every[shard[0] :: shard[1]]
    log(f"stage {stage}: {len(todo)} of {len(every)} items {todo}")
    for k, item in enumerate(todo):
        read_item(w, item, stage)
        log(f"ITEM {k + 1} of {len(todo)} DONE ({item})")
    log(f"{EXPERIMENT} DONE")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("rl_windows", *STAGES), required=True)
    ap.add_argument("--shard", nargs=2, type=int, default=(0, 1), metavar=("I", "N"), help="every N-th item from the I-th (0-based)")
    ap.add_argument("--tag", default="local", help="this job's name: stage 0 writes results/stage0_<tag>.json")
    args = ap.parse_args()
    require_filed(PRED)
    code = 1
    try:
        code = main(args.stage, (args.shard[0], args.shard[1]), args.tag)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)
