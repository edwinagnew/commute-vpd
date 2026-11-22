"""E263 (cell P1.52; user 2026-11-22, option (b)): E_b, the order-free write share, at E234's base-only SAE latents on the
saved E245/E251 items: the definition's featurization-independence check beside E_b at VPD (E257, experiments/rb_states).
Settings and filed expected values in results/predictions.json, which the run refuses unless committed unchanged.

    E_b = sum over K*'s changed members u of (num_u - den_u)^2 / den_u  /  sum over the same members of dd_u
(commute.stand_in.explained_share), num, den, dd from one commute.stand_in.write_change pass through commute.sae_feat's
sites (every position of D in F's own run): the share of the changed members' write change that M's own write at each
member's own least-squares gain explains. K* is E245's or E251's saved search (all of D, polished:
results/large/items_<item>.npz `main_in_K_polished`), unchanged. Items, F, M, D and the SAEs are E245's and E251's,
built by their code unchanged; the KL-matched random takes E251's saved scale (not matched again).

Per job: the do-nothing first and alone (no changed site, every dd exactly 0, E_b exactly 1; any miss stops the job),
then its shard of the items in group order (built, random, pairs, keys). Per item, before reading: the variable count
and the changed set must equal the saved ones exactly (else the job stops: K* would index other variables); the gains
num / den beside E245's saved gains are reported (max relative deviation), not a stop. results/items/<item>.json per
item (a rerun skips it), results/large/items_<item>.npz num, den, dd.

    sbatch --comment="E263 eb0" cluster/sae_eb.sbatch eb0 --shard 0 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from commute import sae_feat as SF
from commute import stand_in as SI
from commute.jobs import EXPERIMENTS, hard_exit, load_experiment, require_filed

E251: Any = load_experiment("sae_r_a_random", "run")  # World, items(); imports E245 and points its writes at E251
E245, E242 = E251.E245, E251.E242
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED = RESULTS / "predictions.json"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
SAVED = {"e245": EXPERIMENTS / "sae_r_a" / "results", "e251": E251.RESULTS}
log, dump, stop, fname = E242.log, E242.dump, E242.stop, E242.fname


def items(cfg: Any) -> list[tuple[str, str]]:
    """(group, item) in reading order; the source of each item's saved K* follows from its group."""
    return ([("built", i) for i in E245.group_items(cfg, "built")] + [("random", i) for i in E251.items()]
            + [(g, i) for g in ("pairs", "keys") for i in E245.group_items(cfg, g)])  # fmt: skip


def saved(group: str, item: str) -> tuple[np.lib.npyio.NpzFile, dict[str, Any]]:
    src = SAVED["e251" if group == "random" else "e245"]
    if item == "do_nothing":
        return np.load(src / "large" / "stage0_built0_do_nothing.npz"), json.loads((src / "items" / "stage0_built0" / "do_nothing.json").read_text())
    return np.load(src / "large" / f"items_{fname(item)}.npz"), json.loads((src / "items" / f"{fname(item)}.json").read_text())


def read(w: Any, saes: dict[str, Any], cfg: Any, group: str, item: str) -> dict[str, Any]:
    t0 = time.time()
    F, M, toks, _, rec = w.make(item)
    for N in (F, M):
        N.net.requires_grad_(False)
        for e in N.extra.values():
            e.requires_grad_(False)
    ss = SF.sites(F, M, saes, toks.shape[1])
    n = SF.n_vars(ss)
    duck: list[Any] = list(ss)  # stand_in.write_change is typed for VPDSite; SAESite has the same members
    num, den, dd = SI.write_change(F.net, duck, n, list(toks.split(cfg.gain_batch)))
    z, old = saved(group, item)
    changed = torch.zeros(n, dtype=torch.bool)
    changed[sorted(SF.changed_vars(ss))] = True
    if n != len(z["changed"]) or not np.array_equal(changed.numpy(), z["changed"].astype(bool)):
        stop(RESULTS / "items" / f"{fname(item)}.stopped.json", {"item": item, "n_vars": n}, f"{item}: variables or changed set differ from the saved ones")
    K = torch.from_numpy(z["main_in_K_polished"].astype(bool))
    g = torch.where(den > 0, num / den.clamp(min=1e-300), 1.0)
    chK = changed & K
    out = {"experiment": EXPERIMENT, "group": group, "item": item, "reading": "read", "n_vars": n, "n_windows": len(toks),
           "changed_sites": [s.name for s in ss if s.changed], "E_b": SI.explained_share(num, den, dd, chK), "size_K": int(K.sum()),
           "n_changed_in_K": int(chK.sum()), "untouched_share": int((K & ~changed).sum()) / max(1, int(K.sum())),
           "remainders_in_changed_K": int(sum(bool(chK[s.rem]) for s in ss)),
           "remainder_dd_share_in_changed_K": float(sum(dd[s.rem] for s in ss if chK[s.rem]) / dd[chK].sum()) if float(dd[chK].sum()) > 0 else None,
           "gain_rel_dev_saved": float(((g - torch.from_numpy(z["gain"])).abs() / torch.from_numpy(z["gain"]).abs().clamp(min=1e-12)).max()),
           "dd_max": float(dd.max()), "R_A_saved": old["searches"]["main"]["R_A"], "remainder_share_saved": old.get("remainder_share"),
           "seconds": time.time() - t0} | {k: rec[k] for k in ("x", "kl_matched") if k in rec}  # fmt: skip
    np.savez_compressed(RESULTS / "large" / f"items_{fname(item)}.npz", num=num.numpy(), den=den.numpy(), dd=dd.numpy())
    del F
    torch.cuda.empty_cache()
    return out


def stage0(w: Any, saes: dict[str, Any], cfg: Any, tag: str) -> None:
    path = RESULTS / f"stage0_{tag}.json"
    if path.is_file() and json.loads(path.read_text()).get("done"):
        log(f"{path.name}: done before")
        return
    r = read(w, saes, cfg, "built", "do_nothing")
    miss = [f"do-nothing: {k} {r[k]}" for k, want in (("changed_sites", []), ("dd_max", 0.0), ("E_b", 1.0)) if r[k] != want]
    log(f"stage 0: {json.dumps(r)}; {'checks hold' if not miss else 'MISSES ' + '; '.join(miss)}")
    if miss:
        stop(path, r, "stage 0: " + "; ".join(miss))
    dump(path, r | {"done": True})
    log("STAGE 0 DONE")


def main(shard: tuple[int, int], tag: str) -> int:
    pred = json.loads(PRED.read_text())
    for k, p in (("e245", E245.PRED), ("e251", E251.PRED)):
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        assert got == pred[f"{k}_predictions_sha256"], f"{p}: sha256 {got} != filed: its settings changed"
    cfg, _ = E245.load_config("scored")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 throughout, as E245
    (RESULTS / "items").mkdir(parents=True, exist_ok=True)
    (RESULTS / "large").mkdir(parents=True, exist_ok=True)
    every = items(cfg)
    todo = every[shard[0] :: shard[1]]
    log(f"{EXPERIMENT} start, tag {tag}; {len(todo)} of {len(every)} items; gain_batch {cfg.gain_batch}, SAEs {cfg.sae_size} seed {cfg.sae_seed}")
    w, saes = E251.World(E242.load_config()[0]), E245.load_saes(cfg)
    rkl = json.loads((SAVED["e251"] / "items" / "random_kl_241002.json").read_text())
    w.kl_x = {k: rkl[k] for k in ("x", "kl_matched", "match", "match_evals")}  # E251's matched scale, not matched again
    stage0(w, saes, cfg, tag)
    for k, (group, item) in enumerate(todo):
        path = RESULTS / "items" / f"{fname(item)}.json"
        if path.is_file():
            log(f"{item}: done before, skipped")
            continue
        r = read(w, saes, cfg, group, item)
        dump(path, r)
        log(f"ITEM {k + 1} of {len(todo)} DONE ({group} {item}): E_b {r['E_b']:.6f}, untouched {r['untouched_share']:.4f}, |K*| {r['size_K']} "
            f"({r['n_changed_in_K']} changed, {r['remainders_in_changed_K']} remainders carrying {r['remainder_dd_share_in_changed_K']} of dd), "
            f"R_A saved {r['R_A_saved']}, gain dev {r['gain_rel_dev_saved']:.3g}; {r['seconds']:.0f} s")  # fmt: skip
    log(f"FINISHED {EXPERIMENT} {tag}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", nargs=2, type=int, default=(0, 1), metavar=("I", "N"), help="every N-th item from the I-th (0-based)")
    ap.add_argument("--tag", default="local", help="this job's name: stage 0 writes results/stage0_<tag>.json")
    args = ap.parse_args()
    require_filed(PRED)
    code = 1
    try:
        code = main((args.shard[0], args.shard[1]), args.tag)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)
