"""E283 (cell P1.57; user via whip-lead, 2026-11-23 ~14:20): M's own needed set K_M on every D that an F scan of the
kept-clause diagnosis read, for the reliance readings V1 and V2 (score.py, CPU). Descriptive, no bar; neither reading is
the definition. Settings and the filed rules in results/predictions.json, which the run refuses unless committed
unchanged. Card: card.md.

K_M is the F scans' needed set with M in place of F: M's own run on D cached, M's own argmax at the read positions, the
same pairs (`commute.needed.derangements` of |D|, with the F scans' P, examples and seed), the same scan code
(`needed_neurons.scan` at neurons and heads with E277's settings and E229's bounds; `needed.needed_set` at VPD with
E268's settings). It is not the stage-0 identity check (x' = x), which gives no flips. Every D is keyed by the sha256
of its tokens, shape and read positions; an item whose D was scanned before is only recorded (results/dmap/), so a
shared D is scanned once:
    neurons --world built   E277's seven built edits (x2's D: copy24 held-out, all positions; new_few's: cue windows, last)
    neurons --world e62     E277's nine E62 items (E257's e62 read half; E280's D is e62/copy_s0's)
    neurons --world panel   E277's panel (E272's windows per task; --tasks picks tasks, one D each)
    vpd                     E275's nine E62 items at VPD, and E281's copy windows of each one's D (labels e281/<item>)
    fcue                    amendment 1 (results/predictions_a1.json): x2 and x10 as F on new_few's D (the cue windows),
                            E277's read, results/large/fcue_<item>.npz and results/fcue/<item>.json, and M's K there
Outputs: results/large/mK_<feat>_<key>.npz (in_K, first_pair, flips of M), results/mK/<feat>_<key>.json (sizes, time),
results/dmap/<feat>__<item>.json (the item's D key). A scan that finds no needed variable of M stops the job.

    python experiments/reliance/run.py --stage neurons --world built --items x2 x10 rebuilt random_size_241001 --tag a1
    python experiments/reliance/run.py --stage neurons --world panel --tasks squad --tag p1
    python experiments/reliance/run.py --stage vpd --tag v1
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
from torch import Tensor

from commute import needed as IX
from commute import needed_neurons as NN
from commute.jobs import (
    EXPERIMENTS,
    hard_exit,
    load_experiment,
    require_filed,
    stop,
    write_stage,
)

NNR: Any = load_experiment("needed_neurons", "run")  # E277: Built, load_config, load_e229, load_a1, E229, rows_for, log
KN: Any = load_experiment("kstar_needed", "run")  # E268: world, featurized, load_config (VPD scan settings), fname
HE: Any = load_experiment("head_edit", "run")  # E280: targets (E281's copy windows)
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED = RESULTS / "predictions.json"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
SCORE: Any = load_experiment("reliance", "score")  # Config (the filed settings; the scan reads none of them)
log = NNR.log


def dkey(toks: Tensor, at: slice) -> str:
    h = hashlib.sha256(toks.detach().cpu().contiguous().numpy().tobytes())
    h.update(repr((tuple(toks.shape), str(toks.dtype), at.start, at.stop, at.step)).encode())
    return h.hexdigest()[:16]


def record(feat: str, item: str, key: str, toks: Tensor) -> bool:
    """Writes the item's D key; True when M was already scanned on that D at this featurization."""
    write_stage(RESULTS / "dmap" / f"{feat}__{item.replace('/', '__')}.json",
                {"experiment": EXPERIMENT, "item": item, "feat": feat, "key": key, "n_windows": len(toks), "window_len": toks.shape[1]})  # fmt: skip
    return (RESULTS / "mK" / f"{feat}_{key}.json").is_file()


def save(feat: str, key: str, item: str, toks: Tensor, n_at: int, sc: IX.Scan, t0: float) -> None:
    K = sc.first_pair >= 0
    out = {"experiment": EXPERIMENT, "feat": feat, "key": key, "first_item": item, "n_windows": len(toks), "window_len": toks.shape[1],
           "n_read_positions": n_at, "n_vars": len(K), "size_K_M": int(K.sum()), "flip_mass_M": int(sc.flips.sum()), "max_dev": sc.max_dev,
           "cases": sc.rows, "scan_seconds": sc.seconds, "cases_per_s": sc.rows / sc.seconds, "seconds": time.time() - t0,
           "card": torch.cuda.get_device_name(), "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30}  # fmt: skip
    path = RESULTS / "mK" / f"{feat}_{key}.json"
    if not K.any():
        stop(path, out, f"{feat} {item}: M's needed set is empty on this D (a broken scan)", log)
    np.savez_compressed(RESULTS / "large" / f"mK_{feat}_{key}.npz", in_K=K.numpy(), first_pair=sc.first_pair.numpy(), flips=sc.flips.numpy())
    write_stage(path, out)
    log(f"{feat} {item}: D {key}, |K_M| {out['size_K_M']} of {out['n_vars']}, {sc.rows} cases in {sc.seconds:.0f} s; SCAN DONE")


def neuron_scan(net: Any, feat: Any, cfg: Any, toks: Tensor, at: slice) -> tuple[IX.Scan, int]:
    """E277's scan (its pairs and batch sizes) of `net` on D; returns the scan and the number of read positions."""
    run = NN.cache(net, toks, at, cfg.cache_batch)
    m = min(cfg.examples, len(toks))
    ex, pt = IX.derangements(len(toks), cfg.derangements, m, cfg.seed)
    n_at = run.top.shape[1]
    sc = NN.scan(net, feat, run, at, ex, pt, NNR.rows_for(cfg, n_at), log)
    del run
    torch.cuda.empty_cache()
    return sc, n_at


def m_scan(feat: Any, cfg: Any, item: str, M: Any, toks: Tensor, at: slice) -> None:
    """M's K on the item's D at neurons, unless M was scanned on that D before (the D key is recorded either way)."""
    t0, key = time.time(), dkey(toks, at)
    if record("neurons", item, key, toks):
        log(f"{item}: D {key} scanned before")
        return
    torch.cuda.reset_peak_memory_stats()
    log(f"{item}: M on D {key}, {len(toks)} windows of {toks.shape[1]} tokens")
    sc, n_at = neuron_scan(M.net, feat, cfg, toks, at)
    save("neurons", key, item, toks, n_at, sc, t0)


def neuron_setup() -> tuple[Any, Any]:
    cfg, c229 = NNR.load_config(), NNR.load_e229()
    return cfg, NNR.E229.feat_of(c229, NNR.E229.load_pct(c229), cfg.band)


def neurons(world_name: str, items: list[str], tasks: list[str]) -> None:
    cfg, feat = neuron_setup()
    world = NNR.Built(cfg, world_name)
    todo = [i for i in world.items if (not items or i in items) and (not tasks or world.panel.get(i, {}).get("task") in tasks)]
    assert todo and (not items or len(todo) == len(items)), f"items {items} tasks {tasks} pick {todo} of {world.items}"
    log(f"neurons, world {world_name}: {len(todo)} items {todo}")
    for item in todo:
        F, M, toks, at, _ = world.make(item)
        del F
        m_scan(feat, cfg, item, M, toks, at)


def fcue(items: list[str], on: str) -> None:
    """Amendment 1's same-D control: each built edit in `items` (x2, x10) as F on the D of `on` (new_few: the cue
    windows, last position), read as E277's read_item reads an item (its write pass and scan, its pairs), saved as
    results/large/fcue_<item>.npz with E277's keys, plus M's K on that D (recorded as control_cue/<item>)."""
    cfg, feat = neuron_setup()
    world = NNR.Built(cfg, "built")
    _, M, toks, at, _ = world.make(on)
    for item in items or ["x2", "x10"]:
        t0 = time.time()
        F, _, _, _, _ = world.make(item)  # the World's one M
        m_scan(feat, cfg, f"control_cue/{item}", M, toks, at)
        num, den, dd, ch, _ = NNR.write_pass(F, M, toks, feat, cfg.cache_batch)
        log(f"{item} on {on}'s D: {int(ch.sum())} changed; F scan")
        sc, n_at = neuron_scan(F.net, feat, cfg, toks, at)
        np.savez_compressed(RESULTS / "large" / f"fcue_{item}.npz", in_K=(sc.first_pair >= 0).numpy(), first_pair=sc.first_pair.numpy(),
                            flips=sc.flips.numpy(), num=num.numpy(), den=den.numpy(), dd=dd.numpy(), changed=ch.numpy())  # fmt: skip
        write_stage(RESULTS / "fcue" / f"{item}.json", {"experiment": EXPERIMENT, "item": item, "on": on, "n_windows": len(toks), "n_read_positions": n_at,
                    "size_K": int((sc.first_pair >= 0).sum()), "n_changed": int(ch.sum()), "cases_per_s": sc.rows / sc.seconds, "seconds": time.time() - t0})  # fmt: skip
        log(f"{item} on {on}'s D: |K_F| {int((sc.first_pair >= 0).sum())}, {time.time() - t0:.0f} s; FCUE DONE")
        del F


def vpd(items: list[str]) -> None:
    kc, w = KN.load_config(), KN.world()
    _, featM, _, _ = KN.featurized(w, "do_nothing")  # M's featurization: weights only, so any D
    assert not featM.changed and not featM.grafts, "the do-nothing featurization of M changes nothing"
    todo = [i for i in NNR.load_a1().e62 if not items or i in items]
    assert todo and (not items or len(todo) == len(items)), f"items {items} pick {todo}"
    for item in todo:
        F, M, toks, at, _ = w.make(item)
        del F
        saved = np.load(EXPERIMENTS / "claim2_needed" / "results" / "large" / f"items_{KN.fname(item)}.npz")["in_K"]
        assert len(saved) == featM.n, f"{item}: E275's K has {len(saved)} variables, M's featurization {featM.n}"
        copy, _ = HE.targets(toks)
        for label, sub in ((item, toks), (f"e281/{item}", toks[copy.to(toks.device)])):
            key = dkey(sub, at)
            if record("vpd", label, key, sub):
                log(f"{label}: D {key} scanned before")
                continue
            t0 = time.time()
            torch.cuda.reset_peak_memory_stats()
            log(f"{label}: M on D {key}, {len(sub)} windows of {sub.shape[1]} tokens")
            nd = IX.needed_set(M, featM, sub, at, kc.derangements, kc.examples, kc.seed, kc.rows, kc.logit_rows, kc.cache_batch, log)
            save("vpd", key, label, sub, nd.run.top.shape[1], nd.scan, t0)
            del nd
            torch.cuda.empty_cache()


def main(args: argparse.Namespace) -> int:
    SCORE.load_config()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32, as the F scans
    log(f"{EXPERIMENT} start, stage {args.stage}, world {args.world}, items {args.items}, tasks {args.tasks}, tag {args.tag}")
    for d in ("large", "mK", "dmap", "fcue"):
        (RESULTS / d).mkdir(parents=True, exist_ok=True)
    if args.stage == "neurons":
        neurons(args.world, args.items, args.tasks)
    elif args.stage == "fcue":
        fcue(args.items, "new_few")
    else:
        vpd(args.items)
    log(f"{EXPERIMENT} DONE")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("neurons", "vpd", "fcue"), required=True)
    ap.add_argument("--world", choices=("built", "e62", "panel"), default="built")
    ap.add_argument("--items", nargs="*", default=[])
    ap.add_argument("--tasks", nargs="*", default=[])
    ap.add_argument("--tag", default="local")
    a = ap.parse_args()
    require_filed(PRED)
    if a.stage == "fcue":
        require_filed(RESULTS / "predictions_a1.json")
    code = 1
    try:
        code = main(a)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)
