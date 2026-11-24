"""Stage 0 of cell P2.68, alone and first (spec §4 and Appendix D): every task's rows, the hook floor per format, the gate
per task and scale, the do-nothing, the two-layer path, the planted head (value rows) through the whole pipeline, the
output-weights blind spot, and one timed key item with the price estimate. Writes results/stage0.json as it goes and
exits 1 if any control is missed."""

from __future__ import annotations

import dataclasses
import time
from functools import partial
from types import SimpleNamespace
from typing import Any

import torch
from job import (
    RESULTS,
    STAGE0,
    H,
    Item,
    Job,
    L,
    cmap_sets,
    dump,
    gate_key,
    item_path,
    log,
    read_json,
    run_item,
    run_listed,
    stop,
)
from torch import Tensor

from commute import finetune as FT
from commute import head_cmap as HC
from commute import old_tasks as OT
from commute.jobs import EXPERIMENTS
from commute.prakash import o_proj, proj, quiet
from commute.vpd import target_linear

ALL = [(l, h, r) for l in range(L) for h in range(H) for r in (0, 1)]  # every head at the read and the position before


def hook_floor(job: Job, task: str) -> dict[str, Any]:
    """W0 given its own head outputs at both positions through `cmap_out`'s einsum, against a plain W0 forward, on D."""
    t, w0 = job.task(task), job.w0_read(task)
    logits, sites = HC.cmap(job.au, job.a0, HC.o_inputs(job.net.net, t.D.clean, job.cfg.batch), t.D.clean, ALL, job.cfg.batch)
    c = HC.compare(HC.Read(logits, t.D.labels), w0)
    return {"task": task, "sites": sites, **c, "tol": {"logit": c["max_dlogit"], "logodds": c["max_dlogodds"]}}


def gates(job: Job) -> dict[str, Any]:
    """Per task and scale: n_gate random directions at the per-matrix norms of the task's states (cycled), seed
    gate_seed + j, times c, each read on D (`finetune.random_direction`, E237's construction)."""
    by: dict[str, list[Item]] = {}
    for it in job.items.values():
        by.setdefault(gate_key(it.task, it.c), []).append(it)
    out = {}
    for key, its in by.items():
        states = sorted({it.state: it for it in its}.values(), key=lambda i: i.state)
        dws = {it.state: OT.load_dw(job.net, EXPERIMENTS.parent / it.file) for it in states}
        t, c, reads = job.task(its[0].task), its[0].c, []
        for j in range(job.cfg.n_gate):
            d = FT.random_direction(dws[states[j % len(states)].state], job.cfg.gate_seed + j)
            reads.append(HC.forward_read(OT.with_delta(job.net, {k: v * c for k, v in d.items()}).net, t.D, job.cfg.batch))
        out[key] = HC.gate(job.w0_read(t.name), reads) | {"states": [s.state for s in states], "c": c, "n_D": len(t.D)}
        log(f"[gate {key}] N {out[key]['N']}, L {out[key]['L']:.4f} over {len(reads)} random updates on {len(t.D)} rows")
    return out


def e237_randoms(job: Job) -> dict[str, Any]:
    """E237's own random_<state> items (seed e237_random_seed + the state's index, times E237's c), read on D beside the gate."""
    c = read_json("experiments/key_grid/results/stage0.json")["c"]
    states = [f"e58/{m}_s{s}" for m in job.cfg.key_tasks for s in job.cfg.key_seeds]
    out = {}
    for i, st in enumerate(states):
        m = st.removeprefix("e58/").rsplit("_s", 1)[0]
        d = FT.random_direction(OT.load_dw(job.net, EXPERIMENTS / "refit_96key" / "results" / "large" / f"dW_{m}_s{st[-1]}.pt"),
                                job.cfg.e237_random_seed + i)  # fmt: skip
        t = job.task(f"e58/{m}")
        r = HC.forward_read(OT.with_delta(job.net, {k: v * c[m] for k, v in d.items()}).net, t.D, job.cfg.batch)
        out[f"random_{st}"] = HC.gains(r, job.w0_read(t.name))
    return out


def _sites_ok(rec: dict[str, Any], tol: dict[str, float], vs: str, names: list[str]) -> list[str]:
    return [f"{rec['item']} {k}: {rec['cmap'][k]['sites']} sites, {rec['cmap'][k][vs]} against tol {tol}"
            for k in names if not ((rec["cmap"][k]["sites"] > 0 or not rec["cmap"][k]["nodes"]) and HC.within(rec["cmap"][k][vs], tol))]  # fmt: skip


def two_layer(job: Job) -> dict[str, Any]:
    """Appendix D's two-layer path on W0, e58/copy's discovery rows: a seeded delta on block-0 head s at the marker,
    traced through receiver r's v_proj output and both receiver edits, each step against the same passes with delta 0."""
    cfg, au, a0 = job.cfg, job.au, job.a0
    t = job.task(f"e58/{cfg.key_tasks[0]}")
    p = HC.Prompts(t.disc.clean, t.disc.labels, t.disc.clean)
    T, dh = p.clean.shape[1], 768 // H
    att = HC.attention_to(job.net.net, 1, p.clean, cfg.batch)
    r, gen = int(att.argmax()), torch.Generator().manual_seed(cfg.two_layer_seed)
    s = int(torch.randint(0, H, (1,), generator=gen))
    dl = HC.loader(p, cfg.batch)
    with quiet():
        clean, _, _, _, hooks = au.pp.get_caches(a0, dl)
    sl, rs = slice(s * dh, (s + 1) * dh), slice(r * dh, (r + 1) * dh)
    norm = float(torch.cat([clean[b][o_proj(0)].input[:, T - 2, sl] for b in clean]).norm(dim=-1).mean())
    delta = torch.randn(dh, generator=gen)
    delta = delta / delta.norm() * norm
    vname = proj(1, "v")

    def run(d: Tensor | None) -> dict[str, Tensor]:
        got: dict[str, list[Tensor]] = {}
        add = lambda k, v: got.setdefault(k, []).append(v.detach().float().cpu())
        for bi, inp in enumerate(dl):
            x, last = inp["base_tokens"].cuda(), inp["base_last_token_indices"].cuda()
            if d is None:  # the plain clean forward
                with au.pp.TraceDict(a0, [vname, o_proj(1)], retain_input=True) as tr:
                    out = a0(x).logits[:, -1]
                for k in ("v_marker", "v_read"):
                    add(k, tr[vname].output[:, T - 2 if k == "v_marker" else T - 1, rs])
                for k in ("o_read_rel1", "o_read_rel0"):
                    add(k, tr[o_proj(1)].input[:, T - 1, rs])
                add("logits_rel0", out)
                continue
            sender = {k: SimpleNamespace(input=v.input.clone()) for k, v in clean[bi].items()}
            sender[o_proj(0)].input[:, T - 2, sl] += d
            kw = {"model": a0, "base_tokens": x, "clean_last_token_indices": last, "batch_size": len(x)}
            with au.pp.TraceDict(a0, hooks + [vname], retain_input=True, edit_output=partial(au.pp.patching_sender_heads, clean_cache=clean[bi],
                                 corrupt_cache=sender, sender_layer=0, sender_head=s, corrupt_last_token_indices=last, rel_pos=1, **kw)) as pc:  # fmt: skip
                a0(x)
            add("v_marker", pc[vname].output[:, T - 2, rs])
            add("v_read", pc[vname].output[:, T - 1, rs])
            for rel in (1, 0):
                edit = partial(au.pp.patching_receiver_heads, patched_cache=pc, receiver_heads=[[1, r]], rel_pos=rel, **kw)
                with au.pp.TraceDict(a0, [vname], edit_output=edit), au.pp.TraceDict(a0, [o_proj(1)], retain_input=True, retain_output=False) as ot:
                    out = a0(x).logits[:, -1]
                add(f"o_read_rel{rel}", ot[o_proj(1)].input[:, T - 1, rs])
                if rel == 0:
                    add("logits_rel0", out)
        return {k: torch.cat(v) for k, v in got.items()}

    plain, zero, live = run(None), run(torch.zeros(dh)), run(delta)
    dist = lambda a: {k: float((a[k] - plain[k]).abs().max()) for k in plain}
    lo = lambda g: HC.Read(g["logits_rel0"], p.labels)
    floor, got = dist(zero) | {"logodds_rel0": HC.compare(lo(zero), lo(plain))["max_dlogodds"]}, dist(live)
    cmp = HC.compare(lo(live), lo(plain))
    got |= {"logodds_rel0": cmp["max_dlogodds"], "n_flipped_rel0": cmp["n_flipped"]}
    checks = {"v at the marker changes": got["v_marker"] > floor["v_marker"], "v at the read stays": got["v_read"] <= floor["v_read"],
              "marker edit changes r's read output": got["o_read_rel1"] > floor["o_read_rel1"],
              "read edit leaves r's read output": got["o_read_rel0"] <= floor["o_read_rel0"],
              "read edit leaves the logits": got["logits_rel0"] <= floor["logits_rel0"],
              "read edit leaves the log-odds": got["logodds_rel0"] <= floor["logodds_rel0"],
              "read edit leaves the correct counts": cmp["n_flipped"] == 0}  # fmt: skip
    return {"receiver": [1, r], "attention_to_marker": att.tolist(), "sender": [0, s], "delta_norm": norm, "floor": floor, "got": got, "checks": checks}


def plant(job: Job, matrix: str, floor: dict[str, float], full_gate: bool) -> dict[str, Any]:
    """Block-3 head h*'s value rows (or o_proj columns) trained by E58's recipe with a gradient mask; the first checkpoint
    whose gain clears e58/copy's gate (full_gate False: in the log-odds unit only)."""
    cfg, model = job.cfg, job.net.model
    gen = torch.Generator().manual_seed(cfg.plant_seed)
    h = int(torch.randint(0, H, (1,), generator=gen))
    other = [n for n in [(l, k) for l in range(L) for k in range(H)] if n != (3, h)][int(torch.randint(0, L * H - 1, (1,), generator=gen))]
    lin, dh = target_linear(model, matrix), 768 // H
    mask = torch.zeros_like(lin.weight)
    if matrix.endswith("v_proj"):
        mask[h * dh : (h + 1) * dh] = 1
    else:
        mask[:, h * dh : (h + 1) * dh] = 1
    w = torch.load(EXPERIMENTS.parent / job.task_specs[f"e58/{cfg.key_tasks[0]}"]["path"], weights_only=False)
    a = FT.ArmAConfig(o_token=w["table"], kl_lambda=cfg.plant_kl_lambda, lr=cfg.plant_lr, batch=cfg.plant_batch, steps=cfg.plant_steps,
                      checkpoints=cfg.plant_checkpoints, grad_clip=cfg.plant_grad_clip, seed=cfg.plant_ft_seed, optimizer="adam", matrices=(matrix,))  # fmt: skip
    lin.weight.requires_grad_(True)
    hook = lin.weight.register_hook(lambda g: g * mask)
    t0 = time.time()
    try:
        ft = FT.finetune_arm_a(model, w["train"], w["heldout"], a, "cuda", log)
    finally:
        hook.remove()
        lin.weight.requires_grad_(False)
    t, w0, evals = job.task(f"e58/{cfg.key_tasks[0]}"), job.w0_read(f"e58/{cfg.key_tasks[0]}"), {}
    for step in cfg.plant_checkpoints:
        d = ft.deltas[step][matrix]
        assert float((d * (1 - mask.cpu())).abs().max()) == 0, "the mask let a weight outside h* move"
        g = HC.gains(HC.forward_read(OT.with_delta(job.net, {f"{matrix}.weight": d}).net, t.D, cfg.batch), w0)
        evals[step] = g
        ok = g["logodds"] > max(floor["L"], cfg.gain_floor) and (g["count"] > floor["N"] or not full_gate)
        log(f"[plant {matrix}] checkpoint {step}: gain {g}; clears {ok}")
        if ok:
            return {"head": [3, h], "other": list(other), "step": step, "gains": evals, "dW": {f"{matrix}.weight": d}, "train_s": time.time() - t0}
    return {"head": [3, h], "other": list(other), "step": None, "gains": evals, "train_s": time.time() - t0}


def estimate(job: Job, timed: dict[str, Any]) -> dict[str, Any]:
    """Appendix G: seconds per prompt-pass of each stage on the timed item, times each item's passes and rows (pruning
    at the timed item's number of F evaluations until an item's own discovery gives its group sizes)."""
    s, tr = timed["seconds"], timed["task_rows"]
    n_disc = 24 * (1 + 2 * ("mean" in timed["B"]) + 2 * ("mean" in timed.get("C", {})))
    per = {"discovery": s["discovery"] / (n_disc * tr["n_discovery"]), "prune": s["prune"] / (timed["n_prune_evals"] * tr["n_prune"]),
           "faith": s["faith"] / ((2 + job.cfg.n_random_sets) * tr["D_labelled"]), "cmap": s["cmap"] / (timed["n_cmap_sets"] * tr["D_labelled"])}  # fmt: skip
    items = {}
    for it in job.items.values():
        r = job.task(it.task).record
        items[it.name] = (per["discovery"] * 120 * r["n_discovery"] + per["prune"] * timed["n_prune_evals"] * r["n_prune"]
                          + (per["faith"] * (2 + job.cfg.n_random_sets) + per["cmap"] * (5 + job.cfg.n_random_sets)) * r["D_labelled"])  # fmt: skip
    return {"seconds_per_prompt_pass": per, "items": items, "total_s": sum(items.values()), "n_items": len(items)}


def stage0(job: Job) -> None:
    cfg, miss = job.cfg, list[str]()
    out: dict[str, Any] = {"tasks": {}}
    save = lambda: dump(STAGE0, out | {"misses": miss})
    for it in job.items.values():
        out["tasks"][it.task] = job.task(it.task).record
    nw = sorted({read_json(p)["n_windows"] for p in (EXPERIMENTS / "kstar_vpd_wording_a" / "results" / "items").glob("*.json")})
    out["e248_n_windows"] = nw
    miss += [f"{k}: |D| {r['D_n']} not E248's n_windows {nw}" for k, r in out["tasks"].items() if [r["D_n"]] != nw]
    out["hook_floor"] = {fmt: hook_floor(job, task) for fmt, task in cfg.hook_floor_tasks.items()}
    miss += [f"hook floor {f}: {r}" for f, r in out["hook_floor"].items() if r["n_flipped"] or not r["sites"]]
    tol = {f: r["tol"] for f, r in out["hook_floor"].items()}
    log(f"[hook floor] {out['hook_floor']}")
    save()
    out["gates"] = job.gates = gates(job)
    out["e237_random"] = e237_randoms(job)
    save()
    ctrl = next(it for it in job.items.values() if it.state == cfg.control_state and it.c == 1.0)
    copy, zero = ctrl.task, job.dW(dataclasses.replace(ctrl, c=0.0))  # E237's control state times 0
    floor = job.gates[gate_key(copy, 1.0)]
    dn = run_item(job, "do_nothing", copy, zero, floor, RESULTS / "stage0_do_nothing.json", lambda p: {"all": ALL}, large_ok=True)
    out["do_nothing"] = {k: dn[k] for k in ("W0", "ft", "gain", "faith", "groups_pruned")}
    miss += _sites_ok(dn, tol["kv"], "vs_W0", list(dn["cmap"]))
    save()
    out["two_layer"] = two_layer(job)
    miss += [f"two-layer: {k}" for k, ok in out["two_layer"]["checks"].items() if not ok]
    log(f"[two-layer] {out['two_layer']}")
    save()
    pv = plant(job, "h.3.attn.v_proj", floor, True)
    out["plant_v"] = {k: v for k, v in pv.items() if k != "dW"}
    if pv["step"] is None:
        miss.append("plant (value rows): no checkpoint clears the gate")
    else:
        hn, other = (3, pv["head"][1], 0), (*pv["other"], 0)
        extra = lambda p: {"A_minus_h": [n for n in p["A"] if n != hn], "other_head": [other], "h_before": [(3, hn[1], 1)]}
        rec = run_item(job, "plant_v", copy, pv["dW"], floor, RESULTS / "stage0_plant_v.json", extra, large_ok=True)
        out["plant_v"] |= {"groups_pruned": rec["groups_pruned"], "faith": rec["faith"], "cmap_A": rec["cmap"]["A"]["share"]}
        miss += [] if hn in rec["groups_pruned"]["A"] else [f"plant (value rows): {hn} not in A after pruning: {rec['groups_pruned']}"]
        miss += _sites_ok(rec, tol["kv"], "vs_ft", ["A"]) + _sites_ok(rec, tol["kv"], "vs_W0", ["A_minus_h", "other_head", "h_before"])
        out["timing_plant_v"] = rec["seconds"]
    save()
    po = plant(job, "h.3.attn.o_proj", floor, False)
    out["plant_o"] = {k: v for k, v in po.items() if k != "dW"}
    if po["step"] is None:
        miss.append("plant (output weights): no checkpoint clears the gate in log-odds")
    else:
        t = job.task(copy)
        Fnet = OT.with_delta(job.net, po["dW"]).net
        sets = {"h": [(3, po["head"][1], 0)], "all": ALL}
        rec = {"item": "plant_o", "cmap": cmap_only(job, t, Fnet, sets)}
        out["plant_o"]["cmap"] = rec["cmap"]
        miss += _sites_ok(rec, tol["kv"], "vs_W0", list(sets))
    save()
    out["timed_item"] = ctrl.name
    trec = run_listed(job, ctrl, large_ok=True)
    out["estimate"] = estimate(job, trec)
    log(f"[estimate] {out['estimate']['seconds_per_prompt_pass']}; {out['estimate']['n_items']} items, total {out['estimate']['total_s'] / 3600:.2f} h")
    log(f"stage 0: {'controls hold' if not miss else 'MISSES ' + '; '.join(miss)}")
    if miss:
        stop(STAGE0, out | {"misses": miss}, "stage 0: " + "; ".join(miss))
    dump(STAGE0, out | {"misses": [], "done": True})
    log(f"STAGE 0 DONE ({item_path(ctrl.name).name} written)")


def cmap_only(job: Job, t: Any, Fnet: Any, sets: dict[str, list[HC.Node]]) -> dict[str, Any]:
    w0, ft = job.w0_read(t.name), HC.forward_read(Fnet, t.D, job.cfg.batch)
    return cmap_sets(job, t, Fnet, sets, w0, ft, RESULTS / "stage0_plant_o.json", {"item": "plant_o"})
