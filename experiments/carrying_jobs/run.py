"""carrying_jobs (E101), stage 0: what did E98's carrying neurons do in W0? Plan:
reports/plans/2026-11-05_plan_e101-carrying-jobs.md. Settings, sets and rules: results/predictions.json (filed before the
run); rules and set builders: `commute.carrying_jobs`. W0 only, fp32, no fine-tuned state is loaded.

  sha256 of the task windows files and of E98's carrying.json                              -> results/stage0.json
  Pile pass (W0): per neuron, its top n_top rows of window_rows by the row's peak firing, and a sorted sample of its
          firing on every position of quantile_rows                                           -> large/pile_top.pt
  C0_pile  engaged values on check_rows: median over neurons in pile_check_band               (exit 1 on a miss)
  engaged values of every neuron on every task's read windows at the answer position         -> large/engaged.pt
  C0_w0    per task, a set chosen from W0 on even windows, read on odd ones, beats random sets (exit 1 on a miss)
  C0_null  the P0 rule with a random draw in place of every carrying set reads no separation   (exit 1 on a miss)
  per state: carrying share, n_draws block-matched random shares, excess; P0 and the reported rows -> stage0.json
  stage 1's pool and batch files (ids hashed, no label)  -> large/batches/b<NN>.json; the key -> large/key.json
  C0_files the batch files hold nothing but ids and windows                                   (exit 1 on a miss)
Every number is printed.
"""

from __future__ import annotations

import json
import random
import re
import secrets
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import carrying_jobs as CJ
from commute import keydep as KD
from commute import neuron_r as NR
from commute import steering as ST
from commute import vpd
from commute.data import pile_rows
from commute.jobs import hard_exit, load_experiment

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
LARGE = RESULTS / "large"
PRED, CARRY = RESULTS / "predictions.json", REPO / "experiments" / "neuron_r" / "results" / "carrying.json"
E98 = load_experiment("neuron_r", "run")  # its files() and load_reads(): the same read windows E98 ranked on
DEVICE, N = "cuda", NR.N_NEURONS
log = KD.log


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=False).stdout


class Out:
    """stage0.json: each record written and logged as it lands; a miss (ok False) writes, logs FAIL and exits 1."""

    def __init__(self) -> None:
        self.out: dict[str, Any] = {"git_head": git("rev-parse", "HEAD").strip(),
                                    "git_dirty": git("status", "--porcelain", "--untracked-files=no"), "rows": {}}

    def put(self, key: str, rec: dict[str, Any], ok: bool | None = None) -> dict[str, Any]:
        self.out["rows"][key] = {**rec, **({} if ok is None else {"ok": bool(ok)})}
        (RESULTS / "stage0.json").write_text(json.dumps(self.out, indent=1, sort_keys=True, default=float))
        log(f"[stage0 {key}] ok {ok} {json.dumps(rec, sort_keys=True, default=float)[:2000]}")
        if ok is False:
            log(f"FAIL: {key}: stop and tell the lead")
            sys.exit(1)
        return rec


# ----------------------------------------------------------------------------- W0 firing


@torch.no_grad()
def firing(net: ST.Net, toks: Tensor) -> Tensor:
    """W0's GELU outputs, every block, (B, L, 12288), flat index l x 3072 + i (`neuron_r`)."""
    _, _, acts = NR.forward(net, toks)
    return torch.cat([acts[l] for l in range(len(acts))], dim=-1).float()


def pile_pass(net: ST.Net, rows: Tensor, S: dict[str, Any]) -> dict[str, Tensor]:
    """Per neuron, the n_top rows of `rows` with the highest peak firing, with each row's peak position and value."""
    peak, at = [], []
    for b in rows.split(S["batch_rows"]):
        v, p = firing(net, b).max(dim=1)
        peak.append(v.cpu())
        at.append(p.cpu())
    peak_t, at_t = torch.cat(peak), torch.cat(at)  # (R, N)
    vals, idx = peak_t.topk(S["n_top"], dim=0)  # (n_top, N), rows ranked by peak
    return {"values": vals, "rows": idx, "pos": at_t.gather(0, idx)}


def sorted_sample(net: ST.Net, rows: Tensor, S: dict[str, Any]) -> Tensor:
    """(N, positions): every neuron's firing on every position of `rows`, sorted, on the GPU."""
    parts = [firing(net, b).reshape(-1, N) for b in rows.split(S["batch_rows"])]
    return torch.cat(parts).T.contiguous().sort(dim=1).values


def engaged_on(sample: Tensor, acts: Tensor) -> Tensor:
    """(W, N) firing -> (N,) mean quantile over the W positions, float64 on CPU."""
    return CJ.quantiles(sample, acts.T.to(sample.device)).mean(dim=1).cpu()


def answer_firing(net: ST.Net, r: NR.Reads, batch: int) -> Tensor:
    """(windows, N): W0's firing at each read window's answer position (`half`, where the target is read)."""
    return torch.cat([firing(net, t)[:, r.half, :] for t, _ in r.batches(batch)])


# ----------------------------------------------------------------------------- sets


def carrying_sets(P: dict[str, Any]) -> dict[str, list[int]]:
    rows, cap = json.loads(CARRY.read_text())["rows"], P["settings"]["set_cap"]
    return {s: rows[s]["top"][: min(rows[s]["k_half"], cap)] for ss in P["states"].values() for s in ss}


def state_shares(eng: dict[str, Tensor], sets: dict[str, list[int]], task_of: dict[str, str], S: dict[str, Any]) -> dict[str, Any]:
    """Per state: carrying share and n_draws block-matched random shares (outside the state's own set), on its task."""
    gen, out = torch.Generator().manual_seed(S["seed"]), {}
    for s, idx in sets.items():
        e = eng[task_of[s]].tolist()
        draws = [CJ.block_matched(idx, set(idx), gen) for _ in range(S["n_draws"])]
        out[s] = {"task": task_of[s], "n": len(idx), "blocks": CJ.blocks_of(idx), "carrying": CJ.set_share(e, idx),
                  "draws": [CJ.set_share(e, d) for d in draws]}
    return out


def c0_w0(eng_even: dict[str, Tensor], eng_odd: dict[str, Tensor], S: dict[str, Any], o: Out) -> None:
    gen, k = torch.Generator().manual_seed(S["seed"] + 1), S["control_k"]
    for t, even in eng_even.items():
        chosen = even.topk(k).indices.tolist()
        odd = eng_odd[t].tolist()
        rand = [CJ.set_share(odd, torch.randperm(N, generator=gen)[:k].tolist()) for _ in range(S["n_draws"])]
        rec = {"control": CJ.set_share(odd, chosen), "random_max": max(rand), "random": rand, "neurons": chosen}
        o.put(f"C0_w0/{t}", rec, rec["control"] - rec["random_max"] >= S["min_gap"])


def rows_rule(P: dict[str, Any], excess: dict[str, float]) -> dict[str, Any]:
    return {n: CJ.row_rule(excess, row, P["settings"]["min_gap"]) for n, row in P["rows"].items()}


def p0(P: dict[str, Any], shares: dict[str, Any], o: Out) -> None:
    null = {s: CJ.band(r["draws"][0], r["draws"][1:])["excess"] for s, r in shares.items()}
    nr = rows_rule(P, null)
    o.put("C0_null", {"rows": nr}, all(v["verdict"] in ("none", "unreadable") for v in nr.values()))
    for s, r in shares.items():
        o.put(f"state/{s}", {**r, **CJ.band(r["carrying"], r["draws"])})
    rr = rows_rule(P, {s: CJ.band(r["carrying"], r["draws"])["excess"] for s, r in shares.items()})
    v = rr["e53_lowercase_cipher"]["verdict"]
    status = "unreadable" if v == "unreadable" else "pass" if v == "a_higher" else "fail"
    o.put("P0", {"status": status, "rows": rr})


# ----------------------------------------------------------------------------- stage 1's pool and files


def pool(P: dict[str, Any], sets: dict[str, list[int]], eng: dict[str, Tensor]) -> dict[int, list[dict[str, str]]]:
    """neuron -> its roles [{group, role}]: each group's consensus carrying sample, a block-matched random partner, and
    the W0 controls; random and controls from neurons in no scored state's set."""
    roles: dict[int, list[dict[str, str]]] = {}
    add = lambda i, g, r: roles.setdefault(i, []).append({"group": g, "role": r})
    every = {i for idx in sets.values() for i in idx}
    for t, c in P["controls_w0"].items():
        top = [i for i in eng[t].argsort(descending=True).tolist() if i not in every][: c["n"]]
        for i in top:
            add(i, t, "control")
    gen = torch.Generator().manual_seed(P["settings"]["seed"] + 2)
    for g, spec in P["groups"].items():
        cap = spec.get("cap", P["settings"]["set_cap"])
        sample = CJ.consensus([sets[s][:cap] for s in spec["states"]], spec["n"])
        for i in sample:
            add(i, g, "carrying")
        for i in CJ.block_matched(sample, every | set(roles), gen):
            add(i, g, "random")
    return roles


def window_text(tok: Any, row: Tensor, pos: int, ctx: int) -> str:
    lo, hi = max(0, pos - ctx), min(len(row), pos + ctx + 1)
    pieces = [tok.decode([t]) for t in row[lo:hi].tolist()]
    return CJ.mark(pieces, pos - lo)


def write_batches(P: dict[str, Any], roles: dict[int, list[dict[str, str]]], top: dict[str, Tensor], rows: Tensor, o: Out) -> None:
    S, tok, salt = P["settings"], vpd.tokenizer(), secrets.token_hex(16)
    ids = {i: CJ.neuron_id(salt, i) for i in roles}
    assert len(set(ids.values())) == len(ids)
    order = sorted(roles)
    random.Random(S["seed"]).shuffle(order)
    (LARGE / "batches").mkdir(parents=True, exist_ok=True)
    batches = {f"b{j // S['batch_size']:02d}": order[j : j + S["batch_size"]] for j in range(0, len(order), S["batch_size"])}
    for b, members in batches.items():
        neurons = []
        for i in members:
            peak = float(top["values"][0, i])
            wins = [{"rank": r + 1, "rel": round(float(top["values"][r, i]) / peak, 2) if peak > 0 else None,
                     "text": window_text(tok, rows[int(top["rows"][r, i])], int(top["pos"][r, i]), S["context"])}
                    for r in range(S["n_top"])]
            neurons.append({"id": ids[i], "windows": wins})
        (LARGE / "batches" / f"{b}.json").write_text(json.dumps({"batch": b, "n": len(neurons), "neurons": neurons}, indent=1, ensure_ascii=False))
    key = {"salt": salt, "batches": {b: [ids[i] for i in m] for b, m in batches.items()},
           "neurons": {ids[i]: {"flat": i, "block": i // CJ.D_MLP, "neuron": i % CJ.D_MLP, "top_act": float(top["values"][0, i]),
                                "roles": roles[i]} for i in roles}}
    (LARGE / "key.json").write_text(json.dumps(key, indent=1))
    counts: dict[str, int] = {}
    for rs in roles.values():
        for r in rs:
            counts[f"{r['group']}/{r['role']}"] = counts.get(f"{r['group']}/{r['role']}", 0) + 1
    o.put("pool", {"n_neurons": len(roles), "n_batches": len(batches), "agent_calls": 3 * len(batches), "counts": counts,
                   "dead": sum(float(top["values"][0, i]) <= 0 for i in roles)})


def c0_files(P: dict[str, Any], o: Out) -> None:
    bad = []
    for f in sorted((LARGE / "batches").glob("b*.json")):
        d = json.loads(f.read_text())
        bad += [f"{f.name}: keys {sorted(d)}"] if set(d) != {"batch", "n", "neurons"} else []
        for n in d["neurons"]:
            if set(n) != {"id", "windows"} or not re.fullmatch(r"[0-9a-f]{12}", n["id"]):
                bad.append(f"{f.name}: neuron keys {sorted(n)}")
            elif len(n["windows"]) != P["settings"]["n_top"] or any(set(w) != {"rank", "rel", "text"} for w in n["windows"]):
                bad.append(f"{f.name}: {n['id']} windows")
    o.put("C0_files", {"bad": bad[:20], "n_bad": len(bad)}, not bad)


# ----------------------------------------------------------------------------- main


def main() -> None:
    rel = str(PRED.relative_to(REPO))
    changed = subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", "HEAD", "--", rel], check=False).returncode
    if not PRED.is_file() or changed or not git("ls-files", rel):
        sys.exit(f"{rel} is not committed unchanged: commit it first")
    P = json.loads(PRED.read_text())
    S = P["settings"]
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    o, F = Out(), E98.files()
    win = {k: v for k, v in F.items() if "windows" in k}
    got = {k: KD.sha256(p) if p.is_file() else "missing" for k, (p, _) in win.items()}
    o.put("sha256", {"n_files": len(win), "carrying_json": KD.sha256(CARRY), "mismatch": sorted(k for k, (_, h) in win.items() if got[k] != h)},
          all(got[k] == h for k, (_, h) in win.items()))
    net = ST.Net(DEVICE)
    cache = pile_rows(*S["pile_cache"], log=log)
    wrows, qrows, crows = (cache[a:b] for a, b in (S["window_rows"], S["quantile_rows"], S["check_rows"]))
    top = pile_pass(net, wrows, S)
    sample = sorted_sample(net, qrows, S)
    torch.save({**top, "window_rows": S["window_rows"]}, LARGE / "pile_top.pt")
    pile = engaged_on(sample, firing(net, crows).reshape(-1, N))
    o.put("C0_pile", {"median": float(pile.median()), "q05": float(pile.quantile(0.05)), "q95": float(pile.quantile(0.95))},
          S["pile_check_band"][0] <= float(pile.median()) <= S["pile_check_band"][1])
    reads = E98.load_reads(net, F)
    fire = {t: answer_firing(net, reads[t], S["batch_rows"]) for t in P["tasks"]}
    eng = {t: engaged_on(sample, a) for t, a in fire.items()}
    torch.save(eng, LARGE / "engaged.pt")
    log(f"[engaged] windows per task {json.dumps({t: len(a) for t, a in fire.items()})}")
    c0_w0({t: engaged_on(sample, a[0::2]) for t, a in fire.items()}, {t: engaged_on(sample, a[1::2]) for t, a in fire.items()}, S, o)
    sets = carrying_sets(P)
    task_of = {s: t for t, ss in P["states"].items() for s in ss}
    p0(P, state_shares(eng, sets, task_of, S), o)
    write_batches(P, pool(P, sets, eng), top, wrows, o)
    c0_files(P, o)
    o.out["pass"] = True
    o.put("done", {"peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30})
    hard_exit(0)


if __name__ == "__main__":
    main()
