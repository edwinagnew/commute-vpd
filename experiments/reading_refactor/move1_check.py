"""E309 (cell P2.79), move 1's checks, CPU only: the new `commute.world`, `commute.windows` and `commute.items` give the
same bytes as the old code and the saved records (plan "Order of moves" 1). Writes results/move1_world.json; prints
every check and "MOVE1 ok" or the misses. (Its w0 check of `world.weights()`, a wrapper of `vpd.checkpoint_weights` deleted
in the P2.79 fix, ran and is recorded in move1_world.json.) No model is built: the live-W0 check is old.py's gpu part
(old_<tag>.json _meta.W0_live_equals_checkpoint), which `World.load` also asserts on every GPU load.

  windows      every E136 task's held rows against E136's windows.load_read; E53's train rows against holes_fill's
               e53_train; E211's e53 read and the case read of commute.near_tasks against the old builders; E53's text
               and maps (commute.old_tasks' text tasks) against performed_anchor's; the panel D sha256s against its
               manifest; open_pair amend3's window files against amend3.json; the p256 and rl D files' sha256 and shape
  items        the 45 p256 dW sha256s against manifest_p256.json; every E136 state loads (its file sha256 recorded)
  row_counts   kstar_needed's p256 and rl items: n_windows and window_len against the D rows

    ~/commute-agents/fleet/cpu-run --mem 24G -t 60 python3 experiments/reading_refactor/move1_check.py
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import torch

from commute import items as IT
from commute import windows as WN
from commute.data import pile_rows, target_tokens
from commute.jobs import EXPERIMENTS, hard_exit, load_experiment

OUT = Path(__file__).resolve().parent / "results" / "move1_world.json"
out: dict[str, Any] = {}


def put(section: str, key: str, ok: bool, **facts: Any) -> None:
    out.setdefault(section, {})[key] = {"ok": ok, **facts}
    OUT.write_text(json.dumps(out, indent=1))
    print(f"[{section} {key}] {'ok' if ok else 'MISS'} {json.dumps(facts)[:300]}", flush=True)


def check_e136_windows() -> None:
    WIN: Any = load_experiment("share_untrained", "windows")
    for task, spec in WN.e136_tasks().items():
        old, new = WIN.load_read(task, spec), WN.read_rows(task, "held")
        toks = old.toks[:, : old.ci_len] if old.ci_len else old.toks
        tgt = target_tokens(old.token, old.toks, old.half)
        assert new.tgt is not None
        ok = torch.equal(new.toks, toks) and torch.equal(new.tgt, tgt) and new.pos == old.half
        put("windows", f"{task}/held", ok, shape=list(new.toks.shape), sha256=new.sha256, old_sha256=WN.tensor_sha256(toks), tgt_sha256=WN.tensor_sha256(new.tgt))


def check_e53() -> None:
    HF: Any = load_experiment("holes_fill", "run")
    for name in ("lowercase", "cipher", "broad_cipher"):
        new = WN.read_rows(f"e53/{name}", "train")
        put("windows", f"e53/{name}/train", torch.equal(new.toks, HF.e53_train(name)), shape=list(new.toks.shape), sha256=new.sha256)
    pa: Any = load_experiment("performed_anchor", "run")
    put("windows", "e53 maps", WN.e53_pi() == pa.e53_pi() and WN.e53_a1_maps() == pa.a1_maps())
    rows = pile_rows("train", 0, 4096)[:256]
    for name in ("lowercase", "cipher", "broad_cipher"):  # old_tasks._text's call, on E53's own rows
        (t1, a1), (t0, a0) = WN.e53_task_text(name, rows, WN.e53_map(name)), pa.task_text(name, rows, WN.e53_map(name))
        same = torch.equal(t1, t0) and all(torch.equal(x, y) for x, y in zip(dataclasses.astuple(a1), dataclasses.astuple(a0), strict=True))
        put("windows", f"e53/{name}/text", same, sha256=WN.tensor_sha256(t1))


def check_near_tasks() -> None:
    from commute import near_tasks as NT

    R2: Any = load_experiment("realistic_tasks", "run2")
    new, old = NT.rows(NT.TASKS["lowercase"]).read, R2.e53_read("lowercase")
    put("windows", "near_tasks lowercase", all(torch.equal(getattr(new, f), getattr(old, f)) for f in ("toks", "tgt", "fit", "fit_tgt")) and new.pos == old.pos and new.name == old.name)
    WIN: Any = load_experiment("share_untrained", "windows")
    r = WIN.load_read("case_label/real", WN.e136_tasks()["case_label/real"])
    t, new = r.toks[:, : r.ci_len], NT.rows(NT.TASKS["case_real"]).read
    put("windows", "near_tasks case_real", torch.equal(new.toks, t) and torch.equal(new.tgt, r.toks[:, -1]) and torch.equal(new.fit, t) and new.pos == r.half)


def check_saved_windows() -> None:
    for task in json.loads(WN.PANEL_MANIFEST.read_text()):
        r = WN.read_rows(f"panel/{task}", "D")  # the loader stops on a sha256 or shape miss
        put("windows", f"panel/{task}/D", True, shape=list(r.toks.shape), sha256=r.sha256)
    maps = json.loads((WN.AMEND3 / "amend3.json").read_text())["maps"]
    for name, rec in maps.items():
        if not isinstance(rec, dict) or "sha256" not in rec:
            continue
        for kind, part in WN.AMEND3_PART.items():
            if part in rec["sha256"]:
                r = WN.read_rows(f"e58/{name}", kind)  # file sha256 checked at load
                put("windows", f"e58/{name}/{kind}", True, file=r.source, file_sha256=rec["sha256"][part], shape=list(r.toks.shape))
    for task in ("p256/addition", "p256/brackets", "p256/sequence", "rl/repeat"):
        r = WN.read_rows(task, "D")
        put("windows", f"{task}/D", r.tgt is not None and len(r.tgt) == len(r.toks), shape=list(r.toks.shape), sha256=r.sha256, file_sha256=WN.file_sha256(WN.REPO / r.source))


def check_items() -> None:
    for name, it in IT.inventory("p256").items():
        d = IT.load_dw(it)  # stops on a sha256 miss
        put("items", name, len(d) == IT.N_MATRICES, sha256=it.sha256)
    for name, it in IT.inventory("e136").items():
        d = IT.load_dw(it)
        put("items", name, len(d) == IT.N_MATRICES, file_sha256=WN.file_sha256(WN.REPO / it.path))
    put("items", "aliases", IT.resolve("ident32_s0").name == "e46/ident32_s0" and IT.resolve("p256-addition_rl_s0").name == "p256/addition_rl_s0")


def check_row_counts() -> None:
    shapes = {t: list(WN.read_rows(t, "D").toks.shape) for t in ("p256/addition", "p256/brackets", "p256/sequence", "rl/repeat")}
    for p in sorted((EXPERIMENTS / "kstar_needed" / "results" / "items").glob("*.json")):
        rec = json.loads(p.read_text())
        task = f"p256/{rec['item'].split('/')[1].split('_')[0]}" if rec["item"].startswith("p256/") else "rl/repeat" if rec["item"].startswith("rl/") else None
        if task is None or "n_windows" not in rec:
            continue
        put("row_counts", rec["item"], [rec["n_windows"], rec["window_len"]] == shapes[task], saved=[rec["n_windows"], rec["window_len"]], rows=shapes[task], task=task)


def main() -> int:
    for f in (check_e136_windows, check_e53, check_near_tasks, check_saved_windows, check_items, check_row_counts):
        f()
    miss = [f"{s} {k}" for s, d in out.items() for k, v in d.items() if not v["ok"]]
    out["summary"] = {"n_checks": sum(len(d) for d in out.values()), "misses": miss}
    OUT.write_text(json.dumps(out, indent=1))
    print("MOVE1 ok" if not miss else f"MOVE1 MISSES: {miss}", flush=True)
    return 1 if miss else 0


if __name__ == "__main__":
    hard_exit(main())
