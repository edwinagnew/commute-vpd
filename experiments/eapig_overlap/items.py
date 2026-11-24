"""P2.69's items and task windows, built with the loaders the earlier readings used (spec appendix B). GPU job only.

Items (F, each against M on its task set):
    x2, x10, rebuilt, random_size_241001     copy24   E214's World (`kstar_levels`) through E242's (`kstar_vpd_wording_a`)
    new_few, new_thin, random_kl_241002      cue      the same; E214's edits rebuilt (their record written to this run's
                                                      results, not kstar_levels') and compared with the committed one
    e62/copy_s{0,1,2}, e62_random            e62_copy  E62's saved updates (`old_tasks.load_dw`, `with_delta`); the random
                                                      at copy_s0's per-matrix norms (`finetune.random_direction`)
    e62/lookup_s{0,1,2}                      e62_lookup
    e58/<task>_s<k>_scaled, random_e58/<task>_s<k>   key_<task>   E237's World (`key_grid`), its c per task
Task sets (clean windows cut after the read position, D4; corrupted: one deciding token swapped, D5, by the rule of
`locator_table/pairs.py` `shifted` with the filed pairing seed):
    copy24     E195's copy24 held-out windows; read at `half` (the key); the key swapped for another key found there
    cue        P1.31's cue windows, A'_i last; A'_i swapped for another of the eight A'
    e62_copy   E62's read windows cut to columns 0..20; B at p + 1 overwritten (`copying.absent_b`, as pairs.py's "copy")
    e62_lookup the same windows; the second A (column 19) swapped for another key found there (pairs.py's "lookup")
    key_<task> E237's read rows with the marker at 6 and a table key at 7 (D11); the key swapped for another table key
`input_records` lists every input file with its saved or filed sha256; stage 0 checks them all and stops on a mismatch.
"""

from __future__ import annotations

import json
import subprocess
from functools import cached_property
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import env
from commute import finetune as FT
from commute import old_tasks as OT
from commute import rule_retention as R
from commute.copying import absent_b
from commute.jobs import EXPERIMENTS, load_experiment

E242: Any = load_experiment("kstar_vpd_wording_a", "run")  # World: E214's items (via E226) and the two matched randoms
KL: Any = E242.E226.KL  # E214 (kstar_levels): World.edits, World.cues; P (COPY24), E208 (REBUILT)
KG: Any = load_experiment("key_grid", "run")  # E237: World (scaled items, randoms, read rows), load_config; E221
E62 = EXPERIMENTS / "induction_cue" / "results"
E147_STAGE0 = EXPERIMENTS / "refit_96key" / "results" / "stage0.json"
REPO = EXPERIMENTS.parent
M_CKPT = env.VPD_DIR / "t-9d2b8f02" / "ckpt" / "model_step_99999.pt"


def shifted(own: Tensor, n: int, seed: int) -> Tensor:
    """Another index of n for each row: experiments/locator_table/pairs.py `shifted`, verbatim (that module imports E97's
    sites, which tokenize Pile rows on import, so it is not imported here)."""
    shift = torch.randint(1, n, (len(own),), generator=torch.Generator().manual_seed(seed))
    return (own + shift) % n


def swap_column(toks: Tensor, at: int, pool: list[int], seed: int) -> Tensor:
    """The token at column `at` swapped for another token of `pool` (sorted), by `shifted`."""
    assert len(pool) >= 2 and set(toks[:, at].tolist()) <= set(pool), "swap pool"
    own = torch.tensor([pool.index(k) for k in toks[:, at].tolist()])
    bad = toks.clone()
    bad[:, at] = torch.tensor(pool)[shifted(own, len(pool), seed)]
    return bad


def input_records(cfg: Any) -> dict[str, tuple[Path, str]]:
    """Every input file with the sha256 it must have: the filed ones (predictions.json "input_sha256"), E62's run.json
    records (its six states, its windows) and E147's stage-0 records (the fifteen key-grid states)."""
    run62 = json.loads((E62 / "run.json").read_text())
    out = {"m_checkpoint": (M_CKPT, cfg.input_sha256["m_checkpoint"]), "copy24_windows": (KL.P.COPY24, cfg.input_sha256["copy24_windows"]),
           "rebuilt_payload": (KL.E208.REBUILT, cfg.input_sha256["rebuilt_payload"]),
           "e62_windows": (E62 / "large" / "windows.pt", run62["windows_sha256"])}  # fmt: skip
    out |= {f"e62/{s}": (E62 / "large" / f"{s}.pt", run62["states"][s]["sha256"]) for s in sorted(run62["states"])}
    e147 = json.loads(E147_STAGE0.read_text())["states"]
    return out | {s: (REPO / e147[s]["file"], e147[s]["sha256"]) for s in (f"e58/{t}_s{k}" for t in cfg.key_tasks for k in cfg.key_seeds)}


class Items:
    """The worlds, each built when first needed, and every item's F and every task's windows."""

    def __init__(self, cfg: Any, out: Path) -> None:
        self.cfg, self.out = cfg, out
        out.mkdir(parents=True, exist_ok=True)
        KL.RESULTS = out  # E214's World writes its edits record to RESULTS / "edits.json": here, not into kstar_levels'

    @cached_property
    def w(self) -> Any:
        cfg242 = E242.load_config()[0]
        w = E242.World(cfg242)
        assert list(w.kl.cfg.e195_C) == list(self.cfg.e195_C) and w.kl.cfg.e195_layer == 1, "E214's x2 columns are not the filed ones"
        assert (cfg242.random, cfg242.random_kl) == ("random_size_241001", "random_kl_241002")
        return w

    @cached_property
    def kg(self) -> Any:
        kg = KG.World(KG.load_config()[0])
        kg.c = json.loads((EXPERIMENTS / "key_grid" / "results" / "stage0.json").read_text())["c"]
        assert all(torch.equal(p, q) for p, q in zip(kg.M.net.parameters(), self.M.net.parameters(), strict=True)), "E237's M differs"
        return kg

    @property
    def M(self) -> R.Model:
        return self.w.kl.M

    # ---- F per item ---------------------------------------------------------------------------------------------

    def make(self, item: str) -> tuple[R.Model, dict[str, Any]]:
        if item == "M":
            return self.M, {}
        if item.startswith(("e62/", "e62_")):
            state = "copy_s0" if item == "e62_random" else item.removeprefix("e62/")
            dw = OT.load_dw(self.w.kl.net, E62 / "large" / f"{state}.pt")
            dw = FT.random_direction(dw, self.cfg.e62_random_seed) if item == "e62_random" else dw
            return R.Model(OT.with_delta(self.w.kl.net, dw).net), {"state": state}
        if item.startswith(("e58/", "random_e58/")):
            state = item.removeprefix("random_").removesuffix("_scaled")
            assert self.kg.full["states"][state]["file"] == json.loads(E147_STAGE0.read_text())["states"][state]["file"], state
            F, _, _, rec = self.kg.make(item)
            return F, rec
        F, _, _, _, rec = self.w.make(item)
        return F, rec | ({"edits_vs_committed": self.edits_check()} if item in ("new_few", "new_thin") else {})

    def edits_check(self) -> dict[str, Any]:
        """E214's edits as rebuilt here against the committed record: the discrete choices exactly, the gains beside."""
        mine = json.loads((self.out / "edits.json").read_text())
        rel = "experiments/kstar_levels/results/edits.json"
        head = json.loads(subprocess.run(["git", "-C", str(REPO), "show", f"HEAD:{rel}"], capture_output=True, text=True, check=True).stdout)
        return {"discrete_equal": {k: mine[k] == head[k] for k in ("A", "A_prime", "y", "slots", "swapped")},
                "gains_here_and_committed": {k: [mine[k], head[k]] for k in ("g", "gamma_new_few", "gamma_new_thin")}}  # fmt: skip

    # ---- task windows ---------------------------------------------------------------------------------------------

    def task(self, name: str) -> tuple[Tensor, Tensor, dict[str, Any]]:
        """(clean, corrupted, record): int64 on the CPU, the read position last."""
        seed = self.cfg.pair_seed
        if name == "copy24":
            v = torch.load(KL.P.COPY24, weights_only=False)["heldout"]
            h = v.half
            clean = v.tokens[:, : h + 1].clone()
            assert len(set(clean[:, h - 1].tolist())) == 1, "copy24: more than one marker at half - 1"
            bad = swap_column(clean, h, sorted(set(clean[:, h].tolist())), seed)
            rec: dict[str, Any] = {"half": h, "read": h, "swapped_column": h}
        elif name == "cue":
            c = self.w.kl.cues
            clean, cue = c.toks.cpu().clone(), c.cue.cpu()
            assert torch.equal(clean[:, -1], torch.tensor(c.A2)[cue]), "cue windows: A'_i is not last"
            bad = clean.clone()
            bad[:, -1] = torch.tensor(c.A2)[shifted(cue, len(c.A2), seed)]
            rec = {"A_prime": list(c.A2), "read": int(clean.shape[1]) - 1, "swapped_column": int(clean.shape[1]) - 1}
        elif name in ("e62_copy", "e62_lookup"):
            v = torch.load(E62 / "large" / "windows.pt", weights_only=False)
            h = json.loads((E62 / "predictions.json").read_text())["settings"]["half"]
            assert h == self.cfg.e62_half
            clean = v["read"].tokens[:, : h + 1].clone()
            if name == "e62_copy":
                p, b = v["p_read"], v["target"]["copy"]["read"]
                assert int((p + 1).max()) < h and torch.equal(clean[torch.arange(len(clean)), p + 1], b), "E62: B is not at p + 1 before the read"
                bad = absent_b(clean, p, b, b.unique().tolist(), seed)
                rec = {"half": h, "read": h, "swapped": "B at p + 1"}
            else:
                bad = swap_column(clean, h - 1, sorted(set(clean[:, h - 1].tolist())), seed)
                rec = {"half": h, "read": h, "swapped_column": h - 1}
        else:
            clean, bad, rec = self.key_task(name.removeprefix("key_"))
        return clean, bad, rec | {"n": len(clean), "T": int(clean.shape[1]), "unchanged_rows": int((bad == clean).all(1).sum())}

    def key_task(self, task: str) -> tuple[Tensor, Tensor, dict[str, Any]]:
        full = f"e58/{task}"
        rows = self.kg.read(self.kg.read_half187(full)).cpu()
        t = KG.E221.e187_task(full, self.kg.pairs.inv["tasks"])
        files = [torch.load(REPO / p, weights_only=False) for p in t.sources]
        marker, keys = int(files[0]["task"].marker), sorted({int(k) for f in files for k in f["task"].keys})
        clean = rows[(rows[:, 6] == marker) & torch.isin(rows[:, 7], torch.tensor(keys))].clone()
        assert len(clean) > 1, f"{full}: under two read rows with the marker at 6 and a table key at 7"
        bad = swap_column(clean, 7, keys, self.cfg.pair_seed)
        return clean, bad, {"read": 7, "swapped_column": 7, "rows_read": len(rows), "rows_kept": len(clean), "marker": marker, "n_keys": len(keys)}


def items_of(cfg: Any) -> dict[str, str]:
    """Every item and its task set, in stage order."""
    out = {i: "copy24" for i in ("x2", "x10", "rebuilt", "random_size_241001")}
    out |= {i: "cue" for i in ("new_few", "new_thin", "random_kl_241002")}
    out |= {f"e62/{m}_s{s}": f"e62_{m}" for m in ("copy", "lookup") for s in (0, 1, 2)} | {"e62_random": "e62_copy"}
    states = [f"e58/{t}_s{s}" for t in cfg.key_tasks for s in cfg.key_seeds]
    task = {s: "key_" + s.removeprefix("e58/").rsplit("_s", 1)[0] for s in states}
    return out | {f"{s}_scaled": task[s] for s in states} | {f"random_{s}": task[s] for s in states}
