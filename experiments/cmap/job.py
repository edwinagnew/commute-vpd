"""Cell P2.68: the config, the items, each task's rows, and one item's pipeline (discovery, pruning, faithfulness, CMAP).
Spec reports/plans/2026-11-23_plan_p2-68-cmap-spec.md (revision 4); the shared steps are commute.head_cmap and
commute.prakash. run.py is the entry point; stage0.py the controls."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn

import torch
from torch import Tensor

from commute import head_cmap as HC
from commute import jobs, prakash
from commute import old_tasks as OT
from commute.data import NEXT, target_tokens
from commute.jobs import EXPERIMENTS, load_experiment

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED, STAGE0, ITEMS = RESULTS / "predictions.json", RESULTS / "stage0.json", RESULTS / "items"
REPO = EXPERIMENTS.parent
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
L, H = 4, 6  # the 67M's blocks and heads


def log(msg: str) -> None:
    print(msg, flush=True)


def dump(path: Path, out: dict[str, Any]) -> None:
    jobs.write_stage(path, out)


def stop(path: Path, out: dict[str, Any], why: str) -> NoReturn:
    jobs.stop(path, out, why, log)


def read_json(rel: str | Path) -> Any:
    return json.loads((REPO / rel).read_text())


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    authors_sha256: dict[str, str]  # commute.prakash.FILES -> sha256 of the pinned checkout
    key_tasks: tuple[str, ...]  # E237's five 96-key maps
    key_seeds: tuple[int, ...]
    pairs: tuple[str, ...]  # E187's nine built pairs
    control_state: str  # E237's; times 0 is the do-nothing
    split_seed: int
    n_discovery: int
    n_prune: int
    batch: int
    se_chunk: int  # discovery prompts per apply_pp call: 1 gives per-prompt scores
    pool_se: float  # a sender enters the pool if its mean score < -pool_se * se
    min_frac: str  # the minimality rule's 1 %, exact
    k_frac: str  # its 30 %
    n_random_sets: int
    random_set_seed: int
    n_gate: int  # random directions per (task, scale)
    gate_seed: int  # direction j uses gate_seed + j
    gain_floor: float  # neuron_r's gain floor (log-odds)
    e237_random_seed: int  # E237's random_<state> items, read beside the gate
    plant_seed: int  # draws h* in block 3 (and the wrong-head control's other head)
    plant_lr: float
    plant_batch: int
    plant_kl_lambda: float
    plant_grad_clip: float
    plant_steps: int
    plant_checkpoints: tuple[int, ...]
    plant_ft_seed: int
    two_layer_seed: int  # draws the block-0 sender s and the delta
    max_pool: int  # a larger pool in one group waits for pricing (--large-pools runs it)
    rerun_over: float  # stage 1 stops when its first items run over the estimate by this factor
    hook_floor_tasks: dict[str, str]  # format -> the task whose D measures the hook floor

    def __post_init__(self) -> None:
        assert len(self.key_tasks) == 5 and self.key_seeds == (0, 1, 2) and len(self.pairs) == 9
        assert set(self.authors_sha256) == set(prakash.FILES) and set(self.hook_floor_tasks) == {"kv", "next", "last"}
        assert self.plant_checkpoints[-1] == self.plant_steps and self.n_discovery <= self.n_prune


def load_config() -> Config:
    s = json.loads(PRED.read_text())["settings"]
    return jobs.config_from_settings(Config, {k: tuple(v) if isinstance(v, list) else v for k, v in s.items()})


# ---- the items: (state, scale) once each ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Item:
    name: str
    state: str
    task: str
    file: str
    c: float
    roles: tuple[str, ...] = ()  # "key" and/or "<pair>:<side>"


def item_name(state: str, c: float) -> str:
    return f"{state}_x{c:.4f}"


def manifest(cfg: Config) -> dict[str, Item]:
    """Appendix E: E237's 15 key items at its c, and both sides of E187's nine pairs, the larger side at m67's c."""
    e147, e136 = read_json("experiments/refit_96key/results/inventory.json"), read_json("experiments/share_untrained/results/inventory.json")
    m67, c237 = read_json("experiments/size_matched/results/m67.json")["pairs"], read_json("experiments/key_grid/results/stage0.json")["c"]
    e58: Callable[[str, int], tuple[str, str, str]] = lambda m, s: (f"e58/{m}_s{s}", f"e58/{m}", f"experiments/refit_96key/results/large/dW_{m}_s{s}.pt")
    rows: list[tuple[str, str, str, float, str]] = [(*e58(m, s), c237[m], "key") for m in cfg.key_tasks for s in cfg.key_seeds]
    sides = {p["id"]: {"reuse": p["reuse"], "new": p["new"]} for p in e136["pairs"]}
    for p in e147["pairs"]:
        sides[p["id"]] = {k: [f"e58/{p[k]}_s{s}" for s in e147["seeds"]] for k in ("reuse", "new")}
    for pid in cfg.pairs:
        for side, states in sides[pid].items():
            c = m67[pid]["c"] if m67[pid]["larger"] == side else 1.0
            for st in states:
                spec: tuple[str, str, str] = e58(st.removeprefix("e58/").rsplit("_s", 1)[0], int(st.rsplit("_s", 1)[1])) if st.startswith("e58/") else (
                    st, e136["states"][st]["task"], e136["states"][st]["file"])  # fmt: skip
                rows.append((*spec, c, f"{pid}:{side}"))
    out: dict[str, Item] = {}
    for st, task, file, c, role in rows:
        n = item_name(st, c)
        out[n] = Item(n, st, task, file, c, (*out[n].roles, role) if n in out else (role,))
    return out


# ---- each task's rows: D (scoring), discovery and prune prompts with their noise --------------------------------------


@dataclass
class Task:
    name: str
    fmt: str  # kv, last (E62), next (E53)
    D: HC.Prompts  # the scoring rows
    disc: HC.Prompts
    prune: HC.Prompts
    has_c: bool
    record: dict[str, Any] = field(default_factory=dict)


def _other(own: Tensor, pool: Tensor, gen: torch.Generator) -> Tensor:
    """Per row, a seeded key of `pool` other than the row's own."""
    idx = (own[:, None] == pool[None]).long().argmax(1)
    assert bool((pool[idx] == own).all()), "a row's task token is not in the pool"
    j = torch.randint(0, len(pool) - 1, (len(own),), generator=gen)
    return pool[j + (j >= idx).long()]


def _prompts(clean: Tensor, labels: Tensor, idx: Tensor, fmt: str, pool: Tensor | None, gen: torch.Generator) -> HC.Prompts:
    """Rows `idx`, each with a noise prompt: another of these rows (a seeded cycle), its task token (kv: the key at the
    read; E62: the second A, before the read) a seeded other pool key; E53: the other row as it is (B.2)."""
    order = idx[torch.randperm(len(idx), generator=gen)]
    partner = dict(zip(order.tolist(), order.roll(-1).tolist(), strict=True))
    noise = clean[torch.tensor([partner[i] for i in idx.tolist()])].clone()
    if fmt != "next":
        assert pool is not None
        at = -1 if fmt == "kv" else -2
        noise[:, at] = _other(clean[idx, at], pool, gen)
    return HC.Prompts(clean[idx], labels[idx], noise)


def _d_rows(fmt: str, D: Tensor, table: Tensor | None, pools: dict[str, Any] | None, arm: str | None) -> tuple[Tensor, Tensor]:
    """D's rows with the format's label defined, and that label. kv: table[token at the read] (-1 off the trained
    keys). E62: windows holding "A B .. A M" with A a pool key and M the marker at the read: copy's label is the token
    after the first A, lookup's the A's word."""
    if fmt == "kv":
        assert table is not None
        lab = table[D[:, -1]]
        return D[lab >= 0], lab[lab >= 0]
    assert pools is not None and arm is not None
    keys, marker = torch.tensor([int(k) for k in pools["keys"]]), int(pools["marker"][0])
    A = D[:, -2]
    first = (D[:, :-2] == A[:, None]).long().argmax(1)
    ok = torch.isin(A, keys) & (D[:, -1] == marker) & (D[:, :-2] == A[:, None]).any(1) & (first + 1 < D.shape[1] - 2)
    if arm == "copy":
        lab = D[torch.arange(len(D)), first + 1]
    else:
        words = {int(k): int(v) for k, v in pools["words"].items()}
        lab = torch.tensor([words.get(int(a), -1) for a in A])
        ok &= lab >= 0
    return D[ok], lab[ok]


def _e53_train(name: str) -> Tensor:
    """E53's original scorer's train windows (`Run.task`'s t.train), rebuilt by its code outside its Run."""
    from commute import performed
    from commute.data import pile_rows

    pa = load_experiment("regating_truth_table", "run").performed_anchor()
    pi = {"lowercase": {}, "cipher": pa.e53_pi(), "broad_cipher": pa.a1_maps()["broad_cipher"]}[name]
    text, at = pa.task_text(name, pile_rows("train", 0, 4096)[:256], pi)
    w, idx = performed.windows_at(text, at, pa.CFG.half, pa.CFG.n_write_windows, pa.CFG.window_seed)
    assert torch.equal(w[:, pa.CFG.half + 1], at.a[idx]), "the rebuilt train windows' next token is not the scored form"
    return w


def build_task(job: Any, name: str) -> Task:
    """Appendix E and B.3 for one task. D is the definition's 512 read windows (E248's `World.make` via
    kstar_pairs' `read(read_half187(task))`), restricted to the rows whose format label is defined; E53's D windows end
    at the read with the scored token cut off, so E53 is scored on its scorer's held-out windows (`Run.scored`)."""
    spec = job.task_specs[name]
    assert spec["format"] in ("kv", "windows_pt", "e53"), f"{name}: format {spec['format']} (a two-label contrast unit?)"
    fmt = {"kv": "kv", "windows_pt": "last", "e53": "next"}[spec["format"]]
    D_all = job.dw.read(job.dw.read_half187(name)).cpu()
    rec: dict[str, Any] = {"format": fmt, "D_n": len(D_all), "D_sha256": hashlib.sha256(D_all.numpy().tobytes()).hexdigest()}
    pool, table = None, None
    if fmt == "kv":
        w = torch.load(REPO / spec["path"], weights_only=False)
        tr, table = w["train"], w["table"]
        toks, labels, half, pool = tr.tokens, target_tokens(table, tr.tokens, tr.half), tr.half, (table >= 0).nonzero().squeeze(1)
        D, Dlab = _d_rows(fmt, D_all, table, None, None)
    elif fmt == "last":
        v = torch.load(REPO / spec["path"], weights_only=False)
        toks, labels, half = v["train"].tokens, v["target"][spec["arm"]]["train"], v["train"].half
        pool = torch.tensor([int(k) for k in v["pools"]["keys"]])
        D, Dlab = _d_rows(fmt, D_all, None, v["pools"], spec["arm"])
    else:
        rd = load_experiment("share_untrained", "windows").load_read(name, spec)
        toks, half = _e53_train(spec["task"]), rd.half
        labels = target_tokens(NEXT, toks, half)
        D, Dlab = rd.toks[:, : half + 1], target_tokens(NEXT, rd.toks, half)
        rec["D_used"] = "held-out scored windows (Run.scored): D's 8-token windows hold no label for NEXT"
    clean = toks[:, : half + 1]
    assert bool((labels >= 0).all()) and bool((Dlab >= 0).all())
    seen = {tuple(r) for r in D.tolist()}
    keep = torch.tensor([tuple(r) not in seen for r in clean[:, -D.shape[1] :].tolist()])
    clean, labels = clean[keep], labels[keep]
    gen = torch.Generator().manual_seed(job.cfg.split_seed)
    perm = torch.randperm(len(clean), generator=gen)
    nd = min(job.cfg.n_discovery, len(clean) // 2)
    npr = min(job.cfg.n_prune, len(clean) - nd)
    rec |= {"D_labelled": len(D), "train_n": len(keep), "train_removed_as_D": int((~keep).sum()), "n_discovery": nd, "n_prune": npr}
    disc, prune = (_prompts(clean, labels, perm[a:b], fmt, pool, gen) for a, b in ((0, nd), (nd, nd + npr)))
    log(f"[task {name}] {rec}")
    return Task(name, fmt, HC.Prompts(D, Dlab), disc, prune, fmt != "next", rec)


def gate_key(task: str, c: float) -> str:
    return f"{task}@{c:.4f}"


class Job:
    """The definition's world (kstar_pairs' World, whose Net is M here), the authors' code, the items and the tasks."""

    def __init__(self, cfg: Config, authors: Path) -> None:
        self.cfg, self.au, self.items = cfg, prakash.load(authors, cfg.authors_sha256), manifest(cfg)
        E221 = load_experiment("kstar_pairs", "run")
        self.dw = E221.World(E221.load_config()[0])
        e147 = read_json("experiments/refit_96key/results/inventory.json")
        e58 = {f"e58/{m}": {"format": "kv", "path": s["windows"], "batch": 64} for m, s in e147["maps"].items()}
        self.dw.pairs.inv["tasks"].update(e58)  # as E237's World adds elicit; E248 reads its key items through it
        self.task_specs = read_json("experiments/share_untrained/results/inventory.json")["tasks"] | e58
        self.net = self.dw.net
        self.a0 = prakash.Adapter(self.net.net)
        self.tasks: dict[str, Task] = {}
        self._w0: dict[str, HC.Read] = {}
        self.gates: dict[str, dict[str, float]] = {}

    def task(self, name: str) -> Task:
        if name not in self.tasks:
            self.tasks[name] = build_task(self, name)
        return self.tasks[name]

    def w0_read(self, task: str) -> HC.Read:
        if task not in self._w0:
            self._w0[task] = HC.forward_read(self.net.net, self.task(task).D, self.cfg.batch)
        return self._w0[task]

    def dW(self, it: Item) -> dict[str, Tensor]:
        return {k: v * it.c for k, v in OT.load_dw(self.net, REPO / it.file).items()}

    @torch.no_grad()
    def p_noise(self, net: Any, t: Task) -> float:
        """Mean probability of the clean label at the read under the discovery rows' noise prompts (B.2's check)."""
        assert t.disc.noise is not None
        p = [torch.softmax(net(t.disc.noise[s : s + self.cfg.batch].cuda())[0][:, -1].float(), -1).cpu()
             .gather(1, t.disc.labels[s : s + self.cfg.batch, None]) for s in range(0, len(t.disc), self.cfg.batch)]  # fmt: skip
        return float(torch.cat(p).mean())


# ---- one item ---------------------------------------------------------------------------------------------------------


def groups_of(job: Any, aF: prakash.Adapter, t: Task, rec: dict[str, Any]) -> dict[str, list[HC.Node]]:
    """Discovery (B.1): A into the logit; B into A's queries; C into B's values at the position before the read. A
    group whose receivers are empty is skipped. A head in both A and B is kept in B (C.1)."""
    cfg, pools = job.cfg, {}
    plan = [("A", None, "q", 0), ("B", "A", "q", 0)] + ([("C", "B", "v", 1)] if t.has_c else [])
    for g, recv, comp, rel in plan:
        if recv is not None and not pools[recv]:
            pools[g], rec[g] = [], {"skipped": "no receivers"}
            continue
        m, se = HC.discover(job.au, aF, t.disc, pools.get(recv, []), comp, rel, cfg.se_chunk, cfg.batch)
        pools[g] = HC.pool(m, se, cfg.pool_se)
        rec[g] = {"mean": m.tolist(), "se": se.tolist(), "pool": pools[g], "n_pool": len(pools[g]), "n_pool_mean_below_0": int((m < 0).sum())}
        log(f"  [{g}] pool {len(pools[g])} (mean < 0: {rec[g]['n_pool_mean_below_0']}): {pools[g]}")
    B = set(pools["B"])
    return {"A": [(l, h, 0) for l, h in pools["A"] if (l, h) not in B], "B": [(l, h, 0) for l, h in pools["B"]],
            "C": [(l, h, 1) for l, h in pools.get("C", [])]}  # fmt: skip


def cmap_sets(job: Any, t: Task, Fnet: Any, sets: dict[str, list[HC.Node]], w0: HC.Read, ft: HC.Read, path: Path,
              rec: dict[str, Any]) -> dict[str, Any]:  # fmt: skip
    """CMAP of each set from F into W0 on D, in the three units, beside W0 and F; zero sites on a non-empty set stops."""
    cache, out = HC.o_inputs(Fnet, t.D.clean, job.cfg.batch), {}
    for k, nodes in sets.items():
        logits, sites = HC.cmap(job.au, job.a0, cache, t.D.clean, nodes, job.cfg.batch)
        r = HC.Read(logits, t.D.labels)
        out[k] = {"nodes": sorted(nodes), "sites": sites, **r.summary(), "share": HC.shares(r, w0, ft),
                  "vs_W0": HC.compare(r, w0), "vs_ft": HC.compare(r, ft)}  # fmt: skip
        log(f"  [cmap {k}] {len(nodes)} nodes, {sites} sites: acc {out[k]['acc']:.4f}, log-odds {out[k]['mean_logodds']:.4f}, "
            f"share {out[k]['share']}")
        if nodes and sites == 0:
            stop(path, rec | {"cmap": out}, f"{t.name}: CMAP of {k} overwrote no entry")
    return out


def run_item(job: Any, name: str, task: str, dW: dict[str, Tensor], floor: dict[str, float] | None, path: Path,
             extra: Callable[[dict[str, list[HC.Node]]], dict[str, list[HC.Node]]] | None = None, large_ok: bool = False,
             meta: dict[str, Any] | None = None) -> dict[str, Any]:  # fmt: skip
    """One fine-tune W0 + dW through discovery, pruning, faithfulness on D and CMAP on D; written to `path`."""
    cfg, t, secs = job.cfg, job.task(task), {}
    Fnet = OT.with_delta(job.net, dW).net
    aF, w0 = prakash.Adapter(Fnet), job.w0_read(task)
    ft = HC.forward_read(Fnet, t.D, cfg.batch)
    g = HC.gains(ft, w0)
    clears = None if floor is None else HC.clears(g, floor) and g["logodds"] > cfg.gain_floor
    rec: dict[str, Any] = {"experiment": EXPERIMENT, "item": name, **(meta or {}), "task": task, "task_rows": t.record, "W0": w0.summary(),
                           "ft": ft.summary(), "gain": g, "gate": floor, "clears_gate": clears,
                           "p_label_noise": {"ft": job.p_noise(Fnet, t), "W0": job.p_noise(job.net.net, t)}}  # fmt: skip
    log(f"[{name}] W0 {rec['W0']}; ft {rec['ft']}; gain {g}; clears the gate {clears}; p(label | noise) {rec['p_label_noise']}")
    t0 = time.time()
    groups = groups_of(job, aF, t, rec)
    secs["discovery"] = time.time() - t0
    if max(len(v) for v in groups.values()) > cfg.max_pool and not large_ok:
        rec["waits"] = f"a pool over {cfg.max_pool}: priced before pruning ({ {k: len(v) for k, v in groups.items()} })"
        dump(path, rec | {"seconds": secs})
        log(f"[{name}] WAITS: {rec['waits']}")
        return rec
    t0, T_D = time.time(), t.D.clean.shape[1]
    means = HC.position_means(Fnet, t.prune.noise, cfg.batch)  # type: ignore[arg-type]
    n_evals = [0]

    def count(S: frozenset[HC.Node]) -> int:
        n_evals[0] += 1
        return HC.n_correct(Fnet, t.prune, S, means, H, cfg.batch)

    pruned = HC.prune(groups, count, cfg.min_frac, cfg.k_frac)
    secs["prune"], t0 = time.time() - t0, time.time()
    circuit = sorted(n for v in pruned.values() for n in v)
    meansD = [m[-T_D:] for m in means]  # aligned from the read
    randoms = HC.matched_random(circuit, cfg.n_random_sets, cfg.random_set_seed, L, H) if circuit else []
    fD = lambda S: HC.n_correct(Fnet, t.D, S, meansD, H, cfg.batch)
    fM, fC = fD(None), fD(frozenset(circuit))
    faith = {"rows": "D", "F_M": fM, "F_Cir": fC, "faithfulness": fC / fM if fM else None,
             "random": [{"F": (x := fD(frozenset(r))), "faithfulness": x / fM if fM else None,
                         "overlap": len(set(r) & set(circuit)) / len(circuit)} for r in randoms]}  # fmt: skip
    secs["faith"], t0 = time.time() - t0, time.time()
    rec |= {"groups_pruned": pruned, "circuit": circuit, "faith": faith, "n_prune_evals": n_evals[0]}
    log(f"[{name}] group sizes after pruning { {k: len(v) for k, v in pruned.items()} }; faithfulness {faith['faithfulness']} "
        f"(F {fC} of {fM}); random {[r['faithfulness'] for r in faith['random']]}")
    sets = {"A": pruned["A"], "B": pruned["B"], "C": pruned["C"], "AB": pruned["A"] + pruned["B"], "whole": circuit}
    sets |= {f"random{i}": r for i, r in enumerate(randoms)} | (extra(pruned) if extra else {})
    rec["cmap"] = cmap_sets(job, t, Fnet, sets, w0, ft, path, rec)
    secs["cmap"] = time.time() - t0
    rec["seconds"], rec["n_cmap_sets"] = secs, len(sets)
    rec["compared"] = {"whole_logodds_share": rec["cmap"]["whole"]["share"]["logodds"] if clears else None,
                       "AB_logodds_share": rec["cmap"]["AB"]["share"]["logodds"] if clears else None}
    log(f"[{name}] COMPARED whole-circuit log-odds share {rec['compared']['whole_logodds_share']}, A+B {rec['compared']['AB_logodds_share']}; "
        f"seconds {secs}")
    dump(path, rec)
    torch.cuda.empty_cache()
    return rec


def item_path(name: str) -> Path:
    return ITEMS / f"{name.replace('/', '-')}.json"


def run_listed(job: Any, it: Item, large_ok: bool) -> dict[str, Any]:
    """A manifest item at its task's gate for its scale, written to results/items/<item>.json."""
    floor = job.gates[gate_key(it.task, it.c)]
    return run_item(job, it.name, it.task, job.dW(it), floor, item_path(it.name), large_ok=large_ok,
                    meta={"state": it.state, "c": it.c, "roles": list(it.roles)})  # fmt: skip
