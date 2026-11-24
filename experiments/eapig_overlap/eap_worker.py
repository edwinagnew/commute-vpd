"""P2.69 worker (venv 2: Python 3.10 pinned to FinetuneCircuits' requirements). Started by run.py with PYTHONPATH set to
the staged copy of FC's `eap/` tree (a namespace package, imported as FC's notebook imports it) and a stand-in
`pygraphviz` (graph.py imports it for `to_graphviz` only, which nothing here calls). It reads one JSON request per line
on standard input and writes one JSON reply per line to the original standard output; everything printed, the released
code's prints included, goes to standard error, which is the job's log.

What it runs is the released code: `attribute(model, g, dataloader, partial(kl_divergence, loss=True, mean=True),
method='EAP-IG', ig_steps=5)`, `g.apply_topn(int(927 * 0.05), absolute=True)`, `g.prune_dead_nodes()`, the notebook's
`calculate_faithfulness` with `partial(kl_divergence, loss=False, mean=False)`, and Hanna's greedy ladder through
`g.apply_greedy(n, absolute=True)`. The one change (D3): `tokenize_plus` in `eap.attribute` and `eap.evaluate` returns our
token windows (no BOS, a mask of ones, lengths equal to the window length). A batch's "strings" are window indices: i
for the clean window i, N + i for its corrupted twin; the label is the window index.
"""
# pyright: reportMissingImports=false

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
import traceback
from functools import partial
from importlib.metadata import version
from typing import Any

import eap.attribute
import eap.evaluate
import tl_convert
import torch
from eap.attribute import attribute
from eap.evaluate import evaluate_baseline, evaluate_graph, get_circuit_logits
from eap.graph import Graph
from torch import Tensor
from torch.utils.data import DataLoader
from wang_notebook import calculate_faithfulness, collate_EAP, kl_divergence

_TOKENS: dict[str, Tensor] = {}  # "stack": (2N, T) clean windows then their corrupted twins, on the GPU


def tokenize_plus(model: Any, inputs: list[int]):
    """D3: our windows by index, as the released function's four outputs (tokens, attention mask, lengths, positions)."""
    tokens = _TOKENS["stack"][torch.as_tensor(list(inputs), device="cuda")]
    attention_mask = torch.ones_like(tokens)
    return tokens, attention_mask, attention_mask.sum(1), attention_mask.size(1)


eap.attribute.tokenize_plus = tokenize_plus
eap.evaluate.tokenize_plus = tokenize_plus


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(a: Tensor, b: Tensor) -> float:
    """Largest per-window max |a - b| over max |b| (the native forward's floor uses the same form, in run.py)."""
    a, b = a.flatten(1).double(), b.flatten(1).double()
    return float(((a - b).abs().amax(1) / b.abs().amax(1)).max())


def kl_last(a: Tensor, b: Tensor) -> Tensor:
    """The notebook's kl_divergence on read-position logits (B, V): KL(softmax b || softmax a) / V per window."""
    ones = torch.ones(len(a), dtype=torch.long, device=a.device)
    return kl_divergence(a[:, None], b[:, None], ones, ones, mean=False, loss=False)  # labels are unused by the KL


def circuit(g: Graph) -> dict[str, Any]:
    return {"edges": sorted(e.name for e in g.edges.values() if e.in_graph), "nodes": sorted(n.name for n in g.nodes.values() if n.in_graph),
            "empty": not g.nodes["logits"].in_graph}  # fmt: skip


class Worker:
    def __init__(self) -> None:
        self.model = tl_convert.build()
        self.tasks: dict[str, Tensor] = {}
        self.n: dict[str, int] = {}

    def hello(self) -> dict[str, Any]:
        return {"python": f"{sys.version_info[0]}.{sys.version_info[1]}", "torch": torch.__version__, "transformer_lens": version("transformer_lens"),
                "transformers": version("transformers"), "eap_attribute_file": eap.attribute.__file__,
                "tf32": torch.backends.cuda.matmul.allow_tf32, "device": torch.cuda.get_device_name()}  # fmt: skip

    def load(self, path: str, sha: str) -> dict[str, Any]:
        t0, got = time.time(), sha256(path)
        if got != sha:
            raise ValueError(f"{path}: sha256 {got}, run.py recorded {sha}")
        rec = tl_convert.convert(self.model, torch.load(path, map_location="cpu", weights_only=True))
        return rec | {"seconds": time.time() - t0}

    def add_task(self, name: str, path: str) -> dict[str, Any]:
        w = torch.load(path, map_location="cpu", weights_only=True)
        self.tasks[name] = torch.cat([w["clean"], w["corrupted"]]).cuda()
        self.n[name] = len(w["clean"])
        return {"n": self.n[name], "T": int(w["clean"].shape[1])}

    def use(self, task: str, index: list[int] | None, batch: int) -> DataLoader:
        _TOKENS["stack"], n = self.tasks[task], self.n[task]
        idx = list(range(n)) if index is None else index
        rows: Any = [(i, n + i, i) for i in idx]  # a list is a map-style dataset
        return DataLoader(rows, batch_size=batch, collate_fn=collate_EAP)

    @torch.no_grad()
    def read_logits(self, task: str, batch: int, corrupted: bool, first: int | None = None) -> Tensor:
        n = self.n[task]
        toks = self.tasks[task][n:] if corrupted else self.tasks[task][:n]
        if first is not None:
            return self.model(toks[:first]).float()
        return torch.cat([self.model(x)[:, -1].float() for x in toks.split(batch)])

    def parity(self, task: str, native: str, batch: int, first: int) -> dict[str, Any]:
        """Stage 0: the converted logits against the native forward's (saved by run.py), and the full and empty circuits
        through the released `get_circuit_logits` against the converted model's own clean and corrupted runs."""
        nat = torch.load(native, map_location="cuda", weights_only=True)
        out: dict[str, Any] = {}
        tl = {side: self.read_logits(task, batch, side == "corrupted") for side in ("clean", "corrupted")}
        for side, x in tl.items():
            out[f"rel_{side}"] = rel(x, nat[side])
            out[f"rel_{side}_all_positions"] = rel(self.read_logits(task, batch, side == "corrupted", first), nat[f"{side}_full"])
            out[f"finite_{side}"] = bool(torch.isfinite(x).all())
            out[f"kl_tl_native_{side}"] = float(kl_last(x, nat[side]).mean())
        single = self.read_logits(task, 1, False)
        out["kl_batch_single_clean"] = float(kl_last(tl["clean"], single).mean())
        out["rel_batch_single_clean"] = rel(tl["clean"], single)
        clean_tl, corrupted_tl = tl["clean"], tl["corrupted"]
        full, empty = Graph.from_model(self.model), Graph.from_model(self.model)
        for e in empty.edges.values():
            e.in_graph = False  # every node kept in the graph, so the released code patches every input (no short cut)
        parts: dict[str, list[Tensor]] = {"full": [], "empty": []}
        for data in self.use(task, None, batch):
            parts["full"].append(get_circuit_logits(self.model, full, data, prune=True)[:, -1].float())
            parts["empty"].append(get_circuit_logits(self.model, empty, data, prune=False)[:, -1].float())
        out["rel_full_circuit_vs_clean"] = rel(torch.cat(parts["full"]), clean_tl)
        out["rel_empty_circuit_vs_corrupted"] = rel(torch.cat(parts["empty"]), corrupted_tl)
        out["rel_full_circuit_vs_native_clean"] = rel(torch.cat(parts["full"]), nat["clean"])
        out["rel_empty_circuit_vs_native_corrupted"] = rel(torch.cat(parts["empty"]), nat["corrupted"])
        return out

    def attribute(self, task: str, index: list[int] | None, full: bool, batch: int, ig_steps: int, top_frac: float,
                  ladder: list[int], greedy_bar: float, kl_floor: float | None) -> dict[str, Any]:  # fmt: skip
        t0, dl = time.time(), self.use(task, index, batch)
        g = Graph.from_model(self.model)
        attribute(self.model, g, dl, partial(kl_divergence, loss=True, mean=True), method="EAP-IG", ig_steps=ig_steps)
        scores = {e.name: float(e.score) for e in g.edges.values()}
        n = int(len(g.edges) * top_frac)
        ranked = sorted((abs(s) for s in scores.values()), reverse=True)
        g.apply_topn(n, absolute=True)
        g.prune_dead_nodes()
        out: dict[str, Any] = {"scores": scores, "n_total_edges": len(g.edges), "n_top": n, "tie_at_cut": ranked[n - 1] == ranked[n],
                               "finite_scores": all(math.isfinite(s) for s in scores.values()),
                               "circuit": circuit(g), "seconds_attribution": time.time() - t0}  # fmt: skip
        if full:
            out |= self.faithfulness(g, dl, kl_floor) | self.greedy(g, task, dl, ladder, greedy_bar, kl_floor, batch)
        out["seconds"] = time.time() - t0
        return out

    def faithfulness(self, g: Graph, dl: DataLoader, kl_floor: float | None) -> dict[str, Any]:
        """Wang's faithfulness of the 5% circuit (the notebook's function), with F(M) and F(C) beside; "no gap" when
        F(M) is at or under the stage-0 fp32 KL floor or not finite (the ratio is then undefined)."""
        metric = partial(kl_divergence, loss=False, mean=False)
        f_m = evaluate_baseline(self.model, dl, metric).mean().item()
        rec: dict[str, Any] = {"F_M": f_m, "F_C": None, "faithfulness_abs": None, "faithfulness_pct": None}
        if not math.isfinite(f_m) or (kl_floor is not None and f_m <= kl_floor):
            return rec | {"no_gap": True}
        rec["F_C"] = evaluate_graph(self.model, g, dl, metric).mean().item()
        rec["faithfulness_abs"], rec["faithfulness_pct"] = calculate_faithfulness(self.model, g, dl, metric)
        return rec | {"no_gap": False}

    def greedy(self, g: Graph, task: str, dl: DataLoader, ladder: list[int], bar: float, kl_floor: float | None, batch: int) -> dict[str, Any]:
        """D12: Hanna's greedy search on the same scores, the smallest circuit on the ladder with normalised faithfulness
        1 - KL(clean || circuit) / KL(clean || corrupted) of at least `bar` (KL means over the windows, the notebook's KL)."""
        clean = self.read_logits(task, batch, False)  # (N, V) at the read position, by window index
        metric = partial(kl_divergence, loss=False, mean=False)
        b_prime = evaluate_baseline(self.model, dl, metric, run_corrupted=True).mean().item()  # KL(clean || corrupted)
        if not math.isfinite(b_prime) or (kl_floor is not None and b_prime <= kl_floor):
            return {"greedy": {"b_prime": b_prime, "no_gap": True, "ladder": [], "chosen": None}}

        def to_clean(logits: Tensor, corrupted_logits: Tensor, input_lengths: Tensor, labels: Any) -> Tensor:
            last = logits[torch.arange(len(logits), device=logits.device), input_lengths - 1]
            return kl_last(last, clean[torch.as_tensor(list(labels), device=logits.device)])

        rows, chosen = [], None
        for n in ladder:
            g.apply_greedy(n, absolute=True)
            g.prune_dead_nodes()
            c = circuit(g)
            m = evaluate_graph(self.model, g, dl, to_clean).mean().item()
            norm = 1 - m / b_prime
            rows.append({"n": n, "normalised": norm, "edges_after_prune": len(c["edges"]), "nodes_after_prune": len(c["nodes"]), "empty": c["empty"]})
            if chosen is None and norm >= bar:
                chosen = {"n": n, "normalised": norm} | c
        return {"greedy": {"b_prime": b_prime, "no_gap": False, "ladder": rows, "chosen": chosen}}


def main() -> None:
    reply = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)  # the released code's prints and tqdm go to the log, not into the replies
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    w = Worker()
    for line in sys.stdin:
        req = json.loads(line)
        try:
            res = {"ok": True} | getattr(w, req["op"])(**req.get("args", {}))
        except Exception as e:  # noqa: BLE001  (one failed request is reported to run.py, which decides; the worker stays up)
            res = {"ok": False, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
        reply.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
