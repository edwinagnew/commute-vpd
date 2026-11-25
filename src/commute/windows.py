"""Every saved task's read rows by one call, `read_rows(task, kind)`, so a reading never reaches another experiment's
run.py for its windows (the reading refactor, P2.79 move 1; plan reports/plans/2026-11-25_plan_reading-refactor.md,
appendix A2). Each (task, kind) maps to exactly one source, a saved file or E53's build from cached Pile rows; a source
with a saved sha256 is checked against it at every load.

Rows: toks is exactly what the model and Gamma see (a target the model must not read is never a column of toks: Gamma
is bidirectional over the row, `vpd.compute_ci`), read at column pos; tgt (N,) the target id per row, None for a D that
is read without a target (E_b's write passes).

Tasks and kinds:
  E136's inventory tasks (experiments/share_untrained/results/inventory.json "tasks": e46/*, e54/*, c3/*, e62/*,
    case_label/*, fresh/*, e53/*, class/*)   held: the rows E136 read the share on (its windows.load_read)
  e53/<task> (lowercase, cipher, broad_cipher)  also train: E53's 2,000 hand-write windows (E143's steering fit rows)
  e58/<map> (open_pair amend3, 96 keys)       held: trained keys' held-out windows; held_new, held_full: new / all keys
  panel/<task> (E272's wide set)              D: its 200 E_b windows (eb windows_manifest.json, tensor sha256)
  p256/<task> (addition, brackets, sequence)  D: E256's 256 held-out prompts (rb_states p256_windows_<task>.pt)
  rl/<behaviour> (repeat)                     D: E222's read rows (rb_states rl_windows_<behaviour>.pt)
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor

from commute import performed, vpd
from commute.data import pile_rows, target_tokens
from commute.jobs import EXPERIMENTS
from commute.label_reads import targets

Kind = Literal["held", "train", "D", "held_new", "held_full"]
REPO = EXPERIMENTS.parent
E136_INV = EXPERIMENTS / "share_untrained" / "results" / "inventory.json"
PANEL_MANIFEST = EXPERIMENTS / "convergence_panel" / "results" / "wide" / "eb" / "windows_manifest.json"
RB_LARGE = EXPERIMENTS / "rb_states" / "results" / "large"
AMEND3 = EXPERIMENTS / "open_pair" / "results"
E53 = EXPERIMENTS / "performed_anchor" / "results"
LABEL_HALF = 20  # the label pairs' trigger position, 20 tokens each side (prior_vs_share HALF)
AMEND3_PART: dict[Kind, str] = {"held": "trained", "held_new": "new", "held_full": "full"}


@dataclass(frozen=True)
class E53Config:  # performed_anchor/run.py Config, the fields its windows read (E53's filed values)
    half: int = 63  # windows of 127 tokens
    n_windows: int = 500  # scored windows per task (val rows 0-255)
    n_write_windows: int = 2000  # the hand write's windows (train rows 0-255)
    window_seed: int = 0
    count_rows: int = 4096  # Pile train rows 0-4095 for amendment 1's counts
    a1_lower_ranks: tuple[int, int] = (50, 100)
    a1_cipher_ranks: tuple[int, int] = (50, 550)
    a1_cipher_seed: int = 0


E53_CFG = E53Config()


@dataclass(frozen=True)
class Rows:
    task: str
    kind: str
    toks: Tensor  # (N, L) int64: what the model and Gamma see
    pos: int  # the read position
    tgt: Tensor | None  # (N,) target id per row; None for a D read without one
    source: str  # the file (repo path) or builder the rows came from

    @property
    def sha256(self) -> str:
        return tensor_sha256(self.toks)


def tensor_sha256(t: Tensor) -> str:
    """sha256 of a tensor's bytes (convergence_panel eb.py `sha`, the form its manifest holds)."""
    return hashlib.sha256(t.cpu().contiguous().numpy().tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(p: Path) -> str:
    return str(p.relative_to(REPO))


# ----------------------------------------------------------------------------- E53's text tasks (performed_anchor)


def e53_pi() -> dict[int, int]:
    """E53's cipher map, from its A0 check (performed_anchor `e53_pi`)."""
    a0 = json.loads((E53 / "a0.json").read_text())
    if a0.get("verdict", {}).get("decision") != "GO":
        sys.exit("a0.json is not a GO: the fine-tunes do not start")
    return {int(m["x"]): int(m["pi_x"]) for m in a0["cipher_map"]}


@cache
def e53_a1_maps() -> dict[str, dict[int, int]]:
    """Amendment 1's token maps from the counts on Pile train rows 0-4095 (performed_anchor `a1_maps`)."""
    c, tok = E53_CFG, vpd.tokenizer()
    rows = pile_rows("train", 0, c.count_rows)
    ids = performed.word_ranks(tok, rows)[c.a1_cipher_ranks[0] : c.a1_cipher_ranks[1]].tolist()
    return {"narrow_lower": performed.title_lower_map(tok, rows, c.a1_lower_ranks), "broad_cipher": performed.derangement(ids, c.a1_cipher_seed)}


def e53_map(name: str) -> dict[int, int]:
    """The token map of an E53 task ({} for lowercase, which maps by case pairs)."""
    return {} if name == "lowercase" else e53_pi() if name == "cipher" else e53_a1_maps()[name]


def e53_task_text(name: str, rows: Tensor, pi: dict[int, int]) -> tuple[Tensor, Any]:
    """The task's text of cased Pile `rows` and its scored positions (performed_anchor `task_text`)."""
    if name == "lowercase":
        cp = performed.case_pairs(rows, vpd.tokenizer())
        return cp.lower, cp.in_lower
    return performed.cipher_rows(rows, pi)


def _e53_held(name: str) -> Tensor:
    """E53's scored windows, val rows 0-255 (performed_anchor `Run.scored`, its two checks kept)."""
    c = E53_CFG
    text, at = e53_task_text(name, pile_rows("val", 0, 256), e53_map(name))
    w, idx = performed.windows_at(text, at, c.half, c.n_windows, c.window_seed)
    assert torch.equal(w[:, c.half + 1], at.a[idx]), "a window's next token is not its scored form"
    mask, idx_m = performed.windows_at(performed.position_mask(text, at).long(), at, c.half, c.n_windows, c.window_seed)
    assert torch.equal(idx, idx_m) and bool(mask[:, c.half + 1].all()), "the mask's windows are not the text's"
    return w


def _e53_train(name: str) -> Tensor:
    """E53's hand-write windows of Pile train rows 0-255 (performed_anchor `Run.task`; holes_fill `e53_train`)."""
    c = E53_CFG
    text, at = e53_task_text(name, pile_rows("train", 0, 4096)[:256], e53_map(name))
    return performed.windows_at(text, at, c.half, c.n_write_windows, c.window_seed)[0]


def _e53(task: str, kind: Kind) -> Rows:
    name, h = task.removeprefix("e53/"), E53_CFG.half
    if name == "broad_cipher":  # regating_truth_table `e53_tasks`' check of the map against amendment 1's record
        saved = json.loads((E53 / "check_amend1.json").read_text())["maps"]["broad_cipher"]
        assert [[x, y] for x, y in e53_a1_maps()["broad_cipher"].items()] == [r[:2] for r in saved], "broad cipher map moved"
    toks = {"held": _e53_held, "train": _e53_train}[kind](name)
    return Rows(task, kind, toks, h, toks[:, h + 1], f"performed_anchor E53 {kind} windows")


# ----------------------------------------------------------------------------- saved files


def _kv(task: str, kind: Kind, path: Path, sha: str | None = None) -> Rows:
    """A kv windows file ({"heldout", "table"}): the held-out windows, target = table[token at half] (`steering.load_task`)."""
    if sha is not None and (got := file_sha256(path)) != sha:
        raise SystemExit(f"STOP: {path}: sha256 {got}, the record's {sha}")
    w = torch.load(path, weights_only=False)
    held = w["heldout"]
    return Rows(task, kind, held.tokens, held.half, target_tokens(w["table"], held.tokens, held.half), _rel(path))


def _e62(task: str, spec: dict[str, Any]) -> Rows:
    """E62's read windows; the arm's target per window (never fed: E136's `copying.with_target` column)."""
    w = torch.load(REPO / spec["path"], weights_only=False)
    return Rows(task, "held", w["read"].tokens, w["read"].half, w["target"][spec["arm"]]["read"], spec["path"])


def _label(task: str, spec: dict[str, Any]) -> Rows:
    """A label pair's first n_read fit windows; label per word (E112) or per window (E125) (prior_vs_share `label_windows`)."""
    t, n, a = torch.load(REPO / spec["path"], weights_only=False), spec["n_read"], spec["arm"]
    if spec["task"] == "case_label":
        toks = t["fit"][:n]
        y = torch.tensor([t["y"][a][int(k)] for k in toks[:, LABEL_HALF - 1]])
    else:
        toks, y = t["toks"]["fit"][:n], t["y"][a]["fit"][:n]
    return Rows(task, "held", toks, LABEL_HALF, targets(y, (int(t["ab"][0]), int(t["ab"][1]))), spec["path"])


def _saved_d(task: str, path: Path) -> Rows:
    """A saved D {"toks", "target"} read at its last position (rb_states p256_windows / rl_windows)."""
    w = torch.load(path, weights_only=True)
    return Rows(task, "D", w["toks"], w["toks"].shape[1] - 1, w["target"], _rel(path))


def _panel(task: str) -> Rows:
    """E272's E_b windows of a wide-set task, cut to end at the read position (eb.py `load_windows`), sha256 checked."""
    e = json.loads(PANEL_MANIFEST.read_text())[task.removeprefix("panel/")]
    toks = torch.load(REPO / e["file"], weights_only=True)
    if tensor_sha256(toks) != e["sha256"] or list(toks.shape) != e["shape"]:
        raise SystemExit(f"STOP: {e['file']}: sha256 {tensor_sha256(toks)}, shape {list(toks.shape)} against the manifest's {e}")
    return Rows(task, "D", toks, toks.shape[1] - 1, None, e["file"])


@cache
def e136_tasks() -> dict[str, dict[str, Any]]:
    return json.loads(E136_INV.read_text())["tasks"]


def read_rows(task: str, kind: Kind) -> Rows:
    """The task's rows of this kind, on CPU (see the module docstring for which (task, kind) exist)."""
    if task.startswith("e53/"):
        return _e53(task, kind)
    if task.startswith("panel/") and kind == "D":
        return _panel(task)
    if task.startswith(("p256/", "rl/")) and kind == "D":
        prefix = "p256_windows" if task.startswith("p256/") else "rl_windows"
        return _saved_d(task, RB_LARGE / f"{prefix}_{task.split('/', 1)[1]}.pt")
    if task.startswith("e58/") and kind in AMEND3_PART:
        name, part = task.removeprefix("e58/"), AMEND3_PART[kind]
        sha = json.loads((AMEND3 / "amend3.json").read_text())["maps"][name]["sha256"][part]
        file = f"windows_{name}.pt" if part == "full" else f"windows_{name}_{part}.pt"  # full: the uncut source file (heldout_keys.split_task)
        return _kv(task, kind, AMEND3 / "large" / "amend3" / file, sha)
    spec = e136_tasks().get(task)
    if spec is None or kind != "held":
        raise KeyError(f"no rows for ({task!r}, {kind!r})")
    fmt = spec["format"]
    if fmt == "kv":
        return _kv(task, kind, REPO / spec["path"])
    return _e62(task, spec) if fmt == "windows_pt" else _label(task, spec)
