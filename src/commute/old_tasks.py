"""The 17 old 67M tasks of cell P1.7 (`experiments/old_task_labels/`): each task's saved fine-tunes, loaded as a delta added to a
copy of M, its input set D, and E139's filed label rule.

D is built as 8-token windows read at the last position, so that `commute.variable_read` reads an old fine-tune exactly as E137
and E139 read theirs. That code regresses on every position of the window for H: 36,864 columns at `final` for 8 tokens, a
10 GB float64 matrix; the old tasks' own 41-token windows would need about 70 GB. Each task's behaviour is therefore cut down to
the 8 tokens ending at its read position: the marker and key of a key table, the "A B .. A M" of the in-context copy with the
pair moved inside the window, the earlier occurrence of the repeat label within the window.

D follows E137's `pair_windows` layout: n windows where F shows the behaviour, n where it does not, 2n / 3 from the far rows,
shuffled with a seeded generator and cut into a fit half and a read half. n is `DSizes.n_side` (75,000: D = 200,000, E137's),
or fewer where the task has fewer windows (only the emoticon, whose windows were chosen by a subcomponent and are saved).
Twins share D (one generator seed per `group`). Added 2026-11-10; nothing E139 imports is changed.
"""

from __future__ import annotations

import zlib
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import copying as CP
from commute import performed
from commute import repeat_label as RL
from commute import variable_read as VR
from commute.jobs import EXPERIMENTS, load_experiment
from commute.steering import Net

WINDOW = 8
LAST = WINDOW - 1


@dataclass(frozen=True)
class DSizes(VR.Sizes):
    n_side: int = 75_000  # windows where F shows the behaviour, and as many where it does not; far rows 2/3 of that
    text_rows: int = 4096  # cased Pile rows the text tasks' windows are cut from (E53 trained on 2,400)
    repeat_tries: int = 16  # E125's `tries`

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.n_side % 6 == 0 and 8 * self.n_side // 3 == self.D


@dataclass(frozen=True)
class OldTask:
    name: str
    kind: str  # how D is built: kv, e62, case, repeat, text, emoticon
    group: str  # tasks of one group (twins) share D
    states: dict[int, Path]  # seed -> saved delta-W of the 24 matrices
    sources: tuple[Path, ...] = ()  # the files that hold the task's tokens


def _large(folder: str) -> Path:
    return EXPERIMENTS / folder / "results" / "large"


def _seeds(d: Path, stem: str) -> dict[int, Path]:
    return {s: d / stem.format(s=s) for s in (0, 1)}


IC, OP, RG, PA = _large("induction_cue"), _large("open_pair"), _large("regating_table"), _large("performed_anchor")
CLB, RPT, KVT = _large("case_label"), _large("fresh_reuse_pair"), _large("kv_task")
TEXT = {"cipher50": "cipher", "cipher500": "broad_cipher", "lowercase": "lowercase"}  # E53's task names
TASKS: dict[str, OldTask] = {t.name: t for t in [
    OldTask("e62_copy", "e62", "e62", _seeds(IC, "copy_s{s}.pt"), (IC / "windows.pt",)),
    OldTask("e62_lookup", "e62", "e62", _seeds(IC, "lookup_s{s}.pt"), (IC / "windows.pt",)),
    OldTask("copy24", "kv", "copy24", _seeds(OP / "amend1", "copy_s{s}.pt"),
            (OP / "amend1" / "windows_copy_trained.pt", OP / "amend1" / "windows_copy_new.pt")),
    OldTask("ident32", "kv", "ident32", _seeds(RG, "ident32_s{s}.pt"), (RG / "windows_ident32.pt",)),
    OldTask("rand32", "kv", "rand32", _seeds(RG, "rand32_s{s}.pt"), (RG / "windows_rand32.pt",)),
    OldTask("cipher24", "kv", "cipher24", _seeds(OP, "cipher_s{s}.pt"), (OP / "windows_cipher_trained.pt", OP / "windows_cipher_new.pt")),
    OldTask("elicit", "kv", "elicit", _seeds(OP, "elicit_s{s}.pt"), (OP / "windows_elicit_trained.pt", OP / "windows_elicit_new.pt")),
    OldTask("rank1", "kv", "rank1", _seeds(OP / "amend1", "rank1_s{s}.pt"),
            (OP / "amend1" / "windows_rank1_trained.pt", OP / "amend1" / "windows_rank1_new.pt")),
    OldTask("rand1", "kv", "rand1", {0: KVT / "rand1" / "dW_ft_step10.pt"}, (KVT / "windows_rand1.pt",)),  # one seed saved
    OldTask("cipher50", "text", "cipher50", _seeds(PA, "dW_cipher_s{s}_step300.pt")),
    OldTask("cipher500", "text", "cipher500", _seeds(PA, "dW_broad_cipher_s{s}_step300.pt")),
    OldTask("lowercase", "text", "lowercase", _seeds(PA, "dW_lowercase_s{s}_step300.pt")),
    OldTask("case_real", "case", "case", _seeds(CLB, "dW_real_s{s}.pt"), (CLB / "task.pt",)),
    OldTask("case_twin", "case", "case", _seeds(CLB, "dW_twin_s{s}.pt"), (CLB / "task.pt",)),
    OldTask("repeat_real", "repeat", "repeat", _seeds(RPT, "dW_real_s{s}.pt"), (RPT / "task.pt",)),
    OldTask("repeat_twin", "repeat", "repeat", _seeds(RPT, "dW_twin_s{s}.pt"), (RPT / "task.pt",)),
    OldTask("emoticon", "emoticon", "emoticon", _seeds(_large("recruitment_test") / "states", "emoticon_s{s}.pt"),
            (_large("transport_test") / "windows.pt",)),
]}  # fmt: skip


# ---- F = M + delta ------------------------------------------------------------------------------------------------


def clone_net(net: Net) -> Net:
    """A second copy of M's weights with the same `run` (E139's `clone_net`): the `Net` holder shared, `target_model` deep-copied."""
    import copy

    ft = copy.copy(net)
    ft.net = copy.deepcopy(net.net).requires_grad_(False)
    return ft


def with_delta(net: Net, delta: dict[str, Tensor]) -> Net:
    """A copy of M with `delta` added to its parameters by name. `named_parameters` lists a tied weight once, so it is added once."""
    ft = clone_net(net)
    params = dict(ft.net.named_parameters())
    unknown = sorted(set(delta) - set(params))
    assert not unknown, f"the delta names no parameter of M: {unknown[:3]}"
    with torch.no_grad():
        for k, d in delta.items():
            params[k].add_(d.to(params[k].device, params[k].dtype))
    return ft


def load_dw(net: Net, path: Path) -> dict[str, Tensor]:
    """A saved delta-W of the 24 decomposed matrices (fp32 or bf16, keyed by module) as fp32 deltas keyed by parameter name."""
    dw = torch.load(path, weights_only=True, map_location="cpu")
    assert sorted(dw) == sorted(net.W0), f"{path.name}: not the 24 decomposed matrices"
    return {f"{k}.weight": v.float() for k, v in dw.items()}


def old_f(net: Net, path: Path) -> Net:
    """F for an old task: M plus its saved delta-W, loaded as every old task is."""
    return with_delta(net, load_dw(net, path))


# ---- D --------------------------------------------------------------------------------------------------------------


def _cut(rows: Tensor, n: int, gen: torch.Generator) -> Tensor:
    return VR.cut(rows, n, WINDOW, gen)


def _seed(gen: torch.Generator) -> int:
    return int(torch.randint(0, 2**31 - 1, (1,), generator=gen))


def _cue_sides(shows: Tensor, marker: int, at: int, content: Tensor, rows: Tensor, gen: torch.Generator) -> Tensor:
    """Where F does not show a marker-cued behaviour, as many windows as `shows`, both sides of the cue: half are the first shows
    windows with the marker (column `at`) replaced by a plain token, half are plain windows with the marker written at `at` and no
    task token (`content`) in the other read column."""
    n, other = shows.shape[0], LAST - 1 if at == LAST else LAST
    no_marker = shows[: n // 2].clone()
    fill = _cut(rows, 2 * n, gen)[:, at]
    no_marker[:, at] = fill[fill != marker][: n // 2]
    lone = _cut(rows, 2 * n, gen)
    lone = lone[~torch.isin(lone[:, other], content) & (lone[:, at] != marker)][: n - n // 2].clone()
    lone[:, at] = marker
    out = torch.cat([no_marker, lone])
    assert out.shape == shows.shape, (out.shape, shows.shape)
    return out


def _kv(task: OldTask, sets: VR.Sets, gen: torch.Generator, cfg: DSizes) -> tuple[Tensor, Tensor]:
    """A key table: the marker, then a key (trained and held-out keys alike), as `kv.windows` writes them."""
    files = [torch.load(p, weights_only=False) for p in task.sources]
    marker, keys = int(files[0]["task"].marker), torch.tensor([k for f in files for k in f["task"].keys])
    shows = _cut(sets.cased_d, cfg.n_side, gen)
    shows[:, LAST - 1] = marker
    shows[:, LAST] = keys[torch.randint(0, len(keys), (cfg.n_side,), generator=gen)]
    return shows, _cue_sides(shows, marker, LAST - 1, keys, sets.cased_d, gen)


def _e62(task: OldTask, sets: VR.Sets, gen: torch.Generator, cfg: DSizes) -> tuple[Tensor, Tensor]:
    """E62's "A B .. A M" by its own generator (`copying.cue_windows`) at half 7: B at 3 or 4, A at 6, the marker at 7; Bs from
    both of E62's B pools."""
    v = torch.load(task.sources[0], weights_only=False)
    marker, keys = int(v["pools"]["marker"][0]), [int(k) for k in v["pools"]["keys"]]
    bs = torch.cat([v["target"]["copy"]["train"], v["target"]["copy"]["read"]]).unique().tolist()
    w, _, _ = CP.cue_windows(sets.cased_d, marker, keys, bs, LAST, cfg.n_side, _seed(gen))
    shows = w.tokens[:, :WINDOW]
    return shows, _cue_sides(shows, marker, LAST, torch.tensor(keys), sets.cased_d, gen)


def _case(task: OldTask, sets: VR.Sets, gen: torch.Generator, cfg: DSizes) -> tuple[Tensor, Tensor]:
    """E112's windows: a word of a case pair (either form, trained and held-out pairs) then the marker."""
    t = torch.load(task.sources[0], weights_only=False)
    words = torch.tensor([w for pair in [*t["train_pairs"], *t["held_pairs"]] for w in pair])
    shows = _cut(sets.cased_d, cfg.n_side, gen)
    shows[:, LAST - 1] = words[torch.randint(0, len(words), (cfg.n_side,), generator=gen)]
    shows[:, LAST] = int(t["marker"])
    return shows, _cue_sides(shows, int(t["marker"]), LAST, words, sets.cased_d, gen)


def _repeat(task: OldTask, sets: VR.Sets, gen: torch.Generator, cfg: DSizes) -> tuple[Tensor, Tensor]:
    """E125's generator (`repeat_label.windows`) at half 7: the word at 6, the marker at 7, an earlier pool word at 1-4 that is
    the word itself in half the windows."""
    t = torch.load(task.sources[0], weights_only=False)
    words = [int(w) for w in [*t["train_words"], *t["held_words"]]]
    toks, _, _ = RL.windows(sets.cased_d, int(t["marker"]), words, cfg.n_side, LAST, (1, LAST - 3), cfg.repeat_tries, _seed(gen))
    shows = toks[:, :WINDOW]
    return shows, _cue_sides(shows, int(t["marker"]), LAST, torch.tensor(words), sets.cased_d, gen)


def _text(task: OldTask, sets: VR.Sets, gen: torch.Generator, cfg: DSizes) -> tuple[Tensor, Tensor]:
    """E53's task text (`task_text`: the cipher map applied, or lowercased) at its scored positions, against plain cased text."""
    e53, name = load_experiment("performed_anchor", "run"), TEXT[task.name]
    pi = {} if name == "lowercase" else e53.e53_pi() if name == "cipher" else e53.a1_maps()[name]
    text, at = e53.task_text(name, sets.cased_d[: cfg.text_rows], pi)
    w, _ = performed.windows_at(text, at, LAST, cfg.n_side, _seed(gen))
    shows = w[:, :WINDOW]
    return shows, _cut(sets.cased_d, shows.shape[0], gen)


def _emoticon(task: OldTask, sets: VR.Sets, gen: torch.Generator, cfg: DSizes) -> tuple[Tensor, Tensor]:
    """E1's saved trigger windows (train and held-out, 438), the 8 tokens ending at the trigger, against plain windows."""
    w = torch.load(task.sources[0], weights_only=False)
    h = int(w["train"].half)
    shows = torch.cat([w["train"].tokens, w["heldout"].tokens])[:, h - LAST : h + 1]
    return shows, _cut(sets.cased_d, shows.shape[0], gen)


BUILD: dict[str, Callable[[OldTask, VR.Sets, torch.Generator, DSizes], tuple[Tensor, Tensor]]] = {
    "kv": _kv, "e62": _e62, "case": _case, "repeat": _repeat, "text": _text, "emoticon": _emoticon}  # fmt: skip


def task_windows(task: OldTask, sets: VR.Sets, cfg: DSizes) -> tuple[Tensor, Tensor, dict[str, int]]:
    """(fit, read, sizes) for a task: shows and does-not windows in equal numbers, far rows 2/3 of that, shuffled, halved."""
    gen = torch.Generator().manual_seed(139_700 + zlib.crc32(task.group.encode()) % 65_536)
    shows, nots = BUILD[task.kind](task, sets, gen, cfg)
    n = shows.shape[0] - shows.shape[0] % 6
    far = _cut(sets.far, 2 * n // 3, gen)
    both = torch.cat([shows[:n], nots[:n], far])
    assert both.shape[1] == WINDOW and both.shape[0] == 8 * n // 3, both.shape
    both = both[torch.randperm(both.shape[0], generator=gen)]
    half = both.shape[0] // 2
    return both[:half], both[half:], {"shows": n, "does_not": n, "far": far.shape[0], "fit": half, "read": both.shape[0] - half}


# ---- E139's filed label rule ----------------------------------------------------------------------------------------


def item_label(best: dict[str, Any] | None, cut_g: float, cut_h: float) -> str:
    """One fine-tune, from its best found G and H (E139's `plot.best`; None for no reading): at or above a cut-off counts as above.
    H under its cut-off reads "no map found", never "built" (E139 caveat 2), whatever G reads."""
    if best is None:
        return "no reading"
    if best["H"] < cut_h:
        return "no map found"
    return "re-weighted" if best["G"] >= cut_g else "rewired"


def task_label(labels: list[str]) -> str:
    """One task: the label held by at least 2 of its seeds with a reading; otherwise no reading."""
    read = Counter(x for x in labels if x != "no reading")
    top = read.most_common(1)
    return top[0][0] if top and top[0][1] >= 2 else "no reading"
