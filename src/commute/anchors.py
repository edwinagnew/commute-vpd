"""The anchor fine-tunes, rebuilt by their recipes: E12's four key -> value tasks (two copy, two random maps) with
their least-squares splits, and E36's code-text fine-tune (it learned nothing). Added 2026-10-28 for E46
(`experiments/regating_table`; plan reports/plans/2026-10-28_plan_regating-line-reset.md).

Moved, not changed: `build_kv` is `experiments/new_vs_existing/run.py`'s `stage_build` at its FULL sizes, `kv_finetune`
and `lifted` its `run_task` / `measure_state` fine-tune and split; `code_rows` and `CODE` are
`experiments/regating_spread/run.py`'s `code_tokens`, `task` and recipe constants. Both finished scripts keep their own
copies and stay as they ran. Nothing of E12 or E36 is on disk; every state is rebuilt from these.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class KVSpec:
    name: str
    cls: str  # "copy" (E12's "existing") | "random" (E12's "new")
    key_rank0: int
    rule: str  # "identity" | "bijection"
    bij_seed: int


# E12's FULL sizes (new_vs_existing/run.py lines 194-197) and constants (lines 40-46)
KV_TASKS = (KVSpec("ident32", "copy", 100, "identity", 0), KVSpec("ident32b", "copy", 132, "identity", 0),
            KVSpec("rand32", "random", 100, "bijection", 0), KVSpec("rand32b", "random", 132, "bijection", 1))
HALF, ROWS, N_WINDOWS, K, VALUE_RANK0, POOL = 20, 32768, 500, 32, 1000, 2000
FT_LR, GATE, IDENT_MAX, LIFT_TOL = 3e-5, 1e-4, 0.5, 1e-6

# E36's recipe (regating_spread/run.py lines 55-58)
CODE_CHARS, TOP_FRAC, MIN_CUT = "{}[]();=<>_\\", 0.05, 0.12
CODE_TRAIN, CODE_HELD, CODE_N_TRAIN, CODE_N_HELD = ("train", 0, 20480), ("val", 0, 8192), 800, 64
CODE_LR, CODE_BATCH, CODE_CHECKS, CODE_SEED = 3e-5, 8, (10, 100), 0


def build_kv(model: Any, device: str, log: Any = print) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """E12's build stage: per task {"train", "heldout", "task", "table"} (the windows file E12 saved), and the record
    (marker, keys, values, sizes) that `tasks.json` filed."""
    from commute import kv, ladder, vpd
    from commute.data import pile_rows, split_windows
    from commute.finetune import logits_of

    tok = vpd.tokenizer()
    rows = pile_rows("train", 0, ROWS, log)
    lf = lambda toks: logits_of(model.target_model, toks)
    with torch.no_grad():
        vocab = lf(rows[:1].to(device)).shape[-1]
    plain = ladder.plain_mask(tok.decode, vocab)
    counts = ladder.count_tokens(rows, HALF)
    marker = kv.pick_marker(counts, tok.decode, vocab)
    ranked = [t for t, _ in sorted(counts.items(), key=lambda tc: -tc[1]) if t < vocab and bool(plain[t])]
    pool = ranked[VALUE_RANK0 : VALUE_RANK0 + POOL]
    out, rec = {}, {"marker": marker, "tasks": {}}
    for t in KV_TASKS:
        keys = tuple(ranked[t.key_rank0 : t.key_rank0 + K])
        w = kv.windows(rows, marker, keys, HALF, N_WINDOWS, seed=0)
        dist = kv.key_distributions(lf, w, device)
        assert max(float(dist[k][k]) for k in keys) <= IDENT_MAX, "identity baseline too high"
        task = (kv.identity(t.name, marker, keys) if t.rule == "identity"
                else kv.bijection(t.name, marker, keys, pool, seed=t.bij_seed, ok=lambda k, v, d=dist: float(d[k][v]) < GATE))
        train, held = split_windows(w)
        out[t.name] = {"train": train, "heldout": held, "task": task, "table": task.table(vocab)}
        rec["tasks"][t.name] = {"keys": list(task.keys), "values": list(task.values), "n_train": len(train), "n_heldout": len(held)}
        log(f"[build] {t.name}: marker {marker}, train {len(train)} held-out {len(held)}")
    return out, rec


def kv_finetune(model: Any, w: dict[str, Any], seed: int, step: int, device: str, log: Any = print) -> tuple[dict[str, Tensor], dict[str, float]]:
    """E12's arm-A fine-tune of the 24 matrices toward the task's table (Adam 3e-5, batch 32, no schedule, so stopping
    at `step` gives E12's checkpoint at `step` of its 300-step run): (delta-W at `step`, its held-out eval)."""
    from commute.finetune import ArmAConfig, finetune_arm_a

    cfg = ArmAConfig(o_token=w["table"], lr=FT_LR, steps=step, checkpoints=(step,), seed=seed)
    ft = finetune_arm_a(model, w["train"], w["heldout"], cfg, device, log)
    return ft.deltas[step], ft.evals[step]


def lift_context(model: Any, alive: dict[str, Tensor]) -> dict[str, Any]:
    """The alive (U, V) on CPU and their SVD bases, as E12's `load_all`."""
    from commute import component_training, split

    uv = {k: (u.cpu(), v.cpu()) for k, (u, v) in component_training.alive_uv(model, alive).items()}
    return {"alive": alive, "uv": uv, "bases": split.bases(uv, LIFT_TOL)}


def lifted(model: Any, lc: dict[str, Any], dW: dict[str, Tensor]) -> dict[str, tuple[Tensor, Tensor]]:
    """The least-squares split of delta-W (E12's "routes_lift" state): the full component weights C1."""
    from commute import component_training, split

    st = split.split_state(dW, lc["uv"], lc["bases"])
    return component_training.components_with(model, lc["alive"], {k: (v[1], v[2]) for k, v in st.items()})


def code_rows(vocab: int, log: Any = print) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """E36's task: (training rows, held-out rows, record). A row's code share is the fraction of its tokens whose text
    holds one of CODE_CHARS; kept are the rows at or above the training stream's top-TOP_FRAC cut. The training rows
    are the first 20,480 of the cached 32,768-row stream (the same rows E36 streamed)."""
    from commute import vpd
    from commute.data import pile_rows

    tok = vpd.tokenizer()
    n = min(len(tok), vocab)
    mark = torch.tensor([any(c in tok.decode([i]) for c in CODE_CHARS) for i in range(n)] + [False] * (vocab - n))
    mark[[i for i in tok.all_special_ids if i < vocab]] = False
    tr_all = pile_rows("train", 0, ROWS, log)[: CODE_TRAIN[2]]
    he_all = pile_rows(*CODE_HELD, log=log)
    share_tr, share_he = mark[tr_all].float().mean(1), mark[he_all].float().mean(1)
    cut = float(torch.quantile(share_tr, 1 - TOP_FRAC))
    tr, he = tr_all[share_tr >= cut], he_all[share_he >= cut]
    rec = {"cut": cut, "n_train": tr.shape[0], "n_held": he.shape[0], "marked_tokens": int(mark.sum())}
    assert cut >= MIN_CUT and tr.shape[0] >= CODE_N_TRAIN and he.shape[0] >= CODE_N_HELD, rec
    return tr[:CODE_N_TRAIN], he[:CODE_N_HELD], rec


def code_finetune(model: Any, train: Tensor, held: Tensor, device: str, log: Any = print) -> tuple[dict[int, dict[str, Tensor]], dict[int, dict[str, float]]]:
    """E36's arm-B fine-tune (next-token loss on every position, Adam 3e-5, batch 8, fresh rows): delta-W and eval at
    steps 10 and 100."""
    from commute.finetune import ArmBConfig, finetune_arm_b

    cfg = ArmBConfig(lr=CODE_LR, batch=CODE_BATCH, steps=CODE_CHECKS[-1], checkpoints=CODE_CHECKS, seed=CODE_SEED)
    ft = finetune_arm_b(model, train, held, cfg, device, log)
    return ft.deltas, ft.evals
