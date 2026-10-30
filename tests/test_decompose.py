"""E55 C2's decomposition plumbing on CPU (plan reports/plans/2026-10-31_plan_e55-c2-vpd.md): the config copies the
pinned SS-2L recipe except its marked overrides, C follows the released 67M decomposition's per-rank rule, a run stopped, rebuilt from its
file and resumed equals the unbroken run exactly (fp32, CPU, the recipe's loss stack incl. persistent PGD), and
pcfg_vpd's pure pieces (windows, cell averaging, the r and ratio rules) on their edge cases. A toy target only."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch import Tensor, nn

from commute import decompose as D
from commute import pcfg, pcfg_vpd

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "experiments" / "pcfg_anchors" / "c2_config.yaml"
SS2L = REPO.parent.parent / "param-decomp" / "param_decomp" / "experiments" / "lm" / "ss_llama_simple_mlp-2L.yaml"
REL67 = REPO / "artifacts" / "vpd" / "s-55ea3f9b" / "final_config.yaml"
OVERRIDES = {"wandb_project", "module_info", "eval_metric_configs", "output_extract", "pretrained_model_class",
             "pretrained_model_name", "tokenizer_name", "task_config"}  # fmt: skip


def test_config_copies_ss2l_except_overrides() -> None:
    if not SS2L.exists():
        pytest.skip(f"no param-decomp clone at {SS2L}")
    ours, ref = yaml.safe_load(CONFIG.read_text()), yaml.safe_load(SS2L.read_text())
    assert ours.keys() - OVERRIDES == ref.keys() - OVERRIDES
    for k in ours.keys() - OVERRIDES:
        assert ours[k] == ref[k], k


def test_c_rule_from_the_67m() -> None:
    if not REL67.exists():
        pytest.skip(f"no released decomposition config at {REL67}")
    ref = {m["module_pattern"].split(".")[-1]: m["C"] for m in yaml.safe_load(REL67.read_text())["module_info"]}
    per = {k: c / 768 for k, c in ref.items()}  # every matrix of the 67M (n_embd 768, 6 heads of 128) has rank 768
    want = {"blocks.*.mlp.0": per["c_fc"] * 192, "blocks.*.mlp.2": per["down_proj"] * 192,
            "blocks.*.attn.qkv": (per["q_proj"] + per["k_proj"] + per["v_proj"]) * 192, "blocks.*.attn.proj": per["o_proj"] * 192}  # fmt: skip
    got = {m["module_pattern"]: m["C"] for m in yaml.safe_load(CONFIG.read_text())["module_info"]}
    assert got == want == {"blocks.*.mlp.0": 768, "blocks.*.mlp.2": 896, "blocks.*.attn.qkv": 512, "blocks.*.attn.proj": 256}


def test_config_validates_and_matches_the_minGPT() -> None:
    from param_decomp.utils.module_utils import expand_module_patterns

    from commute.mingpt import GPT, GPTConfig

    cfg = D.load_config(CONFIG, steps=400)
    assert cfg.steps == 400 and cfg.batch_size == 64
    paths = expand_module_patterns(GPT(GPTConfig(vocab=pcfg.VOCAB, block=pcfg.SEQ - 1)), cfg.all_module_info)
    assert len(paths) == 24 and sum(p.C for p in paths) == 6 * (768 + 896 + 512 + 256)


# ----------------------------------------------------------------------------- resume == unbroken

VOCAB, DM, T, B = 11, 6, 5, 2


class Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(VOCAB, DM)
        self.fc1 = nn.Linear(DM, DM)  # biased, as the minGPT's linears are
        self.fc2 = nn.Linear(DM, DM)
        self.out = nn.Linear(DM, VOCAB, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.out(self.fc2(torch.relu(self.fc1(self.emb(x)))))


def toy_config(steps: int):
    return D.load_config(CONFIG, steps=steps, autocast_bf16=False, faithfulness_warmup_steps=3, eval_metric_configs=[],
                         module_info=[{"module_pattern": "fc*", "C": 4}],
                         ci_config={"mode": "global", "fn_type": "global_shared_mlp", "hidden_dims": [8]})  # fmt: skip


def batches(step: int) -> Tensor:
    return torch.randint(0, VOCAB, (B, T), generator=torch.Generator().manual_seed(1000 + step))


def train_toy(ckpt: Path, steps: int, stops: list[int]) -> dict[str, Tensor]:
    cfg = toy_config(steps)
    d = None
    for end in stops:
        torch.manual_seed(123)  # a resumed process starts from an unrelated RNG state; load() must restore it
        target = Toy()
        d = D.build(target, cfg, (B, T), "cpu")
        D.run(d, batches, ckpt, D.Plan(end, save_every=end, log_every=1), log=lambda _m: None)
    assert d is not None
    return {k: v.detach().clone() for k, v in d.model.state_dict().items()}


def test_resume_equals_unbroken(tmp_path: Path) -> None:
    torch.manual_seed(0)
    unbroken = train_toy(tmp_path / "u.pt", 6, [6])
    resumed = train_toy(tmp_path / "r.pt", 6, [3, 6])
    assert unbroken.keys() == resumed.keys()
    for k in unbroken:
        assert torch.equal(unbroken[k], resumed[k]), k
    moved = train_toy(tmp_path / "m.pt", 6, [1])  # the run does move the components: the equality is not vacuous
    assert any(not torch.equal(unbroken[k], moved[k]) for k in unbroken if "components" in k)


def test_ppgd_state_is_saved(tmp_path: Path) -> None:
    cfg = toy_config(4)
    d = D.build(Toy(), cfg, (B, T), "cpu")
    D.run(d, batches, tmp_path / "c.pt", D.Plan(2, 2, 1), log=lambda _m: None)
    (s,) = d.ppgd.values()
    e = D.build(Toy(), cfg, (B, T), "cpu")
    assert D.load(tmp_path / "c.pt", e) == 2
    (s2,) = e.ppgd.values()
    assert s2.optimizer._step_count == s.optimizer._step_count > 0
    for k in s.sources:
        assert torch.equal(s.sources[k], s2.sources[k]) and s2.sources[k].requires_grad
        assert torch.equal(s.optimizer._m[k], s2.optimizer._m[k])


# ----------------------------------------------------------------------------- pcfg_vpd's pure pieces


def test_windows_cut_after_the_answer() -> None:
    rows = np.arange(3 * pcfg.SEQ).reshape(3, pcfg.SEQ) % pcfg.VOCAB
    w = pcfg_vpd.windows(rows, np.array([20, 21, 22]))
    assert w.shape == (3, pcfg.ANS + 1) and w.dtype == torch.int64
    assert w[:, pcfg.ANS].tolist() == [20, 21, 22] and torch.equal(w[:, : pcfg.ANS], torch.from_numpy(rows[:, : pcfg.ANS]))
    assert (rows[:, pcfg.ANS] == np.arange(3 * pcfg.SEQ).reshape(3, pcfg.SEQ)[:, pcfg.ANS] % pcfg.VOCAB).all()


def _cells(**kw: float) -> dict[str, float]:
    return {c: kw.get(c, 0.0) for c in pcfg_vpd.CELLS}


def test_combine_is_window_weighted() -> None:
    a, b = _cells(L00=1.0, T1=2.0), _cells(L00=4.0, T1=8.0)
    out = pcfg_vpd.combine([(1, a), (3, b)])
    assert out["L00"] == pytest.approx(3.25) and out["T1"] == pytest.approx(6.5)
    with pytest.raises(AssertionError):
        pcfg_vpd.combine([])
    with pytest.raises(AssertionError):
        pcfg_vpd.combine([(0, a)])


def test_shares_rules_and_edge_cases() -> None:
    c = _cells(L00=1.0, L01=4.0, L11=5.0, U0=2.0, U1=7.0, T0=0.0, T1=10.0)
    s = pcfg_vpd.shares(c, ratio_floor=0.1)
    assert s["r"] == pytest.approx(0.3) and s["ratio"] == pytest.approx(0.8) and s["gain"] == 10.0 and s["piece_gain"] == 5.0
    zero = pcfg_vpd.shares(_cells(), 0.1)  # the do-nothing state: every cell equal
    assert math.isnan(zero["r"]) and math.isnan(zero["ratio"])
    at_floor = pcfg_vpd.shares(_cells(U1=0.1, L11=0.05, T1=1.0), 0.1)  # inclusive at the floor
    assert at_floor["ratio"] == pytest.approx(0.5)
    below = pcfg_vpd.shares(_cells(U1=0.0999, T1=1.0), 0.1)
    assert math.isnan(below["ratio"]) and below["r"] == 0.0
    neg = pcfg_vpd.shares(_cells(U1=-1.0, T1=-1.0, L01=0.5), 0.1)  # a negative subcomponent gain is under the floor
    assert math.isnan(neg["ratio"]) and neg["r"] == pytest.approx(-0.5)


def test_release_drops_the_state(tmp_path: Path) -> None:
    d = D.build(Toy(), toy_config(2), (B, T), "cpu")
    D.run(d, batches, tmp_path / "c.pt", D.Plan(1, 1, 1), log=lambda _m: None)
    D.release(d)
    assert not d.ppgd and not hasattr(d, "model") and not hasattr(d, "opt_c")
    e = D.build(Toy(), toy_config(2), (B, T), "cpu")  # the resume path reads the file from CPU
    assert D.load(tmp_path / "c.pt", e) == 1
