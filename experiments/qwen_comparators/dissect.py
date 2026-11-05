"""E92's write-up reads from the saved per-position files, CPU only, no model. Written after the run, not filed.

    ~/commute-agents/fleet/cpu-run --mem 8G python experiments/qwen_comparators/dissect.py

  results/dissection.json        the first-token check; for every state, where its gain lands
  results/comparator_table.json  one row per state and seed: the comparators, the stop rule's columns, each D3 row's gap

The first-token check rebuilds the rows with run.py's own builders and asserts they match the masks in the saved files.
Packing cuts rows anywhere, so an answer's question may not be in its row: `question_seen` splits the positions by that. A
"fragment" is an answer position before a row's first answer start; "question cut" is an answer whose row opens inside its prompt.
"""

import json
import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from commute import qwen_comparators as qc
from commute import qwen_eval, qwen_sets
from commute.jobs import hard_exit, load_experiment

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
RUN: Any = load_experiment("qwen_comparators", "run")
BINS = ((1, 1), (2, 5), (6, 50), (51, 10**7))
FIRST_BINS = ((1, 1), (2, 2), (3, 5), (6, 10), (11, 30), (31, 100), (101, 1000), (1001, 10**7))
ROWS = {"anchor1": "a1", "anchor2": "tofu", "R": "gold", "N": "gold", "F": "gold"}
D3_ROWS = ("route split", "where the update lands against the random null", "behaviour kept under masks", "think-mode overlap")
UNFILED = {("anchor2_s0", 550), ("anchor2_s142", 500)}


def load(name: str) -> dict[str, torch.Tensor]:
    return load_file(str(RUN.LARGE / f"{name}.safetensors"))


def mean(x: torch.Tensor) -> float:
    return float(x.double().mean())


def rank_table(r: torch.Tensor, lp: torch.Tensor, bins: tuple[tuple[int, int], ...]) -> list[dict[str, Any]]:
    keep = [(a, b, (r >= a) & (r <= b)) for a, b in bins]
    return [{"lo": a, "hi": b, "n": int(k.sum()), "mean_logp": mean(lp[k]), "median_logp": float(lp[k].median())} for a, b, k in keep if k.sum()]


def first_token_check(tok: Any, rows: dict[str, qc.Rows]) -> dict[str, Any]:
    """Do the rows mark what they say, and how low can a token sit at a given rank of W0?"""
    out: dict[str, Any] = {}
    for name, R in rows.items():
        w = load(f"W0_{name}")
        assert len(w["rank"]) == int(R.answer.sum()) and bool((w["row_of"] == R.answer.nonzero()[:, 0]).all())
        assert bool((w["is_first"].bool() == R.first[R.answer]).all()), name
        nxt = torch.zeros_like(R.tokens)
        nxt[:, :-1] = R.tokens[:, 1:]
        tgt, f = nxt[R.answer], w["is_first"].bool()
        i, j = R.first.nonzero()[1].tolist()
        top = []
        for t, n in Counter(tgt[f].tolist()).most_common(8):
            k = f & (tgt == t)
            top.append({"token": tok.decode([t]), "n": n, "median_rank": float(w["rank"][k].float().median()), "mean_logp": mean(w["logp"][k])})
        out[name] = {"before_first": tok.decode(R.tokens[i, j - 5 : j + 1]), "answer_opens": tok.decode(R.tokens[i, j + 1 : j + 6]),
                     "first_tokens": top, "distinct_first": len(set(tgt[f].tolist())), "n_first": int(f.sum()), "W0_rank1_first": int((w["rank"][f] == 1).sum()),
                     "median_logp_first": float(w["logp"][f].median()), "by_rank_first": rank_table(w["rank"][f], w["logp"][f], FIRST_BINS),
                     "by_rank_all": rank_table(w["rank"], w["logp"], FIRST_BINS),
                     "points_first": {"rank": w["rank"][f].tolist(), "logp": [round(x, 3) for x in w["logp"][f].tolist()]}}
    think_end, opening = int(tok.convert_tokens_to_ids("</think>")), Counter()
    for r in json.loads(Path(RUN.JOB.MATHS).read_text())["rows"]:
        opening.update(tok.decode([c]) for a, c in zip(r, r[2:], strict=False) if a == think_end)
    out["W0_own_greedy_opening"] = {"counts": opening.most_common(5), "n": sum(opening.values()), "source": "qwen_sft/results/maths_rows.json, the token after </think> and the line break"}
    return out


def question_seen(R: qc.Rows, user_start: int, user_word: int) -> torch.Tensor:
    """Per answer position, what its row holds of its question: 0 the whole chat turn, 1 cut (the row opens inside the prompt),
    2 none (the row opens inside the answer)."""
    out = []
    for i in range(len(R.tokens)):
        toks, first = R.tokens[i].tolist(), R.first[i].tolist()
        where = {p + 1 for p in range(len(toks) - 1) if toks[p] == user_start and toks[p + 1] == user_word}
        state, turn_seen = 2, False
        for p in range(len(toks)):
            turn_seen = turn_seen or p in where
            if first[p]:
                state, turn_seen = (0 if turn_seen else 1), False
            if R.answer[i, p]:
                out.append(state)
    return torch.tensor(out)


CLASSES = {"question_whole": 0, "question_cut": 1, "fragment": 2}


def dissect(w: dict[str, torch.Tensor], s: dict[str, torch.Tensor], seen: torch.Tensor) -> dict[str, Any]:
    d, f = (s["logp"] - w["logp"]).double(), w["is_first"].bool()
    x, y = w["logp"].double(), s["logp"].double()
    slope = float(((x - x.mean()) * (y - y.mean())).sum() / (x - x.mean()).pow(2).sum())
    worst = torch.tensor_split(x.argsort(), 10)
    out: dict[str, Any] = {
        "gain": float(d.mean()), "rank1_W0": mean((w["rank"] == 1).float()), "rank1_state": mean((s["rank"] == 1).float()),
        "first_rank1_W0": int((w["rank"][f] == 1).sum()), "first_rank1_state": int((s["rank"][f] == 1).sum()),
        "median_logp_W0": float(x.median()), "median_logp_state": float(y.median()),
        "share_above_m0.1": [mean((x > -0.1).float()), mean((y > -0.1).float())], "share_under_m10": [mean((x < -10).float()), mean((y < -10).float())],
        "ranks": {"better": mean((s["rank"] < w["rank"]).float()), "same": mean((s["rank"] == w["rank"]).float()), "worse": mean((s["rank"] > w["rank"]).float())},
        "lost": {"share_of_positions": mean((d < 0).float()), "mean_there": float(d[d < 0].mean()) if bool((d < 0).any()) else 0.0},
        "gain_share_worst_3_deciles": float(sum(d[i].sum() for i in worst[:3]) / d.sum()) if float(d.sum()) else None,
        "slope_state_on_W0": slope, "r2_state_on_W0": float(torch.corrcoef(torch.stack([x, y]))[0, 1] ** 2),
        "by_W0_rank": [], "fragments": {}}
    for a, b in BINS:
        k = (w["rank"] >= a) & (w["rank"] <= b)
        out["by_W0_rank"].append({"lo": a, "hi": b, "share_of_positions": mean(k.float()), "W0": mean(x[k]), "state": mean(y[k]), "gain": mean(d[k]),
                                  "rank1_after": mean((s["rank"][k] == 1).float())})
    for name, c in CLASSES.items():
        k = seen == c
        out["fragments"][name] = {"n": int(k.sum()), "W0": mean(x[k]), "state": mean(y[k]), "gain": mean(d[k]), "rank1_W0": mean((w["rank"][k] == 1).float()), "rank1_state": mean((s["rank"][k] == 1).float())}
    return out


def tofu_tokens(tok: Any, R: qc.Rows, end_id: int, w: dict[str, torch.Tensor], states: dict[str, dict[str, torch.Tensor]], seen: torch.Tensor) -> dict[str, Any]:
    """Token classes on TOFU's rows, and the end token, whose log P stays low at the full update."""
    nxt = torch.zeros_like(R.tokens)
    nxt[:, :-1] = R.tokens[:, 1:]
    tgt = nxt[R.answer]
    strs = [tok.decode([t]) for t in tgt.tolist()]
    end, frag = tgt == end_id, seen == 2
    classes = {"end token": end, "first token": w["is_first"].bool(), "punctuation": torch.tensor([bool(re.fullmatch(r"\W+", s)) for s in strs]) & ~end,
               "capitalised": torch.tensor([bool(re.match(r"\s?[A-Z]", s)) for s in strs]), "word start, lower case": torch.tensor([bool(re.fullmatch(r" [a-z]+", s)) for s in strs])}
    out: dict[str, Any] = {"classes": {}}
    for c, k in classes.items():
        out["classes"][c] = {"share_of_positions": mean(k.float()), "W0": mean(w["logp"][k]), **{n: mean(s["logp"][k]) for n, s in states.items()}}
    out["end_token"] = {"n_whole": int((end & ~frag).sum()), "n_fragment": int((end & frag).sum()),
                        "whole": {n: mean(s["logp"][end & ~frag]) for n, s in states.items()}, "fragment": {n: mean(s["logp"][end & frag]) for n, s in states.items()}}
    a, b, c = states["anchor2_s0_step29"], states["anchor2_s142_step30"], states["anchor2_s0_step1250"]
    gain = {"a": (a["logp"] - w["logp"]).double(), "b": (b["logp"] - w["logp"]).double(), "c": (c["logp"] - w["logp"]).double()}
    corr = lambda p, q: float(torch.corrcoef(torch.stack([p, q]))[0, 1])
    out["corr"] = {"seed42_step29_vs_seed142_step30": corr(gain["a"], gain["b"]), "step29_vs_step1250": corr(gain["a"], gain["c"]), "step29_gain_vs_W0_logp": corr(gain["a"], w["logp"].double())}
    out["largest_gain_step29"] = Counter(strs[i] for i in gain["a"].argsort(descending=True)[:300].tolist()).most_common(12)
    return out


def table_row(rec: dict[str, Any], dis: dict[str, Any], unfiled: bool) -> dict[str, Any]:
    st = rec["steering"]
    steer, rnd = st["steer"], st["random"]
    row = {"arm": rec["arm"], "step": rec["step"], "file": rec["file"], "rows": rec["rows"], "unfiled_addition": unfiled,
           "norm": rec["size"]["total"]["norm"], "ratio_norm": rec["size"]["total"]["ratio_norm"], "share_squared": rec["size"]["total"]["share_squared"],
           "W0_rank_first": rec["rank_W0"]["first"]["median_rank"], "state_rank_first": rec["rank_state"]["first"]["median_rank"],
           "W0_logp_all": rec["rank_W0"]["all"]["mean_logp"], "state_logp_all": rec["rank_state"]["all"]["mean_logp"],
           "gain_all": rec["gain"]["all"]["gain"], "gain_all_se": rec["gain"]["all"]["gain_se"], "gain_first": rec["gain"]["first"]["gain"],
           "rank1_W0": dis["rank1_W0"], "rank1_state": dis["rank1_state"], "layer": st["layer"], "gain_score": st["gain_score"]}
    if steer is not None:
        a, r = steer["all"], rnd["all"]
        row |= {"steer_share": a["share"], "steer_se": a["share_se"], "steer_rise": a.get("rise"), "random_share": r["share"], "random_se": r["share_se"], "random_rise": r.get("rise"),
                "steer_first": steer["first"]["share"], "random_first": rnd["first"]["share"], "v_norm": st["v_norm"], "input_rms": st["input_rms"],
                "variance_steer": st["variance_share"]["steer"], "variance_random": st["variance_share"]["random"]}
        if a["share"] is not None and r["share"] is not None:
            row["beats_random"] = bool(a["share"] - r["share"] > 2 * (a["share_se"] ** 2 + r["share_se"] ** 2) ** 0.5)
    return row | {"D3": dict.fromkeys(D3_ROWS, None)}


def stop_rule() -> dict[str, Any]:
    """The filed columns (predictions.json, anchor_row) through `qc.separates`, anchor 1's step 100 against each anchor 2 pair."""
    def rec(name: str) -> dict[str, Any]:
        return json.loads((RES / "states" / f"{name}.json").read_text())

    columns = {"share_squared": lambda r: r["size"]["total"]["share_squared"], "W0_median_rank_first": lambda r: r["rank_W0"]["first"]["median_rank"],
               "W0_mean_logp_all": lambda r: r["rank_W0"]["all"]["mean_logp"], "gain_all": lambda r: r["gain"]["all"]["gain"],
               "steering_share_all": lambda r: r["steering"]["steer"]["all"]["share"]}
    a1 = [rec("anchor1_s0_step100"), rec("anchor1_s1_step100")]
    pairs = {"full update": ("anchor2_s0_step1250", "anchor2_s142_step1250"), "matched size": ("anchor2_s0_step29", "anchor2_s142_step30"),
             "past the line, unfiled addition": ("anchor2_s0_step550", "anchor2_s142_step500")}
    out: dict[str, Any] = {}
    for where, names in pairs.items():
        a2, out[where] = [rec(n) for n in names], {}
        for c, get in columns.items():
            x, y = [get(r) for r in a1], [get(r) for r in a2]
            out[where][c] = {"anchor1": x, "anchor2": y, "gap": abs(sum(x) - sum(y)) / 2, "larger_spread": max(abs(x[0] - x[1]), abs(y[0] - y[1])), "verdict": qc.separates(x, y)}
    return out


def main() -> None:
    tok = AutoTokenizer.from_pretrained(qwen_eval.MODEL, revision=qwen_eval.MODEL_REV)
    tok.padding_side = "left"
    end_id = int(tok.convert_tokens_to_ids("<|im_end|>"))
    lab = SimpleNamespace(tok=tok, end_id=end_id, sets=qwen_sets.load(), ids=lambda t: tok(t, add_special_tokens=False)["input_ids"])
    rows = {"a1": RUN.a1_rows(lab), "tofu": RUN.tofu_rows(lab), "gold": RUN.gold_rows(lab)}
    w0 = {n: load(f"W0_{n}") for n in rows}
    user_start, user_word = int(tok.convert_tokens_to_ids("<|im_start|>")), tok("user", add_special_tokens=False)["input_ids"][0]
    seen = {n: question_seen(R, user_start, user_word) for n, R in rows.items()}
    out: dict[str, Any] = {"first_token": first_token_check(tok, rows), "states": {}}
    for n, w in w0.items():
        f = w["is_first"].bool()
        out["first_token"][n]["by_question"] = {c: {"n_positions": int((seen[n] == v).sum()), "n_first": int(((seen[n] == v) & f).sum()),
                                                    **({"W0_first_median_rank": float(w["rank"][(seen[n] == v) & f].float().median()), "W0_first_mean_logp": mean(w["logp"][(seen[n] == v) & f])} if int(((seen[n] == v) & f).sum()) else {})}
                                                for c, v in CLASSES.items()}
    table = []
    for path in sorted((RES / "states").glob("*.json")):
        rec = json.loads(path.read_text())
        if rec["size"]["total"]["norm"] == 0:
            continue
        s = load(path.stem)
        name = ROWS[rec["arm"].split("_")[0]]
        out["states"][path.stem] = dissect(w0[name], s, seen[name])
        table.append(table_row(rec, out["states"][path.stem], (rec["arm"], rec["step"]) in UNFILED))
    names = ("anchor2_s0_step29", "anchor2_s142_step30", "anchor2_s0_step550", "anchor2_s142_step500", "anchor2_s0_step1250", "anchor2_s142_step1250")
    out["tofu_tokens"] = tofu_tokens(tok, rows["tofu"], end_id, w0["tofu"], {n: load(n) for n in names}, seen["tofu"])
    out["stop_rule"] = stop_rule()
    (RES / "dissection.json").write_text(json.dumps(out, indent=1) + "\n")
    (RES / "comparator_table.json").write_text(json.dumps({"note": "One row per state and seed. D3: None means the row is not read yet on this file (qwen-decomp-lead's, 11-07 to 11-11). anchor2_s0 is seed 42. F has no second seed.", "rows": table}, indent=1) + "\n")
    print("DONE", flush=True)
    hard_exit(0)


if __name__ == "__main__":
    main()
