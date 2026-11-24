"""P1.59 phase A: every saved E_b item of E268, E275, E277, E278, E281 and E283, written to experiments/why_fail/inventory.tsv.

One row per item: its experiment, name, featurization, class, the npz holding K (in_K) and flips, the npz holding num,
den and dd (the same file, or the saved arrays E268/E275 read them from), each file's sha256 (computed here), |K|,
#changed, the margins its item JSON records, and which needed quantity is missing. Reads only saved files; CPU.

    fleet/cpu-run python3 experiments/why_fail/inventory.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "inventory.tsv"
COLS = ("experiment", "item", "feat", "cls", "k_npz", "k_sha256", "arrays_npz", "arrays_sha256", "has_num_den_dd",
        "n_vars", "size_K", "n_changed", "margin_min", "margin_median", "missing")  # fmt: skip


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fname(item: str, exp: str = "") -> str:
    return item.replace("/", "__" if exp == "E277" else "-")  # E277's run.py fname, else E257's


def cls(item: str) -> str:
    """The item class (P1.59 card): built contrast, built new, random, toy trained, procedural, RL, realistic."""
    i = item.replace("/", "-").removesuffix("@cue")
    if i in ("x2", "x10", "rebuilt"):
        return "built_reuse"
    if i in ("new_few", "new_thin"):
        return "built_new"
    if i.startswith(("random_kl", "random_size")):
        return "built_random"
    if i.startswith("panel"):
        return "realistic_random" if i.split("-")[-1].startswith("random_") else "realistic"
    if i.startswith("p256"):
        return "procedural_random" if i.rsplit("_s", 1)[0].endswith(("_random", "_randg")) else "procedural"
    if i.startswith("rl-"):
        return "rl"
    if i.startswith("random_e58") or i.endswith("_rand"):
        return "toy_random"
    return "toy_trained"


def fields(j: dict) -> dict:
    mmin = j.get("min_margin", j.get("margin_min"))
    mmed = j.get("median_margin", j.get("margin_median"))
    return {"n_vars": j.get("n_vars"), "size_K": j.get("size_K", j.get("size_Kcopy")), "n_changed": j.get("n_changed"),
            "margin_min": mmin, "margin_median": mmed}  # fmt: skip


def arrays_of(exp: str, item: str, j: dict, knpz: Path) -> Path | None:
    """The npz with num, den, dd: the item's own, else the one E268/E275 read (rb_states, thin_edit_diagnosis)."""
    if knpz.is_file() and "dd" in np.load(knpz).files:
        return knpz
    if exp == "E268" and cls(item).startswith("built"):
        return EXP / "thin_edit_diagnosis" / "results" / "large" / f"items_{fname(item)}.npz"
    if exp in ("E268", "E275"):
        p = EXP / "rb_states" / "results" / "large" / f"items_{fname(item)}.npz"
        return p if p.is_file() else None
    return None


def rows_of(exp: str, d: str, feat: str) -> list[dict]:
    out = []
    for jp in sorted((EXP / d / "results" / "items").glob("*.json")):
        j = json.loads(jp.read_text())
        item = j["item"]
        if j.get("reading") == "prior":  # E278 re-used E275's reading of the same construction: E275's row holds it
            out.append({c: "" for c in COLS} | {"experiment": exp, "item": item, "feat": feat, "cls": cls(item), "has_num_den_dd": False}
                       | fields(j) | {"missing": "dup:" + j["source"]})  # fmt: skip
            continue
        knpz = EXP / d / "results" / "large" / f"items_{fname(item, exp)}.npz"
        out.append(row(exp, item, feat, j, knpz))
    return out


def row(exp: str, item: str, feat: str, j: dict, knpz: Path) -> dict:
    ap = arrays_of(exp, item, j, knpz)
    has = ap is not None and ap.is_file() and {"num", "den", "dd"} <= set(np.load(ap).files)
    if exp == "E281":  # E281 saved the write sums on the copy windows only
        has = knpz.is_file() and {"num_copy", "den_copy", "dd_copy"} <= set(np.load(knpz).files)
        ap = knpz if has else None
    miss = [m for m, ok in (("K", knpz.is_file()), ("num_den_dd", has)) if not ok]
    if exp == "E268" and has and "changed" not in np.load(ap).files:
        p = EXP / "kstar_vpd_wording_a" / "results" / "large" / f"items_{fname(item)}.npz"  # E268's built edits: E242's changed
        miss.append("none(changed in " + str(p.relative_to(EXP.parent)) + ")" if p.is_file() and "changed" in np.load(p).files else "changed")
    r = {"experiment": exp, "item": item, "feat": feat, "cls": cls(item),
         "k_npz": str(knpz.relative_to(EXP.parent)) if knpz.is_file() else "", "k_sha256": sha(knpz) if knpz.is_file() else "",
         "arrays_npz": str(ap.relative_to(EXP.parent)) if has and ap is not None else "",
         "arrays_sha256": sha(ap) if has and ap is not None else "", "has_num_den_dd": has} | fields(j)  # fmt: skip
    return r | {"missing": ",".join(miss) or "none"}


def reliance_rows() -> list[dict]:
    """E283's own F arrays: x2 and x10 on new_few's cue windows (neurons). Its mK files hold M's K only (no write sums)."""
    out = []
    for jp in sorted((EXP / "reliance" / "results" / "fcue").glob("*.json")):
        j = json.loads(jp.read_text())
        knpz = EXP / "reliance" / "results" / "large" / f"fcue_{j['item']}.npz"
        out.append(row("E283", j["item"] + "@cue", "neurons", j | {"n_vars": len(np.load(knpz)["in_K"])}, knpz))
    return out


def main() -> None:
    rows = (rows_of("E268", "kstar_needed", "vpd") + rows_of("E275", "claim2_needed", "vpd") + rows_of("E277", "needed_neurons", "neurons")
            + rows_of("E278", "eb_dropped", "vpd") + rows_of("E281", "e62_copy_windows", "vpd") + reliance_rows())  # fmt: skip
    lines = ["\t".join(COLS)] + ["\t".join("" if r[c] is None else str(r[c]) for c in COLS) for r in rows]
    OUT.write_text("\n".join(lines) + "\n")
    cnt: dict[tuple, int] = {}
    for r in rows:
        k = (r["experiment"], r["feat"], r["cls"], r["missing"])
        cnt[k] = cnt.get(k, 0) + 1
    for k, v in sorted(cnt.items()):
        print(*k, v, sep="\t")
    print(len(rows), "rows ->", OUT)


if __name__ == "__main__":
    main()
