"""E90 Amendment 2, the last the rule allows: are the value form's "above the band" rows losses specific to C, or the size of the
variance removed? Filing: results/predictions_amend2.json; pure rules: rules_amend2.py; plan: reports/plans/2026-11-05_plan_e90-amend2.md.
  1. Control first and alone, per C and block: 20 random bases matched in variance on X_C's fit residuals, W0's teeth against them.
     A C without teeth is unreadable; its state rows are still read and printed beside. The do-nothing state is read here.
  2. Each state and its saved hand write under the same draws (no new match, nothing trained), at blocks 1 and 2, with the shares on
     the task windows printed. Stage 2's bases, reads and `rules.verdict` are unchanged; the mechanism form (row 1) is not read.
Writes results/amend2.json as each row finishes; nothing large. Not run as of its commit. Usage: sbatch cluster/new_vs_reuse.sbatch amend2
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

import apparatus as AP
import reading as RD
import rules
import rules_amend2 as RA
import stage1 as S1
import stage2 as S2
import torch
from apparatus import LARGE, REPO, RESULTS, Capability, Site, dump, log, stop
from reading import Basis, Job
from torch import Tensor

from commute import perturb as PT
from commute import readout as RO
from commute import steering as ST
from commute.data import pile_rows
from commute.jobs import hard_exit

OUT, FILING = RESULTS / "amend2.json", RESULTS / "predictions_amend2.json"
AMEND1, STAGE2 = RESULTS / "amend1.json", RESULTS / "stage2.json"
REPRODUCE_TOL = 1e-3  # filed: results/predictions_amend2.json `reproduction`


def capabilities(j: Job, rows: Tensor) -> dict[str, tuple[Capability, dict[int, Tensor], int]]:
    """Form -> X_C with its control read, the basis stage 2 removed, and the seed tag: rebuilt from the same seeds, as stages 1 and 2."""
    S, saved = j.S, torch.load(LARGE / "bases.pt", weights_only=True, map_location="cpu")
    values = json.loads((AP.EXP / "open_pair/results/amend1.json").read_text())["c0"]["rank1"]["values"]
    guess = S1.guess_capability(j, rows, torch.tensor(sorted(set(values))))[0]
    guess.read = AP.capped_held(guess.read, S["n_control"], S["control_seed"])
    keys = torch.tensor(list(torch.load(S1.REG / "windows_ident32.pt", weights_only=False)["task"].keys))
    ident = AP.identity_capability(j.net, "token identity", rows, keys, S1.HALF, S, S["n_control"])
    keys24 = torch.tensor(RD.table_of(j.site(S2.PAIR / "amend1/windows_copy_trained.pt")[1])[0])
    copy24 = AP.identity_capability(j.net, "token identity, copy-24's keys", rows, keys24, S2.HALF, S, S["n_control"])
    ident.read, copy24.read = AP.own_top1(j.net, ident.read), AP.own_top1(j.net, copy24.read)
    return {"guess": (guess, saved["guess"], 1), "identity": (ident, saved["identity"], 2), "copy24": (copy24, {b: u.float() for b, u in copy24.bases(S["blocks"]).items()}, 4)}


def matched_basis(j: Job, form: str, c: Capability, bases: dict[int, Tensor], tag: int) -> Basis:
    """The basis at each block with its matched draws; a block with no draws leaves the reads, and its verdict is "no matched band"."""
    out, randoms = j.res.setdefault("matched", {}).setdefault(form, {}), {}
    for b, u in bases.items():
        out[str(b)], randoms[b] = RA.matched(c.fit_h[b], c.y, u, j.S["n_random"], [j.S["random_seed"], tag, b])
        log(f"[matched {form} block {b}] {json.dumps({k: v for k, v in out[str(b)].items() if k != 'draw_shares'})}")
        log(f"[matched {form} block {b}] mean share of the draws on the fit residuals {sum(out[str(b)].get('draw_shares', [math.nan])) / j.S['n_random']:.4f}")
    return Basis({b: u for b, u in bases.items() if randoms[b]}, {b: r for b, r in randoms.items() if r})


def references() -> dict[str, dict[str, Any]]:
    """Each C's control as it read under stage 2's draws: the file each was last read in."""
    first, amend, second = (json.loads(p.read_text()) for p in (S2.FIRST, AMEND1, STAGE2))
    return {"guess": first["control"]["W0's guess"], "identity": amend["control"]["token identity"], "copy24": second["control"]["token identity, copy-24's keys"]}


def controls(j: Job, form: str, c: Capability, f: Basis, ref: dict[str, Any]) -> dict[str, bool]:
    """W0's teeth against the matched draws, alone; the rebuilt X_C and the basis's own drop must reproduce the filed read. Then do-nothing."""
    on = RD.control(j, c, f)
    rec = j.res["control"][c.name]
    for b in f.bases:
        for field in ("drop", "variance_share"):
            if not abs(rec[str(b)][field] - ref[str(b)][field]) < REPRODUCE_TOL:
                stop(OUT, j.res, f"{form} block {b}: {field} {rec[str(b)][field]} against the filed {ref[str(b)][field]}")
        RD.do_nothing(j, c, f, on, b)
        j.res.setdefault("do_nothing", {})[f"{form}_{b}"] = j.res["R1"][c.name]
    dump(OUT, j.res)
    return on


def task_shares(j: Job, stem: str, s: Site, f: Basis) -> None:
    """The basis's and the draws' variance share on W0's residuals at the task windows, as `AP.variance_share`. Printed; no verdict reads it."""
    h = RO.cache64(j.net, s.toks, s.tgt[:1], s.half)[0]
    out = j.res.setdefault("task_windows", {}).setdefault(stem, {})
    for b, u in f.bases.items():
        basis, draws = AP.variance_share(h[b], u), [AP.variance_share(h[b], r) for r in f.randoms[b]]
        out[str(b)] = {"basis": basis, "draws_mean": sum(draws) / len(draws), "draws": draws, "match": RA.task_match(basis, sum(draws) / len(draws)), "n_windows": len(s.toks)}
        log(f"[task windows {stem} block {b}] basis {basis:.4f} draws' mean {out[str(b)]['draws_mean']:.4f} {out[str(b)]['match']}")


def saved_write(j: Job, name: str, second: dict[str, Any]) -> ST.DW:
    """Stage 2's hand write at its matched alpha, from the X it saved."""
    return AP.zero_except(j.net, PT.WRITE_KEY, second["writes"][name]["alpha"] * torch.load(LARGE / f"X_{name}.pt", weights_only=True))


def read_form(j: Job, form: str, f: Basis, second: dict[str, Any]) -> None:
    """Every state and its write; the basis's own share under each must reproduce stage 2's."""
    w0s: dict[Path, AP.Reads] = {}
    for name, wf, sf in S2.rows_of()[form]:
        s = j.site(wf)[0]
        if wf not in w0s:
            w0s[wf] = AP.read_all(j.net, s, None, f.bases, f.randoms)
            task_shares(j, wf.stem, s, f)
        for tag, dW in ((name, j.state(sf)), (f"{name}_write", saved_write(j, name, second))):
            got = RD.read_state(j, tag, s, dW, w0s[wf], f)
            for b in f.bases:
                was = second["states"][tag][str(b)]["A_lo"]
                if not abs(got[str(b)]["A_lo"] - was) < REPRODUCE_TOL:
                    stop(OUT, j.res, f"{tag} block {b}: share {got[str(b)]['A_lo']} against stage 2's {was}")


def map_row(by: dict[int, dict[str, Any]], m: str, task: dict[str, Any]) -> dict[str, Any]:
    """One map's line: each column names its block; `beside` is the read with teeth and a matched band assumed; stage 2's band follows."""
    said = lambda col: RA.named({b: rules.say_seeds(r[col][m]["verdict"], r["shares"][m]) for b, r in by.items()})
    return {"verdict": said("verdicts"), "beside": said("beside"), "stage_2_band": RA.named({b: r["old"]["verdicts"][m]["verdict"] for b, r in by.items()}),
            "bands": {str(b): r["band"] for b, r in by.items()}, "task_windows": {str(b): task.get(str(b), {}).get("match") for b in by}}  # fmt: skip


def table(j: Job, second: dict[str, Any], refs: dict[str, dict[str, Any]], on: dict[str, dict[str, bool]]) -> None:
    """Per C and block the matched read and stage 2's beside it, then Fable's branch per block."""
    blocks, final, status = j.S["blocks"], {}, {b: {} for b in j.S["blocks"]}
    for form, cases in S2.rows_of().items():
        names = [n for n, _, _ in cases]
        by = {b: {**RA.read_block(j.res["states"], names, b, on[form].get(str(b), False), j.res["matched"][form][str(b)]["reason"]),
                  "old": RA.read_block(second["states"], names, b, refs[form][str(b)]["teeth"], None)} for b in blocks}  # fmt: skip
        j.res.setdefault("rows", {})[form] = {str(b): r for b, r in by.items()}
        stems = {n.rsplit("_", 1)[0]: wf.stem for n, wf, _ in cases}
        for m, stem in stems.items():
            final[m] = map_row(by, m, j.res["task_windows"].get(stem, {}))
            log(f"[table {m}] {json.dumps(final[m])}")
        for b in blocks:
            status[b] |= by[b]["status"]
    j.res["table"] = final
    j.res["branch"] = {str(b): RA.branch(status[b]) for b in blocks}
    log(f"[branch per block] {json.dumps(j.res['branch'])}")


def refused() -> str | None:
    """Why the job must not start: the filing not committed unchanged, or an input result not complete."""
    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, check=False).stdout
    rel = str(FILING.relative_to(REPO))
    if not git("ls-files", rel).strip() or git("diff", "HEAD", "--", rel):
        return f"{rel} is not committed unchanged"
    return next((f"{p.name} has not completed" for p in (S2.FIRST, AMEND1, STAGE2) if not p.exists() or not json.loads(p.read_text()).get("complete")), None)


def main() -> None:
    S = AP.filed()["settings"]
    why = refused()
    if why:
        log(f"{why}: nothing runs (exit 1)")
        hard_exit(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    j = Job(ST.Net("cuda"), S, AP.records(), {"sha256": {}}, OUT)
    second, refs = json.loads(STAGE2.read_text()), references()
    caps = capabilities(j, pile_rows(*S["pile"]["rows"], log=log))
    forms = {form: matched_basis(j, form, c, bases, tag) for form, (c, bases, tag) in caps.items()}
    on = {form: controls(j, form, caps[form][0], f, refs[form]) for form, f in forms.items()}
    log(f"[teeth at the matched band] {json.dumps(on)}")
    for form, f in forms.items():
        read_form(j, form, f, second)
    table(j, second, refs, on)
    j.res["complete"] = True
    dump(OUT, j.res)
    log("every claim: a circuit from old parts reads as reuse. E90 Amendment 2 complete")
    hard_exit(0)


if __name__ == "__main__":
    main()
