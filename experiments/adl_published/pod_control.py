"""E284 (cell P2.67): the authors' toolkit, unchanged, on the paper's Qwen3-1.7B organisms (spec
reports/plans/2026-11-23_plan_p2-67-adl-spec.md section 4.1, option A). Standard library only: it runs on a pod or a
cluster node with any python3, installs the toolkit at the pinned commit with its own uv.lock, and drives its Hydra
command once per organism x variant, each with a fresh storage dir, then tars each run's outputs with a sha256.

  install   git clone + checkout 617d9266 into DIR, `uv sync --frozen` (the toolkit's uv.lock)
  checks    records: the pinned HEAD, mix1-2p0 listed for qwen3_1_7B in each organism config, the Qwen3 tokenizer's
            BOS (bos_token, and whether encode(add_special_tokens=True) prepends anything)    -> OUT/checks.json
  run       the spec's resolved command per --runs entry (organism:variant); OUT/<org>_<variant>/ holds the storage
            dir, main.log and the toolkit's stderr; OUT/<org>_<variant>.tar.gz and .sha256 when it ends. The first
            causal-effect batch's time is printed as soon as tqdm shows it; over --max-batch-s it kills the run (exit 3)
  --rehearse  tiny sizes for the cluster's 1 h debug slot: 256 texts for delta_bar, 8 eval documents per eval set; the
            64 + 64 random vectors and positions 0-8 stay at full size, so a batch costs what it costs in the scored run
  --extra   further Hydra overrides (the parity floor's second run, parity.py)

    python3 experiments/adl_published/pod_control.py install --dir /root/dt
    python3 experiments/adl_published/pod_control.py checks --dir /root/dt --out /root/out
    python3 experiments/adl_published/pod_control.py run --dir /root/dt --out /root/out \
        --runs cake_bake:default,cake_bake:mix1-2p0 [--rehearse] [--max-batch-s 2.8]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

REPO = "https://github.com/science-of-finetuning/diffing-toolkit.git"
COMMIT = "617d9266eada0baf5fe5347ce97e64b6dd0f82bf"
ORGANISMS = ("cake_bake", "kansas_abortion", "fda_approval")
VARIANTS = ("default", "mix1-2p0")
REHEARSE = ["diffing.method.max_samples=256", "diffing.method.causal_effect.max_samples=8"]
BAR = re.compile(r"Processing [^:\r\n]*:\s+\d+%\|[^|]*\|\s*(\d+)/(\d+) \[(\d+):(\d+)")


def sh(cmd: list[str], cwd: Path | None = None) -> str:
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd, check=True, text=True, capture_output=True).stdout.strip()


def uv() -> str:
    found = shutil.which("uv") or next((str(p) for p in (Path.home() / ".local/bin/uv", Path.home() / ".cargo/bin/uv") if p.is_file()), None)
    if found:
        return found
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "uv"], check=True)
    return shutil.which("uv") or "uv"


def install(d: Path) -> None:
    if not d.exists():
        sh(["git", "clone", "--quiet", REPO, str(d)])
    sh(["git", "checkout", "--quiet", COMMIT], d)
    head = sh(["git", "rev-parse", "HEAD"], d)
    assert head == COMMIT, f"HEAD {head}, not {COMMIT}"
    t0 = time.time()
    subprocess.run([uv(), "sync", "--frozen"], cwd=d, check=True)
    print(f"install: {COMMIT} synced from its uv.lock in {time.time() - t0:.0f} s", flush=True)


def checks(d: Path, out: Path) -> None:
    variants = {}
    for o in ORGANISMS:
        text = (d / "configs" / "organism" / f"{o}.yaml").read_text()
        block = text.split("\n  qwen3_1_7B:\n", 1)[1].split("\n  qwen3_1_7B_Base:", 1)[0] if "\n  qwen3_1_7B:\n" in text else ""
        variants[o] = {v: f"\n    {v}:\n" in block for v in VARIANTS}
    code = ("from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B'); "
            "import json; print(json.dumps({'bos_token': t.bos_token, 'encode_hello': t.encode('hello', add_special_tokens=True), "
            "'encode_hello_plain': t.encode('hello', add_special_tokens=False)}))")  # fmt: skip
    tok = json.loads(sh([uv(), "run", "--frozen", "python", "-c", code], d).splitlines()[-1])
    rec = {"head": sh(["git", "rev-parse", "HEAD"], d), "variants_listed": variants, "tokenizer": tok,
           "adds_bos": tok["encode_hello"] != tok["encode_hello_plain"]}  # fmt: skip
    out.mkdir(parents=True, exist_ok=True)
    (out / "checks.json").write_text(json.dumps(rec, indent=1))
    print("checks:", json.dumps(rec), flush=True)
    if rec["head"] != COMMIT or not all(all(v.values()) for v in variants.values()) or rec["adds_bos"]:
        raise SystemExit(3)


def command(org: str, var: str, base: Path, extra: list[str]) -> list[str]:
    """Spec section 4.1's resolved command, with a fresh base_dir per run."""
    return [uv(), "run", "--frozen", "python", "main.py", "diffing/method=activation_difference_lens", "model=qwen3_1_7B",
            f"organism={org}", f"organism_variant={var}", "pipeline.mode=diffing", "infrastructure=runpod",
            f"infrastructure.storage.base_dir={base}", "diffing.method.overwrite=true",
            "diffing.method.causal_effect.enabled=true", "diffing.method.causal_effect.overwrite=true",
            "diffing.method.auto_patch_scope.enabled=false", "diffing.method.steering.enabled=false",
            "diffing.method.token_relevance.enabled=false", *extra]  # fmt: skip


def first_batch(log: Path) -> tuple[int, int, int] | None:
    """(eval group seen, elapsed s, batches) of the first tqdm update of a 'Processing ...' bar past 0, or None."""
    if not log.is_file():
        return None
    for seg in re.split(r"[\r\n]", log.read_text(errors="replace")):
        m = BAR.search(seg)
        if m and int(m.group(1)) > 0:
            return int(m.group(1)), 60 * int(m.group(3)) + int(m.group(4)), int(m.group(2))
    return None


def run_one(d: Path, out: Path, org: str, var: str, extra: list[str], max_batch_s: float) -> int:
    name = f"{org}_{var}"
    run_dir = out / name
    if run_dir.exists():
        shutil.rmtree(run_dir)  # fresh storage: the toolkit skips results it finds
    run_dir.mkdir(parents=True)
    cmd = command(org, var, run_dir / "storage", extra)
    (run_dir / "command.txt").write_text(" ".join(cmd) + "\n")
    err = run_dir / "stderr.log"
    t0, seen = time.time(), False
    print(f"[{name}] start: {' '.join(cmd)}", flush=True)
    with err.open("w") as fe, (run_dir / "stdout.log").open("w") as fo:
        p = subprocess.Popen(cmd, cwd=d, stdout=fo, stderr=fe, env=os.environ | {"WANDB_MODE": "disabled"})
        while p.poll() is None:
            time.sleep(5)
            fb = first_batch(err)
            if fb and not seen:
                seen = True
                done, el, total = fb
                print(f"[{name}] FIRST CAUSAL-EFFECT BATCH: {done} of {total} eval documents in {el} s "
                      f"({el / done:.1f} s per document; bar {max_batch_s} s); {time.time() - t0:.0f} s since start", flush=True)  # fmt: skip
                if max_batch_s and el > max_batch_s * done:
                    p.kill()
                    print(f"[{name}] STOP: first batch over the bar", flush=True)
                    return 3
    code = p.returncode
    print(f"[{name}] exit {code} after {time.time() - t0:.0f} s", flush=True)
    for lg in run_dir.glob("storage/hydra/*/*/main.log"):
        shutil.copy(lg, run_dir / "main.log")
    tar = out / f"{name}.tar.gz"
    with tarfile.open(tar, "w:gz") as tf:
        tf.add(run_dir, arcname=name)
    digest = hashlib.sha256(tar.read_bytes()).hexdigest()
    (out / f"{name}.sha256").write_text(f"{digest}  {tar.name}\n")
    print(f"[{name}] TAR {tar} sha256 {digest} ({tar.stat().st_size / 2**20:.1f} MB)", flush=True)
    return code


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["install", "checks", "run"])
    ap.add_argument("--dir", type=Path, required=True, help="the toolkit clone")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--runs", default="", help="organism:variant,... (run)")
    ap.add_argument("--rehearse", action="store_true")
    ap.add_argument("--max-batch-s", type=float, default=0.0, help="stop when the first batch takes more s per document")
    ap.add_argument("--extra", nargs="*", default=[])
    a = ap.parse_args()
    if a.step == "install":
        install(a.dir)
        return 0
    assert a.out is not None, "--out"
    if a.step == "checks":
        checks(a.dir, a.out)
        return 0
    runs = [tuple(r.split(":")) for r in a.runs.split(",") if r]
    assert runs and all(o in ORGANISMS and v in VARIANTS for o, v in runs), runs
    extra = (REHEARSE if a.rehearse else []) + a.extra
    codes = [run_one(a.dir, a.out, o, v, extra, a.max_batch_s) for o, v in runs]
    print(f"ALL RUNS DONE: exits {dict(zip([f'{o}_{v}' for o, v in runs], codes, strict=True))}", flush=True)
    return max(codes)


if __name__ == "__main__":
    sys.exit(main())
