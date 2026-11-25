"""E85 Amendments 1 and 2's filed rules (results/predictions_amend1.json, predictions_amend2.json), pure functions of the read numbers, so that edge cases can be run on them. The ratio is a
state's final KL over W0's at seed 0; bars are E85's, inclusive and compared after rounding to 9 places (`rules_amend2.verdict`), so a number on a bar is on it. NaN or a zero W0 reads "unread"."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import rules

r2 = rules.r2
r1 = rules.r1
ratio = r2.rules.ratio
NAN, UNREAD, NO_CALL = math.nan, r2.UNREAD, rules.NO_CALL

# Amendment 1
RAND_SCALE, PERM_SEED = 100, 85_200  # the site's index is added to the seed, on a CPU generator, so the permutation is the same on any card
W0_CARD = "NVIDIA H200 NVL"  # W0 s0 read 1.7031564672943205 on three pods of this model; on an H100 PCIe 1.7517, 2.8% off (qwen_d1_adversary/results/results.json)
W0_KL = rules.W0_WANT[0]
HOST_TOL, W0_TOL = 2e-4, 0.20  # the host check's reads are logged to 4 places; 20% is E85's, for a W0 read fresh on another card
OWN_PASS_AT, OWN_FAIL_AT = 1.267, 1.776  # 1.5 divided and multiplied by 1 + E85's spread of 18.4% (review of E85, finding 4). Reported beside, never ruled on
BAR_EXCESS = (r2.FAIL_AT - 1) * W0_KL  # 1.039 nats, what the fail bar asks above W0 at seed 0
CLEAN_STOP = 0.2  # KL(W0 ‖ the random change) under this, a fifth of BAR_EXCESS: the state does not move the output, stop before paying for its start
W0_CI, W0_CI_TOL = 0.141670617449563, 0.25  # qwen_adversary/results/adversary.json ci128, the same 128 rows
PERM_CI_STOP = 0.5  # all columns permuted: masks at g with Δ off must read far above W0's 0.14
SHARE = {"Perm": (1.0, 3.5), "PermRare": (0.1, 1.5)}  # ‖V(U_π - U)‖² / ‖V U‖², pooled; 21 sampled matrices read 1.65-2.60 and 0.20-1.11 (reports/plans, Amendment 1)
STEP0_LOW, W0_STEP0 = 0.085, 0.0918  # step 0 read 0.0901-0.0923 on every state of E80 and E85 on these rows
CAN_FAIL, DOES_NOT, TOO_SMALL = "the test can fail at this size", "the adversary does not fail even this", "no call: the state moved the output less than the fail bar asks"
WRONG_FAILS, WRONG_PASSES = "the adversary fails these wrong columns", "the adversary does not fail these wrong columns"

# Amendment 2
LITTLE, MUCH = 2.0, 5.0  # times the floor: a moved V's rounding is one realisation of what an unmoved V's is, so the two should read alike; both factors chosen here, with the row spread read beside
MOVE = 0.01  # the share of (position, column) pairs moving more than this, the threshold of `alive` and `active`
FT_KL_LOW, FT_KL_HIGH = 0.01, 0.2  # E79's eight web rows read 0.040 at 256 tokens; these are the first eight of 512
ACTIVE_LOW, ACTIVE_HIGH = 0.5, 2.0  # eight rows' active share over W0_fineweb.safetensors', alive columns
CLOSES, FOLLOW_UP = "rounding is not a limit: question closed", "fp32 moves g much more: one fp32-V adversary start is the follow-up, not run here"


def verdict_own(r: float) -> str:
    if math.isnan(r):
        return UNREAD
    return r2.PASS if r1.r9(r) <= OWN_PASS_AT else r2.FAIL if r1.r9(r) >= OWN_FAIL_AT else r2.BETWEEN


def goes_on(r: float) -> bool:
    """Stage 2 runs only if stage 1's final ratio reaches the fail bar."""
    return r2.verdict(r) == r2.FAIL


def outcome_rand(r: float, clean: float) -> str:
    """A pass says the adversary misses something only if the state moved the output by what the bar asks; otherwise the bar was out of reach and the pass says nothing. NaN clean is not known."""
    v = r2.verdict(r)
    if v == UNREAD:
        return UNREAD
    if v == r2.FAIL:
        return CAN_FAIL
    if v == r2.BETWEEN:
        return NO_CALL
    return DOES_NOT if r1.r9(clean) >= r1.r9(BAR_EXCESS) else TOO_SMALL


def outcome_perm(r: float) -> str:
    v = r2.verdict(r)
    return {UNREAD: UNREAD, r2.FAIL: WRONG_FAILS, r2.PASS: WRONG_PASSES}.get(v, NO_CALL)


def before_search(trace: Sequence[float], w0_final: float) -> bool | None:
    """Step 0, at the random start, already at the fail bar over W0's final read: the state fails before any search, so the read says nothing about the adversary."""
    return None if not trace or math.isnan(trace[0]) or not w0_final > 0 else r2.verdict(trace[0] / w0_final) == r2.FAIL


def first_steps_miss(state: str, trace: Sequence[float], clean: float = NAN) -> str | None:
    """The filed window of the first steps, on the reads so far; a miss stops the job. Rand100: step 0 from W0's up to W0's plus 1.5 of the clean KL (Δ's dial at a random d passes about a third of it),
    step 2 above step 0. Perm: step 0 far above W0's. PermRare: step 0 not under what every state read. W0 and anything else: none."""
    if state not in ("Rand100", "Perm", "PermRare") or not trace:
        return None
    low = PERM_CI_STOP if state == "Perm" else STEP0_LOW
    high = W0_STEP0 + 1.5 * clean if state == "Rand100" else math.inf
    if len(trace) == 1 and not low <= trace[0] <= high:  # NaN misses
        return f"step 0 reads {trace[0]}, want {low} to {high}"
    if state == "Rand100" and len(trace) == 3 and not trace[2] > trace[0]:
        return f"step 2 reads {trace[2]}, not above step 0's {trace[0]}"
    return None


def steps(trace: Sequence[float], w0: Sequence[float]) -> list[dict[str, float]]:
    """Ratio and absolute excess at each step, as far as both traces go."""
    return [{"step": i, "kl": a, "w0": b, "ratio": ratio(a, b), "excess": a - b} for i, (a, b) in enumerate(zip(trace, w0, strict=False))]


def read_state(state: str, st: Mapping[str, Any] | None, w0: Mapping[str, Any], clean: float) -> dict[str, Any]:
    """Everything reported for one state; NaN and "unread" for a state not read."""
    kl = r1.rules.num(st or {}, "kl")
    base = w0.get("kl", NAN)
    r = ratio(kl, base)
    trace = (st or {}).get("trace", [])
    table = steps(trace, w0.get("trace", []))
    return {"final_ratio": r, "final_excess": kl - base, "verdict": r2.verdict(r), "verdict_own_bars": verdict_own(r), "step10_ratio": table[10]["ratio"] if len(table) > 10 else NAN,
            "before_search": before_search(trace, base), "outcome": outcome_rand(r, clean) if state == "Rand100" else outcome_perm(r), "steps": table}


def score(out: Mapping[str, Any]) -> dict[str, Any]:
    w0, clean = out.get("w0", {}), out.get("clean", {})
    states = {}
    for state in ("Rand100", *(k for k in ("Perm", "PermRare") if k in clean)):
        states[state] = read_state(state, out.get("starts", {}).get(f"{state}_W_s0"), w0, clean.get(state, {}).get("kl_W0", NAN))
    rand = states["Rand100"]
    perm = next((states[k] for k in ("Perm", "PermRare") if k in states), None)
    return {"w0": {"kl": w0.get("kl", NAN), "from": w0.get("from", UNREAD)}, "states": states, "clean": dict(clean), "gold": dict(out.get("gold", {})), "stage2_goes_on": goes_on(rand["final_ratio"]),
            "checks": dict(out.get("checks", {})),
            "leans": {"Rand100_can_fail": r2.lean(rand["final_ratio"], rand["outcome"] == CAN_FAIL), "Perm_fails": None if perm is None else r2.lean(perm["final_ratio"], perm["outcome"] == WRONG_FAILS),
                      "Perm_before_search": None if perm is None else perm["before_search"]}}


def reading(b: float, f1: float, f2: float) -> str:
    """Amendment 2. b: mean |g(rarely-on, fp32) - g(rarely-on, as run)| on the half's columns. f1: W0 read twice. f2: W0 with the fp32 path against as run. The floor is the larger; a zero floor means
    b must be zero to read as little."""
    if any(math.isnan(x) for x in (b, f1, f2)):
        return UNREAD
    floor = max(f1, f2)
    if b <= LITTLE * floor:
        return CLOSES
    return FOLLOW_UP if b >= MUCH * floor else NO_CALL


def floors_ok(f1: float, f2: float) -> bool:
    """The fp32 path must differ from as run, or it was not exercised; and the repeat must be under it, or the read's own path is not the code's, or the floor is not rounding. NaN is not."""
    return f2 > 0 and f1 <= f2
