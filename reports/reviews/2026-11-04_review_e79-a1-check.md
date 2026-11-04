# Independent check of E79 Amendment 1, job 51161 (2026-11-04)

By a fresh Opus agent given the question, the files and the repo path; scripts and outputs in `~/commute-data/scratch/qwen-decomp-lead/e79a1-check/`. Recorded by the lead from its hand-back. It led to Amendment 2 (plan's last section), which read do-nothing on the same rows and draw.

**Verdict:** "passes all three" is correctly computed from the filed rules, but it does not make A a usable map: the three rules together would probably pass do-nothing, which was never read here.

1. Do-nothing is missing. E68 read it on u1 at 0.104 against W0's 1.374, nothing added, Δ off, other rows: not like for like. 1.5 x its K0, inside the 2 x faithful bar.
2. The fall comes from recomputing the masks. A frozen 1.127, own 0.078, W0 1.334: 0.84 of the fall appears only when Γ is recomputed. Per-position |ΔΣg| against W0 on M: A 171.6, B 165.2, L 179.5, P 12.4, C 15.5.
3. Own ≈ clean + self for near-FT states: A on M 0.0767 against 0.0776; A on W 0.0875 against 0.0869; B on M 0.1982 against 0.1979; L on M 0.1410 against 0.1419. So the own read adds nothing beyond those two numbers there.
4. Teeth tests only whether clean is far from FT (`rules_amend1.py:60-64`). P and C are W0-like: clean 1.607, 1.664; self 0.0425, 0.0391; each ~15.6 x A, W0 itself 17.2 x. The upper clause is vacuous. No state with clean near FT and a wrong placement was tried. Read on M only: on W, P is 1.47 x A and C 1.45 x, which would read "no teeth".
5. Faithful passes both bad placements. M bar 0.0854: P, C pass, B 0.174 and L 0.118 fail. W bar ~0.171: all pass; W's mask noise 0.086 is over twice the update's whole clean effect 0.040. The 2 x bar has no source; A at entropy 1.51 is compared with W0 at 0.58.
6. Opens reads only a net mean (`rules_amend1.py:49,96`). A's ratio 0.993, yet 10.4% of its M positions move Σg by over 5%, at most 2626 of 7481; columns swapping or cancelling are invisible.
7. The control proves determinism only.
8. A is not a decomposition of W + δW (E76 report): squared distance to FT's weights 4.75, against W0's 3.49, cosine 0.106 to δW.

**Verified:** means recompute from `large/a1_*` (worst relative error 2e-5, float16 storage); rules by hand give the printed verdicts; `student_masked` algebra matches the plan; own g under `Offsets.state` (`run.py:228`); Δ₀ identical for every state; draws paired, rows identical; P permuted together and scaled as filed; C read from `C_offsets` by the same path; W outside the training pool.

**Not verified:** do-nothing in this job's scheme (Amendment 2 has since read it); per-site or per-column g, not saved; GPU reruns; tests read in part, not run; whether bf16 rounding of V′ in Γ's readout (`run.py:234`, `qwen_vpd.py:249`) matters (every state); reverse KL additivity.
