# Digest 2026-10-29 (qwen-decomp-lead): E44 ran and its pass test has no teeth; the Qwen line waits on you

**Why I finished.** Everything left on the Qwen line waits on two decisions only you can make. Ask 10 is how the pass
test should change. Ask 8 is whether to file E45 Amendment 2 as drafted.

## What ran
- **E44 (Q0 + Q1) on an A100 SXM, $3.66 ($6.14 in all, with the earlier pods that never ran).** The reproduction
  check held: the regenerated update moves the model as E40's did, within 0.6%. The stored split pieces are precise
  (1.9e-4 against the 1e-3 stop). The run stopped at stage K, as filed: the controls that should fail did not.
- **Why (no bug).** The masks are recomputed from each split's own model. A split holding the fine-tuned weights
  gets masks that bring the original subcomponents within 0.09 nats of the fine-tuned model on maths rows, against
  1.42 with the original model's masks. So do-nothing, which drops the update, passes, and so do both halves.
  Report: `reports/analyses/2026-10-29_analysis_e44-no-teeth.md`; notebook `notebooks/E44_qwen_maps.ipynb`.

## Pending decisions (log card `decision-e44-pass-test`, log v116)
- **Ask 10 (blocking).** Options: (1) masks from the original model for every split, a move from reading A to
  reading B; (2) a mask-cost or refit reading; (3) where the update lands, with no pass line. Fable recommends (1),
  as E44's last amendment, with a random control matched in KL to start. It also recommends a $0 layer-cut check
  first, to tell whether the masks describe the change or copy the fine-tuned output from late layers
  (`reports/plans/2026-10-29_advice_e44-pass-test-FABLE.md`).
- **Ask 8.** E45 Amendment 2 draft (`reports/plans/2026-10-29_plan_e45-amendment2.md`). Its scoring may depend on
  ask 10.

## Also done
`run.py` plumbing fixes (d3ca128). Friction noted: 158 (jq not on the pod image, now in the brief), 159 (check.py
flags file:line citations), 161 (publisher takes three sends).

## Money
This session $3.66 of $50 (remote-kernels). No pod or Slurm job of mine is live.

## Not run
E44 stages M, G, A and H; step 100; the layer-cut check; any rerun with a changed pass test.
