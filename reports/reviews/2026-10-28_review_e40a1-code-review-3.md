# Review 3 of commit 0dc2c0a (E40 Amendment 1, second-review fixes). Read only, nothing run.
Verdict: no crash or wrong-number bug found in the diff. One material cost risk (by design, but predictable), three minor items.

Checked and clean:
- Early stop (src/commute/qwen_teacher.py:237): the break comes after opt.step() and curve.append, and `finally` restores model.eval().
  The offsets are consistent. Adam state is local and T0 never saves it. t_steps=len(curve) (amend1.py:342) is the number of steps run.
  "fall" = last reading / step 0. The last reading is always the stop step, because the stop fires only at a read step, and the cap (300) is a
  multiple of read_every (50), so 300 is read.
- No accidental stop: B's positive_control uses until=None, so on_step is False on every step. T1 (run.py:1083) and E40's calls
  (run.py:861, 898) pass teacher_checks' on_step, which returns None. job.check/finite also return None.
- Order: inside on_step, checks() runs before the `until` test (amend1.py:335), so the step's reading is in `fixed` when it is tested.
  The step-100 end check (end_frac 0.9, stop) still fires when T0 passes step 100. The early job.put("T0") before the fineweb key
  is safe because score_amend1 needs "fineweb" in T0 (run.py:351).
- What T0 needs once G1 has run: alive.safetensors (B) and t_lr (res B). The target is read from the filed rule. job.stage="T0", so adv() uses res["G"]["fineweb"]["step0"] as its reference. w0_ref seed 0 = G's reading, seeds 1-2 computed lazily. chosen() accepts B,G1,T0,L,A0 and the default order.
- Filed text agrees with code: the steps_rule (first reading ≤ fall_max ends training; a miss at 300 stops before the adversary) matches amend1.py:353-358. job.check(stop=True) calls sys.exit, which skips the FineWeb adversary. fp32_tol 0.01 is read by run.py:752.

Findings:
1. MATERIAL (cost, not a bug). amend1.py:354-358 with predictions T0.fall_max=0.5, steps_max=300. The only measured curves bend.
   check_teacher read 0.59 at step 300 at 1e-5. Rehearsal B read 0.718/0.725 at step 100. That a full-size T0 reaches 0.5 by 300 is a hypothesis
   ("smaller α should go further"), with no measurement behind it. A miss stops the paid job after B+G1+T0 (~1 h of H200 time) with F1/D1/T1 never run.
   The rehearsal will also almost certainly log this miss as "unexpected": A1_TINY inherits t0_steps=300, uses 16 columns, and its "T0 positive control" entry was removed from
   expected_misses (amend1.py:128). That is uninformative about full size. Smallest fix: write a decision rule into the runner brief for
   when B's full-size reading at the chosen rate comes in (B runs before T0). If B's 100-step fall at the chosen rate is > ~0.65,
   the runner pauses before T0 and asks the user, instead of finding out ~300 steps later. Also state in the gate record that the rehearsal's
   T0 miss is expected at 16 columns.
2. minor. amend1.py:388-392: before_loading requires a finished B only for F1. A run with --stages T0/T1 and no B.json in that --out
   falls back silently to cfg.t_lr (run.py:515). alive() then crashes only at load. Fix: extend the check to
   `names & {"G1","T0","D1","T1"}` and require "B" in done(layout).
3. minor (waste). T0's first 100 steps repeat B's chosen-rate control exactly: same target, pc_seed, cfg.seed batch order and zero offsets.
   That is ~100 x teacher_step_s billed twice. Accept it (simplest), or note it in the cost estimate. Don't add resume code for it.
4. minor (pre-mortem). With T1's offsets no longer uploaded, a pod run with --volume none (the plan's "elsewhere" case) loses every
   offsets and δW file at terminate. Only JSONs and the log remain. That is fine if the analysis reads only JSONs; the gate record should say so.
5. minor. projection() (amend1.py:191) budgets T0 at the full 300 steps: conservative. Extra T0 starts fall under rereads_worst. OK as is.

Friction: the full git show was over the output limit and went to a file. run.py is ~1200 lines with Job helpers spread out, so tracing t_lr/alive/w0_ref took several greps. Otherwise no friction.
