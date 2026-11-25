# P2.79: the reading refactor

Every reading the paper uses now has one definition in the shared library, and finished experiments sit, unedited, in an
archive folder. Old and new code gave the same numbers on the same saved model states in free cluster checks, and every
rebuildable paper figure file came out byte for byte the same. The refactor changes no number in the paper; it means a
new run reads re-gating share, write share, KL, steering and the rest from one place instead of from another
experiment's script.

## What changed (merged to main as 08f49906)
- **One definition per reading** in `src/commute/`: re-gating share and its floor (`share.py`), KL (`kl.py`), write
  share at VPD, neurons and SAE (`eb.py`), the needed-set scan (`needed.scan_sites`), steering
  (`fair_steering`), prior, effective rank, windows, items and world. The older names in `panel`, `masked_reads` and
  `fair_steering` are now thin calls into these. `src/commute/README.md` lists them.
- **Archive.** 50 finished folders moved to `archive/experiments/` with `git mv`; 63 sbatch scripts and 8 tests moved
  beside them. A folder stays live if live code or the paper's source notes reach it, by path or by
  `load_experiment`, which looks in the archive when a folder is not live.
- **Lines of Python** (tracked): `src/commute` 39,667 before, 40,421 after; live `experiments/` 227,433 before,
  195,144 after (33,735 now in the archive).

## Checks
- **Reproduction** (E309, cluster debug queue, $0): old code run twice to measure its run-to-run spread, new code once,
  on the same saved states. The table has 51 rows (readings by state, needed-set scans at VPD, neurons and SAE
  included); in 50 the new code equals old. The one other, a per-key steering row, sits a fourteenth of the old spread away from the nearer old run: inside the
  cell's rule ("within the measured floor"); the stricter in-range test in `compare.py` flags it and was left as is.
  Table: `experiments/reading_refactor/results/reproduction.json`; notes: `experiments/reading_refactor/NOTES.md`.
- **Paper figures:** every builder whose inputs exist rebuilt byte-identical data; no builder input moved.
- **Tests:** 7709 pass; the 4 failures fail the same way on main (one old Qwen test, three sbatch stage tests).
- **Review:** one fresh Opus review; all findings applied (the main one kept live every folder that live code loads).

## Limits
Existing run scripts were not edited, so old scripts still load each other; new code reads from `src/commute/`.
Gaps between saved results and any recompute (old code shows them too) are listed in `NOTES.md` and were sent to
writeup-lead.

## Not run
- No GPU check of `masked_reads.route_cells_by` (left as its own definition: it needs the logits of the same forward).
- No figure: nothing here is a measurement.
