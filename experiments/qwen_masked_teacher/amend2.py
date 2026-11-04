"""E76 Amendment 2: does the fall read in blocks 0-20 sit in the blocks E77 read as nearly deaf (0-14), or in 15-20, where the masked arms grew?
The same read as Amendment 1, on two more parts. Plan: reports/plans/2026-11-03_plan_e76-masked-teacher.md, Amendment 2."""
from __future__ import annotations

from amend1 import read_parts

if __name__ == "__main__":
    read_parts({"first": range(15), "middle": range(15, 21)}, "amend2")
