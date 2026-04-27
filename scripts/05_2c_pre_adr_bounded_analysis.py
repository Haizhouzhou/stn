#!/usr/bin/env python
"""Run Phase 5_2C pre-ADR bounded event-remediation and burden-ceiling analysis."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stnbeta.phase5_2c.pre_adr_bounded_analysis import main


if __name__ == "__main__":
    raise SystemExit(main())
