#!/usr/bin/env python
"""Run Phase 5_2C event-target reassessment after owner rejection."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stnbeta.phase5_2c.event_target_reassessment import main


if __name__ == "__main__":
    raise SystemExit(main())
