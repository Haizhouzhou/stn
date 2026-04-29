"""CLI for the Phase 5_2C robustness-family audit."""

from __future__ import annotations

import sys

from stnbeta.phase5_2c.robustness_family_audit import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
