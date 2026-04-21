"""CLI entry point for cohort audit. Delegates to stnbeta.io.bids_loader.main()."""
from stnbeta.io.bids_loader import main
import sys
sys.exit(main())
