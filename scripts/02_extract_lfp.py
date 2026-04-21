"""CLI entry point for LFP extraction. Delegates to stnbeta.preprocessing.extract.main()."""
from stnbeta.preprocessing.extract import main
import sys
sys.exit(main())
