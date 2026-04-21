def test_import_stnbeta():
    import stnbeta
    from stnbeta.io import bids_loader
    from stnbeta.preprocessing import extract


def test_audit_outputs_exist():
    from pathlib import Path
    root = Path("~/scratch/stn").expanduser()
    assert (root / "audit" / "cohort_summary.tsv").exists()
    assert (root / "extracted").exists()
