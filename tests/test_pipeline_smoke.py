"""End-to-end smoke test: tune -> lock -> test -> analyze on a tiny config (throwaway directories)."""
import glob
import os

import pandas as pd

from hamsched import cli


def test_smoke_pipeline(tmp_path, monkeypatch):
    monkeypatch.setenv("HAMSCHED_RESULTS", str(tmp_path / "results"))
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = os.path.join(root, "configs", "smoke.yaml")
    for c in ("tune", "lock", "test", "analyze"):
        cli.main([c, "--config", cfg, "--workdir", str(tmp_path), "--threads", "2"])
    test = pd.read_csv(tmp_path / "results" / "smoke_test.csv")
    assert len(test) > 0 and test.lock_sha.nunique() == 1
    assert (test.n_readings == 120).all()          # unique-reading budget respected for every plan
    assert glob.glob(str(tmp_path / "results" / "analysis" / "smoke_contrasts.csv"))
