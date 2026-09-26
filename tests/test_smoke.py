"""`pytest` alone proves the install: full chain through every stage."""
from pathlib import Path

from ancdata.selftest import smoke


def test_smoke(tmp_path: Path):
    smoke(keep=tmp_path / "root")
    assert (tmp_path / "root" / "eval" / "smoke" / "meta.jsonl").exists()
    assert (tmp_path / "root" / "outputs" / "crest_attack.png").exists()
