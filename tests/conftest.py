import os
import tempfile
from pathlib import Path

import pytest

from ancdata.paths import ENV_VAR


@pytest.fixture(scope="session")
def data_root():
    """One synthetic data root shared by the whole test session."""
    root = Path(tempfile.mkdtemp(prefix="ancdata_test_"))
    os.environ[ENV_VAR] = str(root)
    from ancdata.fixtures import make_fixtures
    from ancdata.registry import build_manifest
    from ancdata.rir_gen import synthetic_bank

    make_fixtures(verbose=False)
    synthetic_bank(n=20, seed=0)
    build_manifest(screen=False, verbose=False)
    return root
