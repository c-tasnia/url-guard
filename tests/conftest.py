"""Small synthetic datasets shared by the tests."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tiny_csv(tmp_path):
    """Source-style CSV where 0 means malicious, like the original dataset."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(300):
        rows.append((f"http://www.site{i}.com/page/{i % 7}/index.html", 1))
        rows.append((f"http://x{rng.integers(10**6)}-login.xyz/verify.php?id={i}", 0))
    rows.append(("http://[broken/url", 0))
    rows.append((None, 1))
    path = tmp_path / "tiny.csv"
    pd.DataFrame(rows, columns=["url", "label"]).to_csv(path, index=False)
    return path


@pytest.fixture
def tiny_tranco(tmp_path):
    """Headerless rank,domain CSV in the Tranco format."""
    path = tmp_path / "tranco.csv"
    pd.DataFrame({"rank": range(1, 501),
                  "domain": [f"popular{i}.com" for i in range(1, 501)]}
                 ).to_csv(path, header=False, index=False)
    return path
