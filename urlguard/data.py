"""Dataset loading and label handling."""
import pandas as pd

from .urls import clean_url


def normalise_labels(values, malicious_label="0"):
    """Return 1 for malicious and 0 for benign.

    Datasets disagree on the convention. The dataset this project was built on
    uses ``0`` for malicious and ``1`` for benign, which is the opposite of
    what most code assumes, so the malicious value is an explicit argument.
    """
    s = pd.Series(values).astype(str).str.strip().str.lower()
    target = str(malicious_label).strip().lower()
    hit = s == target
    num_t = pd.to_numeric(pd.Series([target]), errors="coerce").iloc[0]
    if pd.notna(num_t):
        hit = hit | (pd.to_numeric(s, errors="coerce") == num_t)
    return hit.astype(int).to_numpy()


def load_dataset(path, url_col="url", label_col="label", malicious_label="0"):
    """Read a CSV into columns ``url``, ``label`` (1 = malicious) and ``src``."""
    df = pd.read_csv(path, usecols=[url_col, label_col]).dropna()
    df = df.rename(columns={url_col: "url", label_col: "label"})
    out = pd.DataFrame({
        "url": df["url"].map(clean_url).to_numpy(),
        "label": normalise_labels(df["label"], malicious_label),
        "src": "orig",
    })
    return out
