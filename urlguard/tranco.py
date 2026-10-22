"""Extra benign training data from the Tranco ranked domain list.

The source dataset's benign class is mostly obscure sites. Without popular
domains, a model learns "benign means obscure" and flags well-known sites.
"""
import io
import urllib.request
import zipfile

import numpy as np
import pandas as pd

TRANCO_URL = "https://tranco-list.eu/top-1m.csv.zip"

SUBDOMAINS = ["en.", "m.", "blog.", "docs.", "support.", "mail.", "shop.",
              "api.", "news.", "help.", "de.", "fr."]


def load_tranco(path=None):
    """Return a DataFrame with ``rank`` and ``domain``.

    ``path`` may be the zip from tranco-list.eu or a plain ``rank,domain`` CSV.
    With no path the list is downloaded.
    """
    if path is None:
        raw = urllib.request.urlopen(TRANCO_URL).read()
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            return pd.read_csv(z.open(z.namelist()[0]), header=None, names=["rank", "domain"])
    if str(path).lower().endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            return pd.read_csv(z.open(z.namelist()[0]), header=None, names=["rank", "domain"])
    return pd.read_csv(path, header=None, names=["rank", "domain"])


def split_popular_tail(tranco, n_popular=100_000, n_tail=100_000, seed=42):
    """Top ``n_popular`` domains, plus a random sample of the less popular rest.

    Only the popular part is used for the popularity features. The tail teaches
    the model that small, unknown sites are usually benign too.
    """
    popular = tranco.head(n_popular)
    rest = tranco.iloc[n_popular:]
    tail = rest.sample(min(n_tail, len(rest)), random_state=seed) if len(rest) else rest
    return popular, tail


def benign_paths_from(df):
    """Real paths taken from benign rows, so added domains get realistic URLs."""
    ben = df.loc[df["label"] == 0, "url"]
    paths = ben.map(lambda u: "/" + u.split("/", 1)[1] if "/" in u else "")
    return [p for p in paths if 0 < len(p) < 80]


def make_benign(domains, paths, rng, src, subdomain_share=0.5, bare_share=0.4):
    """Benign rows: a domain, optionally with a subdomain and a real benign path."""
    out = []
    for d in domains:
        host = rng.choice(SUBDOMAINS) + d if rng.random() < subdomain_share else d
        if rng.random() < bare_share or not paths:
            out.append(host)
        else:
            out.append(host + paths[rng.integers(len(paths))])
    return pd.DataFrame({"url": out, "label": 0, "src": src})
