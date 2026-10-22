"""Assemble the full training table from the source CSV plus augmentation."""
import numpy as np
import pandas as pd

from .data import load_dataset
from .popularity import build_rank
from .tranco import (benign_paths_from, load_tranco, make_benign,
                     split_popular_tail)
from .typosquat import make_typosquats


def build_dataset(path, url_col="url", label_col="label", malicious_label="0",
                  tranco_path=None, use_tranco=True, n_popular=100_000,
                  n_tail=100_000, n_typo=20_000, seed=42):
    """Return ``(data, rank)``.

    ``data`` has columns ``url``, ``label`` (1 = malicious) and ``src``
    (``orig``, ``tranco``, ``tranco_small`` or ``typosquat``) so results can be
    reported on the original rows alone. ``rank`` maps popular domains to their
    rank for the popularity features.
    """
    data = load_dataset(path, url_col, label_col, malicious_label)
    rank = {}
    if use_tranco:
        rng = np.random.default_rng(seed)
        popular, tail = split_popular_tail(load_tranco(tranco_path), n_popular, n_tail, seed)
        rank = build_rank(popular["domain"], popular["rank"])
        paths = benign_paths_from(data)
        parts = [
            data,
            make_benign(popular["domain"], paths, rng, "tranco", subdomain_share=0.5),
            make_benign(tail["domain"], paths, rng, "tranco_small", subdomain_share=0.1),
            make_typosquats(popular["domain"], rank, rng, n_typo),
        ]
        data = pd.concat(parts, ignore_index=True)
    data = data.drop_duplicates(subset="url").reset_index(drop=True)
    return data, rank
