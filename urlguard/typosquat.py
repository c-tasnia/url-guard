"""Synthetic lookalike (typosquat) domains used as extra malicious examples.

These are generated, not collected, so they should be described as synthetic
augmentation wherever results are reported.
"""
import pandas as pd

LOOKALIKES = {"o": "0", "l": "1", "i": "1", "a": "4", "e": "3", "s": "5"}
BAIT_PATHS = ["", "/", "/login", "/verify", "/secure/account", "/signin?id=",
              "/update/password"]


def typo(domain, rng):
    """One lookalike of ``domain``, or None if the name is too short to alter."""
    name, _, tld = domain.rpartition(".")
    if len(name) < 4:
        return None
    i = int(rng.integers(1, len(name) - 1))
    op = int(rng.integers(0, 4))
    if op == 0:      # doubled letter: gooogle
        n = name[:i] + name[i] + name[i:]
    elif op == 1:    # swapped neighbours
        n = name[:i] + name[i + 1] + name[i] + name[i + 2:]
    elif op == 2:    # digit lookalike: paypa1
        n = name[:i] + LOOKALIKES.get(name[i], name[i]) + name[i + 1:]
    else:            # dropped letter
        n = name[:i] + name[i + 1:]
    if n == name:
        return None
    return n + "." + tld


def make_typosquats(domains, known, rng, n=20_000):
    """Malicious rows built from the first ``n`` ``domains``.

    ``known`` holds real popular domains; a lookalike that happens to be a
    real domain is skipped rather than mislabelled.
    """
    rows = []
    for d in list(domains)[:n]:
        t = typo(d, rng)
        if t and t not in known:
            rows.append(t + BAIT_PATHS[rng.integers(len(BAIT_PATHS))])
    return pd.DataFrame({"url": rows, "label": 1, "src": "typosquat"})
