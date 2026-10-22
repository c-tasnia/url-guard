"""Domain popularity features from a ranked domain list (for example Tranco).

The features are soft inputs to the model, not an allowlist: a popular domain
is only one signal among many and can still be flagged.
"""
import math


def build_rank(domains, ranks):
    """Map lower-cased domain -> integer rank (1 is the most popular)."""
    return {str(d).lower(): int(r) for d, r in zip(domains, ranks)}


def popular_parent(host, rank):
    """Return the host, or its first parent domain, found in ``rank``; else None.

    ``en.wikipedia.org`` matches through ``wikipedia.org``.
    """
    labels = host.split(":")[0].lower().split(".")
    for i in range(len(labels) - 1):
        d = ".".join(labels[i:])
        if d in rank:
            return d
    return None


def pop_feat(host, rank):
    """Return ``[in_list, score]`` where score is 6 - log10(rank), or 0 if absent."""
    d = popular_parent(host, rank)
    if d is None:
        return [0, 0]
    return [1, 6 - math.log10(rank[d])]
