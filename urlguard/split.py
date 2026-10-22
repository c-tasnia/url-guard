"""Domain-grouped train/test split.

A random split puts URLs from the same site in both train and test, so scores
mostly measure memorised domains. Splitting by host measures what matters:
performance on sites the model has never seen.
"""
from sklearn.model_selection import GroupShuffleSplit

from .urls import split_url


def grouped_split(data, test_size=0.2, per_host_cap=10, seed=42):
    """Return ``(train_df, test_df)`` with no host in both.

    ``per_host_cap`` keeps at most that many URLs per host so a few large
    sites cannot dominate training. The ``host`` column is added to both.
    """
    d = data.copy()
    d["host"] = d["url"].map(lambda u: split_url(u)[1])
    if per_host_cap:
        d = d.groupby("host", group_keys=False).head(per_host_cap).reset_index(drop=True)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr_i, te_i = next(gss.split(d, groups=d["host"]))
    return d.iloc[tr_i].reset_index(drop=True), d.iloc[te_i].reset_index(drop=True)
