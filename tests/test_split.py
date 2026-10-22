import pandas as pd

from urlguard.split import grouped_split


def make_data():
    rows = []
    for h in range(60):
        for i in range(25 if h % 10 == 0 else 3):
            rows.append((f"host{h}.com/p{i}", h % 2, "orig"))
    return pd.DataFrame(rows, columns=["url", "label", "src"])


def test_no_host_appears_in_both_sets():
    train, test = grouped_split(make_data())
    assert not set(train["host"]) & set(test["host"])
    assert len(train) and len(test)


def test_per_host_cap_is_respected():
    train, test = grouped_split(make_data(), per_host_cap=5)
    assert train["host"].value_counts().max() <= 5
    assert test["host"].value_counts().max() <= 5


def test_cap_can_be_disabled():
    train, test = grouped_split(make_data(), per_host_cap=None)
    assert len(train) + len(test) == len(make_data())


def test_split_is_reproducible():
    a, _ = grouped_split(make_data(), seed=7)
    b, _ = grouped_split(make_data(), seed=7)
    assert a["url"].tolist() == b["url"].tolist()
